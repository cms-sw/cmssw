/** \class HLTLevel1Activity
 *
 *  
 *  This class is an EDFilter
 *  that checks if there was any L1 activity
 *  It can be configured to
 *    - look at different bunch crossings
 *    - use or ignore the L1 trigger mask
 *    - only look at a subset of the L1 bits
 * 
 *
 *  \author Andrea Bocci
 *
 */

#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

// FIXME: these should come form the L1 configuration at runtime
#define PHYSICS_BITS_SIZE    128
#define TECHNICAL_BITS_SIZE   64

//
// class declaration
//

class HLTLevel1Activity : public edm::stream::EDFilter<> {
public:
  explicit HLTLevel1Activity(const edm::ParameterSet&);
  ~HLTLevel1Activity();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual bool filter(edm::Event &, edm::EventSetup const &) override final;

private:
  edm::InputTag                                  m_gtReadoutRecordTag;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> m_gtReadoutRecordToken;
  std::vector<int>  m_bunchCrossings;
  std::vector<bool> m_selectPhysics;
  std::vector<bool> m_selectTechnical;
  std::vector<bool> m_maskedPhysics;
  std::vector<bool> m_maskedTechnical;
  unsigned int      m_daqPartitions;
  bool              m_ignoreL1Mask;
  bool              m_invert;

  edm::ESWatcher<L1GtTriggerMaskAlgoTrigRcd> m_watchPhysicsMask;
  edm::ESWatcher<L1GtTriggerMaskTechTrigRcd> m_watchTechnicalMask;
};

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"

//
// constructors and destructor
//
HLTLevel1Activity::HLTLevel1Activity(const edm::ParameterSet & config) :
  m_gtReadoutRecordTag( config.getParameter<edm::InputTag>  ("L1GtReadoutRecordTag") ),
  m_gtReadoutRecordToken(consumes<L1GlobalTriggerReadoutRecord>(m_gtReadoutRecordTag)),
  m_bunchCrossings(  config.getParameter<std::vector<int>>  ("bunchCrossings") ),
  m_selectPhysics(   PHYSICS_BITS_SIZE ),
  m_selectTechnical( TECHNICAL_BITS_SIZE ),
  m_maskedPhysics(   PHYSICS_BITS_SIZE ),
  m_maskedTechnical( TECHNICAL_BITS_SIZE ),
  m_daqPartitions(   config.getParameter<unsigned int>      ("daqPartitions") ),
  m_ignoreL1Mask(    config.getParameter<bool>              ("ignoreL1Mask") ),
  m_invert(          config.getParameter<bool>              ("invert") )
{
  unsigned long long low  = config.getParameter<unsigned long long>("physicsLoBits");
  unsigned long long high = config.getParameter<unsigned long long>("physicsHiBits");
  unsigned long long tech = config.getParameter<unsigned long long>("technicalBits");
  for (unsigned int i = 0; i < 64; i++) {
    m_selectPhysics[i]    = low  & (0x01ULL << (unsigned long long) i);
    m_maskedPhysics[i]    = low  & (0x01ULL << (unsigned long long) i);
  }
  for (unsigned int i = 0; i < 64; i++) {
    m_selectPhysics[i+64] = high & (0x01ULL << (unsigned long long) i);
    m_maskedPhysics[i+64] = high & (0x01ULL << (unsigned long long) i);
  }
  for (unsigned int i = 0; i < 64; i++) {
    m_selectTechnical[i]  = tech & (0x01ULL << (unsigned long long) i);
    m_maskedTechnical[i]  = tech & (0x01ULL << (unsigned long long) i);
  }
}

HLTLevel1Activity::~HLTLevel1Activity()
{
}

void
HLTLevel1Activity::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("L1GtReadoutRecordTag", edm::InputTag("hltGtDigis"));
  desc.add<std::vector<int>>("bunchCrossings", {0, -1, 1});
  desc.add<unsigned int>("daqPartitions", 1);
  desc.add<bool>("ignoreL1Mask", false);
  desc.add<bool>("invert", false);
  desc.add<unsigned long long int>("physicsLoBits", 0x0000000000000001LL);  // bit 0
  desc.add<unsigned long long int>("physicsHiBits", 0x0000000000040000LL);  // bit 64 + 18 = 82
  desc.add<unsigned long long int>("technicalBits", 0x0000000000000001LL);  // bit 0
  descriptions.add("hltLevel1Activity", desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTLevel1Activity::filter(edm::Event & event, edm::EventSetup const & setup)
{
  /*
  // apply L1 mask to the physics bits
  //  - mask & partition == part. --> fully masked
  //  - mask & partition == 0x00  --> fully unmasked
  //  - mask & partition != part. --> unmasked in some partitions, consider as unmasked
  if (not m_ignoreL1Mask and m_watchPhysicsMask.check(setup)) {
    edm::ESHandle<L1GtTriggerMask> h_mask;
    setup.get<L1GtTriggerMaskAlgoTrigRcd>().get(h_mask);
    const std::vector<unsigned int> & mask = h_mask->gtTriggerMask();
    for (unsigned int i = 0; i < PHYSICS_BITS_SIZE; ++i)
      m_maskedPhysics[i] = m_selectPhysics[i] and ((mask[i] & m_daqPartitions) != m_daqPartitions);
  }
  
  // apply L1 mask to the technical bits
  //  - mask & partition == part. --> fully masked
  //  - mask & partition == 0x00  --> fully unmasked
  //  - mask & partition != part. --> unmasked in some partitions, consider as unmasked
  if (not m_ignoreL1Mask and m_watchTechnicalMask.check(setup)) {
    edm::ESHandle<L1GtTriggerMask> h_mask;
    setup.get<L1GtTriggerMaskTechTrigRcd>().get(h_mask);
    const std::vector<unsigned int> & mask = h_mask->gtTriggerMask();
    for (unsigned int i = 0; i < TECHNICAL_BITS_SIZE; ++i)
      m_maskedTechnical[i] = m_selectTechnical[i] and ((mask[i] & m_daqPartitions) != m_daqPartitions);
  }

  // access the L1 decisions
  edm::Handle<L1GlobalTriggerReadoutRecord> h_gtReadoutRecord;
  event.getByToken(m_gtReadoutRecordToken, h_gtReadoutRecord);

  // compare the results with the requested bits, and return true as soon as the first match is found
  for (int bx : m_bunchCrossings) {
    const std::vector<bool> & physics = h_gtReadoutRecord->decisionWord(bx);
    if (physics.size() != PHYSICS_BITS_SIZE)
      // error in L1 results
      return m_invert;
    for (unsigned int i = 0; i < PHYSICS_BITS_SIZE; ++i)
      if (m_maskedPhysics[i] and physics[i])
        return not m_invert;
    const std::vector<bool> & technical = h_gtReadoutRecord->technicalTriggerWord(bx);
    if (technical.size() != TECHNICAL_BITS_SIZE)
      // error in L1 results
      return m_invert;
    for (unsigned int i = 0; i < TECHNICAL_BITS_SIZE; ++i)
      if (m_maskedTechnical[i] and technical[i])
        return not m_invert;
  }
 
  return m_invert; 
  */
  return false;
}

// define as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTLevel1Activity);
