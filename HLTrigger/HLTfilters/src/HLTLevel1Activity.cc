/** \class HLTLevel1Activity
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) 
 *  that checks if there was any L1 activity
 *  It can be configured to
 *    - look at different bunch crossings
 *    - use or ignore the L1 trigger mask
 *    - only look at a subset of the L1 bits
 * 
 *  $Date: 2009/11/17 14:03:10 $
 *  $Revision: 1.4 $
 *
 *  \author Andrea Bocci
 *
 */

#include <vector>

#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTLevel1Activity : public HLTFilter {
public:
  explicit HLTLevel1Activity(const edm::ParameterSet&);
  ~HLTLevel1Activity();
  virtual bool filter(edm::Event&, const edm::EventSetup&);

private:
  edm::InputTag     m_gtReadoutRecord;

  std::vector<int>  m_bunchCrossings;
  std::vector<bool> m_selectPhysics;
  std::vector<bool> m_selectTechnical;
  std::vector<bool> m_maskedPhysics;
  std::vector<bool> m_maskedTechnical;
  bool              m_ignoreL1Mask;

  edm::ESWatcher<L1GtTriggerMaskAlgoTrigRcd> m_watchPhysicsMask;
  edm::ESWatcher<L1GtTriggerMaskTechTrigRcd> m_watchTechnicalMask;
};

#include <boost/foreach.hpp>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#define PHTSICS_BITS_SIZE    128
#define TECHNICAL_BITS_SIZE   64
#define DAQ_PARTITIONS      0xFF

//
// constructors and destructor
//
HLTLevel1Activity::HLTLevel1Activity(const edm::ParameterSet & config) :
  m_gtReadoutRecord( config.getParameter<edm::InputTag>("L1GtReadoutRecordTag") ),
  m_bunchCrossings( config.getParameter<std::vector<int> >("bunchCrossings") ),
  m_selectPhysics(PHTSICS_BITS_SIZE),
  m_selectTechnical(TECHNICAL_BITS_SIZE),
  m_maskedPhysics(PHTSICS_BITS_SIZE),
  m_maskedTechnical(TECHNICAL_BITS_SIZE),
  m_ignoreL1Mask( config.getParameter<bool>("ignoreL1Mask") )
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

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTLevel1Activity::filter(edm::Event& event, const edm::EventSetup& setup)
{
  // apply L1 mask to the physics bits
  if (not m_ignoreL1Mask and m_watchPhysicsMask.check(setup)) {
    edm::ESHandle<L1GtTriggerMask> h_mask;
    setup.get<L1GtTriggerMaskAlgoTrigRcd>().get(h_mask);
    const std::vector<unsigned int> & mask = h_mask->gtTriggerMask();
    for (unsigned int i = 0; i < PHTSICS_BITS_SIZE; ++i)
      m_maskedPhysics[i] = m_selectPhysics[i] and ((mask[i] & DAQ_PARTITIONS) != DAQ_PARTITIONS);
  }
  
  // apply L1 mask to the technical bits
  if (not m_ignoreL1Mask and m_watchTechnicalMask.check(setup)) {
    edm::ESHandle<L1GtTriggerMask> h_mask;
    setup.get<L1GtTriggerMaskTechTrigRcd>().get(h_mask);
    const std::vector<unsigned int> & mask = h_mask->gtTriggerMask();
    for (unsigned int i = 0; i < TECHNICAL_BITS_SIZE; ++i)
      m_maskedTechnical[i] = m_selectTechnical[i] and ((mask[i] & DAQ_PARTITIONS) != DAQ_PARTITIONS);
  }

  // access the L1 decisions
  edm::Handle<L1GlobalTriggerReadoutRecord> h_gtReadoutRecord;
  event.getByLabel(m_gtReadoutRecord, h_gtReadoutRecord);
 
  // compare the results with the requested bits, and return true as soon as the first match is found
  BOOST_FOREACH(int bx, m_bunchCrossings) {
    const std::vector<bool> & physics = h_gtReadoutRecord->decisionWord(bx);
    for (unsigned int i = 0; i < PHTSICS_BITS_SIZE; ++i)
      if (m_maskedPhysics[i] and physics[i])
        return true;
    const std::vector<bool> & technical = h_gtReadoutRecord->technicalTriggerWord(bx);
    for (unsigned int i = 0; i < TECHNICAL_BITS_SIZE; ++i)
      if (m_maskedTechnical[i] and technical[i])
        return true;
  }
 
  return false; 
}

// define as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTLevel1Activity);
