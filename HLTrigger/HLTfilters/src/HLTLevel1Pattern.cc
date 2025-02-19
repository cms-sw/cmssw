/** \class HLTLevel1Pattern
 *
 *
 *  This class is an EDFilter
 *  that checks for a specific pattern of L1 accept/reject in 5 BX's for a given L1 bit
 *  It can be configured to use or ignore the L1 trigger mask
 *
 *  $Date: 2012/01/23 00:18:04 $
 *  $Revision: 1.9 $
 *
 *  \author Andrea Bocci
 *
 */

#include <vector>

#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"

//
// class declaration
//

class HLTLevel1Pattern : public edm::EDFilter {
public:
  explicit HLTLevel1Pattern(const edm::ParameterSet&);
  ~HLTLevel1Pattern();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual bool filter(edm::Event&, const edm::EventSetup&);

private:
  edm::InputTag     m_gtReadoutRecord;
  std::string       m_triggerBit;
  std::vector<int>  m_bunchCrossings;
  std::vector<int>  m_triggerPattern;
  unsigned int      m_daqPartitions;
  unsigned int      m_triggerNumber;
  bool              m_triggerAlgo;
  bool              m_triggerMasked;
  bool              m_ignoreL1Mask;
  bool              m_invert;
  bool              m_throw;

  edm::ESWatcher<L1GtTriggerMenuRcd>         m_watchL1Menu;
  edm::ESWatcher<L1GtTriggerMaskAlgoTrigRcd> m_watchPhysicsMask;
  edm::ESWatcher<L1GtTriggerMaskTechTrigRcd> m_watchTechnicalMask;
};

#include <boost/foreach.hpp>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

//
// constructors and destructor
//
HLTLevel1Pattern::HLTLevel1Pattern(const edm::ParameterSet & config) :
  m_gtReadoutRecord( config.getParameter<edm::InputTag>     ("L1GtReadoutRecordTag") ),
  m_triggerBit(      config.getParameter<std::string>       ("triggerBit") ),
  m_bunchCrossings(  config.getParameter<std::vector<int> > ("bunchCrossings") ),
  m_triggerPattern(  m_bunchCrossings.size(), false ),
  m_daqPartitions(   config.getParameter<unsigned int>      ("daqPartitions") ),
  m_triggerNumber(   0 ),
  m_triggerAlgo(     true ),
  m_triggerMasked(   false ),
  m_ignoreL1Mask(    config.getParameter<bool>              ("ignoreL1Mask") ),
  m_invert(          config.getParameter<bool>              ("invert") ),
  m_throw (          config.getParameter<bool>              ("throw" ) )
{
  std::vector<int> pattern( config.getParameter<std::vector<int> > ("triggerPattern") );
  if (pattern.size() != m_bunchCrossings.size())
    throw cms::Exception("Configuration") << "\"bunchCrossings\" and \"triggerPattern\" parameters do not match";

  for (unsigned int i = 0; i < pattern.size(); ++i)
    m_triggerPattern[i] = (bool) pattern[i];
}

HLTLevel1Pattern::~HLTLevel1Pattern()
{
}

void
HLTLevel1Pattern::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("L1GtReadoutRecordTag",edm::InputTag("hltGtDigis"));
  desc.add<std::string>("triggerBit","L1Tech_RPC_TTU_pointing_Cosmics.v0");
  {
    std::vector<int> temp1;
    temp1.reserve(5);
    temp1.push_back(-2);
    temp1.push_back(-1);
    temp1.push_back(0);
    temp1.push_back(1);
    temp1.push_back(2);
    desc.add<std::vector<int> >("bunchCrossings",temp1);
  }
  desc.add<unsigned int>("daqPartitions",1);
  desc.add<bool>("ignoreL1Mask",false);
  desc.add<bool>("invert",false);
  desc.add<bool>("throw",true);
  {
    std::vector<int> temp1;
    temp1.reserve(5);
    temp1.push_back(1);
    temp1.push_back(1);
    temp1.push_back(1);
    temp1.push_back(0);
    temp1.push_back(0);
    desc.add<std::vector<int> >("triggerPattern",temp1);
  }
  descriptions.add("hltLevel1Pattern",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTLevel1Pattern::filter(edm::Event& event, const edm::EventSetup& setup)
{
  // determine the L1 algo or tech bit to use
  if (m_watchL1Menu.check(setup)) {
    edm::ESHandle<L1GtTriggerMenu> h_menu;
    setup.get<L1GtTriggerMenuRcd>().get(h_menu);

    // look for an Algo L1 bit
    const AlgorithmMap & algoMap = h_menu->gtAlgorithmAliasMap();
    const AlgorithmMap & techMap = h_menu->gtTechnicalTriggerMap();
    AlgorithmMap::const_iterator entry;
    if ((entry = algoMap.find(m_triggerBit)) != algoMap.end()) {
        m_triggerAlgo = true;
        m_triggerNumber = entry->second.algoBitNumber();
    } else 
    if ((entry = techMap.find(m_triggerBit)) != techMap.end()) {
        m_triggerAlgo = false;
        m_triggerNumber = entry->second.algoBitNumber();
    } else {
      if (m_throw) {
	throw cms::Exception("Configuration") << "requested L1 trigger \"" << m_triggerBit << "\" does not exist in the current L1 menu";
      } else {
	return m_invert;
      }
    }
  }

  if (m_triggerAlgo) {
    // check the L1 algorithms mask
    //  - mask & partition == part. --> fully masked
    //  - mask & partition == 0x00  --> fully unmasked
    //  - mask & partition != part. --> unmasked in some partitions, consider as unmasked
    if (m_watchPhysicsMask.check(setup)) {
      edm::ESHandle<L1GtTriggerMask> h_mask;
      setup.get<L1GtTriggerMaskAlgoTrigRcd>().get(h_mask);
      m_triggerMasked = ((h_mask->gtTriggerMask()[m_triggerNumber] & m_daqPartitions) == m_daqPartitions);
    }
  } else {
    // check the L1 technical triggers mask
    //  - mask & partition == part. --> fully masked
    //  - mask & partition == 0x00  --> fully unmasked
    //  - mask & partition != part. --> unmasked in some partitions, consider as unmasked
    if (m_watchTechnicalMask.check(setup)) {
      edm::ESHandle<L1GtTriggerMask> h_mask;
      setup.get<L1GtTriggerMaskTechTrigRcd>().get(h_mask);
      m_triggerMasked = ((h_mask->gtTriggerMask()[m_triggerNumber] & m_daqPartitions) == m_daqPartitions);
    }
  }

  // is the L1 trigger masked ?
  if (not m_ignoreL1Mask and m_triggerMasked)
    return m_invert;

  // access the L1 decisions
  edm::Handle<L1GlobalTriggerReadoutRecord> h_gtReadoutRecord;
  event.getByLabel(m_gtReadoutRecord, h_gtReadoutRecord);

  // check the L1 algorithms results
  for (unsigned int i = 0; i < m_bunchCrossings.size(); ++i) {
    int bx = m_bunchCrossings[i];
    const std::vector<bool> & word = (m_triggerAlgo) ? h_gtReadoutRecord->decisionWord(bx) : h_gtReadoutRecord->technicalTriggerWord(bx);
    if (word.empty() or m_triggerNumber >= word.size())
      // L1 results not available, bail out
      return m_invert;
    bool result = word[m_triggerNumber];
    if (result != m_triggerPattern[i])
      // comparison failed, bail out
      return m_invert;
  }

  // comparison successful
  return not m_invert;
}

// define as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTLevel1Pattern);
