// -*- C++ -*-
//
// Package:    DumpL1RPCConfig
// Class:      DumpL1RPCConfig
//
/**\class DumpL1RPCConfig DumpL1RPCConfig.cc L1TriggerConfig/DumpL1RPCConfig/src/DumpL1RPCConfig.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Wed Apr  9 14:03:40 CEST 2008
//
//

// system include files
#include <memory>
#include "CondFormats/L1TObjects/interface/L1RPCConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCConfigRcd.h"
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <fstream>

//
// class decleration
//

class DumpL1RPCConfig : public edm::global::EDAnalyzer<> {
public:
  explicit DumpL1RPCConfig(const edm::ParameterSet&);

private:
  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  edm::ESGetToken<L1RPCConfig, L1RPCConfigRcd> getToken_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DumpL1RPCConfig::DumpL1RPCConfig(const edm::ParameterSet& iConfig)
    : getToken_(esConsumes())

{
  //now do what ever initialization is needed
}

//
// member functions
//

// ------------ method called to for each event  ------------
void DumpL1RPCConfig::analyze(edm::StreamID, const edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  edm::ESHandle<L1RPCConfig> l1RPCConfig = iSetup.getHandle(getToken_);

  LogTrace("DumpL1RPCConfig") << std::endl;
  LogDebug("DumpL1RPCConfig") << "\n Printing L1RPCConfigRcd record\n" << std::endl;
  LogTrace("DumpL1RPCConfig") << "\nPacs per tower: " << l1RPCConfig->getPPT() << std::endl;

  LogTrace("DumpL1RPCConfig") << "\nQuality table:" << std::endl;

  {
    for (RPCPattern::TQualityVec::const_iterator it = l1RPCConfig->m_quals.begin(); it != l1RPCConfig->m_quals.end();
         ++it) {
      LogTrace("DumpL1RPCConfig") << "QTN " << (int)it->m_QualityTabNumber << " fp " << (int)it->m_FiredPlanes
                                  << " val " << (int)it->m_QualityValue << " tw " << (int)it->m_tower << " sc "
                                  << (int)it->m_logsector << " sg " << (int)it->m_logsegment << std::endl;
    }
  }

  LogTrace("DumpL1RPCConfig") << "\nPatterns:" << std::endl;
  {
    for (RPCPattern::RPCPatVec::const_iterator it = l1RPCConfig->m_pats.begin(); it != l1RPCConfig->m_pats.end();
         ++it) {
      LogTrace("DumpL1RPCConfig") << "tw " << it->getTower() << " sc " << it->getLogSector() << " sg "
                                  << it->getLogSegment() << " pt " << it->getCode() << " s " << it->getSign() << " n "
                                  << it->getNumber() << " t " << (int)it->getPatternType() << " rg "
                                  << it->getRefGroup() << " QTN " << it->getQualityTabNumber();
      for (int lp = 0; lp < RPCPattern::m_LOGPLANES_COUNT; ++lp) {
        LogTrace("DumpL1RPCConfig") << " (LP" << lp << " " << it->getStripFrom(lp) << " " << it->getStripTo(lp) << ")";
      }
      LogTrace("DumpL1RPCConfig") << std::endl;
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(DumpL1RPCConfig);
