// -*- C++ -*-
//
// Package:    Castor
// Class:      CastorDumpConditions
//
/**\class Castor CastorDumpConditions.cc CondTools/Castor/src/CastorDumpConditions.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Luiz Mundim Filho
//         Created:  Thu Mar 12 14:45:44 CET 2009
// $Id: CastorDumpConditions.cc,v 1.1 2011/05/09 19:38:47 mundim Exp $
//
//

// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/CastorPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/CastorPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorGainsRcd.h"
#include "CondFormats/DataRecord/interface/CastorGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/CastorChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/CastorElectronicsMapRcd.h"
#include "CondFormats/DataRecord/interface/CastorRecoParamsRcd.h"
#include "CondFormats/DataRecord/interface/CastorSaturationCorrsRcd.h"
#include "CondFormats/CastorObjects/interface/AllObjects.h"

#include "CalibCalorimetry/CastorCalib/interface/CastorDbASCIIIO.h"
//
// class decleration
//

class CastorDumpConditions : public edm::one::EDAnalyzer<> {
public:
  explicit CastorDumpConditions(const edm::ParameterSet&);

  template <class S, class SRcd>
  void dumpIt(const std::vector<std::string>& mDumpRequest,
              const edm::Event& e,
              const edm::EventSetup& context,
              const std::string name);

private:
  std::string file_prefix;
  std::vector<std::string> mDumpRequest;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
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
CastorDumpConditions::CastorDumpConditions(const edm::ParameterSet& iConfig)

{
  file_prefix = iConfig.getUntrackedParameter<std::string>("outFilePrefix", "Dump");
  mDumpRequest = iConfig.getUntrackedParameter<std::vector<std::string> >("dump", std::vector<std::string>());
  if (mDumpRequest.empty()) {
    throw cms::Exception("Bad Config") << "CastorDumpConditions: No record to dump.";
  }
}

//
// member functions
//

// ------------ method called to for each event  ------------
void CastorDumpConditions::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  {
    edm::LogAbsolute log("CastorDumpConditions");
    log << "I AM IN THE RUN " << iEvent.id().run() << "\n";
    log << "What to dump? " << std::endl;
    if (mDumpRequest.empty()) {
      log << "CastorDumpConditions: Empty request \n";
      return;
    }
  }

  for (std::vector<std::string>::const_iterator it = mDumpRequest.begin(); it != mDumpRequest.end(); it++)
    LogAbsolute("CastorDumpConditions") << *it << "\n";

  // dumpIt called for all possible ValueMaps. The function checks if the dump is actually requested.
  dumpIt<CastorElectronicsMap, CastorElectronicsMapRcd>(mDumpRequest, iEvent, iSetup, "ElectronicsMap");
  dumpIt<CastorQIEData, CastorQIEDataRcd>(mDumpRequest, iEvent, iSetup, "QIEData");
  dumpIt<CastorPedestals, CastorPedestalsRcd>(mDumpRequest, iEvent, iSetup, "Pedestals");
  dumpIt<CastorPedestalWidths, CastorPedestalWidthsRcd>(mDumpRequest, iEvent, iSetup, "PedestalWidths");
  dumpIt<CastorGains, CastorGainsRcd>(mDumpRequest, iEvent, iSetup, "Gains");
  dumpIt<CastorGainWidths, CastorGainWidthsRcd>(mDumpRequest, iEvent, iSetup, "GainWidths");
  dumpIt<CastorChannelQuality, CastorChannelQualityRcd>(mDumpRequest, iEvent, iSetup, "ChannelQuality");
  dumpIt<CastorRecoParams, CastorRecoParamsRcd>(mDumpRequest, iEvent, iSetup, "RecoParams");
  dumpIt<CastorSaturationCorrs, CastorSaturationCorrsRcd>(mDumpRequest, iEvent, iSetup, "SaturationCorrs");
}

template <class S, class SRcd>
void CastorDumpConditions::dumpIt(const std::vector<std::string>& mDumpRequest,
                                  const edm::Event& e,
                                  const edm::EventSetup& context,
                                  const std::string name) {
  if (std::find(mDumpRequest.begin(), mDumpRequest.end(), name) != mDumpRequest.end()) {
    int myrun = e.id().run();
    edm::ESGetToken<S, SRcd> tok = esConsumes<S, SRcd>();
    const S& myobject = context.getData(tok);

    std::ostringstream file;
    file << file_prefix << name.c_str() << "_Run" << myrun << ".txt";
    std::ofstream outStream(file.str().c_str());
    CastorDbASCIIIO::dumpObject(outStream, myobject);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(CastorDumpConditions);
