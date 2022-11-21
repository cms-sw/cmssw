// -*- C++ -*-
//
// Package:    L1CaloGeometryDump
// Class:      L1CaloGeometryDump
//
/**\class L1CaloGeometryDump L1CaloGeometryDump.cc
 L1TriggerConfig/L1CaloGeometryDump/src/L1CaloGeometryDump.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Mon Sep 28 22:17:24 CEST 2009
// $Id$
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"

//
// class decleration
//

class L1CaloGeometryDump : public edm::global::EDAnalyzer<> {
public:
  explicit L1CaloGeometryDump(const edm::ParameterSet &);

private:
  void analyze(edm::StreamID, const edm::Event &, const edm::EventSetup &) const override;

  // ----------member data ---------------------------
  const edm::ESGetToken<L1CaloGeometry, L1CaloGeometryRecord> geomToken_;
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
L1CaloGeometryDump::L1CaloGeometryDump(const edm::ParameterSet &iConfig) : geomToken_(esConsumes()) {
  // now do what ever initialization is needed
}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1CaloGeometryDump::analyze(edm::StreamID, const edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  LogDebug("L1CaloGeometryDump") << iSetup.getData(geomToken_) << std::endl;
}

// define this as a plug-in
DEFINE_FWK_MODULE(L1CaloGeometryDump);
