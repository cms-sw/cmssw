// -*- C++ -*-
//
// Package:    HcalSimNumberingTester
// Class:      HcalSimNumberingTester
// 
/**\class HcalSimNumberingTester HcalSimNumberingTester.cc test/HcalSimNumberingTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2013/12/26
//


// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"


class HcalSimNumberingTester : public edm::one::EDAnalyzer<> {
public:
  explicit HcalSimNumberingTester( const edm::ParameterSet& );
  ~HcalSimNumberingTester() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

HcalSimNumberingTester::HcalSimNumberingTester(const edm::ParameterSet& ) {}


HcalSimNumberingTester::~HcalSimNumberingTester() {}

// ------------ method called to produce the data  ------------
void HcalSimNumberingTester::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  edm::ESHandle<HcalDDDSimConstants> pHSNDC;
  iSetup.get<HcalSimNumberingRecord>().get( pHSNDC );

  if (pHSNDC.isValid()) {
    edm::LogVerbatim("HcalGeom") << "about to de-reference the edm::ESHandle<HcalDDDSimConstants> pHSNDC";
    const HcalDDDSimConstants hdc (*pHSNDC);
    edm::LogVerbatim("HcalGeom") << "about to getConst for 0..1";
    for (int i=0; i<1; ++i) {
      std::vector<std::pair<double,double> > gcons = hdc.getConstHBHE(i);
      edm::LogVerbatim("HcalGeom") << "Geometry Constants for [" << i 
				   << "] with " << gcons.size() << "  elements";
      for (unsigned int k=0; k<gcons.size(); ++k)
	edm::LogVerbatim("HcalGeom") << "Element[" << k << "] = " 
				     << gcons[k].first << " : "
				     << gcons[k].second;
    }
    hdc.printTiles();
  } else {
    edm::LogVerbatim("HcalGeom") << "No record found with HcalDDDSimConstants";
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(HcalSimNumberingTester);
