// -*- C++ -*-
//
// Package:    FastTimeNumberingTester
// Class:      FastTimeNumberingTester
// 
/**\class FastTimeNumberingTester FastTimeNumberingTester.cc test/FastTimeNumberingTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2014/04/24
// $Id: FastTimeNumberingTester.cc,v 1.0 2014/04/24 14:06:07 sunanda Exp $
//
//

//#define EDM_ML_DEBUG

// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/FastTimeDDDConstants.h"
#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"

class FastTimeNumberingTester : public edm::one::EDAnalyzer<> {
public:
  explicit FastTimeNumberingTester( const edm::ParameterSet& );
  ~FastTimeNumberingTester();

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

FastTimeNumberingTester::FastTimeNumberingTester(const edm::ParameterSet& ) {}

FastTimeNumberingTester::~FastTimeNumberingTester() {}

// ------------ method called to produce the data  ------------
void FastTimeNumberingTester::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {
  
  edm::ESHandle<FastTimeDDDConstants> pFTNDC;

  iSetup.get<IdealGeometryRecord>().get(pFTNDC);
  const FastTimeDDDConstants fTnDC(*pFTNDC);
  std::cout << "Fast timing device with " << fTnDC.getCells(1) << ":"
	    << fTnDC.getCells(2) << " cells" << " for barrel and endcap\n";
  for (int type=1; type<=2; ++type) {
    for (int ix=0; ix<400; ++ix) {
      for (int iy=0; iy<400; ++iy) {
	if (fTnDC.isValidXY(type,ix,iy)) {
	  FastTimeDetId id1(type,ix,iy,1), id2(type,ix,iy,-1);
	  std::cout << "Valid ID " << id1 << " and " << id2 << std::endl;
	} else {
#ifdef EDM_ML_DEBUG
	  std::cout << "ix = " << ix << ", iy = " << iy << " is not valid for "
		    << "FastTime type " << type << std::endl;
#endif
	}
	iy += 9;
      }
      ix += 9;
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(FastTimeNumberingTester);
