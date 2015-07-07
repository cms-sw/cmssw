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
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"

#include "CoralBase/Exception.h"

//
// class decleration
//

class HcalSimNumberingTester : public edm::EDAnalyzer {
public:
  explicit HcalSimNumberingTester( const edm::ParameterSet& );
  ~HcalSimNumberingTester();

  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
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
HcalSimNumberingTester::HcalSimNumberingTester(const edm::ParameterSet& ) {}


HcalSimNumberingTester::~HcalSimNumberingTester() {}


//
// member functions
//

// ------------ method called to produce the data  ------------
void HcalSimNumberingTester::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  edm::ESHandle<HcalDDDSimConstants> pHSNDC;
  iSetup.get<HcalSimNumberingRecord>().get( pHSNDC );

  if (pHSNDC.isValid()) {
    std::cout << "about to de-reference the edm::ESHandle<HcalDDDSimConstants> pHSNDC" << std::endl;
    const HcalDDDSimConstants hdc (*pHSNDC);
    std::cout << "about to getConst for 0..1" << std::endl;
    for (int i=0; i<1; ++i) {
      std::vector<std::pair<double,double> > gcons = hdc.getConstHBHE(i);
      std::cout << "Geometry Constants for [" << i << "] with " 
		<< gcons.size() << "  elements" << std::endl;
      for (unsigned int k=0; k<gcons.size(); ++k)
	std::cout << "Element[" << k << "] = " << gcons[k].first << " : "
		  << gcons[k].second << std::endl;
    }
    hdc.printTiles();
  } else {
    std::cout << "No record found with HcalDDDSimConstants" << std::endl;
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(HcalSimNumberingTester);
