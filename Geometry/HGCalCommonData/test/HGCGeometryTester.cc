// -*- C++ -*-
//
// Package:    HGCGeometryTester
// Class:      HGCGeometryTester
// 
/**\class HGCGeometryTester HGCGeometryTester.cc test/HGCGeometryTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2014/02/07
// $Id: HGCGeometryTester.cc,v 1.0 2014/02/07 14:06:07 sunanda Exp $
//
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
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "CoralBase/Exception.h"

class HGCGeometryTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCGeometryTester( const edm::ParameterSet& );
  ~HGCGeometryTester();

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

HGCGeometryTester::HGCGeometryTester(const edm::ParameterSet& ) {}

HGCGeometryTester::~HGCGeometryTester() {}

// ------------ method called to produce the data  ------------
void HGCGeometryTester::analyze( const edm::Event& iEvent, 
				 const edm::EventSetup& iSetup ) {

  edm::ESTransientHandle<DDCompactView> pDD;
  iSetup.get<IdealGeometryRecord>().get( pDD );
  
  //parse the DD for sensitive volumes
  DDExpandedView eview(*pDD);
  std::map<std::string, std::pair<double,double> > svPars;
  do {
    const DDLogicalPart &logPart=eview.logicalPart();
    std::string name=logPart.name();

    //only EE sensitive volumes for the moment
    if ((name.find("HGCal") != std::string::npos) &&
	(name.find("Sensitive") != std::string::npos)) {
    
      size_t pos=name.find("Sensitive")+9;
      int layer=atoi(name.substr(pos,name.size()-1).c_str());
      if (svPars.find(name) == svPars.end()) {
	//print half height and widths for the trapezoid
	std::vector<double> solidPar=eview.logicalPart().solid().parameters();
	svPars[name] = std::pair<double,double>(solidPar[3],
						0.5*(solidPar[4]+solidPar[5]));
	std::cout << name << " Layer " << layer << " " << solidPar[3] 
		  << " " << solidPar[4] << " " << solidPar[5] << std::endl;
      }
    }
  }while(eview.next() );
}


//define this as a plug-in
DEFINE_FWK_MODULE(HGCGeometryTester);
