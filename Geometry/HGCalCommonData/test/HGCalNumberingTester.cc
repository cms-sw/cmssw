// -*- C++ -*-
//
// Package:    HGCalNumberingTester
// Class:      HGCalNumberingTester
// 
/**\class HGCalNumberingTester HGCalNumberingTester.cc test/HGCalNumberingTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2014/03/21
//
//


// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <string>

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
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

class HGCalNumberingTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalNumberingTester( const edm::ParameterSet& );
  ~HGCalNumberingTester();

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
private:
  std::string nameSense_, nameDetector_;
  double      position_;
  int         increment_;
};

HGCalNumberingTester::HGCalNumberingTester(const edm::ParameterSet& iC) {
  nameSense_    = iC.getParameter<std::string>("NameSense");
  nameDetector_ = iC.getParameter<std::string>("NameDevice");
  position_     = iC.getParameter<double>("LocalPosition")*CLHEP::mm;
  increment_    = iC.getParameter<int>("Increment");
  std::cout << "Test numbering for " << nameDetector_ <<" using constants of "
	    << nameSense_ << " at local position of " << position_/CLHEP::mm
	    << " mm for every " << increment_ << " layers" << std::endl;
}

HGCalNumberingTester::~HGCalNumberingTester() {}

// ------------ method called to produce the data  ------------
void HGCalNumberingTester::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {
  
  edm::ESHandle<HGCalDDDConstants> pHGNDC;

  iSetup.get<IdealGeometryRecord>().get(nameSense_,pHGNDC);
  const HGCalDDDConstants hgdc(*pHGNDC);
  std::cout << nameDetector_ << " Layers = " << hgdc.layers(false) 
	    << " Sectors = " << hgdc.sectors() << std::endl;
  std::pair<int,int> kxy, lxy;
  std::pair<float,float> xy;
  HGCalParameters::hgtrap mytr = hgdc.getModule(0,false,false);
  bool  halfCell = ((mytr.alpha) > 0);
  int   subsec   = (halfCell) ? 1 : 0;
  float localx(position_), localy(position_);
  for (unsigned int i=0; i<hgdc.layers(false); ++i) {
    kxy = hgdc.assignCell(localx,localy,i+1,subsec,false);
    xy  = hgdc.locateCell(kxy.second,i+1,kxy.first,false);
    lxy = hgdc.assignCell(xy.first,xy.second,i+1,0,false);
    std::cout << "Input: (" << localx << "," << localy << "," << i+1 
	      << ", " << subsec << "), assignCell o/p (" << kxy.first << ", " 
	      << kxy.second << ") locateCell o/p (" << xy.first << ", " 
	      << xy.second << ")," << " final (" << lxy.first << ", " 
	      << lxy.second << ")" << std::endl;
    kxy = hgdc.assignCell(-localx,-localy,i+1,subsec,false);
    xy  = hgdc.locateCell(kxy.second,i+1,kxy.first,false);
    lxy = hgdc.assignCell(xy.first,xy.second,i+1,0,false);
    std::cout << "Input: (" <<-localx << "," <<-localy << "," << i+1 
	      << ", " << subsec << "), assignCell o/p (" << kxy.first << ", " 
	      << kxy.second << ") locateCell o/p (" << xy.first << ", " 
	      << xy.second << ")," << " final (" << lxy.first << ", " 
	      << lxy.second << ")" << std::endl;
    std::vector<int> ncells = hgdc.numberCells(i+1,false);
    std::cout << "Layer " << i+1 << " with " << ncells.size() << " rows\n";
    int ntot(0);
    for (unsigned int k=0; k<ncells.size(); ++k) {
      ntot += ncells[k];
      std::cout << "Row " << k << " with " << ncells[k] << " cells\n";
    }
    std::cout << "Total Cells " << ntot << ":" << hgdc.maxCells(i+1,false) 
	      << std::endl;
    i += increment_;
    if (halfCell) subsec = 1-subsec;
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(HGCalNumberingTester);
