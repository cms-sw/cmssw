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
#include <vector>

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
  ~HGCalNumberingTester() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
private:
  std::string         nameSense_, nameDetector_;
  std::vector<double> positionX_, positionY_;
  int                 increment_;
  bool                reco_, hexType_;
};

HGCalNumberingTester::HGCalNumberingTester(const edm::ParameterSet& iC) {
  nameSense_    = iC.getParameter<std::string>("NameSense");
  nameDetector_ = iC.getParameter<std::string>("NameDevice");
  positionX_    = iC.getParameter<std::vector<double> >("LocalPositionX");
  positionY_    = iC.getParameter<std::vector<double> >("LocalPositionY");
  increment_    = iC.getParameter<int>("Increment");
  hexType_      = iC.getParameter<bool>("HexType");
  reco_         = iC.getParameter<bool>("Reco");
  std::string unit("mm");
  if (reco_) {
    for (unsigned int k=0; k<positionX_.size(); ++k) {
      positionX_[k] /= CLHEP::cm; 
      positionY_[k] /= CLHEP::cm; 
    }
    unit = "cm";
  } else {
    for (unsigned int k=0; k<positionX_.size(); ++k) {
      positionX_[k] /= CLHEP::mm;
      positionY_[k] /= CLHEP::mm;
    }
  }
  std::cout << "Test numbering for " << nameDetector_ <<" using constants of "
	    << nameSense_ << " at " << positionX_.size() << " local positions "
	    << "for every " << increment_ << " layers for HexType " 
	    << hexType_ << " and  RecoFlag " << reco_ << std::endl;
  for (unsigned int k=0; k<positionX_.size(); ++k) 
    std::cout << "Position[" << k << "] " << positionX_[k] << " " << unit
	      << ", " << positionY_[k] << " " << unit << std::endl;
}

HGCalNumberingTester::~HGCalNumberingTester() {}

// ------------ method called to produce the data  ------------
void HGCalNumberingTester::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {
  
  edm::ESHandle<HGCalDDDConstants> pHGNDC;

  iSetup.get<IdealGeometryRecord>().get(nameSense_,pHGNDC);
  const HGCalDDDConstants hgdc(*pHGNDC);
  std::cout << nameDetector_ << " Layers = " << hgdc.layers(reco_) 
	    << " Sectors = " << hgdc.sectors() << " Minimum Slope = "
	    << hgdc.minSlope() << std::endl;
  if (hexType_) {
    std::cout << "Minimum Wafer # " << hgdc.waferMin() << " Mamximum Wafer # "
	      << hgdc.waferMax() << " Wafer counts " << hgdc.waferCount(0)
	      << ":" << hgdc.waferCount(1) << std::endl;
    for (unsigned int i=0; i<hgdc.layers(true); ++i) 
      std::cout << "Layer " << i+1 << " Wafers " << hgdc.wafers(i+1,0) << ":"
		<< hgdc.wafers(i+1,1) << ":" << hgdc.wafers(i+1,2) <<std::endl;
  }
  std::cout << std::endl << std::endl;
  std::pair<int,int> kxy, lxy;
  std::pair<float,float> xy;
  std::string        flg;
  HGCalParameters::hgtrap mytr = hgdc.getModule(0,hexType_,reco_);
  bool halfCell(false);
  if (!hexType_) halfCell = ((mytr.alpha) > 0);
  int   subsec   = (halfCell) ? 1 : 0;
  for (unsigned int k=0; k<positionX_.size(); ++k) {
    float localx(positionX_[k]), localy(positionY_[k]);
    for (unsigned int i=0; i<hgdc.layers(reco_); ++i) {
      kxy = hgdc.assignCell(localx,localy,i+1,subsec,reco_);
      xy  = hgdc.locateCell(kxy.second,i+1,kxy.first,reco_);
      lxy = hgdc.assignCell(xy.first,xy.second,i+1,0,reco_);
      flg = (kxy == lxy) ? " " : " ***** Error *****";
      std::cout << "Input: (" << localx << "," << localy << "," << i+1 
		<< ", " << subsec << "), assignCell o/p (" << kxy.first << ", "
		<< kxy.second << ") locateCell o/p (" << xy.first << ", " 
		<< xy.second << ")," << " final (" << lxy.first << ", " 
		<< lxy.second << ")" << flg << std::endl;
      kxy = hgdc.assignCell(-localx,-localy,i+1,subsec,reco_);
      xy  = hgdc.locateCell(kxy.second,i+1,kxy.first,reco_);
      lxy = hgdc.assignCell(xy.first,xy.second,i+1,0,reco_);
      flg = (kxy == lxy) ? " " : " ***** Error *****";
      std::cout << "Input: (" <<-localx << "," <<-localy << "," << i+1 
		<< ", " << subsec << "), assignCell o/p (" << kxy.first << ", "
		<< kxy.second << ") locateCell o/p (" << xy.first << ", " 
		<< xy.second << ")," << " final (" << lxy.first << ", " 
		<< lxy.second << ")" << flg << std::endl;
      if (k == 0 && i==0) {
	std::vector<int> ncells = hgdc.numberCells(i+1,reco_);
	std::cout << "Layer " << i+1 << " with " << ncells.size() << " rows\n";
	int ntot(0);
	for (unsigned int k=0; k<ncells.size(); ++k) {
	  ntot += ncells[k];
	  std::cout << "Row " << k << " with " << ncells[k] << " cells\n";
	}
	std::cout << "Total Cells " << ntot << ":" << hgdc.maxCells(i+1,reco_) 
		  << std::endl;
      }
      i += increment_;
      if (halfCell) subsec = 1-subsec;
    }
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(HGCalNumberingTester);
