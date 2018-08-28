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
  int                 increment_, detType_;
  bool                reco_;
};

HGCalNumberingTester::HGCalNumberingTester(const edm::ParameterSet& iC) {
  nameSense_    = iC.getParameter<std::string>("NameSense");
  nameDetector_ = iC.getParameter<std::string>("NameDevice");
  positionX_    = iC.getParameter<std::vector<double> >("LocalPositionX");
  positionY_    = iC.getParameter<std::vector<double> >("LocalPositionY");
  increment_    = iC.getParameter<int>("Increment");
  detType_      = iC.getParameter<int>("DetType");
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
	    << "for every " << increment_ << " layers for DetType " 
	    << detType_ << " and  RecoFlag " << reco_ << std::endl;
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
  if (detType_ != 0) {
    std::cout << "Minimum Wafer # " << hgdc.waferMin() << " Mamximum Wafer # "
	      << hgdc.waferMax() << " Wafer counts " << hgdc.waferCount(0)
	      << ":" << hgdc.waferCount(1) << std::endl;
    for (unsigned int i=0; i<hgdc.layers(true); ++i) 
      std::cout << "Layer " << i+1 << " Wafers " << hgdc.wafers(i+1,0) << ":"
		<< hgdc.wafers(i+1,1) << ":" << hgdc.wafers(i+1,2) <<std::endl;
  }
  std::cout << std::endl << std::endl;
  std::pair<float,float> xy;
  std::string        flg;
  int   subsec(0);
  int   loff = hgdc.firstLayer();
  double scl = (reco_ ? 10.0 : 1.0);
  for (unsigned int k=0; k<positionX_.size(); ++k) {
    float localx(positionX_[k]), localy(positionY_[k]);
    for (unsigned int i=0; i<hgdc.layers(reco_); ++i) {
      if (detType_ == 1) {
	std::pair<int,int> kxy, lxy;
	kxy = hgdc.assignCell(localx,localy,i+loff,subsec,reco_);
	xy  = hgdc.locateCell(kxy.second,i+loff,kxy.first,reco_);
	lxy = hgdc.assignCell(xy.first,xy.second,i+loff,0,reco_);
	flg = (kxy == lxy) ? " " : " ***** Error *****";
	std::cout << "Input: (" << localx << "," << localy << "," << i+loff
		  << ", " << subsec << "), assignCell o/p (" << kxy.first 
		  << ", " << kxy.second << ") locateCell o/p (" << xy.first 
		  << ", " << xy.second << ")," << " final (" << lxy.first 
		  << ", " << lxy.second << ")" << flg << std::endl;
	kxy = hgdc.assignCell(-localx,-localy,i+loff,subsec,reco_);
	xy  = hgdc.locateCell(kxy.second,i+loff,kxy.first,reco_);
	lxy = hgdc.assignCell(xy.first,xy.second,i+loff,0,reco_);
	flg = (kxy == lxy) ? " " : " ***** Error *****";
	std::cout << "Input: (" <<-localx << "," <<-localy << "," << i+loff
		  << ", " << subsec << "), assignCell o/p (" << kxy.first 
		  << ", " << kxy.second << ") locateCell o/p (" << xy.first
		  << ", " << xy.second << ")," << " final (" << lxy.first 
		  << ", " << lxy.second << ")" << flg << std::endl;
      } else if (detType_ == 0) {
	std::array<int,3> kxy, lxy;
	double zpos = hgdc.waferZ(i+loff,reco_);
	kxy = hgdc.assignCellTrap(localx,localy,zpos,i+loff,reco_);
	xy  = hgdc.locateCellTrap(i+loff,kxy[0],kxy[1],reco_);
	lxy = hgdc.assignCellTrap(xy.first,xy.second,zpos,i+loff,reco_);
	flg = (kxy == lxy) ? " " : " ***** Error *****";
	std::cout << "Input: (" << localx << "," << localy << "," << zpos
		  << ", " << i+loff << "), assignCell o/p (" << kxy[0] << ":" 
		  << kxy[1] << ":" << kxy[2] << ") locateCell o/p (" 
		  << xy.first << ", " << xy.second << ")," << " final (" 
		  << lxy[0] << ":" << lxy[1] << ":" << lxy[2] << ") Dist " 
		  << hgdc.distFromEdgeTrap(scl*localx,scl*localy,scl*zpos)
		  << " " << flg << std::endl;
	kxy = hgdc.assignCellTrap(-localx,-localy,zpos,i+loff,reco_);
	xy  = hgdc.locateCellTrap(i+loff,kxy[0],kxy[1],reco_);
	lxy = hgdc.assignCellTrap(xy.first,xy.second,zpos,i+loff,reco_);
	flg = (kxy == lxy) ? " " : " ***** Error *****";
	std::cout << "Input: (" <<-localx << "," <<-localy << "," << zpos
		  << ", " << i+loff << "), assignCell o/p (" << kxy[0] << ":" 
		  << kxy[1] << ":" << kxy[2] << ") locateCell o/p (" 
		  << xy.first << ", " << xy.second << ")," << " final (" 
		  << lxy[0] << ":" << lxy[1] << ":" << lxy[2] << ") Dist " 
		  << hgdc.distFromEdgeTrap(scl*localx,scl*localy,scl*zpos)
		  << " " << flg << std::endl;
      } else {
	std::array<int,5> kxy, lxy;
	kxy = hgdc.assignCellHex(localx,localy,i+loff,reco_);
	xy  = hgdc.locateCell(i+loff,kxy[0],kxy[1],kxy[3],kxy[4],reco_,true);
	lxy = hgdc.assignCellHex(xy.first,xy.second,i+loff,reco_);
	flg = (kxy == lxy) ? " " : " ***** Error *****";
	double zpos = hgdc.waferZ(i+loff,reco_);
	std::cout << "Input: (" << localx << "," << localy << ", " << i+loff 
		  << "), assignCell o/p (" << kxy[0] << ":" << kxy[1] 
		  << ":" << kxy[2] << ":" << kxy[3] << ":" << kxy[4]
		  << ") locateCell o/p (" << xy.first << ", " << xy.second 
		  << ")," << " final ("  << lxy[0] << ":" << lxy[1] << ":" 
		  << lxy[2] << ":" << lxy[3] << ":" << lxy[4] << ") Dist "
		  << hgdc.distFromEdgeHex(scl*localx,scl*localy,scl*zpos)
		  << " " << flg << std::endl;
	kxy = hgdc.assignCellHex(-localx,-localy,i+loff,reco_);
	xy  = hgdc.locateCell(i+loff,kxy[0],kxy[1],kxy[3],kxy[4],reco_,true);
	lxy = hgdc.assignCellHex(xy.first,xy.second,i+loff,reco_);
	flg = (kxy == lxy) ? " " : " ***** Error *****";
	std::cout << "Input: (" <<-localx << "," <<-localy << ", " << i+loff 
		  << "), assignCell o/p (" << kxy[0] << ":" << kxy[1] 
		  << ":" << kxy[2] << ":" << kxy[3] << ":" << kxy[4]
		  << ") locateCell o/p (" << xy.first << ", " << xy.second 
		  << ")," << " final ("  << lxy[0] << ":" << lxy[1] << ":" 
		  << lxy[2] << ":" << lxy[3] << ":" << lxy[4] << ") Dist "
		  << hgdc.distFromEdgeHex(scl*localx,scl*localy,scl*zpos)
		  << " " << flg << std::endl;
      }
      if (k == 0 && i==0 && detType_ == 1) {
	std::vector<int> ncells = hgdc.numberCells(i+1,reco_);
	std::cout << "Layer " << i+1 << " with " << ncells.size() << " rows"
		  << std::endl;
	int ntot(0);
	for (unsigned int k=0; k<ncells.size(); ++k) {
	  ntot += ncells[k];
	  std::cout << "Row " << k << " with " << ncells[k] << " cells\n";
	}
	std::cout << "Total Cells " << ntot << ":" << hgdc.maxCells(i+1,reco_)
		  << std::endl;
      }
      i += increment_;
    }
  }

  // Test the range variables
  if (hgdc.getParameter()->detectorType_ > 0) {
    unsigned int kk(0);
    for (auto const& zz : hgdc.getParameter()->zLayerHex_) {
      std::pair<double,double> rr = hgdc.rangeR(zz, true);
      std::cout << "[" << kk << "]\t z = " << zz << "\t rMin = " << rr.first
		<< "\t rMax = " << rr.second << std::endl;
      ++kk;
    }
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(HGCalNumberingTester);
