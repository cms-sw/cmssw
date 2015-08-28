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
// $Id: HGCalNumberingTester.cc,v 1.0 2014/032/21 14:06:07 sunanda Exp $
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
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "CoralBase/Exception.h"

class HGCalNumberingTester : public edm::one::EDAnalyzer<>
{
public:
  explicit HGCalNumberingTester( const edm::ParameterSet& );
  ~HGCalNumberingTester();

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

HGCalNumberingTester::HGCalNumberingTester(const edm::ParameterSet& ) {}

HGCalNumberingTester::~HGCalNumberingTester() {}

// ------------ method called to produce the data  ------------
void HGCalNumberingTester::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {
  
  edm::ESHandle<HGCalDDDConstants> pHGNDC;

  iSetup.get<IdealGeometryRecord>().get("HGCalEESensitive",pHGNDC);
  const HGCalDDDConstants hgeedc(*pHGNDC);
  std::cout << "EE Layers = " << hgeedc.layers(false) << " Sectors = " 
	    << hgeedc.sectors() << std::endl;
  std::pair<int,int> kxy, lxy;
  std::pair<float,float> xy;
  float localx(5.0), localy(5.0);
  for (unsigned int i=0; i<hgeedc.layers(false); ++i) {
    kxy = hgeedc.assignCell(localx,localy,i+1,0,false);
    xy  = hgeedc.locateCell(kxy.second,i+1,kxy.first,false);
    lxy = hgeedc.assignCell(xy.first,xy.second,i+1,0,false);
    std::cout << "Input: (" << localx << "," << localy << "," << i+1 
	      << ", 0), assignCell o/p (" << kxy.first << ", " << kxy.second 
	      << ") loatCell o/p (" << xy.first << ", " << xy.second << ")," 
	      << " final (" << lxy.first << ", " << lxy.second << ")"
	      << std::endl;
    kxy = hgeedc.assignCell(-localx,-localy,i+1,0,false);
    xy  = hgeedc.locateCell(kxy.second,i+1,kxy.first,false);
    lxy = hgeedc.assignCell(xy.first,xy.second,i+1,0,false);
    std::cout << "Input: (" <<-localx << "," <<-localy << "," << i+1 
	      << ", 0), assignCell o/p (" << kxy.first << ", " << kxy.second 
	      << ") loatCell o/p (" << xy.first << ", " << xy.second << ")," 
	      << " final (" << lxy.first << ", " << lxy.second << ")" 
	      << std::endl;
    std::vector<int> ncells = hgeedc.numberCells(i+1,false);
    std::cout << "Layer " << i+1 << " with " << ncells.size() << " rows\n";
    int ntot(0);
    for (unsigned int k=0; k<ncells.size(); ++k) {
      ntot += ncells[k];
      std::cout << "Row " << k << " with " << ncells[k] << " cells\n";
    }
    std::cout << "Total Cells " << ntot << ":" << hgeedc.maxCells(i+1,false) 
	      << std::endl;
    i += 19;
  }

  iSetup.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive",pHGNDC);
  const HGCalDDDConstants hghesidc(*pHGNDC);
  std::cout << "HE Silicon Layers = " << hghesidc.layers(false) 
	    << " Sectors = " << hghesidc.sectors() << std::endl;
  for (unsigned int i=0; i<hghesidc.layers(false); ++i) {
    kxy = hghesidc.assignCell(localx,localy,i+1,0,false);
    xy  = hghesidc.locateCell(kxy.second,i+1,kxy.first,false);
    lxy = hghesidc.assignCell(xy.first,xy.second,i+1,0,false);
    std::cout << "Input: (" << localx << "," << localy << "," << i+1 
	      << ", 0), assignCell o/p (" << kxy.first << ", " << kxy.second 
	      << ") loatCell o/p (" << xy.first << ", " << xy.second << ")," 
	      << " final (" << lxy.first << ", " << lxy.second << ")" 
	      << std::endl;
    kxy = hghesidc.assignCell(-localx,-localy,i+1,0,false);
    xy  = hghesidc.locateCell(kxy.second,i+1,kxy.first,false);
    lxy = hghesidc.assignCell(xy.first,xy.second,i+1,0,false);
    std::cout << "Input: (" <<-localx << "," <<-localy << "," << i+1 
	      << ", 0), assignCell o/p (" << kxy.first << ", " << kxy.second 
	      << ") loatCell o/p (" << xy.first << ", " << xy.second << ")," 
	      << " final (" << lxy.first << ", " << lxy.second << ")" 
	      << std::endl;
    std::vector<int> ncells = hghesidc.numberCells(i+1,false);
    std::cout << "Layer " << i+1 << " with " << ncells.size() << " rows\n";
    int ntot(0);
    for (unsigned int k=0; k<ncells.size(); ++k) {
      ntot += ncells[k];
      std::cout << "Row " << k << " with " << ncells[k] << " cells\n";
    }
    std::cout << "Total Cells " << ntot << ":" << hghesidc.maxCells(i+1,false) 
	      << std::endl;
    i += 9;
  }

  iSetup.get<IdealGeometryRecord>().get("HGCalHEScintillatorSensitive",pHGNDC);
  const HGCalDDDConstants hghescdc(*pHGNDC);
  std::cout << "HE Scintillator Layers = " << hghescdc.layers(false) 
	    << " Sectors = " << hghescdc.sectors() << std::endl;
  std::vector<HGCalDDDConstants::hgtrap>::const_iterator itr = hghescdc.getFirstModule(false);
  int subsec = ((itr->alpha) > 0) ? 1 : 0;
  for (unsigned int i=0; i<hghescdc.layers(false); ++i) {
    kxy = hghescdc.assignCell(localx,localy,i+1,subsec,false);
    xy  = hghescdc.locateCell(kxy.second,i+1,kxy.first,false);
    lxy = hghescdc.assignCell(xy.first,xy.second,i+1,0,false);
    std::cout << "Input: (" << localx << "," << localy << "," << i+1 
	      << "," << subsec << "), assignCell o/p (" << kxy.first << ", " 
	      << kxy.second  << ") loatCell o/p (" << xy.first << ", " 
	      << xy.second << "), final (" << lxy.first << ", " << lxy.second 
	      << ")" << std::endl;
    kxy = hghescdc.assignCell(-localx,-localy,i+1,subsec,false);
    xy  = hghescdc.locateCell(kxy.second,i+1,kxy.first,false);
    lxy = hghescdc.assignCell(xy.first,xy.second,i+1,0,false);
    std::cout << "Input: (" <<-localx << "," <<-localy << "," << i+1 
	      << "," << subsec << "), assignCell o/p (" << kxy.first << ", " 
	      << kxy.second  << ") loatCell o/p (" << xy.first << ", " 
	      << xy.second << "), final (" << lxy.first << ", " << lxy.second 
	      << ")"  << std::endl;
    std::vector<int> ncells = hghescdc.numberCells(i+1,false);
    std::cout << "Layer " << i+1 << " with " << ncells.size() << " rows\n";
    int ntot(0);
    for (unsigned int k=0; k<ncells.size(); ++k) {
      ntot += ncells[k];
      std::cout << "Row " << k << " with " << ncells[k] << " cells\n";
    }
    std::cout << "Total Cells " << ntot << ":" << hghescdc.maxCells(i+1,false) 
	      << std::endl;
    i += 10;
    subsec = 1-subsec;
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(HGCalNumberingTester);
