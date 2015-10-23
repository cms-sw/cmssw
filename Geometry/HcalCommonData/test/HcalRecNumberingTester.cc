// -*- C++ -*-
//
// Package:    HcalRecNumberingTester
// Class:      HcalRecNumberingTester
// 
/**\class HcalRecNumberingTester HcalRecNumberingTester.cc test/HcalRecNumberingTester.cc

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

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"

#include "CoralBase/Exception.h"

class HcalRecNumberingTester : public edm::one::EDAnalyzer<> {
public:
  explicit HcalRecNumberingTester( const edm::ParameterSet& );
  ~HcalRecNumberingTester();

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

HcalRecNumberingTester::HcalRecNumberingTester(const edm::ParameterSet& ) {}

HcalRecNumberingTester::~HcalRecNumberingTester() {}

// ------------ method called to produce the data  ------------
void HcalRecNumberingTester::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  edm::ESHandle<HcalDDDRecConstants> pHSNDC;
  iSetup.get<HcalRecNumberingRecord>().get( pHSNDC );

  if (pHSNDC.isValid()) {
    std::cout << "about to de-reference the edm::ESHandle<HcalDDDRecConstants> pHSNDC" << std::endl;
    const HcalDDDRecConstants hdc (*pHSNDC);
    std::cout << "about to getPhiOff and getPhiBin for 0..2" << std::endl;
    int neta = hdc.getNEta();
    std::cout << neta << " eta bins with phi off set for barrel = " 
	      << hdc.getPhiOff(0) << ", endcap = " << hdc.getPhiOff(1) 
	      << std::endl;
    for (int i=0; i<neta; ++i) {
      std::pair<double,double> etas   = hdc.getEtaLimit(i);
      double                   fbin   = hdc.getPhiBin(i);
      std::vector<int>         depths = hdc.getDepth(i);
      std::cout << "EtaBin[" << i << "]: EtaLimit = (" << etas.first << ":"
		<< etas.second << ")  phiBin = " << fbin << " depths = (";
      for (unsigned int k=0; k<depths.size(); ++k) {
	if (k == 0) std::cout << depths[k];
	else        std::cout << ", " << depths[k];
      }
      std::cout << ")" << std::endl;
    }
    std::vector<HcalDDDRecConstants::HcalEtaBin> hbar = hdc.getEtaBins(0);
    std::vector<HcalDDDRecConstants::HcalEtaBin> hcap = hdc.getEtaBins(1);
    std::cout << "Topology Mode " << hdc.getTopoMode() 
	      << " HB with " << hbar.size() << " eta sectors and HE with "
	      << hcap.size() << " eta sectors" << std::endl;
    std::vector<HcalCellType> hbcell = hdc.HcalCellTypes(HcalBarrel);
    std::vector<HcalCellType> hecell = hdc.HcalCellTypes(HcalEndcap);
    std::cout << "HB with " << hbcell.size() << " cells" << std::endl;
    for (unsigned int i=0; i<hbcell.size(); ++i)
      std::cout << "HB[" << i << "] det " << hbcell[i].detType() << " halves "
		<< hbcell[i].nHalves() << ":" << hbcell[i].halfSize()
		<< " RO " << hbcell[i].actualReadoutDirection()
		<< " eta " << hbcell[i].etaBin() << ":" << hbcell[i].etaMin()
		<< ":" << hbcell[i].etaMax() << " phi " << hbcell[i].nPhiBins()
		<< ":" << hbcell[i].nPhiModule() << ":" << hbcell[i].phiOffset()
		<< ":" << hbcell[i].phiBinWidth() << ":" << hbcell[i].unitPhi()
		<< " depth " << hbcell[i].depthSegment()
		<< ":" << hbcell[i].depth() << ":" << hbcell[i].depthMin()
		<< ":" << hbcell[i].depthMax() << ":" << hbcell[i].depthType()
		<< std::endl;
    std::cout << "HE with " << hecell.size() << " cells" << std::endl;
    for (unsigned int i=0; i<hecell.size(); ++i)
      std::cout << "HE[" << i << "] det " << hecell[i].detType() << " halves "
		<< hecell[i].nHalves() << ":" << hecell[i].halfSize()
		<< " RO " << hecell[i].actualReadoutDirection()
		<< " eta " << hecell[i].etaBin() << ":" << hecell[i].etaMin()
		<< ":" << hecell[i].etaMax() << " phi " << hecell[i].nPhiBins()
		<< ":" << hecell[i].nPhiModule() << ":" << hecell[i].phiOffset()
		<< ":" << hecell[i].phiBinWidth() << ":" << hecell[i].unitPhi()
		<< " depth " << hecell[i].depthSegment()
		<< ":" << hecell[i].depth() << ":" << hecell[i].depthMin()
		<< ":" << hecell[i].depthMax() << ":" << hecell[i].depthType()
		<< std::endl;
    for (int type=0; type <= 1; ++type ) {
      std::vector<HcalDDDRecConstants::HcalActiveLength> act = hdc.getThickActive(type);
      std::cout << "Hcal type " << type << " has " << act.size() 
		<< " eta/depth segment " << std::endl;
    }
  } else {
    std::cout << "No record found with HcalDDDRecConstants" << std::endl;
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(HcalRecNumberingTester);
