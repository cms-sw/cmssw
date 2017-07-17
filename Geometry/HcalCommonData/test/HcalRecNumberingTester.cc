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

#define EDM_ML_DEBUG

class HcalRecNumberingTester : public edm::one::EDAnalyzer<> {
public:
  explicit HcalRecNumberingTester( const edm::ParameterSet& );
  ~HcalRecNumberingTester() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

HcalRecNumberingTester::HcalRecNumberingTester(const edm::ParameterSet& ) {}

HcalRecNumberingTester::~HcalRecNumberingTester() {}

// ------------ method called to produce the data  ------------
void HcalRecNumberingTester::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  edm::ESHandle<HcalDDDRecConstants> pHSNDC;
  iSetup.get<HcalRecNumberingRecord>().get(pHSNDC);

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
      std::vector<int>         depths = hdc.getDepth(i,false);
      std::cout << "EtaBin[" << i << "]: EtaLimit = (" << etas.first << ":"
		<< etas.second << ")  phiBin = " << fbin << " depths = (";
      for (unsigned int k=0; k<depths.size(); ++k) {
	if (k == 0) std::cout << depths[k];
	else        std::cout << ", " << depths[k];
      }
      std::cout << ")" << std::endl;
    }
    for (int type=0; type<2; ++type) {
      std::pair<int,int> etar = hdc.getEtaRange(type);
      std::cout << "Detector type: " << type << " with eta ranges "
		<< etar.first << ":" << etar.second << std::endl;
      for (int eta=etar.first; eta<=etar.second; ++eta) {
	std::vector<std::pair<int,double> > phis = hdc.getPhis(type+1, eta);
	for (auto & phi : phis) {
	  std::cout << "Type:Eta:phi " << type << ":" << eta << ":" 
		    << phi.first << " Depth range (+z) "
		    << hdc.getMinDepth(type,eta,phi.first,1) << ":" 
		    << hdc.getMaxDepth(type,eta,phi.first,1) << " (-z) "
		    << hdc.getMinDepth(type,eta,phi.first,-1) << ":" 
		    << hdc.getMaxDepth(type,eta,phi.first,-1) << std::endl;
	}
      }
    }
    std::vector<HcalDDDRecConstants::HcalEtaBin> hbar = hdc.getEtaBins(0);
    std::vector<HcalDDDRecConstants::HcalEtaBin> hcap = hdc.getEtaBins(1);
    std::cout << "Topology Mode " << hdc.getTopoMode() 
	      << " HB with " << hbar.size() << " eta sectors and HE with "
	      << hcap.size() << " eta sectors" << std::endl;
    std::vector<HcalCellType> hbcell = hdc.HcalCellTypes(HcalBarrel);
    std::cout << "HB with " << hbcell.size() << " cells" << std::endl;
    unsigned int i1(0), i2(0), i3(0), i4(0);
    for (const auto& cell : hbcell) {
      std::cout << "HB[" << i1 << "] det " << cell.detType() << " zside "
		<< cell.zside() << ":" << cell.halfSize()
		<< " RO " << cell.actualReadoutDirection()
		<< " eta " << cell.etaBin() << ":" << cell.etaMin()
		<< ":" << cell.etaMax() << " phi " << cell.nPhiBins()
		<< ":" << cell.nPhiModule() << ":" << cell.phiOffset()
		<< ":" << cell.phiBinWidth() << ":" << cell.unitPhi()
		<< " depth " << cell.depthSegment()
		<< ":" << cell.depth() << ":" << cell.depthMin()
		<< ":" << cell.depthMax() << ":" << cell.depthType()
		<< std::endl;
      ++i1;
      std::vector<std::pair<int,double>>phis = cell.phis();
      std::cout << "Phis (" << phis.size() << ") :";
      for (const auto& phi : phis) 
	std::cout << " [" << phi.first << ", " << phi.second << "]";
      std::cout << std::endl;
    }
    std::vector<HcalCellType> hecell = hdc.HcalCellTypes(HcalEndcap);
    std::cout << "HE with " << hecell.size() << " cells" << std::endl;
    for (const auto& cell : hecell) {
      std::cout << "HE[" << i2 << "] det " << cell.detType() << " zside "
		<< cell.zside() << ":" << cell.halfSize()
		<< " RO " << cell.actualReadoutDirection()
		<< " eta " << cell.etaBin() << ":" << cell.etaMin()
		<< ":" << cell.etaMax() << " phi " << cell.nPhiBins()
		<< ":" << cell.nPhiModule() << ":" << cell.phiOffset()
		<< ":" << cell.phiBinWidth() << ":" << cell.unitPhi()
		<< " depth " << cell.depthSegment()
		<< ":" << cell.depth() << ":" << cell.depthMin()
		<< ":" << cell.depthMax() << ":" << cell.depthType()
		<< std::endl;
      ++i2;
      std::vector<std::pair<int,double>>phis = cell.phis();
      std::cout << "Phis (" << phis.size() << ") :";
      for (const auto& phi : phis) 
	std::cout << " [" << phi.first << ", " << phi.second << "]";
      std::cout << std::endl;
    }
    std::vector<HcalCellType> hfcell = hdc.HcalCellTypes(HcalForward);
    std::cout << "HF with " << hfcell.size() << " cells" << std::endl;
    for (const auto& cell : hfcell) {
      std::cout << "HF[" << i3 << "] det " << cell.detType() << " zside "
		<< cell.zside() << ":" << cell.halfSize()
		<< " RO " << cell.actualReadoutDirection()
		<< " eta " << cell.etaBin() << ":" << cell.etaMin()
		<< ":" << cell.etaMax() << " phi " << cell.nPhiBins()
		<< ":" << cell.nPhiModule() << ":" << cell.phiOffset()
		<< ":" << cell.phiBinWidth() << ":" << cell.unitPhi()
		<< " depth " << cell.depthSegment()
		<< ":" << cell.depth() << ":" << cell.depthMin()
		<< ":" << cell.depthMax() << ":" << cell.depthType()
		<< std::endl;
      ++i3;
    }
    std::vector<HcalCellType> hocell = hdc.HcalCellTypes(HcalOuter);
    std::cout << "HO with " << hocell.size() << " cells" << std::endl;
    for (const auto& cell : hocell) {
      std::cout << "HO[" << i4 << "] det " << cell.detType() << " zside "
		<< cell.zside() << ":" << cell.halfSize()
		<< " RO " << cell.actualReadoutDirection()
		<< " eta " << cell.etaBin() << ":" << cell.etaMin()
		<< ":" << cell.etaMax() << " phi " << cell.nPhiBins()
		<< ":" << cell.nPhiModule() << ":" << cell.phiOffset()
		<< ":" << cell.phiBinWidth() << ":" << cell.unitPhi()
		<< " depth " << cell.depthSegment()
		<< ":" << cell.depth() << ":" << cell.depthMin()
		<< ":" << cell.depthMax() << ":" << cell.depthType()
		<< std::endl;
      ++i4;
    }
    for (int type=0; type <= 1; ++type ) {
      std::vector<HcalDDDRecConstants::HcalActiveLength> act = hdc.getThickActive(type);
      std::cout << "Hcal type " << type << " has " << act.size() 
		<< " eta/depth segment " << std::endl;
      for (const auto& active : act) {
	std::cout << "zside " << active.zside << " ieta " << active.ieta
		  << " depth " << active.depth << " type " << active.stype
		  << " eta " << active.eta  << " active thickness " 
		  << active.thick << std::endl;
      }
    }

    // Test merging
    std::vector<int> phiSp;
    HcalSubdetector  subdet = HcalSubdetector(hdc.dddConstants()->ldMap()->validDet(phiSp));
    if (subdet == HcalBarrel || subdet == HcalEndcap) {
      int type = (int)(subdet-1);
      std::pair<int,int> etas = hdc.getEtaRange(type);
      for (int eta=etas.first; eta<=etas.second; ++eta) {
	for (int k : phiSp) {
	  int zside = (k>0) ? 1 : -1;
	  int iphi  = (k>0) ? k : -k;
#ifdef EDM_ML_DEBUG
	  std::cout << "Look for Subdet " << subdet << " Zside " << zside
		    << " Eta " << eta << " Phi " << iphi << " depths "
		    << hdc.getMinDepth(type,eta,iphi,zside) << ":"
		    << hdc.getMaxDepth(type,eta,iphi,zside) << std::endl;
#endif
	  std::vector<HcalDetId> ids;
	  for (int depth=hdc.getMinDepth(type,eta,iphi,zside);
	       depth <= hdc.getMaxDepth(type,eta,iphi,zside); ++depth) {
	    HcalDetId id(subdet,zside*eta,iphi,depth);
	    HcalDetId hid = hdc.mergedDepthDetId(id);
	    hdc.unmergeDepthDetId(hid,ids);
	    std::cout << "Input ID " << id << " Merged ID " << hid
		      << " containing " << ids.size() << " IDS:";
	    for (auto id : ids) 
	      std::cout << " " << id;
	    std::cout << std::endl;
	  }
	}
      }
    }
    // Test merging
    for (const auto& cell : hbcell) {
      int ieta  = cell.etaBin()*cell.zside();
      double rz = hdc.getRZ(HcalBarrel,ieta,cell.phis()[0].first,
			    cell.depthSegment());
      std::cout << "HB (eta=" << ieta << ", phi=" << cell.phis()[0].first
		<< ", depth=" << cell.depthSegment() << ") r/z = " << rz 
		<< std::endl;
    }
    for (const auto& cell : hecell) {
      int ieta  = cell.etaBin()*cell.zside();
      double rz = hdc.getRZ(HcalEndcap,ieta,cell.phis()[0].first,
			    cell.depthSegment());
      std::cout << "HE (eta=" << ieta << ", phi=" << cell.phis()[0].first
		<< ", depth=" << cell.depthSegment() << ") r/z = " << rz 
		<< std::endl;
    }
  } else {
    std::cout << "No record found with HcalDDDRecConstants" << std::endl;
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(HcalRecNumberingTester);
