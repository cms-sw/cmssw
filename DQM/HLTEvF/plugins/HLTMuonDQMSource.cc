// -*- C++ -*-
//
// Package:    HLTMuonDQMSource
// Class:      HLTMuonDQMSource
// 
/**\class HLTMuonDQMSource 

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Muriel VANDER DONCKT *:0
//         Created:  Wed Dec 12 09:55:42 CET 2007
// $Id: HLTMuonDQMSource.cc,v 1.39 2010/05/02 08:13:06 dellaric Exp $
// Modification:  Hwidong Yoo (Purdue University)
// contact: hdyoo@cern.ch
//
//



#include "DQM/HLTEvF/interface/HLTMuonDQMSource.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "TMath.h" 

//using namespace std;
//using namespace edm;
//using namespace reco;
//using namespace l1extra;
//
// constructors and destructor
//
HLTMuonDQMSource::HLTMuonDQMSource( const edm::ParameterSet& ps ) :counterEvt_(0), nTrig_(0)
{
  parameters_ = ps;
  verbose_ = parameters_.getUntrackedParameter < bool > ("verbose", false);
  monitorName_ = parameters_.getUntrackedParameter<std::string>("monitorName","HLT/HLTMonMuon");
  prescaleEvt_ = parameters_.getUntrackedParameter<int>("prescaleEvt", -1);
  coneSize_ = parameters_.getUntrackedParameter<double>("coneSize", 0.24);
  l2seedscollectionTag_ = parameters_.getUntrackedParameter<edm::InputTag>("l2MuonSeedTag",edm::InputTag("hltL2MuonSeeds"));
  l3seedscollectionTag_ = parameters_.getUntrackedParameter<edm::InputTag>("l3MuonSeedTag",edm::InputTag("hltL3TrajectorySeed"));
  l2collectionTag_ = parameters_.getUntrackedParameter<edm::InputTag>("l2MuonTag",edm::InputTag("hltL2MuonCandidates"));
  l3collectionTag_ = parameters_.getUntrackedParameter<edm::InputTag>("l3MuonTag",edm::InputTag("hltL3MuonCandidates"));
  l2isolationTag_ = parameters_.getUntrackedParameter<edm::InputTag>("l2IsolationTag",edm::InputTag("hltL2MuonIsolations"));
  l3isolationTag_ = parameters_.getUntrackedParameter<edm::InputTag>("l3IsolationTag",edm::InputTag("hltL3MuonIsolations"));

  TrigResultInput = parameters_.getUntrackedParameter<edm::InputTag>("TrigResultInputTag", edm::InputTag("TriggerResults"));

  dbe_ = 0 ;
  dbe_ = edm::Service < DQMStore > ().operator->();
  dbe_->setVerbose(0);
 
  outputFile_ =
    parameters_.getUntrackedParameter < std::string > ("outputFile", "");
  if (outputFile_.size() != 0) {
    edm::LogWarning("HLTMuonDQMSource") << "Muon HLT Monitoring histograms will be saved to " 
				   << outputFile_ << std::endl;
  }
  else {
    outputFile_ = "HLTMuonDQM.root";
  }
  
  bool disable =
    parameters_.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) {
    outputFile_ = "";
  }
  
  if (dbe_ != NULL) {
    dbe_->setCurrentFolder(monitorName_);
 }
  
  std::vector<edm::ParameterSet> filters = parameters_.getParameter<std::vector<edm::ParameterSet> >("filters");

  for(std::vector<edm::ParameterSet>::iterator filterconf = filters.begin() ; filterconf != filters.end() ; filterconf++){
    theDirectoryName.push_back(filterconf->getParameter<std::string>("directoryName"));
    //theHLTCollectionLevel.push_back(filterconf->getParameter<std::string>("level"));
    //theTriggerBits.push_back(filterconf->getParameter<std::vector<std::string> >("triggerBits"));
    std::string _tmp_level = filterconf->getParameter<std::string>("directoryName");
    std::vector<std::string> _tmp_bits = filterconf->getParameter<std::vector<std::string> >("triggerBits");
    for( size_t i = 0; i < _tmp_bits.size(); ++i ) {
	theTriggerBits.push_back(_tmp_bits[i]);
	theHLTCollectionLevel.push_back(_tmp_level);
    }
  }
  
  // L1PassThrough, L2PassThrough, L3PassThrough
  nTrigs = theDirectoryName.size();

  for( int trig = 0; trig < nTrigs; trig++ ) striggers_[trig] = "";
}


HLTMuonDQMSource::~HLTMuonDQMSource()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//--------------------------------------------------------
void HLTMuonDQMSource::beginJob()
{
  if (dbe_) {
    //dbe_->setCurrentFolder("monitorName_");
    if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
    edm::LogInfo("HLTMuonDQMSource") << "===>DQM event prescale = " << prescaleEvt_ << " events "<< std::endl;
    
    
    /// book some histograms here
    int NBINS = 50; XMIN = 0; XMAX = 50;
    
    // create and cd into new folder
    char name[512], title[512];
    double pt_max;
    std::string dirname;

    for( int trig = 0; trig < nTrigs; trig++ ) {
      dirname = theDirectoryName[trig]+"/";

      for(unsigned int i = 0; i < theHLTCollectionLevel.size(); ++i){
	if(theHLTCollectionLevel[i] == theDirectoryName[trig]) {
	  if(!strcmp(striggers_[trig].c_str(), "")) striggers_[trig] = theTriggerBits[i];
	  else striggers_[trig] += ", " + theTriggerBits[i];
	}
      }
      dbe_->setCurrentFolder(monitorName_ + dirname);
      sprintf(name,"%s triggers",theDirectoryName[trig].c_str());
      dbe_->bookString(name,striggers_[trig]);

      for ( int level = 1; level < 7; ++level ) {
	if( level < 4 ) sprintf(name,"Level%i",level);
	else if (level == 4 ) sprintf(name,"Level%iSeed", level-2);
	else if (level == 5 ) sprintf(name,"Level%iSeed", level-2);
	else if (level == 6 ) sprintf(name,"Level2Isolation");
	
	if( level == 1 ) pt_max = 145;
	else pt_max = 200;
	dbe_->setCurrentFolder(monitorName_+dirname+name);
	if( level == 1 ) hl1quality[trig] = dbe_->book1D("h1L1Quality","GMT quality Flag", 8, 0., 8.);
	if( level == 2 ) {
	    hnHits[trig][level-1] = dbe_->book1D(name,title, NBINS, 0., 100.);
	    hnValidHits[trig] = dbe_->book1D("HLTMuonL2_nValidHits", "L2 Number of Valid Hits", NBINS, 0., 100.);
	    hnValidHits[trig]->setAxisTitle("Number of Valid Hits", 1);
	}
	if( level == 3 ) {
	  hnTkValidHits[trig] = dbe_->book1D("HLTMuonL3_nTkValidHits", "L3 Number of Valid Tracker Hits", NBINS, 0., 100.);
	  hnTkValidHits[trig]->setAxisTitle("Number of Valid Tracker Hits", 1);
	  hnMuValidHits[trig] = dbe_->book1D("HLTMuonL3_nMuValidHits", "L3 Number of Valid Muon Hits", NBINS, 0., 100.);
	  hnMuValidHits[trig]->setAxisTitle("Number of Valid Muon Hits", 1);
	}
	if( level == 1 ) {
	  int eta_nbin = 62;
	  float eta_bins[63] = {
	      		 -2.40, -2.35, -2.30, -2.25, -2.20, -2.15, -2.10, -2.05,
			 -2.00, -1.95, -1.90, -1.85, -1.80, -1.75, -1.70, -1.60,
			 -1.50, -1.40, -1.30, -1.20, -1.10, -1.00, -0.90, -0.80,
			 -0.70, -0.60, -0.50, -0.40, -0.30, -0.20, -0.10, -0.00,
	                 0.10,  0.20,  0.30,  0.40,  0.50,  0.60,  0.70,  0.80,
	                 0.90,  1.00,  1.10,  1.20,  1.30,  1.40,  1.50,  1.60,
			 1.70,  1.75,  1.80,  1.85,  1.90,  1.95,  2.00,  2.05,
			 2.10,  2.15,  2.20,  2.25,  2.30,  2.35,  2.40
			 };
	  int phi_nbin = 144;		 
	  float phi_bins[145];
	  for( int ibin = 0; ibin < 145; ibin++ ) {
	      phi_bins[ibin] = -3.15 + (ibin)*(6.30/144);
	  }
	  float pt_bins[51];
	  for( int ibin = 0; ibin < 51; ibin++ ) {
	      pt_bins[ibin] = ibin*(145./50.);
	  }
	  sprintf(name,"HLTMuonL%i_NMu",level);
	  sprintf(title,"L%i number of muons",level);
	  hNMu[trig][level-1] = dbe_->book1D(name,title, 5, -0.5, 4.5);
	  hNMu[trig][level-1]->setAxisTitle("Number of muons", 1);
	  sprintf(name,"HLTMuonL%i_pt",level);
	  sprintf(title,"L%i Pt",level);
	  hpt[trig][level-1] = dbe_->book1D(name,title, NBINS, pt_bins);
	  hpt[trig][level-1]->setAxisTitle("Pt", 1);
	  sprintf(name,"HLTMuonL%i_eta",level);
	  sprintf(title,"L%i Muon #eta",level);
	  heta[trig][level-1] = dbe_->book1D(name,title, eta_nbin, eta_bins);
	  heta[trig][level-1]->setAxisTitle("#eta", 1);
	  sprintf(name,"HLTMuonL%i_phi",level);
	  sprintf(title,"L%i Muon #phi",level);
	  hphi[trig][level-1] = dbe_->book1D(name,title, phi_nbin, phi_bins);
	  hphi[trig][level-1]->setAxisTitle("#phi", 1);
	  sprintf(name,"HLTMuonL%i_phi_norm",level);
	  sprintf(title,"L%i Muon #phi_norm",level);
	  hphi_norm[trig][level-1] = dbe_->book1D(name,title, phi_nbin, phi_bins);
	  hphi_norm[trig][level-1]->setAxisTitle("#phi", 1);
	  sprintf(name,"HLTMuonL%i_etaphi",level);
	  sprintf(title,"L%i Muon #eta vs #phi",level);
	  hetaphi[trig][level-1] = dbe_->book2D(name,title, phi_nbin, phi_bins, eta_nbin, eta_bins);
	  hetaphi[trig][level-1]->setAxisTitle("#phi", 1);
	  hetaphi[trig][level-1]->setAxisTitle("#eta", 2); 
	  sprintf(name,"HLTMuonL%i_ptphi",level);
	  sprintf(title,"L%i Muon pt vs #phi",level);         
	  hptphi[trig][level-1] = dbe_->book2D(name,title, NBINS, pt_bins, phi_nbin, phi_bins);
	  hptphi[trig][level-1]->setAxisTitle("pt", 1);
	  hptphi[trig][level-1]->setAxisTitle("#phi", 2);
	  sprintf(name,"HLTMuonL%i_pteta",level);
	  sprintf(title,"L%i Muon pt vs #eta",level);         
	  hpteta[trig][level-1] = dbe_->book2D(name,title, NBINS, pt_bins, eta_nbin, eta_bins);
	  hpteta[trig][level-1]->setAxisTitle("pt", 1);
	  hpteta[trig][level-1]->setAxisTitle("#eta", 2);
	  sprintf(name,"HLTMuonL%i_charge",level);
	  sprintf(title,"L%i Muon Charge",level);         
	  hcharge[trig][level-1]  = dbe_->book1D(name,title, 3, -1.5, 1.5);
	  hcharge[trig][level-1]->setAxisTitle("Charge", 1);
	}
	if( level > 1 && level < 4 ) {
	  sprintf(name,"HLTMuonL%i_NMu",level);
	  sprintf(title,"L%i number of muons",level);
	  hNMu[trig][level-1] = dbe_->book1D(name,title, 5, -0.5, 4.5);
	  hNMu[trig][level-1]->setAxisTitle("Number of muons", 1);
	  sprintf(name,"HLTMuonL%i_pt",level);
	  sprintf(title,"L%i Pt",level);
	  hpt[trig][level-1] = dbe_->book1D(name,title, NBINS, 0., pt_max);
	  hpt[trig][level-1]->setAxisTitle("Pt", 1);
	  sprintf(name,"HLTMuonL%i_eta",level);
	  sprintf(title,"L%i Muon #eta",level);
	  heta[trig][level-1] = dbe_->book1D(name,title, NBINS, -2.5, 2.5);
	  heta[trig][level-1]->setAxisTitle("#eta", 1);
	  sprintf(name,"HLTMuonL%i_phi",level);
	  sprintf(title,"L%i Muon #phi",level);
	  hphi[trig][level-1] = dbe_->book1D(name,title, NBINS, -3.15, 3.15);
	  hphi[trig][level-1]->setAxisTitle("#phi", 1);
	  sprintf(name,"HLTMuonL%i_phi_norm",level);
	  sprintf(title,"L%i Muon #phi_norm",level);
	  hphi_norm[trig][level-1] = dbe_->book1D(name,title, NBINS, -3.15, 3.15);
	  hphi_norm[trig][level-1]->setAxisTitle("#phi", 1);
	  sprintf(name,"HLTMuonL%i_etaphi",level);
	  sprintf(title,"L%i Muon #eta vs #phi",level);
	  hetaphi[trig][level-1] = dbe_->book2D(name,title, NBINS, -3.15, 3.15,NBINS,-2.5, 2.5);
	  hetaphi[trig][level-1]->setAxisTitle("#phi", 1);
	  hetaphi[trig][level-1]->setAxisTitle("#eta", 2); 
	  sprintf(name,"HLTMuonL%i_ptphi",level);
	  sprintf(title,"L%i Muon pt vs #phi",level);         
	  hptphi[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., pt_max,NBINS,-3.15, 3.15);
	  hptphi[trig][level-1]->setAxisTitle("pt", 1);
	  hptphi[trig][level-1]->setAxisTitle("#phi", 2);
	  sprintf(name,"HLTMuonL%i_pteta",level);
	  sprintf(title,"L%i Muon pt vs #eta",level);         
	  hpteta[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., pt_max,NBINS,-2.5, 2.5);
	  hpteta[trig][level-1]->setAxisTitle("pt", 1);
	  hpteta[trig][level-1]->setAxisTitle("#eta", 2);
	  sprintf(name,"HLTMuonL%i_nHits",level);
	  sprintf(title,"L%i Number of Hits",level);         
	  hnHits[trig][level-1] = dbe_->book1D(name,title, NBINS, 0., 100.);
	  hnHits[trig][level-1]->setAxisTitle("Number of Hits", 1);
	  sprintf(name,"HLTMuonL%i_charge",level);
	  sprintf(title,"L%i Muon Charge",level);         
	  hcharge[trig][level-1]  = dbe_->book1D(name,title, 3, -1.5, 1.5);
	  hcharge[trig][level-1]->setAxisTitle("Charge", 1);
	}
	else if( level == 4 || level == 5 ) {
	  sprintf(name,"HLTMuonL%iSeed_NMu",level-2);
	  sprintf(title,"L%iSeed number of muons",level-2);
	  hNMu[trig][level-1] = dbe_->book1D(name,title, 5, -0.5, 4.5);
	  hNMu[trig][level-1]->setAxisTitle("Number of muons", 1);
	  sprintf(name,"HLTMuonL%iSeed_pt",level-2);
	  sprintf(title,"L%iSeed Pt",level-2);
	  hpt[trig][level-1] = dbe_->book1D(name,title, NBINS, 0., pt_max);
	  hpt[trig][level-1]->setAxisTitle("Pt", 1);
	  sprintf(name,"HLTMuonL%iSeed_eta",level-2);
	  sprintf(title,"L%iSeed Muon #eta",level-2);
	  heta[trig][level-1] = dbe_->book1D(name,title, NBINS, -2.5, 2.5);
	  heta[trig][level-1]->setAxisTitle("#eta", 1);
	  sprintf(name,"HLTMuonL%iSeed_phi",level-2);
	  sprintf(title,"L%iSeed Muon #phi",level-2);
	  hphi[trig][level-1] = dbe_->book1D(name,title, NBINS, -3.15, 3.15);
	  hphi[trig][level-1]->setAxisTitle("#phi", 1);
	  sprintf(name,"HLTMuonL%iSeed_etaphi",level-2);
	  sprintf(title,"L%iSeed Muon #eta vs #phi",level-2);
	  hetaphi[trig][level-1] = dbe_->book2D(name,title, NBINS, -3.15, 3.15,NBINS,-2.5, 2.5);
	  hetaphi[trig][level-1]->setAxisTitle("#phi", 1);
	  hetaphi[trig][level-1]->setAxisTitle("#eta", 2); 
	  sprintf(name,"HLTMuonL%iSeed_ptphi",level-2);
	  sprintf(title,"L%iSeed Muon pt vs #phi",level-2);         
	  hptphi[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., pt_max,NBINS,-3.15, 3.15);
	  hptphi[trig][level-1]->setAxisTitle("pt", 1);
	  hptphi[trig][level-1]->setAxisTitle("#phi", 2);
	  sprintf(name,"HLTMuonL%iSeed_pteta",level-2);
	  sprintf(title,"L%iSeed Muon pt vs #eta",level-2);         
	  hpteta[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., pt_max,NBINS,-2.5, 2.5);
	  hpteta[trig][level-1]->setAxisTitle("pt", 1);
	  hpteta[trig][level-1]->setAxisTitle("#eta", 2);
	  sprintf(name,"HLTMuonL%iSeed_charge",level-2);
	  sprintf(title,"L%iSeed Muon Charge",level-2);         
	  hcharge[trig][level-1]  = dbe_->book1D(name,title, 3, -1.5, 1.5);
	  hcharge[trig][level-1]->setAxisTitle("Charge", 1);
	  // pt
	  sprintf(name,"HLTMuonL%iSeedtoL%i_ptres",level-2,level-3);
	  sprintf(title,"L%iSeed1/Pt - L%iMuon1/Pt",level-2,level-3);         
	  hseedptres[trig][level-4] = dbe_->book1D(name,title, NBINS, -0.1, 0.1);
	  sprintf(title,"1/PtL%iSeed - 1/PtL%i",level-2,level-3);         
	  hseedptres[trig][level-4]->setAxisTitle(title, 1);
	  sprintf(name,"HLTMuonL%iSeedtoL%i_ptrelres",level-2,level-3);
	  sprintf(title,"(L%iSeed1/Pt - L%iMuon1/Pt)/(L%iMuon1/Pt)",level-2,level-3,level-3);         
	  hseedptrelres[trig][level-4] = dbe_->book1D(name,title, NBINS, -1.0, 1.0);
	  sprintf(title,"(1/PtL%iSeed - 1/PtL%i)/(1/PtL%i)",level-2,level-3,level-3);         
	  hseedptrelres[trig][level-4]->setAxisTitle(title, 1);
	  // eta
	  sprintf(name,"HLTMuonL%iSeedtoL%i_etares",level-2,level-3);
	  sprintf(title,"L%iSeed#eta - L%iMuon#eta",level-2,level-3);         
	  hseedetares[trig][level-4] =dbe_->book1D(name,title, NBINS, -0.1, 0.1);
	  sprintf(title,"L%iSeed #eta - L%i #eta",level-2,level-3);         
	  hseedetares[trig][level-4]->setAxisTitle(title, 1);
	  sprintf(name,"HLTMuonL%iSeedtoL%i_etarelres",level-2,level-3);
	  sprintf(title,"(L%iSeed#eta - L%iMuon#eta)/L%iMuon#eta",level-2,level-3,level-3);         
	  hseedetarelres[trig][level-4] =dbe_->book1D(name,title, NBINS, -1.0, 1.0);
	  sprintf(title,"(L%iSeed #eta - L%i #eta)/L%i #eta",level-2,level-3,level-3);         
	  hseedetarelres[trig][level-4]->setAxisTitle(title, 1);
	  // phi
	  sprintf(name,"HLTMuonL%iSeedtoL%i_phires",level-2,level-3);
	  sprintf(title,"L%iSeed#phi - L%iMuon#phi",level-2,level-3);         
	  hseedphires[trig][level-4] =dbe_->book1D(name,title, NBINS, -0.1, 0.1);
	  sprintf(title,"L%iSeed #phi - L%i #phi",level-2,level-3);         
	  hseedphires[trig][level-4]->setAxisTitle(title, 1);
	  sprintf(name,"HLTMuonL%iSeedtoL%i_phirelres",level-2,level-3);
	  sprintf(title,"(L%iSeed#phi - L%iMuon#phi)/L%iMuon#phi",level-2,level-3,level-3);         
	  hseedphirelres[trig][level-4] =dbe_->book1D(name,title, NBINS, -0.1, 0.1);
	  sprintf(title,"(L%iSeed #phi - L%i #phi)/L%i #phi",level-2,level-3,level-3);         
	  hseedphirelres[trig][level-4]->setAxisTitle(title, 1);

	  sprintf(name,"HLTMuonL%iSeed_NMuperL%i",level-2,level-3);
	  sprintf(title,"L%iSeedNMu per L%i",level-2,level-3);         
	  hseedNMuper[trig][level-4] = dbe_->book1D(name, title, 5, -0.5, 4.5);
	  hseedNMuper[trig][level-4]->setAxisTitle(title, 1);
	}
	else if( level == 6 ) {
	  double max_energy = 1.0;
	  sprintf(name,"HLTMuonL2_driso");
	  sprintf(title,"dR Cone Size");
	  hL2muonIsoDR[trig] = dbe_->book2D(name, title, 10, 0, 0.24, NBINS, 0, 5.0);
	  hL2muonIsoDR[trig]->setAxisTitle(title, 1);
	  hL2muonIsoDR[trig]->setAxisTitle("Calo Energy in Iso Cone (GeV)", 2);

	  sprintf(name,"HLTCaloTower_dRwithL2");
	  sprintf(title,"#Delta R(L2muon, CaloTower)");
	  htowerdRL2[trig] = dbe_->book1D(name, title, NBINS, 0, 0.5);
	  htowerdRL2[trig]->setAxisTitle(title, 1);
	  sprintf(name,"HLTCaloTower_dRwithL3");
	  sprintf(title,"#Delta R(L3muon, CaloTower)");
	  htowerdRL3[trig] = dbe_->book1D(name, title, NBINS, 0, 0.5);
	  htowerdRL3[trig]->setAxisTitle(title, 1);

	  sprintf(name,"HLTCaloTower_Et");
	  sprintf(title,"HLT CaloTower Et");
	  htowerEt[trig] = dbe_->book1D(name, title, NBINS, 0, max_energy);
	  htowerEt[trig]->setAxisTitle(title, 1);
	  sprintf(name,"HLTCaloTower_Eta");
	  sprintf(title,"HLT CaloTower #eta");
	  htowerEta[trig] = dbe_->book1D(name, title, NBINS, -2.5, 2.5);
	  htowerEta[trig]->setAxisTitle(title, 1);
	  sprintf(name,"HLTCaloTower_Phi");
	  sprintf(title,"HLT CaloTower #phi");
	  htowerPhi[trig] = dbe_->book1D(name, title, NBINS, -3.14, 3.14);
	  htowerPhi[trig]->setAxisTitle(title, 1);
	  
	  sprintf(name,"HLTCaloTower_HadEnergy");
	  sprintf(title,"HLT CaloTower hadEnergy");
	  htowerHadEnergy[trig] = dbe_->book1D(name, title, NBINS, 0, max_energy);
	  htowerHadEnergy[trig]->setAxisTitle(title, 1);
	  sprintf(name,"HLTCaloTower_EmEnergy");
	  sprintf(title,"HLT CaloTower emEnergy");
	  htowerEmEnergy[trig] = dbe_->book1D(name, title, NBINS, 0, max_energy);
	  htowerEmEnergy[trig]->setAxisTitle(title, 1);
	  sprintf(name,"HLTCaloTower_OuterEnergy");
	  sprintf(title,"HLT CaloTower outerEnergy");
	  htowerOuterEnergy[trig] = dbe_->book1D(name, title, NBINS, 0, max_energy);
	  htowerOuterEnergy[trig]->setAxisTitle(title, 1);
	 
	  sprintf(name,"HLTCaloTower_HadEt");
	  sprintf(title,"HLT CaloTower hadEt");
	  htowerHadEt[trig] = dbe_->book1D(name, title, NBINS, 0, max_energy);
	  htowerHadEt[trig]->setAxisTitle(title, 1);
	  sprintf(name,"HLTCaloTower_EmEt");
	  sprintf(title,"HLT CaloTower emEt");
	  htowerEmEt[trig] = dbe_->book1D(name, title, NBINS, 0, max_energy);
	  htowerEmEt[trig]->setAxisTitle(title, 1);
	  sprintf(name,"HLTCaloTower_OuterEt");
	  sprintf(title,"HLT CaloTower outerEt");
	  htowerOuterEt[trig] = dbe_->book1D(name, title, NBINS, 0, max_energy);
	  htowerOuterEt[trig]->setAxisTitle(title, 1);

	  sprintf(name,"HLTCaloTower_EtaHadEt");
	  sprintf(title,"HLT CaloTower hadEt");
	  htowerEtaHadEt[trig] = dbe_->book2D(name, title, NBINS, -2.5, 2.5, NBINS, 0, max_energy);
	  htowerEtaHadEt[trig]->setAxisTitle("CaloTower #eta", 1);
	  htowerEtaHadEt[trig]->setAxisTitle(title, 2);
	  sprintf(name,"HLTCaloTower_EtaEmEt");
	  sprintf(title,"HLT CaloTower emEt");
	  htowerEtaEmEt[trig] = dbe_->book2D(name, title, NBINS, -2.5, 2.5, NBINS, 0, max_energy);
	  htowerEtaEmEt[trig]->setAxisTitle("CaloTower #eta", 1);
	  htowerEtaEmEt[trig]->setAxisTitle(title, 2);
	  sprintf(name,"HLTCaloTower_EtaOuterEt");
	  sprintf(title,"HLT CaloTower outerEt");
	  htowerEtaOuterEt[trig] = dbe_->book2D(name, title, NBINS, -2.5, 2.5, NBINS, 0, max_energy);
	  htowerEtaOuterEt[trig]->setAxisTitle("CaloTower #eta", 1);
	  htowerEtaOuterEt[trig]->setAxisTitle(title, 2);

	  sprintf(name,"HLTCaloTower_PhiHadEt");
	  sprintf(title,"HLT CaloTower hadEt");
	  htowerPhiHadEt[trig] = dbe_->book2D(name, title, NBINS, -3.15, 3.15, NBINS, 0, max_energy);
	  htowerPhiHadEt[trig]->setAxisTitle("CaloTower #phi", 1);
	  htowerPhiHadEt[trig]->setAxisTitle(title, 2);
	  sprintf(name,"HLTCaloTower_PhiEmEt");
	  sprintf(title,"HLT CaloTower emEt");
	  htowerPhiEmEt[trig] = dbe_->book2D(name, title, NBINS, -3.15, 3.15, NBINS, 0, max_energy);
	  htowerPhiEmEt[trig]->setAxisTitle("CaloTower #phi", 1);
	  htowerPhiEmEt[trig]->setAxisTitle(title, 2);
	  sprintf(name,"HLTCaloTower_PhiOuterEt");
	  sprintf(title,"HLT CaloTower outerEt");
	  htowerPhiOuterEt[trig] = dbe_->book2D(name, title, NBINS, -3.15, 3.15, NBINS, 0, max_energy);
	  htowerPhiOuterEt[trig]->setAxisTitle("CaloTower #phi", 1);
	  htowerPhiOuterEt[trig]->setAxisTitle(title, 2);
	}
	
	if (level>1&&level<4){
	  sprintf(name,"HLTMuonL%i_dr",level);
	  sprintf(title,"L%i Muon radial impact vs BeamSpot",level);         
	  hdr[trig][level-2] = dbe_->book1D(name,title, NBINS, -0.3, 0.3);
	  hdr[trig][level-2]->setAxisTitle("R Impact (cm) vs BeamSpot", 1);
	  sprintf(name,"HLTMuonL%i_d0",level);
	  sprintf(title,"L%i Muon radial impact vs (0,0)",level);         
	  hd0[trig][level-2] = dbe_->book1D(name,title, NBINS, -0.3, 0.3);
	  hd0[trig][level-2]->setAxisTitle("R Impact (cm) vs 0,0", 1);
	  sprintf(name,"HLTMuonL%i_dz0",level);
	  sprintf(title,"L%i Muon Z impact vs (0)",level);         
	  hdz0[trig][level-2] = dbe_->book1D(name,title, NBINS, -25., 25.);
	  hdz0[trig][level-2]->setAxisTitle("Z impact (cm) vs 0", 1);
	  sprintf(name,"HLTMuonL%i_dz",level);
	  sprintf(title,"L%i Muon Z impact vs BeamSpot",level);         
	  hdz[trig][level-2] = dbe_->book1D(name,title, NBINS, -25., 25.);
	  hdz[trig][level-2]->setAxisTitle("Z impact (cm) vs BeamSpot", 1);
	  sprintf(name,"HLTMuonL%i_err0",level);
	  sprintf(title,"L%i Muon Error on Pt",level);         
	  herr0[trig][level-2] = dbe_->book1D(name,title,NBINS, 0., 0.03);
	  herr0[trig][level-2]->setAxisTitle("Error on Pt", 1);
	  sprintf(name,"HLTMuonL%i_iso",level);
	  if (level==2)sprintf(title,"L%i Muon Energy in Isolation cone",level);         
	  else if (level==3)sprintf(title,"L%i Muon SumPt in Isolation cone",level);               
	  hiso[trig][level-2]  = dbe_->book1D(name,title, NBINS, 0., 5./(level-1));
	  if ( level==2)hiso[trig][level-2]->setAxisTitle("Calo Energy in Iso Cone (GeV)", 1);
	  else if ( level==3)hiso[trig][level-2]->setAxisTitle("Sum Pt in Iso Cone (GeV)", 1);
	  sprintf(name,"HLTMuonL%i_DiMuMass",level);
	  sprintf(title,"L%i Opposite charge DiMuon invariant Mass",level);         
	  hdimumass[trig][level-2]= dbe_->book1D(name,title, NBINS, 0., 150.);
	  hdimumass[trig][level-2]->setAxisTitle("Di Muon Invariant Mass (GeV)");

	  sprintf(name,"HLTMuonL%i_drphi",level);
	  sprintf(title,"L%i #Deltar vs #phi",level);         
	  hdrphi[trig][level-2] = dbe_->bookProfile(name,title, NBINS, -3.15, 3.15,1,-999.,999.,"s");
	  hdrphi[trig][level-2]->setAxisTitle("#phi", 1);
	  sprintf(title,"L%i Muon radial impact vs BeamSpot",level);         
	  hdrphi[trig][level-2]->setAxisTitle(title, 2);

	  sprintf(name,"HLTMuonL%i_d0phi",level);
	  sprintf(title,"L%i #Delta0 vs #phi",level);         
	  hd0phi[trig][level-2] = dbe_->bookProfile(name,title, NBINS, -3.15, 3.15,1,-999.,999.,"s");
	  hd0phi[trig][level-2]->setAxisTitle("#phi", 1);
	  sprintf(title,"L%i Muon radial impact vs (0,0)",level);         
	  hd0phi[trig][level-2]->setAxisTitle(title, 2);

	  sprintf(name,"HLTMuonL%i_dz0eta",level);
	  sprintf(title,"L%i #Deltaz0 vs #eta",level);         
	  hdz0eta[trig][level-2] = dbe_->bookProfile(name,title, NBINS,-2.5, 2.5,1,-999.,999.,"s");
	  hdz0eta[trig][level-2]->setAxisTitle("#eta", 1);
	  sprintf(title,"L%i Muon Z impact vs (0)",level);         
	  hdz0eta[trig][level-2]->setAxisTitle(title, 2);

	  sprintf(name,"HLTMuonL%i_dzeta",level);
	  sprintf(title,"L%i #Deltaz vs #eta",level);         
	  hdzeta[trig][level-2] = dbe_->bookProfile(name,title, NBINS,-2.5, 2.5,1,-999.,999.,"s");
	  hdzeta[trig][level-2]->setAxisTitle("#eta", 1);
	  sprintf(title,"L%i Muon Z impact vs BeamSpot",level);         
	  hdzeta[trig][level-2]->setAxisTitle(title, 2);
	}
	if(level == 2 ) {
	  sprintf(name,"HLTMuonL%itoL%i_ptpull",level,level+1);
	  sprintf(title,"(L%iMuon1/Pt - L%iMuon1/Pt)/#sigma_{pt}^{L2}",level,level+1);         
	  hptpull[trig] = dbe_->book1D(name,title, NBINS, -10.0, 10.0);
	  sprintf(title,"(1/PtL%i - 1/PtL%i)/#sigma_{pt}^{L2}",level,level+1);         
	  hptpull[trig]->setAxisTitle(title, 1);
	  sprintf(name,"HLTMuonL%itoL%i_etapull",level,level+1);
	  sprintf(title,"(L%iMuon#eta - L%iMuon#eta)/#sigma_{#eta}^{L2}",level,level+1);         
	  hetapull[trig] =dbe_->book1D(name,title, NBINS, -10.0, 10.0);
	  sprintf(title,"(L%i #eta - L%i #eta)/#sigma_{#eta}^{L2}",level,level+1);         
	  hetapull[trig]->setAxisTitle(title, 1);
	  sprintf(name,"HLTMuonL%itoL%i_phipull",level,level+1);
	  sprintf(title,"(L%iMuon#phi - L%iMuon#phi)/#sigma_{#phi}^{L2}",level,level+1);         
	  hphipull[trig] =dbe_->book1D(name,title, NBINS, -10.0, 10.0);
	  sprintf(title,"(L%i #phi - L%i #phi)/#sigma_{#phi}^{L2}",level,level+1);         
	  hphipull[trig]->setAxisTitle(title, 1);

	  sprintf(name,"HLTMuonL%itoL%i_ptpullpt",level,level+1);
	  sprintf(title,"L%i Muon #Delta Pt/#sigma_{pt}^{L2} vs Pt ",level);         
	  hptpullpt[trig] =dbe_->bookProfile(name,title, NBINS, 0, pt_max,1,-999.,999.,"s");
	  sprintf(title,"(1/PtL%i - 1/PtL%i)/#sigma_{pt}^{L2}",level,level+1);         
	  hptpullpt[trig]->setAxisTitle(title, 2);
	  hptpullpt[trig]->setAxisTitle("Pt", 1);
	  sprintf(name,"HLTMuonL%itoL%i_etapulleta",level,level+1);
	  sprintf(title,"L%i Muon #Delta#eta/#sigma_{#eta}^{L2} vs #eta ",level);         
	  hetapulleta[trig] =dbe_->bookProfile(name,title, NBINS,-2.5, 2.5,1,-999.,999.,"s");
	  sprintf(title,"(L%i #eta - L%i #eta)/#sigma_{#eta}^{L2}",level,level+1);         
	  hetapulleta[trig]->setAxisTitle(title, 2);
	  hetapulleta[trig]->setAxisTitle("#eta", 1);
	  sprintf(name,"HLTMuonL%itoL%i_phipullphi",level,level+1);
	  sprintf(title,"L%i Muon #Delta#phi/#sigma_{#phi}^{L2} vs #phi ",level);         
	  hphipullphi[trig] =dbe_->bookProfile(name,title, NBINS, -3.15, 3.15,1,-999.,999.,"s");
	  sprintf(title,"(L%i #phi - L%i #phi)/#sigma_{#phi}^{L2}",level,level+1);         
	  hphipullphi[trig]->setAxisTitle(title, 2);
	  hphipullphi[trig]->setAxisTitle("#phi", 1);
	}
	if (level < 3 ) {
	  // res
	  sprintf(name,"HLTMuonL%itoL%i_ptres",level,level+1);
	  sprintf(title,"L%iMuon1/Pt - L%iMuon1/Pt",level,level+1);         
	  hptres[trig][level-1] = dbe_->book1D(name,title, NBINS, -0.1, 0.1);
	  sprintf(title,"1/PtL%i - 1/PtL%i",level,level+1);         
	  hptres[trig][level-1]->setAxisTitle(title, 1);
	  sprintf(name,"HLTMuonL%itoL%i_etares",level,level+1);
	  sprintf(title,"L%iMuon#eta - L%iMuon#eta",level,level+1);         
	  hetares[trig][level-1] =dbe_->book1D(name,title, NBINS, -0.1, 0.1);
	  sprintf(title,"L%i #eta - L%i #eta",level,level+1);         
	  hetares[trig][level-1]->setAxisTitle(title, 1);
	  sprintf(name,"HLTMuonL%itoL%i_phires",level,level+1);
	  sprintf(title,"L%iMuon#phi - L%iMuon#phi",level,level+1);         
	  hphires[trig][level-1] =dbe_->book1D(name,title, NBINS, -0.1, 0.1);
	  sprintf(title,"L%i #phi - L%i #phi",level,level+1);         
	  hphires[trig][level-1]->setAxisTitle(title, 1);

	  sprintf(name,"HLTMuonL%itoL%i_ptrespt",level,level+1);
	  sprintf(title,"L%i Muon #Delta Pt vs Pt ",level);         
	  hptrespt[trig][level-1] =dbe_->bookProfile(name,title, NBINS, 0, pt_max,1,-999.,999.,"s");
	  sprintf(title,"1/PtL%i - 1/PtL%i",level,level+1);         
	  hptrespt[trig][level-1]->setAxisTitle(title, 2);
	  hptrespt[trig][level-1]->setAxisTitle("Pt", 1);
	  sprintf(name,"HLTMuonL%itoL%i_phiresphi",level,level+1);
	  sprintf(title,"L%i Muon #Delta#phi vs #phi ",level);         
	  hphiresphi[trig][level-1] =dbe_->bookProfile(name,title, NBINS, -3.15, 3.15,1,-999.,999.,"s");
	  sprintf(title,"L%i #phi - L%i #phi",level,level+1);         
	  hphiresphi[trig][level-1]->setAxisTitle(title, 2);
	  hphiresphi[trig][level-1]->setAxisTitle("#phi", 1);
	  sprintf(name,"HLTMuonL%itoL%i_etareseta",level,level+1);
	  sprintf(title,"L%i Muon #Delta#eta vs #eta ",level);         
	  hetareseta[trig][level-1] =dbe_->bookProfile(name,title, NBINS,-2.5, 2.5,1,-999.,999.,"s");
	  sprintf(title,"L%i #eta - L%i #eta",level,level+1);         
	  hetareseta[trig][level-1]->setAxisTitle(title, 2);
	  hetareseta[trig][level-1]->setAxisTitle("#eta", 1);

	  // relres
	  sprintf(name,"HLTMuonL%itoL%i_ptrelres",level,level+1);
	  sprintf(title,"(L%iMuon1/Pt - L%iMuon1/Pt)/(L%iMuon1/Pt)",level,level+1,level+1);         
	  hptrelres[trig][level-1] = dbe_->book1D(name,title, NBINS, -1.0, 1.0);
	  sprintf(title,"(1/PtL%i - 1/PtL%i)/(1/PtL%i)",level,level+1,level+1);         
	  hptrelres[trig][level-1]->setAxisTitle(title, 1);
	  sprintf(name,"HLTMuonL%itoL%i_etarelres",level,level+1);
	  sprintf(title,"(L%iMuon#eta - L%iMuon#eta)/L%iMuon#eta",level,level+1,level+1);         
	  hetarelres[trig][level-1] =dbe_->book1D(name,title, NBINS, -1.0, 1.0);
	  sprintf(title,"(L%i #eta - L%i #eta)/L%i #eta",level,level+1,level+1);         
	  hetarelres[trig][level-1]->setAxisTitle(title, 1);
	  sprintf(name,"HLTMuonL%itoL%i_phirelres",level,level+1);
	  sprintf(title,"(L%iMuon#phi - L%iMuon#phi)/L%iMuon#phi",level,level+1,level+1);         
	  hphirelres[trig][level-1] =dbe_->book1D(name,title, NBINS, -1.0, 1.0);
	  sprintf(title,"(L%i #phi - L%i #phi)/L%i #phi",level,level+1,level+1);         
	  hphirelres[trig][level-1]->setAxisTitle(title, 1);

	  sprintf(name,"HLTMuonL%itoL%i_ptrelrespt",level,level+1);
	  sprintf(title,"L%i Muon #DeltaPt/Pt vs Pt ",level);         
	  hptrelrespt[trig][level-1] =dbe_->bookProfile(name,title, NBINS, 0, pt_max,1,-999.,999.,"s");
	  sprintf(title,"(1/PtL%i - 1/PtL%i)/(1/PtL%i)",level,level+1,level+1);         
	  hptrelrespt[trig][level-1]->setAxisTitle(title, 2);
	  hptrelrespt[trig][level-1]->setAxisTitle("Pt", 1);
	  sprintf(name,"HLTMuonL%itoL%i_phirelresphi",level,level+1);
	  sprintf(title,"L%i Muon #Delta#phi/#phi vs #phi ",level);         
	  hphirelresphi[trig][level-1] =dbe_->bookProfile(name,title, NBINS, -3.15, 3.15,1,-999.,999.,"s");
	  sprintf(title,"(L%i #phi - L%i #phi)/L%i #phi",level,level+1,level+1);         
	  hphirelresphi[trig][level-1]->setAxisTitle(title, 2);
	  hphirelresphi[trig][level-1]->setAxisTitle("#phi", 1);
	  sprintf(name,"HLTMuonL%itoL%i_etarelreseta",level,level+1);
	  sprintf(title,"L%i Muon #Delta#eta/#eta vs #eta ",level);         
	  hetarelreseta[trig][level-1] =dbe_->bookProfile(name,title, NBINS,-2.5, 2.5,1,-999.,999.,"s");
	  sprintf(title,"(L%i #eta - L%i #eta)/L%i #eta",level,level+1,level+1);         
	  hetarelreseta[trig][level-1]->setAxisTitle(title, 2);
	  hetarelreseta[trig][level-1]->setAxisTitle("#eta", 1);
	  // charge conversion
	  sprintf(name,"HLTMuonL%itoL%i_chargeconvers",level,level+1);
	  sprintf(title,"L%i Muon charge #rightarrow L%i Muon charge",level,level+1);         
	  hchargeconv[trig][level-1] =dbe_->book1D(name,title, 4, 0, 4);
	  hchargeconv[trig][level-1]->setAxisTitle(title, 1);
	  hchargeconv[trig][level-1]->setBinLabel(1, "- #rightarrow -", 1);
	  hchargeconv[trig][level-1]->setBinLabel(2, "- #rightarrow +", 1);
	  hchargeconv[trig][level-1]->setBinLabel(3, "+ #rightarrow -", 1);
	  hchargeconv[trig][level-1]->setBinLabel(4, "+ #rightarrow +", 1);
	  // reconstruction fraction with dependence
	  sprintf(name,"HLTMuonL%itoL%i_fracpt",level,level+1);
	  sprintf(title,"#ofL%iMuon/#ofL%iMuon",level+1,level);         
	  hptfrac[trig][level-1] = dbe_->book1D(name,title, 40, 0, pt_max);
	  hptfrac[trig][level-1]->setAxisTitle("Pt", 1);
	  sprintf(name,"HLTMuonL%itoL%i_fraceta",level,level+1);
	  sprintf(title,"#ofL%iMuon/#ofL%iMuon",level+1,level);         
	  hetafrac[trig][level-1] = dbe_->book1D(name,title, 40, -2.5, 2.5);
	  hetafrac[trig][level-1]->setAxisTitle("#eta", 1);
	  sprintf(name,"HLTMuonL%itoL%i_fracphi",level,level+1);
	  sprintf(title,"#ofL%iMuon/#ofL%iMuon",level+1,level);         
	  hphifrac[trig][level-1] = dbe_->book1D(name,title, 40, -3.15, 3.15);
	  hphifrac[trig][level-1]->setAxisTitle("#phi", 1);
	  if (level  == 1 ){
	    // res
	    sprintf(name,"HLTMuonL%itoL3_ptres",level);
	    sprintf(title,"L%iMuon1/Pt - L%iMuon1/Pt",level,level+2);         
	    hptres[trig][level+1] = dbe_->book1D(name,title, NBINS, -0.1, 0.1);
	    sprintf(title,"1/PtL%i - 1/PtL%i",level,level+2);         
	    hptres[trig][level+1]->setAxisTitle(title, 1);
	    sprintf(name,"HLTMuonL%itoL3_etares",level);
	    sprintf(title,"L%iMuon#eta - L3Muon#eta",level);         
	    hetares[trig][level+1] =dbe_->book1D(name,title, NBINS, -0.1, 0.1);
	    sprintf(title,"L%i #eta - L3 #eta",level);         
	    hetares[trig][level+1]->setAxisTitle(title, 1);
	    sprintf(name,"HLTMuonL%itoL3_phires",level);
	    sprintf(title,"L%iMuon#phi - L3Muon#phi",level);         
	    hphires[trig][level+1] =dbe_->book1D(name,title, NBINS, -0.1, 0.1);
	    sprintf(title,"L%i #phi - L3 #phi",level);         
	    hphires[trig][level+1]->setAxisTitle(title, 1);

	    sprintf(name,"HLTMuonL%itoL3_ptrespt",level);
	    sprintf(title,"L%i Muon #Delta Pt vs Pt (wrt L3) ",level);         
	    hptrespt[trig][level+1] =dbe_->bookProfile(name,title, NBINS, 0, pt_max,1,-999.,999.,"s");
	    sprintf(title,"1/PtL%i - 1/PtL3",level);         
	    hptrespt[trig][level+1]->setAxisTitle(title, 2);
	    hptrespt[trig][level+1]->setAxisTitle("Pt", 1);
	    sprintf(name,"HLTMuonL%itoL3_phiresphi",level);
	    sprintf(title,"L%i Muon #Delta#phi vs #phi (wrt L3) ",level);         
	    hphiresphi[trig][level+1] =dbe_->bookProfile(name,title, NBINS, -3.15, 3.15,1,-999.,999.,"s");
	    sprintf(title,"L%i #phi - L3 #phi",level);         
	    hphiresphi[trig][level+1]->setAxisTitle(title, 2);
	    hphiresphi[trig][level+1]->setAxisTitle("#phi", 1);
	    sprintf(name,"HLTMuonL%itoL3_etareseta",level);
	    sprintf(title,"L%i Muon #Delta#eta vs #eta (wrt L3) ",level);         
	    hetareseta[trig][level+1] =dbe_->bookProfile(name,title, NBINS,-2.5, 2.5,1,-999.,999.,"s");
	    sprintf(title,"L%i #eta - L3 #eta",level);         
	    hetareseta[trig][level+1]->setAxisTitle(title, 2);
	    hetareseta[trig][level+1]->setAxisTitle("#eta", 1);

	    // relres
	    sprintf(name,"HLTMuonL%itoL3_ptrelres",level);
	    sprintf(title,"(L%iMuon1/Pt - L%iMuon1/Pt)/(L%iMuon1/Pt)",level,level+2,level+2); 
	    hptrelres[trig][level+1] = dbe_->book1D(name,title, NBINS, -1.0, 1.0);
	    sprintf(title,"(1/PtL%i - 1/PtL3)/(1/PtL3)",level);         
	    hptrelres[trig][level+1]->setAxisTitle(title, 1);
	    sprintf(name,"HLTMuonL%itoL3_etarelres",level);
	    sprintf(title,"(L%iMuon#eta - L3Muon#eta)/L3Muon#eta",level);         
	    hetarelres[trig][level+1] =dbe_->book1D(name,title, NBINS, -1.0, 1.0);
	    sprintf(title,"(L%i #eta - L3 #eta)/L3 #eta",level);         
	    hetarelres[trig][level+1]->setAxisTitle(title, 1);
	    sprintf(name,"HLTMuonL%itoL3_phirelres",level);
	    sprintf(title,"(L%iMuon#phi - L3Muon#phi)/L3Muon#phi",level);         
	    hphirelres[trig][level+1] =dbe_->book1D(name,title, NBINS, -1.0, 1.0);
	    sprintf(title,"(L%i #phi - L3 #phi)/L3 #phi",level);         
	    hphirelres[trig][level+1]->setAxisTitle(title, 1);

	    sprintf(name,"HLTMuonL%itoL3_ptrelrespt",level);
	    sprintf(title,"L%i Muon #DeltaPt/Pt vs Pt (wrt L3) ",level);         
	    hptrelrespt[trig][level+1] =dbe_->bookProfile(name,title, NBINS, 0, pt_max,1,-999.,999.,"s");
	    sprintf(title,"(1/PtL%i - 1/PtL3)/(1/PtL3)",level);         
	    hptrelrespt[trig][level+1]->setAxisTitle(title, 2);
	    hptrelrespt[trig][level+1]->setAxisTitle("Pt", 1);
	    sprintf(name,"HLTMuonL%itoL3_phirelresphi",level);
	    sprintf(title,"L%i Muon #Delta#phi/#phi vs #phi (wrt L3) ",level);         
	    hphirelresphi[trig][level+1] =dbe_->bookProfile(name,title, NBINS, -3.15, 3.15,1,-999.,999.,"s");
	    sprintf(title,"(L%i #phi - L3 #phi)/L3 #phi",level);         
	    hphirelresphi[trig][level+1]->setAxisTitle(title, 2);
	    hphirelresphi[trig][level+1]->setAxisTitle("#phi", 1);
	    sprintf(name,"HLTMuonL%itoL3_etarelreseta",level);
	    sprintf(title,"L%i Muon #Delta#eta/#eta vs #eta (wrt L3) ",level);         
	    hetarelreseta[trig][level+1] =dbe_->bookProfile(name,title, NBINS,-2.5, 2.5,1,-999.,999.,"s");
	    sprintf(title,"(L%i #eta - L3 #eta)/L3 #eta",level);         
	    hetarelreseta[trig][level+1]->setAxisTitle(title, 2);
	    hetarelreseta[trig][level+1]->setAxisTitle("#eta", 1);

	    sprintf(name,"HLTMuonL%itoL3_chargeconvers",level);
	    sprintf(title,"L%i Muon charge #rightarrow L3 Muon charge",level);         
	    hchargeconv[trig][level+1] =dbe_->book1D(name,title, 4, 0, 4);
	    hchargeconv[trig][level+1]->setAxisTitle(title, 1);
	    hchargeconv[trig][level+1]->setBinLabel(1, "- #rightarrow -", 1);
	    hchargeconv[trig][level+1]->setBinLabel(2, "- #rightarrow +", 1);
	    hchargeconv[trig][level+1]->setBinLabel(3, "+ #rightarrow -", 1);
	    hchargeconv[trig][level+1]->setBinLabel(4, "+ #rightarrow +", 1);
	  }
	}
      }
      if(verbose_)dbe_->showDirStructure();
    }
    // Muon det id is 2 pushed in bits 28:31
    const unsigned int detector_id = 2<<28;
    dbe_->tagContents(monitorName_, detector_id);
  } 
  
  for( int trig = 0; trig < nTrigs; trig++ ) {
    for( int level = 1; level < 3; ++level ) {
      char name[512];
      sprintf(name, "DenominatorL%iptTrig%i", level, trig);
      _hpt1[trig][level-1] = new TH1D(name, name, 40, 0, 145);
      sprintf(name, "NumeratorL%iptTrig%i", level, trig);
      _hpt2[trig][level-1] = new TH1D(name, name, 40, 0, 145);
      sprintf(name, "DenominatorL%ietaTrig%i", level, trig);
      _heta1[trig][level-1] = new TH1D(name, name, 40, -2.5, 2.5);
      sprintf(name, "NumeratorL%ietaTrig%i", level, trig);
      _heta2[trig][level-1] = new TH1D(name, name, 40, -2.5, 2.5);
      sprintf(name, "DenominatorL%iphiTrig%i", level, trig);
      _hphi1[trig][level-1] = new TH1D(name, name, 40, -3.15, 3.15);
      sprintf(name, "NumeratorL%iphiTrig%i", level, trig);
      _hphi2[trig][level-1] = new TH1D(name, name, 40, -3.15, 3.15);
    }
  }
}

//--------------------------------------------------------
void HLTMuonDQMSource::beginRun(const edm::Run& r, const edm::EventSetup& context) {
  
}

//--------------------------------------------------------
void HLTMuonDQMSource::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
					    const edm::EventSetup& context) {
  
}

// ----------------------------------------------------------
void HLTMuonDQMSource::analyze(const edm::Event& iEvent, 
			       const edm::EventSetup& iSetup )
{  
  if ( !dbe_) return;
  counterEvt_++;
  if (prescaleEvt_ > 0 && counterEvt_%prescaleEvt_!=0) return;
  LogDebug("HLTMuonDQMSource") << " processing conterEvt_: " << counterEvt_ <<std::endl;

  bool trigFired = false;
  bool FiredTriggers[NTRIG] = {false};
  edm::Handle<edm::TriggerResults> trigResult;
  iEvent.getByLabel(TrigResultInput, trigResult);
  if( !trigResult.failedToGet() ) {
    int ntrigs = trigResult->size();
    const edm::TriggerNames & trigName = iEvent.triggerNames(*trigResult);
    for( int itrig = 0; itrig != ntrigs; ++itrig) {
      //std::cout << "trigName = " << trigName.triggerName(itrig) << " " << itrig << std::endl;
      for( unsigned int n = 0; n < (unsigned int)theTriggerBits.size(); n++) { 
	if( trigName.triggerIndex(theTriggerBits[n]) == (unsigned int)ntrigs ) continue;
        if( trigResult->accept(trigName.triggerIndex(theTriggerBits[n])) ) {
	    for( unsigned int j = 0; j < (unsigned int)theDirectoryName.size(); j++ ) {
	      if( theHLTCollectionLevel[n] == theDirectoryName[j] ) FiredTriggers[j] = true;
	    }
	    trigFired = true;
	}
      }
    }
  }
  //else std::cout << "failed to get trigResult!!" << std::endl;
  // trigger fired
  if( !trigFired ) return;
  nTrig_++;

  //get the field
  edm::ESHandle<MagneticField> magField;
  iSetup.get<IdealMagneticFieldRecord>().get(magField);

  //get the geometry
  edm::ESHandle<GlobalTrackingGeometry> glbTrackingGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(glbTrackingGeometry);
  
  edm::Handle<reco::RecoChargedCandidateCollection> l2mucands, l3mucands;
  edm::Handle<L2MuonTrajectorySeedCollection> l2seeds; 
  edm::Handle<L3MuonTrajectorySeedCollection> l3seeds; 
  reco::RecoChargedCandidateCollection::const_iterator cand, cand2, cand3;
  
  iEvent.getByLabel (l2seedscollectionTag_,l2seeds);
  iEvent.getByLabel (l3seedscollectionTag_,l3seeds);
  iEvent.getByLabel (l2collectionTag_,l2mucands);
  iEvent.getByLabel (l3collectionTag_,l3mucands);

  if( !l2seeds.failedToGet() && l2seeds.isValid() ) {
    iEvent.getByLabel (l2collectionTag_,l2mucands);
    if( !l2mucands.failedToGet() && l2mucands->size() != 0 ) {
      iEvent.getByLabel (l3seedscollectionTag_,l3seeds);
      if( !l3seeds.failedToGet() && l3seeds.isValid() ) iEvent.getByLabel (l3collectionTag_,l3mucands);
    }
  }

  for( int ntrig = 0; ntrig < nTrigs; ntrig++ ) {
    if( !FiredTriggers[ntrig] ) continue;
    //std::cout << "trigger fired!" << std::endl;
    if( !l2seeds.failedToGet() && l2seeds.isValid() ) {
      hNMu[ntrig][3]->Fill(l2seeds->size());
      L2MuonTrajectorySeedCollection::const_iterator l2seed;
      std::map<l1extra::L1MuonParticleRef, int> l1map;
      for (l2seed=l2seeds->begin() ; l2seed != l2seeds->end();++l2seed){
	PTrajectoryStateOnDet state=l2seed->startingState();
	// Transform Trajectory State on Det to a TSOS
	TrajectoryStateTransform tsTransform;
	DetId seedDetId(state.detId());
	const GeomDet* gdet = glbTrackingGeometry->idToDet( seedDetId );
	TrajectoryStateOnSurface tsos = tsTransform.transientState(state, &(gdet->surface()), magField.product());
	float pt = tsos.globalMomentum().perp();
	float eta = tsos.globalPosition().eta();
	float phi = tsos.globalPosition().phi();
	hcharge[ntrig][3]->Fill(tsos.charge());
	hpt[ntrig][3]->Fill(pt);
	hphi[ntrig][3]->Fill(phi);
	heta[ntrig][3]->Fill(eta);
	hetaphi[ntrig][3]->Fill(phi,eta);
	hptphi[ntrig][3]->Fill(pt,phi);
	hpteta[ntrig][3]->Fill(pt,eta);
	l1extra::L1MuonParticleRef l1ref = l2seed->l1Particle();
	l1map[l1ref]++;
	hseedptres[ntrig][0]->Fill(1/pt - 1/l1ref->pt());
	hseedetares[ntrig][0]->Fill(eta - l1ref->eta());
	hseedphires[ntrig][0]->Fill(phi - l1ref->phi());
	hseedptrelres[ntrig][0]->Fill((1/pt - 1/l1ref->pt())/(1/l1ref->pt()));
	hseedetarelres[ntrig][0]->Fill((eta - l1ref->eta())/l1ref->eta());
	hseedphirelres[ntrig][0]->Fill((phi - l1ref->phi())/l1ref->phi());

	hcharge[ntrig][0]->Fill(l1ref->charge());
	hpt[ntrig][0]->Fill(l1ref->pt());
	hphi[ntrig][0]->Fill(l1ref->phi());

	if(hphi[ntrig][0]->getEntries()){
	  for(int ibin = 1; ibin <= hphi[ntrig][0]->getNbinsX(); ++ibin)
	    hphi_norm[ntrig][0]->setBinContent(ibin, 
					       hphi[ntrig][0]->getBinContent(ibin)/hphi[ntrig][0]->getEntries());
	}
	
	heta[ntrig][0]->Fill(l1ref->eta());
	hetaphi[ntrig][0]->Fill(l1ref->phi(),l1ref->eta());
	hptphi[ntrig][0]->Fill(l1ref->pt(),l1ref->phi());
	hpteta[ntrig][0]->Fill(l1ref->pt(),l1ref->eta());
	hl1quality[ntrig]->Fill(l1ref->gmtMuonCand().quality());
	_hpt1[ntrig][0]->Fill(l1ref->pt());
	_heta1[ntrig][0]->Fill(l1ref->eta());
	_hphi1[ntrig][0]->Fill(l1ref->phi());
	if ( !l2mucands.failedToGet() && l2mucands.isValid() ) {
	  for (cand=l2mucands->begin(); cand!=l2mucands->end(); ++cand) {
	    reco::TrackRef tk = cand->get<reco::TrackRef>();
	    edm::RefToBase<TrajectorySeed> seed=tk->seedRef();
	    if ( (l2seed->startingState()).detId() == (seed->startingState()).detId() ) {
	      if(tk->pt()*l1ref->pt() != 0 ) {
		hptres[ntrig][0]->Fill(1/l1ref->pt() - 1/tk->pt());
		hptrespt[ntrig][0]->Fill(tk->pt(), 1/l1ref->pt() - 1/tk->pt());
		hptrelres[ntrig][0]->Fill((1/l1ref->pt() - 1/tk->pt())/(1/tk->pt()));
		hptrelrespt[ntrig][0]->Fill(tk->pt(), (1/l1ref->pt() - 1/tk->pt())/(1/tk->pt()));
	      }
	      _hpt2[ntrig][0]->Fill(l1ref->pt());
	      _heta2[ntrig][0]->Fill(l1ref->eta());
	      _hphi2[ntrig][0]->Fill(l1ref->phi());
	      hetares[ntrig][0]->Fill(l1ref->eta()-tk->eta());
	      hetareseta[ntrig][0]->Fill(tk->eta(),l1ref->eta()-tk->eta());
	      hetarelres[ntrig][0]->Fill((l1ref->eta()-tk->eta())/tk->eta());
	      hetarelreseta[ntrig][0]->Fill(tk->eta(),(l1ref->eta()-tk->eta())/tk->eta());
	      hphires[ntrig][0]->Fill(l1ref->phi()-tk->phi());
	      double dphi=l1ref->phi()-tk->phi();
	      if (dphi>TMath::TwoPi())dphi-=2*TMath::TwoPi();
	      else if (dphi<-TMath::TwoPi()) dphi+=TMath::TwoPi();
	      hphiresphi[ntrig][0]->Fill(tk->phi(),dphi);
	      hphirelres[ntrig][0]->Fill((l1ref->phi()-tk->phi())/tk->phi());
	      hphirelresphi[ntrig][0]->Fill(tk->phi(),dphi/tk->phi());
	      // charge conversion
	      int chargeconv = -1;
	      int l1charge = l1ref->charge();
	      int l2charge = tk->charge();
	      if( l1charge == -1 && l2charge == -1 ) chargeconv = 0;
	      else if( l1charge == -1 && l2charge == 1 ) chargeconv = 1;
	      else if( l1charge == 1 && l2charge == -1 ) chargeconv = 2;
	      else if( l1charge == 1 && l2charge == 1 ) chargeconv = 3;
	      hchargeconv[ntrig][0]->Fill(chargeconv);
	      _hpt1[ntrig][1]->Fill(tk->pt());
	      _heta1[ntrig][1]->Fill(tk->eta());
	      _hphi1[ntrig][1]->Fill(tk->phi());
	      //find the L3 build from this L2
	      if (!l3mucands.failedToGet() && l3mucands.isValid() ) {
		for (cand=l3mucands->begin(); cand!=l3mucands->end(); ++cand) {
		  reco::TrackRef l3tk= cand->get<reco::TrackRef>();
		  if( l3tk->seedRef().castTo<edm::Ref<L3MuonTrajectorySeedCollection> > ().isAvailable() ) {
		    if (l3tk->seedRef().castTo<edm::Ref<L3MuonTrajectorySeedCollection> >()->l2Track() == tk){
		      if(l1ref->pt()*l3tk->pt() != 0 ) {
			hptres[ntrig][2]->Fill(1/l1ref->pt() - 1/l3tk->pt());
			hptrespt[ntrig][2]->Fill(l3tk->pt(), 1/l1ref->pt() - 1/l3tk->pt());
			hptrelres[ntrig][2]->Fill((1/l1ref->pt() - 1/l3tk->pt())/(1/l3tk->pt()));
			hptrelrespt[ntrig][2]->Fill(l3tk->pt(), (1/l1ref->pt() - 1/l3tk->pt())/(1/l3tk->pt()));
		      }
		      hetares[ntrig][2]->Fill(l1ref->eta()-l3tk->eta());
		      hetareseta[ntrig][2]->Fill(l1ref->eta(),l1ref->eta()-l3tk->eta());
		      hetarelres[ntrig][2]->Fill((l1ref->eta()-l3tk->eta())/l3tk->eta());
		      hetarelreseta[ntrig][2]->Fill(l1ref->eta(),(l1ref->eta()-l3tk->eta())/l3tk->eta());
		      hphires[ntrig][2]->Fill(l1ref->phi()-l3tk->phi());
		      double dphi=l1ref->phi()-l3tk->phi();
		      if (dphi>TMath::TwoPi())dphi-=2*TMath::TwoPi();
		      else if (dphi<-TMath::TwoPi()) dphi+=TMath::TwoPi();
		      hphiresphi[ntrig][2]->Fill(l3tk->phi(),dphi);
		      hphirelres[ntrig][2]->Fill((l1ref->phi()-l3tk->phi())/l3tk->phi());
		      hphirelresphi[ntrig][2]->Fill(l3tk->phi(),(dphi)/l3tk->phi());
		      // charge conversion
		      int chargeconv = -1;
		      int l1charge = l1ref->charge();
		      int l3charge = l3tk->charge();
		      if( l1charge == -1 && l3charge == -1 ) chargeconv = 0;
		      else if( l1charge == -1 && l3charge == 1 ) chargeconv = 1;
		      else if( l1charge == 1 && l3charge == -1 ) chargeconv = 2;
		      else if( l1charge == 1 && l3charge == 1 ) chargeconv = 3;
		      hchargeconv[ntrig][2]->Fill(chargeconv);
		      _hpt2[ntrig][1]->Fill(tk->pt());
    		      _heta2[ntrig][1]->Fill(tk->eta());
		      _hphi2[ntrig][1]->Fill(tk->phi());
		      //break; //plot only once per L2?
		    }//if
		  }
		}//for
	      }
	      break;
	    }
	  }
	}
      }
      // mapping
      std::map<l1extra::L1MuonParticleRef, int>::iterator it;
      for( it = l1map.begin(); it != l1map.end(); it++ ) {
	  hseedNMuper[ntrig][0]->Fill(it->second);
      }
      hNMu[ntrig][0]->Fill(l1map.size());
    }
    //else std::cout << "failed to get l2seed!" << std::endl;

    if (!l3seeds.failedToGet() && l3seeds.isValid() ) {
      hNMu[ntrig][4]->Fill(l3seeds->size());
      L3MuonTrajectorySeedCollection::const_iterator l3seed;
      std::map<reco::TrackRef, int> l2map;
      for (l3seed=l3seeds->begin() ; l3seed != l3seeds->end();++l3seed){
	PTrajectoryStateOnDet state=l3seed->startingState();
	// Transform Trajectory State on Det to a TSOS
	TrajectoryStateTransform tsTransform;
	DetId seedDetId(state.detId());
	const GeomDet* gdet = glbTrackingGeometry->idToDet( seedDetId );
	TrajectoryStateOnSurface tsos = tsTransform.transientState(state, &(gdet->surface()), magField.product());

	float pt = tsos.globalMomentum().perp();
	float eta = tsos.globalPosition().eta();
	float phi = tsos.globalPosition().phi();
	hcharge[ntrig][4]->Fill(tsos.charge());
	hpt[ntrig][4]->Fill(pt);
	hphi[ntrig][4]->Fill(phi);
	heta[ntrig][4]->Fill(eta);
	hetaphi[ntrig][4]->Fill(phi,eta);
	hptphi[ntrig][4]->Fill(pt,phi);
	hpteta[ntrig][4]->Fill(pt,eta);

	reco::TrackRef l2tkRef = l3seed->l2Track();
	l2map[l2tkRef]++;
	hseedptres[ntrig][1]->Fill(1/pt - 1/l2tkRef->pt());
	hseedetares[ntrig][1]->Fill(eta - l2tkRef->eta());
	hseedphires[ntrig][1]->Fill(phi - l2tkRef->phi());
	hseedptrelres[ntrig][1]->Fill((1/pt - 1/l2tkRef->pt())/(1/l2tkRef->pt()));
	hseedetarelres[ntrig][1]->Fill((eta - l2tkRef->eta())/l2tkRef->eta());
	hseedphirelres[ntrig][1]->Fill((phi - l2tkRef->phi())/l2tkRef->phi());
      }
      // mapping
      std::map<reco::TrackRef, int>::iterator it;
      for( it = l2map.begin(); it != l2map.end(); it++ ) {
	  hseedNMuper[ntrig][1]->Fill(it->second);
      }
    }
    
    //else std::cout << "failed to get l3seed!" << std::endl;

    reco::BeamSpot beamSpot;
    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByLabel("hltOfflineBeamSpot",recoBeamSpotHandle);
    if (!recoBeamSpotHandle.failedToGet())  beamSpot = *recoBeamSpotHandle;
    
    if (!l2mucands.failedToGet() && l2mucands.isValid() ) {
      LogDebug("HLTMuonDQMSource") << " filling L2 stuff " << std::endl;
      edm::Handle<reco::IsoDepositMap> l2depMap;
      if( l2mucands->size() != 0 ) iEvent.getByLabel (l2isolationTag_,l2depMap);
      hNMu[ntrig][1]->Fill(l2mucands->size());
      for (cand=l2mucands->begin(); cand!=l2mucands->end(); ++cand) {
	reco::TrackRef tk = cand->get<reco::TrackRef>();
	if (!l2depMap.failedToGet() && l2depMap.isValid() ) {
	  LogDebug("HLTMuonDQMSource") << " filling L2 Iso stuff " << std::endl;
	  if ( l2depMap->contains(tk.id()) ){
	    reco::IsoDepositMap::value_type calDeposit = (*l2depMap)[tk];
	    double dephlt = calDeposit.depositWithin(coneSize_);
	    if( dephlt != 0 ) hiso[ntrig][0]->Fill(dephlt);
	    // 2D plot: cone size vs. dep
	    double depL2[10];
	    for( int idep = 0; idep < 10; idep++ ) {
		double conesize = (idep + 1)*coneSize_/10.0;
		depL2[idep] = calDeposit.depositWithin(conesize);
		if( depL2[idep] != 0 ) hL2muonIsoDR[ntrig]->Fill(conesize, depL2[idep]);
	    }
	  }
	}
	
	// eta cut
	hpt[ntrig][1]->Fill(tk->pt());      
	hcharge[ntrig][1]->Fill(tk->charge()); 
	if ( tk->charge() != 0 ) {
	  heta[ntrig][1]->Fill(tk->eta());      
	  hphi[ntrig][1]->Fill(tk->phi()); 

	  if(hphi[ntrig][1]->getEntries()){
	    for(int ibin = 1; ibin <= hphi[ntrig][1]->getNbinsX(); ++ibin)
	      hphi_norm[ntrig][1]->setBinContent(ibin, 
						 hphi[ntrig][1]->getBinContent(ibin)/hphi[ntrig][1]->getEntries());
	  }
	  
	  hetaphi[ntrig][1]->Fill(tk->phi(),tk->eta()); 
	  hptphi[ntrig][1]->Fill(tk->pt(),tk->phi()); 
	  hpteta[ntrig][1]->Fill(tk->pt(),tk->eta()); 
	  const reco::HitPattern& hitp = tk->hitPattern();
	  hnHits[ntrig][1]->Fill(hitp.numberOfHits()); 
	  hnValidHits[ntrig]->Fill(hitp.numberOfValidHits()); 
	  hd0[ntrig][0]->Fill(tk->d0()); 
	  if (!recoBeamSpotHandle.failedToGet()){
	    hdr[ntrig][0]->Fill(tk->dxy(beamSpot.position()));	
	    hdrphi[ntrig][0]->Fill(tk->phi(),tk->dxy(beamSpot.position())); 
	  } 
	  hd0phi[ntrig][0]->Fill(tk->phi(),tk->d0()); 
	  hdz0[ntrig][0]->Fill(tk->dz()); 
	  hdz0eta[ntrig][0]->Fill(tk->eta(),tk->dz());
	  hdz[ntrig][0]->Fill(tk->dz(beamSpot.position())); 
	  hdzeta[ntrig][0]->Fill(tk->eta(),tk->dz(beamSpot.position()));
	  herr0[ntrig][0]->Fill(tk->error(0)); 
	  cand2=cand;
	  ++cand2;
	  for (; cand2!=l2mucands->end(); cand2++) {
	    reco::TrackRef tk2=cand2->get<reco::TrackRef>();
	    if ( tk->charge()*tk2->charge() == -1 ){
	      double mass=(cand->p4()+cand2->p4()).M();
	      hdimumass[ntrig][0]->Fill(mass);
	    }
	  }
	} else edm::LogWarning("HLTMonMuon")<<"stop filling candidate with update@Vtx failure";
      }
    }
    if (!l3mucands.failedToGet() && l3mucands.isValid() ) {
      LogDebug("HLTMuonDQMSource") << " filling L3 stuff " << std::endl;
      hNMu[ntrig][2]->Fill(l3mucands->size());
      edm::Handle<reco::IsoDepositMap> l3depMap;
      if( l3mucands->size() != 0 ) iEvent.getByLabel (l3isolationTag_,l3depMap);
      for (cand=l3mucands->begin(); cand!=l3mucands->end(); ++cand) {
	reco::TrackRef tk = cand->get<reco::TrackRef>();
	if (!l3depMap.failedToGet() && l3depMap.isValid() ) {
	  if ( l3depMap->contains(tk.id()) ){
	    reco::IsoDepositMap::value_type calDeposit= (*l3depMap)[tk];
	    double dephlt = calDeposit.depositWithin(coneSize_);
	    if( dephlt != 0 ) hiso[ntrig][1]->Fill(dephlt);
	  }
	}
	// eta cut
	hpt[ntrig][2]->Fill(tk->pt());      
	heta[ntrig][2]->Fill(tk->eta());      
	hphi[ntrig][2]->Fill(tk->phi()); 

	if(hphi[ntrig][2]->getEntries()){
	  for(int ibin = 1; ibin <= hphi[ntrig][2]->getNbinsX(); ++ibin)
	    hphi_norm[ntrig][2]->setBinContent(ibin, 
					       hphi[ntrig][2]->getBinContent(ibin)/hphi[ntrig][2]->getEntries());
	}

	hetaphi[ntrig][2]->Fill(tk->phi(),tk->eta()); 
	hptphi[ntrig][2]->Fill(tk->pt(),tk->phi()); 
	hpteta[ntrig][2]->Fill(tk->pt(),tk->eta()); 
	const reco::HitPattern& hitp = tk->hitPattern();
	hnHits[ntrig][2]->Fill(hitp.numberOfHits()); 
	hnTkValidHits[ntrig]->Fill(hitp.numberOfValidTrackerHits()); 
	hnMuValidHits[ntrig]->Fill(hitp.numberOfValidMuonHits()); 
	hd0[ntrig][1]->Fill(tk->d0()); 
	if (!recoBeamSpotHandle.failedToGet()) {
	  hdr[ntrig][1]->Fill(tk->dxy(beamSpot.position()));
	  hdrphi[ntrig][1]->Fill(tk->phi(),tk->dxy(beamSpot.position())); 
	}
	hd0phi[ntrig][1]->Fill(tk->phi(),tk->d0()); 
	hdz0[ntrig][1]->Fill(tk->dz()); 
	hdz0eta[ntrig][1]->Fill(tk->eta(),tk->dz());
	hdz[ntrig][1]->Fill(tk->dz(beamSpot.position())); 
	hdzeta[ntrig][1]->Fill(tk->eta(),tk->dz(beamSpot.position()));
	herr0[ntrig][1]->Fill(tk->error(0)); 
	hcharge[ntrig][2]->Fill(tk->charge()); 
	cand2=cand;
	++cand2;
	
	for (; cand2!=l3mucands->end(); cand2++) {
	  reco::TrackRef tk2=cand2->get<reco::TrackRef>();
	  if ( tk->charge()*tk2->charge() == -1 ){
	    double mass=(cand->p4()+cand2->p4()).M();
	    hdimumass[ntrig][1]->Fill(mass);
	  }
	}
	if( tk->seedRef().castTo<edm::Ref<L3MuonTrajectorySeedCollection> >().isAvailable() ) {
	  reco::TrackRef l2tk = tk->seedRef().castTo<edm::Ref<L3MuonTrajectorySeedCollection> >()->l2Track();
	  if(tk->pt()*l2tk->pt() != 0 ) {
	    hptres[ntrig][1]->Fill(1/l2tk->pt() - 1/tk->pt());
	    hptrespt[ntrig][1]->Fill(tk->pt(), 1/l2tk->pt() - 1/tk->pt());
	    hptrelres[ntrig][1]->Fill((1/l2tk->pt() - 1/tk->pt())/(1/tk->pt()));
	    hptrelrespt[ntrig][1]->Fill(tk->pt(), (1/l2tk->pt() - 1/tk->pt())/(1/tk->pt()));
	    double pterr = (tk->ptError()/(tk->pt()*tk->pt()));
	    hptpull[ntrig]->Fill((1/l2tk->pt() - 1/tk->pt())/pterr);
	    hptpullpt[ntrig]->Fill(tk->pt(), (1/l2tk->pt() - 1/tk->pt())/pterr);
	  }
	  hphires[ntrig][1]->Fill(l2tk->phi()-tk->phi());
	  double dphi=l2tk->phi()-tk->phi();
	  if (dphi>TMath::TwoPi())dphi-=2*TMath::TwoPi();
	  else if (dphi<-TMath::TwoPi()) dphi+=TMath::TwoPi();
	  hphiresphi[ntrig][1]->Fill(tk->phi(),dphi);
	  hphirelres[ntrig][1]->Fill((l2tk->phi()-tk->phi())/tk->phi());
	  hphirelresphi[ntrig][1]->Fill(tk->phi(),dphi/tk->phi());
	  hphipull[ntrig]->Fill(dphi/tk->phiError());
	  hphipullphi[ntrig]->Fill(tk->phi(), dphi/tk->phiError());
	  hetares[ntrig][1]->Fill(l2tk->eta()-tk->eta());
	  hetareseta[ntrig][1]->Fill(tk->eta(),l2tk->eta()-tk->eta());
	  hetarelres[ntrig][1]->Fill((l2tk->eta()-tk->eta())/tk->eta());
	  hetarelreseta[ntrig][1]->Fill(tk->eta(),(l2tk->eta()-tk->eta())/tk->eta());
	  hetapull[ntrig]->Fill((l2tk->eta()-tk->eta())/tk->etaError());
	  hetapulleta[ntrig]->Fill(tk->eta(),(l2tk->eta()-tk->eta())/tk->etaError());
	  // charge conversion
	  int chargeconv = -1;
	  int l2charge = l2tk->charge();
	  int l3charge = tk->charge();
	  if( l2charge == -1 && l3charge == -1 ) chargeconv = 0;
	  else if( l2charge == -1 && l3charge == 1 ) chargeconv = 1;
	  else if( l2charge == 1 && l3charge == -1 ) chargeconv = 2;
	  else if( l2charge == 1 && l3charge == 1 ) chargeconv = 3;
	  hchargeconv[ntrig][1]->Fill(chargeconv);
	}
      }
    }  
    
    for( int level = 0; level < 2; level++ ) {  
      for( int nbin = 1; nbin < _hpt1[ntrig][level]->GetNbinsX()+1; nbin++ ) {
	if( _hpt1[ntrig][level]->GetBinContent(nbin) != 0 ) {
	  double frac = _hpt2[ntrig][level]->GetBinContent(nbin)/_hpt1[ntrig][level]->GetBinContent(nbin);
	  double err = sqrt(frac*fabs(1 - frac)/_hpt1[ntrig][level]->GetBinContent(nbin));
	  hptfrac[ntrig][level]->setBinContent(nbin, frac);
	  hptfrac[ntrig][level]->setBinError(nbin, err);
	}
	if( _heta1[ntrig][level]->GetBinContent(nbin) != 0 ) {
	  double frac = _heta2[ntrig][level]->GetBinContent(nbin)/_heta1[ntrig][level]->GetBinContent(nbin);
	  double err = sqrt(frac*fabs(1 - frac)/_heta1[ntrig][level]->GetBinContent(nbin));
	  hetafrac[ntrig][level]->setBinContent(nbin, frac);
	  hetafrac[ntrig][level]->setBinError(nbin, err);
	}
	if( _hphi1[ntrig][level]->GetBinContent(nbin) != 0 ) {
	  double frac = _hphi2[ntrig][level]->GetBinContent(nbin)/_hphi1[ntrig][level]->GetBinContent(nbin);
	  double err = sqrt(frac*fabs(1 - frac)/_hphi1[ntrig][level]->GetBinContent(nbin));
	  hphifrac[ntrig][level]->setBinContent(nbin, frac);
	  hphifrac[ntrig][level]->setBinError(nbin, err);
	}
	else {
	  hptfrac[ntrig][level]->setBinContent(nbin, 0.0);
	  hetafrac[ntrig][level]->setBinContent(nbin, 0.0);
	  hphifrac[ntrig][level]->setBinContent(nbin, 0.0);
	  hptfrac[ntrig][level]->setBinError(nbin, 0.0);
	  hetafrac[ntrig][level]->setBinError(nbin, 0.0);
	  hphifrac[ntrig][level]->setBinError(nbin, 0.0);
	}
      }
    }
  }


  // Tower
  edm::Handle<CaloTowerCollection> caloTower; 
  if( !l2mucands.failedToGet() && l2mucands->size() != 0 )
    iEvent.getByLabel(edm::InputTag("hltTowerMakerForMuons"), caloTower);
  for( int ntrig = 0; ntrig < nTrigs; ntrig++ ) {
    if( !FiredTriggers[ntrig] ) continue;
    if( caloTower.isValid() ) {
      CaloTowerCollection::const_iterator itower;
      for( itower = caloTower->begin(); itower != caloTower->end(); ++itower ) {
	  if( (*itower).et() == 0 ) continue;

	  // deltaR with l2muon
	  if (!l2mucands.failedToGet() && l2mucands.isValid() ) {
	      for (cand=l2mucands->begin(); cand!=l2mucands->end(); ++cand) {
		  reco::TrackRef l2tk= cand->get<reco::TrackRef>();
	          double deltaR_l2 = reco::deltaR(*l2tk, (*itower));
		  htowerdRL2[ntrig]->Fill(deltaR_l2);
	      }
	  }
	  // deltaR with l3muon
	  if (!l3mucands.failedToGet() && l3mucands.isValid() ) {
	      for (cand=l3mucands->begin(); cand!=l3mucands->end(); ++cand) {
		  reco::TrackRef l3tk= cand->get<reco::TrackRef>();
	          double deltaR_l3 = reco::deltaR(*l3tk, (*itower));
		  htowerdRL3[ntrig]->Fill(deltaR_l3);
	      }
	  }
	  htowerEt[ntrig]->Fill((*itower).et());
	  htowerPhi[ntrig]->Fill((*itower).phi());
	  htowerEta[ntrig]->Fill((*itower).eta());
	  
	  // energy contributions from different detectors
	  // energy in HO ("outerEnergy")is not included in "hadEnergy"
	  htowerOuterEnergy[ntrig]->Fill((*itower).outerEnergy());
	  htowerHadEnergy[ntrig]->Fill((*itower).hadEnergy());
	  htowerEmEnergy[ntrig]->Fill((*itower).emEnergy());

	  htowerOuterEt[ntrig]->Fill((*itower).outerEt());
	  htowerHadEt[ntrig]->Fill((*itower).hadEt());
	  htowerEmEt[ntrig]->Fill((*itower).emEt());

	  // 2D plots
	  htowerPhiOuterEt[ntrig]->Fill((*itower).phi(), (*itower).outerEt());
	  htowerPhiHadEt[ntrig]->Fill((*itower).phi(), (*itower).hadEt());
	  htowerPhiEmEt[ntrig]->Fill((*itower).phi(), (*itower).emEt());
	  htowerEtaOuterEt[ntrig]->Fill((*itower).eta(), (*itower).outerEt());
	  htowerEtaHadEt[ntrig]->Fill((*itower).eta(), (*itower).hadEt());
	  htowerEtaEmEt[ntrig]->Fill((*itower).eta(), (*itower).emEt());
      }
    }
  }
  
}




//--------------------------------------------------------
void HLTMuonDQMSource::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
					  const edm::EventSetup& context) {
}
//--------------------------------------------------------
void HLTMuonDQMSource::endRun(const edm::Run& r, const edm::EventSetup& context){
}
//--------------------------------------------------------
void HLTMuonDQMSource::endJob(){
  edm::LogInfo("HLTMonMuon") << "analyzed " << counterEvt_ << " events";
  //std::cout << "analyzed = " << counterEvt_ << " , triggered = " << nTrig_ << std::endl;
  
  
  //if (outputFile_.size() != 0 && dbe_)
  //dbe_->save(outputFile_);
  
  return;
}
