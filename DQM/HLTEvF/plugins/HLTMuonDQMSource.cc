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
// $Id: HLTMuonDQMSource.cc,v 1.40 2010/08/07 14:55:56 wmtan Exp $
// Modification:  Sanghyeon Song (Chonnam National University)
// contact: sanghyeon.song@cern.ch
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

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"

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


  l3seedscollectionTagOIState_ = parameters_.getUntrackedParameter<edm::InputTag>("l3MuonSeedTagOIState",edm::InputTag("hltL3TrajSeedOIState"));
  l3seedscollectionTagOIHit_ = parameters_.getUntrackedParameter<edm::InputTag>("l3MuonSeedTagOIHit",edm::InputTag("hltL3TrajSeedOIHit"));
  l3seedscollectionTagIOHit_ = parameters_.getUntrackedParameter<edm::InputTag>("l3MuonSeedTagIOHit",edm::InputTag("hltL3TrajSeedIOHit"));

  l3trkfindingOIState_ = parameters_.getUntrackedParameter<edm::InputTag>("l3MuonTrkFindingOIState",edm::InputTag("hltL3TrackCandidateFromL2OIState"));
  l3trkfindingOIHit_ = parameters_.getUntrackedParameter<edm::InputTag>("l3MuonTrkFindingOIHit",edm::InputTag("hltL3TrackCandidateFromL2OIHit"));
  l3trkfindingIOHit_ = parameters_.getUntrackedParameter<edm::InputTag>("l3MuonTrkFindingIOHit",edm::InputTag("hltL3TrackCandidateFromL2IOHit"));

  l3trkOIState_ = parameters_.getUntrackedParameter<edm::InputTag>("l3MuonTrkOIState",edm::InputTag("hltL3TkTracksFromL2OIState"));
  l3trkOIHit_ = parameters_.getUntrackedParameter<edm::InputTag>("l3MuonTrkOIHit",edm::InputTag("hltL3TkTracksFromL2OIHit"));
  l3trkIOHit_ = parameters_.getUntrackedParameter<edm::InputTag>("l3MuonTrkIOHit",edm::InputTag("hltL3TkTracksFromL2IOHit"));
  l3tktrk_ = parameters_.getUntrackedParameter<edm::InputTag>("l3MuonTrk",edm::InputTag("hltL3TkTracksFromL2"));

  l3muons_ = parameters_.getUntrackedParameter<edm::InputTag>("l3muons",edm::InputTag("hltL3Muons"));
  l3muonsOIState_ = parameters_.getUntrackedParameter<edm::InputTag>("l3muonsOIState",edm::InputTag("hltL3MuonsOIState"));
  l3muonsOIHit_ = parameters_.getUntrackedParameter<edm::InputTag>("l3muonsOIHit",edm::InputTag("hltL3MuonsOIHit"));
  l3muonsIOHit_ = parameters_.getUntrackedParameter<edm::InputTag>("l3muonsIOHit",edm::InputTag("hltL3MuonsIOHit"));

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
        else if (level == 6 ) sprintf(name,"Isolation");

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

          hnTkValidHits[trig] = dbe_->book1D("HLTMuonL3_nTkValidHits", "L3 Number of Valid Tracker Hits", 30, 0., 30.);
          hnTkValidHits[trig]->setAxisTitle("Number of Valid Tracker Hits", 1);
          hnPixelValidHits[trig] = dbe_->book1D("HLTMuonL3_nPixelValidHits", "L3 Number of Valid Pixel Hits", 10, 0., 10.);
          hnPixelValidHits[trig]->setAxisTitle("Number of Valid Pixel Hits", 1);
          hnStripValidHits[trig] = dbe_->book1D("HLTMuonL3_nStripValidHits", "L3 Number of Valid Strip Hits", 20, 0., 20.);
          hnStripValidHits[trig]->setAxisTitle("Number of Valid Strip Hits", 1);
          hnMuValidHits[trig] = dbe_->book1D("HLTMuonL3_nMuValidHits", "L3 Number of Valid Muon Hits", 100, 0., 100.);
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
          hNMu[trig][level-1] = dbe_->book1D(name,title, 11, -0.5, 10.5);
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
          hNMu[trig][level-1] = dbe_->book1D(name,title,11, -0.5, 10.5);
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
          hNMu[trig][level-1] = dbe_->book1D(name,title, 11, -0.5, 10.5);
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

          if( level == 5 ){
            dbe_->setCurrentFolder(monitorName_+dirname+"Level3Seed");

            sprintf(name,"HLTMuonL3Seed_NMu_comp");
            sprintf(title,"Comparison of each HLT L3 Muon Seed");
            hNMu_l3seed_comp[trig][level-1] = dbe_->book1D(name,title,4,0,4);
            hNMu_l3seed_comp[trig][level-1]->setAxisTitle("Number of muons",1);
            hNMu_l3seed_comp[trig][level-1]->setBinLabel(1, "Cascade", 1);
            hNMu_l3seed_comp[trig][level-1]->setBinLabel(2, "OIState", 1);
            hNMu_l3seed_comp[trig][level-1]->setBinLabel(3, "OIHit", 1);
            hNMu_l3seed_comp[trig][level-1]->setBinLabel(4, "IOHit", 1);

            dbe_->setCurrentFolder(monitorName_+dirname+"Level3Seed/OIState");

            sprintf(name,"HLTMuonL%iSeed_NMu_OIState",level-2);
            sprintf(title,"OIState L%iSeed number of muons",level-2);
            hNMu_OIState[trig][level-1] = dbe_->book1D(name,title, 11, -0.5, 10.5);
            hNMu_OIState[trig][level-1]->setAxisTitle("Number of muons", 1);
            sprintf(name,"HLTMuonL%iSeed_pt_OIState",level-2);
            sprintf(title,"L%iSeed OIState Muon Pt",level-2);
            hpt_OIState[trig][level-1] = dbe_->book1D(name,title, NBINS, 0., pt_max);
            hpt_OIState[trig][level-1]->setAxisTitle("Pt", 1);
            sprintf(name,"HLTMuonL%iSeed_eta_OIState",level-2);
            sprintf(title,"L%iSeed OIState Muon #eta",level-2);
            heta_OIState[trig][level-1] = dbe_->book1D(name,title, NBINS, -2.5, 2.5);
            heta_OIState[trig][level-1]->setAxisTitle("#eta", 1);
            sprintf(name,"HLTMuonL%iSeed_phi_OIState",level-2);
            sprintf(title,"L%iSeed OIState Muon #phi",level-2);
            hphi_OIState[trig][level-1] = dbe_->book1D(name,title, NBINS, -3.15, 3.15);
            hphi_OIState[trig][level-1]->setAxisTitle("#phi", 1);
            sprintf(name,"HLTMuonL%iSeed_etaphi_OIState",level-2);
            sprintf(title,"L%iSeed OIState Muon #eta vs #phi",level-2);
            hetaphi_OIState[trig][level-1] = dbe_->book2D(name,title, NBINS, -3.15, 3.15, NBINS, -2.5, 2.5);
            hetaphi_OIState[trig][level-1]->setAxisTitle("#phi", 1);
            hetaphi_OIState[trig][level-1]->setAxisTitle("#eta", 2);
            sprintf(name,"HLTMuonL%iSeed_ptphi_OIState",level-2);
            sprintf(title,"L%iSeed OIState Muon pt vs #phi",level-2);
            hptphi_OIState[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., pt_max, NBINS, -3.15, 3.15);
            hptphi_OIState[trig][level-1]->setAxisTitle("pt", 1);
            hptphi_OIState[trig][level-1]->setAxisTitle("#phi", 2);
            sprintf(name,"HLTMuonL%iSeed_pteta_OIState",level-2);
            sprintf(title,"L%iSeed OIState Muon pt vs #eta",level-2);
            hpteta_OIState[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., pt_max, NBINS, -2.5, 2.5);
            hpteta_OIState[trig][level-1]->setAxisTitle("pt", 1);
            hpteta_OIState[trig][level-1]->setAxisTitle("#eta", 2);
            sprintf(name,"HLTMuonL%iSeed_charge_OIState",level-2);
            sprintf(title,"L%iSeed OIState Muon Charge",level-2);
            hcharge_OIState[trig][level-1]  = dbe_->book1D(name,title, 3, -1.5, 1.5);
            hcharge_OIState[trig][level-1]->setAxisTitle("Charge", 1);

            dbe_->setCurrentFolder(monitorName_+dirname+"Level3Seed/OIHit");

            sprintf(name,"HLTMuonL%iSeed_NMu_OIHit",level-2);
            sprintf(title,"OIHit L%iSeed number of muons",level-2);
            hNMu_OIHit[trig][level-1] = dbe_->book1D(name,title, 11, -0.5, 10.5);
            hNMu_OIHit[trig][level-1]->setAxisTitle("Number of muons", 1);
            sprintf(name,"HLTMuonL%iSeed_pt_OIHit",level-2);
            sprintf(title,"L%iSeed OIHit Muon Pt",level-2);
            hpt_OIHit[trig][level-1] = dbe_->book1D(name,title, NBINS, 0., pt_max);
            hpt_OIHit[trig][level-1]->setAxisTitle("Pt", 1);
            sprintf(name,"HLTMuonL%iSeed_eta_OIHit",level-2);
            sprintf(title,"L%iSeed OIHit Muon #eta",level-2);
            heta_OIHit[trig][level-1] = dbe_->book1D(name,title, NBINS, -2.5, 2.5);
            heta_OIHit[trig][level-1]->setAxisTitle("#eta", 1);
            sprintf(name,"HLTMuonL%iSeed_phi_OIHit",level-2);
            sprintf(title,"L%iSeed OIHit Muon #phi",level-2);
            hphi_OIHit[trig][level-1] = dbe_->book1D(name,title, NBINS, -3.15, 3.15);
            hphi_OIHit[trig][level-1]->setAxisTitle("#phi", 1);
            sprintf(name,"HLTMuonL%iSeed_etaphi_OIHit",level-2);
            sprintf(title,"L%iSeed OIHit Muon #eta vs #phi",level-2);
            hetaphi_OIHit[trig][level-1] = dbe_->book2D(name,title, NBINS, -3.15, 3.15, NBINS, -2.5, 2.5);
            hetaphi_OIHit[trig][level-1]->setAxisTitle("#phi", 1);
            hetaphi_OIHit[trig][level-1]->setAxisTitle("#eta", 2);
            sprintf(name,"HLTMuonL%iSeed_ptphi_OIHit",level-2);
            sprintf(title,"L%iSeed OIHit Muon pt vs #phi",level-2);
            hptphi_OIHit[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., pt_max, NBINS, -3.15, 3.15);
            hptphi_OIHit[trig][level-1]->setAxisTitle("pt", 1);
            hptphi_OIHit[trig][level-1]->setAxisTitle("#phi", 2);
            sprintf(name,"HLTMuonL%iSeed_pteta_OIHit",level-2);
            sprintf(title,"L%iSeed OIHit Muon pt vs #eta",level-2);
            hpteta_OIHit[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., pt_max, NBINS, -2.5, 2.5);
            hpteta_OIHit[trig][level-1]->setAxisTitle("pt", 1);
            hpteta_OIHit[trig][level-1]->setAxisTitle("#eta", 2);
            sprintf(name,"HLTMuonL%iSeed_charge_OIHit",level-2);
            sprintf(title,"L%iSeed OIHit Muon Charge",level-2);
            hcharge_OIHit[trig][level-1]  = dbe_->book1D(name,title, 3, -1.5, 1.5);
            hcharge_OIHit[trig][level-1]->setAxisTitle("Charge", 1);

            dbe_->setCurrentFolder(monitorName_+dirname+"Level3Seed/IOHit");

            sprintf(name,"HLTMuonL%iSeed_NMu_IOHit",level-2);
            sprintf(title,"IOHit L%iSeed number of muons",level-2);
            hNMu_IOHit[trig][level-1] = dbe_->book1D(name,title, 11, -0.5, 10.5);
            hNMu_IOHit[trig][level-1]->setAxisTitle("Number of muons", 1);
            sprintf(name,"HLTMuonL%iSeed_pt_IOHit",level-2);
            sprintf(title,"L%iSeed IOHit Muon Pt",level-2);
            hpt_IOHit[trig][level-1] = dbe_->book1D(name,title, NBINS, 0., pt_max);
            hpt_IOHit[trig][level-1]->setAxisTitle("Pt", 1);
            sprintf(name,"HLTMuonL%iSeed_eta_IOHit",level-2);
            sprintf(title,"L%iSeed IOHit Muon #eta",level-2);
            heta_IOHit[trig][level-1] = dbe_->book1D(name,title, NBINS, -2.5, 2.5);
            heta_IOHit[trig][level-1]->setAxisTitle("#eta", 1);
            sprintf(name,"HLTMuonL%iSeed_phi_IOHit",level-2);
            sprintf(title,"L%iSeed IOHit Muon #phi",level-2);
            hphi_IOHit[trig][level-1] = dbe_->book1D(name,title, NBINS, -3.15, 3.15);
            hphi_IOHit[trig][level-1]->setAxisTitle("#phi", 1);
            sprintf(name,"HLTMuonL%iSeed_etaphi_IOHit",level-2);
            sprintf(title,"L%iSeed IOHit Muon #eta vs #phi",level-2);
            hetaphi_IOHit[trig][level-1] = dbe_->book2D(name,title, NBINS, -3.15, 3.15, NBINS, -2.5, 2.5);
            hetaphi_IOHit[trig][level-1]->setAxisTitle("#phi", 1);
            hetaphi_IOHit[trig][level-1]->setAxisTitle("#eta", 2);
            sprintf(name,"HLTMuonL%iSeed_ptphi_IOHit",level-2);
            sprintf(title,"L%iSeed IOHit Muon pt vs #phi",level-2);
            hptphi_IOHit[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., pt_max, NBINS, -3.15, 3.15);
            hptphi_IOHit[trig][level-1]->setAxisTitle("pt", 1);
            hptphi_IOHit[trig][level-1]->setAxisTitle("#phi", 2);
            sprintf(name,"HLTMuonL%iSeed_pteta_IOHit",level-2);
            sprintf(title,"L%iSeed IOHit Muon pt vs #eta",level-2);
            hpteta_IOHit[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., pt_max, NBINS, -2.5, 2.5);
            hpteta_IOHit[trig][level-1]->setAxisTitle("pt", 1);
            hpteta_IOHit[trig][level-1]->setAxisTitle("#eta", 2);
            sprintf(name,"HLTMuonL%iSeed_charge_IOHit",level-2);
            sprintf(title,"L%iSeed IOHit Muon Charge",level-2);
            hcharge_IOHit[trig][level-1]  = dbe_->book1D(name,title, 3, -1.5, 1.5);
            hcharge_IOHit[trig][level-1]->setAxisTitle("Charge", 1);

          } 
        }
        else if( level == 6 ) {
          double max_energy = 1.0;

          sprintf(name,"HLTMuonL2_driso");
          sprintf(title,"dR Cone Size");
          hL2muonIsoDR[trig] = dbe_->book2D(name, title, 10, 0, 0.24, NBINS, 0, 5.0);
          hL2muonIsoDR[trig]->setAxisTitle(title, 1);
          hL2muonIsoDR[trig]->setAxisTitle("Calo Energy in Iso Cone (GeV)", 2);

          sprintf(name,"HLTMuonL3_driso");
          sprintf(title,"dR Cone Size");
          hL3muonIsoDR[trig] = dbe_->book2D(name, title, 10, 0, 0.24, NBINS, 0, 5.0);
          hL3muonIsoDR[trig]->setAxisTitle(title, 1);
          hL3muonIsoDR[trig]->setAxisTitle("Calo Energy in Iso Cone (GeV)", 2);

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

          sprintf(name,"HLTMuonL2_iso");
          sprintf(title,"L2 Muon Energy in Isolation cone");         
          hisoL2[trig] = dbe_->book1D(name,title, NBINS, 0., 5./1);
          hisoL2[trig]->setAxisTitle("Calo Energy in Iso Cone (GeV)", 1);
          sprintf(name,"HLTMuonL3_iso");
          sprintf(title,"L3 Muon SumPt in Isolation cone");               
          hisoL3[trig] = dbe_->book1D(name,title, NBINS, 0., 5./2);
          hisoL3[trig]->setAxisTitle("Sum Pt in Iso Cone (GeV)", 1);
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
        if( level == 3 ){
          sprintf(name,"HLTMuonL3_NMu_comp");
          sprintf(title,"Comparison of each HLT L3 Muons");
          hNMu_comp[trig][level-1] = dbe_->book1D(name,title,4,0,4);
          hNMu_comp[trig][level-1]->setAxisTitle("Number of muons",1);
          hNMu_comp[trig][level-1]->setBinLabel(1, "Cascade", 1);
          hNMu_comp[trig][level-1]->setBinLabel(2, "OIState", 1);
          hNMu_comp[trig][level-1]->setBinLabel(3, "OIHit", 1);
          hNMu_comp[trig][level-1]->setBinLabel(4, "IOHit", 1);
          sprintf(name,"HLTMuonL3_trk_NMu_comp");
          sprintf(title,"Comparison of each HLT L3 Muons");
          hNMu_trk_comp[trig][level-1] = dbe_->book1D(name,title,4,0,4);
          hNMu_trk_comp[trig][level-1]->setAxisTitle("Number of muons",1);
          hNMu_trk_comp[trig][level-1]->setBinLabel(1, "Cascade", 1);
          hNMu_trk_comp[trig][level-1]->setBinLabel(2, "OIState", 1);
          hNMu_trk_comp[trig][level-1]->setBinLabel(3, "OIHit", 1);
          hNMu_trk_comp[trig][level-1]->setBinLabel(4, "IOHit", 1);

          sprintf(name,"HLTMuonL%i_trk_NMu",level);
          sprintf(title," L%i tracker number of muons",level);
          hNMu_trk[trig][level-1] = dbe_->book1D(name,title, 10, -0.5, 10.5);
          hNMu_trk[trig][level-1]->setAxisTitle("Number of muons", 1);
          sprintf(name,"HLTMuonL%i_trk_pt",level);
          sprintf(title,"L%i tracker Muon Pt",level);
          hpt_trk[trig][level-1] = dbe_->book1D(name,title, NBINS, 0., 200);
          hpt_trk[trig][level-1]->setAxisTitle("Pt", 1);
          sprintf(name,"HLTMuonL%i_trk_eta",level);
          sprintf(title,"L%i tracker Muon #eta",level);
          heta_trk[trig][level-1] = dbe_->book1D(name,title, NBINS, -2.5, 2.5);
          heta_trk[trig][level-1]->setAxisTitle("#eta", 1);
          sprintf(name,"HLTMuonL%i_trk_phi",level);
          sprintf(title,"L%i tracker Muon #phi",level);
          hphi_trk[trig][level-1] = dbe_->book1D(name,title, NBINS, -3.15, 3.15);
          hphi_trk[trig][level-1]->setAxisTitle("#phi", 1);
          sprintf(name,"HLTMuonL%i_trk_etaphi",level);
          sprintf(title,"L%i tracker Muon #eta vs #phi",level);
          hetaphi_trk[trig][level-1] = dbe_->book2D(name,title, NBINS, -3.15, 3.15, NBINS, -2.5, 2.5);
          hetaphi_trk[trig][level-1]->setAxisTitle("#phi", 1);
          hetaphi_trk[trig][level-1]->setAxisTitle("#eta", 2);
          sprintf(name,"HLTMuonL%i_trk_ptphi",level);
          sprintf(title,"L%i tracker Muon pt vs #phi",level);
          hptphi_trk[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., 200, NBINS, -3.15, 3.15);
          hptphi_trk[trig][level-1]->setAxisTitle("pt", 1);
          hptphi_trk[trig][level-1]->setAxisTitle("#phi", 2);
          sprintf(name,"HLTMuonL%i_trk_pteta",level);
          sprintf(title,"L%i tracker Muon pt vs #eta",level);
          hpteta_trk[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., 200, NBINS, -2.5, 2.5);
          hpteta_trk[trig][level-1]->setAxisTitle("pt", 1);
          hpteta_trk[trig][level-1]->setAxisTitle("#eta", 2);
          sprintf(name,"HLTMuonL%i_trk_charge",level);
          sprintf(title,"L%i tracker Muon Charge",level);
          hcharge_trk[trig][level-1]  = dbe_->book1D(name,title, 3, -1.5, 1.5);
          hcharge_trk[trig][level-1]->setAxisTitle("Charge", 1);
          sprintf(name,"HLTMuonL%i_trk_d0",level);
          sprintf(title,"L%i tracker Muon from  radial impact vs (0,0)",level);
          hd0_trk[trig][level-1] = dbe_->book1D(name,title, 50, -0.3, 0.3 );
          hd0_trk[trig][level-1]->setAxisTitle("R Impact (cm) vs 0,0", 1);
          sprintf(name,"HLTMuonL%i_trk_dz",level);
          sprintf(title,"L%i tracker Muon from Z impact vs BeamSpot",level);
          hdz_trk[trig][level-1] = dbe_->book1D(name,title, NBINS, -25., 25.);
          hdz_trk[trig][level-1]->setAxisTitle("Z impact (cm) vs BeamSpot", 1);
          sprintf(name,"HLTMuonL%i_trk_dr",level);
          sprintf(title,"L%i tracker Muon radial impact vs BeamSpot",level);
          hdr_trk[trig][level-1] = dbe_->book1D(name,title, 50, -0.3, 0.3 );
          hdr_trk[trig][level-1]->setAxisTitle("R Impact (cm) vs BeamSpot", 1);

          hptres_L3L3trk[trig][level-1] = dbe_->book1D("HLTMuonL3toL3trk_ptres","L3Muon 1/Pt - L3trkMuon 1/Pt;L3 1/P_{T} - L3 trk 1/P_{T}",50+1,-0.02,0.02);

          hetares_L3L3trk[trig][level-1] = dbe_->book1D("HLTMuonL3toL3trk_etares","L3Muon #eta - L3trkMuon #eta;L3 #eta - L3 trk #eta",50,-0.02,0.02);
          hphires_L3L3trk[trig][level-1] = dbe_->book1D("HLTMuonL3toL3trk_phires","L3Muon #phi - L3trkMuon #phi;L3 #phi - L3 trk #phi",50,-0.02,0.02);
          hptrelres_L3L3trk[trig][level-1] = dbe_->book1D("HLTMuonL3toL3trk_ptrelres","(L3Muon 1/Pt - L3trkMuon 1/Pt)/(L3Muon 1/Pt);(L3 1/P_{T} - L3 trk 1/P_{T})/(L3 1/P_{T})",50,-0.05,0.05);
          hetarelres_L3L3trk[trig][level-1] = dbe_->book1D("HLTMuonL3toL3trk_etarelres","(L3Muon #eta - L3trkMuon #eta)/L3Muon #eta;(L3 #eta - L3 trk #eta)/L3 #eta",50,-0.02,0.02);
          hphirelres_L3L3trk[trig][level-1] = dbe_->book1D("HLTMuonL3toL3trk_phirelres","(L3Muon #phi - L3trkMuon #phi)/L3Muon #phi;(L3 #phi - L3 trk #phi)/L3 #phi",50,-0.02,0.02);

          dbe_->setCurrentFolder(monitorName_+dirname+"Level3/OIState");

          sprintf(name,"HLTMuonL%i_NMu_OIState",level);
          sprintf(title,"OIState L%i number of muons",level);
          hNMu_OIState[trig][level-1] = dbe_->book1D(name,title, 10, -0.5, 10.5);
          hNMu_OIState[trig][level-1]->setAxisTitle("Number of muons", 1);
          sprintf(name,"HLTMuonL%i_pt_OIState",level);
          sprintf(title,"L%i Pt_OIState",level);
          hpt_OIState[trig][level-1] = dbe_->book1D(name,title, NBINS, 0., 200);
          hpt_OIState[trig][level-1]->setAxisTitle("Pt", 1);
          sprintf(name,"HLTMuonL%i_eta_OIState",level);
          sprintf(title,"L%i Muon #eta_OIState",level);
          heta_OIState[trig][level-1] = dbe_->book1D(name,title, NBINS, -2.5, 2.5);
          heta_OIState[trig][level-1]->setAxisTitle("#eta", 1);
          sprintf(name,"HLTMuonL%i_phi_OIState",level);
          sprintf(title,"L%i Muon #phi_OIState",level);
          hphi_OIState[trig][level-1] = dbe_->book1D(name,title, NBINS, -3.15, 3.15);
          hphi_OIState[trig][level-1]->setAxisTitle("#phi", 1);
          sprintf(name,"HLTMuonL%i_etaphi_OIState",level);
          sprintf(title,"L%i Muon #eta vs #phi_OIState",level);
          hetaphi_OIState[trig][level-1] = dbe_->book2D(name,title, NBINS, -3.15, 3.15, NBINS, -2.5, 2.5);
          hetaphi_OIState[trig][level-1]->setAxisTitle("#phi", 1);
          hetaphi_OIState[trig][level-1]->setAxisTitle("#eta", 2);
          sprintf(name,"HLTMuonL%i_ptphi_OIState",level);
          sprintf(title,"L%i Muon pt vs #phi_OIState",level);
          hptphi_OIState[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., 200, NBINS, -3.15, 3.15);
          hptphi_OIState[trig][level-1]->setAxisTitle("pt", 1);
          hptphi_OIState[trig][level-1]->setAxisTitle("#phi", 2);
          sprintf(name,"HLTMuonL%i_pteta_OIState",level);
          sprintf(title,"L%i Muon pt vs #eta_OIState",level);
          hpteta_OIState[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., 200, NBINS, -2.5, 2.5);
          hpteta_OIState[trig][level-1]->setAxisTitle("pt", 1);
          hpteta_OIState[trig][level-1]->setAxisTitle("#eta", 2);
          sprintf(name,"HLTMuonL%i_charge_OIState",level);
          sprintf(title,"L%i Muon Charge_OIState",level);
          hcharge_OIState[trig][level-1]  = dbe_->book1D(name,title, 3, -1.5, 1.5);
          hcharge_OIState[trig][level-1]->setAxisTitle("Charge", 1);
          sprintf(name,"HLTMuonL%i_d0_OIState",level);
          sprintf(title,"L%i Muon from OIState radial impact vs (0,0)",level);
          hd0_OIState[trig][level-1] = dbe_->book1D(name,title, 50, -0.3, 0.3 );
          hd0_OIState[trig][level-1]->setAxisTitle("R Impact (cm) vs 0,0", 1);
          sprintf(name,"HLTMuonL%i_dz_OIState",level);
          sprintf(title,"L%i Muon from OIStateZ impact vs BeamSpot",level);
          hdz_OIState[trig][level-1] = dbe_->book1D(name,title, NBINS, -25., 25.);
          hdz_OIState[trig][level-1]->setAxisTitle("Z impact (cm) vs BeamSpot", 1);
          sprintf(name,"HLTMuonL%i_dr_OIState",level);
          sprintf(title,"L%i Muon from OIState radial impact vs BeamSpot",level);
          hdr_OIState[trig][level-1] = dbe_->book1D(name,title, 50, -0.3, 0.3 );
          hdr_OIState[trig][level-1]->setAxisTitle("R Impact (cm) vs BeamSpot", 1);

          sprintf(name,"HLTMuonL%i_trk_OIState_NMu",level);
          sprintf(title," L%i OIState tracker number of muons",level);
          hNMu_trk_OIState[trig][level-1] = dbe_->book1D(name,title, 10, -0.5, 10.5);
          hNMu_trk_OIState[trig][level-1]->setAxisTitle("Number of muons", 1);
          sprintf(name,"HLTMuonL%i_trk_OIState_pt",level);
          sprintf(title,"L%i OIState tracker Muon Pt",level);
          hpt_trk_OIState[trig][level-1] = dbe_->book1D(name,title, NBINS, 0., 200);
          hpt_trk_OIState[trig][level-1]->setAxisTitle("Pt", 1);
          sprintf(name,"HLTMuonL%i_trk_OIState_eta",level);
          sprintf(title,"L%i OIState tracker Muon #eta",level);
          heta_trk_OIState[trig][level-1] = dbe_->book1D(name,title, NBINS, -2.5, 2.5);
          heta_trk_OIState[trig][level-1]->setAxisTitle("#eta", 1);
          sprintf(name,"HLTMuonL%i_trk_OIState_phi",level);
          sprintf(title,"L%i OIState tracker Muon #phi",level);
          hphi_trk_OIState[trig][level-1] = dbe_->book1D(name,title, NBINS, -3.15, 3.15);
          hphi_trk_OIState[trig][level-1]->setAxisTitle("#phi", 1);
          sprintf(name,"HLTMuonL%i_trk_OIState_etaphi",level);
          sprintf(title,"L%i OIState tracker Muon #eta vs #phi",level);
          hetaphi_trk_OIState[trig][level-1] = dbe_->book2D(name,title, NBINS, -3.15, 3.15, NBINS, -2.5, 2.5);
          hetaphi_trk_OIState[trig][level-1]->setAxisTitle("#phi", 1);
          hetaphi_trk_OIState[trig][level-1]->setAxisTitle("#eta", 2);
          sprintf(name,"HLTMuonL%i_trk_OIState_ptphi",level);
          sprintf(title,"L%i OIState tracker Muon pt vs #phi",level);
          hptphi_trk_OIState[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., 200, NBINS, -3.15, 3.15);
          hptphi_trk_OIState[trig][level-1]->setAxisTitle("pt", 1);
          hptphi_trk_OIState[trig][level-1]->setAxisTitle("#phi", 2);
          sprintf(name,"HLTMuonL%i_trk_OIState_pteta",level);
          sprintf(title,"L%i OIState tracker Muon pt vs #eta",level);
          hpteta_trk_OIState[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., 200, NBINS, -2.5, 2.5);
          hpteta_trk_OIState[trig][level-1]->setAxisTitle("pt", 1);
          hpteta_trk_OIState[trig][level-1]->setAxisTitle("#eta", 2);
          sprintf(name,"HLTMuonL%i_trk_OIState_charge",level);
          sprintf(title,"L%i OIState tracker Muon Charge",level);
          hcharge_trk_OIState[trig][level-1]  = dbe_->book1D(name,title, 3, -1.5, 1.5);
          hcharge_trk_OIState[trig][level-1]->setAxisTitle("Charge", 1);
          sprintf(name,"HLTMuonL%i_trk_OIState_d0",level);
          sprintf(title,"L%i OIState tracker Muon from  radial impact vs (0,0)",level);
          hd0_trk_OIState[trig][level-1] = dbe_->book1D(name,title, 50, -0.3, 0.3 );
          hd0_trk_OIState[trig][level-1]->setAxisTitle("R Impact (cm) vs 0,0", 1);
          sprintf(name,"HLTMuonL%i_trk_OIState_dz",level);
          sprintf(title,"L%i OIState tracker Muon from Z impact vs BeamSpot",level);
          hdz_trk_OIState[trig][level-1] = dbe_->book1D(name,title, NBINS, -25., 25.);
          hdz_trk_OIState[trig][level-1]->setAxisTitle("Z impact (cm) vs BeamSpot", 1);
          sprintf(name,"HLTMuonL%i_trk_OIState_dr",level);
          sprintf(title,"L%i OIState tracker Muon radial impact vs BeamSpot",level);
          hdr_trk_OIState[trig][level-1] = dbe_->book1D(name,title, 50, -0.3, 0.3 );
          hdr_trk_OIState[trig][level-1]->setAxisTitle("R Impact (cm) vs BeamSpot", 1);

          dbe_->setCurrentFolder(monitorName_+dirname+"Level3/OIHit");

          sprintf(name,"HLTMuonL%i_NMu_OIHit",level);
          sprintf(title,"OIHit L%i number of muons",level);
          hNMu_OIHit[trig][level-1] = dbe_->book1D(name,title, 10, -0.5, 10.5);
          hNMu_OIHit[trig][level-1]->setAxisTitle("Number of muons", 1);

          sprintf(name,"HLTMuonL%i_pt_OIHit",level);
          sprintf(title,"L%i Pt_OIHit",level);
          hpt_OIHit[trig][level-1] = dbe_->book1D(name,title, NBINS, 0., 200);
          hpt_OIHit[trig][level-1]->setAxisTitle("Pt", 1);
          sprintf(name,"HLTMuonL%i_eta_OIHit",level);
          sprintf(title,"L%i Muon #eta_OIHit",level);
          heta_OIHit[trig][level-1] = dbe_->book1D(name,title, NBINS, -2.5, 2.5);
          heta_OIHit[trig][level-1]->setAxisTitle("#eta", 1);
          sprintf(name,"HLTMuonL%i_phi_OIHit",level);
          sprintf(title,"L%i Muon #phi_OIHit",level);
          hphi_OIHit[trig][level-1] = dbe_->book1D(name,title, NBINS, -3.15, 3.15);
          hphi_OIHit[trig][level-1]->setAxisTitle("#phi", 1);
          sprintf(name,"HLTMuonL%i_etaphi_OIHit",level);
          sprintf(title,"L%i Muon #eta vs #phi_OIHit",level);
          hetaphi_OIHit[trig][level-1] = dbe_->book2D(name,title, NBINS, -3.15, 3.15, NBINS, -2.5, 2.5);
          hetaphi_OIHit[trig][level-1]->setAxisTitle("#phi", 1);
          hetaphi_OIHit[trig][level-1]->setAxisTitle("#eta", 2);
          sprintf(name,"HLTMuonL%i_ptphi_OIHit",level);
          sprintf(title,"L%i Muon pt vs #phi_OIHit",level);
          hptphi_OIHit[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., 200, NBINS, -3.15, 3.15);
          hptphi_OIHit[trig][level-1]->setAxisTitle("pt", 1);
          hptphi_OIHit[trig][level-1]->setAxisTitle("#phi", 2);
          sprintf(name,"HLTMuonL%i_pteta_OIHit",level);
          sprintf(title,"L%i Muon pt vs #eta_OIHit",level);
          hpteta_OIHit[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., 200, NBINS, -2.5, 2.5);
          hpteta_OIHit[trig][level-1]->setAxisTitle("pt", 1);
          hpteta_OIHit[trig][level-1]->setAxisTitle("#eta", 2);
          sprintf(name,"HLTMuonL%i_charge_OIHit",level);
          sprintf(title,"L%i Muon Charge_OIHit",level);
          hcharge_OIHit[trig][level-1]  = dbe_->book1D(name,title, 3, -1.5, 1.5);
          hcharge_OIHit[trig][level-1]->setAxisTitle("Charge", 1);
 
          sprintf(name,"HLTMuonL%i_d0_OIHit",level);
          sprintf(title,"L%i Muon from OIHit radial impact vs (0,0)",level);
          hd0_OIHit[trig][level-1] = dbe_->book1D(name,title, 50, -0.3, 0.3 );
          hd0_OIHit[trig][level-1]->setAxisTitle("R Impact (cm) vs 0,0", 1);
          sprintf(name,"HLTMuonL%i_dz_OIHit",level);
          sprintf(title,"L%i Muon from OIHit Z impact vs BeamSpot",level);
          hdz_OIHit[trig][level-1] = dbe_->book1D(name,title, NBINS, -25., 25.);
          hdz_OIHit[trig][level-1]->setAxisTitle("Z impact (cm) vs BeamSpot", 1);
          sprintf(name,"HLTMuonL%i_dr_OIHit",level);
          sprintf(title,"L%i Muon from OIHit radial impact vs BeamSpot",level);
          hdr_OIHit[trig][level-1] = dbe_->book1D(name,title, 50, -0.3, 0.3 );
          hdr_OIHit[trig][level-1]->setAxisTitle("R Impact (cm) vs BeamSpot", 1);

          sprintf(name,"HLTMuonL%i_trk_OIHit_NMu",level);
          sprintf(title," L%i OIHit tracker number of muons",level);
          hNMu_trk_OIHit[trig][level-1] = dbe_->book1D(name,title, 10, -0.5, 10.5);
          hNMu_trk_OIHit[trig][level-1]->setAxisTitle("Number of muons", 1);
          sprintf(name,"HLTMuonL%i_trk_OIHit_pt",level);
          sprintf(title,"L%i OIHit tracker Muon Pt",level);
          hpt_trk_OIHit[trig][level-1] = dbe_->book1D(name,title, NBINS, 0., 200);
          hpt_trk_OIHit[trig][level-1]->setAxisTitle("Pt", 1);
          sprintf(name,"HLTMuonL%i_trk_OIHit_eta",level);
          sprintf(title,"L%i OIHit tracker Muon #eta",level);
          heta_trk_OIHit[trig][level-1] = dbe_->book1D(name,title, NBINS, -2.5, 2.5);
          heta_trk_OIHit[trig][level-1]->setAxisTitle("#eta", 1);
          sprintf(name,"HLTMuonL%i_trk_OIHit_phi",level);
          sprintf(title,"L%i OIHit tracker Muon #phi",level);
          hphi_trk_OIHit[trig][level-1] = dbe_->book1D(name,title, NBINS, -3.15, 3.15);
          hphi_trk_OIHit[trig][level-1]->setAxisTitle("#phi", 1);
          sprintf(name,"HLTMuonL%i_trk_OIHit_etaphi",level);
          sprintf(title,"L%i OIHit tracker Muon #eta vs #phi",level);
          hetaphi_trk_OIHit[trig][level-1] = dbe_->book2D(name,title, NBINS, -3.15, 3.15, NBINS, -2.5, 2.5);
          hetaphi_trk_OIHit[trig][level-1]->setAxisTitle("#phi", 1);
          hetaphi_trk_OIHit[trig][level-1]->setAxisTitle("#eta", 2);
          sprintf(name,"HLTMuonL%i_trk_OIHit_ptphi",level);
          sprintf(title,"L%i OIHit tracker Muon pt vs #phi",level);
          hptphi_trk_OIHit[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., 200, NBINS, -3.15, 3.15);
          hptphi_trk_OIHit[trig][level-1]->setAxisTitle("pt", 1);
          hptphi_trk_OIHit[trig][level-1]->setAxisTitle("#phi", 2);
          sprintf(name,"HLTMuonL%i_trk_OIHit_pteta",level);
          sprintf(title,"L%i OIHit tracker Muon pt vs #eta",level);
          hpteta_trk_OIHit[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., 200, NBINS, -2.5, 2.5);
          hpteta_trk_OIHit[trig][level-1]->setAxisTitle("pt", 1);
          hpteta_trk_OIHit[trig][level-1]->setAxisTitle("#eta", 2);
          sprintf(name,"HLTMuonL%i_trk_OIHit_charge",level);
          sprintf(title,"L%i OIHit tracker Muon Charge",level);
          hcharge_trk_OIHit[trig][level-1]  = dbe_->book1D(name,title, 3, -1.5, 1.5);
          hcharge_trk_OIHit[trig][level-1]->setAxisTitle("Charge", 1);
          sprintf(name,"HLTMuonL%i_trk_OIHit_d0",level);
          sprintf(title,"L%i OIHit tracker Muon from  radial impact vs (0,0)",level);
          hd0_trk_OIHit[trig][level-1] = dbe_->book1D(name,title, 50, -0.3, 0.3 );
          hd0_trk_OIHit[trig][level-1]->setAxisTitle("R Impact (cm) vs 0,0", 1);
          sprintf(name,"HLTMuonL%i_trk_OIHit_dz",level);
          sprintf(title,"L%i OIHit tracker Muon from Z impact vs BeamSpot",level);
          hdz_trk_OIHit[trig][level-1] = dbe_->book1D(name,title, NBINS, -25., 25.);
          hdz_trk_OIHit[trig][level-1]->setAxisTitle("Z impact (cm) vs BeamSpot", 1);
          sprintf(name,"HLTMuonL%i_trk_OIHit_dr",level);
          sprintf(title,"L%i OIHit tracker Muon radial impact vs BeamSpot",level);
          hdr_trk_OIHit[trig][level-1] = dbe_->book1D(name,title, 50, -0.3, 0.3 );
          hdr_trk_OIHit[trig][level-1]->setAxisTitle("R Impact (cm) vs BeamSpot", 1);

          dbe_->setCurrentFolder(monitorName_+dirname+"Level3/IOHit");

          sprintf(name,"HLTMuonL%i_NMu_IOHit",level);
          sprintf(title,"IOHit L%i number of muons",level);
          hNMu_IOHit[trig][level-1] = dbe_->book1D(name,title, 10, -0.5, 10.5);
          hNMu_IOHit[trig][level-1]->setAxisTitle("Number of muons", 1);

          sprintf(name,"HLTMuonL%i_pt_IOHit",level);
          sprintf(title,"L%i Pt_IOHit",level);
          hpt_IOHit[trig][level-1] = dbe_->book1D(name,title, NBINS, 0., 200);
          hpt_IOHit[trig][level-1]->setAxisTitle("Pt", 1);
          sprintf(name,"HLTMuonL%i_eta_IOHit",level);
          sprintf(title,"L%i Muon #eta_IOHit",level);
          heta_IOHit[trig][level-1] = dbe_->book1D(name,title, NBINS, -2.5, 2.5);
          heta_IOHit[trig][level-1]->setAxisTitle("#eta", 1);
          sprintf(name,"HLTMuonL%i_phi_IOHit",level);
          sprintf(title,"L%i Muon #phi_IOHit",level);
          hphi_IOHit[trig][level-1] = dbe_->book1D(name,title, NBINS, -3.15, 3.15);
          hphi_IOHit[trig][level-1]->setAxisTitle("#phi", 1);
          sprintf(name,"HLTMuonL%i_etaphi_IOHit",level);
          sprintf(title,"L%i Muon #eta vs #phi_IOHit",level);
          hetaphi_IOHit[trig][level-1] = dbe_->book2D(name,title, NBINS, -3.15, 3.15, NBINS, -2.5, 2.5);
          hetaphi_IOHit[trig][level-1]->setAxisTitle("#phi", 1);
          hetaphi_IOHit[trig][level-1]->setAxisTitle("#eta", 2);
          sprintf(name,"HLTMuonL%i_ptphi_IOHit",level);
          sprintf(title,"L%i Muon pt vs #phi_IOHit",level);
          hptphi_IOHit[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., 200, NBINS, -3.15, 3.15);
          hptphi_IOHit[trig][level-1]->setAxisTitle("pt", 1);
          hptphi_IOHit[trig][level-1]->setAxisTitle("#phi", 2);
          sprintf(name,"HLTMuonL%i_pteta_IOHit",level);
          sprintf(title,"L%i Muon pt vs #eta_IOHit",level);
          hpteta_IOHit[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., 200, NBINS, -2.5, 2.5);
          hpteta_IOHit[trig][level-1]->setAxisTitle("pt", 1);
          hpteta_IOHit[trig][level-1]->setAxisTitle("#eta", 2);
          sprintf(name,"HLTMuonL%i_charge_IOHit",level);
          sprintf(title,"L%i Muon Charge_IOHit",level);
          hcharge_IOHit[trig][level-1]  = dbe_->book1D(name,title, 3, -1.5, 1.5);
          hcharge_IOHit[trig][level-1]->setAxisTitle("Charge", 1);

          sprintf(name,"HLTMuonL%i_d0_IOHit",level);
          sprintf(title,"L%i Muon from IOHit radial impact vs (0,0)",level);
          hd0_IOHit[trig][level-1] = dbe_->book1D(name,title, 50, -0.3, 0.3 );
          hd0_IOHit[trig][level-1]->setAxisTitle("R Impact (cm) vs 0,0", 1);
          sprintf(name,"HLTMuonL%i_dz_IOHit",level);
          sprintf(title,"L%i Muon from IOHit Z impact vs BeamSpot",level);
          hdz_IOHit[trig][level-1] = dbe_->book1D(name,title, NBINS, -25., 25.);
          hdz_IOHit[trig][level-1]->setAxisTitle("Z impact (cm) vs BeamSpot", 1);
          sprintf(name,"HLTMuonL%i_dr_IOHit",level);
          sprintf(title,"L%i Muon from IOHit radial impact vs BeamSpot",level);
          hdr_IOHit[trig][level-1] = dbe_->book1D(name,title, 50, -0.3, 0.3 );
          hdr_IOHit[trig][level-1]->setAxisTitle("R Impact (cm) vs BeamSpot", 1);

          sprintf(name,"HLTMuonL%i_trk_IOHit_NMu",level);
          sprintf(title," L%i IOHit tracker number of muons",level);
          hNMu_trk_IOHit[trig][level-1] = dbe_->book1D(name,title, 10, -0.5, 10.5);
          hNMu_trk_IOHit[trig][level-1]->setAxisTitle("Number of muons", 1);
          sprintf(name,"HLTMuonL%i_trk_IOHit_pt",level);
          sprintf(title,"L%i IOHit tracker Muon Pt",level);
          hpt_trk_IOHit[trig][level-1] = dbe_->book1D(name,title, NBINS, 0., 200);
          hpt_trk_IOHit[trig][level-1]->setAxisTitle("Pt", 1);
          sprintf(name,"HLTMuonL%i_trk_IOHit_eta",level);
          sprintf(title,"L%i IOHit tracker Muon #eta",level);
          heta_trk_IOHit[trig][level-1] = dbe_->book1D(name,title, NBINS, -2.5, 2.5);
          heta_trk_IOHit[trig][level-1]->setAxisTitle("#eta", 1);
          sprintf(name,"HLTMuonL%i_trk_IOHit_phi",level);
          sprintf(title,"L%i IOHit tracker Muon #phi",level);
          hphi_trk_IOHit[trig][level-1] = dbe_->book1D(name,title, NBINS, -3.15, 3.15);
          hphi_trk_IOHit[trig][level-1]->setAxisTitle("#phi", 1);
          sprintf(name,"HLTMuonL%i_trk_IOHit_etaphi",level);
          sprintf(title,"L%i IOHit tracker Muon #eta vs #phi",level);
          hetaphi_trk_IOHit[trig][level-1] = dbe_->book2D(name,title, NBINS, -3.15, 3.15, NBINS, -2.5, 2.5);
          hetaphi_trk_IOHit[trig][level-1]->setAxisTitle("#phi", 1);
          hetaphi_trk_IOHit[trig][level-1]->setAxisTitle("#eta", 2);
          sprintf(name,"HLTMuonL%i_trk_IOHit_ptphi",level);
          sprintf(title,"L%i IOHit tracker Muon pt vs #phi",level);
          hptphi_trk_IOHit[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., 200, NBINS, -3.15, 3.15);
          hptphi_trk_IOHit[trig][level-1]->setAxisTitle("pt", 1);
          hptphi_trk_IOHit[trig][level-1]->setAxisTitle("#phi", 2);
          sprintf(name,"HLTMuonL%i_trk_IOHit_pteta",level);
          sprintf(title,"L%i IOHit tracker Muon pt vs #eta",level);
          hpteta_trk_IOHit[trig][level-1] = dbe_->book2D(name,title, NBINS, 0., 200, NBINS, -2.5, 2.5);
          hpteta_trk_IOHit[trig][level-1]->setAxisTitle("pt", 1);
          hpteta_trk_IOHit[trig][level-1]->setAxisTitle("#eta", 2);
          sprintf(name,"HLTMuonL%i_trk_IOHit_charge",level);
          sprintf(title,"L%i IOHit tracker Muon Charge",level);
          hcharge_trk_IOHit[trig][level-1]  = dbe_->book1D(name,title, 3, -1.5, 1.5);
          hcharge_trk_IOHit[trig][level-1]->setAxisTitle("Charge", 1);
          sprintf(name,"HLTMuonL%i_trk_IOHit_d0",level);
          sprintf(title,"L%i IOHit tracker Muon from  radial impact vs (0,0)",level);
          hd0_trk_IOHit[trig][level-1] = dbe_->book1D(name,title, 50, -0.3, 0.3 );
          hd0_trk_IOHit[trig][level-1]->setAxisTitle("R Impact (cm) vs 0,0", 1);
          sprintf(name,"HLTMuonL%i_trk_IOHit_dz",level);
          sprintf(title,"L%i IOHit tracker Muon from Z impact vs BeamSpot",level);
          hdz_trk_IOHit[trig][level-1] = dbe_->book1D(name,title, NBINS, -25., 25.);
          hdz_trk_IOHit[trig][level-1]->setAxisTitle("Z impact (cm) vs BeamSpot", 1);
          sprintf(name,"HLTMuonL%i_trk_IOHit_dr",level);
          sprintf(title,"L%i IOHit tracker Muon radial impact vs BeamSpot",level);
          hdr_trk_IOHit[trig][level-1] = dbe_->book1D(name,title, 50, -0.3, 0.3 );
          hdr_trk_IOHit[trig][level-1]->setAxisTitle("R Impact (cm) vs BeamSpot", 1);

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
void HLTMuonDQMSource::beginRun(const edm::Run& run, const edm::EventSetup& eventSetup) {

}

//--------------------------------------------------------
void HLTMuonDQMSource::beginLuminosityBlock(const edm::LuminosityBlock& leventSetupk, 
    const edm::EventSetup& eventSetup) {

}

// ----------------------------------------------------------
void HLTMuonDQMSource::analyze(const edm::Event& event, 
    const edm::EventSetup& eventSetup )
{  
  if ( !dbe_) return;
  counterEvt_++;
  if (prescaleEvt_ > 0 && counterEvt_%prescaleEvt_!=0) return;
  LogDebug("HLTMuonDQMSource") << " processing conterEvt_: " << counterEvt_ <<std::endl;

  bool trigFired = false;
  bool FiredTriggers[NTRIG] = {false};
  edm::Handle<edm::TriggerResults> trigResult;
  event.getByLabel(TrigResultInput, trigResult);
  if( !trigResult.failedToGet() ) {
    int ntrigs = trigResult->size();
    const edm::TriggerNames & trigName = event.triggerNames(*trigResult);
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
  eventSetup.get<IdealMagneticFieldRecord>().get(magField);

  //get the geometry
  edm::ESHandle<GlobalTrackingGeometry> glbTrackingGeometry;
  eventSetup.get<GlobalTrackingGeometryRecord>().get(glbTrackingGeometry);

  edm::Handle<reco::RecoChargedCandidateCollection> l2mucands, l3mucands;
  edm::Handle<L2MuonTrajectorySeedCollection> l2seeds; 
  edm::Handle<L3MuonTrajectorySeedCollection> l3seeds; 
  reco::RecoChargedCandidateCollection::const_iterator cand, cand2, cand3;

  event.getByLabel (l2seedscollectionTag_,l2seeds);
  event.getByLabel (l3seedscollectionTag_,l3seeds);
  event.getByLabel (l2collectionTag_,l2mucands);
  event.getByLabel (l3collectionTag_,l3mucands);

  edm::Handle<L3MuonTrajectorySeedCollection> l3seedsOIState; 
  edm::Handle<L3MuonTrajectorySeedCollection> l3seedsOIHit; 
  edm::Handle<L3MuonTrajectorySeedCollection> l3seedsIOHit; 
  event.getByLabel (l3seedscollectionTagOIState_,l3seedsOIState);
  event.getByLabel (l3seedscollectionTagOIHit_,l3seedsOIHit);
  event.getByLabel (l3seedscollectionTagIOHit_,l3seedsIOHit);

  edm::Handle<std::vector<TrackCandidate> > l3trkCandsOIState;
  edm::Handle<std::vector<TrackCandidate> > l3trkCandsOIHit;
  edm::Handle<std::vector<TrackCandidate> > l3trkCandsIOHit;
  event.getByLabel (l3trkfindingOIState_,l3trkCandsOIState);
  event.getByLabel (l3trkfindingOIHit_,l3trkCandsOIHit);
  event.getByLabel (l3trkfindingIOHit_,l3trkCandsIOHit);

  edm::Handle<std::vector<reco::Track> > l3trkOIState;
  edm::Handle<std::vector<reco::Track> > l3trkOIHit;
  edm::Handle<std::vector<reco::Track> > l3trkIOHit;
  edm::Handle<std::vector<reco::Track> > l3tktrk;
  event.getByLabel (l3trkOIState_,l3trkOIState);
  event.getByLabel (l3trkOIHit_,l3trkOIHit);
  event.getByLabel (l3trkIOHit_,l3trkIOHit);
  event.getByLabel (l3tktrk_, l3tktrk);

// 
  edm::Handle<std::vector<reco::Track> > l3muonOIState;
  edm::Handle<std::vector<reco::Track> > l3muonOIHit;
  edm::Handle<std::vector<reco::Track> > l3muonIOHit;
  edm::Handle<std::vector<reco::Track> > l3muon;
  event.getByLabel (l3muonsOIState_,l3muonOIState);
  event.getByLabel (l3muonsOIHit_,l3muonOIHit);
  event.getByLabel (l3muonsIOHit_,l3muonIOHit);
  event.getByLabel (l3muons_, l3muon);

//  
  if( !l2seeds.failedToGet() && l2seeds.isValid() ) {
    event.getByLabel (l2collectionTag_,l2mucands);
    if( !l2mucands.failedToGet() && l2mucands->size() != 0 ) {
      event.getByLabel (l3seedscollectionTag_,l3seeds);
      if( !l3seeds.failedToGet() && l3seeds.isValid() ) event.getByLabel (l3collectionTag_,l3mucands);
    }
  }
  
  for( int ntrig = 0; ntrig < nTrigs; ntrig++ ) {
    if( !FiredTriggers[ntrig] ) continue;
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

    if (!l3seedsOIState.failedToGet() && l3seedsOIState.isValid() ) {
      hNMu_OIState[ntrig][4]->Fill(l3seedsOIState->size());

      L3MuonTrajectorySeedCollection::const_iterator l3seedOIState;
      for(l3seedOIState=l3seedsOIState->begin(); l3seedOIState != l3seedsOIState->end(); ++l3seedOIState){
        PTrajectoryStateOnDet state=l3seedOIState->startingState();
        TrajectoryStateTransform tsTransform;
        DetId seedDetId(state.detId());
        const GeomDet* gdet = glbTrackingGeometry->idToDet( seedDetId );
        TrajectoryStateOnSurface tsos = tsTransform.transientState(state, &(gdet->surface()), magField.product());

        float pt = tsos.globalMomentum().perp();
        float eta = tsos.globalPosition().eta();
        float phi = tsos.globalPosition().phi();

        hNMu_l3seed_comp[ntrig][4]->Fill(1);
        hpt_OIState[ntrig][4]->Fill(pt);
        hphi_OIState[ntrig][4]->Fill(phi);
        heta_OIState[ntrig][4]->Fill(eta);
        hetaphi_OIState[ntrig][4]->Fill(phi,eta);
        hptphi_OIState[ntrig][4]->Fill(pt,phi);
        hpteta_OIState[ntrig][4]->Fill(pt,eta);
        hcharge_OIState[ntrig][4]->Fill(tsos.charge());

        //    reco::TrackRef tk = cand->get<reco::TrackRef>(); 
        //    hpt_OIState[ntrig][2]->Fill(tk->pt());

        // OIState track candidate
        //reco::TrackRef l2tkRef = l3seedOIState->l2Track();
      }
    }

    if ( !l3seedsOIHit.failedToGet() && l3seedsOIHit.isValid() && l3seedsOIHit->size() != 0 ) {
      hNMu_OIHit[ntrig][4]->Fill(l3seedsOIHit->size());

      L3MuonTrajectorySeedCollection::const_iterator l3seedOIHit;
      for(l3seedOIHit=l3seedsOIHit->begin(); l3seedOIHit != l3seedsOIHit->end(); ++l3seedOIHit){
        PTrajectoryStateOnDet state=l3seedOIHit->startingState();
        TrajectoryStateTransform tsTransform;
        DetId seedDetId(state.detId());
        const GeomDet* gdet = glbTrackingGeometry->idToDet( seedDetId );
        TrajectoryStateOnSurface tsos = tsTransform.transientState(state, &(gdet->surface()), magField.product());

        float pt = tsos.globalMomentum().perp();
        float eta = tsos.globalPosition().eta();
        float phi = tsos.globalPosition().phi();

        hNMu_l3seed_comp[ntrig][4]->Fill(2);
        hpt_OIHit[ntrig][4]->Fill(pt);
        hphi_OIHit[ntrig][4]->Fill(phi);
        heta_OIHit[ntrig][4]->Fill(eta);
        hetaphi_OIHit[ntrig][4]->Fill(phi,eta);
        hptphi_OIHit[ntrig][4]->Fill(pt,phi);
        hpteta_OIHit[ntrig][4]->Fill(pt,eta);
        hcharge_OIHit[ntrig][4]->Fill(tsos.charge());

        //reco::TrackRef l2tkRef = l3seedOIHit->l2Track();
//        std::cout << "== l3 cascade->l2tk muon pt : " << l2tkRef->pt() << "  eta : " << l2tkRef->eta() << "  phi : " << l2tkRef->phi() << std::endl;
      }
    }

    if ( !l3seedsIOHit.failedToGet() && l3seedsIOHit.isValid() && l3seedsIOHit->size() != 0 ) {
      hNMu_IOHit[ntrig][4]->Fill(l3seedsIOHit->size());


      L3MuonTrajectorySeedCollection::const_iterator l3seedIOHit;
      for(l3seedIOHit=l3seedsIOHit->begin(); l3seedIOHit != l3seedsIOHit->end(); ++l3seedIOHit){
        PTrajectoryStateOnDet state=l3seedIOHit->startingState();
        TrajectoryStateTransform tsTransform;
        DetId seedDetId(state.detId());
        const GeomDet* gdet = glbTrackingGeometry->idToDet( seedDetId );
        TrajectoryStateOnSurface tsos = tsTransform.transientState(state, &(gdet->surface()), magField.product());

        float pt = tsos.globalMomentum().perp();
        float eta = tsos.globalPosition().eta();
        float phi = tsos.globalPosition().phi();


        hNMu_l3seed_comp[ntrig][4]->Fill(3);
        hpt_IOHit[ntrig][4]->Fill(pt);
        hphi_IOHit[ntrig][4]->Fill(phi);
        heta_IOHit[ntrig][4]->Fill(eta);
        hetaphi_IOHit[ntrig][4]->Fill(phi,eta);
        hptphi_IOHit[ntrig][4]->Fill(pt,phi);
        hpteta_IOHit[ntrig][4]->Fill(pt,eta);
        hcharge_IOHit[ntrig][4]->Fill(tsos.charge());

        //reco::TrackRef l2tkRef = l3seedIOHit->l2Track();
      }
    } 


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


        hNMu_l3seed_comp[ntrig][4]->Fill(0);
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


    reco::BeamSpot beamSpot;
    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    event.getByLabel("hltOfflineBeamSpot",recoBeamSpotHandle);
    if (!recoBeamSpotHandle.failedToGet())  beamSpot = *recoBeamSpotHandle;

    if (!l2mucands.failedToGet() && l2mucands.isValid() ) {
      LogDebug("HLTMuonDQMSource") << " filling L2 stuff " << std::endl;
      edm::Handle<reco::IsoDepositMap> l2depMap;


      if( l2mucands->size() != 0 ) event.getByLabel (l2isolationTag_,l2depMap);
      hNMu[ntrig][1]->Fill(l2mucands->size());
      for (cand=l2mucands->begin(); cand!=l2mucands->end(); ++cand) {
        reco::TrackRef tk = cand->get<reco::TrackRef>();
        if (!l2depMap.failedToGet() && l2depMap.isValid() ) {
          LogDebug("HLTMuonDQMSource") << " filling L2 Iso stuff " << std::endl;
          if ( tk.isNonnull() &&l2depMap->contains(tk.id()) ){
            reco::IsoDepositMap::value_type calDeposit = (*l2depMap)[tk];
            double dephlt = calDeposit.depositWithin(coneSize_);
            if( dephlt != 0 ) hisoL2[ntrig]->Fill(dephlt);
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
            hdz[ntrig][0]->Fill(tk->dz(beamSpot.position())); 
            hdzeta[ntrig][0]->Fill(tk->eta(),tk->dz(beamSpot.position()));
          } 
          hd0phi[ntrig][0]->Fill(tk->phi(),tk->d0()); 
          hdz0[ntrig][0]->Fill(tk->dz()); 
          hdz0eta[ntrig][0]->Fill(tk->eta(),tk->dz());
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

      if( l3mucands->size() != 0 ) event.getByLabel (l3isolationTag_,l3depMap);
      for (cand=l3mucands->begin(); cand!=l3mucands->end(); ++cand) {
        reco::TrackRef tk = cand->get<reco::TrackRef>();
        if (!l3depMap.failedToGet() && l3depMap.isValid() ) {
          if ( tk.isNonnull() && l3depMap->contains(tk.id()) ){
            reco::IsoDepositMap::value_type calDeposit= (*l3depMap)[tk];
            double dephlt = calDeposit.depositWithin(coneSize_);
            if( dephlt != 0 ) hisoL3[ntrig]->Fill(dephlt);
            double depL3[10];
            for( int idep = 0; idep < 10; idep++ ) {
              double conesize = ( idep + 1 ) * coneSize_/10.0;
              depL3[idep] = calDeposit.depositWithin(conesize);
              if( depL3[idep] != 0 ) hL3muonIsoDR[ntrig]->Fill(conesize,depL3[idep]);
            }
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
        hnPixelValidHits[ntrig]->Fill(hitp.numberOfValidPixelHits()); 
        hnStripValidHits[ntrig]->Fill(hitp.numberOfValidStripHits()); 
        hnMuValidHits[ntrig]->Fill(hitp.numberOfValidMuonHits()); 
        hd0[ntrig][1]->Fill(tk->d0()); 
        if (!recoBeamSpotHandle.failedToGet()) {
          hdr[ntrig][1]->Fill(tk->dxy(beamSpot.position()));
          hdrphi[ntrig][1]->Fill(tk->phi(),tk->dxy(beamSpot.position())); 
          hdz[ntrig][1]->Fill(tk->dz(beamSpot.position())); 
          hdzeta[ntrig][1]->Fill(tk->eta(),tk->dz(beamSpot.position()));
        }
        hd0phi[ntrig][1]->Fill(tk->phi(),tk->d0()); 
        hdz0[ntrig][1]->Fill(tk->dz()); 
        hdz0eta[ntrig][1]->Fill(tk->eta(),tk->dz());
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
    // l3 matching

//    std::vector<reco::Track>::const_iterator l3MatchedTrk = l3tktrk->end();
    double bpt = -999.;
    double beta = -999.;
    double bphi = -999.;
    if( !l3mucands.failedToGet() && l3mucands.isValid() ) {
      for(reco::RecoChargedCandidateCollection::const_iterator i = l3mucands->begin(); i != l3mucands->end(); i++ ) {
        double dR_min_tk=1e9;
        if( !l3tktrk.failedToGet() && l3tktrk.isValid() ) {
          for( std::vector<reco::Track>::const_iterator itl3trk = l3tktrk->begin(); itl3trk !=l3tktrk->end(); itl3trk++ ) {
            double dR = deltaR(i->eta(), i->phi(),itl3trk->eta(),itl3trk->phi());
            if( dR < dR_min_tk ) {
              dR_min_tk=dR ;
              bpt = itl3trk->pt();
              beta = itl3trk->eta();
              bphi = itl3trk->phi();
//              l3MatchedTrk = itl3trk;
            }
          }
        }
        if( bpt !=-999 && beta != -999 && bphi != -999  ) {
          hptres_L3L3trk[ntrig][2]->Fill(1/i->pt() - 1/bpt);
          hetares_L3L3trk[ntrig][2]->Fill(i->eta() - beta);
          hphires_L3L3trk[ntrig][2]->Fill(i->phi() - bphi);
          hptrelres_L3L3trk[ntrig][2]->Fill( (1/i->pt() - 1/bpt)/(1/i->pt()) );
          hetarelres_L3L3trk[ntrig][2]->Fill( (i->eta() - beta)/i->eta() );
          hphirelres_L3L3trk[ntrig][2]->Fill( (i->phi() - bphi)/i->phi() );
        }
      }
    }
    std::vector<reco::Track>::const_iterator itl3mu;
    if( !l3muon.failedToGet() && l3muon.isValid() ) {
      for( itl3mu = l3muon->begin(); itl3mu != l3muon->end(); itl3mu++ ) {
        hNMu_comp[ntrig][2]->Fill(0);
      }
    }

    std::vector<reco::Track>::const_iterator itl3muOIState;
    if( !l3muonOIState.failedToGet() && l3muonOIState.isValid() && l3muonOIState->size() > 0 ) {
      hNMu_OIState[ntrig][2]->Fill(l3muonOIState->size());
      for( itl3muOIState = l3muonOIState->begin(); itl3muOIState != l3muonOIState->end(); itl3muOIState++ ) {
        hNMu_comp[ntrig][2]->Fill(1);
        hcharge_OIState[ntrig][2]->Fill(itl3muOIState->charge());
        hpt_OIState[ntrig][2]->Fill(itl3muOIState->pt());
        hphi_OIState[ntrig][2]->Fill(itl3muOIState->phi());
        heta_OIState[ntrig][2]->Fill(itl3muOIState->eta());
        hetaphi_OIState[ntrig][2]->Fill(itl3muOIState->phi(),itl3muOIState->eta());
        hptphi_OIState[ntrig][2]->Fill(itl3muOIState->pt(),itl3muOIState->phi());
        hpteta_OIState[ntrig][2]->Fill(itl3muOIState->pt(),itl3muOIState->eta());

        hd0_OIState[ntrig][2]->Fill(itl3muOIState->d0()); 
        if( !recoBeamSpotHandle.failedToGet()) {
          hdz_OIState[ntrig][2]->Fill(itl3muOIState->dz(beamSpot.position()));
          hdr_OIState[ntrig][2]->Fill(itl3muOIState->dxy(beamSpot.position()));
        }
       //hpt_l3muOIState[ntrig][2]->Fill(itl3muOIState->pt());
      }
    } 
    std::vector<reco::Track>::const_iterator itl3muOIHit;
    if( !l3muonOIHit.failedToGet() && l3muonOIHit.isValid() && l3muonOIHit->size() > 0 ) {
      hNMu_OIHit[ntrig][2]->Fill(l3muonOIHit->size());
      for( itl3muOIHit = l3muonOIHit->begin(); itl3muOIHit != l3muonOIHit->end(); itl3muOIHit++ ) {
        hNMu_comp[ntrig][2]->Fill(2);
        hcharge_OIHit[ntrig][2]->Fill(itl3muOIHit->charge());
        hpt_OIHit[ntrig][2]->Fill(itl3muOIHit->pt());
        hphi_OIHit[ntrig][2]->Fill(itl3muOIHit->phi());
        heta_OIHit[ntrig][2]->Fill(itl3muOIHit->eta());
        hetaphi_OIHit[ntrig][2]->Fill(itl3muOIHit->phi(),itl3muOIHit->eta());
        hptphi_OIHit[ntrig][2]->Fill(itl3muOIHit->pt(),itl3muOIHit->phi());
        hpteta_OIHit[ntrig][2]->Fill(itl3muOIHit->pt(),itl3muOIHit->eta());

        hd0_OIHit[ntrig][2]->Fill(itl3muOIHit->d0()); 
        if( !recoBeamSpotHandle.failedToGet()) {
          hdz_OIHit[ntrig][2]->Fill(itl3muOIHit->dz(beamSpot.position()));
          hdr_OIHit[ntrig][2]->Fill(itl3muOIHit->dxy(beamSpot.position()));
        }
       //hpt_l3muOIHit[ntrig][2]->Fill(itl3muOIHit->pt());
      }
    } 
    std::vector<reco::Track>::const_iterator itl3muIOHit;
    if( !l3muonIOHit.failedToGet() && l3muonIOHit.isValid() && l3muonIOHit->size() > 0 ) {
      hNMu_IOHit[ntrig][2]->Fill(l3muonIOHit->size());
      for( itl3muIOHit = l3muonIOHit->begin(); itl3muIOHit != l3muonIOHit->end(); itl3muIOHit++ ) {
        hNMu_comp[ntrig][2]->Fill(3);
        hcharge_IOHit[ntrig][2]->Fill(itl3muIOHit->charge());
        hpt_IOHit[ntrig][2]->Fill(itl3muIOHit->pt());
        hphi_IOHit[ntrig][2]->Fill(itl3muIOHit->phi());
        heta_IOHit[ntrig][2]->Fill(itl3muIOHit->eta());
        hetaphi_IOHit[ntrig][2]->Fill(itl3muIOHit->phi(),itl3muIOHit->eta());
        hptphi_IOHit[ntrig][2]->Fill(itl3muIOHit->pt(),itl3muIOHit->phi());
        hpteta_IOHit[ntrig][2]->Fill(itl3muIOHit->pt(),itl3muIOHit->eta());

        hd0_IOHit[ntrig][2]->Fill(itl3muIOHit->d0()); 
        if( !recoBeamSpotHandle.failedToGet()) {
          hdz_IOHit[ntrig][2]->Fill(itl3muIOHit->dz(beamSpot.position()));
          hdr_IOHit[ntrig][2]->Fill(itl3muIOHit->dxy(beamSpot.position()));
        }
      //hpt_l3muIOHit[ntrig][2]->Fill(itl3muIOHit->pt());
      }
    } 
    // l3 muon tracker track
    std::vector<reco::Track>::const_iterator itl3tktrk;
    if( l3tktrk.failedToGet() && l3tktrk.isValid() ) {
      hNMu_trk[ntrig][2]->Fill(l3tktrk->size());
      for( itl3tktrk = l3tktrk->begin(); itl3tktrk != l3tktrk->end(); itl3tktrk++ ) {
        hNMu_trk_comp[ntrig][2]->Fill(0);
        hcharge_trk[ntrig][2]->Fill(itl3tktrk->charge());
        hpt_trk[ntrig][2]->Fill(itl3tktrk->pt());
        hphi_trk[ntrig][2]->Fill(itl3tktrk->phi());
        heta_trk[ntrig][2]->Fill(itl3tktrk->eta());
        hetaphi_trk[ntrig][2]->Fill(itl3tktrk->phi(),itl3tktrk->eta());
        hptphi_trk[ntrig][2]->Fill(itl3tktrk->pt(),itl3tktrk->phi());
        hpteta_trk[ntrig][2]->Fill(itl3tktrk->pt(),itl3tktrk->eta());

        //hd0_trk[ntrig][2]->Fill(tk->d0()); 
        hd0_trk[ntrig][2]->Fill(itl3tktrk->d0()); 
        if( !recoBeamSpotHandle.failedToGet()) {
          hdz_trk[ntrig][2]->Fill(itl3tktrk->dz(beamSpot.position()));
          hdr_trk[ntrig][2]->Fill(itl3tktrk->dxy(beamSpot.position()));
        }
      }
    }
    // l3 OIState tracker info.
    std::vector<reco::Track>::const_iterator itl3trkOIState;
    if( !l3trkOIState.failedToGet() && l3trkOIState.isValid() && l3trkOIState->size() > 0 ) {
      hNMu_trk_OIState[ntrig][2]->Fill(l3trkOIState->size());
      for( itl3trkOIState = l3trkOIState->begin(); itl3trkOIState != l3trkOIState->end(); itl3trkOIState++ ) {

        hNMu_trk_comp[ntrig][2]->Fill(1);
        hcharge_trk_OIState[ntrig][2]->Fill(itl3trkOIState->charge());
        hpt_trk_OIState[ntrig][2]->Fill(itl3trkOIState->pt());
        hphi_trk_OIState[ntrig][2]->Fill(itl3trkOIState->phi());
        heta_trk_OIState[ntrig][2]->Fill(itl3trkOIState->eta());
        hetaphi_trk_OIState[ntrig][2]->Fill(itl3trkOIState->phi(),itl3trkOIState->eta());
        hptphi_trk_OIState[ntrig][2]->Fill(itl3trkOIState->pt(),itl3trkOIState->phi());
        hpteta_trk_OIState[ntrig][2]->Fill(itl3trkOIState->pt(),itl3trkOIState->eta());

        hd0_trk_OIState[ntrig][2]->Fill(itl3trkOIState->d0()); 
        if( !recoBeamSpotHandle.failedToGet()) {
          hdz_trk_OIState[ntrig][2]->Fill(itl3trkOIState->dz(beamSpot.position()));
          hdr_trk_OIState[ntrig][2]->Fill(itl3trkOIState->dxy(beamSpot.position()));
        }
      }
    }
    // l3 OIHit tracker info.
    std::vector<reco::Track>::const_iterator itl3trkOIHit;
    if( !l3trkOIHit.failedToGet() && l3trkOIHit.isValid() && l3trkOIHit->size() >0 ) {
      hNMu_trk_OIHit[ntrig][2]->Fill(l3trkOIHit->size());
      for( itl3trkOIHit = l3trkOIHit->begin(); itl3trkOIHit != l3trkOIHit->end(); itl3trkOIHit++ ) {

        hNMu_trk_comp[ntrig][2]->Fill(2);
        hcharge_trk_OIHit[ntrig][2]->Fill(itl3trkOIHit->charge());
        hpt_trk_OIHit[ntrig][2]->Fill(itl3trkOIHit->pt());
        hphi_trk_OIHit[ntrig][2]->Fill(itl3trkOIHit->phi());
        heta_trk_OIHit[ntrig][2]->Fill(itl3trkOIHit->eta());
        hetaphi_trk_OIHit[ntrig][2]->Fill(itl3trkOIHit->phi(),itl3trkOIHit->eta());
        hptphi_trk_OIHit[ntrig][2]->Fill(itl3trkOIHit->pt(),itl3trkOIHit->phi());
        hpteta_trk_OIHit[ntrig][2]->Fill(itl3trkOIHit->pt(),itl3trkOIHit->eta());

        hd0_trk_OIHit[ntrig][2]->Fill(itl3trkOIHit->d0()); 
        if( !recoBeamSpotHandle.failedToGet()) {
          hdz_trk_OIHit[ntrig][2]->Fill(itl3trkOIHit->dz(beamSpot.position()));
          hdr_trk_OIHit[ntrig][2]->Fill(itl3trkOIHit->dxy(beamSpot.position()));
        }
      }
    }
    // l3 IOHit tracker info.
    std::vector<reco::Track>::const_iterator itl3trkIOHit;
    if( !l3trkIOHit.failedToGet() && l3trkIOHit.isValid() && l3trkIOHit->size() > 0 ) {
      hNMu_trk_IOHit[ntrig][2]->Fill(l3trkIOHit->size());
      for( itl3trkIOHit = l3trkIOHit->begin(); itl3trkIOHit != l3trkIOHit->end(); itl3trkIOHit++ ) {

        hNMu_trk_comp[ntrig][2]->Fill(3);
        hcharge_trk_IOHit[ntrig][2]->Fill(itl3trkIOHit->charge());
        hpt_trk_IOHit[ntrig][2]->Fill(itl3trkIOHit->pt());
        hphi_trk_IOHit[ntrig][2]->Fill(itl3trkIOHit->phi());
        heta_trk_IOHit[ntrig][2]->Fill(itl3trkIOHit->eta());
        hetaphi_trk_IOHit[ntrig][2]->Fill(itl3trkIOHit->phi(),itl3trkIOHit->eta());
        hptphi_trk_IOHit[ntrig][2]->Fill(itl3trkIOHit->pt(),itl3trkIOHit->phi());
        hpteta_trk_IOHit[ntrig][2]->Fill(itl3trkIOHit->pt(),itl3trkIOHit->eta());

        hd0_trk_IOHit[ntrig][2]->Fill(itl3trkIOHit->d0()); 
        if( !recoBeamSpotHandle.failedToGet()) {
          hdr_trk_IOHit[ntrig][2]->Fill(itl3trkIOHit->dxy(beamSpot.position()));
          hdz_trk_IOHit[ntrig][2]->Fill(itl3trkIOHit->dz(beamSpot.position()));
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
    event.getByLabel(edm::InputTag("hltTowerMakerForMuons"), caloTower);
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
void HLTMuonDQMSource::endLuminosityBlock(const edm::LuminosityBlock& lumiBlock, 
    const edm::EventSetup& eventSetup) {
}
//--------------------------------------------------------
void HLTMuonDQMSource::endRun(const edm::Run& run, const edm::EventSetup& eventSetup){
}
//--------------------------------------------------------
void HLTMuonDQMSource::endJob(){
  edm::LogInfo("HLTMonMuon") << "analyzed " << counterEvt_ << " events";
  //std::cout << "analyzed = " << counterEvt_ << " , triggered = " << nTrig_ << std::endl;


  //if (outputFile_.size() != 0 && dbe_)
  //dbe_->save(outputFile_);

  return;
}
