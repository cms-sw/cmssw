#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PFGeometry.h"

#include "RecoParticleFlow/PFRootEvent/interface/PFRootEventManager.h"

#include "RecoParticleFlow/PFRootEvent/interface/IO.h"

#include "RecoParticleFlow/PFRootEvent/interface/PFJetAlgorithm.h" 
#include "RecoJets/JetAlgorithms/interface/JetMaker.h"

#include "RecoParticleFlow/PFRootEvent/interface/Utils.h" 
#include "RecoParticleFlow/PFRootEvent/interface/EventColin.h" 
#include "RecoParticleFlow/PFRootEvent/interface/METManager.h"

#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibrationHF.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFClusterCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"

#include "FWCore/ServiceRegistry/interface/Service.h" 
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/FWLite/interface/AutoLibraryLoader.h"

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TCutG.h"
#include "TVector3.h"
#include "TROOT.h"

#include <iostream>
#include <vector>
#include <stdlib.h>

using namespace std;
using namespace boost;
using namespace reco;

PFRootEventManager::PFRootEventManager() {}



PFRootEventManager::PFRootEventManager(const char* file)
  : 
  iEvent_(0),
  options_(0),
  tree_(0),
  outTree_(0),
  outEvent_(0),
  //   clusters_(new reco::PFClusterCollection),
  clustersECAL_(new reco::PFClusterCollection),
  clustersHCAL_(new reco::PFClusterCollection),
  clustersHFEM_(new reco::PFClusterCollection),
  clustersHFHAD_(new reco::PFClusterCollection),
  clustersPS_(new reco::PFClusterCollection),
  pfBlocks_(new reco::PFBlockCollection),
  pfCandidates_(new reco::PFCandidateCollection),
  //pfJets_(new reco::PFJetCollection),
  outFile_(0),
  calibFile_(0)
{
  
  
  //   iEvent_=0;
  h_deltaETvisible_MCEHT_ 
    = new TH1F("h_deltaETvisible_MCEHT","Jet Et difference CaloTowers-MC"
               ,1000,-50.,50.);
  h_deltaETvisible_MCPF_  
    = new TH1F("h_deltaETvisible_MCPF" ,"Jet Et difference ParticleFlow-MC"
               ,1000,-50.,50.);

  readOptions(file, true, true);
 
       
  //   maxERecHitEcal_ = -1;
  //   maxERecHitHcal_ = -1;

}

void PFRootEventManager::reset() { 

  if(outEvent_) {
    outEvent_->reset();
    outTree_->GetBranch("event")->SetAddress(&outEvent_);
  } 
  
  
 
}

void PFRootEventManager::readOptions(const char* file, 
                                     bool refresh, 
                                     bool reconnect) {
                                     
  readSpecificOptions(file);
  
  cout<<"calling PFRootEventManager::readOptions"<<endl;
   

  reset();
  
  PFGeometry pfGeometry; // initialize geometry
  
  // cout<<"reading options "<<endl;

  try {
    if( !options_ )
      options_ = new IO(file);
    else if( refresh) {
      delete options_;
      options_ = new IO(file);
    }
  }
  catch( const string& err ) {
    cout<<err<<endl;
    return;
  }


  debug_ = false; 
  options_->GetOpt("rootevent", "debug", debug_);

  
  // output text file for calibration
  string calibfilename;
  options_->GetOpt("calib","outfile",calibfilename);
  if (!calibfilename.empty()) { 
    calibFile_ = new std::ofstream(calibfilename.c_str());
    std::cout << "Calib file name " << calibfilename << " " << calibfilename.c_str() << std::endl;
  }

  // output root file   ------------------------------------------

  
  if(!outFile_) {
    string outfilename;
    options_->GetOpt("root","outfile", outfilename);
    if(!outfilename.empty() ) {
      outFile_ = TFile::Open(outfilename.c_str(), "recreate");
      
      bool doOutTree = false;
      options_->GetOpt("root","outtree", doOutTree);
      if(doOutTree) {
        outFile_->cd();
        // cout<<"do tree"<<endl;
        outEvent_ = new EventColin();
        outTree_ = new TTree("Eff","");
        outTree_->Branch("event","EventColin", &outEvent_,32000,2);
      }
      // cout<<"don't do tree"<<endl;
    }
  }
// PFJet benchmark options and output jetfile to be open before input file!!!--

  doPFJetBenchmark_ = false;
  options_->GetOpt("pfjet_benchmark", "on/off", doPFJetBenchmark_);
  
  if (doPFJetBenchmark_) {
    string outjetfilename;
    options_->GetOpt("pfjet_benchmark", "outjetfile", outjetfilename);
        
    bool pfjBenchmarkDebug;
    options_->GetOpt("pfjet_benchmark", "debug", pfjBenchmarkDebug);
    
    bool plotAgainstReco=0;
    options_->GetOpt("pfjet_benchmark", "plotAgainstReco", plotAgainstReco);
    
    bool onlyTwoJets=1;
    options_->GetOpt("pfjet_benchmark", "onlyTwoJets", onlyTwoJets);
    
    double deltaRMax=0.1;
    options_->GetOpt("pfjet_benchmark", "deltaRMax", deltaRMax);

    fastsim_=true;
    options_->GetOpt("Simulation","Fast",fastsim_);
 
    PFJetBenchmark_.setup( outjetfilename, 
                           pfjBenchmarkDebug,
                           plotAgainstReco,
			   onlyTwoJets,
                           deltaRMax );
  }

// PFMET benchmark options and output jetfile to be open before input file!!!--

  doPFMETBenchmark_ = false;
  options_->GetOpt("pfmet_benchmark", "on/off", doPFMETBenchmark_);
  
  if (doPFMETBenchmark_) {

    doMet_ = false;
    options_->GetOpt("MET", "on/off", doMet_);

    JECinCaloMet_ = false;
    options_->GetOpt("pfmet_benchmark", "JECinCaloMET", JECinCaloMet_);

    std::string outmetfilename;
    options_->GetOpt("pfmet_benchmark", "outmetfile", outmetfilename);

    // define here the various benchmark comparison
    metManager_.reset( new METManager(outmetfilename) );
    metManager_->addGenBenchmark("PF");
    metManager_->addGenBenchmark("Calo");
    if ( doMet_ ) metManager_->addGenBenchmark("recompPF");
    if (JECinCaloMet_) metManager_->addGenBenchmark("corrCalo");

    bool pfmetBenchmarkDebug;
    options_->GetOpt("pfmet_benchmark", "debug", pfmetBenchmarkDebug);
        
    MET1cut = 10.0;
    options_->GetOpt("pfmet_benchmark", "truemetcut", MET1cut);
    
    DeltaMETcut = 30.0;
    options_->GetOpt("pfmet_benchmark", "deltametcut", DeltaMETcut);
    
    DeltaPhicut = 0.8;
    options_->GetOpt("pfmet_benchmark", "deltaphicut", DeltaPhicut);
    
    std::vector<unsigned int> vIgnoreParticlesIDs;
    options_->GetOpt("pfmet_benchmark", "trueMetIgnoreParticlesIDs", vIgnoreParticlesIDs);
    //std::cout << "FL: vIgnoreParticlesIDs.size() = " << vIgnoreParticlesIDs.size() << std::endl;
    //std::cout << "FL: first = " << vIgnoreParticlesIDs[0] << std::endl;
    metManager_->SetIgnoreParticlesIDs(&vIgnoreParticlesIDs);

    std::vector<unsigned int> trueMetSpecificIdCut;
    std::vector<double> trueMetSpecificEtaCut;
    options_->GetOpt("pfmet_benchmark", "trueMetSpecificIdCut", trueMetSpecificIdCut);
    options_->GetOpt("pfmet_benchmark", "trueMetSpecificEtaCut", trueMetSpecificEtaCut);
    if (trueMetSpecificIdCut.size()!=trueMetSpecificEtaCut.size()) std::cout << "Warning: PFRootEventManager: trueMetSpecificIdCut.size()!=trueMetSpecificEtaCut.size()" << std::endl;
    else metManager_->SetSpecificIdCut(&trueMetSpecificIdCut,&trueMetSpecificEtaCut);

  }

  // input root file --------------------------------------------

  if( reconnect )
    connect( inFileName_.c_str() );


  // filter --------------------------------------------------------------

  filterNParticles_ = 0;
  options_->GetOpt("filter", "nparticles", filterNParticles_);
  
  filterHadronicTaus_ = true;
  options_->GetOpt("filter", "hadronic_taus", filterHadronicTaus_);
  
  filterTaus_.clear();
  options_->GetOpt("filter", "taus", filterTaus_);
  if( !filterTaus_.empty() &&
      filterTaus_.size() != 2 ) {
    cerr<<"PFRootEventManager::ReadOptions, bad filter/taus option."<<endl
        <<"please use : "<<endl
        <<"\tfilter taus n_charged n_neutral"<<endl;
    filterTaus_.clear();
  }
  
  
  // clustering parameters -----------------------------------------------

  doClustering_ = true;
  options_->GetOpt("clustering", "on/off", doClustering_);
  
  bool clusteringDebug = false;
  options_->GetOpt("clustering", "debug", clusteringDebug );

  findRecHitNeighbours_ = true;
  options_->GetOpt("clustering", "findRecHitNeighbours", 
                   findRecHitNeighbours_);
  
  double threshEcalBarrel = 0.1;
  options_->GetOpt("clustering", "thresh_Ecal_Barrel", threshEcalBarrel);
  
  double threshPtEcalBarrel = 0.0;
  options_->GetOpt("clustering", "thresh_Pt_Ecal_Barrel", threshPtEcalBarrel);
  
  double threshSeedEcalBarrel = 0.3;
  options_->GetOpt("clustering", "thresh_Seed_Ecal_Barrel", 
                   threshSeedEcalBarrel);

  double threshPtSeedEcalBarrel = 0.0;
  options_->GetOpt("clustering", "thresh_Pt_Seed_Ecal_Barrel", 
                   threshPtSeedEcalBarrel);

  double threshEcalEndcap = 0.2;
  options_->GetOpt("clustering", "thresh_Ecal_Endcap", threshEcalEndcap);

  double threshPtEcalEndcap = 0.0;
  options_->GetOpt("clustering", "thresh_Pt_Ecal_Endcap", threshPtEcalEndcap);

  double threshSeedEcalEndcap = 0.8;
  options_->GetOpt("clustering", "thresh_Seed_Ecal_Endcap",
                   threshSeedEcalEndcap);

  double threshPtSeedEcalEndcap = 0.0;
  options_->GetOpt("clustering", "thresh_Pt_Seed_Ecal_Endcap",
                   threshPtSeedEcalEndcap);

  double showerSigmaEcal = 3;  
  options_->GetOpt("clustering", "shower_Sigma_Ecal",
                   showerSigmaEcal);

  int nNeighboursEcal = 4;
  options_->GetOpt("clustering", "neighbours_Ecal", nNeighboursEcal);
  
  int posCalcNCrystalEcal = -1;
  options_->GetOpt("clustering", "posCalc_nCrystal_Ecal", 
                   posCalcNCrystalEcal);

  double posCalcP1Ecal 
    = threshEcalBarrel<threshEcalEndcap ? threshEcalBarrel:threshEcalEndcap;
//   options_->GetOpt("clustering", "posCalc_p1_Ecal", 
//                    posCalcP1Ecal);
  
  bool useCornerCellsEcal = false;
  options_->GetOpt("clustering", "useCornerCells_Ecal",
                   useCornerCellsEcal);


  clusterAlgoECAL_.setThreshBarrel( threshEcalBarrel );
  clusterAlgoECAL_.setThreshSeedBarrel( threshSeedEcalBarrel );
  
  clusterAlgoECAL_.setThreshPtBarrel( threshPtEcalBarrel );
  clusterAlgoECAL_.setThreshPtSeedBarrel( threshPtSeedEcalBarrel );
  
  clusterAlgoECAL_.setThreshEndcap( threshEcalEndcap );
  clusterAlgoECAL_.setThreshSeedEndcap( threshSeedEcalEndcap );

  clusterAlgoECAL_.setThreshPtEndcap( threshPtEcalEndcap );
  clusterAlgoECAL_.setThreshPtSeedEndcap( threshPtSeedEcalEndcap );

  clusterAlgoECAL_.setNNeighbours( nNeighboursEcal );
  clusterAlgoECAL_.setShowerSigma( showerSigmaEcal );

  clusterAlgoECAL_.setPosCalcNCrystal( posCalcNCrystalEcal );
  clusterAlgoECAL_.setPosCalcP1( posCalcP1Ecal );

  clusterAlgoECAL_.setUseCornerCells( useCornerCellsEcal );

  clusterAlgoECAL_.enableDebugging( clusteringDebug ); 

  int dcormode = 0;
  options_->GetOpt("clustering", "depthCor_Mode", dcormode);
  
  double dcora = -1;
  options_->GetOpt("clustering", "depthCor_A", dcora);
  double dcorb = -1;
  options_->GetOpt("clustering", "depthCor_B", dcorb);
  double dcorap = -1;
  options_->GetOpt("clustering", "depthCor_A_preshower", dcorap);
  double dcorbp = -1;
  options_->GetOpt("clustering", "depthCor_B_preshower", dcorbp);

  //   if( dcormode > 0 && 
  //       dcora > -0.5 && 
  //       dcorb > -0.5 && 
  //       dcorap > -0.5 && 
  //       dcorbp > -0.5 ) {

  //     cout<<"set depth correction "
  //    <<dcormode<<" "<<dcora<<" "<<dcorb<<" "<<dcorap<<" "<<dcorbp<<endl;
  reco::PFCluster::setDepthCorParameters( dcormode, 
                                          dcora, dcorb, 
                                          dcorap, dcorbp);
  //   }
  //   else {
  //     reco::PFCluster::setDepthCorParameters( -1, 
  //                                        0,0 , 
  //                                        0,0 );
  //   }

  

  double threshHcalBarrel = 0.8;
  options_->GetOpt("clustering", "thresh_Hcal_Barrel", threshHcalBarrel);
  
  double threshPtHcalBarrel = 0.0;
  options_->GetOpt("clustering", "thresh_Pt_Hcal_Barrel", threshPtHcalBarrel);
  
  double threshSeedHcalBarrel = 1.4;
  options_->GetOpt("clustering", "thresh_Seed_Hcal_Barrel", 
                   threshSeedHcalBarrel);

  double threshPtSeedHcalBarrel = 0.0;
  options_->GetOpt("clustering", "thresh_Pt_Seed_Hcal_Barrel", 
                   threshPtSeedHcalBarrel);

  double threshHcalEndcap = 0.8;
  options_->GetOpt("clustering", "thresh_Hcal_Endcap", threshHcalEndcap);

  double threshPtHcalEndcap = 0.0;
  options_->GetOpt("clustering", "thresh_Pt_Hcal_Endcap", threshPtHcalEndcap);

  double threshSeedHcalEndcap = 1.4;
  options_->GetOpt("clustering", "thresh_Seed_Hcal_Endcap",
                   threshSeedHcalEndcap);

  double threshPtSeedHcalEndcap = 0.0;
  options_->GetOpt("clustering", "thresh_Pt_Seed_Hcal_Endcap",
                   threshPtSeedHcalEndcap);

  double showerSigmaHcal    = 15;
  options_->GetOpt("clustering", "shower_Sigma_Hcal",
                   showerSigmaHcal);
 
  int nNeighboursHcal = 4;
  options_->GetOpt("clustering", "neighbours_Hcal", nNeighboursHcal);

  int posCalcNCrystalHcal = 5;
  options_->GetOpt("clustering", "posCalc_nCrystal_Hcal",
                   posCalcNCrystalHcal);

  bool useCornerCellsHcal = false;
  options_->GetOpt("clustering", "useCornerCells_Hcal",
                   useCornerCellsHcal);

  double posCalcP1Hcal 
    = threshHcalBarrel<threshHcalEndcap ? threshHcalBarrel:threshHcalEndcap;
//   options_->GetOpt("clustering", "posCalc_p1_Hcal", 
//                    posCalcP1Hcal);




  clusterAlgoHCAL_.setThreshBarrel( threshHcalBarrel );
  clusterAlgoHCAL_.setThreshSeedBarrel( threshSeedHcalBarrel );
  
  clusterAlgoHCAL_.setThreshPtBarrel( threshPtHcalBarrel );
  clusterAlgoHCAL_.setThreshPtSeedBarrel( threshPtSeedHcalBarrel );
  
  clusterAlgoHCAL_.setThreshEndcap( threshHcalEndcap );
  clusterAlgoHCAL_.setThreshSeedEndcap( threshSeedHcalEndcap );

  clusterAlgoHCAL_.setThreshPtEndcap( threshPtHcalEndcap );
  clusterAlgoHCAL_.setThreshPtSeedEndcap( threshPtSeedHcalEndcap );

  clusterAlgoHCAL_.setNNeighbours( nNeighboursHcal );
  clusterAlgoHCAL_.setShowerSigma( showerSigmaHcal );

  clusterAlgoHCAL_.setPosCalcNCrystal( posCalcNCrystalHcal );
  clusterAlgoHCAL_.setPosCalcP1( posCalcP1Hcal );

  clusterAlgoHCAL_.setUseCornerCells( useCornerCellsHcal );

  clusterAlgoHCAL_.enableDebugging( clusteringDebug ); 


  // clustering HF EM 

  double threshHFEM = 0.;
  options_->GetOpt("clustering", "thresh_HFEM", threshHFEM);
  
  double threshPtHFEM = 0.;
  options_->GetOpt("clustering", "thresh_Pt_HFEM", threshPtHFEM);
  
  double threshSeedHFEM = 0.001;
  options_->GetOpt("clustering", "thresh_Seed_HFEM", 
                   threshSeedHFEM);
  
  double threshPtSeedHFEM = 0.0;
  options_->GetOpt("clustering", "thresh_Pt_Seed_HFEM", 
                   threshPtSeedHFEM);
  
  double showerSigmaHFEM    = 0.1;
  options_->GetOpt("clustering", "shower_Sigma_HFEM",
                   showerSigmaHFEM);
 
  int nNeighboursHFEM = 4;
  options_->GetOpt("clustering", "neighbours_HFEM", nNeighboursHFEM);

  int posCalcNCrystalHFEM = -1;
  options_->GetOpt("clustering", "posCalc_nCrystal_HFEM",
                   posCalcNCrystalHFEM);

  bool useCornerCellsHFEM = false;
  options_->GetOpt("clustering", "useCornerCells_HFEM",
                   useCornerCellsHFEM);

  double posCalcP1HFEM = threshHFEM;
//   options_->GetOpt("clustering", "posCalc_p1_HFEM", 
//                    posCalcP1HFEM);


  clusterAlgoHFEM_.setThreshEndcap( threshHFEM );
  clusterAlgoHFEM_.setThreshSeedEndcap( threshSeedHFEM );

  clusterAlgoHFEM_.setThreshPtEndcap( threshPtHFEM );
  clusterAlgoHFEM_.setThreshPtSeedEndcap( threshPtSeedHFEM );

  clusterAlgoHFEM_.setNNeighbours( nNeighboursHFEM );
  clusterAlgoHFEM_.setShowerSigma( showerSigmaHFEM );

  clusterAlgoHFEM_.setPosCalcNCrystal( posCalcNCrystalHFEM );
  clusterAlgoHFEM_.setPosCalcP1( posCalcP1HFEM );

  clusterAlgoHFEM_.setUseCornerCells( useCornerCellsHFEM );

  clusterAlgoHFEM_.enableDebugging( clusteringDebug ); 

  
  // clustering HFHAD 

  double threshHFHAD = 0.;
  options_->GetOpt("clustering", "thresh_HFHAD", threshHFHAD);
  
  double threshPtHFHAD = 0.;
  options_->GetOpt("clustering", "thresh_Pt_HFHAD", threshPtHFHAD);
  
  double threshSeedHFHAD = 0.001;
  options_->GetOpt("clustering", "thresh_Seed_HFHAD", 
                   threshSeedHFHAD);
  
  double threshPtSeedHFHAD = 0.0;
  options_->GetOpt("clustering", "thresh_Pt_Seed_HFHAD", 
                   threshPtSeedHFHAD);
  
  double showerSigmaHFHAD    = 0.1;
  options_->GetOpt("clustering", "shower_Sigma_HFHAD",
                   showerSigmaHFHAD);
 
  int nNeighboursHFHAD = 4;
  options_->GetOpt("clustering", "neighbours_HFHAD", nNeighboursHFHAD);

  int posCalcNCrystalHFHAD = -1;
  options_->GetOpt("clustering", "posCalc_nCrystal_HFHAD",
                   posCalcNCrystalHFHAD);

  bool useCornerCellsHFHAD = false;
  options_->GetOpt("clustering", "useCornerCells_HFHAD",
                   useCornerCellsHFHAD);

  double posCalcP1HFHAD = threshHFHAD;
//   options_->GetOpt("clustering", "posCalc_p1_HFHAD", 
//                    posCalcP1HFHAD);


  clusterAlgoHFHAD_.setThreshEndcap( threshHFHAD );
  clusterAlgoHFHAD_.setThreshSeedEndcap( threshSeedHFHAD );

  clusterAlgoHFHAD_.setThreshPtEndcap( threshPtHFHAD );
  clusterAlgoHFHAD_.setThreshPtSeedEndcap( threshPtSeedHFHAD );

  clusterAlgoHFHAD_.setNNeighbours( nNeighboursHFHAD );
  clusterAlgoHFHAD_.setShowerSigma( showerSigmaHFHAD );

  clusterAlgoHFHAD_.setPosCalcNCrystal( posCalcNCrystalHFHAD );
  clusterAlgoHFHAD_.setPosCalcP1( posCalcP1HFHAD );

  clusterAlgoHFHAD_.setUseCornerCells( useCornerCellsHFHAD );

  clusterAlgoHFHAD_.enableDebugging( clusteringDebug ); 

  


  // clustering preshower

  double threshPS = 0.0001;
  options_->GetOpt("clustering", "thresh_PS", threshPS);
  
  double threshPtPS = 0.0;
  options_->GetOpt("clustering", "thresh_Pt_PS", threshPtPS);
  
  double threshSeedPS = 0.001;
  options_->GetOpt("clustering", "thresh_Seed_PS", 
                   threshSeedPS);
  
  double threshPtSeedPS = 0.0;
  options_->GetOpt("clustering", "thresh_Pt_Seed_PS", 
                   threshPtSeedPS);
  
  //Comment Michel: PSBarrel shall be removed?
  double threshPSBarrel     = threshPS;
  double threshSeedPSBarrel = threshSeedPS;

  double threshPtPSBarrel     = threshPtPS;
  double threshPtSeedPSBarrel = threshPtSeedPS;

  double threshPSEndcap     = threshPS;
  double threshSeedPSEndcap = threshSeedPS;

  double threshPtPSEndcap     = threshPtPS;
  double threshPtSeedPSEndcap = threshPtSeedPS;

  double showerSigmaPS    = 0.1;
  options_->GetOpt("clustering", "shower_Sigma_PS",
                   showerSigmaPS);
 
  int nNeighboursPS = 4;
  options_->GetOpt("clustering", "neighbours_PS", nNeighboursPS);

  int posCalcNCrystalPS = -1;
  options_->GetOpt("clustering", "posCalc_nCrystal_PS",
                   posCalcNCrystalPS);

  bool useCornerCellsPS = false;
  options_->GetOpt("clustering", "useCornerCells_PS",
                   useCornerCellsPS);

  double posCalcP1PS = threshPS;
//   options_->GetOpt("clustering", "posCalc_p1_PS", 
//                    posCalcP1PS);




  clusterAlgoPS_.setThreshBarrel( threshPSBarrel );
  clusterAlgoPS_.setThreshSeedBarrel( threshSeedPSBarrel );
  
  clusterAlgoPS_.setThreshPtBarrel( threshPtPSBarrel );
  clusterAlgoPS_.setThreshPtSeedBarrel( threshPtSeedPSBarrel );
  
  clusterAlgoPS_.setThreshEndcap( threshPSEndcap );
  clusterAlgoPS_.setThreshSeedEndcap( threshSeedPSEndcap );

  clusterAlgoPS_.setThreshPtEndcap( threshPtPSEndcap );
  clusterAlgoPS_.setThreshPtSeedEndcap( threshPtSeedPSEndcap );

  clusterAlgoPS_.setNNeighbours( nNeighboursPS );
  clusterAlgoPS_.setShowerSigma( showerSigmaPS );

  clusterAlgoPS_.setPosCalcNCrystal( posCalcNCrystalPS );
  clusterAlgoPS_.setPosCalcP1( posCalcP1PS );

  clusterAlgoPS_.setUseCornerCells( useCornerCellsPS );

  clusterAlgoPS_.enableDebugging( clusteringDebug ); 

  


  // options for particle flow ---------------------------------------------


  doParticleFlow_ = true;
  options_->GetOpt("particle_flow", "on/off", doParticleFlow_);  

  std::vector<double> DPtovPtCut;
  std::vector<unsigned> NHitCut;
  options_->GetOpt("particle_flow", "DPtoverPt_Cut", DPtovPtCut);
  options_->GetOpt("particle_flow", "NHit_Cut", NHitCut);

  try {
    pfBlockAlgo_.setParameters( DPtovPtCut, 
				NHitCut); 
  }  
  catch( std::exception& err ) {
    cerr<<"exception setting PFBlockAlgo parameters: "
        <<err.what()<<". terminating."<<endl;
    delete this;
    exit(1);
  }
  

  bool blockAlgoDebug = false;
  options_->GetOpt("blockAlgo", "debug",  blockAlgoDebug);  
  pfBlockAlgo_.setDebug( blockAlgoDebug );

  bool AlgoDebug = false;
  options_->GetOpt("PFAlgo", "debug",  AlgoDebug);  
  pfAlgo_.setDebug( AlgoDebug );

  // read PFCluster calibration parameters
  

  double e_slope = 1.0;
  options_->GetOpt("particle_flow","calib_ECAL_slope", e_slope);
  double e_offset = 0;
  options_->GetOpt("particle_flow","calib_ECAL_offset", e_offset);
  
  double eh_eslope = 1.05;
  options_->GetOpt("particle_flow","calib_ECAL_HCAL_eslope", eh_eslope);
  double eh_hslope = 1.06;
  options_->GetOpt("particle_flow","calib_ECAL_HCAL_hslope", eh_hslope);
  double eh_offset = 6.11;
  options_->GetOpt("particle_flow","calib_ECAL_HCAL_offset", eh_offset);
  
  double h_slope = 2.17;
  options_->GetOpt("particle_flow","calib_HCAL_slope", h_slope);
  double h_offset = 1.73;
  options_->GetOpt("particle_flow","calib_HCAL_offset", h_offset);
  double h_damping = 2.49;
  options_->GetOpt("particle_flow","calib_HCAL_damping", h_damping);
  

  unsigned newCalib = 0;
  options_->GetOpt("particle_flow", "newCalib", newCalib);  
  std::cout << "New calib = " << newCalib << std::endl;

  shared_ptr<pftools::PFClusterCalibration> 
    clusterCalibration( new pftools::PFClusterCalibration() );
  clusterCalibration_ = clusterCalibration;

  shared_ptr<PFEnergyCalibration> 
    calibration( new PFEnergyCalibration( e_slope,
                                          e_offset, 
                                          eh_eslope,
                                          eh_hslope,
                                          eh_offset,
                                          h_slope,
                                          h_offset,
                                          h_damping,
					  newCalib) );
  calibration_ = calibration;

  //--ab: get calibration factors for HF:
  bool calibHF_use = false;
  std::vector<double>  calibHF_eta_step;
  std::vector<double>  calibHF_a_EMonly;
  std::vector<double>  calibHF_b_HADonly;
  std::vector<double>  calibHF_a_EMHAD;
  std::vector<double>  calibHF_b_EMHAD;

  options_->GetOpt("particle_flow","calib_calibHF_use",calibHF_use);
  options_->GetOpt("particle_flow","calib_calibHF_eta_step",calibHF_eta_step);
  options_->GetOpt("particle_flow","calib_calibHF_a_EMonly",calibHF_a_EMonly);
  options_->GetOpt("particle_flow","calib_calibHF_b_HADonly",calibHF_b_HADonly);
  options_->GetOpt("particle_flow","calib_calibHF_a_EMHAD",calibHF_a_EMHAD);
  options_->GetOpt("particle_flow","calib_calibHF_b_EMHAD",calibHF_b_EMHAD);

  shared_ptr<PFEnergyCalibrationHF>  thepfEnergyCalibrationHF
    ( new PFEnergyCalibrationHF(calibHF_use,calibHF_eta_step,calibHF_a_EMonly,calibHF_b_HADonly,calibHF_a_EMHAD,calibHF_b_EMHAD) ) ;

  thepfEnergyCalibrationHF_ = thepfEnergyCalibrationHF;


  //----------------------------------------
  double nSigmaECAL = 99999;
  options_->GetOpt("particle_flow", "nsigma_ECAL", nSigmaECAL);
  double nSigmaHCAL = 99999;
  options_->GetOpt("particle_flow", "nsigma_HCAL", nSigmaHCAL);

  // pfAlgo_.setNewCalibration(newCalib);

  // Set the parameters for the brand-new calibration
  double g0, g1, e0, e1;
  options_->GetOpt("correction", "globalP0", g0);
  options_->GetOpt("correction", "globalP1", g1);
  options_->GetOpt("correction", "lowEP0", e0);
  options_->GetOpt("correction", "lowEP1", e1);
  clusterCalibration->setCorrections(e0, e1, g0, g1);
  
  int allowNegative(0);
  options_->GetOpt("correction", "allowNegativeEnergy", allowNegative);
  clusterCalibration->setAllowNegativeEnergy(allowNegative);
  
  int doCorrection(1);
  options_->GetOpt("correction", "doCorrection", doCorrection);
  clusterCalibration->setDoCorrection(doCorrection);

  int doEtaCorrection(1);
  options_->GetOpt("correction", "doEtaCorrection", doEtaCorrection);
  clusterCalibration->setDoEtaCorrection(doEtaCorrection);
  
  double barrelEta;
  options_->GetOpt("evolution", "barrelEndcapEtaDiv", barrelEta);
  clusterCalibration->setBarrelBoundary(barrelEta);
  
  double ecalEcut;
  options_->GetOpt("evolution", "ecalECut", ecalEcut);
  double hcalEcut;
  options_->GetOpt("evolution", "hcalECut", hcalEcut);
  clusterCalibration->setEcalHcalEnergyCuts(ecalEcut,hcalEcut);

  std::vector<std::string>* names = clusterCalibration->getKnownSectorNames();
  for(std::vector<std::string>::iterator i = names->begin(); i != names->end(); ++i) {
    std::string sector = *i;
    std::vector<double> params;
    options_->GetOpt("evolution", sector.c_str(), params);
    clusterCalibration->setEvolutionParameters(sector, params);
  }

  std::vector<double> etaCorrectionParams; 
  options_->GetOpt("evolution","etaCorrection", etaCorrectionParams);
  clusterCalibration->setEtaCorrectionParameters(etaCorrectionParams);

  try {
    pfAlgo_.setParameters( nSigmaECAL, nSigmaHCAL, 
                           calibration,
			   clusterCalibration,thepfEnergyCalibrationHF_, newCalib);
  }
  catch( std::exception& err ) {
    cerr<<"exception setting PFAlgo parameters: "
        <<err.what()<<". terminating."<<endl;
    delete this;
    exit(1);
  }

  std::vector<double> muonHCAL;
  std::vector<double> muonECAL;
  options_->GetOpt("particle_flow", "muon_HCAL", muonHCAL);
  options_->GetOpt("particle_flow", "muon_ECAL", muonECAL);
  assert ( muonHCAL.size() == 2 && muonECAL.size() == 2 );

  double nSigmaTRACK = 3.0;
  options_->GetOpt("particle_flow", "nsigma_TRACK", nSigmaTRACK);

  double ptError = 1.0;
  options_->GetOpt("particle_flow", "pt_error", ptError);
  
  std::vector<double> factors45;
  options_->GetOpt("particle_flow", "factors_45", factors45);
  assert ( factors45.size() == 2 );
  

  try { 
    pfAlgo_.setPFMuonAndFakeParameters(muonHCAL,
				       muonECAL,
				       nSigmaTRACK,
				       ptError,
				       factors45);
  }
  catch( std::exception& err ) {
    cerr<<"exception setting PFAlgo Muon and Fake parameters: "
        <<err.what()<<". terminating."<<endl;
    delete this;
    exit(1);
  }
  


  bool usePFElectrons = false;   // set true to use PFElectrons
  options_->GetOpt("particle_flow", "usePFElectrons", usePFElectrons);
  cout<<"use PFElectrons "<<usePFElectrons<<endl;

  if( usePFElectrons ) { 
    // PFElectrons options -----------------------------
    double mvaEleCut = -1.;  // if = -1. get all the pre-id electrons
    options_->GetOpt("particle_flow", "electron_mvaCut", mvaEleCut);

    string mvaWeightFileEleID = "";
    options_->GetOpt("particle_flow", "electronID_mvaWeightFile", 
		     mvaWeightFileEleID);
    mvaWeightFileEleID = expand(mvaWeightFileEleID);
    
    try { 
      pfAlgo_.setPFEleParameters(mvaEleCut,
				 mvaWeightFileEleID,
				 usePFElectrons);
    }
    catch( std::exception& err ) {
      cerr<<"exception setting PFAlgo Electron parameters: "
	  <<err.what()<<". terminating."<<endl;
      delete this;
      exit(1);
    }
  }


  bool usePFConversions = false;   // set true to use PFConversions
  options_->GetOpt("particle_flow", "usePFConversions", usePFConversions);

  try { 
    pfAlgo_.setPFConversionParameters(usePFConversions);
  }
  catch( std::exception& err ) {
    cerr<<"exception setting PFAlgo Conversions parameters: "
        <<err.what()<<". terminating."<<endl;
    delete this;
    exit(1);
  }



  int    algo = 2;
  options_->GetOpt("particle_flow", "algorithm", algo);

  pfAlgo_.setAlgo( algo );
  //   pfAlgoOther_.setAlgo( 1 );


  // jets options ---------------------------------

  doJets_ = false;
  options_->GetOpt("jets", "on/off", doJets_);

  jetsDebug_ = false;
  options_->GetOpt("jets", "debug", jetsDebug_);

  jetAlgoType_=3; //FastJet as Default
  options_->GetOpt("jets", "algo", jetAlgoType_);

  double mEtInputCut = 0.5;
  options_->GetOpt("jets", "EtInputCut",  mEtInputCut);           

  double mEInputCut = 0.;
  options_->GetOpt("jets", "EInputCut",  mEInputCut);  

  double seedThreshold  = 1.0;
  options_->GetOpt("jets", "seedThreshold", seedThreshold);

  double coneRadius = 0.5;
  options_->GetOpt("jets", "coneRadius", coneRadius);             

  double coneAreaFraction= 1.0;
  options_->GetOpt("jets", "coneAreaFraction",  coneAreaFraction);   

  int maxPairSize=2;
  options_->GetOpt("jets", "maxPairSize",  maxPairSize);  

  int maxIterations=100;
  options_->GetOpt("jets", "maxIterations",  maxIterations);      

  double overlapThreshold  = 0.75;
  options_->GetOpt("jets", "overlapThreshold", overlapThreshold);

  double ptMin = 10.;
  options_->GetOpt("jets", "ptMin",  ptMin);      

  double rparam = 1.0;
  options_->GetOpt("jets", "rParam",  rparam);    
 
  jetMaker_.setmEtInputCut (mEtInputCut);
  jetMaker_.setmEInputCut(mEInputCut); 
  jetMaker_.setSeedThreshold(seedThreshold); 
  jetMaker_.setConeRadius(coneRadius);
  jetMaker_.setConeAreaFraction(coneAreaFraction);
  jetMaker_.setMaxPairSize(maxPairSize);
  jetMaker_.setMaxIterations(maxIterations) ;
  jetMaker_.setOverlapThreshold(overlapThreshold) ;
  jetMaker_.setPtMin (ptMin);
  jetMaker_.setRParam (rparam);
  jetMaker_.setDebug(jetsDebug_);
  jetMaker_.updateParameter();
  cout <<"Opt: doJets? " << doJets_  <<endl; 
  cout <<"Opt: jetsDebug " << jetsDebug_  <<endl; 
  cout <<"Opt: algoType " << jetAlgoType_  <<endl; 
  cout <<"----------------------------------" << endl;


  // tau benchmark options ---------------------------------

  doTauBenchmark_ = false;
  options_->GetOpt("tau_benchmark", "on/off", doTauBenchmark_);
  
  if (doTauBenchmark_) {
    double coneAngle = 0.5;
    options_->GetOpt("tau_benchmark", "cone_angle", coneAngle);
    
    double seedEt    = 0.4;
    options_->GetOpt("tau_benchmark", "seed_et", seedEt);
    
    double coneMerge = 100.0;
    options_->GetOpt("tau_benchmark", "cone_merge", coneMerge);
    
    options_->GetOpt("tau_benchmark", "debug", tauBenchmarkDebug_);

    // cout<<"jets debug "<<jetsDebug_<<endl;
    
    if( tauBenchmarkDebug_ ) {
      cout << "Tau Benchmark Options : ";
      cout << "Angle=" << coneAngle << " seedEt=" << seedEt 
           << " Merge=" << coneMerge << endl;
    }

    jetAlgo_.SetConeAngle(coneAngle);
    jetAlgo_.SetSeedEt(seedEt);
    jetAlgo_.SetConeMerge(coneMerge);   
  }



  // print flags -------------

  printRecHits_ = false;
  options_->GetOpt("print", "rechits", printRecHits_ );
  
  printClusters_ = false;
  options_->GetOpt("print", "clusters", printClusters_ );
  
  printPFBlocks_ = false;
  options_->GetOpt("print", "PFBlocks", printPFBlocks_ );
  
  printPFCandidates_ = true;
  options_->GetOpt("print", "PFCandidates", printPFCandidates_ );
  
  printPFJets_ = true;
  printPFPt_ = 0.;
  options_->GetOpt("print", "jets", printPFJets_ );
  options_->GetOpt("print", "ptjets", printPFPt_ );
 
  printSimParticles_ = true;
  options_->GetOpt("print", "simParticles", printSimParticles_ );

  printGenParticles_ = true;
  options_->GetOpt("print", "genParticles", printGenParticles_ );

  //MCTruthMatching Tool set to false by default
  //can only be used with fastsim and the UnFoldedMode set to true
  //when generating the simulated file
  printMCTruthMatching_ = false; 
  options_->GetOpt("print", "mctruthmatching", printMCTruthMatching_ );  


  verbosity_ = VERBOSE;
  options_->GetOpt("print", "verbosity", verbosity_ );
  cout<<"verbosity : "<<verbosity_<<endl;

}

void PFRootEventManager::connect( const char* infilename ) {

  string fname = infilename;
  if( fname.empty() ) 
    fname = inFileName_;

  
  cout<<"opening input root file"<<endl;

  options_->GetOpt("root","file", inFileName_);
  


  try {
    AutoLibraryLoader::enable();
  }
  catch(string& err) {
    cout<<err<<endl;
  }




  file_ = TFile::Open(inFileName_.c_str() );


  if(!file_ ) return;
  else if(file_->IsZombie() ) {
    return;
  }
  else 
    cout<<"rootfile "<<inFileName_
        <<" opened"<<endl;

  

  tree_ = (TTree*) file_->Get("Events");  
  if(!tree_) {
    cerr<<"PFRootEventManager::ReadOptions :";
    cerr<<"input TTree Events not found in file "
        <<inFileName_<<endl;
    return; 
  }

  tree_->GetEntry();
   
  
  // hits branches ----------------------------------------------

  string rechitsECALbranchname;
  options_->GetOpt("root","rechits_ECAL_branch", rechitsECALbranchname);
  
  rechitsECALBranch_ = tree_->GetBranch(rechitsECALbranchname.c_str());
  if(!rechitsECALBranch_) {
    cerr<<"PFRootEventManager::ReadOptions : rechits_ECAL_branch not found : "
        <<rechitsECALbranchname<<endl;
  }

  string rechitsHCALbranchname;
  options_->GetOpt("root","rechits_HCAL_branch", rechitsHCALbranchname);
  
  rechitsHCALBranch_ = tree_->GetBranch(rechitsHCALbranchname.c_str());
  if(!rechitsHCALBranch_) {
    cerr<<"PFRootEventManager::ReadOptions : rechits_HCAL_branch not found : "
        <<rechitsHCALbranchname<<endl;
  }

  string rechitsHFEMbranchname;
  options_->GetOpt("root","rechits_HFEM_branch", rechitsHFEMbranchname);
  
  rechitsHFEMBranch_ = tree_->GetBranch(rechitsHFEMbranchname.c_str());
  if(!rechitsHFEMBranch_) {
    cerr<<"PFRootEventManager::ReadOptions : rechits_HFEM_branch not found : "
        <<rechitsHFEMbranchname<<endl;
  }

  string rechitsHFHADbranchname;
  options_->GetOpt("root","rechits_HFHAD_branch", rechitsHFHADbranchname);
  
  rechitsHFHADBranch_ = tree_->GetBranch(rechitsHFHADbranchname.c_str());
  if(!rechitsHFHADBranch_) {
    cerr<<"PFRootEventManager::ReadOptions : rechits_HFHAD_branch not found : "
        <<rechitsHFHADbranchname<<endl;
  }

  string rechitsPSbranchname;
  options_->GetOpt("root","rechits_PS_branch", rechitsPSbranchname);
  
  rechitsPSBranch_ = tree_->GetBranch(rechitsPSbranchname.c_str());
  if(!rechitsPSBranch_) {
    cerr<<"PFRootEventManager::ReadOptions : rechits_PS_branch not found : "
        <<rechitsPSbranchname<<endl;
  }


  // clusters branches ----------------------------------------------

  
  clustersECALBranch_ = 0;
  clustersHCALBranch_ = 0;
  //COLIN not adding a branch to read HF clusters from the file. 
  // we never use this functionality anyway for the other detectors
  clustersPSBranch_ = 0;


  if( !doClustering_ ) {
    string clustersECALbranchname;
    options_->GetOpt("root","clusters_ECAL_branch", clustersECALbranchname);
    
    clustersECALBranch_ = tree_->GetBranch(clustersECALbranchname.c_str());
    if(!clustersECALBranch_) {
      cerr <<"PFRootEventManager::ReadOptions : clusters_ECAL_branch not found:"
           <<clustersECALbranchname<<endl;
    }
  

    string clustersHCALbranchname;
    options_->GetOpt("root","clusters_HCAL_branch", clustersHCALbranchname);
    
    clustersHCALBranch_ = tree_->GetBranch(clustersHCALbranchname.c_str());
    if(!clustersHCALBranch_) {
      cerr<<"PFRootEventManager::ReadOptions : clusters_HCAL_branch not found : "
          <<clustersHCALbranchname<<endl;
    }
  
    string clustersPSbranchname;
    options_->GetOpt("root","clusters_PS_branch", clustersPSbranchname);

    clustersPSBranch_ = tree_->GetBranch(clustersPSbranchname.c_str());
    if(!clustersPSBranch_) {
      cerr<<"PFRootEventManager::ReadOptions : clusters_PS_branch not found : "
          <<clustersPSbranchname<<endl;
    }
  }

  // other branches ----------------------------------------------
  
  
//   string clustersIslandBarrelbranchname;
//   clustersIslandBarrelBranch_ = 0;
//   options_->GetOpt("root","clusters_island_barrel_branch", 
//                    clustersIslandBarrelbranchname);
//   if(!clustersIslandBarrelbranchname.empty() ) {
//     clustersIslandBarrelBranch_ 
//       = tree_->GetBranch(clustersIslandBarrelbranchname.c_str());
//     if(!clustersIslandBarrelBranch_) {
//       cerr<<"PFRootEventManager::ReadOptions : clusters_island_barrel_branch not found : "
//           <<clustersIslandBarrelbranchname<< endl;
//     }
//   }
//   else {
//     cerr<<"branch not found: root/clusters_island_barrel_branch"<<endl;
//   }

  string recTracksbranchname;
  options_->GetOpt("root","recTracks_branch", recTracksbranchname);

  recTracksBranch_ = tree_->GetBranch(recTracksbranchname.c_str());
  if(!recTracksBranch_) {
    cerr<<"PFRootEventManager::ReadOptions : recTracks_branch not found : "
        <<recTracksbranchname<< endl;
  }


  string primaryVertexbranchname;
  options_->GetOpt("root","primaryVertex_branch", primaryVertexbranchname);

  primaryVertexBranch_ = tree_->GetBranch(primaryVertexbranchname.c_str());
  if(!primaryVertexBranch_) {
    cerr<<"PFRootEventManager::ReadOptions : primaryVertex_branch not found : "
        <<primaryVertexbranchname<< endl;
  }

  string stdTracksbranchname;
  options_->GetOpt("root","stdTracks_branch", stdTracksbranchname);

  stdTracksBranch_ = tree_->GetBranch(stdTracksbranchname.c_str());
  if(!stdTracksBranch_) {
    cerr<<"PFRootEventManager::ReadOptions : stdTracks_branch not found : "
        <<stdTracksbranchname<< endl;
  }
  
  string gsfTracksbranchname; 
  options_->GetOpt("root","gsfrecTracks_branch",gsfTracksbranchname); 
  gsfrecTracksBranch_ = tree_->GetBranch(gsfTracksbranchname.c_str()); 
  if(!gsfrecTracksBranch_) { 
    cerr<<"PFRootEventManager::ReadOptions : gsfrecTracks_branch not found : " 
        <<gsfTracksbranchname<< endl; 
  } 

  //muons
  string muonbranchname;
  options_->GetOpt("root","muon_branch",muonbranchname); 
  muonsBranch_= tree_->GetBranch(muonbranchname.c_str());
  if(!muonsBranch_) { 
    cerr<<"PFRootEventManager::ReadOptions : muon_branch not found : " 
        <<muonbranchname<< endl; 
  } 
  //nuclear
  useNuclear_=false;
   options_->GetOpt("particle_flow", "useNuclear", useNuclear_);
   if( useNuclear_ ) {

  string nuclearbranchname;
  options_->GetOpt("root","nuclear_branch",nuclearbranchname); 
  nuclearBranch_= tree_->GetBranch(nuclearbranchname.c_str());
  if(!nuclearBranch_) { 
    cerr<<"PFRootEventManager::ReadOptions : nuclear_branch not found : " 
        <<nuclearbranchname<< endl; 
  } 
  }
  //conversion

   useConversions_=false;
   options_->GetOpt("particle_flow", "usePFConversions", useConversions_);
   if( useConversions_ ) {
     string conversionbranchname;
     options_->GetOpt("root","conversion_branch",conversionbranchname); 
     conversionBranch_= tree_->GetBranch(conversionbranchname.c_str());
     if(!conversionBranch_) { 
       cerr<<"PFRootEventManager::ReadOptions : conversion_branch not found : " 
	   <<conversionbranchname<< endl; 
     } 
  }

  //V0

  useV0_=false;
  options_->GetOpt("particle_flow", "useV0", useV0_);
  if( useV0_ ) {
    
    string V0branchname;
    options_->GetOpt("root","V0_branch",V0branchname); 
    v0Branch_= tree_->GetBranch(V0branchname.c_str());
    if(!v0Branch_) { 
      cerr<<"PFRootEventManager::ReadOptions : V0_branch not found : " 
	  <<V0branchname<< endl; 
    } 
  }


  string trueParticlesbranchname;
  options_->GetOpt("root","trueParticles_branch", trueParticlesbranchname);

  trueParticlesBranch_ = tree_->GetBranch(trueParticlesbranchname.c_str());
  if(!trueParticlesBranch_) {
    cerr<<"PFRootEventManager::ReadOptions : trueParticles_branch not found : "
        <<trueParticlesbranchname<< endl;
  }
  

  string MCTruthbranchname;
  options_->GetOpt("root","MCTruth_branch", MCTruthbranchname);

  MCTruthBranch_ = tree_->GetBranch(MCTruthbranchname.c_str());
  if(!MCTruthBranch_) {
    cerr<<"PFRootEventManager::ReadOptions : MCTruth_branch not found : "
        <<MCTruthbranchname << endl;
  }

  string caloTowersBranchName;
  caloTowersBranch_ = 0;
  options_->GetOpt("root","caloTowers_branch", caloTowersBranchName);
  if(!caloTowersBranchName.empty() ) {
    caloTowersBranch_ = tree_->GetBranch(caloTowersBranchName.c_str()); 
    if(!caloTowersBranch_) {
      cerr<<"PFRootEventManager::ReadOptions : caloTowers_branch not found : "
          <<caloTowersBranchName<< endl;
    }
  }    
  
  // GenParticlesCand   
  string genParticleCandBranchName;
  genParticlesforJetsBranch_ = 0;
  options_->GetOpt("root","genParticleforJets_branch", 
                   genParticleCandBranchName);
  if(!genParticleCandBranchName.empty() ){  
    genParticlesforJetsBranch_= 
      tree_->GetBranch(genParticleCandBranchName.c_str()); 
    if(!genParticlesforJetsBranch_) {
      cerr<<"PFRootEventanager::ReadOptions : "
          <<"genParticleforJets_branch not found : "
          <<genParticleCandBranchName<< endl;
    }  
  }
       
  // calo tower base candidates 
  string caloTowerCandBranchName;
  caloTowerBaseCandidatesBranch_ = 0;
  options_->GetOpt("root","caloTowerBaseCandidates_branch", 
		   caloTowerCandBranchName);
  if(!caloTowerCandBranchName.empty() ){  
    caloTowerBaseCandidatesBranch_= 
      tree_->GetBranch(caloTowerCandBranchName.c_str()); 
    if(!caloTowerBaseCandidatesBranch_) {
      cerr<<"PFRootEventanager::ReadOptions : "
	  <<"caloTowerBaseCandidates_branch not found : "
          <<caloTowerCandBranchName<< endl;
    }  
  }

      
  string genJetBranchName; 
  options_->GetOpt("root","genJetBranchName", genJetBranchName);
  if(!genJetBranchName.empty() ) {
    genJetBranch_= tree_->GetBranch(genJetBranchName.c_str()); 
    if(!genJetBranch_) {
      cerr<<"PFRootEventManager::ReadOptions :genJetBranch_ not found : "
          <<genJetBranchName<< endl;
    }
  }
  
  string recCaloBranchName;
  options_->GetOpt("root","recCaloJetBranchName", recCaloBranchName);
  if(!recCaloBranchName.empty() ) {
    recCaloBranch_= tree_->GetBranch(recCaloBranchName.c_str()); 
    if(!recCaloBranch_) {
      cerr<<"PFRootEventManager::ReadOptions :recCaloBranch_ not found : "
          <<recCaloBranchName<< endl;
    }
  }
  string reccorrCaloBranchName;
  options_->GetOpt("root","reccorrCaloJetBranchName", reccorrCaloBranchName);
  if(!reccorrCaloBranchName.empty() ) {
    reccorrCaloBranch_= tree_->GetBranch(reccorrCaloBranchName.c_str()); 
    if(!reccorrCaloBranch_) {
      cerr<<"PFRootEventManager::ReadOptions :reccorrCaloBranch_ not found : "
          <<reccorrCaloBranchName<< endl;
    }
  }
  string recPFBranchName; 
  options_->GetOpt("root","recPFJetBranchName", recPFBranchName);
  if(!recPFBranchName.empty() ) {
    recPFBranch_= tree_->GetBranch(recPFBranchName.c_str()); 
    if(!recPFBranch_) {
      cerr<<"PFRootEventManager::ReadOptions :recPFBranch_ not found : "
          <<recPFBranchName<< endl;
    }
  }

  string recPFMETBranchName; 
  options_->GetOpt("root","recPFMETBranchName", recPFMETBranchName);
  if(!recPFMETBranchName.empty() ) {
    recPFMETBranch_= tree_->GetBranch(recPFMETBranchName.c_str()); 
    if(!recPFMETBranch_) {
      cerr<<"PFRootEventManager::ReadOptions :recPFMETBranch_ not found : "
          <<recPFMETBranchName<< endl;
    }
  }
  string recCaloMETBranchName; 
  options_->GetOpt("root","recCaloMETBranchName", recCaloMETBranchName);
  if(!recCaloMETBranchName.empty() ) {
    recCaloMETBranch_= tree_->GetBranch(recCaloMETBranchName.c_str()); 
    if(!recCaloMETBranch_) {
      cerr<<"PFRootEventManager::ReadOptions :recCaloMETBranch_ not found : "
          <<recCaloMETBranchName<< endl;
    }
  }
  string recTCMETBranchName; 
  options_->GetOpt("root","recTCMETBranchName", recTCMETBranchName);
  if(!recTCMETBranchName.empty() ) {
    recTCMETBranch_= tree_->GetBranch(recTCMETBranchName.c_str()); 
    if(!recTCMETBranch_) {
      cerr<<"PFRootEventManager::ReadOptions :recTCMETBranch_ not found : "
          <<recTCMETBranchName<< endl;
    }
  }
  string genParticlesforMETBranchName; 
  options_->GetOpt("root","genParticlesforMETBranchName", genParticlesforMETBranchName);
  if(!genParticlesforMETBranchName.empty() ) {
    genParticlesforMETBranch_= tree_->GetBranch(genParticlesforMETBranchName.c_str()); 
    if(!genParticlesforMETBranch_) {
      cerr<<"PFRootEventManager::ReadOptions :genParticlesforMETBranch_ not found : "
          <<genParticlesforMETBranchName<< endl;
    }
  }

  setAddresses();

}




void PFRootEventManager::setAddresses() {
  if( rechitsECALBranch_ ) rechitsECALBranch_->SetAddress(&rechitsECAL_);
  if( rechitsHCALBranch_ ) rechitsHCALBranch_->SetAddress(&rechitsHCAL_);
  if( rechitsHFEMBranch_ ) rechitsHFEMBranch_->SetAddress(&rechitsHFEM_);
  if( rechitsHFHADBranch_ ) rechitsHFHADBranch_->SetAddress(&rechitsHFHAD_);
  if( rechitsPSBranch_ ) rechitsPSBranch_->SetAddress(&rechitsPS_);
  if( clustersECALBranch_ ) clustersECALBranch_->SetAddress( clustersECAL_.get() );
  if( clustersHCALBranch_ ) clustersHCALBranch_->SetAddress( clustersHCAL_.get() );
  if( clustersPSBranch_ ) clustersPSBranch_->SetAddress( clustersPS_.get() );
//   if( clustersIslandBarrelBranch_ ) 
//     clustersIslandBarrelBranch_->SetAddress(&clustersIslandBarrel_);
  if( primaryVertexBranch_ ) primaryVertexBranch_->SetAddress(&primaryVertices_);
  if( recTracksBranch_ ) recTracksBranch_->SetAddress(&recTracks_);
  if( stdTracksBranch_ ) stdTracksBranch_->SetAddress(&stdTracks_);
  if( gsfrecTracksBranch_ ) gsfrecTracksBranch_->SetAddress(&gsfrecTracks_); 
  if( muonsBranch_ ) muonsBranch_->SetAddress(&muons_); 
  if( nuclearBranch_ ) nuclearBranch_->SetAddress(&nuclear_); 
  if( conversionBranch_ ) conversionBranch_->SetAddress(&conversion_); 
  if( v0Branch_ ) v0Branch_->SetAddress(&v0_);

  if( trueParticlesBranch_ ) trueParticlesBranch_->SetAddress(&trueParticles_);
  if( MCTruthBranch_ ) { 
    MCTruthBranch_->SetAddress(&MCTruth_);
  }
  if( caloTowersBranch_ ) caloTowersBranch_->SetAddress(&caloTowers_);
  if( genParticlesforJetsBranch_ ) 
    genParticlesforJetsBranch_->SetAddress(&genParticlesforJets_);
//   if( caloTowerBaseCandidatesBranch_ ) {
//     caloTowerBaseCandidatesBranch_->SetAddress(&caloTowerBaseCandidates_);
//   }
  if (genJetBranch_) genJetBranch_->SetAddress(&genJetsCMSSW_);
  if (recCaloBranch_) recCaloBranch_->SetAddress(&caloJetsCMSSW_);
  if (reccorrCaloBranch_) reccorrCaloBranch_->SetAddress(&corrcaloJetsCMSSW_);
  if (recPFBranch_) recPFBranch_->SetAddress(&pfJetsCMSSW_); 

  if (recCaloMETBranch_) recCaloMETBranch_->SetAddress(&caloMetsCMSSW_);
  if (recTCMETBranch_) recTCMETBranch_->SetAddress(&tcMetsCMSSW_);
  if (recPFMETBranch_) recPFMETBranch_->SetAddress(&pfMetsCMSSW_); 
  if (genParticlesforMETBranch_) genParticlesforMETBranch_->SetAddress(&genParticlesCMSSW_); 
}


PFRootEventManager::~PFRootEventManager() {

  if(outFile_) {
    outFile_->Close();
  }

  if(outEvent_) delete outEvent_;

  delete options_;

}


void PFRootEventManager::write() {

  if(doPFJetBenchmark_) PFJetBenchmark_.write();
  if(doPFMETBenchmark_) metManager_->write();

  if(!outFile_) return;
  else {
    outFile_->cd(); 
    // write histos here
    cout<<"writing output to "<<outFile_->GetName()<<endl;
    h_deltaETvisible_MCEHT_->Write();
    h_deltaETvisible_MCPF_->Write();
    if(outTree_) outTree_->Write();
  }
}


bool PFRootEventManager::processEntry(int entry) {

  reset();

  iEvent_ = entry;
 
  if( outEvent_ ) outEvent_->setNumber(entry);

  if(verbosity_ == VERBOSE  || 
     // entry < 3000 ||
     entry < 100 && entry%10 == 0 || 
     entry < 1000 && entry%100 == 0 || 
     entry%1000 == 0 ) 
    cout<<"process entry "<< entry << endl;
  
  bool goodevent =  readFromSimulation(entry);

  if(verbosity_ == VERBOSE ) {
    cout<<"number of vertices       : "<<primaryVertices_.size()<<endl;
    cout<<"number of recTracks      : "<<recTracks_.size()<<endl;
    cout<<"number of gsfrecTracks   : "<<gsfrecTracks_.size()<<endl;
    cout<<"number of muons          : "<<muons_.size()<<endl;
    cout<<"number of nuclear ints   : "<<nuclear_.size()<<endl;
    cout<<"number of conversions    : "<<conversion_.size()<<endl;
    cout<<"number of v0             : "<<v0_.size()<<endl;
    cout<<"number of stdTracks      : "<<stdTracks_.size()<<endl;
    cout<<"number of true particles : "<<trueParticles_.size()<<endl;
    cout<<"number of ECAL rechits   : "<<rechitsECAL_.size()<<endl;
    cout<<"number of HCAL rechits   : "<<rechitsHCAL_.size()<<endl;
    cout<<"number of HFEM rechits   : "<<rechitsHFEM_.size()<<endl;
    cout<<"number of HFHAD rechits   : "<<rechitsHFHAD_.size()<<endl;
    cout<<"number of PS rechits     : "<<rechitsPS_.size()<<endl;
  }  

  if( doClustering_ ) clustering(); 
  else if( verbosity_ == VERBOSE )
    cout<<"clustering is OFF - clusters come from the input file"<<endl; 

  if(verbosity_ == VERBOSE ) {
    if(clustersECAL_.get() ) {
      cout<<"number of ECAL clusters : "<<clustersECAL_->size()<<endl;
    }
    if(clustersHCAL_.get() ) {
      cout<<"number of HCAL clusters : "<<clustersHCAL_->size()<<endl;
    }
    if(clustersHFEM_.get() ) {
      cout<<"number of HFEM clusters : "<<clustersHFEM_->size()<<endl;
    }
    if(clustersHFHAD_.get() ) {
      cout<<"number of HFHAD clusters : "<<clustersHFHAD_->size()<<endl;
    }
    if(clustersPS_.get() ) {
      cout<<"number of PS clusters : "<<clustersPS_->size()<<endl;
    }
  }

  
  if(doParticleFlow_) particleFlow();

  if(doJets_) {
    reconstructGenJets();
    reconstructCaloJets();
    reconstructPFJets();
  }    
        
 
  // call print() in verbose mode
  if( verbosity_ == VERBOSE ) print();
  
  // evaluate PFJet Benchmark 
  
  if(doPFJetBenchmark_) { // start PFJet Benchmark

          
    PFJetBenchmark_.process(pfJets_, genJets_);
    double resPt = PFJetBenchmark_.resPtMax();
    double resChargedHadEnergy = PFJetBenchmark_.resChargedHadEnergyMax();
    double resNeutralHadEnergy = PFJetBenchmark_.resNeutralHadEnergyMax();
    double resNeutralEmEnergy = PFJetBenchmark_.resNeutralEmEnergyMax();
          
    if( verbosity_ == VERBOSE ){ //start debug print

      cout << " =====================PFJetBenchmark =================" << endl;
      cout<<"Resol Pt max "<<resPt
          <<" resChargedHadEnergy Max " << resChargedHadEnergy
          <<" resNeutralHadEnergy Max " << resNeutralHadEnergy
          << " resNeutralEmEnergy Max "<< resNeutralEmEnergy << endl;
    } // end debug print

    // PJ : printout for bad events (selected by the "if")
    if ( resPt < -1. ) { 
      cout << " =====================PFJetBenchmark =================" << endl;
      cout<<"process entry "<< entry << endl;
      cout<<"Resol Pt max "<<resPt
	  <<" resChargedHadEnergy Max " << resChargedHadEnergy
	  <<" resNeutralHadEnergy Max " << resNeutralHadEnergy
	  << " resNeutralEmEnergy Max "<< resNeutralEmEnergy 
	  << " Jet pt " << genJets_[0].pt() << endl;
      // return true;
    } else { 
      // return false;
    }
    //   if (resNeutralEmEnergy>0.5) return true;
    //   else return false;
  }// end PFJet Benchmark

  if(doPFMETBenchmark_) { // start PFMet Benchmark

    // Fill here the various met benchmarks
    // pfMET vs GenMET
    metManager_->setMET1(&genParticlesCMSSW_);
    metManager_->setMET2(&pfMetsCMSSW_[0]);
    metManager_->FillHisto("PF");
    // cout events in tail
    metManager_->coutTailEvents(entry,DeltaMETcut,DeltaPhicut, MET1cut);

    // caloMET vs GenMET
    metManager_->setMET2(&caloMetsCMSSW_[0]);
    metManager_->FillHisto("Calo");

    if ( doMet_ ) { 
      // recomputed pfMET vs GenMET
      metManager_->setMET2(*pfCandidates_);
      metManager_->FillHisto("recompPF");
      metManager_->coutTailEvents(entry,DeltaMETcut,DeltaPhicut, MET1cut);
    }

    if (JECinCaloMet_)
    {
      // corrCaloMET vs GenMET
      metManager_->setMET2(&caloMetsCMSSW_[0]);
      metManager_->propagateJECtoMET2(caloJetsCMSSW_, corrcaloJetsCMSSW_);
      metManager_->FillHisto("corrCalo");
    }
  }// end PFMET Benchmark
    
  // evaluate tau Benchmark   
  if( goodevent && doTauBenchmark_) { // start tau Benchmark
    double deltaEt = 0.;
    deltaEt  = tauBenchmark( *pfCandidates_ ); 
    if( verbosity_ == VERBOSE ) cout<<"delta E_t ="<<deltaEt <<endl;
    //      cout<<"delta E_t ="<<deltaEt<<" delta E_t Other ="<<deltaEt1<<endl;


    //   if( deltaEt>0.4 ) {
    //     cout<<deltaEt<<endl;
    //     return true;
    //   }  
    //   else return false;

  
  } // end tau Benchmark
  
  if(goodevent && outTree_) 
    outTree_->Fill();

  if(calibFile_)
    printMCCalib(*calibFile_);
  
  return goodevent;

}



bool PFRootEventManager::readFromSimulation(int entry) {

  if (verbosity_ == VERBOSE ) {
    cout <<"start reading from simulation"<<endl;
  }


  if(!tree_) return false;
  
  setAddresses();

  if(stdTracksBranch_) { 
    stdTracksBranch_->GetEntry(entry);
  }
  if(MCTruthBranch_) { 
    MCTruthBranch_->GetEntry(entry);
  }
  if(trueParticlesBranch_ ) {
    trueParticlesBranch_->GetEntry(entry);
  }
  if(rechitsECALBranch_) {
    rechitsECALBranch_->GetEntry(entry);
  }
  if(rechitsHCALBranch_) {
    rechitsHCALBranch_->GetEntry(entry);
  }
  if(rechitsHFEMBranch_) {
    rechitsHFEMBranch_->GetEntry(entry);
  }
  if(rechitsHFHADBranch_) {
    rechitsHFHADBranch_->GetEntry(entry);
  }
  if(rechitsPSBranch_) {
    rechitsPSBranch_->GetEntry(entry);  
  }
  if(clustersECALBranch_ && !doClustering_) {
    clustersECALBranch_->GetEntry(entry);
  }
  if(clustersHCALBranch_ && !doClustering_) {
    clustersHCALBranch_->GetEntry(entry);
  }
  if(clustersPSBranch_ && !doClustering_) {
    clustersPSBranch_->GetEntry(entry);
  }
  if(clustersIslandBarrelBranch_) {
    clustersIslandBarrelBranch_->GetEntry(entry);
  }
  if(caloTowersBranch_) {
    caloTowersBranch_->GetEntry(entry);
  } 
  if(primaryVertexBranch_) {
    primaryVertexBranch_->GetEntry(entry);
  }
  if(recTracksBranch_) {
    recTracksBranch_->GetEntry(entry);
  }
  if(gsfrecTracksBranch_) {
    gsfrecTracksBranch_->GetEntry(entry);
  }
  if(muonsBranch_) {
    muonsBranch_->GetEntry(entry);
  }
  if(nuclearBranch_) {
    nuclearBranch_->GetEntry(entry);
  }
  if(conversionBranch_) {
    conversionBranch_->GetEntry(entry);
  }
  if(v0Branch_) {
    v0Branch_->GetEntry(entry);
  }

  if(genParticlesforJetsBranch_) {
    genParticlesforJetsBranch_->GetEntry(entry);
  }
//   if(caloTowerBaseCandidatesBranch_) {
//     caloTowerBaseCandidatesBranch_->GetEntry(entry);
//   }
  if(genJetBranch_) {
    genJetBranch_->GetEntry(entry);
  }
  if(recCaloBranch_) {
    recCaloBranch_->GetEntry(entry);
  }
  if(reccorrCaloBranch_) {
    reccorrCaloBranch_->GetEntry(entry);
  }
  if(recPFBranch_) {
    recPFBranch_->GetEntry(entry);
  }

  tree_->GetEntry( entry, 0 );

  // now can use the tree

  bool goodevent = true;
  if(trueParticlesBranch_ ) {
    // this is a filter to select single particle events.
    if(filterNParticles_ && doTauBenchmark_ &&
       trueParticles_.size() != filterNParticles_ ) {
      cout << "PFRootEventManager : event discarded Nparticles="
           <<filterNParticles_<< endl; 
      goodevent = false;
    }
    if(goodevent && doTauBenchmark_ && filterHadronicTaus_ && !isHadronicTau() ) {
      cout << "PFRootEventManager : leptonic tau discarded " << endl; 
      goodevent =  false;
    }
    if( goodevent && doTauBenchmark_ && !filterTaus_.empty() 
        && !countChargedAndPhotons() ) {
      assert( filterTaus_.size() == 2 );
      cout <<"PFRootEventManager : tau discarded: "
           <<"number of charged and neutral particles different from "
           <<filterTaus_[0]<<","<<filterTaus_[1]<<endl;
      goodevent =  false;      
    } 
    
    if(goodevent)
      fillOutEventWithSimParticles( trueParticles_ );

  }
  
  //   if(caloTowersBranch_) {
  //     if(goodevent)
  //       fillOutEventWithCaloTowers( caloTowers_ );
  //   } 

  if(rechitsECALBranch_) {
    PreprocessRecHits( rechitsECAL_ , findRecHitNeighbours_);
  }
  if(rechitsHCALBranch_) {
    PreprocessRecHits( rechitsHCAL_ , findRecHitNeighbours_);
  }
  if(rechitsHFEMBranch_) {
    PreprocessRecHits( rechitsHFEM_ , findRecHitNeighbours_);
  }
  if(rechitsHFHADBranch_) {
    PreprocessRecHits( rechitsHFHAD_ , findRecHitNeighbours_);
  }
  if(rechitsPSBranch_) {
    PreprocessRecHits( rechitsPS_ , findRecHitNeighbours_);
  }

  if ( recTracksBranch_ ) { 
    PreprocessRecTracks( recTracks_);
  }

  if(gsfrecTracksBranch_) {
    PreprocessRecTracks( gsfrecTracks_);
  }
   
  //   if(clustersECALBranch_ && !doClustering_) {
  //     for(unsigned i=0; i<clustersECAL_->size(); i++) 
  //       (*clustersECAL_)[i].calculatePositionREP();
  //   }
  //   if(clustersHCALBranch_ && !doClustering_) {
  //     for(unsigned i=0; i<clustersHCAL_->size(); i++) 
  //       (*clustersHCAL_)[i].calculatePositionREP();    
  //   }
  //   if(clustersPSBranch_ && !doClustering_) {
  //     for(unsigned i=0; i<clustersPS_->size(); i++) 
  //       (*clustersPS_)[i].calculatePositionREP();    
  //   }

  return goodevent;
}


bool PFRootEventManager::isHadronicTau() const {

  for ( unsigned i=0;  i < trueParticles_.size(); i++) {
    const reco::PFSimParticle& ptc = trueParticles_[i];
    const std::vector<int>& ptcdaughters = ptc.daughterIds();
    if (abs(ptc.pdgCode()) == 15) {
      for ( unsigned int dapt=0; dapt < ptcdaughters.size(); ++dapt) {
        
        const reco::PFSimParticle& daughter 
          = trueParticles_[ptcdaughters[dapt]];
        

        int pdgdaugther = daughter.pdgCode();
        int abspdgdaughter = abs(pdgdaugther);


        if (abspdgdaughter == 11 || 
            abspdgdaughter == 13) { 
          return false; 
        }//electron or muons?
      }//loop daughter
    }//tau
  }//loop particles


  return true;
}



bool PFRootEventManager::countChargedAndPhotons() const {
  
  int nPhoton = 0;
  int nCharged = 0;
  
  for ( unsigned i=0;  i < trueParticles_.size(); i++) {
    const reco::PFSimParticle& ptc = trueParticles_[i];
   
    const std::vector<int>& daughters = ptc.daughterIds();

    // if the particle decays before ECAL, we do not want to 
    // consider it.
    if(!daughters.empty() ) continue; 

    double charge = ptc.charge();
    double pdgCode = ptc.pdgCode();
    
    if( abs(charge)>1e-9) 
      nCharged++;
    else if( pdgCode==22 )
      nPhoton++;
  }    

  //   const HepMC::GenEvent* myGenEvent = MCTruth_.GetEvent();
  //   if(!myGenEvent) {
  //     cerr<<"impossible to filter on the number of charged and "
  //    <<"neutral particles without the HepMCProduct. "
  //    <<"Please check that the branch edmHepMCProduct_*_*_* is found"<<endl;
  //     exit(1);
  //   }
  
  //   for ( HepMC::GenEvent::particle_const_iterator 
  //      piter  = myGenEvent->particles_begin();
  //    piter != myGenEvent->particles_end(); 
  //    ++piter ) {
    
  //     const HepMC::GenParticle* p = *piter;
  //     int partId = p->pdg_id();Long64_t lines = T->ReadFile("mon_fichier","i:j:k:x:y:z");
    
  // //     pdgTable_->GetParticle( partId )->Print();
       
  //     int charge = chargeValue(partId);
  //     cout<<partId <<" "<<charge/3.<<endl;

  //     if(charge) 
  //       nCharged++;
  //     else 
  //       nNeutral++;
  //   }
  
  if( nCharged == filterTaus_[0] && 
      nPhoton == filterTaus_[1]  )
    return true;
  else 
    return false;
}



int PFRootEventManager::chargeValue(const int& Id) const {

  
  //...Purpose: to give three times the charge for a particle/parton.

  //      ID     = particle ID
  //      hepchg = particle charge times 3

  int kqa,kq1,kq2,kq3,kqj,irt,kqx,kqn;
  int hepchg;


  int ichg[109]={-1,2,-1,2,-1,2,-1,2,0,0,-3,0,-3,0,-3,0,
                 -3,0,0,0,0,0,0,3,0,0,0,0,0,0,3,0,3,6,0,0,3,6,0,0,-1,2,-1,2,-1,2,0,0,0,0,
                 -3,0,-3,0,-3,0,0,0,0,0,-1,2,-1,2,-1,2,0,0,0,0,
                 -3,0,-3,0,-3,0,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


  //...Initial values. Simple case of direct readout.
  hepchg=0;
  kqa=abs(Id);
  kqn=kqa/1000000000%10;
  kqx=kqa/1000000%10;
  kq3=kqa/1000%10;
  kq2=kqa/100%10;
  kq1=kqa/10%10;
  kqj=kqa%10;
  irt=kqa%10000;

  //...illegal or ion
  //...set ion charge to zero - not enough information
  if(kqa==0 || kqa >= 10000000) {

    if(kqn==1) {hepchg=0;}
  }
  //... direct translation
  else if(kqa<=100) {hepchg = ichg[kqa-1];}
  //... deuteron or tritium
  else if(kqa==100 || kqa==101) {hepchg = -3;}
  //... alpha or He3
  else if(kqa==102 || kqa==104) {hepchg = -6;}
  //... KS and KL (and undefined)
  else if(kqj == 0) {hepchg = 0;}
  //C... direct translation
  else if(kqx>0 && irt<100)
    {
      hepchg = ichg[irt-1];
      if(kqa==1000017 || kqa==1000018) {hepchg = 0;}
      if(kqa==1000034 || kqa==1000052) {hepchg = 0;}
      if(kqa==1000053 || kqa==1000054) {hepchg = 0;}
      if(kqa==5100061 || kqa==5100062) {hepchg = 6;}
    }
  //...Construction from quark content for heavy meson, diquark, baryon.
  //...Mesons.
  else if(kq3==0)
    {
      hepchg = ichg[kq2-1]-ichg[kq1-1];
      //...Strange or beauty mesons.
      if((kq2==3) || (kq2==5)) {hepchg = ichg[kq1-1]-ichg[kq2-1];}
    }
  else if(kq1 == 0) {
    //...Diquarks.
    hepchg = ichg[kq3-1] + ichg[kq2-1];
  }

  else{
    //...Baryons
    hepchg = ichg[kq3-1]+ichg[kq2-1]+ichg[kq1-1];
  }

  //... fix sign of charge
  if(Id<0 && hepchg!=0) {hepchg = -1*hepchg;}

  // cout << hepchg<< endl;
  return hepchg;
}



void 
PFRootEventManager::PreprocessRecTracks(reco::PFRecTrackCollection& recTracks) {  
  for( unsigned i=0; i<recTracks.size(); ++i ) {     
    recTracks[i].calculatePositionREP();
  }
}

void 
PFRootEventManager::PreprocessRecTracks(reco::GsfPFRecTrackCollection& recTracks) {  
  for( unsigned i=0; i<recTracks.size(); ++i ) {     
    recTracks[i].calculatePositionREP();
    recTracks[i].calculateBremPositionREP();
  }
}


 
void 
PFRootEventManager::PreprocessRecHits(reco::PFRecHitCollection& rechits, 
                                      bool findNeighbours) {
  
 
  map<unsigned, unsigned> detId2index;

  for(unsigned i=0; i<rechits.size(); i++) { 
    rechits[i].calculatePositionREP();
    
    if(findNeighbours) 
      detId2index.insert( make_pair(rechits[i].detId(), i) );
  }
  
  if(findNeighbours) {
    for(unsigned i=0; i<rechits.size(); i++) { 
      setRecHitNeigbours( rechits[i], detId2index );
    }
  }
}


void PFRootEventManager::setRecHitNeigbours
( reco::PFRecHit& rh, 
  const map<unsigned, unsigned>& detId2index ) {

  rh.clearNeighbours();

  vector<unsigned> neighbours4DetId = rh.neighboursIds4();
  vector<unsigned> neighbours8DetId = rh.neighboursIds8();
  
  for( unsigned i=0; i<neighbours4DetId.size(); i++) {
    unsigned detId = neighbours4DetId[i];
    //     cout<<"finding n for detId "<<detId<<endl;
    const map<unsigned, unsigned>::const_iterator& it = detId2index.find(detId);
    
    if(it != detId2index.end() ) {
      //       cout<<"found n index "<<it->second<<endl;
      rh.add4Neighbour( it->second );
    }
  }

  for( unsigned i=0; i<neighbours8DetId.size(); i++) {
    unsigned detId = neighbours8DetId[i];
    //     cout<<"finding n for detId "<<detId<<endl;
    const map<unsigned, unsigned>::const_iterator& it = detId2index.find(detId);
    
    if(it != detId2index.end() ) {
      //       cout<<"found n index "<<it->second<<endl;
      rh.add8Neighbour( it->second );
    }
  }

  
}


void PFRootEventManager::clustering() {

  if (verbosity_ == VERBOSE ) {
    cout <<"start clustering"<<endl;
  }
  
  // ECAL clustering -------------------------------------------

  vector<bool> mask;
  fillRecHitMask( mask, rechitsECAL_ );
  clusterAlgoECAL_.setMask( mask );  

  //   edm::OrphanHandle< reco::PFRecHitCollection > rechitsHandleECAL( &rechitsECAL_, edm::ProductID(10001) );
  clusterAlgoECAL_.doClustering( rechitsECAL_ );
  clustersECAL_ = clusterAlgoECAL_.clusters();

  assert(clustersECAL_.get() );

  fillOutEventWithClusters( *clustersECAL_ );

  // HCAL clustering -------------------------------------------

  fillRecHitMask( mask, rechitsHCAL_ );
  clusterAlgoHCAL_.setMask( mask );  
  //   edm::OrphanHandle< reco::PFRecHitCollection > rechitsHandleHCAL( &rechitsHCAL_, edm::ProductID(10002) );
  clusterAlgoHCAL_.doClustering( rechitsHCAL_ );
  clustersHCAL_ = clusterAlgoHCAL_.clusters();

  fillOutEventWithClusters( *clustersHCAL_ );

  // HF clustering -------------------------------------------

  clusterAlgoHFEM_.doClustering( rechitsHFEM_ );
  clustersHFEM_ = clusterAlgoHFEM_.clusters();
  
  clusterAlgoHFHAD_.doClustering( rechitsHFHAD_ );
  clustersHFHAD_ = clusterAlgoHFHAD_.clusters();
  

  // PS clustering -------------------------------------------

  fillRecHitMask( mask, rechitsPS_ );
  clusterAlgoPS_.setMask( mask );  
  //   edm::OrphanHandle< reco::PFRecHitCollection > rechitsHandlePS( &rechitsPS_, edm::ProductID(10003) );
  clusterAlgoPS_.doClustering( rechitsPS_ );
  clustersPS_ = clusterAlgoPS_.clusters();

  fillOutEventWithClusters( *clustersPS_ );
  
}



void 
PFRootEventManager::fillOutEventWithClusters(const reco::PFClusterCollection& 
                                             clusters) {

  if(!outEvent_) return;
  
  for(unsigned i=0; i<clusters.size(); i++) {
    EventColin::Cluster cluster;
    cluster.eta = clusters[i].position().Eta();
    cluster.phi = clusters[i].position().Phi();
    cluster.e = clusters[i].energy();
    cluster.layer = clusters[i].layer();
    cluster.type = 1;

    reco::PFTrajectoryPoint::LayerType tpLayer = 
      reco::PFTrajectoryPoint::NLayers;
    switch( clusters[i].layer() ) {
    case PFLayer::ECAL_BARREL:
    case PFLayer::ECAL_ENDCAP:
      tpLayer = reco::PFTrajectoryPoint::ECALEntrance;
      break;
    case PFLayer::HCAL_BARREL1:
    case PFLayer::HCAL_ENDCAP:
      tpLayer = reco::PFTrajectoryPoint::HCALEntrance;
      break;
    default:
      break;
    }
    if(tpLayer < reco::PFTrajectoryPoint::NLayers) {
      try {
        double peta = -10;
        double phi = -10;
        double pe = -10;
	
        const reco::PFSimParticle& ptc 
          = closestParticle( tpLayer, 
                             cluster.eta, cluster.phi, 
                             peta, phi, pe );

        
        cluster.particle.eta = peta;
        cluster.particle.phi = phi;
        cluster.particle.e = pe;
        cluster.particle.pdgCode = ptc.pdgCode();
        
        
      }
      catch( std::exception& err ) {
        // cerr<<err.what()<<endl;
      } 
    }

    outEvent_->addCluster(cluster);
  }   
}


void 
PFRootEventManager::fillOutEventWithSimParticles(const reco::PFSimParticleCollection& trueParticles ) {

  if(!outEvent_) return;
  
  for ( unsigned i=0;  i < trueParticles.size(); i++) {

    const reco::PFSimParticle& ptc = trueParticles[i];

    unsigned ntrajpoints = ptc.nTrajectoryPoints();
    
    if(ptc.daughterIds().empty() ) { // stable
      reco::PFTrajectoryPoint::LayerType ecalEntrance 
        = reco::PFTrajectoryPoint::ECALEntrance;

      if(ntrajpoints == 3) { 
        // old format for PFSimCandidates. 
        // in this case, the PFSimCandidate which does not decay 
        // before ECAL has 3 points: initial, ecal entrance, hcal entrance
        ecalEntrance = static_cast<reco::PFTrajectoryPoint::LayerType>(1);
      }
      // else continue; // endcap case we do not care;

      const reco::PFTrajectoryPoint& tpatecal 
        = ptc.extrapolatedPoint( ecalEntrance );
        
      EventColin::Particle outptc;
      outptc.eta = tpatecal.position().Eta();
      outptc.phi = tpatecal.position().Phi();    
      outptc.e = tpatecal.momentum().E();
      outptc.pdgCode = ptc.pdgCode();
    
      
      outEvent_->addParticle(outptc);
    }  
  }   
}      

void 
PFRootEventManager::fillOutEventWithPFCandidates(const reco::PFCandidateCollection& pfCandidates ) {

  if(!outEvent_) return;
  
  for ( unsigned i=0;  i < pfCandidates.size(); i++) {

    const reco::PFCandidate& candidate = pfCandidates[i];
    
    EventColin::Particle outptc;
    outptc.eta = candidate.eta();
    outptc.phi = candidate.phi();    
    outptc.e = candidate.energy();
    outptc.pdgCode = candidate.particleId();
    
    
    outEvent_->addCandidate(outptc);  
  }   
}      


void 
PFRootEventManager::fillOutEventWithCaloTowers( const CaloTowerCollection& cts ){

  if(!outEvent_) return;
  
  for ( unsigned i=0;  i < cts.size(); i++) {

    const CaloTower& ct = cts[i];
    
    EventColin::CaloTower outct;
    outct.e  = ct.energy();
    outct.ee = ct.emEnergy();
    outct.eh = ct.hadEnergy();

    outEvent_->addCaloTower( outct );
  }
}


void 
PFRootEventManager::fillOutEventWithBlocks( const reco::PFBlockCollection& 
                                            blocks ) {

  if(!outEvent_) return;
  
  for ( unsigned i=0;  i < blocks.size(); i++) {

    //    const reco::PFBlock& block = blocks[i];
    
    EventColin::Block outblock;
 
    outEvent_->addBlock( outblock );
  }
}



void PFRootEventManager::particleFlow() {
  
  if (verbosity_ == VERBOSE ) {
    cout <<"start particle flow"<<endl;
  }


  if( debug_) {
    cout<<"PFRootEventManager::particleFlow start"<<endl;
    //     cout<<"number of elements in memory: "
    //  <<reco::PFBlockElement::instanceCounter()<<endl;
  }


  edm::OrphanHandle< reco::PFRecTrackCollection > trackh( &recTracks_, 
                                                          edm::ProductID(1) );  
  
  edm::OrphanHandle< reco::PFClusterCollection > ecalh( clustersECAL_.get(), 
                                                        edm::ProductID(2) );  
  
  edm::OrphanHandle< reco::PFClusterCollection > hcalh( clustersHCAL_.get(), 
                                                        edm::ProductID(3) );  

  edm::OrphanHandle< reco::PFClusterCollection > hfemh( clustersHFEM_.get(), 
                                                        edm::ProductID(31) );  

  edm::OrphanHandle< reco::PFClusterCollection > hfhadh( clustersHFHAD_.get(), 
                                                        edm::ProductID(32) );  

  edm::OrphanHandle< reco::PFClusterCollection > psh( clustersPS_.get(), 
                                                      edm::ProductID(4) );   

  edm::OrphanHandle< reco::GsfPFRecTrackCollection > gsftrackh( &gsfrecTracks_, 
                                                          edm::ProductID(5) );  

  edm::OrphanHandle< reco::MuonCollection > muonh( &muons_, 
						   edm::ProductID(6) );

  edm::OrphanHandle< reco::PFNuclearInteractionCollection > nuclh( &nuclear_, 
                                                          edm::ProductID(7) );

  edm::OrphanHandle< reco::PFConversionCollection > convh( &conversion_, 
							   edm::ProductID(8) );

  edm::OrphanHandle< reco::PFV0Collection > v0( &v0_, 
						edm::ProductID(9) );

  edm::OrphanHandle< reco::VertexCollection > vertexh( &primaryVertices_, 
						       edm::ProductID(10) );  
  
  vector<bool> trackMask;
  fillTrackMask( trackMask, recTracks_ );
  vector<bool> gsftrackMask;
  fillTrackMask( gsftrackMask, gsfrecTracks_ );
  vector<bool> ecalMask;
  fillClusterMask( ecalMask, *clustersECAL_ );
  vector<bool> hcalMask;
  fillClusterMask( hcalMask, *clustersHCAL_ );
  vector<bool> psMask;
  fillClusterMask( psMask, *clustersPS_ );
  
  pfBlockAlgo_.setInput( trackh, gsftrackh, 
			 muonh,nuclh,convh,v0,
			 ecalh, hcalh, hfemh, hfhadh, psh,
			 trackMask,gsftrackMask,
			 ecalMask, hcalMask, psMask );

  pfBlockAlgo_.findBlocks();
  
  if( debug_) cout<<pfBlockAlgo_<<endl;

  pfBlocks_ = pfBlockAlgo_.transferBlocks();

  pfAlgo_.setPFVertexParameters(true, primaryVertices_); 

  pfAlgo_.reconstructParticles( *pfBlocks_.get() );
  //   pfAlgoOther_.reconstructParticles( blockh );
  if( debug_) cout<< pfAlgo_<<endl;
  pfCandidates_ = pfAlgo_.transferCandidates();
  //   pfCandidatesOther_ = pfAlgoOther_.transferCandidates();
  
  fillOutEventWithPFCandidates( *pfCandidates_ );

  if( debug_) cout<<"PFRootEventManager::particleFlow stop"<<endl;
}



void PFRootEventManager::reconstructGenJets() {

  if (verbosity_ == VERBOSE || jetsDebug_) {
    cout<<endl;
    cout<<"start reconstruct GenJets  --- "<<endl;
    cout<< " input gen particles for jet: all neutrinos removed ; muons present" << endl;
  }

  genJets_.clear();
  genParticlesforJetsPtrs_.clear();

  for(unsigned i=0; i<genParticlesforJets_.size(); i++) {

    const reco::GenParticle&    genPart = *(genParticlesforJets_[i]);

    // remove all muons/neutrinos for PFJet studies
    //    if (reco::isNeutrino( genPart ) || reco::isMuon( genPart )) continue;
    //    remove all neutrinos for PFJet studies
    if (reco::isNeutrino( genPart )) continue;
    // Work-around a bug in the pythia di-jet gun.
    if (abs(genPart.pdgId())<7 || abs(genPart.pdgId())==21 ) continue;

    if (jetsDebug_ ) {
      cout << "      #" << i << "  PDG code:" << genPart.pdgId() 
	   << " status " << genPart.status()
	   << ", p/pt/eta/phi: " << genPart.p() << '/' << genPart.pt() 
	   << '/' << genPart.eta() << '/' << genPart.phi() << endl;
    }
    
    genParticlesforJetsPtrs_.push_back( refToPtr(genParticlesforJets_[i]) );
  }
  
  vector<ProtoJet> protoJets;
  reconstructFWLiteJets(genParticlesforJetsPtrs_, protoJets );


  // Convert Protojets to GenJets
    int ijet = 0;
  typedef vector <ProtoJet>::const_iterator IPJ;
  for  (IPJ ipj = protoJets.begin(); ipj != protoJets.end (); ipj++) {
    const ProtoJet& protojet = *ipj;
    const ProtoJet::Constituents& constituents = protojet.getTowerList();
          
    reco::Jet::Point vertex (0,0,0); // do not have true vertex yet, use default
    GenJet::Specific specific;
    JetMaker::makeSpecific(constituents, &specific);
    // constructor without constituents
    GenJet newJet (protojet.p4(), vertex, specific);
          
    // last step is to copy the constituents into the jet (new jet definition since 18X)
    // namespace reco {
    //class Jet : public CompositeRefBaseCandidate {
    // public:
    //  typedef reco::CandidateBaseRefVector Constituents;
          
    ProtoJet::Constituents::const_iterator constituent = constituents.begin();
    for (; constituent != constituents.end(); ++constituent) {
      // find index of this ProtoJet constituent in the overall collection PFconstit
      // see class IndexedCandidate in JetRecoTypes.h
      uint index = constituent->index();
      newJet.addDaughter( genParticlesforJetsPtrs_[index] );
    }  // end loop on ProtoJet constituents
    // last step: copy ProtoJet Variables into Jet
    newJet.setJetArea(protojet.jetArea()); 
    newJet.setPileup(protojet.pileup());
    newJet.setNPasses(protojet.nPasses());
    ++ijet;
    if (jetsDebug_ ) cout<<" gen jet "<<ijet<<" "<<newJet.print()<<endl;
    genJets_.push_back (newJet);
          
  } // end loop on protojets iterator IPJ
  
}

void PFRootEventManager::reconstructCaloJets() {

  if (verbosity_ == VERBOSE || jetsDebug_ ) {
    cout<<endl;
    cout<<"start reconstruct CaloJets --- "<<endl;
  }
  caloJets_.clear();
  caloTowersPtrs_.clear();

  for( unsigned i=0; i<caloTowers_.size(); i++) {
    reco::CandidatePtr candPtr( &caloTowers_, i );
    caloTowersPtrs_.push_back( candPtr );
  }
 
  reconstructFWLiteJets( caloTowersPtrs_, caloJets_ );

  if (jetsDebug_ ) {
    for(unsigned ipj=0; ipj<caloJets_.size(); ipj++) {
      const ProtoJet& protojet = caloJets_[ipj];      
      cout<<" calo jet "<<ipj<<" "<<protojet.pt() <<endl;
    }
  }

}


void PFRootEventManager::reconstructPFJets() {

  if (verbosity_ == VERBOSE || jetsDebug_) {
    cout<<endl;
    cout<<"start reconstruct PF Jets --- "<<endl;
  }
  pfJets_.clear();
  pfCandidatesPtrs_.clear();
        
  for( unsigned i=0; i<pfCandidates_->size(); i++) {
    reco::CandidatePtr candPtr( pfCandidates_.get(), i );
    pfCandidatesPtrs_.push_back( candPtr );
  }

  vector<ProtoJet> protoJets;
  reconstructFWLiteJets(pfCandidatesPtrs_, protoJets );

  // Convert Protojets to PFJets

  int ijet = 0;
  typedef vector <ProtoJet>::const_iterator IPJ;
  for  (IPJ ipj = protoJets.begin(); ipj != protoJets.end (); ipj++) {
    const ProtoJet& protojet = *ipj;
    const ProtoJet::Constituents& constituents = protojet.getTowerList();
        
    reco::Jet::Point vertex (0,0,0); // do not have true vertex yet, use default
    PFJet::Specific specific;
    JetMaker::makeSpecific(constituents, &specific);
    // constructor without constituents
    PFJet newJet (protojet.p4(), vertex, specific);
        
    // last step is to copy the constituents into the jet (new jet definition since 18X)
    // namespace reco {
    //class Jet : public CompositeRefBaseCandidate {
    // public:
    //  typedef reco::CandidateBaseRefVector Constituents;
        
    ProtoJet::Constituents::const_iterator constituent = constituents.begin();
    for (; constituent != constituents.end(); ++constituent) {
      // find index of this ProtoJet constituent in the overall collection PFconstit
      // see class IndexedCandidate in JetRecoTypes.h
      uint index = constituent->index();
      newJet.addDaughter(pfCandidatesPtrs_[index]);
    }  // end loop on ProtoJet constituents
    // last step: copy ProtoJet Variables into Jet
    newJet.setJetArea(protojet.jetArea()); 
    newJet.setPileup(protojet.pileup());
    newJet.setNPasses(protojet.nPasses());
    ++ijet;
    if (jetsDebug_ )  cout<<" PF jet "<<ijet<<" "<<newJet.print()<<endl;
    pfJets_.push_back (newJet);
        
  } // end loop on protojets iterator IPJ

}

void 
PFRootEventManager::reconstructFWLiteJets(const reco::CandidatePtrVector& Candidates, vector<ProtoJet>& output ) {

  // cout<<"!!! Make FWLite Jets  "<<endl;  
  JetReco::InputCollection input;
  // vector<ProtoJet> output;
  jetMaker_.applyCuts (Candidates, &input); 
  if (jetAlgoType_==1) {// ICone 
    /// Produce jet collection using CMS Iterative Cone Algorithm
    jetMaker_.makeIterativeConeJets(input, &output);
  }
  if (jetAlgoType_==2) {// MCone
    jetMaker_.makeMidpointJets(input, &output);
  }     
  if (jetAlgoType_==3) {// Fastjet
    jetMaker_.makeFastJets(input, &output);  
  }
  if((jetAlgoType_>3)||(jetAlgoType_<0)) {
    cout<<"Unknown Jet Algo ! " <<jetAlgoType_ << endl;
  }
  //if (jetsDebug_) cout<<"Proto Jet Size " <<output.size()<<endl;

}




double 
PFRootEventManager::tauBenchmark( const reco::PFCandidateCollection& candidates) {
  //std::cout << "building jets from MC particles," 
  //    << "PF particles and caloTowers" << std::endl;
  
  //initialize Jets Reconstruction
  jetAlgo_.Clear();

  //COLIN The following comment is not really adequate, 
  // since partTOTMC is not an action..
  // one should say what this variable is for.
  // see my comment later 
  //MAKING TRUE PARTICLE JETS
//   TLorentzVector partTOTMC;

  // colin: the following is not necessary
  // since the lorentz vectors are initialized to 0,0,0,0. 
  // partTOTMC.SetPxPyPzE(0.0, 0.0, 0.0, 0.0);

  //MAKING JETS WITH TAU DAUGHTERS
  //Colin: this vector vectPART is not necessary !!
  //it was just an efficient copy of trueparticles_.....
//   vector<reco::PFSimParticle> vectPART;
//   for ( unsigned i=0;  i < trueParticles_.size(); i++) {
//     const reco::PFSimParticle& ptc = trueParticles_[i];
//     vectPART.push_back(ptc);
//   }//loop


  //COLIN one must not loop on trueParticles_ to find taus. 
  //the code was giving wrong results on non single tau events. 

  // first check that this is a single tau event. 

  TLorentzVector partTOTMC;
  bool tauFound = false;
  bool tooManyTaus = false;
  if (fastsim_){

    for ( unsigned i=0;  i < trueParticles_.size(); i++) {
      const reco::PFSimParticle& ptc = trueParticles_[i];
      if (abs(ptc.pdgCode()) == 15) {
	// this is a tau
	if( i ) tooManyTaus = true;
	else tauFound=true;
      }
    }
    
    if(!tauFound || tooManyTaus ) {
      cerr<<"PFRootEventManager::tauBenchmark : not a single tau event"<<endl;
      return -9999;
    }
    
    // loop on the daugthers of the tau
    const std::vector<int>& ptcdaughters = trueParticles_[0].daughterIds();
    
    // will contain the sum of the lorentz vectors of the visible daughters
    // of the tau.
    
    
    for ( unsigned int dapt=0; dapt < ptcdaughters.size(); ++dapt) {
      
      const reco::PFTrajectoryPoint& tpatvtx 
	= trueParticles_[ptcdaughters[dapt]].trajectoryPoint(0);
      TLorentzVector partMC;
      partMC.SetPxPyPzE(tpatvtx.momentum().Px(),
			tpatvtx.momentum().Py(),
			tpatvtx.momentum().Pz(),
			tpatvtx.momentum().E());
      
      partTOTMC += partMC;
      if (tauBenchmarkDebug_) {
	//pdgcode
	int pdgcode =  trueParticles_[ptcdaughters[dapt]].pdgCode();
	cout << pdgcode << endl;
	cout << tpatvtx << endl;
	cout << partMC.Px() << " " << partMC.Py() << " " 
	     << partMC.Pz() << " " << partMC.E()
	     << " PT=" 
	     << sqrt(partMC.Px()*partMC.Px()+partMC.Py()*partMC.Py()) 
	     << endl;
      }//debug
    }//loop daughter
  }else{

    uint itau=0;
    const HepMC::GenEvent* myGenEvent = MCTruth_.GetEvent();
    for ( HepMC::GenEvent::particle_const_iterator 
	    piter  = myGenEvent->particles_begin();
	  piter != myGenEvent->particles_end(); 
	  ++piter ) {
      
    
      if (abs((*piter)->pdg_id())==15){
	itau++;
	tauFound=true;
	for ( HepMC::GenVertex::particles_out_const_iterator bp =
		(*piter)->end_vertex()->particles_out_const_begin();
	      bp != (*piter)->end_vertex()->particles_out_const_end(); ++bp ) {
	  uint nuId=abs((*bp)->pdg_id());
	  bool isNeutrino=(nuId==12)||(nuId==14)||(nuId==16);
	  if (!isNeutrino){
	    

	    TLorentzVector partMC;
	    partMC.SetPxPyPzE((*bp)->momentum().x(),
			      (*bp)->momentum().y(),
			      (*bp)->momentum().z(),
			      (*bp)->momentum().e());
	    partTOTMC += partMC;
	  }
	}
      }
    }
    if (itau>1) tooManyTaus=true;

    if(!tauFound || tooManyTaus ) {
      cerr<<"PFRootEventManager::tauBenchmark : not a single tau event"<<endl;
      return -9999;
    }
  }







  EventColin::Jet jetmc;

  jetmc.eta = partTOTMC.Eta();
  jetmc.phi = partTOTMC.Phi();
  jetmc.et = partTOTMC.Et();
  jetmc.e = partTOTMC.E();
  
  if(outEvent_) outEvent_->addJetMC( jetmc );

  /*
  //MC JETS RECONSTRUCTION (visible energy)
  for ( unsigned i=0;  i < trueParticles_.size(); i++) {
  const reco::PFSimParticle& ptc = trueParticles_[i];
  const std::vector<int>& ptcdaughters = ptc.daughterIds();
    
  //PARTICULE NOT DISINTEGRATING BEFORE ECAL
  if(ptcdaughters.size() != 0) continue;
    
  //TAKE INFO AT VERTEX //////////////////////////////////////////////////
  const reco::PFTrajectoryPoint& tpatvtx = ptc.trajectoryPoint(0);
  TLorentzVector partMC;
  partMC.SetPxPyPzE(tpatvtx.momentum().Px(),
  tpatvtx.momentum().Py(),
  tpatvtx.momentum().Pz(),
  tpatvtx.momentum().E());
    
  partTOTMC += partMC;
  if (tauBenchmarkDebug_) {
  //pdgcode
  int pdgcode = ptc.pdgCode();
  cout << pdgcode << endl;
  cout << tpatvtx << endl;
  cout << partMC.Px() << " " << partMC.Py() << " " 
  << partMC.Pz() << " " << partMC.E() 
  << " PT=" 
  << sqrt(partMC.Px()*partMC.Px()+partMC.Py()*partMC.Py()) 
  << endl;
  }//debug?
  }//loop true particles
  */
  if (tauBenchmarkDebug_) {
    cout << " ET Vector=" << partTOTMC.Et() 
         << " " << partTOTMC.Eta() 
         << " " << partTOTMC.Phi() << endl; cout << endl;
  }//debug

  //////////////////////////////////////////////////////////////////////////
  //CALO TOWER JETS (ECAL+HCAL Towers)
  //cout << endl;  
  //cout << "THERE ARE " << caloTowers_.size() << " CALO TOWERS" << endl;

  vector<TLorentzVector> allcalotowers;
  //   vector<double>         allemenergy;
  //   vector<double>         allhadenergy;
  double threshCaloTowers = 1E-10;
  for ( unsigned int i = 0; i < caloTowers_.size(); ++i) {
    
    if(caloTowers_[i].energy() < threshCaloTowers) {
      //     cout<<"skipping calotower"<<endl;
      continue;
    }

    TLorentzVector caloT;
    TVector3 pepr( caloTowers_[i].eta(),
                   caloTowers_[i].phi(),
                   caloTowers_[i].energy());
    TVector3 pxyz = Utils::VectorEPRtoXYZ( pepr );
    caloT.SetPxPyPzE(pxyz.X(),pxyz.Y(),pxyz.Z(),caloTowers_[i].energy());
    allcalotowers.push_back(caloT);
    //     allemenergy.push_back( caloTowers_[i].emEnergy() );
    //     allhadenergy.push_back( caloTowers_[i].hadEnergy() );
  }//loop calo towers
  if ( tauBenchmarkDebug_)  
    cout << " RETRIEVED " << allcalotowers.size() 
         << " CALOTOWER 4-VECTORS " << endl;
  
  //ECAL+HCAL tower jets computation
  jetAlgo_.Clear();
  const vector< PFJetAlgorithm::Jet >&  caloTjets 
    = jetAlgo_.FindJets( &allcalotowers );
  
  //cout << caloTjets.size() << " CaloTower Jets found" << endl;
  double JetEHTETmax = 0.0;
  for ( unsigned i = 0; i < caloTjets.size(); i++) {
    TLorentzVector jetmom = caloTjets[i].GetMomentum();
    double jetcalo_pt = sqrt(jetmom.Px()*jetmom.Px()+jetmom.Py()*jetmom.Py());
    double jetcalo_et = jetmom.Et();



    EventColin::Jet jet;
    jet.eta = jetmom.Eta();
    jet.phi = jetmom.Phi();
    jet.et  = jetmom.Et();
    jet.e   = jetmom.E();
    
    const vector<int>& indexes = caloTjets[i].GetIndexes();
    for( unsigned ii=0; ii<indexes.size(); ii++){
      jet.ee   +=  caloTowers_[ indexes[ii] ].emEnergy();
      jet.eh   +=  caloTowers_[ indexes[ii] ].hadEnergy();
      jet.ete   +=  caloTowers_[ indexes[ii] ].emEt();
      jet.eth   +=  caloTowers_[ indexes[ii] ].hadEt();
    }
    
    if(outEvent_) outEvent_->addJetEHT( jet );

    if ( tauBenchmarkDebug_) {
      cout << " ECAL+HCAL jet : " << caloTjets[i] << endl;
      cout << jetmom.Px() << " " << jetmom.Py() << " " 
           << jetmom.Pz() << " " << jetmom.E() 
           << " PT=" << jetcalo_pt << endl;
    }//debug

    if (jetcalo_et >= JetEHTETmax) 
      JetEHTETmax = jetcalo_et;
  }//loop MCjets

  //////////////////////////////////////////////////////////////////
  //PARTICLE FLOW JETS
  vector<TLorentzVector> allrecparticles;
  //   if ( tauBenchmarkDebug_) {
  //     cout << endl;
  //     cout << " THERE ARE " << pfBlocks_.size() << " EFLOW BLOCKS" << endl;
  //   }//debug

  //   for ( unsigned iefb = 0; iefb < pfBlocks_.size(); iefb++) {
  //       const std::vector< PFBlockParticle >& recparticles 
  //    = pfBlocks_[iefb].particles();

  
  
  for(unsigned i=0; i<candidates.size(); i++) {
  
    //       if (tauBenchmarkDebug_) 
    //  cout << " there are " << recparticles.size() 
    //       << " particle in this block" << endl;
    
    const reco::PFCandidate& candidate = candidates[i];

    if (tauBenchmarkDebug_) {
      cout << i << " " << candidate << endl;
      int type = candidate.particleId();
      cout << " type= " << type << " " << candidate.charge() 
           << endl;
    }//debug

    const math::XYZTLorentzVector& PFpart = candidate.p4();
    
    TLorentzVector partRec(PFpart.Px(), 
                           PFpart.Py(), 
                           PFpart.Pz(),
                           PFpart.E());
    
    //loading 4-vectors of Rec particles
    allrecparticles.push_back( partRec );

  }//loop on candidates
  

  if (tauBenchmarkDebug_) 
    cout << " THERE ARE " << allrecparticles.size() 
         << " RECONSTRUCTED 4-VECTORS" << endl;

  jetAlgo_.Clear();
  const vector< PFJetAlgorithm::Jet >&  PFjets 
    = jetAlgo_.FindJets( &allrecparticles );

  if (tauBenchmarkDebug_) 
    cout << PFjets.size() << " PF Jets found" << endl;
  double JetPFETmax = 0.0;
  for ( unsigned i = 0; i < PFjets.size(); i++) {
    TLorentzVector jetmom = PFjets[i].GetMomentum();
    double jetpf_pt = sqrt(jetmom.Px()*jetmom.Px()+jetmom.Py()*jetmom.Py());
    double jetpf_et = jetmom.Et();

    EventColin::Jet jet;
    jet.eta = jetmom.Eta();
    jet.phi = jetmom.Phi();
    jet.et  = jetmom.Et();
    jet.e   = jetmom.E();

    if(outEvent_) outEvent_->addJetPF( jet );

    if (tauBenchmarkDebug_) {
      cout <<" Rec jet : "<< PFjets[i] <<endl;
      cout << jetmom.Px() << " " << jetmom.Py() << " " 
           << jetmom.Pz() << " " << jetmom.E() 
           << " PT=" << jetpf_pt << " eta="<< jetmom.Eta() 
           << " Phi=" << jetmom.Phi() << endl;
      cout << "------------------------------------------------" << endl;
    }//debug
    
    if (jetpf_et >= JetPFETmax)  
      JetPFETmax = jetpf_et;
  }//loop PF jets

  //fill histos

  double deltaEtEHT = JetEHTETmax - partTOTMC.Et();
  h_deltaETvisible_MCEHT_->Fill(deltaEtEHT);
  
  double deltaEt = JetPFETmax - partTOTMC.Et();
  h_deltaETvisible_MCPF_ ->Fill(deltaEt);

  if (verbosity_ == VERBOSE ) {
    cout << "tau benchmark E_T(PF) - E_T(true) = " << deltaEt << endl;
  }

  return deltaEt/partTOTMC.Et();
}//Makejets





/*

void PFRootEventManager::lookForGenParticle(unsigned barcode) {
  
const HepMC::GenEvent* event = MCTruth_.GetEvent();
if(!event) {
cerr<<"no GenEvent"<<endl;
return;
}
  
const HepMC::GenParticle* particle = event->barcode_to_particle(barcode);
if(!particle) {
cerr<<"no particle with barcode "<<barcode<<endl;
return;
}

math::XYZTLorentzVector momentum(particle->momentum().px(),
particle->momentum().py(),
particle->momentum().pz(),
particle->momentum().e());

double eta = momentum.Eta();
double phi = momentum.phi();

double phisize = 0.05;
double etasize = 0.05;
  
double etagate = displayZoomFactor_ * etasize;
double phigate = displayZoomFactor_ * phisize;
  
if(displayHist_[EPE]) {
displayHist_[EPE]->GetXaxis()->SetRangeUser(eta-etagate, eta+etagate);
displayHist_[EPE]->GetYaxis()->SetRangeUser(phi-phigate, phi+phigate);
}
if(displayHist_[EPH]) {
displayHist_[EPH]->GetXaxis()->SetRangeUser(eta-etagate, eta+etagate);
displayHist_[EPH]->GetYaxis()->SetRangeUser(phi-phigate, phi+phigate);
}
  
updateDisplay();

}
*/



string PFRootEventManager::expand(const string& oldString) const {

  string newString = oldString;
 
  string dollar = "$";
  string slash  = "/";
  
  // protection necessary or segv !!
  int dollarPos = newString.find(dollar,0);
  if( dollarPos == -1 ) return oldString;

  int    lengh  = newString.find(slash,0) - newString.find(dollar,0) + 1;
  string env_variable =
    newString.substr( ( newString.find(dollar,0) + 1 ), lengh -2);
  // the env var could be defined between { }
  int begin = env_variable.find_first_of("{");
  int end = env_variable.find_last_of("}");
  
  // cout << "var=" << env_variable << begin<<" "<<end<< endl;
  

  env_variable = env_variable.substr( begin+1, end-1 );
  // cout << "var=" << env_variable <<endl;


  // cerr<<"call getenv "<<endl;
  char* directory = getenv( env_variable.c_str() );

  if(!directory) {
    cerr<<"please define environment variable $"<<env_variable<<endl;
    delete this;
    exit(1);
  }
  string sdir = directory;
  sdir += "/";

  newString.replace( 0, lengh , sdir);

  if (verbosity_ == VERBOSE ) {
    cout << "expand " <<oldString<<" to "<< newString << endl;
  }

  return newString;
}


void 
PFRootEventManager::printMCCalib(ofstream& out) const {

  if(!out) return;
  // if (!out.is_open()) return;

  // Use only for one PFSimParticle/GenParticles
  const HepMC::GenEvent* myGenEvent = MCTruth_.GetEvent();
  if(!myGenEvent) return;
  int nGen = 0;
  for ( HepMC::GenEvent::particle_const_iterator 
          piter  = myGenEvent->particles_begin();
          piter != myGenEvent->particles_end(); 
        ++piter ) nGen++;
  int nSim = trueParticles_.size();
  if ( nGen != 1 || nSim != 1 ) return;

  // One GenJet 
  if ( genJets_.size() != 1 ) return;
  double true_E = genJets_[0].p();
  double true_eta = genJets_[0].eta();
  double true_phi = genJets_[0].phi();

  // One particle-flow jet
  if ( pfJets_.size() != 1 ) return;
  double rec_ECALEnergy = pfJets_[0].neutralEmEnergy();
  double rec_HCALEnergy = pfJets_[0].neutralHadronEnergy();

  double col_ECALEnergy = rec_ECALEnergy * 1.05;
  double col_HCALEnergy = rec_HCALEnergy;
  if ( col_HCALEnergy > 1E-6 ) 
    col_HCALEnergy = col_ECALEnergy > 1E-6 ? 
    6. + 1.06*rec_HCALEnergy : (2.17*rec_HCALEnergy+1.73)/(1.+std::exp(2.49/rec_HCALEnergy));

  double jam_ECALEnergy = rec_ECALEnergy;
  double jam_HCALEnergy = rec_HCALEnergy;
  clusterCalibration_->
    getCalibratedEnergyEmbedAInHcal(jam_ECALEnergy, jam_HCALEnergy, true_eta, true_phi);

  out << true_eta << " " << true_phi << " " << true_E 
      << " " <<  rec_ECALEnergy << " " << rec_HCALEnergy
      << " " <<  col_ECALEnergy << " " << col_HCALEnergy
      << " " <<  jam_ECALEnergy << " " << jam_HCALEnergy << std::endl;

}

void  PFRootEventManager::print(ostream& out,int maxNLines ) const {

  if(!out) return;

  //If McTruthMatching print a detailed list 
  //of matching between simparticles and PFCandidates
  //MCTruth Matching vectors.
  std::vector< std::list <simMatch> > candSimMatchTrack;
  std::vector< std::list <simMatch> >  candSimMatchEcal;  
  if( printMCTruthMatching_){
    mcTruthMatching( std::cout,
		     *pfCandidates_,
		     candSimMatchTrack,
		     candSimMatchEcal);
  }


  if( printRecHits_ ) {
    out<<"ECAL RecHits =============================================="<<endl;
    for(unsigned i=0; i<rechitsECAL_.size(); i++) {
      string seedstatus = "    ";
      if(clusterAlgoECAL_.isSeed(i) ) 
        seedstatus = "SEED";
      printRecHit(rechitsECAL_[i], seedstatus.c_str(), out );
    }
    out<<endl;
    out<<"HCAL RecHits =============================================="<<endl;
    for(unsigned i=0; i<rechitsHCAL_.size(); i++) {
      string seedstatus = "    ";
      if(clusterAlgoHCAL_.isSeed(i) ) 
        seedstatus = "SEED";
      printRecHit(rechitsHCAL_[i], seedstatus.c_str(), out);
    }
    out<<endl;
    out<<"HFEM RecHits =============================================="<<endl;
    for(unsigned i=0; i<rechitsHFEM_.size(); i++) {
      string seedstatus = "    ";
      if(clusterAlgoHFEM_.isSeed(i) ) 
        seedstatus = "SEED";
      printRecHit(rechitsHFEM_[i], seedstatus.c_str(), out);
    }
    out<<endl;
    out<<"HFHAD RecHits =============================================="<<endl;
    for(unsigned i=0; i<rechitsHFHAD_.size(); i++) {
      string seedstatus = "    ";
      if(clusterAlgoHFHAD_.isSeed(i) ) 
        seedstatus = "SEED";
      printRecHit(rechitsHFHAD_[i], seedstatus.c_str(), out);
    }
    out<<endl;
    out<<"PS RecHits ================================================"<<endl;
    for(unsigned i=0; i<rechitsPS_.size(); i++) {
      string seedstatus = "    ";
      if(clusterAlgoPS_.isSeed(i) ) 
        seedstatus = "SEED";
      printRecHit(rechitsPS_[i], seedstatus.c_str(), out);
    }
    out<<endl;
  }
  if( printClusters_ ) {
    out<<"ECAL Clusters ============================================="<<endl;
    for(unsigned i=0; i<clustersECAL_->size(); i++) {
      printCluster((*clustersECAL_)[i], out);
    }    
    out<<endl;
    out<<"HCAL Clusters ============================================="<<endl;
    for(unsigned i=0; i<clustersHCAL_->size(); i++) {
      printCluster((*clustersHCAL_)[i], out);
    }    
    out<<endl;
    out<<"HFEM Clusters ============================================="<<endl;
    for(unsigned i=0; i<clustersHFEM_->size(); i++) {
      printCluster((*clustersHFEM_)[i], out);
    }    
    out<<endl;
    out<<"HFHAD Clusters ============================================="<<endl;
    for(unsigned i=0; i<clustersHFHAD_->size(); i++) {
      printCluster((*clustersHFHAD_)[i], out);
    }    
    out<<endl;
    out<<"PS Clusters   ============================================="<<endl;
    for(unsigned i=0; i<clustersPS_->size(); i++) {
      printCluster((*clustersPS_)[i], out);
    }    
    out<<endl;
  }
  bool printTracks = true;
  if( printTracks) {
    
  }
  if( printPFBlocks_ ) {
    out<<"Particle Flow Blocks ======================================"<<endl;
    for(unsigned i=0; i<pfBlocks_->size(); i++) {
      out<<i<<" "<<(*pfBlocks_)[i]<<endl;
    }    
    out<<endl;
  }
  if(printPFCandidates_) {
    out<<"Particle Flow Candidates =================================="<<endl;
    for(unsigned i=0; i<pfCandidates_->size(); i++) {
      out<<i<<" " <<(*pfCandidates_)[i]<<endl;
    }    
    out<<endl;

    //print a detailed list of PFSimParticles matching
    //the PFCandiates
    if(printMCTruthMatching_){
      cout<<"MCTruthMatching Results"<<endl;
      for(unsigned icand=0; icand<pfCandidates_->size(); 
	  icand++) {
	out <<icand<<" " <<(*pfCandidates_)[icand]<<endl;
	out << "is matching:" << endl;

	//tracking
	ITM it_t    = candSimMatchTrack[icand].begin();
	ITM itend_t = candSimMatchTrack[icand].end();
	for(;it_t!=itend_t;++it_t){
	  unsigned simid = it_t->second;
	  out << "\tSimParticle " << trueParticles_[simid]
	      <<endl;
	  out << "\t\tthrough Track matching pTrectrack=" 
	      << it_t->first << " GeV" << endl;
	}//loop simparticles

	ITM it_e    = candSimMatchEcal[icand].begin();
	ITM itend_e = candSimMatchEcal[icand].end();
	for(;it_e!=itend_e;++it_e){
	  unsigned simid = it_e->second;
	  out << "\tSimParticle " << trueParticles_[simid]
	      << endl; 
	  out << "\t\tsimparticle contributing to a total of " 
	      << it_e->first
	      << " GeV of its ECAL cluster"
	      << endl;  
	}//loop simparticles
	cout<<"________________"<<endl;
      }//loop candidates 
    }////print mc truth matching
  }
  if(printPFJets_) {
    out<<"Jets  ====================================================="<<endl;
    out<<"Particle Flow: "<<endl;
    for(unsigned i=0; i<pfJets_.size(); i++) {      
      if (pfJets_[i].pt() > printPFPt_ )
	out<<i<<pfJets_[i].print()<<endl;
    }    
    out<<endl;
    out<<"Generated: "<<endl;
    for(unsigned i=0; i<genJets_.size(); i++) {
      if (genJets_[i].pt() > printPFPt_ )
	out<<i<<genJets_[i].print()<<endl;
      // <<" invisible energy = "<<genJets_[i].invisibleEnergy()<<endl;
    }        
    out<<endl;
    out<<"Calo: "<<endl;
    for(unsigned i=0; i<caloJets_.size(); i++) {      
      out<<"pt = "<<caloJets_[i].pt()<<endl;
    }        
    out<<endl;  
  }
  if( printSimParticles_ ) {
    out<<"Sim Particles  ==========================================="<<endl;

    for(unsigned i=0; i<trueParticles_.size(); i++) {
      if( trackInsideGCut( trueParticles_[i]) ) 
        out<<"\t"<<trueParticles_[i]<<endl;
    }   
 
    //print a detailed list of PFSimParticles matching
    //the PFCandiates
    if(printMCTruthMatching_) {
      cout<<"MCTruthMatching Results"<<endl;
      for ( unsigned i=0;  i < trueParticles_.size(); i++) {
	cout << "==== Particle Simulated " << i << endl;
	const reco::PFSimParticle& ptc = trueParticles_[i];
	out <<i<<" "<<trueParticles_[i]<<endl;
	
	if(!ptc.daughterIds().empty()){
	  cout << "Look at the desintegration products" << endl;
	  cout << endl;
	  continue;
	}
	
	//TRACKING
	if(ptc.rectrackId() != 99999){
	  cout << "matching pfCandidate (trough tracking): " << endl;
	  for( unsigned icand=0; icand<pfCandidates_->size()
		 ; icand++ ) 
	    {
	      ITM it    = candSimMatchTrack[icand].begin();
	      ITM itend = candSimMatchTrack[icand].end();
	      for(;it!=itend;++it)
		if( i == it->second ){
		  out<<icand<<" "<<(*pfCandidates_)[icand]<<endl;
		  cout << endl;
		}
	    }//loop candidate
	}//trackmatch
	
	//CALORIMETRY
	vector<unsigned> rechitSimIDs  
	  = ptc.recHitContrib();
	vector<double>   rechitSimFrac 
	  = ptc.recHitContribFrac();
	//cout << "Number of rechits contrib =" << rechitSimIDs.size() << endl;
	if( !rechitSimIDs.size() ) continue; //no rechit
	
	cout << "matching pfCandidate (through ECAL): " << endl;
	
	//look at total ECAL desposition:
	double totalEcalE = 0.0;
	for(unsigned irh=0; irh<rechitsECAL_.size();++irh)
	  for ( unsigned isimrh=0;  isimrh < rechitSimIDs.size(); 
		isimrh++ )
	    if(rechitSimIDs[isimrh] == rechitsECAL_[irh].detId())
	      totalEcalE += (rechitsECAL_[irh].energy()*rechitSimFrac[isimrh]/100.0);
	cout << "For info, this particle deposits E=" << totalEcalE 
	     << "(GeV) in the ECAL" << endl;
	
	for( unsigned icand=0; icand<pfCandidates_->size()
	       ; icand++ ) 
	  {
	    ITM it    = candSimMatchEcal[icand].begin();
	    ITM itend = candSimMatchEcal[icand].end();
	    for(;it!=itend;++it)
	      if( i == it->second )
		out<<icand<<" "<<it->first<<"GeV "<<(*pfCandidates_)[icand]<<endl;	  
	  }//loop candidate
	cout << endl;      
      }//loop particles  
    }//mctruthmatching

  }

  
  if ( printGenParticles_ ) { 
    printGenParticles(out,maxNLines);
  }
}


void
PFRootEventManager::printGenParticles(std::ostream& out,
                                      int maxNLines) const {
                                 
                                 
  const HepMC::GenEvent* myGenEvent = MCTruth_.GetEvent();
  if(!myGenEvent) return;

  out<<"GenParticles ==========================================="<<endl;

  std::cout << "Id  Gen Name       eta    phi     pT     E    Vtx1   " 
            << " x      y      z   " 
            << "Moth  Vtx2  eta   phi     R      Z   Da1  Da2 Ecal?" 
            << std::endl;

  int nLines = 0;
  for ( HepMC::GenEvent::particle_const_iterator 
          piter  = myGenEvent->particles_begin();
        piter != myGenEvent->particles_end(); 
        ++piter ) {
    
    if( nLines == maxNLines) break;
    nLines++;
    
    HepMC::GenParticle* p = *piter;
    /* */
    int partId = p->pdg_id();

    // We have here a subset of particles only. 
    // To be filled according to the needs.
    /*switch(partId) {
      case    1: { name = "d"; break; } 
      case    2: { name = "u"; break; } 
      case    3: { name = "s"; break; } 
      case    4: { name = "c"; break; } 
      case    5: { name = "b"; break; } 
      case    6: { name = "t"; break; } 
      case   -1: { name = "~d"; break; } 
      case   -2: { name = "~u"; break; } 
      case   -3: { name = "~s"; break; } 
      case   -4: { name = "~c"; break; } 
      case   -5: { name = "~b"; break; } 
      case   -6: { name = "~t"; break; } 
      case   11: { name = "e-"; break; }
      case  -11: { name = "e+"; break; }
      case   12: { name = "nu_e"; break; }
      case  -12: { name = "~nu_e"; break; }
      case   13: { name = "mu-"; break; }
      case  -13: { name = "mu+"; break; }
      case   14: { name = "nu_mu"; break; }
      case  -14: { name = "~nu_mu"; break; }
      case   15: { name = "tau-"; break; }
      case  -15: { name = "tau+"; break; }
      case   16: { name = "nu_tau"; break; }
      case  -16: { name = "~nu_tau"; break; }
      case   21: { name = "gluon"; break; }
      case   22: { name = "gamma"; break; }
      case   23: { name = "Z0"; break; }
      case   24: { name = "W+"; break; }
      case   25: { name = "H0"; break; }
      case  -24: { name = "W-"; break; }
      case  111: { name = "pi0"; break; }
      case  113: { name = "rho0"; break; }
      case  223: { name = "omega"; break; }
      case  333: { name = "phi"; break; }
      case  443: { name = "J/psi"; break; }
      case  553: { name = "Upsilon"; break; }
      case  130: { name = "K0L"; break; }
      case  211: { name = "pi+"; break; }
      case -211: { name = "pi-"; break; }
      case  213: { name = "rho+"; break; }
      case -213: { name = "rho-"; break; }
      case  221: { name = "eta"; break; }
      case  331: { name = "eta'"; break; }
      case  441: { name = "etac"; break; }
      case  551: { name = "etab"; break; }
      case  310: { name = "K0S"; break; }
      case  311: { name = "K0"; break; }
      case -311: { name = "Kbar0"; break; }
      case  321: { name = "K+"; break; }
      case -321: { name = "K-"; break; }
      case  411: { name = "D+"; break; }
      case -411: { name = "D-"; break; }
      case  421: { name = "D0"; break; }
      case  431: { name = "Ds_+"; break; }
      case -431: { name = "Ds_-"; break; }
      case  511: { name = "B0"; break; }
      case  521: { name = "B+"; break; }
      case -521: { name = "B-"; break; }
      case  531: { name = "Bs_0"; break; }
      case  541: { name = "Bc_+"; break; }
      case -541: { name = "Bc_+"; break; }
      case  313: { name = "K*0"; break; }
      case -313: { name = "K*bar0"; break; }
      case  323: { name = "K*+"; break; }
      case -323: { name = "K*-"; break; }
      case  413: { name = "D*+"; break; }
      case -413: { name = "D*-"; break; }
      case  423: { name = "D*0"; break; }
      case  513: { name = "B*0"; break; }
      case  523: { name = "B*+"; break; }
      case -523: { name = "B*-"; break; }
      case  533: { name = "B*_s0"; break; }
      case  543: { name = "B*_c+"; break; }
      case -543: { name = "B*_c-"; break; }
      case  1114: { name = "Delta-"; break; }
      case -1114: { name = "Deltabar+"; break; }
      case -2112: { name = "nbar0"; break; }
      case  2112: { name = "n"; break; }
      case  2114: { name = "Delta0"; break; }
      case -2114: { name = "Deltabar0"; break; }
      case  3122: { name = "Lambda0"; break; }
      case -3122: { name = "Lambdabar0"; break; }
      case  3112: { name = "Sigma-"; break; }
      case -3112: { name = "Sigmabar+"; break; }
      case  3212: { name = "Sigma0"; break; }
      case -3212: { name = "Sigmabar0"; break; }
      case  3214: { name = "Sigma*0"; break; }
      case -3214: { name = "Sigma*bar0"; break; }
      case  3222: { name = "Sigma+"; break; }
      case -3222: { name = "Sigmabar-"; break; }
      case  2212: { name = "p"; break; }
      case -2212: { name = "~p"; break; }
      case -2214: { name = "Delta-"; break; }
      case  2214: { name = "Delta+"; break; }
      case -2224: { name = "Deltabar--"; break; }
      case  2224: { name = "Delta++"; break; }
      default: { 
      name = "unknown"; 
      cout << "Unknown code : " << partId << endl;
      }   
      }
    */
    std::string latexString;
    std::string name = getGenParticleName(partId,latexString);

    math::XYZTLorentzVector momentum1(p->momentum().px(),
                                      p->momentum().py(),
                                      p->momentum().pz(),
                                      p->momentum().e() );

    int vertexId1 = 0;

    if ( !p->production_vertex() && p->pdg_id() == 2212 ) continue;

    math::XYZVector vertex1;
    vertexId1 = -1;

    if(p->production_vertex() ) {
      vertex1.SetCoordinates( p->production_vertex()->position().x()/10.,
			      p->production_vertex()->position().y()/10.,
			      p->production_vertex()->position().z()/10. );
      vertexId1 = p->production_vertex()->barcode();
    }

    out.setf(std::ios::fixed, std::ios::floatfield);
    out.setf(std::ios::right, std::ios::adjustfield);
    
    out << std::setw(4) << p->barcode() << " " 
        << name;
    
    for(unsigned int k=0;k<11-name.length() && k<12; k++) out << " ";  
    
    double eta = momentum1.eta();
    if ( eta > +10. ) eta = +10.;
    if ( eta < -10. ) eta = -10.;
    
    out << std::setw(6) << std::setprecision(2) << eta << " " 
        << std::setw(6) << std::setprecision(2) << momentum1.phi() << " " 
        << std::setw(7) << std::setprecision(2) << momentum1.pt() << " " 
        << std::setw(7) << std::setprecision(2) << momentum1.e() << " " 
        << std::setw(4) << vertexId1 << " " 
        << std::setw(6) << std::setprecision(1) << vertex1.x() << " " 
        << std::setw(6) << std::setprecision(1) << vertex1.y() << " " 
        << std::setw(6) << std::setprecision(1) << vertex1.z() << " ";


    if( p->production_vertex() ) {
      if ( p->production_vertex()->particles_in_size() ) {
	const HepMC::GenParticle* mother = 
	  *(p->production_vertex()->particles_in_const_begin());
	
	out << std::setw(4) << mother->barcode() << " ";
      }
      else 
	out << "     " ;
    }    

    if ( p->end_vertex() ) {  
      math::XYZTLorentzVector vertex2(p->end_vertex()->position().x()/10.,
                                      p->end_vertex()->position().y()/10.,
                                      p->end_vertex()->position().z()/10.,
                                      p->end_vertex()->position().t()/10.);
      int vertexId2 = p->end_vertex()->barcode();

      std::vector<const HepMC::GenParticle*> children;
      HepMC::GenVertex::particles_out_const_iterator firstDaughterIt = 
        p->end_vertex()->particles_out_const_begin();
      HepMC::GenVertex::particles_out_const_iterator lastDaughterIt = 
        p->end_vertex()->particles_out_const_end();
      for ( ; firstDaughterIt != lastDaughterIt ; ++firstDaughterIt ) {
        children.push_back(*firstDaughterIt);
      }      

      out << std::setw(4) << vertexId2 << " "
          << std::setw(6) << std::setprecision(2) << vertex2.eta() << " " 
          << std::setw(6) << std::setprecision(2) << vertex2.phi() << " " 
          << std::setw(5) << std::setprecision(1) << vertex2.pt() << " " 
          << std::setw(6) << std::setprecision(1) << vertex2.z() << " ";

      for ( unsigned id=0; id<children.size(); ++id )
        out << std::setw(4) << children[id]->barcode() << " ";
    }
    out << std::endl;
  }
}


void  PFRootEventManager::printRecHit(const reco::PFRecHit& rh, 
                                      const char* seedstatus,
                                      ostream& out) const {

  if(!out) return;
  
  double eta = rh.position().Eta();
  double phi = rh.position().Phi();

  
  TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");
  if( !cutg || cutg->IsInside( eta, phi ) ) 
    out<<seedstatus<<" "<<rh<<endl;;
}

void  PFRootEventManager::printCluster(const reco::PFCluster& cluster,
                                       ostream& out ) const {
  
  if(!out) return;

  double eta = cluster.position().Eta();
  double phi = cluster.position().Phi();

  TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");
  if( !cutg || cutg->IsInside( eta, phi ) ) 
    out<<cluster<<endl;
}





bool PFRootEventManager::trackInsideGCut( const reco::PFTrack& track ) const {

  TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");
  if(!cutg) return true;
  
  const vector< reco::PFTrajectoryPoint >& points = track.trajectoryPoints();
  
  for( unsigned i=0; i<points.size(); i++) {
    if( ! points[i].isValid() ) continue;
    
    const math::XYZPoint& pos = points[i].position();
    if( cutg->IsInside( pos.Eta(), pos.Phi() ) ) return true;
  }

  // no point inside cut
  return false;
}


void  
PFRootEventManager::fillRecHitMask( vector<bool>& mask, 
                                    const reco::PFRecHitCollection& rechits ) 
  const {

  TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");
  if(!cutg) return;

  mask.clear();
  mask.reserve( rechits.size() );
  for(unsigned i=0; i<rechits.size(); i++) {
    
    double eta = rechits[i].position().Eta();
    double phi = rechits[i].position().Phi();

    if( cutg->IsInside( eta, phi ) )
      mask.push_back( true );
    else 
      mask.push_back( false );   
  }
}

void  
PFRootEventManager::fillClusterMask(vector<bool>& mask, 
                                    const reco::PFClusterCollection& clusters) 
  const {
  
  TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");
  if(!cutg) return;

  mask.clear();
  mask.reserve( clusters.size() );
  for(unsigned i=0; i<clusters.size(); i++) {
    
    double eta = clusters[i].position().Eta();
    double phi = clusters[i].position().Phi();

    if( cutg->IsInside( eta, phi ) )
      mask.push_back( true );
    else 
      mask.push_back( false );   
  }
}

void  
PFRootEventManager::fillTrackMask(vector<bool>& mask, 
                                  const reco::PFRecTrackCollection& tracks) 
  const {
  
  TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");
  if(!cutg) return;

  mask.clear();
  mask.reserve( tracks.size() );
  for(unsigned i=0; i<tracks.size(); i++) {
    if( trackInsideGCut( tracks[i] ) )
      mask.push_back( true );
    else 
      mask.push_back( false );   
  }
}

void  
PFRootEventManager::fillTrackMask(vector<bool>& mask, 
                                  const reco::GsfPFRecTrackCollection& tracks) 
  const {
  
  TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");
  if(!cutg) return;

  mask.clear();
  mask.reserve( tracks.size() );
  for(unsigned i=0; i<tracks.size(); i++) {
    if( trackInsideGCut( tracks[i] ) )
      mask.push_back( true );
    else 
      mask.push_back( false );   
  }
}


const reco::PFSimParticle&
PFRootEventManager::closestParticle( reco::PFTrajectoryPoint::LayerType layer, 
                                     double eta, double phi,
                                     double& peta, double& pphi, double& pe) 
  const {
  

  if( trueParticles_.empty() ) {
    string err  = "PFRootEventManager::closestParticle : ";
    err        += "vector of PFSimParticles is empty";
    throw std::length_error( err.c_str() );
  }

  double mindist2 = 99999999;
  unsigned iClosest=0;
  for(unsigned i=0; i<trueParticles_.size(); i++) {
    
    const reco::PFSimParticle& ptc = trueParticles_[i];

    // protection for old version of the PFSimParticle 
    // dataformats. 
    if( layer >= reco::PFTrajectoryPoint::NLayers ||
        ptc.nTrajectoryMeasurements() + layer >= 
        ptc.nTrajectoryPoints() ) {
      continue;
    }

    const reco::PFTrajectoryPoint& tp
      = ptc.extrapolatedPoint( layer );

    peta = tp.position().Eta();
    pphi = tp.position().Phi();
    pe = tp.momentum().E();

    double deta = peta - eta;
    double dphi = pphi - phi;

    double dist2 = deta*deta + dphi*dphi;

    if(dist2<mindist2) {
      mindist2 = dist2;
      iClosest = i;
    }
  }

  return trueParticles_[iClosest];
}



//-----------------------------------------------------------
void 
PFRootEventManager::readCMSSWJets() {

  cout<<"CMSSW Gen jets : size = " <<  genJetsCMSSW_.size() << endl;
  for ( unsigned i = 0; i < genJetsCMSSW_.size(); i++) {
    cout<<"Gen jet Et : " <<  genJetsCMSSW_[i].et() << endl;
  }
  cout<<"CMSSW PF jets : size = " <<  pfJetsCMSSW_.size() << endl;
  for ( unsigned i = 0; i < pfJetsCMSSW_.size(); i++) {
    cout<<"PF jet Et : " <<  pfJetsCMSSW_[i].et() << endl;
  }
  cout<<"CMSSW Calo jets : size = " <<  caloJetsCMSSW_.size() << endl;
  for ( unsigned i = 0; i < caloJetsCMSSW_.size(); i++) {
    cout<<"Calo jet Et : " << caloJetsCMSSW_[i].et() << endl;
  }
}
//________________________________________________________________
std::string PFRootEventManager::getGenParticleName(int partId, std::string &latexString) const
{
  std::string  name;
  switch(partId) {
  case    1: { name = "d";latexString="d"; break; } 
  case    2: { name = "u";latexString="u";break; } 
  case    3: { name = "s";latexString="s" ;break; } 
  case    4: { name = "c";latexString="c" ; break; } 
  case    5: { name = "b";latexString="b" ; break; } 
  case    6: { name = "t";latexString="t" ; break; } 
  case   -1: { name = "~d";latexString="#bar{d}" ; break; } 
  case   -2: { name = "~u";latexString="#bar{u}" ; break; } 
  case   -3: { name = "~s";latexString="#bar{s}" ; break; } 
  case   -4: { name = "~c";latexString="#bar{c}" ; break; } 
  case   -5: { name = "~b";latexString="#bar{b}" ; break; } 
  case   -6: { name = "~t";latexString="#bar{t}" ; break; } 
  case   11: { name = "e-";latexString=name ; break; }
  case  -11: { name = "e+";latexString=name ; break; }
  case   12: { name = "nu_e";latexString="#nu_{e}" ; break; }
  case  -12: { name = "~nu_e";latexString="#bar{#nu}_{e}" ; break; }
  case   13: { name = "mu-";latexString="#mu-" ; break; }
  case  -13: { name = "mu+";latexString="#mu+" ; break; }
  case   14: { name = "nu_mu";latexString="#nu_{mu}" ; break; }
  case  -14: { name = "~nu_mu";latexString="#bar{#nu}_{#mu}"; break; }
  case   15: { name = "tau-";latexString="#tau^{-}" ; break; }
  case  -15: { name = "tau+";latexString="#tau^{+}" ; break; }
  case   16: { name = "nu_tau";latexString="#nu_{#tau}" ; break; }
  case  -16: { name = "~nu_tau";latexString="#bar{#nu}_{#tau}"; break; }
  case   21: { name = "gluon";latexString= name; break; }
  case   22: { name = "gamma";latexString= "#gamma"; break; }
  case   23: { name = "Z0";latexString="Z^{0}" ; break; }
  case   24: { name = "W+";latexString="W^{+}" ; break; }
  case   25: { name = "H0";latexString=name ; break; }
  case  -24: { name = "W-";latexString="W^{-}" ; break; }
  case  111: { name = "pi0";latexString="#pi^{0}" ; break; }
  case  113: { name = "rho0";latexString="#rho^{0}" ; break; }
  case  223: { name = "omega";latexString="#omega" ; break; }
  case  333: { name = "phi";latexString= "#phi"; break; }
  case  443: { name = "J/psi";latexString="J/#psi" ; break; }
  case  553: { name = "Upsilon";latexString="#Upsilon" ; break; }
  case  130: { name = "K0L";latexString=name ; break; }
  case  211: { name = "pi+";latexString="#pi^{+}" ; break; }
  case -211: { name = "pi-";latexString="#pi^{-}" ; break; }
  case  213: { name = "rho+";latexString="#rho^{+}" ; break; }
  case -213: { name = "rho-";latexString="#rho^{-}" ; break; }
  case  221: { name = "eta";latexString="#eta" ; break; }
  case  331: { name = "eta'";latexString="#eta'" ; break; }
  case  441: { name = "etac";latexString="#eta_{c}" ; break; }
  case  551: { name = "etab";latexString= "#eta_{b}"; break; }
  case  310: { name = "K0S";latexString=name ; break; }
  case  311: { name = "K0";latexString="K^{0}" ; break; }
  case -311: { name = "Kbar0";latexString="#bar{#Kappa}^{0}" ; break; }
  case  321: { name = "K+";latexString= "K^{+}"; break; }
  case -321: { name = "K-";latexString="K^{-}"; break; }
  case  411: { name = "D+";latexString="D^{+}" ; break; }
  case -411: { name = "D-";latexString="D^{-}"; break; }
  case  421: { name = "D0";latexString=name ; break; }
  case  431: { name = "Ds_+";latexString="Ds_{+}" ; break; }
  case -431: { name = "Ds_-";latexString="Ds_{-}" ; break; }
  case  511: { name = "B0";latexString= name; break; }
  case  521: { name = "B+";latexString="B^{+}" ; break; }
  case -521: { name = "B-";latexString="B^{-}" ; break; }
  case  531: { name = "Bs_0";latexString="Bs_{0}" ; break; }
  case  541: { name = "Bc_+";latexString="Bc_{+}" ; break; }
  case -541: { name = "Bc_+";latexString="Bc_{+}" ; break; }
  case  313: { name = "K*0";latexString="K^{*0}" ; break; }
  case -313: { name = "K*bar0";latexString="#bar{K}^{*0}" ; break; }
  case  323: { name = "K*+";latexString="#K^{*+}"; break; }
  case -323: { name = "K*-";latexString="#K^{*-}" ; break; }
  case  413: { name = "D*+";latexString= "D^{*+}"; break; }
  case -413: { name = "D*-";latexString= "D^{*-}" ; break; }
  case  423: { name = "D*0";latexString="D^{*0}" ; break; }
  case  513: { name = "B*0";latexString="B^{*0}" ; break; }
  case  523: { name = "B*+";latexString="B^{*+}" ; break; }
  case -523: { name = "B*-";latexString="B^{*-}" ; break; }
  case  533: { name = "B*_s0";latexString="B^{*}_{s0}" ; break; }
  case  543: { name = "B*_c+";latexString= "B^{*}_{c+}"; break; }
  case -543: { name = "B*_c-";latexString= "B^{*}_{c-}"; break; }
  case  1114: { name = "Delta-";latexString="#Delta^{-}" ; break; }
  case -1114: { name = "Deltabar+";latexString="#bar{#Delta}^{+}" ; break; }
  case -2112: { name = "nbar0";latexString="{bar}n^{0}" ; break; }
  case  2112: { name = "n"; latexString=name ;break;}
  case  2114: { name = "Delta0"; latexString="#Delta^{0}" ;break; }
  case -2114: { name = "Deltabar0"; latexString="#bar{#Delta}^{0}" ;break; }
  case  3122: { name = "Lambda0";latexString= "#Lambda^{0}"; break; }
  case -3122: { name = "Lambdabar0";latexString="#bar{#Lambda}^{0}" ; break; }
  case  3112: { name = "Sigma-"; latexString="#Sigma" ;break; }
  case -3112: { name = "Sigmabar+"; latexString="#bar{#Sigma}^{+}" ;break; }
  case  3212: { name = "Sigma0";latexString="#Sigma^{0}" ; break; }
  case -3212: { name = "Sigmabar0";latexString="#bar{#Sigma}^{0}" ; break; }
  case  3214: { name = "Sigma*0"; latexString="#Sigma^{*0}" ;break; }
  case -3214: { name = "Sigma*bar0";latexString="#bar{#Sigma}^{*0}" ; break; }
  case  3222: { name = "Sigma+"; latexString="#Sigma^{+}" ;break; }
  case -3222: { name = "Sigmabar-"; latexString="#bar{#Sigma}^{-}";break; }
  case  2212: { name = "p";latexString=name ; break; }
  case -2212: { name = "~p";latexString="#bar{p}" ; break; }
  case -2214: { name = "Delta-";latexString="#Delta^{-}" ; break; }
  case  2214: { name = "Delta+";latexString="#Delta^{+}" ; break; }
  case -2224: { name = "Deltabar--"; latexString="#bar{#Delta}^{--}" ;break; }
  case  2224: { name = "Delta++"; latexString= "#Delta^{++}";break; }
  default:
    {
      name = "unknown"; 
      cout << "Unknown code : " << partId << endl;
      break;
    } 
                
                  
  }
  return name;  

}

//_____________________________________________________________________________
void PFRootEventManager::mcTruthMatching( std::ostream& out,
					  const reco::PFCandidateCollection& candidates,
					  std::vector< std::list <simMatch> >& candSimMatchTrack,
					  std::vector< std::list <simMatch> >& candSimMatchEcal) const
{
  
  if(!out) return;
  out << endl;
  out << "Running Monte Carlo Truth Matching Tool" << endl;
  out << endl;

  //resize matching vectors
  candSimMatchTrack.resize(candidates.size());
  candSimMatchEcal.resize(candidates.size());

  for(unsigned i=0; i<candidates.size(); i++) {
    const reco::PFCandidate& pfCand = candidates[i];
    
    //Matching with ECAL clusters
    if (verbosity_ == VERBOSE ) {
      out <<i<<" " <<(*pfCandidates_)[i]<<endl;
      out << "is matching:" << endl;
    }
    
    PFCandidate::ElementsInBlocks eleInBlocks 
      = pfCand.elementsInBlocks();

    for(unsigned iel=0; iel<eleInBlocks.size(); ++iel) {
      PFBlockRef blockRef   = eleInBlocks[iel].first;
      unsigned indexInBlock = eleInBlocks[iel].second;
      
      //Retrieving elements of the block
      const reco::PFBlock& blockh 
	= *blockRef;
      const edm::OwnVector< reco::PFBlockElement >& 
	elements_h = blockh.elements();
      
      reco::PFBlockElement::Type type 
	= elements_h[ indexInBlock ].type();   
//       cout <<"(" << blockRef.key() << "|" <<indexInBlock <<"|" 
// 	   << elements_h[ indexInBlock ].type() << ")," << endl;
      
      //TRACK=================================
      if(type == reco::PFBlockElement::TRACK){
	const reco::PFRecTrackRef trackref 
	  = elements_h[ indexInBlock ].trackRefPF();
	assert( !trackref.isNull() );	  
	const reco::PFRecTrack& track = *trackref; 
	const reco::TrackRef trkREF = track.trackRef();
	unsigned rtrkID = track.trackId();

	//looking for the matching charged simulated particle:
	for ( unsigned isim=0;  isim < trueParticles_.size(); isim++) {
	  const reco::PFSimParticle& ptc = trueParticles_[isim];
	  unsigned trackIDM = ptc.rectrackId();
	  if(trackIDM != 99999 
	     && trackIDM == rtrkID){

	    if (verbosity_ == VERBOSE ) 
	      out << "\tSimParticle " << isim 
		  << " through Track matching pTrectrack=" 
		  << trkREF->pt() << " GeV" << endl;	 
	    
	    //store info
	    std::pair<double, unsigned> simtrackmatch
	      = make_pair(trkREF->pt(),trackIDM);
	    candSimMatchTrack[i].push_back(simtrackmatch);
	  }//match
	}//loop simparticles 
	
      }//TRACK

      //ECAL=================================
      if(type == reco::PFBlockElement::ECAL)
	{
	  const reco::PFClusterRef clusterref 
	    = elements_h[ indexInBlock ].clusterRef();
	  assert( !clusterref.isNull() );	  
	  const reco::PFCluster& cluster = *clusterref; 
	  
	  const std::vector< reco::PFRecHitFraction >& 
	    fracs = cluster.recHitFractions();  

// 	  cout << "This is an ecal cluster of energy " 
// 	       << cluster.energy() << endl;
	  vector<unsigned> simpID;
	  vector<double>   simpEC(trueParticles_.size(),0.0);	  
	  vector<unsigned> simpCN(trueParticles_.size(),0);	 
	  for(unsigned int rhit = 0; rhit < fracs.size(); ++rhit){
	    
	    const reco::PFRecHitRef& rh = fracs[rhit].recHitRef();
	    if(rh.isNull()) continue;
	    const reco::PFRecHit& rechit_cluster = *rh;
//  	    cout << rhit << " ID=" << rechit_cluster.detId() 
//  		 << " E=" << rechit_cluster.energy() 
//  		 << " fraction=" << fracs[rhit].fraction() << " ";
	    
	    //loop on sim particules
// 	    cout << "coming from sim particles: ";
	    for ( unsigned isim=0;  isim < trueParticles_.size(); isim++) {
	      const reco::PFSimParticle& ptc = trueParticles_[isim];
	      
	      vector<unsigned> rechitSimIDs  
		= ptc.recHitContrib();
	      vector<double>   rechitSimFrac 
		= ptc.recHitContribFrac();
	      //cout << "Number of rechits contrib =" << rechitSimIDs.size() << endl;
	      if( !rechitSimIDs.size() ) continue; //no rechit
								       
	      for ( unsigned isimrh=0;  isimrh < rechitSimIDs.size(); isimrh++) {
		if( rechitSimIDs[isimrh] == rechit_cluster.detId() ){
		  
		  bool takenalready = false;
		  for(unsigned iss = 0; iss < simpID.size(); ++iss)
		    if(simpID[iss] == isim) takenalready = true;
		  if(!takenalready) simpID.push_back(isim);
		  
		  simpEC[isim] += 
		    ((rechit_cluster.energy()*rechitSimFrac[isimrh])/100.0)
		    *fracs[rhit].fraction();
		  
		  simpCN[isim]++; //counting rechits

//   		  cout << isim << " with contribution of =" 
//   		       << rechitSimFrac[isimrh] << "%, "; 
		}//match rechit
	      }//loop sim rechit
	    }//loop sim particules
//  	    cout << endl;
	  }//loop cand rechit 

	  for(unsigned is=0; is < simpID.size(); ++is)
	    {
	      double frac_of_cluster 
		= (simpEC[simpID[is]]/cluster.energy())*100.0;
	      
	      //store info
	      std::pair<double, unsigned> simecalmatch
		= make_pair(simpEC[simpID[is]],simpID[is]);
	      candSimMatchEcal[i].push_back(simecalmatch);
	      
	      if (verbosity_ == VERBOSE ) {
		out << "\tSimParticle " << simpID[is] 
		    << " through ECAL matching Epfcluster=" 
		    << cluster.energy() 
		    << " GeV with N=" << simpCN[simpID[is]]
		    << " rechits in common "
		    << endl; 
		out << "\t\tsimparticle contributing to a total of " 
		    << simpEC[simpID[is]]
		    << " GeV of this cluster (" 
		    <<  frac_of_cluster << "%) " 
		    << endl;
	      }
	    }//loop particle matched
	}//ECAL clusters

    }//loop elements

    if (verbosity_ == VERBOSE )
      cout << "===============================================================" 
	   << endl;

  }//loop pfCandidates_

  if (verbosity_ == VERBOSE ){

    cout << "=================================================================="
	 << endl;
    cout << "SimParticles" << endl;
    
    //loop simulated particles  
    for ( unsigned i=0;  i < trueParticles_.size(); i++) {
      cout << "==== Particle Simulated " << i << endl;
      const reco::PFSimParticle& ptc = trueParticles_[i];
      out <<i<<" "<<trueParticles_[i]<<endl;

      if(!ptc.daughterIds().empty()){
	cout << "Look at the desintegration products" << endl;
	cout << endl;
	continue;
      }
      
      //TRACKING
      if(ptc.rectrackId() != 99999){
	cout << "matching pfCandidate (trough tracking): " << endl;
	for( unsigned icand=0; icand<candidates.size(); icand++ ) 
	  {
	    ITM it    = candSimMatchTrack[icand].begin();
	    ITM itend = candSimMatchTrack[icand].end();
	    for(;it!=itend;++it)
	      if( i == it->second ){
		out<<icand<<" "<<(*pfCandidates_)[icand]<<endl;
		cout << endl;
	      }
	  }//loop candidate
      }//trackmatch
      
      
      //CALORIMETRY
      vector<unsigned> rechitSimIDs  
	= ptc.recHitContrib();
      vector<double>   rechitSimFrac 
	= ptc.recHitContribFrac();
      //cout << "Number of rechits contrib =" << rechitSimIDs.size() << endl;
      if( !rechitSimIDs.size() ) continue; //no rechit
      
      cout << "matching pfCandidate (through ECAL): " << endl;
      
      //look at total ECAL desposition:
      double totalEcalE = 0.0;
      for(unsigned irh=0; irh<rechitsECAL_.size();++irh)
	for ( unsigned isimrh=0;  isimrh < rechitSimIDs.size(); 
	      isimrh++ )
	  if(rechitSimIDs[isimrh] == rechitsECAL_[irh].detId())
	    totalEcalE += (rechitsECAL_[irh].energy()*rechitSimFrac[isimrh]/100.0);
      cout << "For info, this particle deposits E=" << totalEcalE 
	   << "(GeV) in the ECAL" << endl;
      
      for( unsigned icand=0; icand<candidates.size(); icand++ ) 
	{
	  ITM it    = candSimMatchEcal[icand].begin();
	  ITM itend = candSimMatchEcal[icand].end();
	  for(;it!=itend;++it)
	    if( i == it->second )
	      out<<icand<<" "<<it->first<<"GeV "<<(*pfCandidates_)[icand]<<endl;	  
	}//loop candidate
      cout << endl;
    }//loop particles  
  }//verbose

}//mctruthmatching
//_____________________________________________________________________________
