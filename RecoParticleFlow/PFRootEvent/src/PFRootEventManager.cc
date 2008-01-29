

//#include "FWCore/Framework/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
//#include "DataFormats/Common/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include "DataFormats/Math/interface/Point3D.h"

#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

#include "RecoParticleFlow/PFClusterAlgo/interface/PFClusterAlgo.h"
// #include "RecoParticleFlow/PFBlockAlgo/interface/PFBlockAlgo.h"
#include "RecoParticleFlow/PFBlockAlgo/interface/PFGeometry.h"
// #include "RecoParticleFlow/PFAlgo/interface/PFAlgo.h"


#include "RecoParticleFlow/PFRootEvent/interface/PFRootEventManager.h"

#include "RecoParticleFlow/PFRootEvent/interface/IO.h"

#include "RecoParticleFlow/PFRootEvent/interface/PFJetAlgorithm.h" 
#include "RecoParticleFlow/PFRootEvent/interface/Utils.h" 
#include "RecoParticleFlow/PFRootEvent/interface/EventColin.h" 
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"

#include "FWCore/FWLite/interface/AutoLibraryLoader.h"

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TCutG.h>
#include <TVector3.h>
// #include <TDatabasePDG.h>

#include <iostream>
#include <vector>
#include <stdlib.h>

using namespace std;


PFRootEventManager::PFRootEventManager() {
//   energyCalibration_ = new PFEnergyCalibration();
//   energyResolution_ = new PFEnergyResolution();
}



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
  clustersPS_(new reco::PFClusterCollection),
  pfBlocks_(new reco::PFBlockCollection),
  pfCandidates_(new reco::PFCandidateCollection),
//   pfCandidatesOther_(new reco::PFCandidateCollection),
  outFile_(0)
{
  
//   options_ = 0;
//   tree_ = 0;

//   outEvent_ = 0;
//   outTree_ = 0;

//   jetAlgo_ = 0;
  
//   iEvent_=0;
  h_deltaETvisible_MCEHT_ 
    = new TH1F("h_deltaETvisible_MCEHT","Jet Et difference CaloTowers-MC"
	       ,500,-50,50);
  h_deltaETvisible_MCPF_  
    = new TH1F("h_deltaETvisible_MCPF" ,"Jet Et difference ParticleFlow-MC"
	       ,500,-50,50);

  readOptions(file, true, true);
 
       
//   maxERecHitEcal_ = -1;
//   maxERecHitHcal_ = -1;

//   energyCalibration_ = new PFEnergyCalibration();
//   energyResolution_ = new PFEnergyResolution();
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

  clusteringIsOn_ = true;
  options_->GetOpt("clustering", "on/off", clusteringIsOn_);

//   clusteringMode_ = 0;
//   options_->GetOpt("clustering", "mode", clusteringMode_);  
  

  bool clusteringDebug = false;
  options_->GetOpt("clustering", "debug", clusteringDebug );


  debug_ = false; 
  options_->GetOpt("rootevent", "debug", debug_);

  findRecHitNeighbours_ = true;
  options_->GetOpt("clustering", "findRecHitNeighbours", 
		   findRecHitNeighbours_);
  
  
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


  // input root file --------------------------------------------

  if( reconnect )
    connect( inFileName_.c_str() );


  // various parameters ------------------------------------------

  vector<int> algos;
  options_->GetOpt("display", "cluster_algos", algos);
  algosToDisplay_.clear();
  for(unsigned i=0; i< algos.size(); i++) algosToDisplay_.insert( algos[i] );

  displayClusterLines_ = false;
  options_->GetOpt("display", "cluster_lines", displayClusterLines_);
  
//   if(displayClusterLines_) 
//     cout<<"will display cluster lines "<<endl;

  viewSizeEtaPhi_.clear();
  options_->GetOpt("display", "viewsize_etaphi", viewSizeEtaPhi_);
  if(viewSizeEtaPhi_.size() != 2) {
    cerr<<"PFRootEventManager::ReadOptions, bad display/viewsize_etaphi tag...using 700/350"
	<<endl;
    viewSizeEtaPhi_.clear();
    viewSizeEtaPhi_.push_back(700); 
    viewSizeEtaPhi_.push_back(350); 
  }

 viewSize_.clear();
  options_->GetOpt("display", "viewsize_xy", viewSize_);
  if(viewSize_.size() != 2) {
    cerr<<"PFRootEventManager::ReadOptions, bad display/viewsize_xy tag...using 700/350"
	<<endl;
    viewSize_.clear();
    viewSize_.push_back(600); 
    viewSize_.push_back(600); 
  }


  // display parametes ------------------------------------

  displayXY_ = true;
  options_->GetOpt("display", "x/y", displayXY_);
  
  displayEtaPhi_ = true;
  options_->GetOpt("display", "eta/phi", displayEtaPhi_);

  displayRZ_ = true;
  options_->GetOpt("display", "r/z", displayRZ_);

 
  displayColorClusters_ = false;
  options_->GetOpt("display", "color_clusters", displayColorClusters_);
  
  displayRecHits_= true;
  options_->GetOpt("display", "rechits",displayRecHits_);
  
  displayClusters_ = true;
  options_->GetOpt("display", "clusters",displayClusters_);
 
  displayRecTracks_ = true;
  options_->GetOpt("display", "rectracks", displayRecTracks_);

  displayTrueParticles_ = true;
  options_->GetOpt("display", "particles", displayTrueParticles_);

  displayZoomFactor_ = 10;  
  options_->GetOpt("display", "zoom_factor", displayZoomFactor_);

  displayJetColors_ = false;
  options_->GetOpt("display", "jet_colors", displayJetColors_);
  

  displayTrueParticlesPtMin_ = -1;
  options_->GetOpt("display", "particles_ptmin", displayTrueParticlesPtMin_);
  
  displayRecTracksPtMin_ = -1;
  options_->GetOpt("display", "rectracks_ptmin", displayRecTracksPtMin_);
  
  displayRecHitsEnMin_ = -1;
  options_->GetOpt("display","rechits_enmin",displayRecHitsEnMin_);
  
  displayClustersEnMin_ = -1;
  options_->GetOpt("display","clusters_enmin",displayClustersEnMin_);
  

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

  double threshEcalBarrel = 0.1;
  options_->GetOpt("clustering", "thresh_Ecal_Barrel", threshEcalBarrel);
  
  double threshSeedEcalBarrel = 0.3;
  options_->GetOpt("clustering", "thresh_Seed_Ecal_Barrel", 
		   threshSeedEcalBarrel);

  double threshEcalEndcap = 0.2;
  options_->GetOpt("clustering", "thresh_Ecal_Endcap", threshEcalEndcap);

  double threshSeedEcalEndcap = 0.8;
  options_->GetOpt("clustering", "thresh_Seed_Ecal_Endcap",
		   threshSeedEcalEndcap);

  double showerSigmaEcal = 3;  
  options_->GetOpt("clustering", "shower_Sigma_Ecal",
		   showerSigmaEcal);

  int nNeighboursEcal = 4;
  options_->GetOpt("clustering", "neighbours_Ecal", nNeighboursEcal);
  
  int posCalcNCrystalEcal = -1;
  options_->GetOpt("clustering", "posCalc_nCrystal_Ecal", 
		   posCalcNCrystalEcal);

  double posCalcP1Ecal = -1;
  options_->GetOpt("clustering", "posCalc_p1_Ecal", 
		   posCalcP1Ecal);
  

  clusterAlgoECAL_.setThreshBarrel( threshEcalBarrel );
  clusterAlgoECAL_.setThreshSeedBarrel( threshSeedEcalBarrel );
  
  clusterAlgoECAL_.setThreshEndcap( threshEcalEndcap );
  clusterAlgoECAL_.setThreshSeedEndcap( threshSeedEcalEndcap );

  clusterAlgoECAL_.setNNeighbours( nNeighboursEcal );
  clusterAlgoECAL_.setShowerSigma( showerSigmaEcal );

  clusterAlgoECAL_.setPosCalcNCrystal( posCalcNCrystalEcal );
  clusterAlgoECAL_.setPosCalcP1( posCalcP1Ecal );

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
// 	<<dcormode<<" "<<dcora<<" "<<dcorb<<" "<<dcorap<<" "<<dcorbp<<endl;
  reco::PFCluster::setDepthCorParameters( dcormode, 
					  dcora, dcorb, 
					  dcorap, dcorbp);
//   }
//   else {
//     reco::PFCluster::setDepthCorParameters( -1, 
// 					    0,0 , 
// 					    0,0 );
//   }

  

  double threshHcalBarrel = 0.8;
  options_->GetOpt("clustering", "thresh_Hcal_Barrel", threshHcalBarrel);
  
  double threshSeedHcalBarrel = 1.4;
  options_->GetOpt("clustering", "thresh_Seed_Hcal_Barrel", 
		   threshSeedHcalBarrel);

  double threshHcalEndcap = 0.8;
  options_->GetOpt("clustering", "thresh_Hcal_Endcap", threshHcalEndcap);

  double threshSeedHcalEndcap = 1.4;
  options_->GetOpt("clustering", "thresh_Seed_Hcal_Endcap",
		   threshSeedHcalEndcap);

  double showerSigmaHcal    = 15;
  options_->GetOpt("clustering", "shower_Sigma_Hcal",
                   showerSigmaHcal);
 
  int nNeighboursHcal = 4;
  options_->GetOpt("clustering", "neighbours_Hcal", nNeighboursHcal);

  int posCalcNCrystalHcal = 5;
  options_->GetOpt("clustering", "posCalc_nCrystal_Hcal",
                   posCalcNCrystalHcal);

  double posCalcP1Hcal = 1.0;
  options_->GetOpt("clustering", "posCalc_p1_Hcal", 
		   posCalcP1Hcal);




  clusterAlgoHCAL_.setThreshBarrel( threshHcalBarrel );
  clusterAlgoHCAL_.setThreshSeedBarrel( threshSeedHcalBarrel );
  
  clusterAlgoHCAL_.setThreshEndcap( threshHcalEndcap );
  clusterAlgoHCAL_.setThreshSeedEndcap( threshSeedHcalEndcap );

  clusterAlgoHCAL_.setNNeighbours( nNeighboursHcal );
  clusterAlgoHCAL_.setShowerSigma( showerSigmaHcal );

  clusterAlgoHCAL_.setPosCalcNCrystal( posCalcNCrystalHcal );
  clusterAlgoHCAL_.setPosCalcP1( posCalcP1Hcal );

  clusterAlgoHCAL_.enableDebugging( clusteringDebug ); 


  // clustering preshower

  double threshPS = 0.0001;
  options_->GetOpt("clustering", "thresh_PS", threshPS);
  
  double threshSeedPS = 0.001;
  options_->GetOpt("clustering", "thresh_Seed_PS", 
		   threshSeedPS);
  
  //Comment Michel: PSBarrel shall be removed?
  double threshPSBarrel     = threshPS;
  double threshSeedPSBarrel = threshSeedPS;

  double threshPSEndcap     = threshPS;
  double threshSeedPSEndcap = threshSeedPS;

  double showerSigmaPS    = 0.1;
  options_->GetOpt("clustering", "shower_Sigma_PS",
                   showerSigmaPS);
 
  int nNeighboursPS = 4;
  options_->GetOpt("clustering", "neighbours_PS", nNeighboursPS);

  int posCalcNCrystalPS = -1;
  options_->GetOpt("clustering", "posCalc_nCrystal_PS",
                   posCalcNCrystalPS);

  double posCalcP1PS = 0.;
  options_->GetOpt("clustering", "posCalc_p1_PS", 
		   posCalcP1PS);




  clusterAlgoPS_.setThreshBarrel( threshPSBarrel );
  clusterAlgoPS_.setThreshSeedBarrel( threshSeedPSBarrel );
  
  clusterAlgoPS_.setThreshEndcap( threshPSEndcap );
  clusterAlgoPS_.setThreshSeedEndcap( threshSeedPSEndcap );

  clusterAlgoPS_.setNNeighbours( nNeighboursPS );
  clusterAlgoPS_.setShowerSigma( showerSigmaPS );

  clusterAlgoPS_.setPosCalcNCrystal( posCalcNCrystalPS );
  clusterAlgoPS_.setPosCalcP1( posCalcP1PS );

  clusterAlgoPS_.enableDebugging( clusteringDebug ); 

  


  // options for particle flow ---------------------------------------------


  string map_ECAL_eta;
  options_->GetOpt("particle_flow", "resolution_map_ECAL_eta", map_ECAL_eta);
  string map_ECAL_phi;
  options_->GetOpt("particle_flow", "resolution_map_ECAL_phi", map_ECAL_phi);
  string map_ECALec_x;
  options_->GetOpt("particle_flow", "resolution_map_ECALec_x", map_ECALec_x);
  string map_ECALec_y;
  options_->GetOpt("particle_flow", "resolution_map_ECALec_y", map_ECALec_y);
  string map_HCAL_eta;
  options_->GetOpt("particle_flow", "resolution_map_HCAL_eta", map_HCAL_eta);
  string map_HCAL_phi;
  options_->GetOpt("particle_flow", "resolution_map_HCAL_phi", map_HCAL_phi);

  //getting resolution maps
  map_ECAL_eta = expand(map_ECAL_eta);
  map_ECAL_phi = expand(map_ECAL_phi);
  map_HCAL_eta = expand(map_HCAL_eta);
  map_HCAL_phi = expand(map_HCAL_phi);

  double DPtovPtCut = 999.;
  options_->GetOpt("particle_flow", "DPtoverPt_Cut", DPtovPtCut);
  double chi2TrackECAL=100;
  options_->GetOpt("particle_flow", "chi2_ECAL_Track", chi2TrackECAL);
  double chi2TrackHCAL=100;
  options_->GetOpt("particle_flow", "chi2_HCAL_Track", chi2TrackHCAL);
  double chi2ECALHCAL=100;
  options_->GetOpt("particle_flow", "chi2_ECAL_HCAL", chi2ECALHCAL);
  double chi2PSECAL=100;
  options_->GetOpt("particle_flow", "chi2_PS_ECAL", chi2PSECAL);
  double chi2PSTrack=100;
  options_->GetOpt("particle_flow", "chi2_PS_Track", chi2PSTrack);
  double chi2PSHV=100;
  options_->GetOpt("particle_flow", "chi2_PSH_PSV", chi2PSHV);
  bool   multiLink = false;
  options_->GetOpt("particle_flow", "multilink", multiLink);

  try {
    pfBlockAlgo_.setParameters( map_ECAL_eta.c_str(),
				map_ECAL_phi.c_str(),
				map_HCAL_eta.c_str(),
				map_HCAL_phi.c_str(),
				DPtovPtCut, 
				chi2TrackECAL,
				chi2TrackHCAL,
				chi2ECALHCAL,
				chi2PSECAL, 
				chi2PSTrack,
				chi2PSHV,
				multiLink ); 
  }  
  catch( std::exception& err ) {
    cerr<<"exception setting PFBlockAlgo parameters: "
	<<err.what()<<". terminating."<<endl;
    exit(1);
  }
  

  bool blockAlgoDebug = false;
  options_->GetOpt("blockAlgo", "debug",  blockAlgoDebug);  
  pfBlockAlgo_.setDebug( blockAlgoDebug );

  bool AlgoDebug = false;
  options_->GetOpt("PFAlgo", "debug",  AlgoDebug);  
  pfAlgo_.setDebug( AlgoDebug );

  double eCalibP0 = 0;
  double eCalibP1 = 1;
  vector<double> ecalib;
  options_->GetOpt("particle_flow", "ecalib", ecalib);
  if(ecalib.size() == 2) {
    eCalibP0 = ecalib[0];
    eCalibP1 = ecalib[1]; 
  }
  else {
    cerr<<"PFRootEventManager::readOptions : WARNING : "
	<<"wrong calibration coefficients for ECAL"<<endl;
  }


  double nSigmaECAL = 99999;
  options_->GetOpt("particle_flow", "nsigma_ECAL", nSigmaECAL);
  double nSigmaHCAL = 99999;
  options_->GetOpt("particle_flow", "nsigma_HCAL", nSigmaHCAL);


  double mvaCut = 999999;
  options_->GetOpt("particle_flow", "mergedPhotons_mvaCut", mvaCut);
  
  string mvaWeightFile = "";
  options_->GetOpt("particle_flow", "mergedPhotons_mvaWeightFile", 
		   mvaWeightFile);  
  mvaWeightFile = expand( mvaWeightFile );

  try {
    pfAlgo_.setParameters( eCalibP0, eCalibP1, nSigmaECAL, nSigmaHCAL,
			   mvaCut, mvaWeightFile.c_str() );
//     pfAlgoOther_.setParameters( eCalibP0, eCalibP1, nSigmaECAL, nSigmaHCAL,
// 			    mvaCut, mvaWeightFile.c_str() );
  }
  catch( std::exception& err ) {
    cerr<<"exception setting PFAlgo parameters: "
	<<err.what()<<". terminating."<<endl;
    exit(1);
  }

  int    algo = 2;
  options_->GetOpt("particle_flow", "algorithm", algo);

  pfAlgo_.setAlgo( algo );
//   pfAlgoOther_.setAlgo( 1 );


  bool pfAlgoDebug = false;
  options_->GetOpt("particle_flow", "debug", pfAlgoDebug );  

  pfAlgo_.setDebug( pfAlgoDebug );
//   pfAlgoOther_.setDebug( pfAlgoDebug );

  // print flags -------------

  printRecHits_ = false;
  options_->GetOpt("print", "rechits", printRecHits_ );
  
  printClusters_ = false;
  options_->GetOpt("print", "clusters", printClusters_ );
  
  printPFBlocks_ = true;
  options_->GetOpt("print", "PFBlocks", printPFBlocks_ );
  
  printPFCandidates_ = true;
  options_->GetOpt("print", "PFCandidates", printPFCandidates_ );
  
  printTrueParticles_ = true;
  options_->GetOpt("print", "true_particles", printTrueParticles_ );
  
  printMCtruth_ = true;
  options_->GetOpt("print", "MC_truth", printMCtruth_ );
  
  verbosity_ = VERBOSE;
  options_->GetOpt("print", "verbosity", verbosity_ );
  cout<<"verbosity : "<<verbosity_<<endl;

  // jets options ---------------------------------
  doJets_ = false;
  options_->GetOpt("jets", "dojets", doJets_);
  
  jetsDebug_ = false;
  
  if (doJets_) {
    double coneAngle = 0.5;
    options_->GetOpt("jets", "cone_angle", coneAngle);
    
    double seedEt    = 0.4;
    options_->GetOpt("jets", "seed_et", seedEt);
    
    double coneMerge = 100.0;
    options_->GetOpt("jets", "cone_merge", coneMerge);
    
    options_->GetOpt("jets", "jets_debug", jetsDebug_);

    // cout<<"jets debug "<<jetsDebug_<<endl;
    
    if( jetsDebug_ ) {
      cout << "Jet Options : ";
      cout << "Angle=" << coneAngle << " seedEt=" << seedEt 
	   << " Merge=" << coneMerge << endl;
    }

    jetAlgo_.SetConeAngle(coneAngle);
    jetAlgo_.SetSeedEt(seedEt);
    jetAlgo_.SetConeMerge(coneMerge);   
  }

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
  clustersPSBranch_ = 0;


  if( !clusteringIsOn_ ) {
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
  
  
  string clustersIslandBarrelbranchname;
  clustersIslandBarrelBranch_ = 0;
  options_->GetOpt("root","clusters_island_barrel_branch", 
		   clustersIslandBarrelbranchname);
  if(!clustersIslandBarrelbranchname.empty() ) {
    clustersIslandBarrelBranch_ 
      = tree_->GetBranch(clustersIslandBarrelbranchname.c_str());
    if(!clustersIslandBarrelBranch_) {
      cerr<<"PFRootEventManager::ReadOptions : clusters_island_barrel_branch not found : "
	  <<clustersIslandBarrelbranchname<< endl;
    }
  }
  else {
    cerr<<"branch not found: root/clusters_island_barrel_branch"<<endl;
  }

  string recTracksbranchname;
  options_->GetOpt("root","recTracks_branch", recTracksbranchname);

  recTracksBranch_ = tree_->GetBranch(recTracksbranchname.c_str());
  if(!recTracksBranch_) {
    cerr<<"PFRootEventManager::ReadOptions : recTracks_branch not found : "
	<<recTracksbranchname<< endl;
  }

  string stdTracksbranchname;
  options_->GetOpt("root","stdTracks_branch", stdTracksbranchname);

  stdTracksBranch_ = tree_->GetBranch(stdTracksbranchname.c_str());
  if(!stdTracksBranch_) {
    cerr<<"PFRootEventManager::ReadOptions : stdTracks_branch not found : "
	<<stdTracksbranchname<< endl;
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

  setAddresses();
} 



void PFRootEventManager::setAddresses() {
  if( rechitsECALBranch_ ) rechitsECALBranch_->SetAddress(&rechitsECAL_);
  if( rechitsHCALBranch_ ) rechitsHCALBranch_->SetAddress(&rechitsHCAL_);
  if( rechitsPSBranch_ ) rechitsPSBranch_->SetAddress(&rechitsPS_);
  if( clustersECALBranch_ ) clustersECALBranch_->SetAddress( clustersECAL_.get() );
  if( clustersHCALBranch_ ) clustersHCALBranch_->SetAddress( clustersHCAL_.get() );
  if( clustersPSBranch_ ) clustersPSBranch_->SetAddress( clustersPS_.get() );
  if( clustersIslandBarrelBranch_ ) 
    clustersIslandBarrelBranch_->SetAddress(&clustersIslandBarrel_);
  if( recTracksBranch_ ) recTracksBranch_->SetAddress(&recTracks_);
  if( stdTracksBranch_ ) stdTracksBranch_->SetAddress(&stdTracks_);
  if( trueParticlesBranch_ ) trueParticlesBranch_->SetAddress(&trueParticles_);
  if( MCTruthBranch_ ) { 
    MCTruthBranch_->SetAddress(&MCTruth_);
  }
  if( caloTowersBranch_ ) caloTowersBranch_->SetAddress(&caloTowers_);
}


PFRootEventManager::~PFRootEventManager() {

  if(outFile_) {
    outFile_->Close();
  }

  if(outEvent_) delete outEvent_;


  delete options_;
  
//   delete energyCalibration_;
//   PFBlock::setEnergyCalibration(NULL);
//   delete energyResolution_;
//   PFBlock::setEnergyResolution(NULL);
}


void PFRootEventManager::write() {
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
     entry%10 == 0) 
    cout<<"process entry "<< entry << endl;
  

//   if(fromRealData_) {
//     if( !readFromRealData(entry) ) return false;
//   }
//   else {
//     if(! readFromSimulation(entry) ) return false;
//   } 

  bool goodevent =  readFromSimulation(entry);

  if(verbosity_ == VERBOSE ) {
    cout<<"number of recTracks      : "<<recTracks_.size()<<endl;
    cout<<"number of stdTracks      : "<<stdTracks_.size()<<endl;
    cout<<"number of true particles : "<<trueParticles_.size()<<endl;
    cout<<"number of ECAL rechits   : "<<rechitsECAL_.size()<<endl;
    cout<<"number of HCAL rechits   : "<<rechitsHCAL_.size()<<endl;
    cout<<"number of PS rechits     : "<<rechitsPS_.size()<<endl;
  }  

  if( clusteringIsOn_ ) clustering(); 
  else if( verbosity_ == VERBOSE )
    cout<<"clustering is OFF - clusters come from the input file"<<endl; 

  if(verbosity_ == VERBOSE ) {
    if(clustersECAL_.get() ) {
      cout<<"number of ECAL clusters : "<<clustersECAL_->size()<<endl;
    }
    if(clustersHCAL_.get() ) {
      cout<<"number of HCAL clusters : "<<clustersHCAL_->size()<<endl;
    }
    if(clustersPS_.get() ) {
      cout<<"number of PS clusters : "<<clustersPS_->size()<<endl;
    }
  }

  particleFlow();

  // call print() in verbose mode
  if( verbosity_ == VERBOSE ) print();
  double deltaEt=0;
  // double deltaEt1=0;
  if( goodevent && doJets_) { 
    deltaEt  = makeJets( *pfCandidates_ ); 
    // deltaEt1 = makeJets( *pfCandidatesOther_ ); 
  }
  
  if(goodevent && outTree_) 
    outTree_->Fill();
  
 
  if( verbosity_ == VERBOSE )
     cout<<"delta E_t ="<<deltaEt<<endl;
//      cout<<"delta E_t ="<<deltaEt<<" delta E_t Other ="<<deltaEt1<<endl;

  
  
//   if( deltaEt>0.1 && deltaEt<0.2 ) {
//     cout<<deltaEt<<endl;
//     return true;
//   }  
//   else return false;
//   //  if(trueParticles_.size() != 1 ) return false;
//   else 
//     return false;
  
  return goodevent;

}



bool PFRootEventManager::readFromSimulation(int entry) {

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
  if(rechitsPSBranch_) {
    rechitsPSBranch_->GetEntry(entry);  
  }
  if(clustersECALBranch_ && !clusteringIsOn_) {
    clustersECALBranch_->GetEntry(entry);
  }
  if(clustersHCALBranch_ && !clusteringIsOn_) {
    clustersHCALBranch_->GetEntry(entry);
  }
  if(clustersPSBranch_ && !clusteringIsOn_) {
    clustersPSBranch_->GetEntry(entry);
  }
  if(clustersIslandBarrelBranch_) {
    clustersIslandBarrelBranch_->GetEntry(entry);
  }
  if(caloTowersBranch_) {
    caloTowersBranch_->GetEntry(entry);
  } 
  if(recTracksBranch_) {
    recTracksBranch_->GetEntry(entry);
  }
  tree_->GetEntry( entry, 0 );

  // now can use the tree

  bool goodevent = true;
  if(trueParticlesBranch_ ) {
    // this is a filter to select single particle events.
    if(filterNParticles_ && 
       trueParticles_.size() != filterNParticles_ ) {
      cout << "PFRootEventManager : event discarded Nparticles="
	   <<filterNParticles_<< endl; 
      goodevent = false;
    }
    if(goodevent && filterHadronicTaus_ && !isHadronicTau() ) {
      cout << "PFRootEventManager : leptonic tau discarded " << endl; 
      goodevent =  false;
    }
    if( goodevent && !filterTaus_.empty() 
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
  if(rechitsPSBranch_) {
    PreprocessRecHits( rechitsPS_ , findRecHitNeighbours_);
  }
  if(clustersECALBranch_ && !clusteringIsOn_) {
    for(unsigned i=0; i<clustersECAL_->size(); i++) 
      (*clustersECAL_)[i].calculatePositionREP();
  }
  if(clustersHCALBranch_ && !clusteringIsOn_) {
    for(unsigned i=0; i<clustersHCAL_->size(); i++) 
      (*clustersHCAL_)[i].calculatePositionREP();    
  }
  if(clustersPSBranch_ && !clusteringIsOn_) {
    for(unsigned i=0; i<clustersPS_->size(); i++) 
      (*clustersPS_)[i].calculatePositionREP();    
  }

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
// 	<<"neutral particles without the HepMCProduct. "
// 	<<"Please check that the branch edmHepMCProduct_*_*_* is found"<<endl;
//     exit(1);
//   }
  
//   for ( HepMC::GenEvent::particle_const_iterator 
// 	  piter  = myGenEvent->particles_begin();
// 	piter != myGenEvent->particles_end(); 
// 	++piter ) {
    
//     const HepMC::GenParticle* p = *piter;
//     int partId = p->pdg_id();
    
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
  
  // ECAL clustering -------------------------------------------

  vector<bool> mask;
  fillRecHitMask( mask, rechitsECAL_ );
  clusterAlgoECAL_.setMask( mask );  

  edm::OrphanHandle< reco::PFRecHitCollection > rechitsHandleECAL( &rechitsECAL_, edm::ProductID(10001) );
  clusterAlgoECAL_.doClustering( rechitsHandleECAL );
  clustersECAL_ = clusterAlgoECAL_.clusters();

  assert(clustersECAL_.get() );

  fillOutEventWithClusters( *clustersECAL_ );

  // HCAL clustering -------------------------------------------

  fillRecHitMask( mask, rechitsHCAL_ );
  clusterAlgoHCAL_.setMask( mask );  
  edm::OrphanHandle< reco::PFRecHitCollection > rechitsHandleHCAL( &rechitsHCAL_, edm::ProductID(10002) );
  clusterAlgoHCAL_.doClustering( rechitsHandleHCAL );
  //clusterAlgoHCAL_.doClustering( rechitsHCAL_ );
  clustersHCAL_ = clusterAlgoHCAL_.clusters();

  fillOutEventWithClusters( *clustersHCAL_ );

  // PS clustering -------------------------------------------

  fillRecHitMask( mask, rechitsPS_ );
  clusterAlgoPS_.setMask( mask );  
  edm::OrphanHandle< reco::PFRecHitCollection > rechitsHandlePS( &rechitsPS_, edm::ProductID(10003) );
  clusterAlgoPS_.doClustering( rechitsHandlePS );
  //clusterAlgoPS_.doClustering( rechitsPS_ );
  clustersPS_ = clusterAlgoPS_.clusters();

  fillOutEventWithClusters( *clustersPS_ );
  
}



void 
PFRootEventManager::fillOutEventWithClusters(const reco::PFClusterCollection& 
					     clusters) {

  if(!outEvent_) return;
  
  for(unsigned i=0; i<clusters.size(); i++) {
    EventColin::Cluster cluster;
    cluster.eta = clusters[i].positionXYZ().Eta();
    cluster.phi = clusters[i].positionXYZ().Phi();
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
	cerr<<err.what()<<endl;
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
      outptc.eta = tpatecal.positionXYZ().Eta();
      outptc.phi = tpatecal.positionXYZ().Phi();    
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

    const reco::PFBlock& block = blocks[i];
    
    EventColin::Block outblock;
 
    outEvent_->addBlock( outblock );
  }
}



void PFRootEventManager::particleFlow() {
  
  if( debug_) {
    cout<<"PFRootEventManager::particleFlow start"<<endl;
//     cout<<"number of elements in memory: "
// 	<<reco::PFBlockElement::instanceCounter()<<endl;
  }

  edm::OrphanHandle< reco::PFRecTrackCollection > trackh( &recTracks_, 
							  edm::ProductID(1) );  
  
  edm::OrphanHandle< reco::PFClusterCollection > ecalh( clustersECAL_.get(), 
							edm::ProductID(2) );  
  
  edm::OrphanHandle< reco::PFClusterCollection > hcalh( clustersHCAL_.get(), 
							edm::ProductID(3) );  

  edm::OrphanHandle< reco::PFClusterCollection > psh( clustersPS_.get(), 
						      edm::ProductID(4) );   


  vector<bool> trackMask;
  fillTrackMask( trackMask, recTracks_ );
  vector<bool> ecalMask;
  fillClusterMask( ecalMask, *clustersECAL_ );
  vector<bool> hcalMask;
  fillClusterMask( hcalMask, *clustersHCAL_ );
  vector<bool> psMask;
  fillClusterMask( psMask, *clustersPS_ );
  
  pfBlockAlgo_.setInput( trackh, ecalh, hcalh, psh,
			 trackMask, ecalMask, hcalMask, psMask ); 
  pfBlockAlgo_.findBlocks();
  
  if( debug_) cout<<pfBlockAlgo_<<endl;

  pfBlocks_ = pfBlockAlgo_.transferBlocks();


  edm::OrphanHandle< reco::PFBlockCollection > blockh( pfBlocks_.get(), 
						       edm::ProductID(5) );  
  
  pfAlgo_.reconstructParticles( blockh );
//   pfAlgoOther_.reconstructParticles( blockh );
  if( debug_) cout<< pfAlgo_<<endl;
  pfCandidates_ = pfAlgo_.transferCandidates();
//   pfCandidatesOther_ = pfAlgoOther_.transferCandidates();
  
  fillOutEventWithPFCandidates( *pfCandidates_ );

  if( debug_) cout<<"PFRootEventManager::particleFlow stop"<<endl;
}

double PFRootEventManager::makeJets( const reco::PFCandidateCollection& candidates) {
  //std::cout << "building jets from MC particles," 
  //    << "PF particles and caloTowers" << std::endl;
  
  //initialize Jets Reconstruction
  jetAlgo_.Clear();

  //MAKING TRUE PARTICLE JETS
  TLorentzVector partTOTMC;

  // colin: the following is not necessary
  // partTOTMC.SetPxPyPzE(0.0, 0.0, 0.0, 0.0);

  //MAKING JETS WITH TAU DAUGHTERS
  vector<reco::PFSimParticle> vectPART;
  for ( unsigned i=0;  i < trueParticles_.size(); i++) {
    const reco::PFSimParticle& ptc = trueParticles_[i];
    vectPART.push_back(ptc);
  }//loop

  for ( unsigned i=0;  i < trueParticles_.size(); i++) {
    const reco::PFSimParticle& ptc = trueParticles_[i];
    const std::vector<int>& ptcdaughters = ptc.daughterIds();

    if (abs(ptc.pdgCode()) == 15) {
      for ( unsigned int dapt=0; dapt < ptcdaughters.size(); ++dapt) {

	const reco::PFTrajectoryPoint& tpatvtx 
	  = vectPART[ptcdaughters[dapt]].trajectoryPoint(0);
	TLorentzVector partMC;
	partMC.SetPxPyPzE(tpatvtx.momentum().Px(),
			  tpatvtx.momentum().Py(),
			  tpatvtx.momentum().Pz(),
			  tpatvtx.momentum().E());

	partTOTMC += partMC;
	if (jetsDebug_) {
	  //pdgcode
	  int pdgcode = vectPART[ptcdaughters[dapt]].pdgCode();
	  cout << pdgcode << endl;
	  cout << tpatvtx << endl;
	  cout << partMC.Px() << " " << partMC.Py() << " " 
	       << partMC.Pz() << " " << partMC.E()
	       << " PT=" 
	       << sqrt(partMC.Px()*partMC.Px()+partMC.Py()*partMC.Py()) 
	       << endl;
	}//debug
      }//loop daughter
    }//tau?
  }//loop particles

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
    if (jetsDebug_) {
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
  if (jetsDebug_) {
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
  double threshCaloTowers = 0;
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
  if ( jetsDebug_)  
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

    if ( jetsDebug_) {
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
//   if ( jetsDebug_) {
//     cout << endl;
//     cout << " THERE ARE " << pfBlocks_.size() << " EFLOW BLOCKS" << endl;
//   }//debug

//   for ( unsigned iefb = 0; iefb < pfBlocks_.size(); iefb++) {
//       const std::vector< PFBlockParticle >& recparticles 
// 	= pfBlocks_[iefb].particles();

  
  
  for(unsigned i=0; i<candidates.size(); i++) {
  
//       if (jetsDebug_) 
// 	cout << " there are " << recparticles.size() 
// 	     << " particle in this block" << endl;
    
    const reco::PFCandidate& candidate = candidates[i];

    if (jetsDebug_) {
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
  

  if (jetsDebug_) 
    cout << " THERE ARE " << allrecparticles.size() 
	 << " RECONSTRUCTED 4-VECTORS" << endl;

  jetAlgo_.Clear();
  const vector< PFJetAlgorithm::Jet >&  PFjets 
    = jetAlgo_.FindJets( &allrecparticles );

  if (jetsDebug_) 
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

    if (jetsDebug_) {
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
    cout << "makeJets E_T(PF) - E_T(true) = " << deltaEt << endl;
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

void  PFRootEventManager::print(ostream& out) const {

  if(!out) return;

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
      out<<(*pfBlocks_)[i]<<endl;
    }    
    out<<endl;
  }
  if(printPFCandidates_) {
    out<<"Particle Flow Candidates =================================="<<endl;
    out<<pfAlgo_<<endl;
    for(unsigned i=0; i<pfCandidates_->size(); i++) {
      out<<(*pfCandidates_)[i]<<endl;
    }    
    out<<endl;
  }
  if( printTrueParticles_ ) {
    out<<"True Particles  ==========================================="<<endl;
    for(unsigned i=0; i<trueParticles_.size(); i++) {
       if( trackInsideGCut( trueParticles_[i]) ) 
	 out<<"\t"<<trueParticles_[i]<<endl;
     }    
 
  }

  
  if ( printMCtruth_ ) { 
    out<<"MC truth  ==========================================="<<endl;
    printMCTruth(out);
  }
}


void
PFRootEventManager::printMCTruth(std::ostream& out,
                                 int maxNLines) const {

  const HepMC::GenEvent* myGenEvent = MCTruth_.GetEvent();
  if(!myGenEvent) return;

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
    std::string name;

    // We have here a subset of particles only. 
    // To be filled according to the needs.
    switch(partId) {
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
    case  221: { name = "eta"; break; }
    case  331: { name = "eta'"; break; }
    case  441: { name = "etac"; break; }
    case  551: { name = "etab"; break; }
    case -213: { name = "rho-"; break; }
    case  310: { name = "K0S"; break; }
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
    case  2112: { name = "n"; break; }
    case  3122: { name = "Lambda0"; break; }
    case  3112: { name = "Sigma-"; break; }
    case -3112: { name = "Sigma+"; break; }
    case  3212: { name = "Sigma0"; break; }
    case  2212: { name = "p"; break; }
    case -2212: { name = "~p"; break; }
    default: { 
      name = "unknown"; 
      cout << "Unknown code : " << partId << endl;
    }   
    }

    math::XYZTLorentzVector momentum1(p->momentum().px(),
				      p->momentum().py(),
				      p->momentum().pz(),
				      p->momentum().e());

    int vertexId1 = 0;

    if ( !p->production_vertex() ) continue;

    math::XYZVector vertex1 (p->production_vertex()->position().x()/10.,
			     p->production_vertex()->position().y()/10.,
			     p->production_vertex()->position().z()/10.);
    vertexId1 = p->production_vertex()->barcode();
    
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.setf(std::ios::right, std::ios::adjustfield);
    
    std::cout << std::setw(4) << p->barcode() << " " 
	 << name;
    
    for(unsigned int k=0;k<11-name.length() && k<12; k++) std::cout << " ";  
    
    double eta = momentum1.eta();
    if ( eta > +10. ) eta = +10.;
    if ( eta < -10. ) eta = -10.;
    std::cout << std::setw(6) << std::setprecision(2) << eta << " " 
	      << std::setw(6) << std::setprecision(2) << momentum1.phi() << " " 
	      << std::setw(7) << std::setprecision(2) << momentum1.pt() << " " 
	      << std::setw(7) << std::setprecision(2) << momentum1.e() << " " 
	      << std::setw(4) << vertexId1 << " " 
	      << std::setw(6) << std::setprecision(1) << vertex1.x() << " " 
	      << std::setw(6) << std::setprecision(1) << vertex1.y() << " " 
	      << std::setw(6) << std::setprecision(1) << vertex1.z() << " ";

    const HepMC::GenParticle* mother = 
      *(p->production_vertex()->particles_in_const_begin());

    if ( mother )
      std::cout << std::setw(4) << mother->barcode() << " ";
    else 
      std::cout << "     " ;
    
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

      std::cout << std::setw(4) << vertexId2 << " "
		<< std::setw(6) << std::setprecision(2) << vertex2.eta() << " " 
		<< std::setw(6) << std::setprecision(2) << vertex2.phi() << " " 
		<< std::setw(5) << std::setprecision(1) << vertex2.pt() << " " 
		<< std::setw(6) << std::setprecision(1) << vertex2.z() << " ";
      for ( unsigned id=0; id<children.size(); ++id )
	std::cout << std::setw(4) << children[id]->barcode() << " ";
    }
    std::cout << std::endl;

  }
}


void  PFRootEventManager::printRecHit(const reco::PFRecHit& rh, 
				      const char* seedstatus,
				      ostream& out) const {

  if(!out) return;
  
  double eta = rh.positionREP().Eta();
  double phi = rh.positionREP().Phi();

  
  TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");
  if( !cutg || cutg->IsInside( eta, phi ) ) 
    out<<seedstatus<<" "<<rh<<endl;;
}

void  PFRootEventManager::printCluster(const reco::PFCluster& cluster,
				       ostream& out ) const {
  
  if(!out) return;

  double eta = cluster.positionREP().Eta();
  double phi = cluster.positionREP().Phi();

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
    
    const math::XYZPoint& pos = points[i].positionXYZ();
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
    
    double eta = rechits[i].positionREP().Eta();
    double phi = rechits[i].positionREP().Phi();

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
    
    double eta = clusters[i].positionREP().Eta();
    double phi = clusters[i].positionREP().Phi();

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

    peta = tp.positionXYZ().Eta();
    pphi = tp.positionXYZ().Phi();
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



