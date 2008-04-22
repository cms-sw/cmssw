

#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include "DataFormats/Math/interface/Point3D.h"

#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

#include "RecoParticleFlow/PFClusterAlgo/interface/PFClusterAlgo.h"
#include "RecoParticleFlow/PFBlockAlgo/interface/PFGeometry.h"

#include "RecoParticleFlow/PFRootEvent/interface/PFRootEventManager.h"

#include "RecoParticleFlow/PFRootEvent/interface/IO.h"

#include "RecoParticleFlow/PFRootEvent/interface/PFJetAlgorithm.h" 
#include "RecoJets/JetAlgorithms/interface/JetMaker.h"


#include "RecoParticleFlow/PFRootEvent/interface/Utils.h" 
#include "RecoParticleFlow/PFRootEvent/interface/EventColin.h" 
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"


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
  clustersPS_(new reco::PFClusterCollection),
  pfBlocks_(new reco::PFBlockCollection),
  pfCandidates_(new reco::PFCandidateCollection),
  //pfJets_(new reco::PFJetCollection),
  outFile_(0) {
  
  
  //   iEvent_=0;
  h_deltaETvisible_MCEHT_ 
    = new TH1F("h_deltaETvisible_MCEHT","Jet Et difference CaloTowers-MC"
	       ,100,-100,100);
  h_deltaETvisible_MCPF_  
    = new TH1F("h_deltaETvisible_MCPF" ,"Jet Et difference ParticleFlow-MC"
	       ,100,-100,100);

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
    
    bool PlotAgainstReco=0;
    options_->GetOpt("pfjet_benchmark", "PlotAgainstReco", PlotAgainstReco);
    
    double deltaRMax=0.1;
    options_->GetOpt("pfjet_benchmark", "deltaRMax", deltaRMax);


          PFJetBenchmark_.setup( outjetfilename, 
      			   pfjBenchmarkDebug,
      			   PlotAgainstReco,
     			   deltaRMax );
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


  doParticleFlow_ = true;
  options_->GetOpt("particle_flow", "on/off", doParticleFlow_);  

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

  // read PFCluster calibration parameters
  

  double e_slope = 1;
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
  

  shared_ptr<PFEnergyCalibration> 
    calibration( new PFEnergyCalibration( e_slope,
					  e_offset, 
					  eh_eslope,
					  eh_hslope,
					  eh_offset,
					  h_slope,
					  h_offset,
					  h_damping ) );


  double nSigmaECAL = 99999;
  options_->GetOpt("particle_flow", "nsigma_ECAL", nSigmaECAL);
  double nSigmaHCAL = 99999;
  options_->GetOpt("particle_flow", "nsigma_HCAL", nSigmaHCAL);

  bool   clusterRecoveryAlgo = false;
  options_->GetOpt("particle_flow", "clusterRecovery", clusterRecoveryAlgo );

  double mvaCut = 999999;
  options_->GetOpt("particle_flow", "mergedPhotons_mvaCut", mvaCut);
  
  string mvaWeightFile = "";
  options_->GetOpt("particle_flow", "mergedPhotons_mvaWeightFile", 
                   mvaWeightFile);  
  mvaWeightFile = expand( mvaWeightFile );
  

  // new for PS PFAlgo validation (MDN)
  double PSCut = 999999;
  options_->GetOpt("particle_flow", "mergedPhotons_PSCut", PSCut);

  try {
//     pfAlgo_.setParameters( eCalibP0, eCalibP1, nSigmaECAL, nSigmaHCAL,
//                            PSCut, mvaCut, mvaWeightFile.c_str() );
    pfAlgo_.setParameters( nSigmaECAL, nSigmaHCAL, 
			   calibration,
			   clusterRecoveryAlgo,
                           PSCut, mvaCut, mvaWeightFile.c_str() );
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
  options_->GetOpt("print", "jets", printPFJets_ );
 
  printSimParticles_ = true;
  options_->GetOpt("print", "simParticles", printSimParticles_ );

  printGenParticles_ = true;
  options_->GetOpt("print", "genParticles", printGenParticles_ );
  
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
  
  // GenParticlesCand   
  string genParticleCandBranchName;
  genParticleforJetsBranch_ = 0;
  options_->GetOpt("root","genParticleforJets_branch", 
		   genParticleCandBranchName);
  if(!genParticleCandBranchName.empty() ){  
    genParticleforJetsBranch_= 
      tree_->GetBranch(genParticleCandBranchName.c_str()); 
    if(!genParticleforJetsBranch_) {
      cerr<<"PFRootEventanager::ReadOptions : "
	  <<"genParticleBaseCandidates_branch not found : "
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
  string recPFBranchName; 
  options_->GetOpt("root","recPFJetBranchName", recPFBranchName);
  if(!recPFBranchName.empty() ) {
    recPFBranch_= tree_->GetBranch(recPFBranchName.c_str()); 
    if(!recPFBranch_) {
      cerr<<"PFRootEventManager::ReadOptions :recPFBranch_ not found : "
          <<recPFBranchName<< endl;
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
  if( genParticleforJetsBranch_ ) 
    genParticleforJetsBranch_->SetAddress(&genParticleRef_);
  if( caloTowerBaseCandidatesBranch_ ) {
    caloTowerBaseCandidatesBranch_->SetAddress(&caloTowerBaseCandidates_);
  }
  if (genJetBranch_) genJetBranch_->SetAddress(&genJetsCMSSW_);
  if (recCaloBranch_) recCaloBranch_->SetAddress(&caloJetsCMSSW_);
  if (recPFBranch_) recPFBranch_->SetAddress(&pfJetsCMSSW_); 
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
  // if(doPFJetBenchmark_) PFJetBenchmark_.write();
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
  
  bool goodevent =  readFromSimulation(entry);

  if(verbosity_ == VERBOSE ) {
    cout<<"number of recTracks      : "<<recTracks_.size()<<endl;
    cout<<"number of stdTracks      : "<<stdTracks_.size()<<endl;
    cout<<"number of true particles : "<<trueParticles_.size()<<endl;
    cout<<"number of ECAL rechits   : "<<rechitsECAL_.size()<<endl;
    cout<<"number of HCAL rechits   : "<<rechitsHCAL_.size()<<endl;
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
 //	 if (resNeutralEmEnergy>0.5) return true;
//	 else return false;
	}// end PFJet Benchmark
  
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
  if(recTracksBranch_) {
    recTracksBranch_->GetEntry(entry);
  }
  if(genParticleforJetsBranch_) {
    genParticleforJetsBranch_->GetEntry(entry);
  }
  if(caloTowerBaseCandidatesBranch_) {
    caloTowerBaseCandidatesBranch_->GetEntry(entry);
  }
  if(genJetBranch_) {
    genJetBranch_->GetEntry(entry);
  }
  if(recCaloBranch_) {
    recCaloBranch_->GetEntry(entry);
  }
  if(recPFBranch_) {
    recPFBranch_->GetEntry(entry);
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

   if ( recTracksBranch_ ) { 
     PreprocessRecTracks( recTracks_);
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
PFRootEventManager::PreprocessRecTracks(reco::PFRecTrackCollection& recTracks) {  
  for( unsigned i=0; i<recTracks.size(); ++i ) {     
    recTracks[i].calculatePositionREP();
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


//   edm::OrphanHandle< reco::PFBlockCollection > blockh( pfBlocks_.get(), 
//                                                        edm::ProductID(5) );  
  
  pfAlgo_.reconstructParticles( *pfBlocks_.get() );
  //   pfAlgoOther_.reconstructParticles( blockh );
  if( debug_) cout<< pfAlgo_<<endl;
  pfCandidates_ = pfAlgo_.transferCandidates();
  //   pfCandidatesOther_ = pfAlgoOther_.transferCandidates();
  
  fillOutEventWithPFCandidates( *pfCandidates_ );

  if( debug_) cout<<"PFRootEventManager::particleFlow stop"<<endl;
}



void PFRootEventManager::reconstructGenJets() {

  genJets_.clear();
  genParticleBaseCandidates_.clear();
  if (verbosity_ == VERBOSE || jetsDebug_) {
    cout <<"start reconstruct GenJets"<<endl;
	cout << " input gen particles for jet: all muons/neutrinos removed " << endl;
  }
  // need to convert reco::GenParticleRefVector genParticleRef_ 
  // into Candidate Collection input for reconstructFWLiteJets.
  // Warning: the selection of gen particles to be used for jet
  // has changed!!!
  // in 1_6_9  all muons/neutrinos are removed
  // for > 1_8_0  only muons/neutrinos coming from Bosons (pdg id from 23 to 39)
  // are removed. For instance muons/neutrinos from  tau decays are kept.
  // The motivation is: calo jet corrections should include corrections due
  // to muons/neutrinos from heavy flavors (b or c) decays inside jets.
  for(unsigned i=0; i<genParticleRef_.size(); i++) {
  const reco::GenParticle mcpart = *(genParticleRef_[i]);
	  // remove all muons/neutrinos for PFJet studies
	  if (reco::isNeutrino (mcpart) || reco::isMuon (mcpart)) continue;
	   if (jetsDebug_ ) {
	   cout << "      #" << i << "  PDG code:" << mcpart.pdgId() 
		    << " status " << mcpart.status()
		    << ", p/pt/eta/phi: " << mcpart.p() << '/' << mcpart.pt() 
		    << '/' << mcpart.eta() << '/' << mcpart.phi() << endl;
	   }
  genParticleBaseCandidates_.push_back(mcpart.clone() );
  }
  
  // convert Candidate Collection to RefTobase  vector GenConstit
  // Jet constituents are stored as CandidateBaseRef(see CandidateFwd.h)
  reco::CandidateBaseRefVector Genconstit;

  
  for(unsigned i=0;i<genParticleRef_.size(); i++) {
	  // conversion in two steps: cand->Ref->RefTobase
	  // transient Ref:
	  // edm::Ref <CandidateCollection> candRef(&genParticleBaseCandidates_, i); does not compile yet
	  edm::Ref <CandidateCollection>
	  candRef(const_cast<const CandidateCollection*>(&genParticleBaseCandidates_),i); 
	  const CandidateBaseRef constit (candRef); 
	  // conversion in one step does not compile?
	  //      edm::RefToBase <CandidateCollection> constit(pfCandh, i); 
	  Genconstit.push_back(constit);
	  
  }
  
  

  vector<ProtoJet> protoJets;
  reconstructFWLiteJets(genParticleBaseCandidates_, protoJets );

    // Convert Protojets to GenJets
  
  // For each Protojet in turn
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
		  newJet.addDaughter(Genconstit[index]);
	  }  // end loop on ProtoJet constituents
		 // last step: copy ProtoJet Variables into Jet
	  newJet.setJetArea(protojet.jetArea()); 
	  newJet.setPileup(protojet.pileup());
	  newJet.setNPasses(protojet.nPasses());
	  ++ijet;
	  if (jetsDebug_ )cout<<ijet<<newJet.print()<<endl;
	  genJets_.push_back (newJet);
	  
	  } // end loop on protojets iterator IPJ
  
}

void PFRootEventManager::reconstructCaloJets() {

  caloJets_.clear();
  if (verbosity_ == VERBOSE || jetsDebug_ ) {
    cout <<"start reconstruct CaloJets"<<endl;
  }

  //   reco::CandidateCollection baseCandidates;
  //   for(unsigned i=0; i<caloTowers_.size(); i++) {
  //     baseCandidates.push_back( caloTowers_[i].clone() );
  //   }
 
  reconstructFWLiteJets(caloTowerBaseCandidates_, caloJets_ );

  //COLIN: geometry needed to make a calo jet from a proto jet !!
  //   JetMaker mjet;
  //   typedef vector <ProtoJet>::const_iterator IPJ;
  //   for  (IPJ ipj = protoJets.begin(); ipj != protoJets.end (); ipj++) {
  //     caloJets_.push_back(mjet.makeCaloJet(*ipj));  
  //   } 
}


void PFRootEventManager::reconstructPFJets() {

	pfJets_.clear();
    basePFCandidates_.clear();
	/// basePFCandidates_ to be declared global in PFRootEventManager.h
	//reco::CandidateCollection basePFCandidates_;
    if (verbosity_ == VERBOSE || jetsDebug_) {
		cout <<"start reconstruct PFJets"<<endl;
	}
	
	// Copy PFCandidates into std::vector<Candidate> format
	// as input for jet algorithms
	// Warning:
	// basePFCandidates_ Collection lifetime ==  pfJets_ Collection lifetime
	// transform PFCandidates to Candidates
	for(unsigned i=0; i<pfCandidates_->size(); i++) { 
		basePFCandidates_.push_back( (*pfCandidates_)[i].clone() );	  
	}
	
	// Jet constituents are stored as CandidateBaseRef(see CandidateFwd.h)
	reco::CandidateBaseRefVector PFconstit;
		
	// convert Candidate Collection to RefTobase  vector PFConstit
	
	for(unsigned i=0;i<pfCandidates_->size(); i++) {
		// conversion in two steps: cand->Ref->RefTobase
		// transient Ref:
		// edm::Ref <CandidateCollection> candRef(&basePFCandidates_, i); does not compile yet
		edm::Ref <CandidateCollection> 
		candRef(const_cast<const CandidateCollection*>(&basePFCandidates_),i); 
		const CandidateBaseRef constit (candRef);
		// conversion in one step does not compile?
		//      edm::RefToBase <CandidateCollection> constit(pfCandh, i); 
		PFconstit.push_back(constit);
		
	}
	
// Reconstruct ProtoJets from basePFCandidates_

vector<ProtoJet> protoJets;
reconstructFWLiteJets(basePFCandidates_, protoJets );

// Convert Protojets to PFJets

// For each Protojet in turn
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
		newJet.addDaughter(PFconstit[index]);
	}  // end loop on ProtoJet constituents
	   // last step: copy ProtoJet Variables into Jet
	newJet.setJetArea(protojet.jetArea()); 
	newJet.setPileup(protojet.pileup());
	newJet.setNPasses(protojet.nPasses());
	++ijet;
	if (jetsDebug_ )cout<<ijet<<newJet.print()<<endl;
	pfJets_.push_back (newJet);
	
	} // end loop on protojets iterator IPJ

	}



void PFRootEventManager::reconstructFWLiteJets(const reco::CandidateCollection& Candidates, vector<ProtoJet>& output ) {

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
        if (tauBenchmarkDebug_) {
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

void  PFRootEventManager::print(ostream& out,int maxNLines ) const {

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
  }
  if(printPFJets_) {
    out<<"Jets  ====================================================="<<endl;
    out<<"Particle Flow: "<<endl;
    for(unsigned i=0; i<pfJets_.size(); i++) {      
      out<<i<<pfJets_[i].print()<<endl;
    }    
    out<<endl;
    out<<"Generated: "<<endl;
    for(unsigned i=0; i<genJets_.size(); i++) {      
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
 
  }

  
  if ( printGenParticles_ ) { 
    //out<<"GenParticles ==========================================="<<endl;
    printGenParticles(out);
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
                                      p->momentum().e());

    int vertexId1 = 0;

    if ( !p->production_vertex() ) continue;

    math::XYZVector vertex1 (p->production_vertex()->position().x()/10.,
                             p->production_vertex()->position().y()/10.,
                             p->production_vertex()->position().z()/10.);
    vertexId1 = p->production_vertex()->barcode();
    
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


    const HepMC::GenParticle* mother = 
      *(p->production_vertex()->particles_in_const_begin());

    if ( mother )
      out << std::setw(4) << mother->barcode() << " ";
    else 
      out << "     " ;
    
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
  
  double eta = rh.positionXYZ().Eta();
  double phi = rh.positionXYZ().Phi();

  
  TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");
  if( !cutg || cutg->IsInside( eta, phi ) ) 
    out<<seedstatus<<" "<<rh<<endl;;
}

void  PFRootEventManager::printCluster(const reco::PFCluster& cluster,
                                       ostream& out ) const {
  
  if(!out) return;

  double eta = cluster.positionXYZ().Eta();
  double phi = cluster.positionXYZ().Phi();

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
    
    double eta = rechits[i].positionXYZ().Eta();
    double phi = rechits[i].positionXYZ().Phi();

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
    
    double eta = clusters[i].positionXYZ().Eta();
    double phi = clusters[i].positionXYZ().Phi();

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
