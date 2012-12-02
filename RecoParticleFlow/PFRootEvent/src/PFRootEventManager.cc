#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

#include "DataFormats/FWLite/interface/ChainEvent.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterAlgo.h"
#include "RecoParticleFlow/PFTracking/interface/PFGeometry.h"

#include "RecoParticleFlow/PFRootEvent/interface/PFRootEventManager.h"

#include "RecoParticleFlow/PFRootEvent/interface/IO.h"

#include "RecoParticleFlow/PFRootEvent/interface/PFJetAlgorithm.h" 
#include "RecoParticleFlow/PFRootEvent/interface/JetMaker.h"

#include "RecoParticleFlow/PFRootEvent/interface/Utils.h" 
#include "RecoParticleFlow/PFRootEvent/interface/EventColin.h" 
#include "RecoParticleFlow/PFRootEvent/interface/METManager.h"

#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFSCEnergyCalibration.h"
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
#include <cstdlib>

using namespace std;
// using namespace boost;
using namespace reco;

using boost::shared_ptr;

PFRootEventManager::PFRootEventManager() {}



PFRootEventManager::PFRootEventManager(const char* file)
  : 
  iEvent_(0),
  options_(0),
  ev_(0),
  tree_(0),
  outTree_(0),
  outEvent_(0),
  //   clusters_(new reco::PFClusterCollection),
  eventAuxiliary_( new edm::EventAuxiliary ),
  clustersECAL_(new reco::PFClusterCollection),
  clustersHCAL_(new reco::PFClusterCollection),
  clustersHO_(new reco::PFClusterCollection),
  clustersHFEM_(new reco::PFClusterCollection),
  clustersHFHAD_(new reco::PFClusterCollection),
  clustersPS_(new reco::PFClusterCollection),
  pfBlocks_(new reco::PFBlockCollection),
  pfCandidates_(new reco::PFCandidateCollection),
  pfCandidateElectronExtras_(new reco::PFCandidateElectronExtraCollection),
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
 
  initializeEventInformation();
       
  //   maxERecHitEcal_ = -1;
  //   maxERecHitHcal_ = -1;

}


void PFRootEventManager::initializeEventInformation() {

  unsigned int nev = ev_->size();
  for ( unsigned int entry = 0; entry < nev; ++entry ) { 
    ev_->to(entry);
    const edm::EventBase& iEv = *ev_;
    mapEventToEntry_[iEv.id().run()][iEv.id().luminosityBlock()][iEv.id().event()] = entry;
  }

  cout<<"Number of events: "<< nev
      <<" starting with event: "<<mapEventToEntry_.begin()->first<<endl;
}


void PFRootEventManager::reset() { 

  if(outEvent_) {
    outEvent_->reset();
    outTree_->GetBranch("event")->SetAddress(&outEvent_);
  }  

  if ( ev_ && ev_->isValid() ) 
    ev_->getTFile()->cd();
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
    bool doOutTree = false;
    options_->GetOpt("root","outtree", doOutTree);
    if(doOutTree) {
      if(!outfilename.empty() ) {
	outFile_ = TFile::Open(outfilename.c_str(), "recreate");
	
	outFile_->cd();
	//options_->GetOpt("root","outtree", doOutTree);
	//if(doOutTree) {
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
    //COLIN : looks like this benchmark is not written in the standard output file. Any particular reason for it? 
    
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

  doPFCandidateBenchmark_ = true;
  options_->GetOpt("pfCandidate_benchmark", "on/off", doPFCandidateBenchmark_); 
  if (doPFCandidateBenchmark_) {    
    cout<<"+++ Setting PFCandidate benchmark"<<endl;
    TDirectory* dir = outFile_->mkdir("DQMData");
    dir = dir->mkdir("PFTask");    
    dir = dir->mkdir("particleFlowManager");
    pfCandidateManager_.setDirectory( dir );

    float dRMax = 0.5; 
    options_->GetOpt("pfCandidate_benchmark", "dRMax", dRMax); 
    float ptMin = 2; 
    options_->GetOpt("pfCandidate_benchmark", "ptMin", ptMin); 
    bool matchCharge = true; 
    options_->GetOpt("pfCandidate_benchmark", "matchCharge", matchCharge); 
    int mode = static_cast<int>(Benchmark::DEFAULT);
    options_->GetOpt("pfCandidate_benchmark", "mode", mode); 
    
    pfCandidateManager_.setParameters( dRMax, matchCharge, 
				       static_cast<Benchmark::Mode>(mode));
    pfCandidateManager_.setRange( ptMin, 10e5, -4.5, 4.5, -10, 10);
    pfCandidateManager_.setup();
    //COLIN need to set the subdirectory.  
    cout<<"+++ Done "<<endl;
  }
  // Addition to have DQM histograms : by S. Dutta   
  doPFDQM_ = true;
  options_->GetOpt("pfDQM_monitor", "on/off", doPFDQM_); 

  if (doPFDQM_) {
    cout<<"+++ Setting PFDQM Monitoring " <<endl;
    string dqmfilename;
    dqmFile_ = 0;
    options_->GetOpt("pfDQM_monitor","DQMFilename", dqmfilename);
    dqmFile_ = TFile::Open(dqmfilename.c_str(), "recreate");

    TDirectory* dir = dqmFile_->mkdir("DQMData");
    TDirectory* dir1 = dir->mkdir("PFJetValidation");
    TDirectory* dir2 = dir->mkdir("PFMETValidation");
    pfJetMonitor_.setDirectory( dir1 );
    pfMETMonitor_.setDirectory( dir2 );
    float dRMax = 0.5;
    options_->GetOpt("pfCandidate_benchmark", "dRMax", dRMax);
    float ptMin = 2;
    options_->GetOpt("pfCandidate_benchmark", "ptMin", ptMin);
    bool matchCharge = true;
    options_->GetOpt("pfCandidate_benchmark", "matchCharge", matchCharge);
    int mode = static_cast<int>(Benchmark::DEFAULT);
    options_->GetOpt("pfCandidate_benchmark", "mode", mode);

    pfJetMonitor_.setParameters( dRMax, matchCharge, static_cast<Benchmark::Mode>(mode),
                                 ptMin, 10e5, -4.5, 4.5, -10.0, 10.0, false);
    pfJetMonitor_.setup();

    pfMETMonitor_.setParameters( static_cast<Benchmark::Mode>(mode),
                                 ptMin, 10e5, -4.5, 4.5, -10.0, 10.0, false);
    pfMETMonitor_.setup();
  }
//-----------------------------------------------


  std::string outputFile0;
  TFile* file0 = 0;
  TH2F* hBNeighbour0 = 0;
  TH2F* hENeighbour0 = 0;
  options_->GetOpt("clustering", "ECAL", outputFile0);
  if(!outputFile0.empty() ) {
    file0 = TFile::Open(outputFile0.c_str(),"recreate");
    hBNeighbour0 = new TH2F("BNeighbour0","f_{Neighbours} vs 1/E_{Seed} (ECAL Barrel)",500, 0., 0.5, 1000,0.,1.);
    hENeighbour0 = new TH2F("ENeighbour0","f_{Neighbours} vs 1/E_{Seed} (ECAL Endcap)",500, 0., 0.2, 1000,0.,1.);
  }

  std::string outputFile1;
  TFile* file1 = 0;
  TH2F* hBNeighbour1 = 0;
  TH2F* hENeighbour1 = 0;
  options_->GetOpt("clustering", "HCAL", outputFile1);
  if(!outputFile1.empty() ) {
    file1 = TFile::Open(outputFile1.c_str(),"recreate");
    hBNeighbour1 = new TH2F("BNeighbour1","f_{Neighbours} vs 1/E_{Seed} (HCAL Barrel)",500, 0., 0.05, 400,0.,1.);
    hENeighbour1 = new TH2F("ENeighbour1","f_{Neighbours} vs 1/E_{Seed} (HCAL Endcap)",500, 0., 0.05, 400,0.,1.);
  }

  std::string outputFile2;
  TFile* file2 = 0;
  TH2F* hBNeighbour2 = 0;
  TH2F* hENeighbour2 = 0;
  options_->GetOpt("clustering", "HFEM", outputFile2);
  if(!outputFile2.empty() ) {
    file2 = TFile::Open(outputFile2.c_str(),"recreate");
    hBNeighbour2 = new TH2F("BNeighbour2","f_{Neighbours} vs 1/E_{Seed} (HFEM Barrel)",500, 0., 0.02, 400,0.,1.);
    hENeighbour2 = new TH2F("ENeighbour2","f_{Neighbours} vs 1/E_{Seed} (HFEM Endcap)",500, 0., 0.02, 400,0.,1.);
  }

  std::string outputFile3;
  TFile* file3 = 0;
  TH2F* hBNeighbour3 = 0;
  TH2F* hENeighbour3 = 0;
  options_->GetOpt("clustering", "HFHAD", outputFile3);
  if(!outputFile3.empty() ) {
    file3 = TFile::Open(outputFile3.c_str(),"recreate");
    hBNeighbour3 = new TH2F("BNeighbour3","f_{Neighbours} vs 1/E_{Seed} (HFHAD Barrel)",500, 0., 0.02, 400,0.,1.);
    hENeighbour3 = new TH2F("ENeighbour3","f_{Neighbours} vs 1/E_{Seed} (HFHAD Endcap)",500, 0., 0.02, 400,0.,1.);
  }

  std::string outputFile4;
  TFile* file4 = 0;
  TH2F* hBNeighbour4 = 0;
  TH2F* hENeighbour4 = 0;
  options_->GetOpt("clustering", "Preshower", outputFile4);
  if(!outputFile4.empty() ) {
    file4 = TFile::Open(outputFile4.c_str(),"recreate");
    hBNeighbour4 = new TH2F("BNeighbour4","f_{Neighbours} vs 1/E_{Seed} (Preshower Barrel)",200, 0., 1000., 400,0.,1.);
    hENeighbour4 = new TH2F("ENeighbour4","f_{Neighbours} vs 1/E_{Seed} (Preshower Endcap)",200, 0., 1000., 400,0.,1.);
  }

  // input root file --------------------------------------------

  if( reconnect )
    connect(); 

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

  //Rechit for Ring 0 and +-1/2
  double threshold_R0=0.4;
  options_->GetOpt("clustering", "threshold_Hit_R0", threshold_R0);

  double threshold_R1=1.0;
  options_->GetOpt("clustering", "threshold_Hit_R1", threshold_R1);

  //Clustering
  double threshHO_Seed_Ring0=1.0;
  options_->GetOpt("clustering", "threshHO_Seed_Ring0", threshHO_Seed_Ring0);

  double threshHO_Ring0=0.5;
  options_->GetOpt("clustering", "threshHO_Ring0", threshHO_Ring0);

  double threshHO_Seed_Outer=3.1;
  options_->GetOpt("clustering", "threshHO_Seed_Outer", threshHO_Seed_Outer);

  double threshHO_Outer=1.0;
  options_->GetOpt("clustering", "threshHO_Outer", threshHO_Outer);


  doClustering_ = true;
  //options_->GetOpt("clustering", "on/off", doClustering_);
  
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

  double threshCleanEcalBarrel = 1E5;
  options_->GetOpt("clustering", "thresh_Clean_Ecal_Barrel", 
                   threshCleanEcalBarrel);

  std::vector<double> minS4S1CleanEcalBarrel;
  options_->GetOpt("clustering", "minS4S1_Clean_Ecal_Barrel", 
                   minS4S1CleanEcalBarrel);

  double threshDoubleSpikeEcalBarrel = 10.;
  options_->GetOpt("clustering", "thresh_DoubleSpike_Ecal_Barrel", 
                   threshDoubleSpikeEcalBarrel);

  double minS6S2DoubleSpikeEcalBarrel = 0.04;
  options_->GetOpt("clustering", "minS6S2_DoubleSpike_Ecal_Barrel", 
                   minS6S2DoubleSpikeEcalBarrel);

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

  double threshCleanEcalEndcap = 1E5;
  options_->GetOpt("clustering", "thresh_Clean_Ecal_Endcap", 
                   threshCleanEcalEndcap);

  std::vector<double> minS4S1CleanEcalEndcap;
  options_->GetOpt("clustering", "minS4S1_Clean_Ecal_Endcap", 
                   minS4S1CleanEcalEndcap);

  double threshDoubleSpikeEcalEndcap = 1E9;
  options_->GetOpt("clustering", "thresh_DoubleSpike_Ecal_Endcap", 
                   threshDoubleSpikeEcalEndcap);

  double minS6S2DoubleSpikeEcalEndcap = -1.;
  options_->GetOpt("clustering", "minS6S2_DoubleSpike_Ecal_Endcap", 
                   minS6S2DoubleSpikeEcalEndcap);

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

  clusterAlgoECAL_.setHistos(file0,hBNeighbour0,hENeighbour0);

  clusterAlgoECAL_.setThreshBarrel( threshEcalBarrel );
  clusterAlgoECAL_.setThreshSeedBarrel( threshSeedEcalBarrel );
  
  clusterAlgoECAL_.setThreshPtBarrel( threshPtEcalBarrel );
  clusterAlgoECAL_.setThreshPtSeedBarrel( threshPtSeedEcalBarrel );
  
  clusterAlgoECAL_.setThreshCleanBarrel(threshCleanEcalBarrel);
  clusterAlgoECAL_.setS4S1CleanBarrel(minS4S1CleanEcalBarrel);

  clusterAlgoECAL_.setThreshDoubleSpikeBarrel( threshDoubleSpikeEcalBarrel );
  clusterAlgoECAL_.setS6S2DoubleSpikeBarrel( minS6S2DoubleSpikeEcalBarrel );

  clusterAlgoECAL_.setThreshEndcap( threshEcalEndcap );
  clusterAlgoECAL_.setThreshSeedEndcap( threshSeedEcalEndcap );

  clusterAlgoECAL_.setThreshPtEndcap( threshPtEcalEndcap );
  clusterAlgoECAL_.setThreshPtSeedEndcap( threshPtSeedEcalEndcap );

  clusterAlgoECAL_.setThreshCleanEndcap(threshCleanEcalEndcap);
  clusterAlgoECAL_.setS4S1CleanEndcap(minS4S1CleanEcalEndcap);

  clusterAlgoECAL_.setThreshDoubleSpikeEndcap( threshDoubleSpikeEcalEndcap );
  clusterAlgoECAL_.setS6S2DoubleSpikeEndcap( minS6S2DoubleSpikeEcalEndcap );

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

  
  // And now the HCAL
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

  double threshCleanHcalBarrel = 1E5;
  options_->GetOpt("clustering", "thresh_Clean_Hcal_Barrel", 
                   threshCleanHcalBarrel);

  std::vector<double> minS4S1CleanHcalBarrel;
  options_->GetOpt("clustering", "minS4S1_Clean_Hcal_Barrel", 
                   minS4S1CleanHcalBarrel);

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

  double threshCleanHcalEndcap = 1E5;
  options_->GetOpt("clustering", "thresh_Clean_Hcal_Endcap", 
                   threshCleanHcalEndcap);

  std::vector<double> minS4S1CleanHcalEndcap;
  options_->GetOpt("clustering", "minS4S1_Clean_Hcal_Endcap", 
                   minS4S1CleanHcalEndcap);

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

  bool cleanRBXandHPDs = false;
  options_->GetOpt("clustering", "cleanRBXandHPDs_Hcal",
                   cleanRBXandHPDs);

  double posCalcP1Hcal 
    = threshHcalBarrel<threshHcalEndcap ? threshHcalBarrel:threshHcalEndcap;
//   options_->GetOpt("clustering", "posCalc_p1_Hcal", 
//                    posCalcP1Hcal);


  clusterAlgoHCAL_.setHistos(file1,hBNeighbour1,hENeighbour1);


  clusterAlgoHCAL_.setThreshBarrel( threshHcalBarrel );
  clusterAlgoHCAL_.setThreshSeedBarrel( threshSeedHcalBarrel );
  
  clusterAlgoHCAL_.setThreshPtBarrel( threshPtHcalBarrel );
  clusterAlgoHCAL_.setThreshPtSeedBarrel( threshPtSeedHcalBarrel );
  
  clusterAlgoHCAL_.setThreshCleanBarrel(threshCleanHcalBarrel);
  clusterAlgoHCAL_.setS4S1CleanBarrel(minS4S1CleanHcalBarrel);

  clusterAlgoHCAL_.setThreshEndcap( threshHcalEndcap );
  clusterAlgoHCAL_.setThreshSeedEndcap( threshSeedHcalEndcap );

  clusterAlgoHCAL_.setThreshPtEndcap( threshPtHcalEndcap );
  clusterAlgoHCAL_.setThreshPtSeedEndcap( threshPtSeedHcalEndcap );

  clusterAlgoHCAL_.setThreshCleanEndcap(threshCleanHcalEndcap);
  clusterAlgoHCAL_.setS4S1CleanEndcap(minS4S1CleanHcalEndcap);

  clusterAlgoHCAL_.setNNeighbours( nNeighboursHcal );
  clusterAlgoHCAL_.setShowerSigma( showerSigmaHcal );

  clusterAlgoHCAL_.setPosCalcNCrystal( posCalcNCrystalHcal );
  clusterAlgoHCAL_.setPosCalcP1( posCalcP1Hcal );

  clusterAlgoHCAL_.setUseCornerCells( useCornerCellsHcal );
  clusterAlgoHCAL_.setCleanRBXandHPDs( cleanRBXandHPDs );

  clusterAlgoHCAL_.enableDebugging( clusteringDebug ); 


  // And now the HO
  double threshHOBarrel = 0.5;
  options_->GetOpt("clustering", "thresh_HO_Barrel", threshHOBarrel);
  
  double threshPtHOBarrel = 0.0;
  options_->GetOpt("clustering", "thresh_Pt_HO_Barrel", threshPtHOBarrel);
  
  double threshSeedHOBarrel = 1.0;
  options_->GetOpt("clustering", "thresh_Seed_HO_Barrel", 
                   threshSeedHOBarrel);

  double threshPtSeedHOBarrel = 0.0;
  options_->GetOpt("clustering", "thresh_Pt_Seed_HO_Barrel", 
                   threshPtSeedHOBarrel);

  double threshCleanHOBarrel = 1E5;
  options_->GetOpt("clustering", "thresh_Clean_HO_Barrel", 
                   threshCleanHOBarrel);

  std::vector<double> minS4S1CleanHOBarrel;
  options_->GetOpt("clustering", "minS4S1_Clean_HO_Barrel", 
                   minS4S1CleanHOBarrel);

  double threshDoubleSpikeHOBarrel = 1E9;
  options_->GetOpt("clustering", "thresh_DoubleSpike_HO_Barrel", 
                   threshDoubleSpikeHOBarrel);

  double minS6S2DoubleSpikeHOBarrel = -1;
  options_->GetOpt("clustering", "minS6S2_DoubleSpike_HO_Barrel", 
                   minS6S2DoubleSpikeHOBarrel);

  double threshHOEndcap = 1.0;
  options_->GetOpt("clustering", "thresh_HO_Endcap", threshHOEndcap);

  double threshPtHOEndcap = 0.0;
  options_->GetOpt("clustering", "thresh_Pt_HO_Endcap", threshPtHOEndcap);

  double threshSeedHOEndcap = 3.1;
  options_->GetOpt("clustering", "thresh_Seed_HO_Endcap",
                   threshSeedHOEndcap);

  double threshPtSeedHOEndcap = 0.0;
  options_->GetOpt("clustering", "thresh_Pt_Seed_HO_Endcap",
                   threshPtSeedHOEndcap);

  double threshCleanHOEndcap = 1E5;
  options_->GetOpt("clustering", "thresh_Clean_HO_Endcap", 
                   threshCleanHOEndcap);

  std::vector<double> minS4S1CleanHOEndcap;
  options_->GetOpt("clustering", "minS4S1_Clean_HO_Endcap", 
                   minS4S1CleanHOEndcap);

  double threshDoubleSpikeHOEndcap = 1E9;
  options_->GetOpt("clustering", "thresh_DoubleSpike_HO_Endcap", 
                   threshDoubleSpikeHOEndcap);

  double minS6S2DoubleSpikeHOEndcap = -1;
  options_->GetOpt("clustering", "minS6S2_DoubleSpike_HO_Endcap", 
                   minS6S2DoubleSpikeHOEndcap);

  double showerSigmaHO    = 15;
  options_->GetOpt("clustering", "shower_Sigma_HO",
                   showerSigmaHO);
 
  int nNeighboursHO = 4;
  options_->GetOpt("clustering", "neighbours_HO", nNeighboursHO);

  int posCalcNCrystalHO = 5;
  options_->GetOpt("clustering", "posCalc_nCrystal_HO",
                   posCalcNCrystalHO);

  bool useCornerCellsHO = false;
  options_->GetOpt("clustering", "useCornerCells_HO",
                   useCornerCellsHO);

  bool cleanRBXandHPDsHO = false;
  options_->GetOpt("clustering", "cleanRBXandHPDs_HO",
                   cleanRBXandHPDsHO);

  double posCalcP1HO 
    = threshHOBarrel<threshHOEndcap ? threshHOBarrel:threshHOEndcap;
//   options_->GetOpt("clustering", "posCalc_p1_HO", 
//                    posCalcP1HO);


  clusterAlgoHO_.setHistos(file1,hBNeighbour1,hENeighbour1);


  clusterAlgoHO_.setThreshBarrel( threshHOBarrel );
  clusterAlgoHO_.setThreshSeedBarrel( threshSeedHOBarrel );
  
  clusterAlgoHO_.setThreshPtBarrel( threshPtHOBarrel );
  clusterAlgoHO_.setThreshPtSeedBarrel( threshPtSeedHOBarrel );
  
  clusterAlgoHO_.setThreshCleanBarrel(threshCleanHOBarrel);
  clusterAlgoHO_.setS4S1CleanBarrel(minS4S1CleanHOBarrel);

  clusterAlgoHO_.setThreshDoubleSpikeBarrel( threshDoubleSpikeHOBarrel );
  clusterAlgoHO_.setS6S2DoubleSpikeBarrel( minS6S2DoubleSpikeHOBarrel );

  clusterAlgoHO_.setThreshEndcap( threshHOEndcap );
  clusterAlgoHO_.setThreshSeedEndcap( threshSeedHOEndcap );

  clusterAlgoHO_.setThreshPtEndcap( threshPtHOEndcap );
  clusterAlgoHO_.setThreshPtSeedEndcap( threshPtSeedHOEndcap );

  clusterAlgoHO_.setThreshCleanEndcap(threshCleanHOEndcap);
  clusterAlgoHO_.setS4S1CleanEndcap(minS4S1CleanHOEndcap);

  clusterAlgoHO_.setThreshDoubleSpikeEndcap( threshDoubleSpikeHOEndcap );
  clusterAlgoHO_.setS6S2DoubleSpikeEndcap( minS6S2DoubleSpikeHOEndcap );

  clusterAlgoHO_.setNNeighbours( nNeighboursHO );
  clusterAlgoHO_.setShowerSigma( showerSigmaHO );

  clusterAlgoHO_.setPosCalcNCrystal( posCalcNCrystalHO );
  clusterAlgoHO_.setPosCalcP1( posCalcP1HO );

  clusterAlgoHO_.setUseCornerCells( useCornerCellsHO );
  clusterAlgoHO_.setCleanRBXandHPDs( cleanRBXandHPDs );

  clusterAlgoHO_.enableDebugging( clusteringDebug ); 


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
  
  double threshCleanHFEM = 1E5;
  options_->GetOpt("clustering", "thresh_Clean_HFEM", 
                   threshCleanHFEM);

  std::vector<double> minS4S1CleanHFEM;
  options_->GetOpt("clustering", "minS4S1_Clean_HFEM", 
                   minS4S1CleanHFEM);

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


  clusterAlgoHFEM_.setHistos(file2,hBNeighbour2,hENeighbour2);

  clusterAlgoHFEM_.setThreshEndcap( threshHFEM );
  clusterAlgoHFEM_.setThreshSeedEndcap( threshSeedHFEM );

  clusterAlgoHFEM_.setThreshPtEndcap( threshPtHFEM );
  clusterAlgoHFEM_.setThreshPtSeedEndcap( threshPtSeedHFEM );

  clusterAlgoHFEM_.setThreshCleanEndcap(threshCleanHFEM);
  clusterAlgoHFEM_.setS4S1CleanEndcap(minS4S1CleanHFEM);

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
  
  double threshCleanHFHAD = 1E5;
  options_->GetOpt("clustering", "thresh_Clean_HFHAD", 
                   threshCleanHFHAD);

  std::vector<double> minS4S1CleanHFHAD;
  options_->GetOpt("clustering", "minS4S1_Clean_HFHAD", 
                   minS4S1CleanHFHAD);

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


  clusterAlgoHFHAD_.setHistos(file3,hBNeighbour3,hENeighbour3);

  clusterAlgoHFHAD_.setThreshEndcap( threshHFHAD );
  clusterAlgoHFHAD_.setThreshSeedEndcap( threshSeedHFHAD );

  clusterAlgoHFHAD_.setThreshPtEndcap( threshPtHFHAD );
  clusterAlgoHFHAD_.setThreshPtSeedEndcap( threshPtSeedHFHAD );

  clusterAlgoHFHAD_.setThreshCleanEndcap(threshCleanHFHAD);
  clusterAlgoHFHAD_.setS4S1CleanEndcap(minS4S1CleanHFHAD);

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
  options_->GetOpt("clustering", "thresh_Pt_Seed_PS", threshPtSeedPS);
  
  double threshCleanPS = 1E5;
  options_->GetOpt("clustering", "thresh_Clean_PS", threshCleanPS);

  std::vector<double> minS4S1CleanPS;
  options_->GetOpt("clustering", "minS4S1_Clean_PS", minS4S1CleanPS);

  //Comment Michel: PSBarrel shall be removed?
  double threshPSBarrel     = threshPS;
  double threshSeedPSBarrel = threshSeedPS;

  double threshPtPSBarrel     = threshPtPS;
  double threshPtSeedPSBarrel = threshPtSeedPS;

  double threshCleanPSBarrel = threshCleanPS;
  std::vector<double> minS4S1CleanPSBarrel = minS4S1CleanPS;

  double threshPSEndcap     = threshPS;
  double threshSeedPSEndcap = threshSeedPS;

  double threshPtPSEndcap     = threshPtPS;
  double threshPtSeedPSEndcap = threshPtSeedPS;

  double threshCleanPSEndcap = threshCleanPS;
  std::vector<double> minS4S1CleanPSEndcap = minS4S1CleanPS;

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


  clusterAlgoPS_.setHistos(file4,hBNeighbour4,hENeighbour4);



  clusterAlgoPS_.setThreshBarrel( threshPSBarrel );
  clusterAlgoPS_.setThreshSeedBarrel( threshSeedPSBarrel );
  
  clusterAlgoPS_.setThreshPtBarrel( threshPtPSBarrel );
  clusterAlgoPS_.setThreshPtSeedBarrel( threshPtSeedPSBarrel );
  
  clusterAlgoPS_.setThreshCleanBarrel(threshCleanPSBarrel);
  clusterAlgoPS_.setS4S1CleanBarrel(minS4S1CleanPSBarrel);

  clusterAlgoPS_.setThreshEndcap( threshPSEndcap );
  clusterAlgoPS_.setThreshSeedEndcap( threshSeedPSEndcap );

  clusterAlgoPS_.setThreshPtEndcap( threshPtPSEndcap );
  clusterAlgoPS_.setThreshPtSeedEndcap( threshPtSeedPSEndcap );

  clusterAlgoPS_.setThreshCleanEndcap(threshCleanPSEndcap);
  clusterAlgoPS_.setS4S1CleanEndcap(minS4S1CleanPSEndcap);

  clusterAlgoPS_.setNNeighbours( nNeighboursPS );
  clusterAlgoPS_.setShowerSigma( showerSigmaPS );

  clusterAlgoPS_.setPosCalcNCrystal( posCalcNCrystalPS );
  clusterAlgoPS_.setPosCalcP1( posCalcP1PS );

  clusterAlgoPS_.setUseCornerCells( useCornerCellsPS );

  clusterAlgoPS_.enableDebugging( clusteringDebug ); 

  // options for particle flow ---------------------------------------------


  doParticleFlow_ = true;
  doCompare_ = false;
  options_->GetOpt("particle_flow", "on/off", doParticleFlow_);  
  options_->GetOpt("particle_flow", "comparison", doCompare_);  

  useKDTreeTrackEcalLinker_ = true;
  options_->GetOpt("particle_flow", "useKDTreeTrackEcalLinker", useKDTreeTrackEcalLinker_);  
  std::cout << "Use Track-ECAL link optimization: " << useKDTreeTrackEcalLinker_ << std::endl;
  pfBlockAlgo_.setUseOptimization(useKDTreeTrackEcalLinker_);

  std::vector<double> DPtovPtCut;
  std::vector<unsigned> NHitCut;
  bool useIterTracking;
  int nuclearInteractionsPurity;
  options_->GetOpt("particle_flow", "DPtoverPt_Cut", DPtovPtCut);
  options_->GetOpt("particle_flow", "NHit_Cut", NHitCut);
  options_->GetOpt("particle_flow", "useIterTracking", useIterTracking);
  options_->GetOpt("particle_flow", "nuclearInteractionsPurity", nuclearInteractionsPurity);
  
  
  std::vector<double> PhotonSelectionCuts;
  options_->GetOpt("particle_flow","useEGPhotons",useEGPhotons_);
  options_->GetOpt("particle_flow","photonSelection", PhotonSelectionCuts);

  try {
    pfBlockAlgo_.setParameters( DPtovPtCut, 
				NHitCut,
				useIterTracking,
				useConvBremPFRecTracks_,
				nuclearInteractionsPurity,
				useEGPhotons_,
				PhotonSelectionCuts); 
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
  boost::shared_ptr<PFEnergyCalibration> 
    calibration( new PFEnergyCalibration() );
  calibration_ = calibration;

  bool usePFSCEleCalib;
  std::vector<double>  calibPFSCEle_Fbrem_barrel; 
  std::vector<double>  calibPFSCEle_Fbrem_endcap;
  std::vector<double>  calibPFSCEle_barrel;
  std::vector<double>  calibPFSCEle_endcap;
  options_->GetOpt("particle_flow","usePFSCEleCalib",usePFSCEleCalib);
  options_->GetOpt("particle_flow","calibPFSCEle_Fbrem_barrel",calibPFSCEle_Fbrem_barrel);
  options_->GetOpt("particle_flow","calibPFSCEle_Fbrem_endcap",calibPFSCEle_Fbrem_endcap);
  options_->GetOpt("particle_flow","calibPFSCEle_barrel",calibPFSCEle_barrel);
  options_->GetOpt("particle_flow","calibPFSCEle_endcap",calibPFSCEle_endcap);
  boost::shared_ptr<PFSCEnergyCalibration>  
    thePFSCEnergyCalibration (new PFSCEnergyCalibration(calibPFSCEle_Fbrem_barrel,calibPFSCEle_Fbrem_endcap,
							calibPFSCEle_barrel,calibPFSCEle_endcap ));
  
  bool useEGammaSupercluster;
  double sumEtEcalIsoForEgammaSC_barrel;
  double sumEtEcalIsoForEgammaSC_endcap;
  double coneEcalIsoForEgammaSC;
  double sumPtTrackIsoForEgammaSC_barrel;
  double sumPtTrackIsoForEgammaSC_endcap;
  unsigned int nTrackIsoForEgammaSC;
  double coneTrackIsoForEgammaSC;
  options_->GetOpt("particle_flow","useEGammaSupercluster",useEGammaSupercluster);
  options_->GetOpt("particle_flow","sumEtEcalIsoForEgammaSC_barrel",sumEtEcalIsoForEgammaSC_barrel);
  options_->GetOpt("particle_flow","sumEtEcalIsoForEgammaSC_endcap",sumEtEcalIsoForEgammaSC_endcap);
  options_->GetOpt("particle_flow","coneEcalIsoForEgammaSC",coneEcalIsoForEgammaSC);
  options_->GetOpt("particle_flow","sumPtTrackIsoForEgammaSC_barrel",sumPtTrackIsoForEgammaSC_barrel);
  options_->GetOpt("particle_flow","sumPtTrackIsoForEgammaSC_endcap",sumPtTrackIsoForEgammaSC_endcap);
  options_->GetOpt("particle_flow","nTrackIsoForEgammaSC",nTrackIsoForEgammaSC);
  options_->GetOpt("particle_flow","coneTrackIsoForEgammaSC",coneTrackIsoForEgammaSC);
  options_->GetOpt("particle_flow","useEGammaElectrons",useEGElectrons_);

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

  boost::shared_ptr<PFEnergyCalibrationHF>  thepfEnergyCalibrationHF
    ( new PFEnergyCalibrationHF(calibHF_use,calibHF_eta_step,calibHF_a_EMonly,calibHF_b_HADonly,calibHF_a_EMHAD,calibHF_b_EMHAD) ) ;

  thepfEnergyCalibrationHF_ = thepfEnergyCalibrationHF;


  //----------------------------------------
  double nSigmaECAL = 99999;
  options_->GetOpt("particle_flow", "nsigma_ECAL", nSigmaECAL);
  double nSigmaHCAL = 99999;
  options_->GetOpt("particle_flow", "nsigma_HCAL", nSigmaHCAL);

  try {
    pfAlgo_.setParameters( nSigmaECAL, nSigmaHCAL, 
                           calibration, thepfEnergyCalibrationHF_);
  }
  catch( std::exception& err ) {
    cerr<<"exception setting PFAlgo parameters: "
        <<err.what()<<". terminating."<<endl;
    delete this;
    exit(1);
  }

  std::vector<double> muonHCAL;
  std::vector<double> muonECAL;
  std::vector<double> muonHO;
  options_->GetOpt("particle_flow", "muon_HCAL", muonHCAL);
  options_->GetOpt("particle_flow", "muon_ECAL", muonECAL);
  options_->GetOpt("particle_flow", "muon_HO", muonHO);

  assert ( muonHCAL.size() == 2 && muonECAL.size() == 2 && muonHO.size() == 2);

  double nSigmaTRACK = 3.0;
  options_->GetOpt("particle_flow", "nsigma_TRACK", nSigmaTRACK);

  double ptError = 1.0;
  options_->GetOpt("particle_flow", "pt_error", ptError);
  
  std::vector<double> factors45;
  options_->GetOpt("particle_flow", "factors_45", factors45);
  assert ( factors45.size() == 2 );

  edm::ParameterSet iConfig;


  double maxDPtOPt;
  options_->GetOpt("particle_flow", "maxDPtOPt", maxDPtOPt);
  iConfig.addParameter<double>("maxDPtOPt",maxDPtOPt);

  int minTrackerHits;
  options_->GetOpt("particle_flow", "minTrackerHits", minTrackerHits);
  iConfig.addParameter<int>("minTrackerHits",minTrackerHits);

  int minPixelHits;
  options_->GetOpt("particle_flow", "minPixelHits", minPixelHits);
  iConfig.addParameter<int>("minPixelHits",minPixelHits);

  std::string trackQuality;
  options_->GetOpt("particle_flow", "trackQuality", trackQuality);
  iConfig.addParameter<std::string>("trackQuality",trackQuality);

  double ptErrorScale;
  options_->GetOpt("particle_flow", "ptErrorScale", ptErrorScale);
  iConfig.addParameter<double>("ptErrorScale",ptErrorScale);

  double eventFractionForCleaning;
  options_->GetOpt("particle_flow", "eventFractionForCleaning", eventFractionForCleaning);
  iConfig.addParameter<double>("eventFractionForCleaning",eventFractionForCleaning);

  double dzpv;
  options_->GetOpt("particle_flow", "dzPV", dzpv);
  iConfig.addParameter<double>("dzPV",dzpv);

  bool postMuonCleaning;
  options_->GetOpt("particle_flow", "postMuonCleaning", postMuonCleaning);
  iConfig.addParameter<bool>("postMuonCleaning",postMuonCleaning);

  double minPtForPostCleaning;
  options_->GetOpt("particle_flow", "minPtForPostCleaning", minPtForPostCleaning);
  iConfig.addParameter<double>("minPtForPostCleaning",minPtForPostCleaning);

  double eventFactorForCosmics;
  options_->GetOpt("particle_flow", "eventFactorForCosmics", eventFactorForCosmics);
  iConfig.addParameter<double>("eventFactorForCosmics",eventFactorForCosmics);
  
  double minSignificanceForCleaning;
  options_->GetOpt("particle_flow", "metSignificanceForCleaning", minSignificanceForCleaning);
  iConfig.addParameter<double>("metSignificanceForCleaning",minSignificanceForCleaning);

  double minSignificanceForRejection;
  options_->GetOpt("particle_flow", "metSignificanceForRejection", minSignificanceForRejection);
  iConfig.addParameter<double>("metSignificanceForRejection",minSignificanceForRejection);

  double metFactorForCleaning;
  options_->GetOpt("particle_flow", "metFactorForCleaning", metFactorForCleaning);
  iConfig.addParameter<double>("metFactorForCleaning",metFactorForCleaning);

  double eventFractionForRejection;
  options_->GetOpt("particle_flow", "eventFractionForRejection", eventFractionForRejection);
  iConfig.addParameter<double>("eventFractionForRejection",eventFractionForRejection);

  double metFactorForRejection;
  options_->GetOpt("particle_flow", "metFactorForRejection", metFactorForRejection);
  iConfig.addParameter<double>("metFactorForRejection",metFactorForRejection);

  double metFactorForHighEta;
  options_->GetOpt("particle_flow", "metFactorForHighEta", metFactorForHighEta);
  iConfig.addParameter<double>("metFactorForHighEta",metFactorForHighEta);

  double ptFactorForHighEta;
  options_->GetOpt("particle_flow", "ptFactorForHighEta", ptFactorForHighEta);
  iConfig.addParameter<double>("ptFactorForHighEta",ptFactorForHighEta);


  double metFactorForFakes;
  options_->GetOpt("particle_flow", "metFactorForFakes", metFactorForFakes);
  iConfig.addParameter<double>("metFactorForFakes",metFactorForFakes);

  double minMomentumForPunchThrough;
  options_->GetOpt("particle_flow", "minMomentumForPunchThrough", minMomentumForPunchThrough);
  iConfig.addParameter<double>("minMomentumForPunchThrough",minMomentumForPunchThrough);

  double minEnergyForPunchThrough;
  options_->GetOpt("particle_flow", "minEnergyForPunchThrough", minEnergyForPunchThrough);
  iConfig.addParameter<double>("minEnergyForPunchThrough",minEnergyForPunchThrough);


  double punchThroughFactor;
  options_->GetOpt("particle_flow", "punchThroughFactor", punchThroughFactor);
  iConfig.addParameter<double>("punchThroughFactor",punchThroughFactor);

  double punchThroughMETFactor;
  options_->GetOpt("particle_flow", "punchThroughMETFactor", punchThroughMETFactor);
  iConfig.addParameter<double>("punchThroughMETFactor",punchThroughMETFactor);


  double cosmicRejectionDistance;
  options_->GetOpt("particle_flow", "cosmicRejectionDistance", cosmicRejectionDistance);
  iConfig.addParameter<double>("cosmicRejectionDistance",cosmicRejectionDistance);

  try { 
    pfAlgo_.setPFMuonAndFakeParameters(iConfig);  
}
  catch( std::exception& err ) {
    cerr<<"exception setting PFAlgo Muon and Fake parameters: "
        <<err.what()<<". terminating."<<endl;
    delete this;
    exit(1);
  }
  
  bool postHFCleaning = true;
  options_->GetOpt("particle_flow", "postHFCleaning", postHFCleaning);
  double minHFCleaningPt = 5.;
  options_->GetOpt("particle_flow", "minHFCleaningPt", minHFCleaningPt);
  double minSignificance = 2.5;
  options_->GetOpt("particle_flow", "minSignificance", minSignificance);
  double maxSignificance = 2.5;
  options_->GetOpt("particle_flow", "maxSignificance", maxSignificance);
  double minSignificanceReduction = 1.4;
  options_->GetOpt("particle_flow", "minSignificanceReduction", minSignificanceReduction);
  double maxDeltaPhiPt = 7.0;
  options_->GetOpt("particle_flow", "maxDeltaPhiPt", maxDeltaPhiPt);
  double minDeltaMet = 0.4;
  options_->GetOpt("particle_flow", "minDeltaMet", minDeltaMet);

  // Set post HF cleaning muon parameters
  try { 
    pfAlgo_.setPostHFCleaningParameters(postHFCleaning,
					minHFCleaningPt,
					minSignificance,
					maxSignificance,
					minSignificanceReduction,
					maxDeltaPhiPt,
					minDeltaMet);
  }
  catch( std::exception& err ) {
    cerr<<"exception setting post HF cleaning parameters: "
        <<err.what()<<". terminating."<<endl;
    delete this;
    exit(1);
  }
  
  useAtHLT_ = false;
  options_->GetOpt("particle_flow", "useAtHLT", useAtHLT_);
  cout<<"use HLT tracking "<<useAtHLT_<<endl;

  useHO_ = true;
  options_->GetOpt("particle_flow", "useHO", useHO_);
  cout<<"use of HO "<<useHO_<<endl;


  usePFElectrons_ = false;   // set true to use PFElectrons
  options_->GetOpt("particle_flow", "usePFElectrons", usePFElectrons_);
  cout<<"use PFElectrons "<<usePFElectrons_<<endl;

  if( usePFElectrons_ ) { 
    // PFElectrons options -----------------------------
    double mvaEleCut = -1.;  // if = -1. get all the pre-id electrons
    options_->GetOpt("particle_flow", "electron_mvaCut", mvaEleCut);

    bool applyCrackCorrections=true;
    options_->GetOpt("particle_flow","electron_crackCorrection",applyCrackCorrections);

    string mvaWeightFileEleID = "";
    options_->GetOpt("particle_flow", "electronID_mvaWeightFile", 
		     mvaWeightFileEleID);
    mvaWeightFileEleID = expand(mvaWeightFileEleID);

    std::string egammaElectronstagname;
    options_->GetOpt("particle_flow","egammaElectrons",egammaElectronstagname);
    egammaElectronsTag_ =  edm::InputTag(egammaElectronstagname);
    
    //HO in the algorithm or not
    pfBlockAlgo_.setHOTag(useHO_);
    pfAlgo_.setHOTag(useHO_);

    try { 
      pfAlgo_.setPFEleParameters(mvaEleCut,
				 mvaWeightFileEleID,
				 usePFElectrons_,
				 thePFSCEnergyCalibration,
				 calibration,
				 sumEtEcalIsoForEgammaSC_barrel,
				 sumEtEcalIsoForEgammaSC_endcap,
				 coneEcalIsoForEgammaSC,
				 sumPtTrackIsoForEgammaSC_barrel,
				 sumPtTrackIsoForEgammaSC_endcap,
				 nTrackIsoForEgammaSC,
				 coneTrackIsoForEgammaSC,
				 applyCrackCorrections,
				 usePFSCEleCalib,
				 useEGElectrons_,
				 useEGammaSupercluster);
    }
    catch( std::exception& err ) {
      cerr<<"exception setting PFAlgo Electron parameters: "
	  <<err.what()<<". terminating."<<endl;
      delete this;
      exit(1);
    }
  }

  bool usePFPhotons = true;
  bool useReg=false;
  string mvaWeightFileConvID = "";
  string mvaWeightFileRegLCEB="";
  string mvaWeightFileRegLCEE="";    
  string mvaWeightFileRegGCEB="";
  string mvaWeightFileRegGCEEhr9="";
  string mvaWeightFileRegGCEElr9="";
  string mvaWeightFileRegRes="";
  string X0Map="";
  double mvaConvCut=-1.;
  double sumPtTrackIsoForPhoton=2.0;
  double sumPtTrackIsoSlopeForPhoton=0.001;
  options_->GetOpt("particle_flow", "usePFPhotons", usePFPhotons);
  options_->GetOpt("particle_flow", "conv_mvaCut", mvaConvCut);
  options_->GetOpt("particle_flow", "useReg", useReg);
  options_->GetOpt("particle_flow", "convID_mvaWeightFile", mvaWeightFileConvID);
  options_->GetOpt("particle_flow", "mvaWeightFileRegLCEB", mvaWeightFileRegLCEB);
  options_->GetOpt("particle_flow", "mvaWeightFileRegLCEE", mvaWeightFileRegLCEE);
  options_->GetOpt("particle_flow", "mvaWeightFileRegGCEB", mvaWeightFileRegGCEB);
  options_->GetOpt("particle_flow", "mvaWeightFileRegGCEEHr9", mvaWeightFileRegGCEEhr9);
  options_->GetOpt("particle_flow", "mvaWeightFileRegGCEELr9", mvaWeightFileRegGCEElr9);
  options_->GetOpt("particle_flow", "mvaWeightFileRegRes", mvaWeightFileRegRes);
  options_->GetOpt("particle_flow", "X0Map", X0Map);
  options_->GetOpt("particle_flow","sumPtTrackIsoForPhoton",sumPtTrackIsoForPhoton);
  options_->GetOpt("particle_flow","sumPtTrackIsoSlopeForPhoton",sumPtTrackIsoSlopeForPhoton);
  // cout<<"use PFPhotons "<<usePFPhotons<<endl;

  if( usePFPhotons ) { 
    // PFPhoton options -----------------------------
    TFile *infile_PFLCEB = new TFile(mvaWeightFileRegLCEB.c_str(),"READ");
    TFile *infile_PFLCEE = new TFile(mvaWeightFileRegLCEE.c_str(),"READ");
    TFile *infile_PFGCEB = new TFile(mvaWeightFileRegGCEB.c_str(),"READ");
    TFile *infile_PFGCEEhR9 = new TFile(mvaWeightFileRegGCEEhr9.c_str(),"READ");
    TFile *infile_PFGCEElR9 = new TFile(mvaWeightFileRegGCEElr9.c_str(),"READ");
    TFile *infile_PFRes = new TFile(mvaWeightFileRegRes.c_str(),"READ");
   
    const GBRForest *gbrLCBar = (GBRForest*)infile_PFLCEB->Get("PFLCorrEB");
    const GBRForest *gbrLCEnd = (GBRForest*)infile_PFLCEE->Get("PFLCorrEE");
    const GBRForest *gbrGCEB = (GBRForest*)infile_PFGCEB->Get("PFGCorrEB");
    const GBRForest *gbrGCEEhr9 = (GBRForest*)infile_PFGCEEhR9->Get("PFGCorrEEHr9");
    const GBRForest *gbrGCEElr9 = (GBRForest*)infile_PFGCEElR9->Get("PFGCorrEELr9");
    const GBRForest *gbrRes = (GBRForest*)infile_PFRes->Get("PFRes");
    try { 
      pfAlgo_.setPFPhotonParameters
	(usePFPhotons,
	 mvaWeightFileConvID,
	 mvaConvCut,
	 useReg,
	 X0Map,
	 calibration,
	 sumPtTrackIsoForPhoton,
	 sumPtTrackIsoSlopeForPhoton
	 );
      pfAlgo_.setPFPhotonRegWeights(gbrLCBar, gbrLCEnd,gbrGCEB,
				    gbrGCEEhr9,gbrGCEElr9,
				    gbrRes
				    );
      
    }
    catch( std::exception& err ) {
      cerr<<"exception setting PFAlgo Photon parameters: "
	  <<err.what()<<". terminating."<<endl;
      delete this;
      exit(1);
    }
  }



  bool rejectTracks_Bad = true;
  bool rejectTracks_Step45 = true;
  bool usePFConversions = false;   // set true to use PFConversions
  bool usePFNuclearInteractions = false;
  bool usePFV0s = false;


  double dptRel_DispVtx = 10;
  

  options_->GetOpt("particle_flow", "usePFConversions", usePFConversions);
  options_->GetOpt("particle_flow", "usePFV0s", usePFV0s);
  options_->GetOpt("particle_flow", "usePFNuclearInteractions", usePFNuclearInteractions);
  options_->GetOpt("particle_flow", "dptRel_DispVtx",  dptRel_DispVtx);

  try { 
    pfAlgo_.setDisplacedVerticesParameters(rejectTracks_Bad,
					   rejectTracks_Step45,
					   usePFNuclearInteractions,
					   usePFConversions,
					   usePFV0s,
					   dptRel_DispVtx);

  }
  catch( std::exception& err ) {
    cerr<<"exception setting PFAlgo displaced vertex parameters: "
        <<err.what()<<". terminating."<<endl;
    delete this;
    exit(1);
  }

  bool bCorrect = false;
  bool bCalibPrimary = false;
  double dptRel_PrimaryTrack = 0;
  double dptRel_MergedTrack = 0;
  double ptErrorSecondary = 0;
  vector<double> nuclCalibFactors;

  options_->GetOpt("particle_flow", "bCorrect", bCorrect);
  options_->GetOpt("particle_flow", "bCalibPrimary", bCalibPrimary);
  options_->GetOpt("particle_flow", "dptRel_PrimaryTrack", dptRel_PrimaryTrack);
  options_->GetOpt("particle_flow", "dptRel_MergedTrack", dptRel_MergedTrack);
  options_->GetOpt("particle_flow", "ptErrorSecondary", ptErrorSecondary);
  options_->GetOpt("particle_flow", "nuclCalibFactors", nuclCalibFactors);

  try { 
    pfAlgo_.setCandConnectorParameters(bCorrect, bCalibPrimary, dptRel_PrimaryTrack, dptRel_MergedTrack, ptErrorSecondary, nuclCalibFactors);
  }
  catch( std::exception& err ) {
    cerr<<"exception setting PFAlgo cand connector parameters: "
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
  printRecHitsEMin_ = 0.;
  options_->GetOpt("print", "rechits", printRecHits_ );
  options_->GetOpt("print", "rechits_emin", printRecHitsEMin_ );
  
  printClusters_ = false;
  printClustersEMin_ = 0.;
  options_->GetOpt("print", "clusters", printClusters_ );
  options_->GetOpt("print", "clusters_emin", printClustersEMin_ );

  printPFBlocks_ = false;
  options_->GetOpt("print", "PFBlocks", printPFBlocks_ );
  
  printPFCandidates_ = true;
  printPFCandidatesPtMin_ = 0.;
  options_->GetOpt("print", "PFCandidates", printPFCandidates_ );
  options_->GetOpt("print", "PFCandidates_ptmin", printPFCandidatesPtMin_ );
  
  printPFJets_ = true;
  printPFJetsPtMin_ = 0.;
  options_->GetOpt("print", "jets", printPFJets_ );
  options_->GetOpt("print", "jets_ptmin", printPFJetsPtMin_ );
 
  printSimParticles_ = true;
  printSimParticlesPtMin_ = 0.;
  options_->GetOpt("print", "simParticles", printSimParticles_ );
  options_->GetOpt("print", "simParticles_ptmin", printSimParticlesPtMin_ );

  printGenParticles_ = true;
  printGenParticlesPtMin_ = 0.;
  options_->GetOpt("print", "genParticles", printGenParticles_ );
  options_->GetOpt("print", "genParticles_ptmin", printGenParticlesPtMin_ );

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

  cout<<"Opening input root files"<<endl;

  options_->GetOpt("root","file", inFileNames_);
  


  try {
    AutoLibraryLoader::enable();
  }
  catch(string& err) {
    cout<<err<<endl;
  }

  ev_ = new fwlite::ChainEvent(inFileNames_);


  if ( !ev_ || !ev_->isValid() ) { 
    cout << "The rootfile(s) " << endl;
    for ( unsigned int ifile=0; ifile<inFileNames_.size(); ++ifile ) 
      std::cout << " - " << inFileNames_[ifile] << std::endl;
    cout << " is (are) not valid file(s) to open" << endl;
    return;
  } else { 
    cout << "The rootfile(s) : " << endl;
    for ( unsigned int ifile=0; ifile<inFileNames_.size(); ++ifile ) 
      std::cout << " - " << inFileNames_[ifile] << std::endl;
    cout<<" are opened with " << ev_->size() << " events." <<endl;
  }
  
  // hits branches ----------------------------------------------
  std::string rechitsECALtagname;
  options_->GetOpt("root","rechits_ECAL_inputTag", rechitsECALtagname);
  rechitsECALTag_ = edm::InputTag(rechitsECALtagname);

  std::string rechitsHCALtagname;
  options_->GetOpt("root","rechits_HCAL_inputTag", rechitsHCALtagname);
  rechitsHCALTag_ = edm::InputTag(rechitsHCALtagname);

  std::string rechitsHOtagname;
  options_->GetOpt("root","rechits_HO_inputTag", rechitsHOtagname);
  rechitsHOTag_ = edm::InputTag(rechitsHOtagname);

  std::string rechitsHFEMtagname;
  options_->GetOpt("root","rechits_HFEM_inputTag", rechitsHFEMtagname);
  rechitsHFEMTag_ = edm::InputTag(rechitsHFEMtagname);

  std::string rechitsHFHADtagname;
  options_->GetOpt("root","rechits_HFHAD_inputTag", rechitsHFHADtagname);
  rechitsHFHADTag_ = edm::InputTag(rechitsHFHADtagname);

  std::vector<string> rechitsCLEANEDtagnames;
  options_->GetOpt("root","rechits_CLEANED_inputTags", rechitsCLEANEDtagnames);
  for ( unsigned tags = 0; tags<rechitsCLEANEDtagnames.size(); ++tags )
    rechitsCLEANEDTags_.push_back(edm::InputTag(rechitsCLEANEDtagnames[tags]));
  rechitsCLEANEDV_.resize(rechitsCLEANEDTags_.size());
  rechitsCLEANEDHandles_.resize(rechitsCLEANEDTags_.size());


  // Tracks branches
  std::string rechitsPStagname;
  options_->GetOpt("root","rechits_PS_inputTag", rechitsPStagname);
  rechitsPSTag_ = edm::InputTag(rechitsPStagname);

  std::string recTrackstagname;
  options_->GetOpt("root","recTracks_inputTag", recTrackstagname);
  recTracksTag_ = edm::InputTag(recTrackstagname);

  std::string displacedRecTrackstagname;
  options_->GetOpt("root","displacedRecTracks_inputTag", displacedRecTrackstagname);
  displacedRecTracksTag_ = edm::InputTag(displacedRecTrackstagname);

  std::string primaryVerticestagname;
  options_->GetOpt("root","primaryVertices_inputTag", primaryVerticestagname);
  primaryVerticesTag_ = edm::InputTag(primaryVerticestagname);

  std::string stdTrackstagname;
  options_->GetOpt("root","stdTracks_inputTag", stdTrackstagname);
  stdTracksTag_ = edm::InputTag(stdTrackstagname);

  std::string gsfrecTrackstagname;
  options_->GetOpt("root","gsfrecTracks_inputTag", gsfrecTrackstagname);
  gsfrecTracksTag_ = edm::InputTag(gsfrecTrackstagname);

  useConvBremGsfTracks_ = false;
  options_->GetOpt("particle_flow", "useConvBremGsfTracks", useConvBremGsfTracks_);
  if ( useConvBremGsfTracks_ ) { 
    std::string convBremGsfrecTrackstagname;
    options_->GetOpt("root","convBremGsfrecTracks_inputTag", convBremGsfrecTrackstagname);
    convBremGsfrecTracksTag_ = edm::InputTag(convBremGsfrecTrackstagname);
  }

  useConvBremPFRecTracks_ = false;
  options_->GetOpt("particle_flow", "useConvBremPFRecTracks", useConvBremPFRecTracks_);


  // muons branch
  std::string muonstagname;
  options_->GetOpt("root","muon_inputTag", muonstagname);
  muonsTag_ = edm::InputTag(muonstagname);

  // conversion
  usePFConversions_=false;
  options_->GetOpt("particle_flow", "usePFConversions", usePFConversions_);
  if( usePFConversions_ ) {
    std::string conversiontagname;
    options_->GetOpt("root","conversion_inputTag", conversiontagname);
    conversionTag_ = edm::InputTag(conversiontagname);
  }

  // V0
  usePFV0s_=false;
  options_->GetOpt("particle_flow", "usePFV0s", usePFV0s_);
  if( usePFV0s_ ) {
    std::string v0tagname;
    options_->GetOpt("root","V0_inputTag", v0tagname);
    v0Tag_ = edm::InputTag(v0tagname);
  }

  // Photons
  std::string photontagname;
  options_->GetOpt("root","Photon_inputTag",photontagname);
  photonTag_ = edm::InputTag(photontagname);

 //Displaced Vertices
  usePFNuclearInteractions_=false;
  options_->GetOpt("particle_flow", "usePFNuclearInteractions", usePFNuclearInteractions_);
  if( usePFNuclearInteractions_ ) {
    std::string pfNuclearTrackerVertextagname;
    options_->GetOpt("root","PFDisplacedVertex_inputTag", pfNuclearTrackerVertextagname);
    pfNuclearTrackerVertexTag_ = edm::InputTag(pfNuclearTrackerVertextagname);
  }

  std::string trueParticlestagname;
  options_->GetOpt("root","trueParticles_inputTag", trueParticlestagname);
  trueParticlesTag_ = edm::InputTag(trueParticlestagname);

  std::string MCTruthtagname;
  options_->GetOpt("root","MCTruth_inputTag", MCTruthtagname);
  MCTruthTag_ = edm::InputTag(MCTruthtagname);

  std::string caloTowerstagname;
  options_->GetOpt("root","caloTowers_inputTag", caloTowerstagname);
  caloTowersTag_ = edm::InputTag(caloTowerstagname);

  std::string genJetstagname;
  options_->GetOpt("root","genJets_inputTag", genJetstagname);
  genJetsTag_ = edm::InputTag(genJetstagname);

  
  std::string genParticlesforMETtagname;
  options_->GetOpt("root","genParticlesforMET_inputTag", genParticlesforMETtagname);
  genParticlesforMETTag_ = edm::InputTag(genParticlesforMETtagname);

  std::string genParticlesforJetstagname;
  options_->GetOpt("root","genParticlesforJets_inputTag", genParticlesforJetstagname);
  genParticlesforJetsTag_ = edm::InputTag(genParticlesforJetstagname);

  // PF candidates 
  std::string pfCandidatetagname;
  options_->GetOpt("root","particleFlowCand_inputTag", pfCandidatetagname);
  pfCandidateTag_ = edm::InputTag(pfCandidatetagname);

  std::string caloJetstagname;
  options_->GetOpt("root","CaloJets_inputTag", caloJetstagname);
  caloJetsTag_ = edm::InputTag(caloJetstagname);

  std::string corrcaloJetstagname;
  options_->GetOpt("root","corrCaloJets_inputTag", corrcaloJetstagname);
  corrcaloJetsTag_ = edm::InputTag(corrcaloJetstagname);

  std::string pfJetstagname;
  options_->GetOpt("root","PFJets_inputTag", pfJetstagname);
  pfJetsTag_ = edm::InputTag(pfJetstagname);

  std::string pfMetstagname;
  options_->GetOpt("root","PFMET_inputTag", pfMetstagname);
  pfMetsTag_ = edm::InputTag(pfMetstagname);

  std::string caloMetstagname;
  options_->GetOpt("root","CaloMET_inputTag", caloMetstagname);
  caloMetsTag_ = edm::InputTag(caloMetstagname);

  std::string tcMetstagname;
  options_->GetOpt("root","TCMET_inputTag", tcMetstagname);
  tcMetsTag_ = edm::InputTag(tcMetstagname);

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
  clusterAlgoECAL_.write();
  clusterAlgoHCAL_.write();
  clusterAlgoHO_.write();
  clusterAlgoPS_.write();
  clusterAlgoHFEM_.write();
  clusterAlgoHFHAD_.write();
  
  // Addition to have DQM histograms : by S. Dutta                                                                                                       
  if (doPFDQM_) {
    cout << " Writing DQM root file " << endl;
    pfJetMonitor_.write();
    pfMETMonitor_.write();
    dqmFile_->Write();
  }
  //-----------------------------------------------                                                                                                           
  if(outFile_) {
    outFile_->Write();
//     outFile_->cd(); 
//     // write histos here
//     cout<<"writing output to "<<outFile_->GetName()<<endl;
//     h_deltaETvisible_MCEHT_->Write();
//     h_deltaETvisible_MCPF_->Write();
//     if(outTree_) outTree_->Write();
//     if(doPFCandidateBenchmark_) pfCandidateBenchmark_.write();
  }
}


int PFRootEventManager::eventToEntry(int run, int lumi, int event) const {
  
  RunsMap::const_iterator iR = mapEventToEntry_.find( run );
  if( iR != mapEventToEntry_.end() ) {
    LumisMap::const_iterator iL = iR->second.find( lumi );
    if( iL != iR->second.end() ) {
      EventToEntry::const_iterator iE = iL->second.find( event );
      if( iE != iL->second.end() ) {
	return iE->second;
      }  
      else {
	cout<<"event "<<event<<" not found in run "<<run<<", lumi "<<lumi<<endl;
      }
    }
    else {
      cout<<"lumi "<<lumi<<" not found in run "<<run<<endl;
    }
  }
  else{
    cout<<"run "<<run<<" not found"<<endl;
  }
  return -1;    
}

bool PFRootEventManager::processEvent(int run, int lumi, int event) {

  int entry = eventToEntry(run, lumi, event);
  if( entry < 0 ) {
    cout<<"event "<<event<<" is not present, sorry."<<endl;
    return false;
  }
  else
    return processEntry( entry ); 
} 


bool PFRootEventManager::processEntry(int entry) {

  reset();

  iEvent_ = entry;

  bool exists = ev_->to(entry);
  if ( !exists ) { 
    std::cout << "Entry " << entry << " does not exist " << std::endl; 
    return false;
  }
  const edm::EventBase& iEvent = *ev_;

  if( outEvent_ ) outEvent_->setNumber(entry);

  if(verbosity_ == VERBOSE  || 
     //entry < 10000 ||
     entry < 10 ||
     (entry < 100 && entry%10 == 0) || 
     (entry < 1000 && entry%100 == 0) || 
     entry%1000 == 0 ) 
    cout<<"process entry "<< entry 
	<<", run "<<iEvent.id().run()
	<<", lumi "<<iEvent.id().luminosityBlock()	
	<<", event:"<<iEvent.id().event()
	<< endl;

  //ev_->getTFile()->cd();

  bool goodevent =  readFromSimulation(entry);

  /* 
  std::cout << "Rechits cleaned : " << std::endl;
  for(unsigned i=0; i<rechitsCLEANED_.size(); i++) {
    string seedstatus = "    ";
    printRecHit(rechitsCLEANED_[i], i, seedstatus.c_str());
  }
  */

  if(verbosity_ == VERBOSE ) {
    cout<<"number of vertices             : "<<primaryVertices_.size()<<endl;
    cout<<"number of recTracks            : "<<recTracks_.size()<<endl;
    cout<<"number of gsfrecTracks         : "<<gsfrecTracks_.size()<<endl;
    cout<<"number of convBremGsfrecTracks : "<<convBremGsfrecTracks_.size()<<endl;
    cout<<"number of muons                : "<<muons_.size()<<endl;
    cout<<"number of displaced vertices   : "<<pfNuclearTrackerVertex_.size()<<endl;
    cout<<"number of conversions          : "<<conversion_.size()<<endl;
    cout<<"number of v0                   : "<<v0_.size()<<endl;
    cout<<"number of stdTracks            : "<<stdTracks_.size()<<endl;
    cout<<"number of true particles       : "<<trueParticles_.size()<<endl;
    cout<<"number of ECAL rechits         : "<<rechitsECAL_.size()<<endl;
    cout<<"number of HCAL rechits         : "<<rechitsHCAL_.size()<<endl;
    cout<<"number of HO rechits           : "<<rechitsHO_.size()<<endl;
    cout<<"number of HFEM rechits         : "<<rechitsHFEM_.size()<<endl;
    cout<<"number of HFHAD rechits        : "<<rechitsHFHAD_.size()<<endl;
    cout<<"number of HF Cleaned rechits   : "<<rechitsCLEANED_.size()<<endl;
    cout<<"number of PS rechits           : "<<rechitsPS_.size()<<endl;
  }  

  if( doClustering_ ) {
    clustering(); 

  } else if( verbosity_ == VERBOSE ) {
    cout<<"clustering is OFF - clusters come from the input file"<<endl; 
  }

  if(verbosity_ == VERBOSE ) {
    if(clustersECAL_.get() ) {
      cout<<"number of ECAL clusters : "<<clustersECAL_->size()<<endl;
    }
    if(clustersHCAL_.get() ) {
      cout<<"number of HCAL clusters : "<<clustersHCAL_->size()<<endl;
    }

    if(useHO_ && clustersHO_.get() ) {
      cout<<"number of HO clusters : "<<clustersHO_->size()<<endl;
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

  if(doParticleFlow_) { 
    particleFlow();
    if (doCompare_) pfCandCompare(entry);
  }

  if(doJets_) {
    reconstructGenJets();
    reconstructCaloJets();
    reconstructPFJets();
  }    

  // call print() in verbose mode
  if( verbosity_ == VERBOSE ) print();
  
  //COLIN the PFJet and PFMET benchmarks are very messy. 
  //COLIN    compare with the filling of the PFCandidateBenchmark, which is one line. 
  
  goodevent = eventAccepted(); 

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
    /*
    if ( fabs(resPt) > 0.4 )
      std::cout << "'" << iEvent.id().run() << ":" << iEvent.id().event() << "-" 
		<< iEvent.id().run() << ":" << iEvent.id().event() << "'," << std::endl;
    */
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
  
  // Addition to have DQM histograms : by S. Dutta 
  reco::MET reComputedMet_;    
  reco::MET computedGenMet_;
  //-----------------------------------------------

  //COLIN would  be nice to move this long code to a separate function. 
  // is it necessary to re-set everything at each event?? 
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

  if( goodevent && doPFCandidateBenchmark_ ) {
    pfCandidateManager_.fill( *pfCandidates_, genParticlesCMSSW_);
  }
  
  // Addition to have DQM histograms : by S. Dutta                                                                                                          
  if( goodevent && doPFDQM_ ) {
    float deltaMin, deltaMax;
    pfJetMonitor_.fill( pfJets_, genJets_, deltaMin, deltaMax);
    if (doPFMETBenchmark_) {
      pfMETMonitor_.fillOne( reComputedMet_, computedGenMet_, deltaMin, deltaMax);
    }
  }
  //-----------------------------------------------                                                                    
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



bool PFRootEventManager::eventAccepted() const {
  // return highPtJet(10); 
  //return highPtPFCandidate( 10, PFCandidate::h ); 
  return true;
} 

bool PFRootEventManager::highPtJet(double ptMin) const {
  for( unsigned i=0; i<pfJets_.size(); ++i) {
    if( pfJets_[i].pt() > ptMin ) return true;
  }
  return false;
}

bool PFRootEventManager::highPtPFCandidate( double ptMin, 
					    PFCandidate::ParticleType type) const {
  for( unsigned i=0; i<pfCandidates_->size(); ++i) {

    const PFCandidate& pfc = (*pfCandidates_)[i];
    if(type!= PFCandidate::X &&  
       pfc.particleId() != type ) continue;
    if( pfc.pt() > ptMin ) return true;
  }
  return false;
}


bool PFRootEventManager::readFromSimulation(int entry) {

  if (verbosity_ == VERBOSE ) {
    cout <<"start reading from simulation"<<endl;
  }


  // if(!tree_) return false;
  
  const edm::EventBase& iEvent = *ev_;
  

  bool foundstdTracks = iEvent.getByLabel(stdTracksTag_,stdTracksHandle_);
  if ( foundstdTracks ) { 
    stdTracks_ = *stdTracksHandle_;
    // cout << "Found " << stdTracks_.size() << " standard tracks" << endl;
  } else { 
    cerr<<"PFRootEventManager::ProcessEntry : stdTracks Collection not found : "
        <<entry << " " << stdTracksTag_<<endl;
  }

  bool foundMCTruth = iEvent.getByLabel(MCTruthTag_,MCTruthHandle_);
  if ( foundMCTruth ) { 
    MCTruth_ = *MCTruthHandle_;
    // cout << "Found MC truth" << endl;
  } else { 
    // cerr<<"PFRootEventManager::ProcessEntry : MCTruth Collection not found : "
    //    <<entry << " " << MCTruthTag_<<endl;
  }

  bool foundTP = iEvent.getByLabel(trueParticlesTag_,trueParticlesHandle_);
  if ( foundTP ) { 
    trueParticles_ = *trueParticlesHandle_;
    // cout << "Found " << trueParticles_.size() << " true particles" << endl;
  } else { 
    //cerr<<"PFRootEventManager::ProcessEntry : trueParticles Collection not found : "
    //    <<entry << " " << trueParticlesTag_<<endl;
  }

  bool foundECAL = iEvent.getByLabel(rechitsECALTag_,rechitsECALHandle_);
  if ( foundECAL ) { 
    rechitsECAL_ = *rechitsECALHandle_;
    // cout << "Found " << rechitsECAL_.size() << " ECAL rechits" << endl;
  } else { 
    cerr<<"PFRootEventManager::ProcessEntry : rechitsECAL Collection not found : "
        <<entry << " " << rechitsECALTag_<<endl;
  }

  bool foundHCAL = iEvent.getByLabel(rechitsHCALTag_,rechitsHCALHandle_);
  if ( foundHCAL ) { 
    rechitsHCAL_ = *rechitsHCALHandle_;
    // cout << "Found " << rechitsHCAL_.size() << " HCAL rechits" << endl;
  } else { 
    cerr<<"PFRootEventManager::ProcessEntry : rechitsHCAL Collection not found : "
        <<entry << " " << rechitsHCALTag_<<endl;
  }

  if (useHO_) {
    bool foundHO = iEvent.getByLabel(rechitsHOTag_,rechitsHOHandle_);
    if ( foundHO ) { 
      rechitsHO_ = *rechitsHOHandle_;
      // cout << "Found " << rechitsHO_.size() << " HO rechits" << endl;
    } else { 
      cerr<<"PFRootEventManager::ProcessEntry : rechitsHO Collection not found : "
	  <<entry << " " << rechitsHOTag_<<endl;
    }
  }

  bool foundHFEM = iEvent.getByLabel(rechitsHFEMTag_,rechitsHFEMHandle_);
  if ( foundHFEM ) { 
    rechitsHFEM_ = *rechitsHFEMHandle_;
    // cout << "Found " << rechitsHFEM_.size() << " HFEM rechits" << endl;
  } else { 
    cerr<<"PFRootEventManager::ProcessEntry : rechitsHFEM Collection not found : "
        <<entry << " " << rechitsHFEMTag_<<endl;
  }

  bool foundHFHAD = iEvent.getByLabel(rechitsHFHADTag_,rechitsHFHADHandle_);
  if ( foundHFHAD ) { 
    rechitsHFHAD_ = *rechitsHFHADHandle_;
    // cout << "Found " << rechitsHFHAD_.size() << " HFHAD rechits" << endl;
  } else { 
    cerr<<"PFRootEventManager::ProcessEntry : rechitsHFHAD Collection not found : "
        <<entry << " " << rechitsHFHADTag_<<endl;
  }

  for ( unsigned i=0; i<rechitsCLEANEDTags_.size(); ++i ) { 
    bool foundCLEANED = iEvent.getByLabel(rechitsCLEANEDTags_[i],
					  rechitsCLEANEDHandles_[i]);
    if ( foundCLEANED ) { 
      rechitsCLEANEDV_[i] = *(rechitsCLEANEDHandles_[i]);
      // cout << "Found " << rechitsCLEANEDV_[i].size() << " CLEANED rechits" << endl;
    } else { 
      cerr<<"PFRootEventManager::ProcessEntry : rechitsCLEANED Collection not found : "
	  <<entry << " " << rechitsCLEANEDTags_[i]<<endl;
    }

  }

  bool foundPS = iEvent.getByLabel(rechitsPSTag_,rechitsPSHandle_);
  if ( foundPS ) { 
    rechitsPS_ = *rechitsPSHandle_;
    // cout << "Found " << rechitsPS_.size() << " PS rechits" << endl;
  } else { 
    cerr<<"PFRootEventManager::ProcessEntry : rechitsPS Collection not found : "
        <<entry << " " << rechitsPSTag_<<endl;
  }

  bool foundCT = iEvent.getByLabel(caloTowersTag_,caloTowersHandle_);
  if ( foundCT ) { 
    caloTowers_ = *caloTowersHandle_;
    // cout << "Found " << caloTowers_.size() << " calo Towers" << endl;
  } else { 
    cerr<<"PFRootEventManager::ProcessEntry : caloTowers Collection not found : "
        <<entry << " " << caloTowersTag_<<endl;
  }

  bool foundPV = iEvent.getByLabel(primaryVerticesTag_,primaryVerticesHandle_);
  if ( foundPV ) { 
    primaryVertices_ = *primaryVerticesHandle_;
    // cout << "Found " << primaryVertices_.size() << " primary vertices" << endl;
  } else { 
    cerr<<"PFRootEventManager::ProcessEntry : primaryVertices Collection not found : "
        <<entry << " " << primaryVerticesTag_<<endl;
  }


  bool foundPFV = iEvent.getByLabel(pfNuclearTrackerVertexTag_,pfNuclearTrackerVertexHandle_);
  if ( foundPFV ) { 
    pfNuclearTrackerVertex_ = *pfNuclearTrackerVertexHandle_;
    // cout << "Found " << pfNuclearTrackerVertex_.size() << " secondary PF vertices" << endl;
  } else if ( usePFNuclearInteractions_ ) { 
    cerr<<"PFRootEventManager::ProcessEntry : pfNuclearTrackerVertex Collection not found : "
        <<entry << " " << pfNuclearTrackerVertexTag_<<endl;
  }

  bool foundrecTracks = iEvent.getByLabel(recTracksTag_,recTracksHandle_);
  if ( foundrecTracks ) { 
    recTracks_ = *recTracksHandle_;
    // cout << "Found " << recTracks_.size() << " PFRecTracks" << endl;
  } else { 
    cerr<<"PFRootEventManager::ProcessEntry : recTracks Collection not found : "
        <<entry << " " << recTracksTag_<<endl;
  }

  bool founddisplacedRecTracks = iEvent.getByLabel(displacedRecTracksTag_,displacedRecTracksHandle_);
  if ( founddisplacedRecTracks ) { 
    displacedRecTracks_ = *displacedRecTracksHandle_;
    // cout << "Found " << displacedRecTracks_.size() << " PFRecTracks" << endl;
  } else { 
    cerr<<"PFRootEventManager::ProcessEntry : displacedRecTracks Collection not found : "
        <<entry << " " << displacedRecTracksTag_<<endl;
  }


  bool foundgsfrecTracks = iEvent.getByLabel(gsfrecTracksTag_,gsfrecTracksHandle_);
  if ( foundgsfrecTracks ) { 
    gsfrecTracks_ = *gsfrecTracksHandle_;
    // cout << "Found " << gsfrecTracks_.size() << " GsfPFRecTracks" << endl;
  } else { 
    cerr<<"PFRootEventManager::ProcessEntry : gsfrecTracks Collection not found : "
        <<entry << " " << gsfrecTracksTag_<<endl;
  }

  bool foundconvBremGsfrecTracks = iEvent.getByLabel(convBremGsfrecTracksTag_,convBremGsfrecTracksHandle_);
  if ( foundconvBremGsfrecTracks ) { 
    convBremGsfrecTracks_ = *convBremGsfrecTracksHandle_;
    // cout << "Found " << convBremGsfrecTracks_.size() << " ConvBremGsfPFRecTracks" << endl;
  } else if ( useConvBremGsfTracks_ ) { 
    cerr<<"PFRootEventManager::ProcessEntry : convBremGsfrecTracks Collection not found : "
        <<entry << " " << convBremGsfrecTracksTag_<<endl;
  }

  bool foundmuons = iEvent.getByLabel(muonsTag_,muonsHandle_);
  if ( foundmuons ) { 
    muons_ = *muonsHandle_;
    /*
    cout << "Found " << muons_.size() << " muons" << endl;
    for ( unsigned imu=0; imu<muons_.size(); ++imu ) { 
      std::cout << " Muon " << imu << ":" << std::endl;
      reco::MuonRef muonref( &muons_, imu );
      PFMuonAlgo::printMuonProperties(muonref);
    }
    */
  } else { 
    cerr<<"PFRootEventManager::ProcessEntry : muons Collection not found : "
        <<entry << " " << muonsTag_<<endl;
  }

  bool foundconversion = iEvent.getByLabel(conversionTag_,conversionHandle_);
  if ( foundconversion ) { 
    conversion_ = *conversionHandle_;
    // cout << "Found " << conversion_.size() << " conversion" << endl;
  } else if ( usePFConversions_ ) { 
    cerr<<"PFRootEventManager::ProcessEntry : conversion Collection not found : "
        <<entry << " " << conversionTag_<<endl;
  }



  bool foundv0 = iEvent.getByLabel(v0Tag_,v0Handle_);
  if ( foundv0 ) { 
    v0_ = *v0Handle_;
    // cout << "Found " << v0_.size() << " v0" << endl;
  } else if ( usePFV0s_ ) { 
    cerr<<"PFRootEventManager::ProcessEntry : v0 Collection not found : "
        <<entry << " " << v0Tag_<<endl;
  }

  if(useEGPhotons_) {
    bool foundPhotons = iEvent.getByLabel(photonTag_,photonHandle_);
    if ( foundPhotons) {
      photons_ = *photonHandle_;    
    } else {
      cerr <<"PFRootEventManager::ProcessEntry : photon collection not found : " 
	   << entry << " " << photonTag_ << endl;
    }
  }

  if(useEGElectrons_) {
    bool foundElectrons = iEvent.getByLabel(egammaElectronsTag_,egammaElectronHandle_);
    if ( foundElectrons) {
      //      std::cout << " Found collection " << std::endl;
      egammaElectrons_ = *egammaElectronHandle_;
    } else
      {
	cerr <<"PFRootEventManager::ProcessEntry : electron collection not found : "
	     << entry << " " << egammaElectronsTag_ << endl;
      }
  }

  bool foundgenJets = iEvent.getByLabel(genJetsTag_,genJetsHandle_);
  if ( foundgenJets ) { 
    genJetsCMSSW_ = *genJetsHandle_;
    // cout << "Found " << genJetsCMSSW_.size() << " genJets" << endl;
  } else { 
    //cerr<<"PFRootEventManager::ProcessEntry : genJets Collection not found : "
    //    <<entry << " " << genJetsTag_<<endl;
  }

  bool foundgenParticlesforJets = iEvent.getByLabel(genParticlesforJetsTag_,genParticlesforJetsHandle_);
  if ( foundgenParticlesforJets ) { 
    genParticlesforJets_ = *genParticlesforJetsHandle_;
    // cout << "Found " << genParticlesforJets_.size() << " genParticlesforJets" << endl;
  } else { 
    //cerr<<"PFRootEventManager::ProcessEntry : genParticlesforJets Collection not found : "
    //    <<entry << " " << genParticlesforJetsTag_<<endl;
  }

  bool foundgenParticlesforMET = iEvent.getByLabel(genParticlesforMETTag_,genParticlesforMETHandle_);
  if ( foundgenParticlesforMET ) { 
    genParticlesCMSSW_ = *genParticlesforMETHandle_;
    // cout << "Found " << genParticlesCMSSW_.size() << " genParticlesforMET" << endl;
  } else { 
    //cerr<<"PFRootEventManager::ProcessEntry : genParticlesforMET Collection not found : "
    //    <<entry << " " << genParticlesforMETTag_<<endl;
  }

  bool foundcaloJets = iEvent.getByLabel(caloJetsTag_,caloJetsHandle_);
  if ( foundcaloJets ) { 
    caloJetsCMSSW_ = *caloJetsHandle_;
    // cout << "Found " << caloJetsCMSSW_.size() << " caloJets" << endl;
  } else { 
    cerr<<"PFRootEventManager::ProcessEntry : caloJets Collection not found : "
        <<entry << " " << caloJetsTag_<<endl;
  }

  bool foundcorrcaloJets = iEvent.getByLabel(corrcaloJetsTag_,corrcaloJetsHandle_);
  if ( foundcorrcaloJets ) { 
    corrcaloJetsCMSSW_ = *corrcaloJetsHandle_;
    // cout << "Found " << corrcaloJetsCMSSW_.size() << " corrcaloJets" << endl;
  } else { 
    //cerr<<"PFRootEventManager::ProcessEntry : corrcaloJets Collection not found : "
    //    <<entry << " " << corrcaloJetsTag_<<endl;
  }

  bool foundpfJets = iEvent.getByLabel(pfJetsTag_,pfJetsHandle_);
  if ( foundpfJets ) { 
    pfJetsCMSSW_ = *pfJetsHandle_;
    // cout << "Found " << pfJetsCMSSW_.size() << " PFJets" << endl;
  } else { 
    cerr<<"PFRootEventManager::ProcessEntry : PFJets Collection not found : "
        <<entry << " " << pfJetsTag_<<endl;
  }

  bool foundpfCands = iEvent.getByLabel(pfCandidateTag_,pfCandidateHandle_);
  if ( foundpfCands ) { 
    pfCandCMSSW_ = *pfCandidateHandle_;
    // cout << "Found " << pfCandCMSSW_.size() << " PFCandidates" << endl;
  } else { 
    cerr<<"PFRootEventManager::ProcessEntry : PFCandidate Collection not found : "
        <<entry << " " << pfCandidateTag_<<endl;
  }

  bool foundpfMets = iEvent.getByLabel(pfMetsTag_,pfMetsHandle_);
  if ( foundpfMets ) { 
    pfMetsCMSSW_ = *pfMetsHandle_;
    //cout << "Found " << pfMetsCMSSW_.size() << " PFMets" << endl;
  } else { 
    cerr<<"PFRootEventManager::ProcessEntry : PFMets Collection not found : "
        <<entry << " " << pfMetsTag_<<endl;
  }

  bool foundtcMets = iEvent.getByLabel(tcMetsTag_,tcMetsHandle_);
  if ( foundtcMets ) { 
    tcMetsCMSSW_ = *tcMetsHandle_;
    //cout << "Found " << tcMetsCMSSW_.size() << " TCMets" << endl;
  } else { 
    cerr<<"TCRootEventManager::ProcessEntry : TCMets Collection not found : "
        <<entry << " " << tcMetsTag_<<endl;
  }

  bool foundcaloMets = iEvent.getByLabel(caloMetsTag_,caloMetsHandle_);
  if ( foundcaloMets ) { 
    caloMetsCMSSW_ = *caloMetsHandle_;
    //cout << "Found " << caloMetsCMSSW_.size() << " CALOMets" << endl;
  } else { 
    cerr<<"CALORootEventManager::ProcessEntry : CALOMets Collection not found : "
        <<entry << " " << caloMetsTag_<<endl;
  }

  // now can use the tree

  bool goodevent = true;
  if(trueParticles_.size() ) {
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

  if(rechitsECAL_.size()) {
    PreprocessRecHits( rechitsECAL_ , findRecHitNeighbours_);
  }
  if(rechitsHCAL_.size()) {
    PreprocessRecHits( rechitsHCAL_ , findRecHitNeighbours_);
  }
  
  if (useHO_) {
    if(rechitsHO_.size()) {
      PreprocessRecHits( rechitsHO_ , findRecHitNeighbours_);
    }
  }

  if(rechitsHFEM_.size()) {
    PreprocessRecHits( rechitsHFEM_ , findRecHitNeighbours_);
  }
  if(rechitsHFHAD_.size()) {
    PreprocessRecHits( rechitsHFHAD_ , findRecHitNeighbours_);
  }
  rechitsCLEANED_.clear();
  for ( unsigned i=0; i<rechitsCLEANEDV_.size(); ++i ) { 
    if(rechitsCLEANEDV_[i].size()) {
      PreprocessRecHits( rechitsCLEANEDV_[i] , false);
      for ( unsigned j=0; j<rechitsCLEANEDV_[i].size(); ++j ) { 
	rechitsCLEANED_.push_back( (rechitsCLEANEDV_[i])[j] );
      }
    }
  }

  if(rechitsPS_.size()) {
    PreprocessRecHits( rechitsPS_ , findRecHitNeighbours_);
  }

  /*
  if ( recTracks_.size() ) { 
    PreprocessRecTracks( recTracks_);
  }

  if ( displacedRecTracks_.size() ) { 
    //   cout << "preprocessing rec tracks" << endl;
    PreprocessRecTracks( displacedRecTracks_);
  }


  if(gsfrecTracks_.size()) {
    PreprocessRecTracks( gsfrecTracks_);
  }
   
  if(convBremGsfrecTracks_.size()) {
    PreprocessRecTracks( convBremGsfrecTracks_);
  }
  */

  return goodevent;
}


bool PFRootEventManager::isHadronicTau() const {

  for ( unsigned i=0;  i < trueParticles_.size(); i++) {
    const reco::PFSimParticle& ptc = trueParticles_[i];
    const std::vector<int>& ptcdaughters = ptc.daughterIds();
    if (std::abs(ptc.pdgCode()) == 15) {
      for ( unsigned int dapt=0; dapt < ptcdaughters.size(); ++dapt) {
        
        const reco::PFSimParticle& daughter 
          = trueParticles_[ptcdaughters[dapt]];
        

        int pdgdaugther = daughter.pdgCode();
        int abspdgdaughter = std::abs(pdgdaugther);


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
    
    if( std::abs(charge)>1e-9) 
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
  kqa=std::abs(Id);
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
  /*
  for( unsigned i=0; i<recTracks.size(); ++i ) {     
    recTracks[i].calculatePositionREP();
  }
  */
}

void 
PFRootEventManager::PreprocessRecTracks(reco::GsfPFRecTrackCollection& recTracks) {  
  /*
  for( unsigned i=0; i<recTracks.size(); ++i ) {     
    recTracks[i].calculatePositionREP();
    recTracks[i].calculateBremPositionREP();
  }
  */
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
  
  vector<bool> mask;
  // ECAL clustering -------------------------------------------

  fillRecHitMask( mask, rechitsECAL_ );
  clusterAlgoECAL_.doClustering( rechitsECAL_, mask );
  clustersECAL_ = clusterAlgoECAL_.clusters();

  assert(clustersECAL_.get() );

  fillOutEventWithClusters( *clustersECAL_ );

  // HCAL clustering -------------------------------------------

  fillRecHitMask( mask, rechitsHCAL_ );
  clusterAlgoHCAL_.doClustering( rechitsHCAL_, mask );
  clustersHCAL_ = clusterAlgoHCAL_.clusters();

  fillOutEventWithClusters( *clustersHCAL_ );

  // HO clustering -------------------------------------------

  if (useHO_) {
    fillRecHitMask( mask, rechitsHO_ );
    
    clusterAlgoHO_.doClustering( rechitsHO_, mask );
    
    clustersHO_ = clusterAlgoHO_.clusters();
    
    fillOutEventWithClusters( *clustersHO_ );
  }

  // HF clustering -------------------------------------------

  fillRecHitMask( mask, rechitsHFEM_ );
  clusterAlgoHFEM_.doClustering( rechitsHFEM_, mask );
  clustersHFEM_ = clusterAlgoHFEM_.clusters();
  
  fillRecHitMask( mask, rechitsHFHAD_ );
  clusterAlgoHFHAD_.doClustering( rechitsHFHAD_, mask );
  clustersHFHAD_ = clusterAlgoHFHAD_.clusters();
  
  // PS clustering -------------------------------------------

  fillRecHitMask( mask, rechitsPS_ );
  clusterAlgoPS_.doClustering( rechitsPS_, mask );
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
    if (!useHO_ && cluster.layer==PFLayer::HCAL_BARREL2) continue;
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

    case PFLayer::HCAL_BARREL2:
      tpLayer = reco::PFTrajectoryPoint::HOLayer;
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
  
  edm::OrphanHandle< reco::PFRecTrackCollection > displacedtrackh( &displacedRecTracks_, 
                                                          edm::ProductID(77) );  

  edm::OrphanHandle< reco::PFClusterCollection > ecalh( clustersECAL_.get(), 
                                                        edm::ProductID(2) );  
  
  edm::OrphanHandle< reco::PFClusterCollection > hcalh( clustersHCAL_.get(), 
                                                        edm::ProductID(3) );  

  edm::OrphanHandle< reco::PFClusterCollection > hoh( clustersHO_.get(), 
						      edm::ProductID(21) );  //GMA put this four

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

  edm::OrphanHandle< reco::PFDisplacedTrackerVertexCollection > nuclearh( &pfNuclearTrackerVertex_, 
                                                          edm::ProductID(7) );


  //recoPFRecTracks_pfNuclearTrackerVertex__TEST.

  edm::OrphanHandle< reco::PFConversionCollection > convh( &conversion_, 
							   edm::ProductID(8) );

  edm::OrphanHandle< reco::PFV0Collection > v0( &v0_, 
						edm::ProductID(9) );


  edm::OrphanHandle< reco::VertexCollection > vertexh( &primaryVertices_, 
						       edm::ProductID(10) );  

  edm::OrphanHandle< reco::GsfPFRecTrackCollection > convBremGsftrackh( &convBremGsfrecTracks_, 
									edm::ProductID(11) );  

  edm::OrphanHandle< reco::PhotonCollection > photonh( &photons_, edm::ProductID(12) ) ;
  
  vector<bool> trackMask;
  fillTrackMask( trackMask, recTracks_ );
  vector<bool> gsftrackMask;
  fillTrackMask( gsftrackMask, gsfrecTracks_ );
  vector<bool> ecalMask;
  fillClusterMask( ecalMask, *clustersECAL_ );
  vector<bool> hcalMask;
  fillClusterMask( hcalMask, *clustersHCAL_ );


  vector<bool> hoMask;
  if (useHO_) {fillClusterMask( hoMask, *clustersHO_ );}

  vector<bool> hfemMask;
  fillClusterMask( hfemMask, *clustersHFEM_ );
  vector<bool> hfhadMask;
  fillClusterMask( hfhadMask, *clustersHFHAD_ );
  vector<bool> psMask;
  fillClusterMask( psMask, *clustersPS_ );
  vector<bool> photonMask;
  fillPhotonMask( photonMask, photons_ );

  if ( !useAtHLT_ )
    pfBlockAlgo_.setInput( trackh, gsftrackh, convBremGsftrackh,
			   muonh, nuclearh, displacedtrackh, convh, v0,
			   ecalh, hcalh, hoh, hfemh, hfhadh, psh, 
			   photonh, trackMask,gsftrackMask, 
			   ecalMask, hcalMask, hoMask, hfemMask, hfhadMask, psMask,photonMask );
  else    
    pfBlockAlgo_.setInput( trackh, muonh, ecalh, hcalh, hfemh, hfhadh, psh, hoh,
			   trackMask, ecalMask, hcalMask, hoMask, psMask);

  pfBlockAlgo_.findBlocks();
  
  if( debug_) cout<<pfBlockAlgo_<<endl;

  pfBlocks_ = pfBlockAlgo_.transferBlocks();

  pfAlgo_.setPFVertexParameters(true, primaryVerticesHandle_.product()); 
  if(useEGElectrons_)
    pfAlgo_.setEGElectronCollection(egammaElectrons_);

  pfAlgo_.reconstructParticles( *pfBlocks_.get() );
  //   pfAlgoOther_.reconstructParticles( blockh );

  //  pfAlgo_.postMuonCleaning(muonsHandle_, *vertexh);

  
  if(usePFElectrons_) {
    pfCandidateElectronExtras_= pfAlgo_.transferElectronExtra();
    edm::OrphanHandle<reco::PFCandidateElectronExtraCollection > electronExtraProd(&(*pfCandidateElectronExtras_),edm::ProductID(20));
    pfAlgo_.setElectronExtraRef(electronExtraProd);
  }

  pfAlgo_.checkCleaning( rechitsCLEANED_ );

  if( debug_) cout<< pfAlgo_<<endl;
  pfCandidates_ = pfAlgo_.transferCandidates();
  //   pfCandidatesOther_ = pfAlgoOther_.transferCandidates();
  
  fillOutEventWithPFCandidates( *pfCandidates_ );

  if( debug_) cout<<"PFRootEventManager::particleFlow stop"<<endl;
}

void PFRootEventManager::pfCandCompare(int entry) {

  /*
  cout << "ievt " << entry <<" : PFCandidate : "
       << " original size : " << pfCandCMSSW_.size()
       << " current  size : " << pfCandidates_->size() << endl;
  */

  bool differentSize = pfCandCMSSW_.size() != pfCandidates_->size();
  if ( differentSize ) { 
    cout << "+++WARNING+++ PFCandidate size changed for entry " 
	 << entry << " !" << endl
	 << " - original size : " << pfCandCMSSW_.size() << endl 
	 << " - current  size : " << pfCandidates_->size() << endl;
  } else { 
    for(unsigned i=0; i<pfCandidates_->size(); i++) {
      double deltaE = (*pfCandidates_)[i].energy()-pfCandCMSSW_[i].energy();
      double deltaEta = (*pfCandidates_)[i].eta()-pfCandCMSSW_[i].eta();
      double deltaPhi = (*pfCandidates_)[i].phi()-pfCandCMSSW_[i].phi();
      if ( fabs(deltaE) > 1E-4 ||
	   fabs(deltaEta) > 1E-9 ||
	   fabs(deltaPhi) > 1E-9 ) { 
	cout << "+++WARNING+++ PFCandidate " << i 
	     << " changed  for entry " << entry << " ! " << endl 
	     << " - Original : " << pfCandCMSSW_[i] << endl
	     << " - Current  : " << (*pfCandidates_)[i] << endl
	     << " DeltaE   = : " << deltaE << endl
	     << " DeltaEta = : " << deltaEta << endl
	     << " DeltaPhi = : " << deltaPhi << endl << endl;
      }
    }
  }
}


void PFRootEventManager::reconstructGenJets() {

  if (verbosity_ == VERBOSE || jetsDebug_) {
    cout<<endl;
    cout<<"start reconstruct GenJets  --- "<<endl;
    cout<< " input gen particles for jet: all neutrinos removed ; muons present" << endl;
  }

  genJets_.clear();
  genParticlesforJetsPtrs_.clear();

  if ( !genParticlesforJets_.size() ) return;

  for(unsigned i=0; i<genParticlesforJets_.size(); i++) {

    const reco::GenParticle&    genPart = *(genParticlesforJets_[i]);

    // remove all muons/neutrinos for PFJet studies
    //    if (reco::isNeutrino( genPart ) || reco::isMuon( genPart )) continue;
    //    remove all neutrinos for PFJet studies
    if (reco::isNeutrino( genPart )) continue;
    // Work-around a bug in the pythia di-jet gun.
    if (std::abs(genPart.pdgId())<7 || std::abs(genPart.pdgId())==21 ) continue;

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



///COLIN need to get rid of this mess. 
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
      if (std::abs(ptc.pdgCode()) == 15) {
	// this is a tau
	if( i ) tooManyTaus = true;
	else tauFound=true;
      }
    }
    
    if(!tauFound || tooManyTaus ) {
      // cerr<<"PFRootEventManager::tauBenchmark : not a single tau event"<<endl;
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
      
    
      if (std::abs((*piter)->pdg_id())==15){
	itau++;
	tauFound=true;
	for ( HepMC::GenVertex::particles_out_const_iterator bp =
		(*piter)->end_vertex()->particles_out_const_begin();
	      bp != (*piter)->end_vertex()->particles_out_const_end(); ++bp ) {
	  uint nuId=std::abs((*bp)->pdg_id());
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
  // if ( pfJets_.size() != 1 ) return;
  double rec_ECALEnergy = 0.;
  double rec_HCALEnergy = 0.;
  double deltaRMin = 999.;
  unsigned int theJet = 0;
  for ( unsigned int ijet=0; ijet<pfJets_.size(); ++ijet ) { 
    double rec_ECAL = pfJets_[ijet].neutralEmEnergy();
    double rec_HCAL = pfJets_[ijet].neutralHadronEnergy();
    double rec_eta = pfJets_[0].eta();
    double rec_phi = pfJets_[0].phi();
    double deltaR = std::sqrt( (rec_eta-true_eta)*(rec_eta-true_eta)
			     + (rec_phi-true_phi)*(rec_phi-true_phi) ); 
    if ( deltaR < deltaRMin ) { 
      deltaRMin = deltaR;
      rec_ECALEnergy = rec_ECAL;
      rec_HCALEnergy = rec_HCAL;
    }
  }
  if ( deltaRMin > 0.1 ) return;
  
  std::vector < PFCandidatePtr > constituents = pfJets_[theJet].getPFConstituents ();
  double pat_ECALEnergy = 0.;
  double pat_HCALEnergy = 0.;
  for (unsigned ic = 0; ic < constituents.size (); ++ic) {
    if ( constituents[ic]->particleId() < 4 ) continue;
    if ( constituents[ic]->particleId() == 4 ) 
      pat_ECALEnergy += constituents[ic]->rawEcalEnergy();
    else if ( constituents[ic]->particleId() == 5 ) 
      pat_HCALEnergy += constituents[ic]->rawHcalEnergy();
  }

  out << true_eta << " " << true_phi << " " << true_E 
      << " " <<  rec_ECALEnergy << " " << rec_HCALEnergy
      << " " <<  pat_ECALEnergy << " " << pat_HCALEnergy
      << " " << deltaRMin << std::endl;
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
    out<<"ECAL RecHits ==============================================="<<endl;
    printRecHits(rechitsECAL_, clusterAlgoECAL_, out );             out<<endl;
    out<<"HCAL RecHits ==============================================="<<endl;
    printRecHits(rechitsHCAL_, clusterAlgoHCAL_, out );             out<<endl;
    if (useHO_) {
      out<<"HO RecHits ================================================="<<endl;
      printRecHits(rechitsHO_, clusterAlgoHO_, out );                 out<<endl;
    }
    out<<"HFEM RecHits ==============================================="<<endl;
    printRecHits(rechitsHFEM_, clusterAlgoHFEM_, out );             out<<endl;
    out<<"HFHAD RecHits =============================================="<<endl;
    printRecHits(rechitsHFHAD_, clusterAlgoHFHAD_, out );           out<<endl;
    out<<"PS RecHits ================================================="<<endl;
    printRecHits(rechitsPS_, clusterAlgoPS_, out );                 out<<endl;
  }

  if( printClusters_ ) {
    out<<"ECAL Clusters ============================================="<<endl;
    printClusters( *clustersECAL_, out);                           out<<endl;
    out<<"HCAL Clusters ============================================="<<endl;
    printClusters( *clustersHCAL_, out);                           out<<endl;
    if (useHO_) {
      out<<"HO Clusters ==============================================="<<endl;
      printClusters( *clustersHO_, out);                             out<<endl;
    }
    out<<"HFEM Clusters ============================================="<<endl;
    printClusters( *clustersHFEM_, out);                           out<<endl;
    out<<"HFHAD Clusters ============================================"<<endl;
    printClusters( *clustersHFHAD_, out);                          out<<endl;
    out<<"PS Clusters   ============================================="<<endl;
    printClusters( *clustersPS_, out);                             out<<endl;
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
    double mex = 0.;
    double mey = 0.;
    for(unsigned i=0; i<pfCandidates_->size(); i++) {
      const PFCandidate& pfc = (*pfCandidates_)[i];
      mex -= pfc.px();
      mey -= pfc.py();
      if(pfc.pt()>printPFCandidatesPtMin_)
      out<<i<<" " <<(*pfCandidates_)[i]<<endl;
    }    
    out<<endl;
    out<< "MEX, MEY, MET ============================================" <<endl 
       << mex << " " << mey << " " << sqrt(mex*mex+mey*mey);
    out<<endl;
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
      if (pfJets_[i].pt() > printPFJetsPtMin_ )
	out<<i<<pfJets_[i].print()<<endl;
    }    
    out<<endl;
    out<<"Generated: "<<endl;
    for(unsigned i=0; i<genJets_.size(); i++) {
      if (genJets_[i].pt() > printPFJetsPtMin_ )
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
      if( trackInsideGCut( trueParticles_[i]) ){ 

	const reco::PFSimParticle& ptc = trueParticles_[i];

	// get trajectory at start point
	const reco::PFTrajectoryPoint& tp0 = ptc.extrapolatedPoint( 0 );

	if(tp0.momentum().pt()>printSimParticlesPtMin_)
	  out<<"\t"<<trueParticles_[i]<<endl;
      }
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

    if(momentum1.pt()<printGenParticlesPtMin_) continue;

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

void PFRootEventManager::printRecHits(const reco::PFRecHitCollection& rechits, const PFClusterAlgo& clusterAlgo, ostream& out) const{

    for(unsigned i=0; i<rechits.size(); i++) {
      string seedstatus = "    ";
      if(clusterAlgo.isSeed(i) ) 
        seedstatus = "SEED";
      printRecHit(rechits[i], i, seedstatus.c_str(), out);
    }
    return;
}

void  PFRootEventManager::printRecHit(const reco::PFRecHit& rh,
				      unsigned index,  
                                      const char* seedstatus,
                                      ostream& out) const {

  if(!out) return;
  double eta = rh.position().Eta();
  double phi = rh.position().Phi();
  double energy = rh.energy();

  if(energy<printRecHitsEMin_)  return;

  TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");
  if( !cutg || cutg->IsInside( eta, phi ) ) 
    out<<index<<"\t"<<seedstatus<<" "<<rh<<endl; 
}

void PFRootEventManager::printClusters(const reco::PFClusterCollection& clusters,
                                       ostream& out ) const {  
  for(unsigned i=0; i<clusters.size(); i++) {
    printCluster(clusters[i], out);
  }
  return;
}

void  PFRootEventManager::printCluster(const reco::PFCluster& cluster,
                                       ostream& out ) const {
  
  if(!out) return;

  double eta = cluster.position().Eta();
  double phi = cluster.position().Phi();
  double energy = cluster.energy();

  if(energy<printClustersEMin_)  return;

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
  if(!cutg) {
    mask.resize( rechits.size(), true);
    return;
  }

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
PFRootEventManager::fillPhotonMask(vector<bool>& mask, 
                                  const reco::PhotonCollection& photons) 
  const {
  
  TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");
  if(!cutg) return;

  mask.clear();
  mask.reserve( photons.size() );
  for(unsigned i=0; i<photons.size(); i++) {
    double eta = photons[i].caloPosition().Eta();
    double phi = photons[i].caloPosition().Phi();
    if( cutg->IsInside( eta, phi ) )
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
  case  421: { name = "D0";latexString="D^{0}" ; break; }
  case -421: { name = "D0-bar";latexString="#overline{D^{0}}" ; break; }
  case  423: { name = "D*0";latexString="D^{*0}" ; break; }
  case -423: { name = "D*0-bar";latexString="#overline{D^{*0}}" ; break; }
  case  431: { name = "Ds_+";latexString="Ds_{+}" ; break; }
  case -431: { name = "Ds_-";latexString="Ds_{-}" ; break; }
  case  511: { name = "B0";latexString= name; break; }
  case  521: { name = "B+";latexString="B^{+}" ; break; }
  case -521: { name = "B-";latexString="B^{-}" ; break; }
  case  531: { name = "Bs_0";latexString="Bs_{0}" ; break; }
  case -531: { name = "anti-Bs_0";latexString="#overline{Bs_{0}}" ; break; }
  case  541: { name = "Bc_+";latexString="Bc_{+}" ; break; }
  case -541: { name = "Bc_+";latexString="Bc_{+}" ; break; }
  case  313: { name = "K*0";latexString="K^{*0}" ; break; }
  case -313: { name = "K*bar0";latexString="#bar{K}^{*0}" ; break; }
  case  323: { name = "K*+";latexString="#K^{*+}"; break; }
  case -323: { name = "K*-";latexString="#K^{*-}" ; break; }
  case  413: { name = "D*+";latexString= "D^{*+}"; break; }
  case -413: { name = "D*-";latexString= "D^{*-}" ; break; }

  case  433: { name = "Ds*+";latexString="D_{s}^{*+}" ; break; }
  case -433: { name = "Ds*-";latexString="B_{S}{*-}" ; break; }

  case  513: { name = "B*0";latexString="B^{*0}" ; break; }
  case -513: { name = "anti-B*0";latexString="#overline{B^{*0}}" ; break; }
  case  523: { name = "B*+";latexString="B^{*+}" ; break; }
  case -523: { name = "B*-";latexString="B^{*-}" ; break; }

  case  533: { name = "B*_s0";latexString="B^{*}_{s0}" ; break; }
  case -533 : {name="anti-B_s0"; latexString= "#overline{B_{s}^{0}}";break; }

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
  case  3114: { name = "Sigma*-"; latexString="#Sigma^{*}" ;break; }
  case -3114: { name = "Sigmabar*+"; latexString="#bar{#Sigma}^{*+}" ;break; }


  case  3212: { name = "Sigma0";latexString="#Sigma^{0}" ; break; }
  case -3212: { name = "Sigmabar0";latexString="#bar{#Sigma}^{0}" ; break; }
  case  3214: { name = "Sigma*0"; latexString="#Sigma^{*0}" ;break; }
  case -3214: { name = "Sigma*bar0";latexString="#bar{#Sigma}^{*0}" ; break; }
  case  3222: { name = "Sigma+"; latexString="#Sigma^{+}" ;break; }
  case -3222: { name = "Sigmabar-"; latexString="#bar{#Sigma}^{-}";break; }
  case  3224: { name = "Sigma*+"; latexString="#Sigma^{*+}" ;break; }
  case -3224: { name = "Sigmabar*-"; latexString="#bar{#Sigma}^{*-}";break; }

  case  2212: { name = "p";latexString=name ; break; }
  case -2212: { name = "~p";latexString="#bar{p}" ; break; }
  case -2214: { name = "Delta-";latexString="#Delta^{-}" ; break; }
  case  2214: { name = "Delta+";latexString="#Delta^{+}" ; break; }
  case -2224: { name = "Deltabar--"; latexString="#bar{#Delta}^{--}" ;break; }
  case  2224: { name = "Delta++"; latexString= "#Delta^{++}";break; }

  case  3312: { name = "Xi-"; latexString= "#Xi^{-}";break; }
  case -3312: { name = "Xi+"; latexString= "#Xi^{+}";break; }
  case  3314: { name = "Xi*-"; latexString= "#Xi^{*-}";break; }
  case -3314: { name = "Xi*+"; latexString= "#Xi^{*+}";break; }

  case  3322: { name = "Xi0"; latexString= "#Xi^{0}";break; }
  case -3322: { name = "anti-Xi0"; latexString= "#overline{Xi^{0}}";break; }
  case  3324: { name = "Xi*0"; latexString= "#Xi^{*0}";break; }
  case -3324: { name = "anti-Xi*0"; latexString= "#overline{Xi^{*0}}";break; }

  case  3334: { name = "Omega-"; latexString= "#Omega^{-}";break; }
  case -3334: { name = "anti-Omega+"; latexString= "#Omega^{+}";break; }

  case  4122: { name = "Lambda_c+"; latexString= "#Lambda_{c}^{+}";break; }
  case -4122: { name = "Lambda_c-"; latexString= "#Lambda_{c}^{-}";break; }
  case  4222: { name = "Sigma_c++"; latexString= "#Sigma_{c}^{++}";break; }
  case -4222: { name = "Sigma_c--"; latexString= "#Sigma_{c}^{--}";break; }


  case 92 : {name="String"; latexString= "String";break; }
    
  case  2101 : {name="ud_0"; latexString= "ud_{0}";break; }
  case -2101 : {name="anti-ud_0"; latexString= "#overline{ud}_{0}";break; }
  case  2103 : {name="ud_1"; latexString= "ud_{1}";break; }
  case -2103 : {name="anti-ud_1"; latexString= "#overline{ud}_{1}";break; }
  case  2203 : {name="uu_1"; latexString= "uu_{1}";break; }
  case -2203 : {name="anti-uu_1"; latexString= "#overline{uu}_{1}";break; }
  case  3303 : {name="ss_1"; latexString= "#overline{ss}_{1}";break; }
  case  3101 : {name="sd_0"; latexString= "sd_{0}";break; }
  case -3101 : {name="anti-sd_0"; latexString= "#overline{sd}_{0}";break; }
  case  3103 : {name="sd_1"; latexString= "sd_{1}";break; }
  case -3103 : {name="anti-sd_1"; latexString= "#overline{sd}_{1}";break; }

  case 20213 : {name="a_1+"; latexString= "a_{1}^{+}";break; }
  case -20213 : {name="a_1-"; latexString= "a_{1}^{-}";break; }

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

edm::InputTag 
PFRootEventManager::stringToTag(const std::vector< std::string >& tagname) { 

  if ( tagname.size() == 1 ) 
    return edm::InputTag(tagname[0]);

  else if ( tagname.size() == 2 ) 
    return edm::InputTag(tagname[0], tagname[1]);

  else if ( tagname.size() == 3 ) 
    return tagname[2] == '*' ? 
      edm::InputTag(tagname[0], tagname[1]) :
      edm::InputTag(tagname[0], tagname[1], tagname[2]);
  else {
    cout << "Invalid tag name with " << tagname.size() << " strings "<< endl;
    return edm::InputTag();
  }
  
}
