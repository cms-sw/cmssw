#ifndef RecoParticleFlow_PFRootEvent_PFRootEventManager_h
#define RecoParticleFlow_PFRootEvent_PFRootEventManager_h

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"
#include "DataFormats/ParticleFlowReco/interface/PFSimParticleFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

#include "RecoParticleFlow/PFClusterAlgo/interface/PFClusterAlgo.h"
#include "RecoParticleFlow/PFBlockAlgo/interface/PFBlockAlgo.h"
#include "RecoParticleFlow/PFAlgo/interface/PFAlgo.h"

#include "RecoParticleFlow/PFRootEvent/interface/PFJetAlgorithm.h"
// #include "RecoParticleFlow/Benchmark/interface/PFJetBenchmark.h"

#include "RecoParticleFlow/PFRootEvent/interface/FWLiteJetProducer.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include <TObject.h>
#include "TEllipse.h"
#include "TBox.h"

#include <string>
#include <map>
#include <set>
#include <vector>
#include <memory>

class TTree;
class TBranch;
class TFile;
class TCanvas;
class TH2F;
class TH1F;


class IO;


class PFBlockElement;

class EventColin;
class PFEnergyCalibration;
class PFEnergyResolution;

/// \brief ROOT interface to particle flow package
/*!
  This base class allows to perform clustering and particle flow from 
  ROOT CINT (or any program). It is designed to support analysis and 
  developpement. Users should feel free to create their own PFRootEventManager,
  inheriting from this base class. Just reimplement the ProcessEntry function

  An example:

  \code
  gSystem->Load("libFWCoreFWLite.so");
  gSystem->Load("libRecoParticleFlowPFRootEvent.so");
  AutoLibraryLoader::enable();
  gSystem->Load("libCintex.so");
  ROOT::Cintex::Cintex::Enable();

  PFRootEventManager em("pfRootEvent.opt");
  int i=0;
  em.processEntry( i++ )
  \endcode
  
  pfRootEvent.opt is an option file (see IO class):
  \verbatim
  root file test.root

  root hits_branch  recoPFRecHits_pfcluster__Demo.obj
  root recTracks_branch  recoPFRecTracks_pf_PFRecTrackCollection_Demo.obj

  display algos 1 

  display  viewsize_etaphi 600 400
  display  viewsize_xy     400 400

  display  color_clusters               1

  clustering thresh_Ecal_Barrel           0.2
  clustering thresh_Seed_Ecal_Barrel      0.3
  clustering thresh_Ecal_Endcap           0.2
  clustering thresh_Seed_Ecal_Endcap      0.9
  clustering neighbours_Ecal            4

  clustering depthCor_Mode          1
  clustering depthCor_A                   0.89
  clustering depthCor_B                   7.3
  clustering depthCor_A_preshower   0.89
  clustering depthCor_B_preshower   4.0

  clustering thresh_Hcal_Barrel           1.0
  clustering thresh_Seed_Hcal_Barrel      1.4
  clustering thresh_Hcal_Endcap           1.0
  clustering thresh_Seed_Hcal_Endcap      1.4
  clustering neighbours_Hcal            4
  \endverbatim
  
  \author Colin Bernet, Renaud Bruneliere

  \date July 2006

  \todo test
*/
class PFRootEventManager {

 public:

  /// viewport definition
  enum View_t { XY = 0, RZ = 1, EPE = 2, EPH = 3, NViews = 4 };
  enum Verbosity {SHUTUP = 0, VERBOSE};

  /// default constructor
  PFRootEventManager();
  
  /// \param is an option file, see IO
  PFRootEventManager(const char* file);

  /// destructor
  virtual ~PFRootEventManager();
  
  virtual void write();
  
  /// reset before next event
  void reset();

  /// parse option file
  /// if(reconnect), the rootfile will be reopened, and the tree reconnected
  void readOptions(const char* file, 
                   bool refresh=true,
                   bool reconnect=false);


  virtual void readSpecificOptions(const char* file) {}
  
  /// open the root file and connect to the tree
  void connect(const char* infilename="");

  /// sets addresses for all branches
  void setAddresses();
  
  /// process one entry 
  virtual bool processEntry(int entry);

  /// read data from simulation tree
  bool readFromSimulation(int entry);

  /// study the sim event to check if the tau decay is hadronic
  bool isHadronicTau() const;

  /// study the sim event to check if the 
  /// number of stable charged particles and stable photons
  /// match the selection
  bool countChargedAndPhotons() const;
  
  /// return the chargex3
  /// \todo function stolen from famos. remove when it it possible to 
  /// use the particle data table in FWLite
  int chargeValue(const int& pdgId) const;
   
  
  /// preprocess a rectrack vector from a given rectrack branch
  void PreprocessRecTracks( reco::PFRecTrackCollection& rectracks); 
  
  /// preprocess a rechit vector from a given rechit branch
  void PreprocessRecHits( reco::PFRecHitCollection& rechits, 
                          bool findNeighbours);
  
  /// for a given rechit, find the indices of the rechit neighbours, 
  /// and store these indices in the rechit. The search is done in a
  /// detid to index map
  void setRecHitNeigbours( reco::PFRecHit& rh, 
                           const std::map<unsigned, unsigned>& detId2index );

  /// read data from testbeam tree
  //  bool readFromRealData(int entry);

  /// performs clustering 
  void clustering();

  /// performs particle flow
  void particleFlow();

  /// reconstruct gen jets
  void reconstructGenJets();   

  /// reconstruct calo jets
  void reconstructCaloJets();   
  
  /// reconstruct pf jets
  void reconstructPFJets();
  

  /// used by the reconstruct*Jets functions
  void reconstructFWLiteJets(const reco::CandidateCollection& Candidates,
			     std::vector<ProtoJet>& output);


  /// performs the tau benchmark 
  double tauBenchmark( const reco::PFCandidateCollection& candidates);


  /// fills OutEvent with clusters
  void fillOutEventWithClusters(const reco::PFClusterCollection& clusters);

  /// fills OutEvent with candidates
  void fillOutEventWithPFCandidates(const reco::PFCandidateCollection& pfCandidates );

  /// fills OutEvent with sim particles
  void fillOutEventWithSimParticles(const reco::PFSimParticleCollection& ptcs);

  /// fills outEvent with calo towers
  void fillOutEventWithCaloTowers(const CaloTowerCollection& cts);

  /// fills outEvent with blocks
  void fillOutEventWithBlocks(const reco::PFBlockCollection& blocks);
  


  /// print information
  void   print(  std::ostream& out = std::cout,
                 int maxNLines = -1 ) const;


  /// get tree
  TTree* tree() {return tree_;}

  // protected:

  // expand environment variable in a string
  std::string  expand(const std::string& oldString) const;

  /// print a rechit
  void   printRecHit(const reco::PFRecHit& rh, 
                     const char* seed="    ",
                     std::ostream& out = std::cout) const;
  
  /// print a cluster
  void   printCluster(const reco::PFCluster& cluster,
                      std::ostream& out = std::cout) const;

  

  /// print the HepMC truth
  void printGenParticles(std::ostream& out = std::cout,
                         int maxNLines = -1) const;
  
  /*   /// is inside cut G?  */
  /*   bool   insideGCut(double eta, double phi) const; */
  
  /// is PFTrack inside cut G ? yes if at least one trajectory point is inside.
  bool trackInsideGCut( const reco::PFTrack& track ) const;
  
  /// rechit mask set to true for rechits inside TCutG
  void fillRecHitMask( vector<bool>& mask, 
                       const reco::PFRecHitCollection& rechits ) const;
                       
  /// cluster mask set to true for rechits inside TCutG
  void fillClusterMask( vector<bool>& mask, 
                        const reco::PFClusterCollection& clusters ) const;

  /// track mask set to true for rechits inside TCutG
  void fillTrackMask( vector<bool>& mask, 
                      const reco::PFRecTrackCollection& tracks ) const;
                       
  /// find the closest PFSimParticle to a point (eta,phi) in a given detector
  const reco::PFSimParticle& 
    closestParticle( reco::PFTrajectoryPoint::LayerType  layer, 
                     double eta, double phi, 
                     double& peta, double& pphi, double& pe) const;
                     
  
  const  reco::PFBlockCollection& blocks() const { return *pfBlocks_; }

  
  int eventNumber()   {return iEvent_;}

  /*   std::vector<int> getViewSizeEtaPhi() {return viewSizeEtaPhi_;} */
  /*   std::vector<int> getViewSize()       {return viewSize_;} */
  
  void readCMSSWJets();
  
  
  
  // data members -------------------------------------------------------

  /// current event
  int         iEvent_;
  
  /// options file parser 
  IO*         options_;      
  
  /// input tree  
  TTree*      tree_;          
  
  /// output tree
  TTree*      outTree_;

  /// event for output tree 
  /// \todo change the name EventColin to something else
  EventColin* outEvent_;

  /// output histo dET ( EHT - MC)
  TH1F*            h_deltaETvisible_MCEHT_;

  /// output histo dET ( PF - MC)  
  TH1F*            h_deltaETvisible_MCPF_;
 

  // MC branches --------------------------
  
  /// rechits branch  
  TBranch*   hitsBranch_;          
  
  /// ECAL rechits branch  
  TBranch*   rechitsECALBranch_;          
  
  /// HCAL rechits branch  
  TBranch*   rechitsHCALBranch_;          
  
  /// PS rechits branch  
  TBranch*   rechitsPSBranch_;          

  /// ECAL clusters branch  
  TBranch*   clustersECALBranch_;          
  
  /// HCAL clusters branch  
  TBranch*   clustersHCALBranch_;          
   
  /// PS clusters branch  
  TBranch*   clustersPSBranch_;          

  /// ECAL island clusters branch_;
  TBranch*   clustersIslandBarrelBranch_;

  /// calotowers
  TBranch* caloTowersBranch_;
  
  /// reconstructed tracks branch  
  TBranch*   recTracksBranch_;          
  
  /// standard reconstructed tracks branch  
  TBranch*   stdTracksBranch_;          
  
  /// true particles branch
  TBranch*   trueParticlesBranch_;          

  /// MCtruth branch
  TBranch*   MCTruthBranch_;          

  /// Gen Particles base Candidates branch
  TBranch*   genParticleBaseCandidatesBranch_;

  /// Calo Tower base Candidates branch
  TBranch*   caloTowerBaseCandidatesBranch_;
  
  ///CMSSW Gen Jet branch
  TBranch*   genJetBranch_;
  
  ///CMSSW Calo Jet branch
  TBranch*   recCaloBranch_;
  
  ///CMSSW  PF Jet branch
  TBranch*   recPFBranch_;

  
  
  /// rechits ECAL
  reco::PFRecHitCollection rechitsECAL_;

  /// rechits HCAL
  reco::PFRecHitCollection rechitsHCAL_;

  /// rechits PS 
  reco::PFRecHitCollection rechitsPS_;

  /// clusters ECAL
  std::auto_ptr< reco::PFClusterCollection > clustersECAL_;

  /// clusters HCAL
  std::auto_ptr< reco::PFClusterCollection > clustersHCAL_;

  /// clusters PS
  std::auto_ptr< reco::PFClusterCollection > clustersPS_;

  /// clusters ECAL island barrel
  std::vector<reco::BasicCluster>  clustersIslandBarrel_;
  
  CaloTowerCollection     caloTowers_;

  /// reconstructed tracks
  reco::PFRecTrackCollection    recTracks_;
  
  /// standard reconstructed tracks
  reco::TrackCollection    stdTracks_;
  
  /// true particles
  reco::PFSimParticleCollection trueParticles_;

  /// MC truth
  edm::HepMCProduct MCTruth_;
  
  /// reconstructed pfblocks  
  std::auto_ptr< reco::PFBlockCollection >   pfBlocks_;

  /// reconstructed pfCandidates 
  std::auto_ptr< reco::PFCandidateCollection > pfCandidates_;
  
  /// has to be global to print out pfjets constituents
  reco::CandidateCollection basePFCandidates_;

  /// gen particle base candidates (input for gen jets)
  reco::CandidateCollection genParticleBaseCandidates_;

  /// calo tower base candidates (input for calo jets)
  reco::CandidateCollection caloTowerBaseCandidates_;

  /// PF Jets
  reco::PFJetCollection pfJets_;

  /// gen Jets
  reco::GenJetCollection genJets_;

  /// calo Jets
  std::vector<ProtoJet> caloJets_;

  /// CMSSW PF Jets
  reco::PFJetCollection pfJetsCMSSW_;

  /// CMSSW  gen Jets
  reco::GenJetCollection genJetsCMSSW_;

  /// calo Jets
  std::vector<CaloJet> caloJetsCMSSW_;
  /// input file
  TFile*     file_; 

  /// input file name
  std::string     inFileName_;   

  /// output file 
  TFile*     outFile_;

  /// output filename
  std::string     outFileName_;   

  // algos --------------------------------------------------------
  
  /// clustering algorithm for ECAL
  /// \todo try to make all the algorithms concrete. 
  PFClusterAlgo   clusterAlgoECAL_;

  /// clustering algorithm for ECAL
  PFClusterAlgo   clusterAlgoHCAL_;

  /// clustering algorithm for ECAL
  PFClusterAlgo   clusterAlgoPS_;


  /// algorithm for building the particle flow blocks 
  PFBlockAlgo     pfBlockAlgo_;

  /// particle flow algorithm
  PFAlgo          pfAlgo_;

  /// PFJet Benchmark
  // PFJetBenchmark PFJetBenchmark_;

  /// native jet algorithm 
  /// \todo make concrete
  PFJetAlgorithm  jetAlgo_;
  
  /// wrapper to official jet algorithms
  FWLiteJetProducer jetMaker_;


  //----------------- print flags --------------------------------

  /// print rechits yes/no
  bool                     printRecHits_;

  /// print clusters yes/no
  bool                     printClusters_;

  /// print PFBlocks yes/no
  bool                     printPFBlocks_;

  /// print PFCandidates yes/no
  bool                     printPFCandidates_; 

  /// print PFJets yes/no
  bool                     printPFJets_;
  
  /// print true particles yes/no
  bool                     printSimParticles_;

  /// print MC truth  yes/no
  bool                     printGenParticles_;

  /// verbosity
  int                      verbosity_;

  //----------------- filter ------------------------------------
  
  unsigned                 filterNParticles_;
  
  bool                     filterHadronicTaus_;

  std::vector<int>         filterTaus_;

  // --------

  /// clustering on/off. If on, rechits from tree are used to form 
  /// clusters. If off, clusters from tree are used.
  bool   doClustering_;

  /// particle flow on/off
  bool   doParticleFlow_;

  /// jets on/off
  bool   doJets_;
  
  /// jet algo type
  int    jetAlgoType_;

  /// tau benchmark on/off
  bool   doTauBenchmark_;

  /// tau benchmark debug
  bool   tauBenchmarkDebug_;
  
  /// PFJet benchmark on/off
  bool   doPFJetBenchmark_;

  /// debug printouts for this PFRootEventManager on/off
  bool   debug_;  

  /// find rechit neighbours ? 
  bool   findRecHitNeighbours_;

  
  /// debug printouts for jet algo on/off
  bool   jetsDebug_;
      
  // MC Truth tools              ---------------------------------------

  /// particle data table.
  /// \todo this could be concrete, but reflex generate code to copy the table,
  /// and the copy constructor is protected...
  /*   TDatabasePDG*   pdgTable_; */

};
#endif
