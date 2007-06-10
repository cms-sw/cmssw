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

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

#include "RecoParticleFlow/PFClusterAlgo/interface/PFClusterAlgo.h"
#include "RecoParticleFlow/PFBlockAlgo/interface/PFBlockAlgo.h"
#include "RecoParticleFlow/PFAlgo/interface/PFAlgo.h"

#include "RecoParticleFlow/PFRootEvent/interface/PFJetAlgorithm.h"

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
class TGraph;


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
  em.display(i++);
  \endcode
  
  pfRootEvent.opt is an option file (see IO class):
  \verbatim
  root file test.root

  root hits_branch  recoPFRecHits_pfcluster__Demo.obj
  root recTracks_branch  recoPFRecTracks_pf_PFRecTrackCollection_Demo.obj

  display algos 1 

  display  viewsize_etaphi 600 400
  display  viewsize_xy     400 400

  display  color_clusters		1

  clustering thresh_Ecal_Barrel           0.2
  clustering thresh_Seed_Ecal_Barrel      0.3
  clustering thresh_Ecal_Endcap           0.2
  clustering thresh_Seed_Ecal_Endcap      0.9
  clustering neighbours_Ecal		4

  clustering depthCor_Mode          1
  clustering depthCor_A 		  0.89
  clustering depthCor_B 		  7.3
  clustering depthCor_A_preshower   0.89
  clustering depthCor_B_preshower   4.0

  clustering thresh_Hcal_Barrel           1.0
  clustering thresh_Seed_Hcal_Barrel      1.4
  clustering thresh_Hcal_Endcap           1.0
  clustering thresh_Seed_Hcal_Endcap      1.4
  clustering neighbours_Hcal		4
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

  /// fills OutEvent with clusters
  void fillOutEventWithClusters(const reco::PFClusterCollection& clusters);

  /// fills OutEvent with sim particles
  void fillOutEventWithSimParticles(const reco::PFSimParticleCollection& ptcs);

  /// performs particle flow
  void particleFlow();

  //performs the jets reconstructions
  double makeJets();


  // display functions ------------------------------------------------

  /// process and display one entry 
  void display(int ientry);
  
  /// display next selected entry. if init, restart from i=0
  void displayNext(bool init);

  /// display current entry
  void display();

  /// display x/y or r/z
  void displayView(unsigned viewType);

  /// display reconstructed calorimeter hits in x/y or r/z view
  void displayRecHits(unsigned viewType, double phi0 = 0.);

  /// display a reconstructed calorimeter hit in x/y or r/z view
  void displayRecHit(reco::PFRecHit& rh, unsigned viewType,
		     double maxe, double phi0 = 0., int color=4);

  /// display clusters in x/y or r/z view
  void displayClusters(unsigned viewType, double phi0 = 0.);

  /// display one cluster
  void displayCluster(const reco::PFCluster& cluster,
		      unsigned viewType, double phi0 = 0.);
  
  /// display cluster-to-rechits lines
  void displayClusterLines(const reco::PFCluster& cluster);

  /// display reconstructed tracks
  void displayRecTracks(unsigned viewType, double phi0 = 0.);

  /// display true particles
  void displayTrueParticles(unsigned viewType, double phi0 = 0.);

  /// display track (for rectracks and particles)
  void displayTrack(const std::vector<reco::PFTrajectoryPoint>& points, 
		    unsigned viewType, double phi0, 
		    double sign, bool displayInitial, 
		    int linestyle, int markerstyle, double markersize, 
		    int color);  


  /// unzooms all support histograms
  void unZoom();

  /// updates all displays
  void updateDisplay();

  /// look for rechit with max energy in ecal or hcal.
  /// 
  /// \todo look for rechit with max transverse energy, look for other objects
  void lookForMaxRecHit(bool ecal);


  /// finds max rechit energy in a given layer 
  double getMaxE(int layer) const;

  /// max rechit energy in ecal 
  double getMaxEEcal();

  /// max rechit energy in hcal 
  double getMaxEHcal();


  /// print information
  void   print(  std::ostream& out = std::cout ) const;

  /// print event display 
  void   printDisplay( const char* directory="" ) const;

  /// get tree
  TTree* tree() {return tree_;}

 protected:

  // retrieve resolution maps
  void   getMap(std::string& map);

  /// print a rechit
  void   printRecHit(const reco::PFRecHit& rh, 
		     const char* seed="    ",
		     std::ostream& out = std::cout) const;
  
  /// print a cluster
  void   printCluster(const reco::PFCluster& cluster,
		      std::ostream& out = std::cout) const;

  /// print the HepMC truth
  void printMCTruth(const HepMC::GenEvent*) const;
  
  /// is inside cut G? 
  bool   insideGCut(double eta, double phi) const;

  /// is PFTrack inside cut G ? yes if at least one trajectory point is inside.
  bool   trackInsideGCut( const reco::PFTrack* track ) const;

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

  /// jet algorithm 
  /// \todo make concrete
  PFJetAlgorithm  jetAlgo_;
  
  // display ------------------------------------------------------

  /// canvases for eta/phi display, one per algo
  /// each is split in 2 : HCAL, ECAL
  // std::map<int, TCanvas* > displayEtaPhi_;        

  /// algos to display
  std::set<int>            algosToDisplay_;  

  /// display cluster-to-rechits lines ? 
  bool                     displayClusterLines_;

  /// display pad xy size for eta/phi view
  std::vector<int>         viewSizeEtaPhi_;        

  //------------ display settings -----------------------------

  /// display x/y ?
  bool displayXY_;

  /// display eta/phi ?
  bool displayEtaPhi_;

  /// display r/z ?
  bool displayRZ_;  
  
  /// display cluster color ? (then color = cluster type )
  bool displayColorClusters_;

  /// display rectracks ? 
  bool displayRecTracks_;

  /// display true particles ? 
  bool displayTrueParticles_;

  /// size of view in number of cells when centering on a rechit
  double displayZoomFactor_;

  /// pt threshold to display rec tracks
  double displayRecTracksPtMin_;

  /// pt threshold to display true particles
  double displayTrueParticlesPtMin_;

  /// vector of canvas for x/y or r/z display
  std::vector<TCanvas*> displayView_;

  /// display pad xy size for (x,y) or (r,z) display
  std::vector<int>      viewSize_;     

  /// support histogram for x/y or r/z display. 
  std::vector<TH2F*>    displayHist_;

  /// ECAL in XY view. \todo should be attribute ?
  TEllipse frontFaceECALXY_;

  /// ECAL in RZ view. \todo should be attribute ?
  TBox     frontFaceECALRZ_;

  /// HCAL in XY view. \todo should be attribute ?
  TEllipse frontFaceHCALXY_;

  /// max rechit energy in ecal
  double                   maxERecHitEcal_;

  /// max rechit energy in hcal
  double                   maxERecHitHcal_;


  //----------------- print flags --------------------------------

  /// print rechits yes/no
  bool                     printRecHits_;

  /// print clusters yes/no
  bool                     printClusters_;

  /// print PFBlocks yes/no
  bool                     printPFBlocks_;

   /// print PFCandidates yes/no
  bool                     printPFCandidates_; 

  /// print true particles yes/no
  bool                     printTrueParticles_;

  /// print MC truth  yes/no
  bool                     printMCtruth_;

  /// verbosity
  int                      verbosity_;

  //----------------- filter ------------------------------------
  
  unsigned                 filterNParticles_;
  
  bool                     filterHadronicTaus_;

  //----------------- clustering parameters ---------------------

  /// clustering on/off. If on, rechits from tree are used to form 
  /// clusters. If off, clusters from tree are used.
  bool   clusteringIsOn_;

  /// clustering mode. 
  // int    clusteringMode_;


  /// debug printouts for this PFRootEventManager on/off
  bool   debug_;  

  /// find rechit neighbours ? 
  bool   findRecHitNeighbours_;

  /// not yet used ?
  bool   displayJetColors_;

  // jets parameters             ----------------------------------------

  /// jet reconstruction (for taus) on/off 
  /// \todo make jet reconstruction more general.
  bool   doJets_;
  
  /// debug printouts for jet algo on/off
  bool   jetsDebug_;

};
#endif
