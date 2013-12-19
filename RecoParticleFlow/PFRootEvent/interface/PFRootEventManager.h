#ifndef RecoParticleFlow_PFRootEvent_PFRootEventManager_h
#define RecoParticleFlow_PFRootEvent_PFRootEventManager_h

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/Provenance/interface/EventAuxiliary.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"
#include "DataFormats/ParticleFlowReco/interface/PFSimParticleFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateElectronExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateElectronExtraFwd.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "DataFormats/ParticleFlowReco/interface/PFDisplacedTrackerVertex.h"

#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0Fwd.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "CommonTools/CandUtils/interface/pdgIdUtils.h"

/* #include "DataFormats/EgammaReco/interface/BasicCluster.h" */
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PFBlockAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PFAlgo.h"

#include "RecoParticleFlow/PFRootEvent/interface/PFJetAlgorithm.h"
#include "RecoParticleFlow/Benchmark/interface/PFJetBenchmark.h"
#include "RecoParticleFlow/Benchmark/interface/PFMETBenchmark.h"
#include "DQMOffline/PFTau/interface/PFCandidateManager.h"
#include "DQMOffline/PFTau/interface/PFJetMonitor.h"
#include "DQMOffline/PFTau/interface/PFMETMonitor.h"

#include "RecoParticleFlow/PFRootEvent/interface/FWLiteJetProducer.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/MET.h"


#include "RecoParticleFlow/PFRootEvent/interface/METManager.h"


#include <TObject.h>
#include "TEllipse.h"
#include "TBox.h"

#include <string>
#include <map>
#include <set>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>

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
class PFEnergyCalibrationHF;
class PFEnergyResolution;

namespace pftools { 
  class PFClusterCalibration;
}

class METManager;

// NEW
namespace fwlite {
  class ChainEvent;
}

typedef std::pair<double, unsigned> simMatch;
typedef std::list< std::pair<double, unsigned> >::iterator ITM;

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
  
  void initializeEventInformation();

  virtual void write();
  
  /// reset before next event
  void reset();

  /// get name of genParticle
  std::string getGenParticleName(int partId,std::string &latexStringName) const;
  
                         
  /// parse option file
  /// if(reconnect), the rootfile will be reopened, and the tree reconnected
  void readOptions(const char* file, 
                   bool refresh=true,
                   bool reconnect=false);


  virtual void readSpecificOptions(const char* file) {}
  
  /// open the root file and connect to the tree
  void connect(const char* infilename="");

  int eventToEntry(int run, int lumi, int event) const;
  
  /// process one event (pass the CMS event number)
  virtual bool processEvent(int run, int lumi, int event); 

  /// process one entry (pass the TTree entry)
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
  void PreprocessRecTracks( reco::GsfPFRecTrackCollection& rectracks); 
  
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

  /// compare particle flow
  void pfCandCompare(int);

  /// reconstruct gen jets
  void reconstructGenJets();   

  /// reconstruct calo jets
  void reconstructCaloJets();   
  
  /// reconstruct pf jets
  void reconstructPFJets();
  
  /// used by the reconstruct*Jets functions
  void reconstructFWLiteJets(const reco::CandidatePtrVector& Candidates,
                             std::vector<ProtoJet>& output);

  void mcTruthMatching( std::ostream& out,
			const reco::PFCandidateCollection& candidates,
			std::vector< std::list <simMatch> >& candSimMatchTrack,
			std::vector< std::list <simMatch> >&  candSimMatchEcal) const;

  /// performs the tau benchmark 
  ///TODO move this function and the associated datamembers out of here
  ///use an official benchmark from RecoParticleFlow/Benchmark
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

  /// print calibration information
  void   printMCCalib(  std::ofstream& out ) const;


  /// get tree
  TTree* tree() {return tree_;}

  // protected:

  // expand environment variable in a string
  std::string  expand(const std::string& oldString) const;

  /// print rechits
  void printRecHits(const reco::PFRecHitCollection& rechits, 
		    const PFClusterAlgo& clusterAlgo,
		    std::ostream& out = std::cout) const;

  void   printRecHit(const reco::PFRecHit& rh, unsigned index, 
                     const char* seed="    ",
                     std::ostream& out = std::cout) const;
  
  /// print clusters
  void   printClusters(const reco::PFClusterCollection& clusters,
                      std::ostream& out = std::cout) const;

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
  void fillRecHitMask( std::vector<bool>& mask, 
                       const reco::PFRecHitCollection& rechits ) const;
                       
  /// cluster mask set to true for rechits inside TCutG
  void fillClusterMask( std::vector<bool>& mask, 
                        const reco::PFClusterCollection& clusters ) const;

  /// track mask set to true for rechits inside TCutG
  void fillTrackMask( std::vector<bool>& mask, 
                      const reco::PFRecTrackCollection& tracks ) const;
  void fillTrackMask( std::vector<bool>& mask, 
                      const reco::GsfPFRecTrackCollection& tracks ) const;
            
  /// photon mask set to true for photons inside TCutG
  void fillPhotonMask (std::vector<bool>& mask,
		       const reco::PhotonCollection& photons) const;
           
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
  
  /// returns true if the event is accepted(have a look at the function implementation)
  bool eventAccepted() const;

  /// returns true if there is at least one jet with pT>pTmin
  bool highPtJet( double ptMin ) const; 

  /// returns true if there is a PFCandidate of a given type over a given pT
  bool highPtPFCandidate( double ptMin, 
			  reco::PFCandidate::ParticleType type = reco::PFCandidate::X) const;

  /// returns an InputTag from a vector of strings
  edm::InputTag stringToTag(const std::vector< std::string >& tagname); 
  // data members -------------------------------------------------------

  /// current event
  int         iEvent_;
  
  /// options file parser 
  IO*         options_;      
  
  /// NEW: input event
  fwlite::ChainEvent* ev_;

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
 


  // branches --------------------------
  
  TBranch*   eventAuxiliaryBranch_;

  /// event auxiliary information
  edm::EventAuxiliary*      eventAuxiliary_;

  /// rechits ECAL
  edm::Handle<reco::PFRecHitCollection> rechitsECALHandle_;
  edm::InputTag rechitsECALTag_;
  reco::PFRecHitCollection rechitsECAL_;

  /// rechits HCAL
  edm::Handle<reco::PFRecHitCollection> rechitsHCALHandle_;
  edm::InputTag rechitsHCALTag_;
  reco::PFRecHitCollection rechitsHCAL_;

  /// rechits HO
  edm::Handle<reco::PFRecHitCollection> rechitsHOHandle_;
  edm::InputTag rechitsHOTag_;
  reco::PFRecHitCollection rechitsHO_;

  /// rechits HF EM
  edm::Handle<reco::PFRecHitCollection> rechitsHFEMHandle_;
  edm::InputTag rechitsHFEMTag_;
  reco::PFRecHitCollection rechitsHFEM_;

  /// rechits HF HAD
  edm::Handle<reco::PFRecHitCollection> rechitsHFHADHandle_;
  edm::InputTag rechitsHFHADTag_;
  reco::PFRecHitCollection rechitsHFHAD_;

  /// rechits HF CLEANED
  std::vector< reco::PFRecHitCollection > rechitsCLEANEDV_;
  std::vector< edm::Handle<reco::PFRecHitCollection> > rechitsCLEANEDHandles_;
  std::vector< edm::InputTag > rechitsCLEANEDTags_;
  reco::PFRecHitCollection rechitsCLEANED_;

  /// rechits PS 
  edm::Handle<reco::PFRecHitCollection> rechitsPSHandle_;
  edm::InputTag rechitsPSTag_;
  reco::PFRecHitCollection rechitsPS_;

  /// clusters ECAL
  edm::Handle<reco::PFClusterCollection> clustersECALHandle_;
  edm::InputTag clustersECALTag_;
  std::auto_ptr< reco::PFClusterCollection > clustersECAL_;

  /// clusters HCAL
  edm::Handle<reco::PFClusterCollection> clustersHCALHandle_;
  edm::InputTag clustersHCALTag_;
  std::auto_ptr< reco::PFClusterCollection > clustersHCAL_;

  /// clusters HO
  edm::Handle<reco::PFClusterCollection> clustersHOHandle_;
  edm::InputTag clustersHOTag_;
  std::auto_ptr< reco::PFClusterCollection > clustersHO_;

  /// clusters HCAL
  edm::Handle<reco::PFClusterCollection> clustersHFEMHandle_;
  edm::InputTag clustersHFEMTag_;
  std::auto_ptr< reco::PFClusterCollection > clustersHFEM_;

  /// clusters HCAL
  edm::Handle<reco::PFClusterCollection> clustersHFHADHandle_;
  edm::InputTag clustersHFHADTag_;
  std::auto_ptr< reco::PFClusterCollection > clustersHFHAD_;

  /// clusters PS
  edm::Handle<reco::PFClusterCollection> clustersPSHandle_;
  edm::InputTag clustersPSTag_;
  std::auto_ptr< reco::PFClusterCollection > clustersPS_;

  /// input collection of calotowers
  edm::Handle<CaloTowerCollection> caloTowersHandle_;
  edm::InputTag caloTowersTag_;
  CaloTowerCollection     caloTowers_;

  /// for the reconstruction of jets. The elements will point 
  /// to the objects in caloTowers_
  /// has to be global to have a lifetime = lifetime of PFJets
  reco::CandidatePtrVector caloTowersPtrs_;


  /// reconstructed primary vertices
  edm::Handle<reco::VertexCollection> primaryVerticesHandle_;
  edm::InputTag primaryVerticesTag_;
  reco::VertexCollection primaryVertices_;

  /// reconstructed tracks
  edm::Handle<reco::PFRecTrackCollection> recTracksHandle_;
  edm::Handle<reco::PFRecTrackCollection> displacedRecTracksHandle_;
  edm::InputTag recTracksTag_;
  edm::InputTag displacedRecTracksTag_;
  reco::PFRecTrackCollection    recTracks_;
  reco::PFRecTrackCollection    displacedRecTracks_;

  /// reconstructed GSF tracks
  edm::Handle<reco::GsfPFRecTrackCollection> gsfrecTracksHandle_;
  edm::InputTag gsfrecTracksTag_;
  reco::GsfPFRecTrackCollection  gsfrecTracks_; 
  
  // egamma electrons
  edm::Handle<reco::GsfElectronCollection> egammaElectronHandle_;
  edm::InputTag egammaElectronsTag_;
  reco::GsfElectronCollection egammaElectrons_;

  /// reconstructed secondary GSF tracks
  edm::Handle<reco::GsfPFRecTrackCollection> convBremGsfrecTracksHandle_;
  edm::InputTag convBremGsfrecTracksTag_;
  reco::GsfPFRecTrackCollection  convBremGsfrecTracks_; 

  /// standard reconstructed tracks
  edm::Handle<reco::TrackCollection> stdTracksHandle_;
  edm::InputTag stdTracksTag_;
  reco::TrackCollection    stdTracks_;
  
  /// muons
  edm::Handle<reco::MuonCollection> muonsHandle_;
  edm::InputTag muonsTag_;
  reco::MuonCollection  muons_;

  /// conversions
  edm::Handle<reco::PFConversionCollection> conversionHandle_;
  edm::InputTag conversionTag_;
  reco::PFConversionCollection conversion_;
  
  /// photons
  edm::Handle<reco::PhotonCollection> photonHandle_;
  edm::InputTag photonTag_;
  reco::PhotonCollection photons_;

  ///superclusters
  reco::SuperClusterCollection ebsc_;                                                                                                                                                                 
  reco::SuperClusterCollection eesc_;   
  
  /// V0
  edm::Handle<reco::PFV0Collection> v0Handle_;
  edm::InputTag v0Tag_;
  reco::PFV0Collection v0_;

  /// PFDisplacedVertex
  edm::Handle<reco::PFDisplacedTrackerVertexCollection> pfNuclearTrackerVertexHandle_;
  edm::InputTag pfNuclearTrackerVertexTag_;
  reco::PFDisplacedTrackerVertexCollection pfNuclearTrackerVertex_;

  /// true particles
  edm::Handle<reco::PFSimParticleCollection> trueParticlesHandle_;
  edm::InputTag trueParticlesTag_;
  reco::PFSimParticleCollection trueParticles_;

  /// MC truth
  edm::Handle<edm::HepMCProduct> MCTruthHandle_;
  edm::InputTag MCTruthTag_;
  edm::HepMCProduct MCTruth_;
  
  /// input collection of gen particles 
  edm::Handle<reco::GenParticleRefVector> genParticlesforJetsHandle_;
  edm::InputTag genParticlesforJetsTag_;
  reco::GenParticleRefVector genParticlesforJets_;
  
  /// gen Jets
  reco::GenJetCollection genJets_;

  /// CMSSW GenParticles
  edm::Handle<reco::GenParticleCollection> genParticlesforMETHandle_;
  edm::InputTag genParticlesforMETTag_;
  reco::GenParticleCollection genParticlesCMSSW_;

  /// gen particle base candidates (input for gen jets new since 1_8_0)
  /// the vector of references to genParticles genParticlesforJets_
  /// is converted to a PtrVector, which is the input to jet reco
  reco::CandidatePtrVector genParticlesforJetsPtrs_;

  /// reconstructed pfblocks  
  std::auto_ptr< reco::PFBlockCollection >   pfBlocks_;

  /// reconstructed pfCandidates 
  std::auto_ptr< reco::PFCandidateCollection > pfCandidates_;
 
  /// PFCandidateElectronExtra
  std::auto_ptr< reco::PFCandidateElectronExtraCollection > pfCandidateElectronExtras_;
  
  /// for the reconstruction of jets. The elements will point 
  /// to the objects in pfCandidates_
  /// has to be global to have a lifetime = lifetime of PFJets
  ///TODO make the other candidate PtrVectors for jets global as well
  reco::CandidatePtrVector pfCandidatesPtrs_;

  /// PF Jets
  reco::PFJetCollection pfJets_;

  /// calo Jets
  std::vector<ProtoJet> caloJets_;

  /// CMSSW PF Jets
  edm::Handle<reco::PFJetCollection> pfJetsHandle_;
  edm::InputTag pfJetsTag_;
  reco::PFJetCollection pfJetsCMSSW_;

  /// CMSSW  gen Jets
  edm::Handle<reco::GenJetCollection> genJetsHandle_;
  edm::InputTag genJetsTag_;
  reco::GenJetCollection genJetsCMSSW_;

  /// CMSSW calo Jets
  edm::Handle< std::vector<reco::CaloJet> >caloJetsHandle_;
  edm::InputTag caloJetsTag_;
  std::vector<reco::CaloJet> caloJetsCMSSW_;

  /// CMSSW corrected calo Jets
  edm::Handle< std::vector<reco::CaloJet> > corrcaloJetsHandle_;
  edm::InputTag corrcaloJetsTag_;
  std::vector<reco::CaloJet> corrcaloJetsCMSSW_;

  /// PF MET
  reco::PFMETCollection pfMets_;

  /// Calo MET
  reco::CaloMETCollection caloMets_;

  /// TCMET
  reco::METCollection tcMets_;

  /// CMSSW Calo MET
  edm::Handle<reco::CaloMETCollection> caloMetsHandle_;
  edm::InputTag caloMetsTag_;
  reco::CaloMETCollection caloMetsCMSSW_;

  /// CMSSW TCMET
  edm::Handle<reco::METCollection> tcMetsHandle_;
  edm::InputTag tcMetsTag_;
  reco::METCollection tcMetsCMSSW_;

  /// CMSSW PF MET
  edm::Handle<reco::PFMETCollection> pfMetsHandle_;
  edm::InputTag pfMetsTag_;
  reco::PFMETCollection pfMetsCMSSW_;

  /// CMSSW PF candidates
  edm::Handle<reco::PFCandidateCollection> pfCandidateHandle_;
  edm::InputTag pfCandidateTag_;
  reco::PFCandidateCollection pfCandCMSSW_;

  /// input file
  TFile*     file_; 

  /// input file names
  std::vector<std::string> inFileNames_;

  /// output file 
  TFile*     outFile_;

  /// output filename
  std::string     outFileName_;   

  // algos --------------------------------------------------------
  
  /// clustering algorithm for ECAL
  /// \todo try to make all the algorithms concrete. 
  PFClusterAlgo   clusterAlgoECAL_;

  /// clustering algorithm for HCAL
  PFClusterAlgo   clusterAlgoHCAL_;

  /// clustering algorithm for HO
  PFClusterAlgo   clusterAlgoHO_;

  /// clustering algorithm for HF, electro-magnetic layer
  PFClusterAlgo   clusterAlgoHFEM_;

  /// clustering algorithm for HF, hadronic layer
  PFClusterAlgo   clusterAlgoHFHAD_;

  /// clustering algorithm for PS
  PFClusterAlgo   clusterAlgoPS_;


  /// algorithm for building the particle flow blocks 
  PFBlockAlgo     pfBlockAlgo_;

  /// particle flow algorithm
  PFAlgo          pfAlgo_;

  // ------------------- benchmarks -------------------------------
  
  /// PFJet Benchmark
  PFJetBenchmark PFJetBenchmark_;

  /// PFMET Benchmark
  double MET1cut;
  double DeltaMETcut;
  double DeltaPhicut;

  PFCandidateManager   pfCandidateManager_;
  bool                 doPFCandidateBenchmark_;

  /// native jet algorithm 
  /// \todo make concrete
  PFJetAlgorithm  jetAlgo_;
  
  /// wrapper to official jet algorithms
  FWLiteJetProducer jetMaker_;

  // Addition to have DQM histograms : by S. Dutta 
  PFJetMonitor   pfJetMonitor_;
  PFMETMonitor   pfMETMonitor_;
  bool           doPFDQM_;
  TFile*         dqmFile_;
  //-----------------------------------------------

  //----------------- print flags --------------------------------

  /// print rechits yes/no
  bool                     printRecHits_;
  double                   printRecHitsEMin_;

  /// print clusters yes/no
  bool                     printClusters_;
  double                   printClustersEMin_;

  /// print PFBlocks yes/no
  bool                     printPFBlocks_;

  /// print PFCandidates yes/no
  bool                     printPFCandidates_; 
  double                   printPFCandidatesPtMin_; 

  /// print PFJets yes/no
  bool                     printPFJets_;
  double                   printPFJetsPtMin_;
  
  /// print true particles yes/no
  bool                     printSimParticles_;
  double                   printSimParticlesPtMin_;

  /// print MC truth  yes/no
  bool                     printGenParticles_;
  double                   printGenParticlesPtMin_;

  // print MC truth matching with PFCandidate yes/no
  bool                     printMCTruthMatching_;

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
  
  /// comparison with pf CMSSW
  bool   doCompare_;

  /// ECAL-track link optimization
  bool useKDTreeTrackEcalLinker_;

  /// jets on/off
  bool   doJets_;

  /// MET on/off
  bool   doMet_;  

  /// propagate the Jet Energy Corrections to the caloMET on/off
  bool JECinCaloMet_;

  /// jet algo type
  int    jetAlgoType_;

  /// tau benchmark on/off
  bool   doTauBenchmark_;

  /// tau benchmark debug
  bool   tauBenchmarkDebug_;
  
  /// PFJet benchmark on/off
  bool   doPFJetBenchmark_;

  /// PFMET benchmark on/off
  bool doPFMETBenchmark_; 

  /// debug printouts for this PFRootEventManager on/off
  bool   debug_;  

  /// find rechit neighbours ? 
  bool   findRecHitNeighbours_;

  
  /// debug printouts for jet algo on/off
  bool   jetsDebug_;
      

  /// Fastsim or fullsim
  bool  fastsim_;

  /// Use of conversions in PFAlgo 
  bool   usePFConversions_;  

  /// Use of V0 in PFAlgo
  bool   usePFV0s_;

  /// Use of PFDisplacedVertex in PFAlgo
  bool   usePFNuclearInteractions_;

  /// Use Secondary Gsf Tracks
  bool useConvBremGsfTracks_;

  /// Use Conv Brem KF Tracks
  bool useConvBremPFRecTracks_;

  /// Use EGPhotons
  bool useEGPhotons_;

  /// Use PFElectrons
  bool usePFElectrons_;

  /// Use EGElectrons
  bool useEGElectrons_;

  /// Use HLT tracking
  bool useAtHLT_;

  /// Use of HO in links with tracks/HCAL and in particle flow reconstruction
  bool useHO_;

  // MC Truth tools              ---------------------------------------

  /// particle data table.
  /// \todo this could be concrete, but reflex generate code to copy the table,
  /// and the copy constructor is protected...
  /*   TDatabasePDG*   pdgTable_; */

  // Needed for single particle calibration rootTuple
  boost::shared_ptr<pftools::PFClusterCalibration> clusterCalibration_;
  boost::shared_ptr<PFEnergyCalibration> calibration_;
  boost::shared_ptr<PFEnergyCalibrationHF> thepfEnergyCalibrationHF_;

  std::ofstream* calibFile_; 
  
  std::auto_ptr<METManager>   metManager_; 
  
  typedef std::map<int, int>  EventToEntry;
  typedef std::map<int, EventToEntry> LumisMap;
  typedef std::map<int, LumisMap> RunsMap;
  RunsMap  mapEventToEntry_;
};
#endif
