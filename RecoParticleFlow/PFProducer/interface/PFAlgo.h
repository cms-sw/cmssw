#ifndef RecoParticleFlow_PFProducer_PFAlgo_h
#define RecoParticleFlow_PFProducer_PFAlgo_h 

#include <iostream>


// #include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Common/interface/Handle.h"
// #include "FWCore/Framework/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

// next include is necessary for inline functions. 
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateElectronExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateElectronExtraFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "RecoParticleFlow/PFProducer/interface/PFCandConnector.h"

/// \brief Particle Flow Algorithm
/*!
  \author Colin Bernet
  \date January 2006
*/


class PFEnergyCalibration;
class PFSCEnergyCalibration;
class PFEnergyCalibrationHF;
class PFElectronAlgo;
class PFConversionAlgo;

namespace pftools { 
  class PFClusterCalibration;
}

class PFAlgo {

 public:

  /// constructor
  PFAlgo();

  /// destructor
  virtual ~PFAlgo();

  void setAlgo( int algo ) {algo_ = algo;}

  void setDebug( bool debug ) {debug_ = debug; connector_.setDebug(debug_);}

  void setParameters(double nSigmaECAL,
                     double nSigmaHCAL, 
                     const boost::shared_ptr<PFEnergyCalibration>& calibration,
                     const boost::shared_ptr<pftools::PFClusterCalibration>& clusterCalibration,
		     const boost::shared_ptr<PFEnergyCalibrationHF>& thepfEnergyCalibrationHF,
		     unsigned int newCalib);
  
  void setCandConnectorParameters( const edm::ParameterSet& iCfgCandConnector ){
    connector_.setParameters(iCfgCandConnector);
  }

  void setPFMuonAndFakeParameters(std::vector<double> muonHCAL,
				  std::vector<double> muonECAL,
				  double nSigmaTRACK,
				  double ptError,
				  std::vector<double> factors45,
				  bool usePFMuonMomAssign);   

  void setPFEleParameters(double mvaEleCut,
			  std::string mvaWeightFileEleID,
			  bool usePFElectrons,
			  const boost::shared_ptr<PFSCEnergyCalibration>& thePFSCEnergyCalibration,
			  double sumEtEcalIsoForEgammaSC_barrel,
			  double sumEtEcalIsoForEgammaSC_endcap,
			  double coneEcalIsoForEgammaSC,
			  double sumPtTrackIsoForEgammaSC_barrel,
			  double sumPtTrackIsoForEgammaSC_endcap,
			  unsigned int nTrackIsoForEgammaSC,
			  double coneTrackIsoForEgammaSC,
			  bool applyCrackCorrections=false,
			  bool usePFSCEleCalib=true,
			  bool useEGElectrons=false,
			  bool useEGammaSupercluster = true);

  void setPostHFCleaningParameters(bool postHFCleaning,
				   double minHFCleaningPt,
				   double minSignificance,
				   double maxSignificance,
				   double minSignificanceReduction,
				   double maxDeltaPhiPt,
				   double minDeltaMet);

  void setDisplacedVerticesParameters(bool rejectTracks_Bad,
				      bool rejectTracks_Step45,
				      bool usePFNuclearInteractions,
				      bool usePFConversions,
				      bool usePFDecays,
				      double dptRel_DispVtx);
  
  //MIKEB : Parameters for the vertices..
  void setPFVertexParameters(bool useVertex,
			   const reco::VertexCollection& primaryVertices);			   
  
  // FlorianB : Collection of e/g electrons
  void setEGElectronCollection(const reco::GsfElectronCollection & egelectrons);

  /// reconstruct particles (full framework case)
  /// will keep track of the block handle to build persistent references,
  /// and call reconstructParticles( const reco::PFBlockCollection& blocks )
  void reconstructParticles( const reco::PFBlockHandle& blockHandle );

  /// reconstruct particles 
  virtual void reconstructParticles( const reco::PFBlockCollection& blocks );
  
  /// Check HF Cleaning
  void checkCleaning( const reco::PFRecHitCollection& cleanedHF );

  // Post Muon Cleaning
  void postMuonCleaning( const edm::Handle<reco::MuonCollection>& muonh,
			 const reco::VertexCollection& primaryVertices );

  // Post Electron Extra Ref
  void setElectronExtraRef(const edm::OrphanHandle<reco::PFCandidateElectronExtraCollection >& extrah);		   

  /// \return collection of candidates
  const std::auto_ptr< reco::PFCandidateCollection >& pfCandidates() const {
    return pfCandidates_;
  }

  /// \return the unfiltered electron collection
  std::auto_ptr< reco::PFCandidateCollection> transferElectronCandidates()  {
    return pfElectronCandidates_;
  }

  /// \return the unfiltered electron extra collection
  // done this way because the pfElectronExtra is needed later in the code to create the Refs and with an auto_ptr, it would be destroyed
  std::auto_ptr< reco::PFCandidateElectronExtraCollection> transferElectronExtra()  {
    std::auto_ptr< reco::PFCandidateElectronExtraCollection> result(new reco::PFCandidateElectronExtraCollection);
    result->insert(result->end(),pfElectronExtra_.begin(),pfElectronExtra_.end());
    return result;
  }


  /// \return collection of cleaned HF candidates
  std::auto_ptr< reco::PFCandidateCollection >& transferCleanedCandidates() {
    return pfCleanedCandidates_;
  }
  
  /// \return collection of  cosmics cleaned muon candidates
  std::auto_ptr< reco::PFCandidateCollection > transferCosmicsMuonCleanedCandidates() {
    return pfCosmicsMuonCleanedCandidates_;
  }

  /// \return collection of  tracker/global cleaned muon candidates
  std::auto_ptr< reco::PFCandidateCollection > transferCleanedTrackerAndGlobalMuonCandidates() {
    return pfCleanedTrackerAndGlobalMuonCandidates_;
  }

  /// \return collection of  fake cleaned muon candidates
  std::auto_ptr< reco::PFCandidateCollection > transferFakeMuonCleanedCandidates() {
    return pfFakeMuonCleanedCandidates_;
  }

  /// \return collection of  punch-through cleaned muon candidates
  std::auto_ptr< reco::PFCandidateCollection > transferPunchThroughMuonCleanedCandidates() {
    return pfPunchThroughMuonCleanedCandidates_;
  }

  /// \return collection of  punch-through cleaned neutral hadron candidates
  std::auto_ptr< reco::PFCandidateCollection > transferPunchThroughHadronCleanedCandidates() {
    return pfPunchThroughHadronCleanedCandidates_;
  }

  /// \return collection of  added muon candidates
  std::auto_ptr< reco::PFCandidateCollection > transferAddedMuonCandidates() {
    return pfAddedMuonCandidates_;
  }
   
    /// \return auto_ptr to the collection of candidates (transfers ownership)
  std::auto_ptr< reco::PFCandidateCollection >  transferCandidates() {
    return connector_.connect(pfCandidates_);
  }
  
  friend std::ostream& operator<<(std::ostream& out, const PFAlgo& algo);
  
 protected:

  /// process one block. can be reimplemented in more sophisticated 
  /// algorithms
  virtual void processBlock( const reco::PFBlockRef& blockref,
                             std::list<reco::PFBlockRef>& hcalBlockRefs, 
                             std::list<reco::PFBlockRef>& ecalBlockRefs ); 
  
  /// Reconstruct a charged hadron from a track
  /// Returns the index of the newly created candidate in pfCandidates_
  unsigned reconstructTrack( const reco::PFBlockElement& elt );

  /// Reconstruct a neutral particle from a cluster. 
  /// If chargedEnergy is specified, the neutral 
  /// particle is created only if the cluster energy is significantly 
  /// larger than the chargedEnergy. In this case, the energy of the 
  /// neutral particle is cluster energy - chargedEnergy

  unsigned reconstructCluster( const reco::PFCluster& cluster,
                               double particleEnergy,
			       bool useDirection = false,
			       double particleX=0.,
			       double particleY=0.,
			       double particleZ=0.);


  /// \return calibrated energy of a photon
  // double gammaCalibratedEnergy( double clusterEnergy ) const;

  /// \return calibrated energy of a neutral hadron, 
  /// which can leave some energy in the ECAL ( energyECAL>0 )
  // double neutralHadronCalibratedEnergy( double energyHCAL, 
  //                                    double energyECAL=-1) const;
  

  /// todo: use PFClusterTools for this
  double neutralHadronEnergyResolution( double clusterEnergy,
					double clusterEta ) const;

 
  double nSigmaHCAL( double clusterEnergy, 
		     double clusterEta ) const;

  std::auto_ptr< reco::PFCandidateCollection >    pfCandidates_;
  /// the unfiltered electron collection 
  std::auto_ptr< reco::PFCandidateCollection >    pfElectronCandidates_;
  // the post-HF-cleaned candidates
  std::auto_ptr< reco::PFCandidateCollection >    pfCleanedCandidates_;
  /// the collection of  cosmics cleaned muon candidates
  std::auto_ptr< reco::PFCandidateCollection >    pfCosmicsMuonCleanedCandidates_;
  /// the collection of  tracker/global cleaned muon candidates
  std::auto_ptr< reco::PFCandidateCollection >    pfCleanedTrackerAndGlobalMuonCandidates_;
  /// the collection of  fake cleaned muon candidates
  std::auto_ptr< reco::PFCandidateCollection >    pfFakeMuonCleanedCandidates_;
  /// the collection of  punch-through cleaned muon candidates
  std::auto_ptr< reco::PFCandidateCollection >    pfPunchThroughMuonCleanedCandidates_;
  /// the collection of  punch-through cleaned neutral hadron candidates
  std::auto_ptr< reco::PFCandidateCollection >    pfPunchThroughHadronCleanedCandidates_;
  /// the collection of  added muon candidates
  std::auto_ptr< reco::PFCandidateCollection >    pfAddedMuonCandidates_;

  /// the unfiltered electron collection 
  reco::PFCandidateElectronExtraCollection    pfElectronExtra_;

  ///Checking if a given cluster is a satellite cluster
  ///of a given charged hadron (track)
  bool isSatelliteCluster( const reco::PFRecTrack& track,
                           const reco::PFCluster& cluster );

  /// Associate PS clusters to a given ECAL cluster, and return their energy
  void associatePSClusters(unsigned iEcal,
			   reco::PFBlockElement::Type psElementType,
			   const reco::PFBlock& block,
			   const edm::OwnVector< reco::PFBlockElement >& elements, 
			   const reco::PFBlock::LinkData& linkData, 
			   std::vector<bool>& active, 
			   std::vector<double>& psEne);

  bool isFromSecInt(const reco::PFBlockElement& eTrack,  std::string order) const;


  // Post HF Cleaning
  void postCleaning();




 private:
  /// create a reference to a block, transient or persistent 
  /// depending on the needs
  reco::PFBlockRef createBlockRef( const reco::PFBlockCollection& blocks, 
				   unsigned bi );
    
  /// input block handle (full framework case)
  reco::PFBlockHandle    blockHandle_;

  /// number of sigma to judge energy excess in ECAL
  double             nSigmaECAL_;
  
  /// number of sigma to judge energy excess in HCAL
  double             nSigmaHCAL_;
  
  boost::shared_ptr<PFEnergyCalibration>  calibration_;
  boost::shared_ptr<pftools::PFClusterCalibration>  clusterCalibration_;
  boost::shared_ptr<PFEnergyCalibrationHF>  thepfEnergyCalibrationHF_;
  boost::shared_ptr<PFSCEnergyCalibration> thePFSCEnergyCalibration_;

  unsigned int newCalib_;

  // std::vector<unsigned> hcalBlockUsed_;
  
  int                algo_;
  bool               debug_;

  /// Variables for PFElectrons
  std::string mvaWeightFileEleID_;
  std::vector<double> setchi2Values_;
  double mvaEleCut_;
  bool usePFElectrons_;
  bool applyCrackCorrectionsElectrons_;
  bool usePFSCEleCalib_;
  bool useEGElectrons_;
  bool useEGammaSupercluster_;
  double sumEtEcalIsoForEgammaSC_barrel_;
  double sumEtEcalIsoForEgammaSC_endcap_;
  double coneEcalIsoForEgammaSC_;
  double sumPtTrackIsoForEgammaSC_barrel_;
  double sumPtTrackIsoForEgammaSC_endcap_;
  double coneTrackIsoForEgammaSC_;
  unsigned int nTrackIsoForEgammaSC_;
  PFElectronAlgo *pfele_;

  // Option to let PF decide the muon momentum
  bool usePFMuonMomAssign_;

  /// Flags to use the protection against fakes 
  /// and not reconstructed displaced vertices
  bool rejectTracks_Bad_;
  bool rejectTracks_Step45_;

  bool usePFNuclearInteractions_;
  bool usePFConversions_;
  PFConversionAlgo* pfConversion_;
  bool usePFDecays_;

  /// Maximal relative uncertainty on the tracks going to or incoming from the 
  /// displcaed vertex to be used in the PFAlgo
  double dptRel_DispVtx_;


  /// A tool used for a postprocessing of displaced vertices
  /// based on reconstructed PFCandidates
  PFCandConnector connector_;
    
  /// Variables for muons and fakes
  std::vector<double> muonHCAL_;
  std::vector<double> muonECAL_;
  double nSigmaTRACK_;
  double ptError_;
  std::vector<double> factors45_;

  // Parameters for post HF cleaning
  bool postHFCleaning_;
  bool postMuonCleaning_;
  double minHFCleaningPt_;
  double minSignificance_;
  double maxSignificance_;
  double minSignificanceReduction_;
  double maxDeltaPhiPt_;
  double minDeltaMet_;

  //MIKE -May19th: Add option for the vertices....
  reco::Vertex       primaryVertex_;
  bool               useVertices_; 

};


#endif


