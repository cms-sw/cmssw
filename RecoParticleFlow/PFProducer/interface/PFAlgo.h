#ifndef RecoParticleFlow_PFProducer_PFAlgo_h
#define RecoParticleFlow_PFProducer_PFAlgo_h 

#include <iostream>


// #include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Common/interface/Handle.h"
// #include "FWCore/Framework/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

// next include is necessary for inline functions. 
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoParticleFlow/PFProducer/interface/PFCandConnector.h"

/// \brief Particle Flow Algorithm
/*!
  \author Colin Bernet
  \date January 2006
*/


class PFEnergyCalibration;
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

  void setDebug( bool debug ) {debug_ = debug;}

  void setParameters(double nSigmaECAL,
                     double nSigmaHCAL, 
                     const boost::shared_ptr<PFEnergyCalibration>& calibration,
                     const boost::shared_ptr<pftools::PFClusterCalibration>& clusterCalibration,
		     const boost::shared_ptr<PFEnergyCalibrationHF>& thepfEnergyCalibrationHF,
		     unsigned int newCalib);
  
  void setPFMuonAndFakeParameters(std::vector<double> muonHCAL,
				  std::vector<double> muonECAL,
				  double nSigmaTRACK,
				  double ptError,
				  std::vector<double> factors45);   
  void setPFEleParameters(double mvaEleCut,
			  std::string mvaWeightFileEleID,
			  bool usePFElectrons);

  void setPFConversionParameters( bool usePFConversions );
  
  //MIKEB : Parameters for the vertices..
  void setPFVertexParameters(bool useVertex,
			   const reco::VertexCollection& primaryVertices);			   
  
  /// reconstruct particles (full framework case)
  /// will keep track of the block handle to build persistent references,
  /// and call reconstructParticles( const reco::PFBlockCollection& blocks )
  void reconstructParticles( const reco::PFBlockHandle& blockHandle );

  /// reconstruct particles 
  virtual void reconstructParticles( const reco::PFBlockCollection& blocks );
  

  /// \return collection of candidates
  const std::auto_ptr< reco::PFCandidateCollection >& pfCandidates() const {
    return pfCandidates_;
  }

  /// \return the unfiltered electron collection
   std::auto_ptr< reco::PFCandidateCollection> transferElectronCandidates()  {
      return pfElectronCandidates_;
    }

  
  /// \return auto_ptr to the collection of candidates (transfers ownership)
  std::auto_ptr< reco::PFCandidateCollection >  transferCandidates() {
    PFCandConnector connector;
    return connector.connect(pfCandidates_);
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
  

  // todo: use PFClusterTools for this
  double neutralHadronEnergyResolution( double clusterEnergy,
					double clusterEta ) const;

  // 
  double nSigmaHCAL( double clusterEnergy, 
		     double clusterEta ) const;

  std::auto_ptr< reco::PFCandidateCollection >    pfCandidates_;
  // the unfiltered electron collection 
  std::auto_ptr< reco::PFCandidateCollection >    pfElectronCandidates_;


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
  unsigned int newCalib_;

  // std::vector<unsigned> hcalBlockUsed_;
  
  int                algo_;
  bool               debug_;

  // Variables for PFElectrons
  std::string mvaWeightFileEleID_;
  std::vector<double> setchi2Values_;
  double mvaEleCut_;
  bool usePFElectrons_;
  PFElectronAlgo *pfele_;
  bool usePFConversions_;
  PFConversionAlgo* pfConversion_;
  
  // Variables for muons and fakes
  std::vector<double> muonHCAL_;
  std::vector<double> muonECAL_;
  double nSigmaTRACK_;
  double ptError_;
  std::vector<double> factors45_;

  //MIKE -May19th: Add option for the vertices....
  reco::Vertex       primaryVertex_;
  bool               useVertices_; 

};


#endif


