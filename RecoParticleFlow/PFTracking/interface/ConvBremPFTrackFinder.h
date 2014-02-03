#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedTrackerVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0Fwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0.h"
#include "TMVA/Reader.h"

class PFEnergyCalibration;

class ConvBremPFTrackFinder {
  
 public:
  ConvBremPFTrackFinder(const TransientTrackBuilder& builder,
			double mvaBremConvCut,
			std::string mvaWeightFileConvBrem);  
  ~ConvBremPFTrackFinder();
  
  bool foundConvBremPFRecTrack(const edm::Handle<reco::PFRecTrackCollection>& thePfRecTrackCol,
			       const edm::Handle<reco::VertexCollection>& primaryVertex,
			       const edm::Handle<reco::PFDisplacedTrackerVertexCollection>& pfNuclears,
			       const edm::Handle<reco::PFConversionCollection >& pfConversions,
			       const edm::Handle<reco::PFV0Collection >& pfV0,
			       bool useNuclear,
			       bool useConversions,
			       bool useV0,
			       const reco::PFClusterCollection & theEClus,
			       const reco::GsfPFRecTrack& gsfpfrectk)
  {
    found_ = false;
    runConvBremFinder(thePfRecTrackCol,primaryVertex,
		      pfNuclears,pfConversions,
		      pfV0,useNuclear,
		      useConversions,useV0,
		      theEClus,gsfpfrectk);
    return found_;};
  
  
  const std::vector<reco::PFRecTrackRef>& getConvBremPFRecTracks() {return  pfRecTrRef_vec_;};

 private:
  void runConvBremFinder(const edm::Handle<reco::PFRecTrackCollection>& thePfRecTrackCol,
			 const edm::Handle<reco::VertexCollection>& primaryVertex,
			 const edm::Handle<reco::PFDisplacedTrackerVertexCollection>& pfNuclears,
			 const edm::Handle<reco::PFConversionCollection >& pfConversions,
			 const edm::Handle<reco::PFV0Collection >& pfV0,
			 bool useNuclear,
			 bool useConversions,
			 bool useV0,
			 const reco::PFClusterCollection & theEClus,
			 const reco::GsfPFRecTrack& gsfpfrectk);
  


  bool found_;
  TransientTrackBuilder builder_;
  double mvaBremConvCut_;
  std::string mvaWeightFileConvBrem_;
  TMVA::Reader    *tmvaReader_;
  std::vector<reco::PFRecTrackRef> pfRecTrRef_vec_;
  float secR,secPout,ptRatioGsfKF,sTIP,Epout,detaBremKF,secPin;
  //int nHITS1;
  float nHITS1;

  PFEnergyCalibration* pfcalib_;

};
