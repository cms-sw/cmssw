#ifndef RecoTracker_TkTrackingRegions_TrackingRegionsFromSuperClustersProducer_H 
#define RecoTracker_TkTrackingRegions_TrackingRegionsFromSuperClustersProducer_H

//******************************************************************************
//
// Part of the refactorisation of of the E/gamma pixel matching pre-2017
// This refactorisation converts the monolithic  approach to a series of 
// independent producer modules, with each modules performing  a specific 
// job as recommended by the 2017 tracker framework
//
// This module is called a Producer even though its not an ED producer
// This was done to be consistant with other TrackingRegion producers
// in RecoTracker/TkTrackingRegions
//
// The module closely follows the other TrackingRegion producers
// in RecoTracker/TkTrackingRegions and is intended to become an EDProducer
// by TrackingRegionEDProducerT<TrackingRegionsFromSuperClustersProducer>

// This module c tracking regions from the superclusters. It mostly
// replicates the functionality of the SeedFilter class
// although unlike that class, it does not actually create seeds
//
// Author : Sam Harper (RAL), 2017
//
//*******************************************************************************

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"


#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/FTSFromVertexToPointFactory.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"

//stick this in common tools
#include "TEnum.h"
#include "TEnumConstant.h"
namespace{ 
  template<typename MyEnum> MyEnum strToEnum(std::string const& enumConstName){
    TEnum* en = TEnum::GetEnum(typeid(MyEnum));
    if (en != nullptr){
      if (TEnumConstant const* enc = en->GetConstant(enumConstName.c_str())){
	return static_cast<MyEnum>(enc->GetValue());
      }else{
	throw cms::Exception("Configuration") <<enumConstName<<" is not a valid member of "<<typeid(MyEnum).name();
      }
    }
    throw cms::Exception("LogicError") <<typeid(MyEnum).name()<<" not recognised by ROOT";
  }
  template<> RectangularEtaPhiTrackingRegion::UseMeasurementTracker strToEnum(std::string const& enumConstName){
    using MyEnum = RectangularEtaPhiTrackingRegion::UseMeasurementTracker;
    if(enumConstName=="kNever") return MyEnum::kNever;
    else if(enumConstName=="kForSiStrips") return MyEnum::kForSiStrips;
    else if(enumConstName=="kAlways") return MyEnum::kAlways;
    else{
      throw cms::Exception("Configuration") <<enumConstName<<" is not a valid member of "<<typeid(MyEnum).name()<<" (or strToEnum needs updating, this is a manual translation found at "<<__FILE__<<" line "<<__LINE__<<")";
    }
  }

} 
class TrackingRegionsFromSuperClustersProducer : public TrackingRegionProducer
{
public: 
  enum class Charge{
    NEG=-1,POS=+1
  };
  
public:

  TrackingRegionsFromSuperClustersProducer(const edm::ParameterSet& cfg,
					   edm::ConsumesCollector && cc);

  virtual ~TrackingRegionsFromSuperClustersProducer(){}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    
  virtual std::vector<std::unique_ptr<TrackingRegion> >
  regions (const edm::Event& iEvent, const edm::EventSetup& iSetup)const override;

private:
  GlobalPoint getVtxPos(const edm::Event& iEvent,double& deltaZVertex)const;
  
  std::unique_ptr<TrackingRegion> 
  createTrackingRegion(const reco::SuperCluster& superCluster,const GlobalPoint& vtxPos,
		       const double deltaZVertex,const Charge charge,
		       const MeasurementTrackerEvent* measTrackerEvent,
		       const MagneticField& magField)const;
  

private:
  double ptMin_; 
  double originRadius_; 
  double originHalfLength_;
  double deltaEtaRegion_;
  double deltaPhiRegion_;
  bool useZInVertex_;
  bool precise_;
  RectangularEtaPhiTrackingRegion::UseMeasurementTracker whereToUseMeasTracker_;
  
  edm::EDGetTokenT<reco::VertexCollection> verticesToken_; 
  edm::EDGetTokenT<reco::BeamSpot>  beamSpotToken_; 
  edm::EDGetTokenT<MeasurementTrackerEvent> measTrackerEventToken_;
  std::vector<edm::EDGetTokenT<std::vector<reco::SuperClusterRef>> > superClustersTokens_;

};



namespace {
  template<typename T>
  edm::Handle<T> getHandle(const edm::Event& event,const edm::EDGetTokenT<T> & token){
    edm::Handle<T> handle;
    event.getByToken(token,handle);
    return handle;
  }
}

TrackingRegionsFromSuperClustersProducer::
TrackingRegionsFromSuperClustersProducer(const edm::ParameterSet& cfg,
					 edm::ConsumesCollector && iC)
{ 
  edm::ParameterSet regionPSet = cfg.getParameter<edm::ParameterSet>("RegionPSet");
  
  ptMin_                 = regionPSet.getParameter<double>("ptMin");
  originRadius_          = regionPSet.getParameter<double>("originRadius");
  originHalfLength_      = regionPSet.getParameter<double>("originHalfLength");
  deltaPhiRegion_        = regionPSet.getParameter<double>("deltaPhiRegion");
  deltaEtaRegion_        = regionPSet.getParameter<double>("deltaEtaRegion");
  useZInVertex_          = regionPSet.getParameter<bool>("useZInVertex");
  precise_               = regionPSet.getParameter<bool>("precise");
  whereToUseMeasTracker_ = strToEnum<RectangularEtaPhiTrackingRegion::UseMeasurementTracker>(regionPSet.getParameter<std::string>("whereToUseMeasTracker"));
  
  auto verticesTag         = regionPSet.getParameter<edm::InputTag>("vertices");
  auto beamSpotTag         = regionPSet.getParameter<edm::InputTag>("beamSpot");
  auto superClustersTags   = regionPSet.getParameter<std::vector<edm::InputTag> >("superClusters");
  auto measTrackerEventTag = regionPSet.getParameter<edm::InputTag>("measurementTrackerEvent");
  
  if(useZInVertex_){
    verticesToken_    = iC.consumes<reco::VertexCollection>(verticesTag);
  }else{
    beamSpotToken_    = iC.consumes<reco::BeamSpot>(beamSpotTag);
  }
  if(whereToUseMeasTracker_ != RectangularEtaPhiTrackingRegion::UseMeasurementTracker::kNever){
    measTrackerEventToken_ = iC.consumes<MeasurementTrackerEvent>(measTrackerEventTag);
  }
  for(const auto& tag : superClustersTags){
    superClustersTokens_.emplace_back(iC.consumes<std::vector<reco::SuperClusterRef>>(tag));
  }
}   



void TrackingRegionsFromSuperClustersProducer::
fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{ 
  edm::ParameterSetDescription desc;
  
  desc.add<double>("ptMin", 1.5);
  desc.add<double>("originRadius", 0.2);
  desc.add<double>("originHalfLength", 15.0);
  desc.add<double>("deltaPhiRegion",0.4);
  desc.add<double>("deltaEtaRegion",0.1);
  desc.add<bool>("useZInVertex", false);
  desc.add<bool>("precise", true);
  desc.add<std::string>("whereToUseMeasTracker","kNever");
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
  desc.add<edm::InputTag>("vertices", edm::InputTag());
  desc.add<std::vector<edm::InputTag> >("superClusters", std::vector<edm::InputTag>{edm::InputTag{"hltEgammaSuperClustersToPixelMatch"}});
  desc.add<edm::InputTag>("measurementTrackerEvent",edm::InputTag()); 
  
  edm::ParameterSetDescription descRegion;
  descRegion.add<edm::ParameterSetDescription>("RegionPSet", desc);

  descriptions.add("trackingRegionsFromSuperClusters", descRegion);
}



std::vector<std::unique_ptr<TrackingRegion> >
TrackingRegionsFromSuperClustersProducer::
regions(const edm::Event& iEvent, const edm::EventSetup& iSetup)const
{
  std::vector<std::unique_ptr<TrackingRegion> > trackingRegions;
  
  double deltaZVertex=0;
  GlobalPoint vtxPos = getVtxPos(iEvent,deltaZVertex);
  
  const MeasurementTrackerEvent *measTrackerEvent = nullptr;    
  if(!measTrackerEventToken_.isUninitialized()){
    measTrackerEvent = getHandle(iEvent,measTrackerEventToken_).product();
  }
  edm::ESHandle<MagneticField> magFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(magFieldHandle);
  
  for(auto& superClustersToken : superClustersTokens_){
    auto superClustersHandle = getHandle(iEvent,superClustersToken);
    for(auto& superClusterRef : *superClustersHandle){
      //do both charge hypothesises 
      trackingRegions.emplace_back(createTrackingRegion(*superClusterRef,vtxPos,deltaZVertex,Charge::POS,measTrackerEvent,*magFieldHandle));
      trackingRegions.emplace_back(createTrackingRegion(*superClusterRef,vtxPos,deltaZVertex,Charge::NEG,measTrackerEvent,*magFieldHandle));
    }
  }
  return trackingRegions;
}

GlobalPoint TrackingRegionsFromSuperClustersProducer::
getVtxPos(const edm::Event& iEvent,double& deltaZVertex)const
{
  if(useZInVertex_){
    auto verticesHandle = getHandle(iEvent,verticesToken_);   
    if(!verticesHandle->empty()){
      deltaZVertex = originHalfLength_;
      const auto& pv = verticesHandle->front();
      return GlobalPoint(pv.x(),pv.y(),pv.z());
    }
  }
  
  //if the vertex collection is empty or we dont want to use the z in the vertex 
  //we fall back to beamspot mode
  auto beamSpotHandle = getHandle(iEvent,beamSpotToken_);
  const reco::BeamSpot::Point& bsPos = beamSpotHandle->position();
  //SH: this is what SeedFilter did, no idea what its trying to achieve....
  const double sigmaZ = beamSpotHandle->sigmaZ();
  const double sigmaZ0Error = beamSpotHandle->sigmaZ0Error();
  deltaZVertex = 3*std::sqrt(sigmaZ*sigmaZ+sigmaZ0Error*sigmaZ0Error);
  //  std::cout <<"z "<<bsPos.z()<<" deltaZ "<<deltaZVertex<<std::endl;
  //deltaZVertex = 30.;
  return GlobalPoint(bsPos.x(),bsPos.y(),bsPos.z());
  //return GlobalPoint(bsPos.x(),bsPos.y(),0);
}

std::unique_ptr<TrackingRegion>
TrackingRegionsFromSuperClustersProducer::
createTrackingRegion(const reco::SuperCluster& superCluster,const GlobalPoint& vtxPos,
		     const double deltaZVertex,const Charge charge,
		     const MeasurementTrackerEvent* measTrackerEvent,
		     const MagneticField& magField)const
{   
  const GlobalPoint clusterPos(superCluster.position().x(), superCluster.position().y(), superCluster.position().z());
  const double energy = superCluster.energy();
  
  FreeTrajectoryState freeTrajState = FTSFromVertexToPointFactory::get(magField, clusterPos, vtxPos, energy, static_cast<int>(charge));
  return std::make_unique<RectangularEtaPhiTrackingRegion>(freeTrajState.momentum(),
							   vtxPos,
							   ptMin_,
							   originRadius_,
							   deltaZVertex,
							   deltaEtaRegion_,
							   deltaPhiRegion_,
							   whereToUseMeasTracker_,
							   precise_,
							   measTrackerEvent);
}

#endif 
