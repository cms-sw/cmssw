#include "RecoTauTag/RecoTau/interface/CaloRecoTauTagInfoAlgorithm.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

using namespace reco;

CaloRecoTauTagInfoAlgorithm::CaloRecoTauTagInfoAlgorithm(const edm::ParameterSet& parameters){
  // parameters of the considered rec. Tracks (catched through a JetTracksAssociation object) :
  tkminPt_                            = parameters.getParameter<double>("tkminPt");
  tkminPixelHitsn_                    = parameters.getParameter<int>("tkminPixelHitsn");
  tkminTrackerHitsn_                  = parameters.getParameter<int>("tkminTrackerHitsn");
  tkmaxipt_                           = parameters.getParameter<double>("tkmaxipt");
  tkmaxChi2_                          = parameters.getParameter<double>("tkmaxChi2");
  // 
  UsePVconstraint_                    = parameters.getParameter<bool>("UsePVconstraint");
  tkPVmaxDZ_                          = parameters.getParameter<double>("tkPVmaxDZ");
  //
  UseTrackQuality_                    = parameters.getParameter<bool>("UseTrackQuality");
  if (UseTrackQuality_) {
    tkQuality_ = reco::TrackBase::qualityByName(parameters.getParameter<std::string>("tkQuality"));
  }
  // parameters of the considered EcalRecHits 
  BarrelBasicClusters_                = parameters.getParameter<edm::InputTag>("BarrelBasicClustersSource"); 
  EndcapBasicClusters_                = parameters.getParameter<edm::InputTag>("EndcapBasicClustersSource"); 
  // parameters of the considered neutral ECAL BasicClusters
  ECALBasicClustersAroundCaloJet_DRConeSize_      = parameters.getParameter<double>("ECALBasicClustersAroundCaloJet_DRConeSize");
  ECALBasicClusterminE_                           = parameters.getParameter<double>("ECALBasicClusterminE");
  ECALBasicClusterpropagTrack_matchingDRConeSize_ = parameters.getParameter<double>("ECALBasicClusterpropagTrack_matchingDRConeSize");
}
  
CaloTauTagInfo CaloRecoTauTagInfoAlgorithm::buildCaloTauTagInfo(edm::Event& theEvent,const edm::EventSetup& theEventSetup,const CaloJetRef& theCaloJet,const TrackRefVector& theTracks,const Vertex& thePV){
  CaloTauTagInfo resultExtended;
  resultExtended.setcalojetRef(theCaloJet);

  TrackRefVector theFilteredTracks;
  if (UsePVconstraint_) theFilteredTracks=TauTagTools::filteredTracks(theTracks,tkminPt_,tkminPixelHitsn_,tkminTrackerHitsn_,tkmaxipt_,tkmaxChi2_,tkPVmaxDZ_,thePV, thePV.z());
  else theFilteredTracks=TauTagTools::filteredTracks(theTracks,tkminPt_,tkminPixelHitsn_,tkminTrackerHitsn_,tkmaxipt_,tkmaxChi2_,thePV);
  if (UseTrackQuality_) theFilteredTracks = filterTracksByQualityBit(theFilteredTracks,tkQuality_);
  resultExtended.setTracks(theFilteredTracks);
  
  //resultExtended.setpositionAndEnergyECALRecHits(getPositionAndEnergyEcalRecHits(theEvent,theEventSetup,theCaloJet));

  std::vector<BasicClusterRef> theNeutralEcalBasicClusters=getNeutralEcalBasicClusters(theEvent,theEventSetup,theCaloJet,theFilteredTracks,ECALBasicClustersAroundCaloJet_DRConeSize_,ECALBasicClusterminE_,ECALBasicClusterpropagTrack_matchingDRConeSize_);
  resultExtended.setneutralECALBasicClusters(theNeutralEcalBasicClusters);
  
  return resultExtended; 
}
CaloTauTagInfo CaloRecoTauTagInfoAlgorithm::buildCaloTauTagInfo(edm::Event& theEvent,const edm::EventSetup& theEventSetup,const JetBaseRef& theJet,const TrackRefVector& theTracks,const Vertex& thePV){
  CaloTauTagInfo resultExtended;
  resultExtended.setJetRef(theJet);

  TrackRefVector theFilteredTracks;
  if (UsePVconstraint_) theFilteredTracks=TauTagTools::filteredTracks(theTracks,tkminPt_,tkminPixelHitsn_,tkminTrackerHitsn_,tkmaxipt_,tkmaxChi2_,tkPVmaxDZ_,thePV, thePV.z());
  else theFilteredTracks=TauTagTools::filteredTracks(theTracks,tkminPt_,tkminPixelHitsn_,tkminTrackerHitsn_,tkmaxipt_,tkmaxChi2_,thePV);
  if (UseTrackQuality_) theFilteredTracks = filterTracksByQualityBit(theFilteredTracks,tkQuality_);
  resultExtended.setTracks(theFilteredTracks);

  //resultExtended.setpositionAndEnergyECALRecHits(getPositionAndEnergyEcalRecHits(theEvent,theEventSetup,theCaloJet));

  //reco::JPTJetRef const theJPTJetRef = theJet.castTo<reco::JPTJetRef>();
  //reco::CaloJetRef const theCaloJet = (theJPTJetRef->getCaloJetRef()).castTo<reco::CaloJetRef>();
  reco::CaloJetRef const theCaloJet = resultExtended.calojetRef();
  std::vector<BasicClusterRef> theNeutralEcalBasicClusters=getNeutralEcalBasicClusters(theEvent,theEventSetup,theCaloJet,theFilteredTracks,ECALBasicClustersAroundCaloJet_DRConeSize_,ECALBasicClusterminE_,ECALBasicClusterpropagTrack_matchingDRConeSize_);
  resultExtended.setneutralECALBasicClusters(theNeutralEcalBasicClusters);

  return resultExtended;
}
/*
std::vector<std::pair<math::XYZPoint,float> > CaloRecoTauTagInfoAlgorithm::getPositionAndEnergyEcalRecHits(edm::Event& theEvent,const edm::EventSetup& theEventSetup,const CaloJetRef& theCaloJet){
  std::vector<std::pair<math::XYZPoint,float> > thePositionAndEnergyEcalRecHits;
  std::vector<CaloTowerPtr> theCaloTowers=theCaloJet->getCaloConstituents();
  ESHandle<CaloGeometry> theCaloGeometry;
  theEventSetup.get<CaloGeometryRecord>().get(theCaloGeometry);
  const CaloSubdetectorGeometry* theCaloSubdetectorGeometry;  
  edm::Handle<EBRecHitCollection> EBRecHits;
  edm::Handle<EERecHitCollection> EERecHits; 
  edm::Handle<ESRecHitCollection> ESRecHits; 
  theEvent.getByLabel(EBRecHitsLabel_,EBRecHits);
  theEvent.getByLabel(EERecHitsLabel_,EERecHits);
  theEvent.getByLabel(ESRecHitsLabel_,ESRecHits);
  for(std::vector<CaloTowerPtr>::const_iterator i_Tower=theCaloTowers.begin();i_Tower!=theCaloTowers.end();i_Tower++){
    size_t numRecHits = (**i_Tower).constituentsSize();
    for(size_t j=0;j<numRecHits;j++) {
      DetId RecHitDetID=(**i_Tower).constituent(j);      


      DetId::Detector DetNum=RecHitDetID.det();     
      if(DetNum==DetId::Ecal){
	if((EcalSubdetector)RecHitDetID.subdetId()==EcalBarrel){
	  theCaloSubdetectorGeometry = theCaloGeometry->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
	  EBDetId EcalID=RecHitDetID;
	  EBRecHitCollection::const_iterator theRecHit=EBRecHits->find(EcalID);
	  const CaloCellGeometry* theRecHitCell=theCaloSubdetectorGeometry->getGeometry(RecHitDetID);
	  math::XYZPoint theRecHitCell_XYZPoint(theRecHitCell->getPosition().x(),theRecHitCell->getPosition().y(),theRecHitCell->getPosition().z());
	  pair<math::XYZPoint,float> thePositionAndEnergyEcalRecHit(theRecHitCell_XYZPoint,theRecHit->energy());
	  thePositionAndEnergyEcalRecHits.push_back(thePositionAndEnergyEcalRecHit);
	}else if((EcalSubdetector)RecHitDetID.subdetId()==EcalEndcap){
	  theCaloSubdetectorGeometry = theCaloGeometry->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);
	  EEDetId EcalID = RecHitDetID;
	  EERecHitCollection::const_iterator theRecHit=EERecHits->find(EcalID);	    
	  const CaloCellGeometry* theRecHitCell=theCaloSubdetectorGeometry->getGeometry(RecHitDetID);
	  math::XYZPoint theRecHitCell_XYZPoint(theRecHitCell->getPosition().x(),theRecHitCell->getPosition().y(),theRecHitCell->getPosition().z());
	  pair<math::XYZPoint,float> thePositionAndEnergyEcalRecHit(theRecHitCell_XYZPoint,theRecHit->energy());
	  thePositionAndEnergyEcalRecHits.push_back(thePositionAndEnergyEcalRecHit);
	}else if((EcalSubdetector)RecHitDetID.subdetId()==EcalPreshower){
	  theCaloSubdetectorGeometry = theCaloGeometry->getSubdetectorGeometry(DetId::Ecal,EcalPreshower);
	  ESDetId EcalID = RecHitDetID;
	  ESRecHitCollection::const_iterator theRecHit=ESRecHits->find(EcalID);	    
	  const CaloCellGeometry* theRecHitCell=theCaloSubdetectorGeometry->getGeometry(RecHitDetID);
	  math::XYZPoint theRecHitCell_XYZPoint(theRecHitCell->getPosition().x(),theRecHitCell->getPosition().y(),theRecHitCell->getPosition().z());
	  pair<math::XYZPoint,float> thePositionAndEnergyEcalRecHit(theRecHitCell_XYZPoint,theRecHit->energy());
	  thePositionAndEnergyEcalRecHits.push_back(thePositionAndEnergyEcalRecHit);
	}	 
      }	
    }
  }
  return thePositionAndEnergyEcalRecHits;
}
*/

std::vector<DetId> CaloRecoTauTagInfoAlgorithm::getVectorDetId(const CaloJetRef& theCaloJet){
  std::vector<CaloTowerPtr> theCaloTowers=theCaloJet->getCaloConstituents();
  std::vector<DetId> myDetIds;
  myDetIds.clear();

  for(std::vector<CaloTowerPtr>::const_iterator i_Tower=theCaloTowers.begin();i_Tower!=theCaloTowers.end();i_Tower++){
    size_t numRecHits = (**i_Tower).constituentsSize();
    for(size_t j=0;j<numRecHits;j++) {
      DetId RecHitDetID=(**i_Tower).constituent(j);      

      myDetIds.push_back(RecHitDetID);
    }
  }
  return myDetIds;
}


std::vector<BasicClusterRef> CaloRecoTauTagInfoAlgorithm::getNeutralEcalBasicClusters(edm::Event& theEvent,const edm::EventSetup& theEventSetup,const CaloJetRef& theCaloJet,const TrackRefVector& theTracks,float theECALBasicClustersAroundCaloJet_DRConeSize,float theECALBasicClusterminE,float theECALBasicClusterpropagTrack_matchingDRConeSize){
  std::vector<math::XYZPoint> thepropagTracksECALSurfContactPoints;
  edm::ESHandle<MagneticField> theMF;
  theEventSetup.get<IdealMagneticFieldRecord>().get(theMF);
  const MagneticField* theMagField=theMF.product();
  for(TrackRefVector::const_iterator i_Track=theTracks.begin();i_Track!=theTracks.end();i_Track++){
    math::XYZPoint thepropagTrackECALSurfContactPoint=TauTagTools::propagTrackECALSurfContactPoint(theMagField,*i_Track);
    if(thepropagTrackECALSurfContactPoint.R()!=0.) thepropagTracksECALSurfContactPoints.push_back(thepropagTrackECALSurfContactPoint);
  }
  
  math::XYZPoint aCaloJetFakePosition((*theCaloJet).px(),(*theCaloJet).py(),(*theCaloJet).pz());
    
  std::vector<BasicClusterRef> theBasicClusters; 
  
  edm::Handle<BasicClusterCollection> theBarrelBCCollection;
  //  theEvent.getByLabel("islandBasicClusters","islandBarrelBasicClusters",theBarrelBCCollection);
  theEvent.getByLabel(BarrelBasicClusters_,theBarrelBCCollection);
  for(unsigned int i_BC=0;i_BC!=theBarrelBCCollection->size();i_BC++) { 
    BasicClusterRef theBasicClusterRef(theBarrelBCCollection,i_BC);    
    if (theBasicClusterRef.isNull()) continue;  
    if (ROOT::Math::VectorUtil::DeltaR(aCaloJetFakePosition,(*theBasicClusterRef).position())<=theECALBasicClustersAroundCaloJet_DRConeSize && (*theBasicClusterRef).energy()>=theECALBasicClusterminE) theBasicClusters.push_back(theBasicClusterRef);
  }
  edm::Handle<BasicClusterCollection> theEndcapBCCollection;
  //  theEvent.getByLabel("islandBasicClusters","islandEndcapBasicClusters",theEndcapBCCollection);
  theEvent.getByLabel(EndcapBasicClusters_,theEndcapBCCollection);
  for(unsigned int j_BC=0;j_BC!=theEndcapBCCollection->size();j_BC++) { 
    BasicClusterRef theBasicClusterRef(theEndcapBCCollection,j_BC); 
    if (theBasicClusterRef.isNull()) continue;  
    if (ROOT::Math::VectorUtil::DeltaR(aCaloJetFakePosition,(*theBasicClusterRef).position())<=theECALBasicClustersAroundCaloJet_DRConeSize && (*theBasicClusterRef).energy()>=theECALBasicClusterminE) theBasicClusters.push_back(theBasicClusterRef);
  }  
  
  std::vector<BasicClusterRef> theNeutralBasicClusters=theBasicClusters;  
  std::vector<BasicClusterRef>::iterator kmatchedBasicCluster;
  for (std::vector<math::XYZPoint>::iterator ipropagTrackECALSurfContactPoint=thepropagTracksECALSurfContactPoints.begin();ipropagTrackECALSurfContactPoint!=thepropagTracksECALSurfContactPoints.end();ipropagTrackECALSurfContactPoint++) {
    double theMatchedEcalBasicClusterpropagTrack_minDR=theECALBasicClusterpropagTrack_matchingDRConeSize;
    bool Track_matchedwithEcalBasicCluster=false;
    for (std::vector<BasicClusterRef>::iterator jBasicCluster=theNeutralBasicClusters.begin();jBasicCluster!=theNeutralBasicClusters.end();jBasicCluster++) {
      if(ROOT::Math::VectorUtil::DeltaR((*ipropagTrackECALSurfContactPoint),(**jBasicCluster).position())<theMatchedEcalBasicClusterpropagTrack_minDR){
      	Track_matchedwithEcalBasicCluster=true;
	theMatchedEcalBasicClusterpropagTrack_minDR=ROOT::Math::VectorUtil::DeltaR((*ipropagTrackECALSurfContactPoint),(**jBasicCluster).position());
	kmatchedBasicCluster=jBasicCluster;
      }
    }
    if(Track_matchedwithEcalBasicCluster) kmatchedBasicCluster=theNeutralBasicClusters.erase(kmatchedBasicCluster);
  }
  return theNeutralBasicClusters;
}

TrackRefVector CaloRecoTauTagInfoAlgorithm::filterTracksByQualityBit(const TrackRefVector& tracks, reco::TrackBase::TrackQuality quality) const
{
  TrackRefVector filteredTracks;
  for (TrackRefVector::const_iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++) {
    if ((*iTrack)->quality(quality)) filteredTracks.push_back(*iTrack);
  }
  return filteredTracks;
}
