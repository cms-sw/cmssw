#include "RecoTauTag/RecoTau/interface/CaloRecoTauTagInfoAlgorithm.h"

CaloRecoTauTagInfoAlgorithm::CaloRecoTauTagInfoAlgorithm(const ParameterSet& parameters){
  // parameters of the considered rec. Tracks (catched through a JetTracksAssociation object) :
  tkminPt_                            = parameters.getParameter<double>("tkminPt");
  tkminPixelHitsn_                    = parameters.getParameter<int>("tkminPixelHitsn");
  tkminTrackerHitsn_                  = parameters.getParameter<int>("tkminTrackerHitsn");
  tkmaxipt_                           = parameters.getParameter<double>("tkmaxipt");
  tkmaxChi2_                          = parameters.getParameter<double>("tkmaxChi2");
  // 
  UsePVconstraint_                    = parameters.getParameter<bool>("UsePVconstraint");
  tkPVmaxDZ_                          = parameters.getParameter<double>("tkPVmaxDZ");
  // parameters of the considered EcalRecHits 
  EBRecHitsLabel_                     = parameters.getParameter<InputTag>("EBRecHitsSource"); 
  EERecHitsLabel_                     = parameters.getParameter<InputTag>("EERecHitsSource"); 
  ESRecHitsLabel_                     = parameters.getParameter<InputTag>("ESRecHitsSource"); 
  // parameters of the considered neutral ECAL BasicClusters
  ECALBasicClustersAroundCaloJet_DRConeSize_      = parameters.getParameter<double>("ECALBasicClustersAroundCaloJet_DRConeSize");
  ECALBasicClusterminE_                           = parameters.getParameter<double>("ECALBasicClusterminE");
  ECALBasicClusterpropagTrack_matchingDRConeSize_ = parameters.getParameter<double>("ECALBasicClusterpropagTrack_matchingDRConeSize");
}
  
CaloTauTagInfo CaloRecoTauTagInfoAlgorithm::buildCaloTauTagInfo(Event& theEvent,const EventSetup& theEventSetup,const CaloJetRef& theCaloJet,const TrackRefVector& theTracks,const Vertex& thePV){
  CaloTauTagInfo resultExtended;
  resultExtended.setcalojetRef(theCaloJet);

  TrackRefVector theFilteredTracks;
  if (UsePVconstraint_) theFilteredTracks=TauTagTools::filteredTracks(theTracks,tkminPt_,tkminPixelHitsn_,tkminTrackerHitsn_,tkmaxipt_,tkmaxChi2_,tkPVmaxDZ_,thePV, thePV.z());
  else theFilteredTracks=TauTagTools::filteredTracks(theTracks,tkminPt_,tkminPixelHitsn_,tkminTrackerHitsn_,tkmaxipt_,tkmaxChi2_,thePV);
  resultExtended.setTracks(theFilteredTracks);
  
  resultExtended.setpositionAndEnergyECALRecHits(getPositionAndEnergyEcalRecHits(theEvent,theEventSetup,theCaloJet));

  BasicClusterRefVector theNeutralEcalBasicClusters=getNeutralEcalBasicClusters(theEvent,theEventSetup,theCaloJet,theFilteredTracks,ECALBasicClustersAroundCaloJet_DRConeSize_,ECALBasicClusterminE_,ECALBasicClusterpropagTrack_matchingDRConeSize_);
  resultExtended.setneutralECALBasicClusters(theNeutralEcalBasicClusters);
  
  return resultExtended; 
}

vector<pair<math::XYZPoint,float> > CaloRecoTauTagInfoAlgorithm::getPositionAndEnergyEcalRecHits(Event& theEvent,const EventSetup& theEventSetup,const CaloJetRef& theCaloJet){
  vector<pair<math::XYZPoint,float> > thePositionAndEnergyEcalRecHits;
  vector<CaloTowerRef> theCaloTowers=theCaloJet->getConstituents();
  ESHandle<CaloGeometry> theCaloGeometry;
  theEventSetup.get<IdealGeometryRecord>().get(theCaloGeometry);
  const CaloSubdetectorGeometry* theCaloSubdetectorGeometry;  
  Handle<EBRecHitCollection> EBRecHits;
  Handle<EERecHitCollection> EERecHits; 
  Handle<ESRecHitCollection> ESRecHits; 
  theEvent.getByLabel(EBRecHitsLabel_,EBRecHits);
  theEvent.getByLabel(EERecHitsLabel_,EERecHits);
  theEvent.getByLabel(ESRecHitsLabel_,ESRecHits);
  for(vector<CaloTowerRef>::const_iterator i_Tower=theCaloTowers.begin();i_Tower!=theCaloTowers.end();i_Tower++){
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

BasicClusterRefVector CaloRecoTauTagInfoAlgorithm::getNeutralEcalBasicClusters(Event& theEvent,const EventSetup& theEventSetup,const CaloJetRef& theCaloJet,const TrackRefVector& theTracks,float theECALBasicClustersAroundCaloJet_DRConeSize,float theECALBasicClusterminE,float theECALBasicClusterpropagTrack_matchingDRConeSize){
  vector<math::XYZPoint> thepropagTracksECALSurfContactPoints;
  ESHandle<MagneticField> theMF;
  theEventSetup.get<IdealMagneticFieldRecord>().get(theMF);
  const MagneticField* theMagField=theMF.product();
  for(TrackRefVector::const_iterator i_Track=theTracks.begin();i_Track!=theTracks.end();i_Track++){
    math::XYZPoint thepropagTrackECALSurfContactPoint=TauTagTools::propagTrackECALSurfContactPoint(theMagField,*i_Track);
    if(thepropagTrackECALSurfContactPoint.z()!=0.) thepropagTracksECALSurfContactPoints.push_back(thepropagTrackECALSurfContactPoint);
  }
  
  math::XYZPoint aCaloJetFakePosition((*theCaloJet).px(),(*theCaloJet).py(),(*theCaloJet).pz());
    
  BasicClusterRefVector theBasicClusters; 
  
  double ECALcorner_tantheta=ECALBounds::barrel_innerradius()/ECALBounds::barrel_halfLength();
  if (fabs((*theCaloJet).pt()/(*theCaloJet).pz())>ECALcorner_tantheta){ 
    Handle<BasicClusterCollection> theBarrelBCCollection;
    theEvent.getByLabel("islandBasicClusters","islandBarrelBasicClusters",theBarrelBCCollection);
    for(unsigned int i_BC=0;i_BC!=theBarrelBCCollection->size();i_BC++) { 
      BasicClusterRef theBasicClusterRef(theBarrelBCCollection,i_BC);    
      if (theBasicClusterRef.isNull()) continue;  
      if (ROOT::Math::VectorUtil::DeltaR(aCaloJetFakePosition,(*theBasicClusterRef).position())<=theECALBasicClustersAroundCaloJet_DRConeSize && (*theBasicClusterRef).energy()>=theECALBasicClusterminE) theBasicClusters.push_back(theBasicClusterRef);
    }  
  }else{    
    Handle<BasicClusterCollection> theEndcapBCCollection;
    theEvent.getByLabel("islandBasicClusters","islandEndcapBasicClusters",theEndcapBCCollection);
    for(unsigned int j_BC=0;j_BC!=theEndcapBCCollection->size();j_BC++) { 
      BasicClusterRef theBasicClusterRef(theEndcapBCCollection,j_BC); 
      if (theBasicClusterRef.isNull()) continue;  
      if (ROOT::Math::VectorUtil::DeltaR(aCaloJetFakePosition,(*theBasicClusterRef).position())<=theECALBasicClustersAroundCaloJet_DRConeSize && (*theBasicClusterRef).energy()>=theECALBasicClusterminE) theBasicClusters.push_back(theBasicClusterRef);
    }  
  }
  
  BasicClusterRefVector theNeutralBasicClusters=theBasicClusters;  
  BasicClusterRefVector::iterator kmatchedBasicCluster;
  for (vector<math::XYZPoint>::iterator ipropagTrackECALSurfContactPoint=thepropagTracksECALSurfContactPoints.begin();ipropagTrackECALSurfContactPoint!=thepropagTracksECALSurfContactPoints.end();ipropagTrackECALSurfContactPoint++) {
    double theMatchedEcalBasicClusterpropagTrack_minDR=theECALBasicClusterpropagTrack_matchingDRConeSize;
    bool Track_matchedwithEcalBasicCluster=false;
    for (BasicClusterRefVector::iterator jBasicCluster=theNeutralBasicClusters.begin();jBasicCluster!=theNeutralBasicClusters.end();jBasicCluster++) {
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
