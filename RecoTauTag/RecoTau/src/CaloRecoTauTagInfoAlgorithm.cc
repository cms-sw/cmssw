#include "RecoTauTag/RecoTau/interface/CaloRecoTauTagInfoAlgorithm.h"

CaloRecoTauTagInfoAlgorithm::CaloRecoTauTagInfoAlgorithm(const ParameterSet& parameters) : chargedpi_mass_(0.13957018){
  // parameters of the considered rec. Tracks (catched through a JetTracksAssociator object) :
  tkminPt_                            = parameters.getParameter<double>("tkminPt");
  tkminPixelHitsn_                    = parameters.getParameter<int>("tkminPixelHitsn");
  tkminTrackerHitsn_                  = parameters.getParameter<int>("tkminTrackerHitsn");
  tkmaxipt_                           = parameters.getParameter<double>("tkmaxipt");
  tkmaxChi2_                          = parameters.getParameter<double>("tkmaxChi2");
  // 
  UsePVconstraint_                    = parameters.getParameter<bool>("UsePVconstraint");
  tkPVmaxDZ_                          = parameters.getParameter<double>("tkPVmaxDZ");
  // parameters of the considered neutral ECAL BasicClusters
  ECALBasicClustersAroundCaloJet_DRConeSize_      = parameters.getParameter<double>("ECALBasicClustersAroundCaloJet_DRConeSize");
  ECALBasicClusterminE_                           = parameters.getParameter<double>("ECALBasicClusterminE");
  ECALBasicClusterpropagTrack_matchingDRConeSize_ = parameters.getParameter<double>("ECALBasicClusterpropagTrack_matchingDRConeSize");
}
  
CaloTauTagInfo CaloRecoTauTagInfoAlgorithm::buildCaloTauTagInfo(Event& theEvent,const EventSetup& theEventSetup,const CaloJetRef& theCaloJet,const TrackRefVector& theTracks,const Vertex& thePV){
  CaloTauTagInfo resultExtended;
  resultExtended.setcalojetRef(theCaloJet);

  TrackRefVector theFilteredTracks;
  if (UsePVconstraint_) theFilteredTracks=TauTagTools::filteredTracks(theTracks,tkminPt_,tkminPixelHitsn_,tkminTrackerHitsn_,tkmaxipt_,tkmaxChi2_,tkPVmaxDZ_,thePV.z());
  else theFilteredTracks=TauTagTools::filteredTracks(theTracks,tkminPt_,tkminPixelHitsn_,tkminTrackerHitsn_,tkmaxipt_,tkmaxChi2_);
  resultExtended.setTracks(theFilteredTracks);
  
  resultExtended.setpositionAndEnergyECALRecHits(getPositionAndEnergyEcalRecHits(theEvent,theEventSetup,theCaloJet));

  BasicClusterRefVector theNeutralEcalBasicClusters=getNeutralEcalBasicClusters(theEvent,theEventSetup,theCaloJet,theFilteredTracks,ECALBasicClustersAroundCaloJet_DRConeSize_,ECALBasicClusterminE_,ECALBasicClusterpropagTrack_matchingDRConeSize_);
  resultExtended.setneutralECALBasicClusters(theNeutralEcalBasicClusters);

  math::XYZTLorentzVector alternatLorentzVect;
  alternatLorentzVect.SetPx(0.);
  alternatLorentzVect.SetPy(0.);
  alternatLorentzVect.SetPz(0.);
  alternatLorentzVect.SetE(0.);
  for(TrackRefVector::iterator iTrack=theFilteredTracks.begin();iTrack!=theFilteredTracks.end();iTrack++) {
    // build a charged pion candidate Lorentz vector from a Track
    math::XYZTLorentzVector iChargedPionCand_XYZTLorentzVect((**iTrack).momentum().x(),(**iTrack).momentum().y(),(**iTrack).momentum().z(),sqrt(pow((double)(**iTrack).momentum().r(),2)+pow(chargedpi_mass_,2)));
    alternatLorentzVect+=iChargedPionCand_XYZTLorentzVect;
  }
  for(BasicClusterRefVector::iterator iBasicCluster=theNeutralEcalBasicClusters.begin();iBasicCluster!=theNeutralEcalBasicClusters.end();iBasicCluster++) {
    // build a gamma candidate Lorentz vector from a neutral ECAL BasicCluster
    double iGammaCand_px=(**iBasicCluster).energy()*sin((**iBasicCluster).position().theta())*cos((**iBasicCluster).position().phi());
    double iGammaCand_py=(**iBasicCluster).energy()*sin((**iBasicCluster).position().theta())*sin((**iBasicCluster).position().phi());
    double iGammaCand_pz=(**iBasicCluster).energy()*cos((**iBasicCluster).position().theta());
    math::XYZTLorentzVector iGammaCand_XYZTLorentzVect(iGammaCand_px,iGammaCand_py,iGammaCand_pz,(**iBasicCluster).energy());
    alternatLorentzVect+=iGammaCand_XYZTLorentzVect;
  }
  resultExtended.setalternatLorentzVect(alternatLorentzVect);
  
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
  theEvent.getByLabel("ecalRecHit","EcalRecHitsEB",EBRecHits);
  theEvent.getByLabel("ecalRecHit","EcalRecHitsEE",EERecHits);
  for(vector<CaloTowerRef>::const_iterator i_Tower=theCaloTowers.begin();i_Tower!=theCaloTowers.end();i_Tower++){
    size_t numRecHits = (**i_Tower).constituentsSize();
    for(size_t j=0;j<numRecHits;j++) {
      DetId RecHitDetID=(**i_Tower).constituent(j);
      DetId::Detector DetNum=RecHitDetID.det();     
      if(DetNum==DetId::Ecal){
	int EcalNum=RecHitDetID.subdetId();
	cout<<"EcalNum "<<EcalNum<<endl;
	if(EcalNum==1){
	  theCaloSubdetectorGeometry = theCaloGeometry->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
	  EBDetId EcalID=RecHitDetID;
	  EBRecHitCollection::const_iterator theRecHit=EBRecHits->find(EcalID);
	  const CaloCellGeometry* theRecHitCell=theCaloSubdetectorGeometry->getGeometry(RecHitDetID);
	  math::XYZPoint theRecHitCell_XYZPoint(theRecHitCell->getPosition().x(),theRecHitCell->getPosition().y(),theRecHitCell->getPosition().z());
	  pair<math::XYZPoint,float> thePositionAndEnergyEcalRecHit(theRecHitCell_XYZPoint,theRecHit->energy());
	  thePositionAndEnergyEcalRecHits.push_back(thePositionAndEnergyEcalRecHit);
	}else if(EcalNum==2){
	  theCaloSubdetectorGeometry = theCaloGeometry->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);
	  EEDetId EcalID = RecHitDetID;
	  EERecHitCollection::const_iterator theRecHit=EERecHits->find(EcalID);	    
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
  
  BasicClusterRefVector theBasicClusters;  
  Handle<BasicClusterCollection> theBarrelBCCollection;
  theEvent.getByLabel("islandBasicClusters","islandBarrelBasicClusters",theBarrelBCCollection);
  int iBC=0;
  for(BasicClusterCollection::const_iterator i_BC=theBarrelBCCollection->begin();i_BC!=theBarrelBCCollection->end();i_BC++) {
    math::XYZPoint aCaloJetFakePosition((*theCaloJet).px(),(*theCaloJet).py(),(*theCaloJet).pz());
    if (ROOT::Math::VectorUtil::DeltaR(aCaloJetFakePosition,(*i_BC).position())<=theECALBasicClustersAroundCaloJet_DRConeSize && (*i_BC).energy()>=theECALBasicClusterminE){
      BasicClusterRef theBasicClusterRef(theBarrelBCCollection,iBC);    
      theBasicClusters.push_back(theBasicClusterRef);
    }
    ++iBC;
  }
  Handle<BasicClusterCollection> theEndcapBCCollection;
  theEvent.getByLabel("islandBasicClusters","islandEndcapBasicClusters",theEndcapBCCollection);
  iBC=0;
  for(BasicClusterCollection::const_iterator i_BC=theEndcapBCCollection->begin();i_BC!=theEndcapBCCollection->end();i_BC++) {
    math::XYZPoint aCaloJetFakePosition((*theCaloJet).px(),(*theCaloJet).py(),(*theCaloJet).pz());
    if (ROOT::Math::VectorUtil::DeltaR(aCaloJetFakePosition,(*i_BC).position())<=theECALBasicClustersAroundCaloJet_DRConeSize && (*i_BC).energy()>=theECALBasicClusterminE){
      BasicClusterRef theBasicClusterRef(theEndcapBCCollection,iBC);    
    theBasicClusters.push_back(theBasicClusterRef);
    }
    ++iBC;
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
