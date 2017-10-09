#include "RecoTauTag/RecoTau/interface/CaloRecoTauAlgorithm.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/JetReco/interface/JetCollection.h"

using namespace reco;

CaloRecoTauAlgorithm::CaloRecoTauAlgorithm() : TransientTrackBuilder_(0),MagneticField_(0),chargedpi_mass_(0.13957018){}  
CaloRecoTauAlgorithm::CaloRecoTauAlgorithm(const edm::ParameterSet& iConfig) : TransientTrackBuilder_(0),MagneticField_(0),chargedpi_mass_(0.13957018){
  LeadTrack_minPt_                    = iConfig.getParameter<double>("LeadTrack_minPt");
  Track_minPt_                        = iConfig.getParameter<double>("Track_minPt");
  IsolationTrack_minPt_               = iConfig.getParameter<double>("IsolationTrack_minPt");
  IsolationTrack_minHits_             = iConfig.getParameter<unsigned int>("IsolationTrack_minHits");
  UseTrackLeadTrackDZconstraint_      = iConfig.getParameter<bool>("UseTrackLeadTrackDZconstraint");
  TrackLeadTrack_maxDZ_               = iConfig.getParameter<double>("TrackLeadTrack_maxDZ");
  ECALRecHit_minEt_                   = iConfig.getParameter<double>("ECALRecHit_minEt");       
  
  MatchingConeMetric_                 = iConfig.getParameter<std::string>("MatchingConeMetric");
  MatchingConeSizeFormula_            = iConfig.getParameter<std::string>("MatchingConeSizeFormula");
  MatchingConeSize_min_               = iConfig.getParameter<double>("MatchingConeSize_min");
  MatchingConeSize_max_               = iConfig.getParameter<double>("MatchingConeSize_max");
  TrackerSignalConeMetric_            = iConfig.getParameter<std::string>("TrackerSignalConeMetric");
  TrackerSignalConeSizeFormula_       = iConfig.getParameter<std::string>("TrackerSignalConeSizeFormula");
  TrackerSignalConeSize_min_          = iConfig.getParameter<double>("TrackerSignalConeSize_min");
  TrackerSignalConeSize_max_          = iConfig.getParameter<double>("TrackerSignalConeSize_max");
  TrackerIsolConeMetric_              = iConfig.getParameter<std::string>("TrackerIsolConeMetric"); 
  TrackerIsolConeSizeFormula_         = iConfig.getParameter<std::string>("TrackerIsolConeSizeFormula"); 
  TrackerIsolConeSize_min_            = iConfig.getParameter<double>("TrackerIsolConeSize_min");
  TrackerIsolConeSize_max_            = iConfig.getParameter<double>("TrackerIsolConeSize_max");
  ECALSignalConeMetric_               = iConfig.getParameter<std::string>("ECALSignalConeMetric");
  ECALSignalConeSizeFormula_          = iConfig.getParameter<std::string>("ECALSignalConeSizeFormula");    
  ECALSignalConeSize_min_             = iConfig.getParameter<double>("ECALSignalConeSize_min");
  ECALSignalConeSize_max_             = iConfig.getParameter<double>("ECALSignalConeSize_max");
  ECALIsolConeMetric_                 = iConfig.getParameter<std::string>("ECALIsolConeMetric");
  ECALIsolConeSizeFormula_            = iConfig.getParameter<std::string>("ECALIsolConeSizeFormula");      
  ECALIsolConeSize_min_               = iConfig.getParameter<double>("ECALIsolConeSize_min");
  ECALIsolConeSize_max_               = iConfig.getParameter<double>("ECALIsolConeSize_max");
  
  EBRecHitsLabel_                     = iConfig.getParameter<edm::InputTag>("EBRecHitsSource"); 
  EERecHitsLabel_                     = iConfig.getParameter<edm::InputTag>("EERecHitsSource"); 
  ESRecHitsLabel_                     = iConfig.getParameter<edm::InputTag>("ESRecHitsSource"); 



  AreaMetric_recoElements_maxabsEta_  = iConfig.getParameter<double>("AreaMetric_recoElements_maxabsEta");

  //Computing the TFormula
  myTrackerSignalConeSizeTFormula=TauTagTools::computeConeSizeTFormula(TrackerSignalConeSizeFormula_,"Tracker signal cone size");
  myTrackerIsolConeSizeTFormula=TauTagTools::computeConeSizeTFormula(TrackerIsolConeSizeFormula_,"Tracker isolation cone size");
  myECALSignalConeSizeTFormula=TauTagTools::computeConeSizeTFormula(ECALSignalConeSizeFormula_,"ECAL signal cone size");
  myECALIsolConeSizeTFormula=TauTagTools::computeConeSizeTFormula(ECALIsolConeSizeFormula_,"ECAL isolation cone size");
  myMatchingConeSizeTFormula=TauTagTools::computeConeSizeTFormula(MatchingConeSizeFormula_,"Matching cone size");

  mySelectedDetId_.clear();
}
void CaloRecoTauAlgorithm::setTransientTrackBuilder(const TransientTrackBuilder* x){TransientTrackBuilder_=x;}
void CaloRecoTauAlgorithm::setMagneticField(const MagneticField* x){MagneticField_=x;} 

CaloTau CaloRecoTauAlgorithm::buildCaloTau(edm::Event& iEvent,const edm::EventSetup& iSetup,const CaloTauTagInfoRef& myCaloTauTagInfoRef,const Vertex& myPV){
  CaloJetRef myCaloJet=(*myCaloTauTagInfoRef).calojetRef(); // catch a ref to the initial CaloJet  
  const std::vector<CaloTowerPtr> myCaloTowers=(*myCaloJet).getCaloConstituents();
  JetBaseRef jetRef = myCaloTauTagInfoRef->jetRef();
  CaloTau myCaloTau(std::numeric_limits<int>::quiet_NaN(),jetRef->p4()); // create the CaloTau with the corrected Lorentz-vector
  
  myCaloTau.setcaloTauTagInfoRef(myCaloTauTagInfoRef);
  
  TrackRefVector myTks=(*myCaloTauTagInfoRef).Tracks();
  
  CaloTauElementsOperators myCaloTauElementsOperators(myCaloTau);
  double myMatchingConeSize=myCaloTauElementsOperators.computeConeSize(myMatchingConeSizeTFormula,MatchingConeSize_min_,MatchingConeSize_max_);
  TrackRef myleadTk=myCaloTauElementsOperators.leadTk(MatchingConeMetric_,myMatchingConeSize,LeadTrack_minPt_);
  double myCaloTau_refInnerPosition_x=0.;
  double myCaloTau_refInnerPosition_y=0.;
  double myCaloTau_refInnerPosition_z=0.;
  if(myleadTk.isNonnull()){
    myCaloTau.setleadTrack(myleadTk);
    double myleadTkDZ=(*myleadTk).dz(myPV.position());
    if(TransientTrackBuilder_!=0)
    { 
      const TransientTrack myleadTransientTk=TransientTrackBuilder_->build(&(*myleadTk));
      GlobalVector myCaloJetdir((*myCaloJet).px(),(*myCaloJet).py(),(*myCaloJet).pz());
      if(IPTools::signedTransverseImpactParameter(myleadTransientTk,myCaloJetdir,myPV).first)
	myCaloTau.setleadTracksignedSipt(IPTools::signedTransverseImpactParameter(myleadTransientTk,myCaloJetdir,myPV).second.significance());
    }
    if((*myleadTk).innerOk()){
      myCaloTau_refInnerPosition_x=(*myleadTk).innerPosition().x(); 
      myCaloTau_refInnerPosition_y=(*myleadTk).innerPosition().y(); 
      myCaloTau_refInnerPosition_z=(*myleadTk).innerPosition().z(); 
    }
    
    if(MagneticField_!=0){ 
      math::XYZPoint mypropagleadTrackECALSurfContactPoint=TauTagTools::propagTrackECALSurfContactPoint(MagneticField_,myleadTk);
      if(mypropagleadTrackECALSurfContactPoint.R()!=0.){
	double myleadTrackHCAL3x3hottesthitDEta=0.;
	double myleadTrackHCAL3x3hottesthitEt=0.;
	double myleadTrackHCAL3x3hitsEtSum=0.;
	edm::ESHandle<CaloGeometry> myCaloGeometry;
	iSetup.get<CaloGeometryRecord>().get(myCaloGeometry);
	const CaloSubdetectorGeometry* myCaloSubdetectorGeometry=(*myCaloGeometry).getSubdetectorGeometry(DetId::Calo,CaloTowerDetId::SubdetId);
	edm::ESHandle<CaloTowerTopology> caloTowerTopology;
	iSetup.get<HcalRecNumberingRecord>().get(caloTowerTopology);
	CaloTowerDetId mypropagleadTrack_closestCaloTowerId((*myCaloSubdetectorGeometry).getClosestCell(GlobalPoint(mypropagleadTrackECALSurfContactPoint.x(),
														    mypropagleadTrackECALSurfContactPoint.y(),
														    mypropagleadTrackECALSurfContactPoint.z())));
	std::vector<CaloTowerDetId> mypropagleadTrack_closestCaloTowerNeighbourIds=getCaloTowerneighbourDetIds(myCaloSubdetectorGeometry, *caloTowerTopology, mypropagleadTrack_closestCaloTowerId);
	for(std::vector<CaloTowerPtr>::const_iterator iCaloTower=myCaloTowers.begin();iCaloTower!=myCaloTowers.end();iCaloTower++){
	  CaloTowerDetId iCaloTowerId((**iCaloTower).id());
	  bool CaloTower_inside3x3matrix=false;
	  if (iCaloTowerId==mypropagleadTrack_closestCaloTowerId) CaloTower_inside3x3matrix=true;
	  if (!CaloTower_inside3x3matrix){
	    for(std::vector<CaloTowerDetId>::const_iterator iCaloTowerDetId=mypropagleadTrack_closestCaloTowerNeighbourIds.begin();iCaloTowerDetId!=mypropagleadTrack_closestCaloTowerNeighbourIds.end();iCaloTowerDetId++){
	      if (iCaloTowerId==(*iCaloTowerDetId)){ 
		CaloTower_inside3x3matrix=true;
		break;
	      }
	    }
	  }
	  if (!CaloTower_inside3x3matrix) continue;	  
	  myleadTrackHCAL3x3hitsEtSum+=(**iCaloTower).hadEt();
	  if((**iCaloTower).hadEt()>=myleadTrackHCAL3x3hottesthitEt ){
	    if ((**iCaloTower).hadEt()!=myleadTrackHCAL3x3hottesthitEt || 
		((**iCaloTower).hadEt()==myleadTrackHCAL3x3hottesthitEt && fabs((**iCaloTower).eta()-mypropagleadTrackECALSurfContactPoint.Eta())<myleadTrackHCAL3x3hottesthitDEta)) myleadTrackHCAL3x3hottesthitDEta = fabs((**iCaloTower).eta()-mypropagleadTrackECALSurfContactPoint.Eta());
	    myleadTrackHCAL3x3hottesthitEt=(**iCaloTower).hadEt();
	  }	
	}	
	myCaloTau.setleadTrackHCAL3x3hitsEtSum(myleadTrackHCAL3x3hitsEtSum);
	if (myleadTrackHCAL3x3hottesthitEt!=0.) myCaloTau.setleadTrackHCAL3x3hottesthitDEta(myleadTrackHCAL3x3hottesthitDEta);
      }
    }
    
    if (UseTrackLeadTrackDZconstraint_){
      TrackRefVector myTksbis;
      for (TrackRefVector::const_iterator iTrack=myTks.begin();iTrack!=myTks.end();++iTrack) {
	if (fabs((**iTrack).dz(myPV.position())-myleadTkDZ)<=TrackLeadTrack_maxDZ_) myTksbis.push_back(*iTrack);
      }
      myTks=myTksbis;
    }


    double myTrackerSignalConeSize=myCaloTauElementsOperators.computeConeSize(myTrackerSignalConeSizeTFormula,TrackerSignalConeSize_min_,TrackerSignalConeSize_max_);
    double myTrackerIsolConeSize=myCaloTauElementsOperators.computeConeSize(myTrackerIsolConeSizeTFormula,TrackerIsolConeSize_min_,TrackerIsolConeSize_max_);     	
    double myECALSignalConeSize=myCaloTauElementsOperators.computeConeSize(myECALSignalConeSizeTFormula,ECALSignalConeSize_min_,ECALSignalConeSize_max_);
    double myECALIsolConeSize=myCaloTauElementsOperators.computeConeSize(myECALIsolConeSizeTFormula,ECALIsolConeSize_min_,ECALIsolConeSize_max_);     	

    TrackRefVector mySignalTks;
    if (UseTrackLeadTrackDZconstraint_) mySignalTks=myCaloTauElementsOperators.tracksInCone((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,Track_minPt_,TrackLeadTrack_maxDZ_,myleadTkDZ, myPV);
    else mySignalTks=myCaloTauElementsOperators.tracksInCone((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,Track_minPt_);
    myCaloTau.setsignalTracks(mySignalTks);
    
    // setting invariant mass of the signal Tracks system
    math::XYZTLorentzVector mySignalTksInvariantMass(0.,0.,0.,0.);
    if((int)(mySignalTks.size())!=0){
      int mySignalTks_qsum=0;       
      for(int i=0;i<(int)mySignalTks.size();i++){
	mySignalTks_qsum+=mySignalTks[i]->charge();
	math::XYZTLorentzVector mychargedpicand_fromTk_LorentzVect(mySignalTks[i]->momentum().x(),mySignalTks[i]->momentum().y(),mySignalTks[i]->momentum().z(),sqrt(std::pow((double)mySignalTks[i]->momentum().r(),2)+std::pow(chargedpi_mass_,2)));
	mySignalTksInvariantMass+=mychargedpicand_fromTk_LorentzVect;
      }
      myCaloTau.setCharge(mySignalTks_qsum);    
    }
    myCaloTau.setsignalTracksInvariantMass(mySignalTksInvariantMass.mass());
    
    TrackRefVector myIsolTks;
    if (UseTrackLeadTrackDZconstraint_) myIsolTks=myCaloTauElementsOperators.tracksInAnnulus((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,IsolationTrack_minPt_,TrackLeadTrack_maxDZ_,myleadTkDZ, myPV);
    else myIsolTks=myCaloTauElementsOperators.tracksInAnnulus((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,IsolationTrack_minPt_);
    myIsolTks = TauTagTools::filteredTracksByNumTrkHits(myIsolTks,IsolationTrack_minHits_);
    myCaloTau.setisolationTracks(myIsolTks);
    
    // setting sum of Pt of the isolation annulus Tracks
    float myIsolTks_Ptsum=0.;
    for(int i=0;i<(int)myIsolTks.size();i++) myIsolTks_Ptsum+=myIsolTks[i]->pt();
    myCaloTau.setisolationTracksPtSum(myIsolTks_Ptsum);


    //getting the EcalRecHits. Just take them all
  std::vector<std::pair<math::XYZPoint,float> > thePositionAndEnergyEcalRecHits;
  mySelectedDetId_.clear();
  //  std::vector<CaloTowerPtr> theCaloTowers=myCaloJet->getCaloConstituents();
  edm::ESHandle<CaloGeometry> theCaloGeometry;
  iSetup.get<CaloGeometryRecord>().get(theCaloGeometry);
  const CaloSubdetectorGeometry* theCaloSubdetectorGeometry;  
  edm::Handle<EBRecHitCollection> EBRecHits;
  edm::Handle<EERecHitCollection> EERecHits; 
  edm::Handle<ESRecHitCollection> ESRecHits; 
  iEvent.getByLabel(EBRecHitsLabel_,EBRecHits);
  iEvent.getByLabel(EERecHitsLabel_,EERecHits);
  iEvent.getByLabel(ESRecHitsLabel_,ESRecHits);
  double maxDeltaR = 0.8;
    math::XYZPoint myCaloJetdir((*myCaloJet).px(),(*myCaloJet).py(),(*myCaloJet).pz());
    
  for(EBRecHitCollection::const_iterator theRecHit = EBRecHits->begin();theRecHit != EBRecHits->end(); theRecHit++){
    theCaloSubdetectorGeometry = theCaloGeometry->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
    const CaloCellGeometry* theRecHitCell=theCaloSubdetectorGeometry->getGeometry(theRecHit->id());  
    math::XYZPoint theRecHitCell_XYZPoint(theRecHitCell->getPosition().x(),theRecHitCell->getPosition().y(),theRecHitCell->getPosition().z());
    if(ROOT::Math::VectorUtil::DeltaR(myCaloJetdir,theRecHitCell_XYZPoint) < maxDeltaR){
      std::pair<math::XYZPoint,float> thePositionAndEnergyEcalRecHit(theRecHitCell_XYZPoint,theRecHit->energy());
      thePositionAndEnergyEcalRecHits.push_back(thePositionAndEnergyEcalRecHit);
      mySelectedDetId_.push_back(theRecHit->id());
    }
  }

for(EERecHitCollection::const_iterator theRecHit = EERecHits->begin();theRecHit != EERecHits->end(); theRecHit++){
    theCaloSubdetectorGeometry = theCaloGeometry->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);
    const CaloCellGeometry* theRecHitCell=theCaloSubdetectorGeometry->getGeometry(theRecHit->id());  
    math::XYZPoint theRecHitCell_XYZPoint(theRecHitCell->getPosition().x(),theRecHitCell->getPosition().y(),theRecHitCell->getPosition().z());
    if(ROOT::Math::VectorUtil::DeltaR(myCaloJetdir,theRecHitCell_XYZPoint) < maxDeltaR){
      std::pair<math::XYZPoint,float> thePositionAndEnergyEcalRecHit(theRecHitCell_XYZPoint,theRecHit->energy());
      thePositionAndEnergyEcalRecHits.push_back(thePositionAndEnergyEcalRecHit);
      mySelectedDetId_.push_back(theRecHit->id());
    }
}
 for(ESRecHitCollection::const_iterator theRecHit = ESRecHits->begin();theRecHit != ESRecHits->end(); theRecHit++){
  theCaloSubdetectorGeometry = theCaloGeometry->getSubdetectorGeometry(DetId::Ecal,EcalPreshower);
    const CaloCellGeometry* theRecHitCell=theCaloSubdetectorGeometry->getGeometry(theRecHit->id());  
    math::XYZPoint theRecHitCell_XYZPoint(theRecHitCell->getPosition().x(),theRecHitCell->getPosition().y(),theRecHitCell->getPosition().z());
    if(ROOT::Math::VectorUtil::DeltaR(myCaloJetdir,theRecHitCell_XYZPoint) < maxDeltaR){
      std::pair<math::XYZPoint,float> thePositionAndEnergyEcalRecHit(theRecHitCell_XYZPoint,theRecHit->energy());
      thePositionAndEnergyEcalRecHits.push_back(thePositionAndEnergyEcalRecHit);
      mySelectedDetId_.push_back(theRecHit->id());
    }
 }

  /*
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
	  std::pair<math::XYZPoint,float> thePositionAndEnergyEcalRecHit(theRecHitCell_XYZPoint,theRecHit->energy());
	  thePositionAndEnergyEcalRecHits.push_back(thePositionAndEnergyEcalRecHit);
	}else if((EcalSubdetector)RecHitDetID.subdetId()==EcalEndcap){
	  theCaloSubdetectorGeometry = theCaloGeometry->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);
	  EEDetId EcalID = RecHitDetID;
	  EERecHitCollection::const_iterator theRecHit=EERecHits->find(EcalID);	    
	  const CaloCellGeometry* theRecHitCell=theCaloSubdetectorGeometry->getGeometry(RecHitDetID);
	  math::XYZPoint theRecHitCell_XYZPoint(theRecHitCell->getPosition().x(),theRecHitCell->getPosition().y(),theRecHitCell->getPosition().z());
	  std::pair<math::XYZPoint,float> thePositionAndEnergyEcalRecHit(theRecHitCell_XYZPoint,theRecHit->energy());
	  thePositionAndEnergyEcalRecHits.push_back(thePositionAndEnergyEcalRecHit);
	}else if((EcalSubdetector)RecHitDetID.subdetId()==EcalPreshower){
	  theCaloSubdetectorGeometry = theCaloGeometry->getSubdetectorGeometry(DetId::Ecal,EcalPreshower);
	  ESDetId EcalID = RecHitDetID;
	  ESRecHitCollection::const_iterator theRecHit=ESRecHits->find(EcalID);	    
	  const CaloCellGeometry* theRecHitCell=theCaloSubdetectorGeometry->getGeometry(RecHitDetID);
	
	  math::XYZPoint theRecHitCell_XYZPoint(theRecHitCell->getPosition().x(),theRecHitCell->getPosition().y(),theRecHitCell->getPosition().z());
	  std::pair<math::XYZPoint,float> thePositionAndEnergyEcalRecHit(theRecHitCell_XYZPoint,theRecHit->energy());
	  thePositionAndEnergyEcalRecHits.push_back(thePositionAndEnergyEcalRecHit);
	}	 
      }	
    }
  }
  */
    
    // setting sum of Et of the isolation annulus ECAL RecHits
    float myIsolEcalRecHits_EtSum=0.;

    std::vector< std::pair<math::XYZPoint,float> > myIsolPositionAndEnergyEcalRecHits=myCaloTauElementsOperators.EcalRecHitsInAnnulus((*myleadTk).momentum(),ECALSignalConeMetric_,myECALSignalConeSize,ECALIsolConeMetric_,myECALIsolConeSize,ECALRecHit_minEt_,thePositionAndEnergyEcalRecHits);
    for(std::vector< std::pair<math::XYZPoint,float> >::const_iterator iEcalRecHit=myIsolPositionAndEnergyEcalRecHits.begin();iEcalRecHit!=myIsolPositionAndEnergyEcalRecHits.end();iEcalRecHit++){
      myIsolEcalRecHits_EtSum+=(*iEcalRecHit).second*fabs(sin((*iEcalRecHit).first.theta()));
    }
    myCaloTau.setisolationECALhitsEtSum(myIsolEcalRecHits_EtSum);    
  }

  math::XYZTLorentzVector myTks_XYZTLorentzVect(0.,0.,0.,0.);
  math::XYZTLorentzVector alternatLorentzVect(0.,0.,0.,0.);
  for(TrackRefVector::iterator iTrack=myTks.begin();iTrack!=myTks.end();iTrack++) {
    // build a charged pion candidate Lorentz vector from a Track
    math::XYZTLorentzVector iChargedPionCand_XYZTLorentzVect((**iTrack).momentum().x(),(**iTrack).momentum().y(),(**iTrack).momentum().z(),sqrt(std::pow((double)(**iTrack).momentum().r(),2)+std::pow(chargedpi_mass_,2)));
    myTks_XYZTLorentzVect+=iChargedPionCand_XYZTLorentzVect;
    alternatLorentzVect+=iChargedPionCand_XYZTLorentzVect;
  }
  myCaloTau.setTracksInvariantMass(myTks_XYZTLorentzVect.mass());

  std::vector<BasicClusterRef> myneutralECALBasicClusters=(*myCaloTauTagInfoRef).neutralECALBasicClusters();
  for(std::vector<BasicClusterRef>::const_iterator iBasicCluster=myneutralECALBasicClusters.begin();iBasicCluster!=myneutralECALBasicClusters.end();iBasicCluster++) {
    // build a gamma candidate Lorentz vector from a neutral ECAL BasicCluster
    double iGammaCand_px=(**iBasicCluster).energy()*sin((**iBasicCluster).position().theta())*cos((**iBasicCluster).position().phi());
    double iGammaCand_py=(**iBasicCluster).energy()*sin((**iBasicCluster).position().theta())*sin((**iBasicCluster).position().phi());
    double iGammaCand_pz=(**iBasicCluster).energy()*cos((**iBasicCluster).position().theta());
    math::XYZTLorentzVector iGammaCand_XYZTLorentzVect(iGammaCand_px,iGammaCand_py,iGammaCand_pz,(**iBasicCluster).energy());
    alternatLorentzVect+=iGammaCand_XYZTLorentzVect;
  }
  myCaloTau.setalternatLorentzVect(alternatLorentzVect);
  
  
  myCaloTau.setVertex(math::XYZPoint(myCaloTau_refInnerPosition_x,myCaloTau_refInnerPosition_y,myCaloTau_refInnerPosition_z));
    
  // setting Et of the highest Et HCAL CaloTower
  double mymaxEtHCALtower_Et=0.; 
  for(unsigned int iTower=0;iTower<myCaloTowers.size();iTower++){
    if((*myCaloTowers[iTower]).hadEt()>=mymaxEtHCALtower_Et) mymaxEtHCALtower_Et=(*myCaloTowers[iTower]).hadEt();
  }
  myCaloTau.setmaximumHCALhitEt(mymaxEtHCALtower_Et);

  return myCaloTau;  
}

std::vector<CaloTowerDetId> CaloRecoTauAlgorithm::getCaloTowerneighbourDetIds(const CaloSubdetectorGeometry* myCaloSubdetectorGeometry, const CaloTowerTopology & myCaloTowerTopology, CaloTowerDetId myCaloTowerDetId){
  std::vector<CaloTowerDetId> myCaloTowerneighbourDetIds;
  std::vector<DetId> northDetIds=myCaloTowerTopology.north(myCaloTowerDetId);
  std::vector<DetId> westDetIds=myCaloTowerTopology.west(myCaloTowerDetId);
  std::vector<DetId> northwestDetIds,southwestDetIds;
  if (westDetIds.size()>0){
    northwestDetIds=myCaloTowerTopology.north(westDetIds[0]);
    southwestDetIds=myCaloTowerTopology.south(westDetIds[(int)westDetIds.size()-1]);
  }
  std::vector<DetId> southDetIds=myCaloTowerTopology.south(myCaloTowerDetId);
  std::vector<DetId> eastDetIds=myCaloTowerTopology.east(myCaloTowerDetId);
  std::vector<DetId> northeastDetIds,southeastDetIds;
  if (eastDetIds.size()>0){
    northeastDetIds=myCaloTowerTopology.north(eastDetIds[0]);
    southeastDetIds=myCaloTowerTopology.south(eastDetIds[(int)eastDetIds.size()-1]);
  }
  std::vector<DetId> myneighbourDetIds=northDetIds;
  myneighbourDetIds.insert(myneighbourDetIds.end(),westDetIds.begin(),westDetIds.end());
  myneighbourDetIds.insert(myneighbourDetIds.end(),northwestDetIds.begin(),northwestDetIds.end());
  myneighbourDetIds.insert(myneighbourDetIds.end(),southwestDetIds.begin(),southwestDetIds.end());
  myneighbourDetIds.insert(myneighbourDetIds.end(),southDetIds.begin(),southDetIds.end());
  myneighbourDetIds.insert(myneighbourDetIds.end(),eastDetIds.begin(),eastDetIds.end());
  myneighbourDetIds.insert(myneighbourDetIds.end(),northeastDetIds.begin(),northeastDetIds.end());
  myneighbourDetIds.insert(myneighbourDetIds.end(),southeastDetIds.begin(),southeastDetIds.end());
  for(std::vector<DetId>::const_iterator iDetId=myneighbourDetIds.begin();iDetId!=myneighbourDetIds.end();iDetId++){
    CaloTowerDetId iCaloTowerId(*iDetId);
    myCaloTowerneighbourDetIds.push_back(iCaloTowerId);
  }
  return myCaloTowerneighbourDetIds;
}

