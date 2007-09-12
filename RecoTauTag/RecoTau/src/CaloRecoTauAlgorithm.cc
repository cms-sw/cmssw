#include "RecoTauTag/RecoTau/interface/CaloRecoTauAlgorithm.h"

CaloRecoTauAlgorithm::CaloRecoTauAlgorithm() : TransientTrackBuilder_(0),chargedpi_mass_(0.13957018){}  
CaloRecoTauAlgorithm::CaloRecoTauAlgorithm(const ParameterSet& iConfig) : TransientTrackBuilder_(0),chargedpi_mass_(0.13957018){
  LeadTrack_minPt_                    = iConfig.getParameter<double>("LeadTrack_minPt");
  Track_minPt_                        = iConfig.getParameter<double>("Track_minPt");
  UseTrackLeadTrackDZconstraint_      = iConfig.getParameter<bool>("UseTrackLeadTrackDZconstraint");
  TrackLeadTrack_maxDZ_               = iConfig.getParameter<double>("TrackLeadTrack_maxDZ");
  ECALRecHit_minEt_                   = iConfig.getParameter<double>("ECALRecHit_minEt");       
  
  MatchingConeMetric_                 = iConfig.getParameter<string>("MatchingConeMetric");
  MatchingConeSizeFormula_            = iConfig.getParameter<string>("MatchingConeSizeFormula");
  MatchingConeVariableSize_min_       = iConfig.getParameter<double>("MatchingConeVariableSize_min");
  MatchingConeVariableSize_max_       = iConfig.getParameter<double>("MatchingConeVariableSize_max");
  TrackerSignalConeMetric_            = iConfig.getParameter<string>("TrackerSignalConeMetric");
  TrackerSignalConeSizeFormula_       = iConfig.getParameter<string>("TrackerSignalConeSizeFormula");
  TrackerSignalConeVariableSize_min_  = iConfig.getParameter<double>("TrackerSignalConeVariableSize_min");
  TrackerSignalConeVariableSize_max_  = iConfig.getParameter<double>("TrackerSignalConeVariableSize_max");
  TrackerIsolConeMetric_              = iConfig.getParameter<string>("TrackerIsolConeMetric"); 
  TrackerIsolConeSizeFormula_         = iConfig.getParameter<string>("TrackerIsolConeSizeFormula"); 
  TrackerIsolConeVariableSize_min_    = iConfig.getParameter<double>("TrackerIsolConeVariableSize_min");
  TrackerIsolConeVariableSize_max_    = iConfig.getParameter<double>("TrackerIsolConeVariableSize_max");
  ECALSignalConeMetric_               = iConfig.getParameter<string>("ECALSignalConeMetric");
  ECALSignalConeSizeFormula_          = iConfig.getParameter<string>("ECALSignalConeSizeFormula");    
  ECALSignalConeVariableSize_min_     = iConfig.getParameter<double>("ECALSignalConeVariableSize_min");
  ECALSignalConeVariableSize_max_     = iConfig.getParameter<double>("ECALSignalConeVariableSize_max");
  ECALIsolConeMetric_                 = iConfig.getParameter<string>("ECALIsolConeMetric");
  ECALIsolConeSizeFormula_            = iConfig.getParameter<string>("ECALIsolConeSizeFormula");      
  ECALIsolConeVariableSize_min_       = iConfig.getParameter<double>("ECALIsolConeVariableSize_min");
  ECALIsolConeVariableSize_max_       = iConfig.getParameter<double>("ECALIsolConeVariableSize_max");
  
  AreaMetric_recoElements_maxabsEta_  = iConfig.getParameter<double>("AreaMetric_recoElements_maxabsEta");
}
void CaloRecoTauAlgorithm::setTransientTrackBuilder(const TransientTrackBuilder* x){TransientTrackBuilder_=x;}

CaloTau CaloRecoTauAlgorithm::buildCaloTau(Event& iEvent,const CaloTauTagInfoRef& myCaloTauTagInfoRef,const Vertex& myPV){
  CaloJetRef myCaloJet=(*myCaloTauTagInfoRef).calojetRef(); // catch a ref to the initial CaloJet  
  CaloTau myCaloTau(numeric_limits<int>::quiet_NaN(),myCaloJet->p4()); // create the CaloTau with the initial CaloJet Lorentz-vector
  
  myCaloTau.setcaloTauTagInfoRef(myCaloTauTagInfoRef);

  myCaloTau.setalternatLorentzVect((*myCaloTauTagInfoRef).alternatLorentzVect());

  TrackRefVector myTks=(*myCaloTauTagInfoRef).Tracks();
  // setting invariant mass of the Tracks system
  math::XYZTLorentzVector myTks_XYZTLorentzVect(0.,0.,0.,0.);
  if((int)(myTks.size())!=0){
    for(int i=0;i<(int)myTks.size();i++){
      math::XYZTLorentzVector mychargedpicand_fromTk_LorentzVect(myTks[i]->momentum().x(),myTks[i]->momentum().y(),myTks[i]->momentum().z(),sqrt(pow((double)myTks[i]->momentum().r(),2)+pow(chargedpi_mass_,2)));
      myTks_XYZTLorentzVect+=mychargedpicand_fromTk_LorentzVect;
    }
  }
  myCaloTau.setTracksInvariantMass(myTks_XYZTLorentzVect.mass());

  CaloTauElementsOperators myCaloTauElementsOperators(myCaloTau);
  TFormula myMatchingConeSizeTFormula=myCaloTauElementsOperators.computeConeSizeTFormula(MatchingConeSizeFormula_,"Matching cone size");
  double myMatchingConeSize=myCaloTauElementsOperators.computeConeSize(myMatchingConeSizeTFormula,MatchingConeVariableSize_min_,MatchingConeVariableSize_max_);
  TrackRef myleadTk=myCaloTauElementsOperators.leadTk(MatchingConeMetric_,myMatchingConeSize,LeadTrack_minPt_);
  double myCaloTau_refInnerPosition_x=0.;
  double myCaloTau_refInnerPosition_y=0.;
  double myCaloTau_refInnerPosition_z=0.;
  if(myleadTk.isNonnull()){
    myCaloTau.setleadTrack(myleadTk);
    double myleadTkDZ=(*myleadTk).dz();
    if(TransientTrackBuilder_!=0){ 
      const TransientTrack myleadTransientTk=TransientTrackBuilder_->build(&(*myleadTk));
      GlobalVector myCaloJetdir((*myCaloJet).px(),(*myCaloJet).py(),(*myCaloJet).pz());
      myCaloTau.setleadTracksignedSipt(IPTools::signedTransverseImpactParameter(myleadTransientTk,myCaloJetdir,myPV).second.significance());
    }
    if((*myleadTk).innerOk()){
      myCaloTau_refInnerPosition_x=(*myleadTk).innerPosition().x(); 
      myCaloTau_refInnerPosition_y=(*myleadTk).innerPosition().y(); 
      myCaloTau_refInnerPosition_z=(*myleadTk).innerPosition().z(); 
    }
    TFormula myTrackerSignalConeSizeTFormula=myCaloTauElementsOperators.computeConeSizeTFormula(TrackerSignalConeSizeFormula_,"Tracker signal cone size");
    double myTrackerSignalConeSize=myCaloTauElementsOperators.computeConeSize(myTrackerSignalConeSizeTFormula,TrackerSignalConeVariableSize_min_,TrackerSignalConeVariableSize_max_);
    TFormula myTrackerIsolConeSizeTFormula=myCaloTauElementsOperators.computeConeSizeTFormula(TrackerIsolConeSizeFormula_,"Tracker isolation cone size");
    double myTrackerIsolConeSize=myCaloTauElementsOperators.computeConeSize(myTrackerIsolConeSizeTFormula,TrackerIsolConeVariableSize_min_,TrackerIsolConeVariableSize_max_);     	
    TFormula myECALSignalConeSizeTFormula=myCaloTauElementsOperators.computeConeSizeTFormula(ECALSignalConeSizeFormula_,"ECAL signal cone size");
    double myECALSignalConeSize=myCaloTauElementsOperators.computeConeSize(myECALSignalConeSizeTFormula,ECALSignalConeVariableSize_min_,ECALSignalConeVariableSize_max_);
    TFormula myECALIsolConeSizeTFormula=myCaloTauElementsOperators.computeConeSizeTFormula(ECALIsolConeSizeFormula_,"ECAL isolation cone size");
    double myECALIsolConeSize=myCaloTauElementsOperators.computeConeSize(myECALIsolConeSizeTFormula,ECALIsolConeVariableSize_min_,ECALIsolConeVariableSize_max_);     	
    
    TrackRefVector mySignalTks;
    if (UseTrackLeadTrackDZconstraint_) mySignalTks=myCaloTauElementsOperators.tracksInCone((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,Track_minPt_,TrackLeadTrack_maxDZ_,myleadTkDZ);
    else mySignalTks=myCaloTauElementsOperators.tracksInCone((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,Track_minPt_);
    myCaloTau.setsignalTracks(mySignalTks);
    
    // setting invariant mass of the signal Tracks system
    math::XYZTLorentzVector mySignalTksInvariantMass(0.,0.,0.,0.);
    if((int)(mySignalTks.size())!=0){
      int mySignalTks_qsum=0;       
      for(int i=0;i<(int)mySignalTks.size();i++){
	mySignalTks_qsum+=mySignalTks[i]->charge();
	math::XYZTLorentzVector mychargedpicand_fromTk_LorentzVect(mySignalTks[i]->momentum().x(),mySignalTks[i]->momentum().y(),mySignalTks[i]->momentum().z(),sqrt(pow((double)mySignalTks[i]->momentum().r(),2)+pow(chargedpi_mass_,2)));
	mySignalTksInvariantMass+=mychargedpicand_fromTk_LorentzVect;
      }
      myCaloTau.setCharge(mySignalTks_qsum);    
    }
    myCaloTau.setsignalTracksInvariantMass(mySignalTksInvariantMass.mass());
    
    TrackRefVector myIsolTks;
    if (UseTrackLeadTrackDZconstraint_) myIsolTks=myCaloTauElementsOperators.tracksInAnnulus((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,Track_minPt_,TrackLeadTrack_maxDZ_,myleadTkDZ);
    else myIsolTks=myCaloTauElementsOperators.tracksInAnnulus((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,Track_minPt_);
    myCaloTau.setisolationTracks(myIsolTks);
    
    // setting sum of Pt of the isolation annulus Tracks
    float myIsolTks_Ptsum=0.;
    for(int i=0;i<(int)myIsolTks.size();i++) myIsolTks_Ptsum+=myIsolTks[i]->pt();
    myCaloTau.setisolationTracksPtSum(myIsolTks_Ptsum);
    
    // setting sum of Et of the isolation annulus ECAL RecHits
    float myIsolEcalRecHits_EtSum=0.;
    vector<pair<math::XYZPoint,float> > myIsolPositionAndEnergyEcalRecHits=myCaloTauElementsOperators.EcalRecHitsInAnnulus((*myleadTk).momentum(),ECALSignalConeMetric_,myECALSignalConeSize,ECALIsolConeMetric_,myECALIsolConeSize,ECALRecHit_minEt_);
    for(vector<pair<math::XYZPoint,float> >::const_iterator iEcalRecHit=myIsolPositionAndEnergyEcalRecHits.begin();iEcalRecHit!=myIsolPositionAndEnergyEcalRecHits.end();iEcalRecHit++){
      myIsolEcalRecHits_EtSum+=(*iEcalRecHit).second*fabs(sin((*iEcalRecHit).first.theta()));
    }
    myCaloTau.setisolationECALhitsEtSum(myIsolEcalRecHits_EtSum);    
  }
  myCaloTau.setVertex(math::XYZPoint(myCaloTau_refInnerPosition_x,myCaloTau_refInnerPosition_y,myCaloTau_refInnerPosition_z));
    
  // setting Et of the highest Et HCAL CaloTower
  const vector<CaloTowerRef> myCaloTowers=(*myCaloJet).getConstituents();
  double mymaxEtHCALtower_Et=0.; 
  for(unsigned int iTower=0;iTower<myCaloTowers.size();iTower++){
    if((*myCaloTowers[iTower]).hadEt()>=mymaxEtHCALtower_Et) mymaxEtHCALtower_Et=(*myCaloTowers[iTower]).hadEt();
  }
  myCaloTau.setmaximumHCALhitEt(mymaxEtHCALtower_Et);

  return myCaloTau;  
}
