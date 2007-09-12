#include "RecoTauTag/RecoTau/interface/PFRecoTauAlgorithm.h"

PFRecoTauAlgorithm::PFRecoTauAlgorithm() : TransientTrackBuilder_(0){}  
PFRecoTauAlgorithm::PFRecoTauAlgorithm(const ParameterSet& iConfig) : TransientTrackBuilder_(0){
  LeadChargedHadrCand_minPt_             = iConfig.getParameter<double>("LeadChargedHadrCand_minPt"); 
  ChargedHadrCand_minPt_                 = iConfig.getParameter<double>("ChargedHadrCand_minPt");
  UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint_ = iConfig.getParameter<bool>("UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint");
  ChargedHadrCandLeadChargedHadrCand_tksmaxDZ_ = iConfig.getParameter<double>("ChargedHadrCandLeadChargedHadrCand_tksmaxDZ");
  NeutrHadrCand_minPt_                   = iConfig.getParameter<double>("NeutrHadrCand_minPt");
  GammaCand_minPt_                       = iConfig.getParameter<double>("GammaCand_minPt");       
  LeadTrack_minPt_                       = iConfig.getParameter<double>("LeadTrack_minPt");
  Track_minPt_                           = iConfig.getParameter<double>("Track_minPt");
  UseTrackLeadTrackDZconstraint_         = iConfig.getParameter<bool>("UseTrackLeadTrackDZconstraint");
  TrackLeadTrack_maxDZ_                  = iConfig.getParameter<double>("TrackLeadTrack_maxDZ");
  
  MatchingConeMetric_                    = iConfig.getParameter<string>("MatchingConeMetric");
  MatchingConeSizeFormula_               = iConfig.getParameter<string>("MatchingConeSizeFormula");
  MatchingConeVariableSize_min_          = iConfig.getParameter<double>("MatchingConeVariableSize_min");
  MatchingConeVariableSize_max_          = iConfig.getParameter<double>("MatchingConeVariableSize_max");
  TrackerSignalConeMetric_               = iConfig.getParameter<string>("TrackerSignalConeMetric");
  TrackerSignalConeSizeFormula_          = iConfig.getParameter<string>("TrackerSignalConeSizeFormula");
  TrackerSignalConeVariableSize_min_     = iConfig.getParameter<double>("TrackerSignalConeVariableSize_min");
  TrackerSignalConeVariableSize_max_     = iConfig.getParameter<double>("TrackerSignalConeVariableSize_max");
  TrackerIsolConeMetric_                 = iConfig.getParameter<string>("TrackerIsolConeMetric"); 
  TrackerIsolConeSizeFormula_            = iConfig.getParameter<string>("TrackerIsolConeSizeFormula"); 
  TrackerIsolConeVariableSize_min_       = iConfig.getParameter<double>("TrackerIsolConeVariableSize_min");
  TrackerIsolConeVariableSize_max_       = iConfig.getParameter<double>("TrackerIsolConeVariableSize_max");
  ECALSignalConeMetric_               = iConfig.getParameter<string>("ECALSignalConeMetric");
  ECALSignalConeSizeFormula_          = iConfig.getParameter<string>("ECALSignalConeSizeFormula");    
  ECALSignalConeVariableSize_min_     = iConfig.getParameter<double>("ECALSignalConeVariableSize_min");
  ECALSignalConeVariableSize_max_     = iConfig.getParameter<double>("ECALSignalConeVariableSize_max");
  ECALIsolConeMetric_                 = iConfig.getParameter<string>("ECALIsolConeMetric");
  ECALIsolConeSizeFormula_            = iConfig.getParameter<string>("ECALIsolConeSizeFormula");      
  ECALIsolConeVariableSize_min_       = iConfig.getParameter<double>("ECALIsolConeVariableSize_min");
  ECALIsolConeVariableSize_max_       = iConfig.getParameter<double>("ECALIsolConeVariableSize_max");
  HCALSignalConeMetric_               = iConfig.getParameter<string>("HCALSignalConeMetric");
  HCALSignalConeSizeFormula_          = iConfig.getParameter<string>("HCALSignalConeSizeFormula");    
  HCALSignalConeVariableSize_min_     = iConfig.getParameter<double>("HCALSignalConeVariableSize_min");
  HCALSignalConeVariableSize_max_     = iConfig.getParameter<double>("HCALSignalConeVariableSize_max");
  HCALIsolConeMetric_                 = iConfig.getParameter<string>("HCALIsolConeMetric");
  HCALIsolConeSizeFormula_            = iConfig.getParameter<string>("HCALIsolConeSizeFormula");      
  HCALIsolConeVariableSize_min_       = iConfig.getParameter<double>("HCALIsolConeVariableSize_min");
  HCALIsolConeVariableSize_max_       = iConfig.getParameter<double>("HCALIsolConeVariableSize_max");
  
  AreaMetric_recoElements_maxabsEta_  = iConfig.getParameter<double>("AreaMetric_recoElements_maxabsEta");
}
void PFRecoTauAlgorithm::setTransientTrackBuilder(const TransientTrackBuilder* x){TransientTrackBuilder_=x;}

PFTau PFRecoTauAlgorithm::buildPFTau(const PFTauTagInfoRef& myPFTauTagInfoRef,const Vertex& myPV){
  PFJetRef myPFJet=(*myPFTauTagInfoRef).pfjetRef();  // catch a ref to the initial PFJet  
  PFTau myPFTau(numeric_limits<int>::quiet_NaN(),myPFJet->p4());   // create the PFTau
   
  myPFTau.setpfTauTagInfoRef(myPFTauTagInfoRef);
  
  myPFTau.setalternatLorentzVect((*myPFTauTagInfoRef).alternatLorentzVect());

  PFCandidateRefVector myPFCands=(*myPFTauTagInfoRef).PFCands();
  PFCandidateRefVector myPFChargedHadrCands=(*myPFTauTagInfoRef).PFChargedHadrCands();
  PFCandidateRefVector myPFNeutrHadrCands=(*myPFTauTagInfoRef).PFNeutrHadrCands();
  PFCandidateRefVector myPFGammaCands=(*myPFTauTagInfoRef).PFGammaCands();
  
  PFTauElementsOperators myPFTauElementsOperators(myPFTau);
  TFormula myMatchingConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(MatchingConeSizeFormula_,"Matching cone size");
  double myMatchingConeSize=myPFTauElementsOperators.computeConeSize(myMatchingConeSizeTFormula,MatchingConeVariableSize_min_,MatchingConeVariableSize_max_);
  PFCandidateRef myleadPFCand=myPFTauElementsOperators.leadPFCand(MatchingConeMetric_,myMatchingConeSize,LeadChargedHadrCand_minPt_);
  bool myleadPFCand_rectkavailable=false;
  double myleadPFCand_rectkDZ=0.;
  double myPFTau_refInnerPosition_x=0.;
  double myPFTau_refInnerPosition_y=0.;
  double myPFTau_refInnerPosition_z=0.;
  if(myleadPFCand.isNonnull()){
    myPFTau.setleadPFChargedHadrCand(myleadPFCand);
    if ((*myleadPFCand).blockRef()->elements().size()!=0){
      for (OwnVector<PFBlockElement>::const_iterator iPFBlockElement=(*myleadPFCand).blockRef()->elements().begin();iPFBlockElement!=(*myleadPFCand).blockRef()->elements().end();iPFBlockElement++){
	if ((*iPFBlockElement).type()==PFBlockElement::TRACK && ROOT::Math::VectorUtil::DeltaR((*myleadPFCand).momentum(),(*iPFBlockElement).trackRef()->momentum())<0.001){
	  TrackRef myleadPFCand_rectk=(*iPFBlockElement).trackRef();
	  if(myleadPFCand_rectk.isNonnull()){
	    myleadPFCand_rectkavailable=true;
	    myleadPFCand_rectkDZ=(*myleadPFCand_rectk).dz();
	    if(TransientTrackBuilder_!=0){ 
	      const TransientTrack myleadPFCand_rectransienttk=TransientTrackBuilder_->build(&(*myleadPFCand_rectk));
	      GlobalVector myPFJetdir((*myPFJet).px(),(*myPFJet).py(),(*myPFJet).pz());
	      myPFTau.setleadPFChargedHadrCandsignedSipt(IPTools::signedTransverseImpactParameter(myleadPFCand_rectransienttk,myPFJetdir,myPV).second.significance());
	    }
	    if((*myleadPFCand_rectk).innerOk()){
	      myPFTau_refInnerPosition_x=(*myleadPFCand_rectk).innerPosition().x(); 
	      myPFTau_refInnerPosition_y=(*myleadPFCand_rectk).innerPosition().y(); 
	      myPFTau_refInnerPosition_z=(*myleadPFCand_rectk).innerPosition().z(); 
	    }
	  }
	}
      }
    }
    TFormula myTrackerSignalConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(TrackerSignalConeSizeFormula_,"Tracker signal cone size");
    double myTrackerSignalConeSize=myPFTauElementsOperators.computeConeSize(myTrackerSignalConeSizeTFormula,TrackerSignalConeVariableSize_min_,TrackerSignalConeVariableSize_max_);
    TFormula myTrackerIsolConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(TrackerIsolConeSizeFormula_,"Tracker isolation cone size");
    double myTrackerIsolConeSize=myPFTauElementsOperators.computeConeSize(myTrackerIsolConeSizeTFormula,TrackerIsolConeVariableSize_min_,TrackerIsolConeVariableSize_max_);     	
    TFormula myECALSignalConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(ECALSignalConeSizeFormula_,"ECAL signal cone size");
    double myECALSignalConeSize=myPFTauElementsOperators.computeConeSize(myECALSignalConeSizeTFormula,ECALSignalConeVariableSize_min_,ECALSignalConeVariableSize_max_);
    TFormula myECALIsolConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(ECALIsolConeSizeFormula_,"ECAL isolation cone size");
    double myECALIsolConeSize=myPFTauElementsOperators.computeConeSize(myECALIsolConeSizeTFormula,ECALIsolConeVariableSize_min_,ECALIsolConeVariableSize_max_);     	
    TFormula myHCALSignalConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(HCALSignalConeSizeFormula_,"HCAL signal cone size");
    double myHCALSignalConeSize=myPFTauElementsOperators.computeConeSize(myHCALSignalConeSizeTFormula,HCALSignalConeVariableSize_min_,HCALSignalConeVariableSize_max_);
    TFormula myHCALIsolConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(HCALIsolConeSizeFormula_,"HCAL isolation cone size");
    double myHCALIsolConeSize=myPFTauElementsOperators.computeConeSize(myHCALIsolConeSizeTFormula,HCALIsolConeVariableSize_min_,HCALIsolConeVariableSize_max_);     	
    
    PFCandidateRefVector mySignalPFChargedHadrCands,mySignalPFNeutrHadrCands,mySignalPFGammaCands,mySignalPFCands;
    if (UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint_ && myleadPFCand_rectkavailable) mySignalPFChargedHadrCands=myPFTauElementsOperators.PFChargedHadrCandsInCone((*myleadPFCand).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,ChargedHadrCand_minPt_,ChargedHadrCandLeadChargedHadrCand_tksmaxDZ_,myleadPFCand_rectkDZ);
    else mySignalPFChargedHadrCands=myPFTauElementsOperators.PFChargedHadrCandsInCone((*myleadPFCand).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,ChargedHadrCand_minPt_);
    myPFTau.setsignalPFChargedHadrCands(mySignalPFChargedHadrCands);
    mySignalPFNeutrHadrCands=myPFTauElementsOperators.PFNeutrHadrCandsInCone((*myleadPFCand).momentum(),HCALSignalConeMetric_,myHCALSignalConeSize,NeutrHadrCand_minPt_);
    myPFTau.setsignalPFNeutrHadrCands(mySignalPFNeutrHadrCands);
    mySignalPFGammaCands=myPFTauElementsOperators.PFGammaCandsInCone((*myleadPFCand).momentum(),ECALSignalConeMetric_,myECALSignalConeSize,GammaCand_minPt_);
    myPFTau.setsignalPFGammaCands(mySignalPFGammaCands);
    
    if((int)(mySignalPFChargedHadrCands.size())!=0){
      int mySignalPFChargedHadrCands_qsum=0;       
      for(int i=0;i<(int)mySignalPFChargedHadrCands.size();i++){
	mySignalPFChargedHadrCands_qsum+=mySignalPFChargedHadrCands[i]->charge();
	mySignalPFCands.push_back(mySignalPFChargedHadrCands[i]);
      }
      myPFTau.setCharge(mySignalPFChargedHadrCands_qsum);    
    }
    for(int i=0;i<(int)mySignalPFNeutrHadrCands.size();i++) mySignalPFCands.push_back(mySignalPFNeutrHadrCands[i]);
    for(int i=0;i<(int)mySignalPFGammaCands.size();i++) mySignalPFCands.push_back(mySignalPFGammaCands[i]);
    myPFTau.setsignalPFCands(mySignalPFCands);
    
    PFCandidateRefVector myIsolPFChargedHadrCands,myIsolPFNeutrHadrCands,myIsolPFGammaCands,myIsolPFCands;
    if (UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint_ && myleadPFCand_rectkavailable) myIsolPFChargedHadrCands=myPFTauElementsOperators.PFChargedHadrCandsInAnnulus((*myleadPFCand).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,ChargedHadrCand_minPt_,ChargedHadrCandLeadChargedHadrCand_tksmaxDZ_,myleadPFCand_rectkDZ);
    else myIsolPFChargedHadrCands=myPFTauElementsOperators.PFChargedHadrCandsInAnnulus((*myleadPFCand).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,ChargedHadrCand_minPt_);
    myPFTau.setisolationPFChargedHadrCands(myIsolPFChargedHadrCands);
    myIsolPFNeutrHadrCands=myPFTauElementsOperators.PFNeutrHadrCandsInAnnulus((*myleadPFCand).momentum(),HCALSignalConeMetric_,myHCALSignalConeSize,HCALIsolConeMetric_,myHCALIsolConeSize,NeutrHadrCand_minPt_);
    myPFTau.setisolationPFNeutrHadrCands(myIsolPFNeutrHadrCands);
    myIsolPFGammaCands=myPFTauElementsOperators.PFGammaCandsInAnnulus((*myleadPFCand).momentum(),ECALSignalConeMetric_,myECALSignalConeSize,ECALIsolConeMetric_,myECALIsolConeSize,GammaCand_minPt_);  
    myPFTau.setisolationPFGammaCands(myIsolPFGammaCands);

    float myIsolPFChargedHadrCands_Ptsum=0.;
    float myIsolPFGammaCands_Etsum=0.;
    for(int i=0;i<(int)myIsolPFChargedHadrCands.size();i++){
      myIsolPFChargedHadrCands_Ptsum+=myIsolPFChargedHadrCands[i]->pt();
      myIsolPFCands.push_back(myIsolPFChargedHadrCands[i]);
    }
    myPFTau.setisolationPFChargedHadrCandsPtSum(myIsolPFChargedHadrCands_Ptsum);
    for(int i=0;i<(int)myIsolPFNeutrHadrCands.size();i++)myIsolPFCands.push_back(myIsolPFNeutrHadrCands[i]);
    for(int i=0;i<(int)myIsolPFGammaCands.size();i++){
      myIsolPFGammaCands_Etsum+=myIsolPFGammaCands[i]->et();
      myIsolPFCands.push_back(myIsolPFGammaCands[i]);
    } 
    myPFTau.setisolationPFGammaCandsEtSum(myIsolPFGammaCands_Etsum);
    myPFTau.setisolationPFCands(myIsolPFCands);
     
    float mymaximumHCALPFClusterEt=0.;
    for(int i=0;i<(int)myPFCands.size();i++){ 
      if (myPFCands[i]->blockRef()->elements().size()!=0){
	for (OwnVector<PFBlockElement>::const_iterator iPFBlockElement=myPFCands[i]->blockRef()->elements().begin();iPFBlockElement!=myPFCands[i]->blockRef()->elements().end();iPFBlockElement++){
	  if ((*iPFBlockElement).type()==PFBlockElement::HCAL && (*iPFBlockElement).clusterRef()->energy()*fabs(sin((*iPFBlockElement).clusterRef()->positionXYZ().Theta()))>mymaximumHCALPFClusterEt) mymaximumHCALPFClusterEt=(*iPFBlockElement).clusterRef()->energy()*fabs(sin((*iPFBlockElement).clusterRef()->positionXYZ().Theta()));
	}
      }
    }    
    myPFTau.setmaximumHCALPFClusterEt(mymaximumHCALPFClusterEt);    
  }
  myPFTau.setVertex(math::XYZPoint(myPFTau_refInnerPosition_x,myPFTau_refInnerPosition_y,myPFTau_refInnerPosition_z));
  
  // set the leading, signal cone and isolation annulus Tracks (the initial list of Tracks was catched through a JetTracksAssociation object, not through the charged hadr. PFCandidates inside the PFJet ; the motivation for considering these objects is the need for checking that a selection by the charged hadr. PFCandidates is equivalent to a selection by the rec. Tracks.)
  TrackRef myleadTk=myPFTauElementsOperators.leadTk(MatchingConeMetric_,myMatchingConeSize,LeadTrack_minPt_);
  myPFTau.setleadTrack(myleadTk);
  if(myleadTk.isNonnull()){
    double myleadTkDZ=(*myleadTk).dz();
    TFormula myTrackerSignalConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(TrackerSignalConeSizeFormula_,"Tracker signal cone size");
    double myTrackerSignalConeSize=myPFTauElementsOperators.computeConeSize(myTrackerSignalConeSizeTFormula,TrackerSignalConeVariableSize_min_,TrackerSignalConeVariableSize_max_);
    TFormula myTrackerIsolConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(TrackerIsolConeSizeFormula_,"Tracker isolation cone size");
    double myTrackerIsolConeSize=myPFTauElementsOperators.computeConeSize(myTrackerIsolConeSizeTFormula,TrackerIsolConeVariableSize_min_,TrackerIsolConeVariableSize_max_);     
    if (UseTrackLeadTrackDZconstraint_){
      myPFTau.setsignalTracks(myPFTauElementsOperators.tracksInCone((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,Track_minPt_,TrackLeadTrack_maxDZ_,myleadTkDZ));
      myPFTau.setisolationTracks(myPFTauElementsOperators.tracksInAnnulus((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,Track_minPt_,TrackLeadTrack_maxDZ_,myleadTkDZ));
    }else{
      myPFTau.setsignalTracks(myPFTauElementsOperators.tracksInCone((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,Track_minPt_));
      myPFTau.setisolationTracks(myPFTauElementsOperators.tracksInAnnulus((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,Track_minPt_));
    }
  }

  return myPFTau;  
}
