#include "RecoTauTag/RecoTau/interface/PFRecoTauAlgorithm.h"

PFRecoTauAlgorithm::PFRecoTauAlgorithm() : TransientTrackBuilder_(0){}  
PFRecoTauAlgorithm::PFRecoTauAlgorithm(const ParameterSet& iConfig) : TransientTrackBuilder_(0){
  LeadChargedHadrCand_minPt_          = iConfig.getParameter<double>("LeadChargedHadrCand_minPt"); 
  ChargedHadrCand_minPt_              = iConfig.getParameter<double>("ChargedHadrCand_minPt");
  UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint_ = iConfig.getParameter<bool>("UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint");
  ChargedHadrCandLeadChargedHadrCand_tksmaxDZ_ = iConfig.getParameter<double>("ChargedHadrCandLeadChargedHadrCand_tksmaxDZ");
  NeutrHadrCand_minPt_                = iConfig.getParameter<double>("NeutrHadrCand_minPt");
  GammaCand_minPt_                    = iConfig.getParameter<double>("GammaCand_minPt");       
  LeadTrack_minPt_                    = iConfig.getParameter<double>("LeadTrack_minPt");
  Track_minPt_                        = iConfig.getParameter<double>("Track_minPt");
  UseTrackLeadTrackDZconstraint_      = iConfig.getParameter<bool>("UseTrackLeadTrackDZconstraint");
  TrackLeadTrack_maxDZ_               = iConfig.getParameter<double>("TrackLeadTrack_maxDZ");
  
  MatchingConeMetric_                 = iConfig.getParameter<string>("MatchingConeMetric");
  MatchingConeSizeFormula_            = iConfig.getParameter<string>("MatchingConeSizeFormula");
  MatchingConeSize_min_               = iConfig.getParameter<double>("MatchingConeSize_min");
  MatchingConeSize_max_               = iConfig.getParameter<double>("MatchingConeSize_max");
  TrackerSignalConeMetric_            = iConfig.getParameter<string>("TrackerSignalConeMetric");
  TrackerSignalConeSizeFormula_       = iConfig.getParameter<string>("TrackerSignalConeSizeFormula");
  TrackerSignalConeSize_min_          = iConfig.getParameter<double>("TrackerSignalConeSize_min");
  TrackerSignalConeSize_max_          = iConfig.getParameter<double>("TrackerSignalConeSize_max");
  TrackerIsolConeMetric_              = iConfig.getParameter<string>("TrackerIsolConeMetric"); 
  TrackerIsolConeSizeFormula_         = iConfig.getParameter<string>("TrackerIsolConeSizeFormula"); 
  TrackerIsolConeSize_min_            = iConfig.getParameter<double>("TrackerIsolConeSize_min");
  TrackerIsolConeSize_max_            = iConfig.getParameter<double>("TrackerIsolConeSize_max");
  ECALSignalConeMetric_               = iConfig.getParameter<string>("ECALSignalConeMetric");
  ECALSignalConeSizeFormula_          = iConfig.getParameter<string>("ECALSignalConeSizeFormula");    
  ECALSignalConeSize_min_             = iConfig.getParameter<double>("ECALSignalConeSize_min");
  ECALSignalConeSize_max_             = iConfig.getParameter<double>("ECALSignalConeSize_max");
  ECALIsolConeMetric_                 = iConfig.getParameter<string>("ECALIsolConeMetric");
  ECALIsolConeSizeFormula_            = iConfig.getParameter<string>("ECALIsolConeSizeFormula");      
  ECALIsolConeSize_min_               = iConfig.getParameter<double>("ECALIsolConeSize_min");
  ECALIsolConeSize_max_               = iConfig.getParameter<double>("ECALIsolConeSize_max");
  HCALSignalConeMetric_               = iConfig.getParameter<string>("HCALSignalConeMetric");
  HCALSignalConeSizeFormula_          = iConfig.getParameter<string>("HCALSignalConeSizeFormula");    
  HCALSignalConeSize_min_             = iConfig.getParameter<double>("HCALSignalConeSize_min");
  HCALSignalConeSize_max_             = iConfig.getParameter<double>("HCALSignalConeSize_max");
  HCALIsolConeMetric_                 = iConfig.getParameter<string>("HCALIsolConeMetric");
  HCALIsolConeSizeFormula_            = iConfig.getParameter<string>("HCALIsolConeSizeFormula");      
  HCALIsolConeSize_min_               = iConfig.getParameter<double>("HCALIsolConeSize_min");
  HCALIsolConeSize_max_               = iConfig.getParameter<double>("HCALIsolConeSize_max");
  
  AreaMetric_recoElements_maxabsEta_  = iConfig.getParameter<double>("AreaMetric_recoElements_maxabsEta");
  ChargedHadrCand_IsolAnnulus_minNhits_ = iConfig.getParameter<uint32_t>("ChargedHadrCand_IsolAnnulus_minNhits");
  Track_IsolAnnulus_minNhits_           = iConfig.getParameter<uint32_t>("Track_IsolAnnulus_minNhits");

  ElecPreIDLeadTkMatch_maxDR_ = iConfig.getParameter<double>("ElecPreIDLeadTkMatch_maxDR");
  EcalStripSumE_minClusEnergy_ = iConfig.getParameter<double>("EcalStripSumE_minClusEnergy");
  EcalStripSumE_deltaEta_ = iConfig.getParameter<double>("EcalStripSumE_deltaEta");
  EcalStripSumE_deltaPhiOverQ_minValue_ = iConfig.getParameter<double>("EcalStripSumE_deltaPhiOverQ_minValue");
  EcalStripSumE_deltaPhiOverQ_maxValue_ = iConfig.getParameter<double>("EcalStripSumE_deltaPhiOverQ_maxValue");


}
void PFRecoTauAlgorithm::setTransientTrackBuilder(const TransientTrackBuilder* x){TransientTrackBuilder_=x;}

PFTau PFRecoTauAlgorithm::buildPFTau(const PFTauTagInfoRef& myPFTauTagInfoRef,const Vertex& myPV){
  PFJetRef myPFJet=(*myPFTauTagInfoRef).pfjetRef();  // catch a ref to the initial PFJet  
  PFTau myPFTau(numeric_limits<int>::quiet_NaN(),myPFJet->p4());   // create the PFTau
   
  myPFTau.setpfTauTagInfoRef(myPFTauTagInfoRef);
  
  // retrieve jet candidates
  PFCandidateRefVector myPFCands=(*myPFTauTagInfoRef).PFCands();
  PFCandidateRefVector myPFChargedHadrCands=(*myPFTauTagInfoRef).PFChargedHadrCands();
  PFCandidateRefVector myPFNeutrHadrCands=(*myPFTauTagInfoRef).PFNeutrHadrCands();
  PFCandidateRefVector myPFGammaCands=(*myPFTauTagInfoRef).PFGammaCands();
  
  //initialize utility class used to build tau
  PFTauElementsOperators myPFTauElementsOperators(myPFTau);
  //find lead track about jet axis
  TFormula myMatchingConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(MatchingConeSizeFormula_,"Matching cone size");
  double myMatchingConeSize=myPFTauElementsOperators.computeConeSize(myMatchingConeSizeTFormula,MatchingConeSize_min_,MatchingConeSize_max_);
  PFCandidateRef myleadPFCand=myPFTauElementsOperators.leadPFChargedHadrCand(MatchingConeMetric_,myMatchingConeSize,LeadChargedHadrCand_minPt_);
  bool myleadPFCand_rectkavailable=false;
  double myleadPFCand_rectkDZ=0.;
  double myPFTau_refInnerPosition_x=0.;
  double myPFTau_refInnerPosition_y=0.;
  double myPFTau_refInnerPosition_z=0.;
  if(myleadPFCand.isNonnull()){ //lead track found, build PFTau
    myPFTau.setleadPFChargedHadrCand(myleadPFCand);
    TrackRef myleadPFCand_rectk=(*myleadPFCand).trackRef();
    if(myleadPFCand_rectk.isNonnull()){
      myleadPFCand_rectkavailable=true;
      myleadPFCand_rectkDZ=(*myleadPFCand_rectk).dz();
      if(TransientTrackBuilder_!=0){ 
	const TransientTrack myleadPFCand_rectransienttk=TransientTrackBuilder_->build(&(*myleadPFCand_rectk));
	GlobalVector myPFJetdir((*myPFJet).px(),(*myPFJet).py(),(*myPFJet).pz());
	if(IPTools::signedTransverseImpactParameter(myleadPFCand_rectransienttk,myPFJetdir,myPV).first)
	  myPFTau.setleadPFChargedHadrCandsignedSipt(IPTools::signedTransverseImpactParameter(myleadPFCand_rectransienttk,myPFJetdir,myPV).second.significance());
      }
    }
    //filter tracks by distance to DCA of lead track to PV, if desired
    if (UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint_ && myleadPFCand_rectkavailable){
      PFCandidateRefVector myPFChargedHadrCandsbis;
      for(PFCandidateRefVector::const_iterator iPFCand=myPFChargedHadrCands.begin();iPFCand!=myPFChargedHadrCands.end();iPFCand++){
	TrackRef iPFChargedHadrCand_track=(**iPFCand).trackRef();
	if (!iPFChargedHadrCand_track)continue;
	if (fabs((*iPFChargedHadrCand_track).dz()-myleadPFCand_rectkDZ)<=ChargedHadrCandLeadChargedHadrCand_tksmaxDZ_) myPFChargedHadrCandsbis.push_back(*iPFCand);
      }
      myPFChargedHadrCands=myPFChargedHadrCandsbis;
    }

    // candidate collections to insert into PFTau
    PFCandidateRefVector mySignalPFChargedHadrCands,mySignalPFNeutrHadrCands,mySignalPFGammaCands,mySignalPFCands;
    PFCandidateRefVector myUnfilteredIsolPFChargedHadrCands,myIsolPFNeutrHadrCands,myIsolPFGammaCands,myIsolPFCands;
    PFCandidateRefVector myIsolPFChargedHadrCands;

    // parse algorithm choice
    bool useTrackerConeAlgo     = false;
    bool useECALConeAlgo        = false; 
    bool useHCALConeAlgo        = false;
    bool useAConelessAlgo       = true;

    if (!TrackerSignalConeMetric_.compare("DR") || !TrackerSignalConeMetric_.compare("angle") || !TrackerSignalConeMetric_.compare("area"))
       useTrackerConeAlgo       = true;
    if (!ECALSignalConeMetric_.compare("DR")    || !ECALSignalConeMetric_.compare("angle")    || !ECALSignalConeMetric_.compare("area"))
       useECALConeAlgo          = true;
    if (!HCALSignalConeMetric_.compare("DR")    || !HCALSignalConeMetric_.compare("angle")    || !HCALSignalConeMetric_.compare("area"))
       useHCALConeAlgo          = true;
    // if all three object types use a cone algo don't use a coneless
    if (useTrackerConeAlgo && useECALConeAlgo && useHCALConeAlgo)
       useAConelessAlgo         = false;


    if (useAConelessAlgo)
    {
       // Do inside-out tau signal object identification
       if (!TrackerSignalConeMetric_.compare("InsideOut") || !TrackerSignalConeMetric_.compare("InsideOutStaticAxis"))
       {
          //warn if necessary
          if (useECALConeAlgo)
             edm::LogWarning("PFRecoTauAlgorithm") << "Warning: Using Inside-Out signal algorithm, but ECAL is set to use another algorithm!";

          bool shiftingAxis = true;
          if ( !TrackerSignalConeMetric_.compare("InsideOutStaticAxis") )
             shiftingAxis = false;

          TFormula insideOutGrowthConstraint = myPFTauElementsOperators.computeConeSizeTFormula(TrackerSignalConeSizeFormula_,"Inside out growth constraint");
          myPFTauElementsOperators.computeInsideOutContents(myPFChargedHadrCands, myPFGammaCands,
                                                            myleadPFCand->momentum(), insideOutGrowthConstraint, TauTagTools::computeDeltaR, 
                                                            TrackerSignalConeSize_min_, TrackerSignalConeSize_max_, ECALSignalConeSize_min_, ECALSignalConeSize_max_,
                                                            ChargedHadrCand_minPt_, GammaCand_minPt_,
                                                            TrackerIsolConeMetric_, TrackerIsolConeSize_max_,
                                                            mySignalPFChargedHadrCands, myUnfilteredIsolPFChargedHadrCands,
                                                            mySignalPFGammaCands, myIsolPFGammaCands, shiftingAxis);

       }
    }
    
    if (useTrackerConeAlgo)
    {
       TFormula myTrackerSignalConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(TrackerSignalConeSizeFormula_,"Tracker signal cone size");
       double myTrackerSignalConeSize=myPFTauElementsOperators.computeConeSize(myTrackerSignalConeSizeTFormula,TrackerSignalConeSize_min_,TrackerSignalConeSize_max_);
       TFormula myTrackerIsolConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(TrackerIsolConeSizeFormula_,"Tracker isolation cone size");
       double myTrackerIsolConeSize=myPFTauElementsOperators.computeConeSize(myTrackerIsolConeSizeTFormula,TrackerIsolConeSize_min_,TrackerIsolConeSize_max_);     	
       if (UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint_ && myleadPFCand_rectkavailable) mySignalPFChargedHadrCands=myPFTauElementsOperators.PFChargedHadrCandsInCone((*myleadPFCand).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,ChargedHadrCand_minPt_,ChargedHadrCandLeadChargedHadrCand_tksmaxDZ_,myleadPFCand_rectkDZ);
       else mySignalPFChargedHadrCands=myPFTauElementsOperators.PFChargedHadrCandsInCone((*myleadPFCand).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,ChargedHadrCand_minPt_);
       if (UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint_ && myleadPFCand_rectkavailable) myUnfilteredIsolPFChargedHadrCands=myPFTauElementsOperators.PFChargedHadrCandsInAnnulus((*myleadPFCand).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,ChargedHadrCand_minPt_,ChargedHadrCandLeadChargedHadrCand_tksmaxDZ_,myleadPFCand_rectkDZ);
       else myUnfilteredIsolPFChargedHadrCands=myPFTauElementsOperators.PFChargedHadrCandsInAnnulus((*myleadPFCand).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,ChargedHadrCand_minPt_);
    }

    if (useECALConeAlgo)
    {
       TFormula myECALSignalConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(ECALSignalConeSizeFormula_,"ECAL signal cone size");
       double myECALSignalConeSize=myPFTauElementsOperators.computeConeSize(myECALSignalConeSizeTFormula,ECALSignalConeSize_min_,ECALSignalConeSize_max_);
       TFormula myECALIsolConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(ECALIsolConeSizeFormula_,"ECAL isolation cone size");
       double myECALIsolConeSize=myPFTauElementsOperators.computeConeSize(myECALIsolConeSizeTFormula,ECALIsolConeSize_min_,ECALIsolConeSize_max_);     	
       mySignalPFGammaCands=myPFTauElementsOperators.PFGammaCandsInCone((*myleadPFCand).momentum(),ECALSignalConeMetric_,myECALSignalConeSize,GammaCand_minPt_);
       myIsolPFGammaCands=myPFTauElementsOperators.PFGammaCandsInAnnulus((*myleadPFCand).momentum(),ECALSignalConeMetric_,myECALSignalConeSize,ECALIsolConeMetric_,myECALIsolConeSize,GammaCand_minPt_);  
    }

    if (useHCALConeAlgo)
    {
       TFormula myHCALSignalConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(HCALSignalConeSizeFormula_,"HCAL signal cone size");
       double myHCALSignalConeSize=myPFTauElementsOperators.computeConeSize(myHCALSignalConeSizeTFormula,HCALSignalConeSize_min_,HCALSignalConeSize_max_);
       TFormula myHCALIsolConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(HCALIsolConeSizeFormula_,"HCAL isolation cone size");
       double myHCALIsolConeSize=myPFTauElementsOperators.computeConeSize(myHCALIsolConeSizeTFormula,HCALIsolConeSize_min_,HCALIsolConeSize_max_);     	
       mySignalPFNeutrHadrCands=myPFTauElementsOperators.PFNeutrHadrCandsInCone((*myleadPFCand).momentum(),HCALSignalConeMetric_,myHCALSignalConeSize,NeutrHadrCand_minPt_);
       myIsolPFNeutrHadrCands=myPFTauElementsOperators.PFNeutrHadrCandsInAnnulus((*myleadPFCand).momentum(),HCALSignalConeMetric_,myHCALSignalConeSize,HCALIsolConeMetric_,myHCALIsolConeSize,NeutrHadrCand_minPt_);
    }

    // set computed signal cone objects
    myPFTau.setsignalPFChargedHadrCands(mySignalPFChargedHadrCands);
    myPFTau.setsignalPFNeutrHadrCands(mySignalPFNeutrHadrCands);
    myPFTau.setsignalPFGammaCands(mySignalPFGammaCands);

    //compute charge
    if((int)(mySignalPFChargedHadrCands.size())!=0){
      int mySignalPFChargedHadrCands_qsum=0;       
      for(int i=0;i<(int)mySignalPFChargedHadrCands.size();i++){
	mySignalPFChargedHadrCands_qsum+=mySignalPFChargedHadrCands[i]->charge();
	mySignalPFCands.push_back(mySignalPFChargedHadrCands[i]);
      }
      myPFTau.setCharge(mySignalPFChargedHadrCands_qsum);    
    }

    // merge signal charged and gammas into total signal cone collection
    for(int i=0;i<(int)mySignalPFNeutrHadrCands.size();i++) mySignalPFCands.push_back(mySignalPFNeutrHadrCands[i]);
    for(int i=0;i<(int)mySignalPFGammaCands.size();i++) mySignalPFCands.push_back(mySignalPFGammaCands[i]);
    myPFTau.setsignalPFCands(mySignalPFCands);

    // filter isolation annulus charged candidates with additional nHits quality cut (note that other cuts [pt, chi2, are already cut on])
    myIsolPFChargedHadrCands = TauTagTools::filteredPFChargedHadrCandsByNumTrkHits(myUnfilteredIsolPFChargedHadrCands, ChargedHadrCand_IsolAnnulus_minNhits_); 

    // set isolation objects
    myPFTau.setisolationPFChargedHadrCands(myIsolPFChargedHadrCands);
    myPFTau.setisolationPFGammaCands(myIsolPFGammaCands);
    myPFTau.setisolationPFNeutrHadrCands(myIsolPFNeutrHadrCands);

    // compute discrimination quantities
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
     
  }

  math::XYZTLorentzVector alternatLorentzVect(0.,0.,0.,0.);
  for (PFCandidateRefVector::const_iterator iGammaCand=myPFGammaCands.begin();iGammaCand!=myPFGammaCands.end();iGammaCand++) alternatLorentzVect+=(**iGammaCand).p4();
  for (PFCandidateRefVector::const_iterator iChargedHadrCand=myPFChargedHadrCands.begin();iChargedHadrCand!=myPFChargedHadrCands.end();iChargedHadrCand++) alternatLorentzVect+=(**iChargedHadrCand).p4();  
  myPFTau.setalternatLorentzVect(alternatLorentzVect);
  
  myPFTau.setVertex(math::XYZPoint(myPFTau_refInnerPosition_x,myPFTau_refInnerPosition_y,myPFTau_refInnerPosition_z));
  
  // set the leading, signal cone and isolation annulus Tracks (the initial list of Tracks was catched through a JetTracksAssociation object, not through the charged hadr. PFCandidates inside the PFJet ; the motivation for considering these objects is the need for checking that a selection by the charged hadr. PFCandidates is equivalent to a selection by the rec. Tracks.)
  TrackRef myleadTk=myPFTauElementsOperators.leadTk(MatchingConeMetric_,myMatchingConeSize,LeadTrack_minPt_);
  myPFTau.setleadTrack(myleadTk);
  if(myleadTk.isNonnull()){
    double myleadTkDZ=(*myleadTk).dz();
    TFormula myTrackerSignalConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(TrackerSignalConeSizeFormula_,"Tracker signal cone size");
    double myTrackerSignalConeSize=myPFTauElementsOperators.computeConeSize(myTrackerSignalConeSizeTFormula,TrackerSignalConeSize_min_,TrackerSignalConeSize_max_);
    TFormula myTrackerIsolConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(TrackerIsolConeSizeFormula_,"Tracker isolation cone size");
    double myTrackerIsolConeSize=myPFTauElementsOperators.computeConeSize(myTrackerIsolConeSizeTFormula,TrackerIsolConeSize_min_,TrackerIsolConeSize_max_);     
    if (UseTrackLeadTrackDZconstraint_){
      myPFTau.setsignalTracks(myPFTauElementsOperators.tracksInCone((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,Track_minPt_,TrackLeadTrack_maxDZ_,myleadTkDZ));

      TrackRefVector myUnfilteredTracks = myPFTauElementsOperators.tracksInAnnulus((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,Track_minPt_,TrackLeadTrack_maxDZ_,myleadTkDZ);
      TrackRefVector myFilteredTracks = TauTagTools::filteredTracksByNumTrkHits(myUnfilteredTracks, Track_IsolAnnulus_minNhits_);
      myPFTau.setisolationTracks(myFilteredTracks);

    }else{
      myPFTau.setsignalTracks(myPFTauElementsOperators.tracksInCone((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,Track_minPt_));
      
      TrackRefVector myUnfilteredTracks = myPFTauElementsOperators.tracksInAnnulus((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,Track_minPt_);
      TrackRefVector myFilteredTracks = TauTagTools::filteredTracksByNumTrkHits(myUnfilteredTracks, Track_IsolAnnulus_minNhits_);
      myPFTau.setisolationTracks(myFilteredTracks);
    }
  }

  
  /* For elecron rejection */
  double myECALenergy=0.;
  double myHCALenergy=0.;
  double myHCALenergy3x3=0.;
  double myMaximumHCALPFClusterE=0.;
  double myMaximumHCALPFClusterEt=0.;
  double myStripClusterE=0.;
  double myEmfrac = -1.;
  bool   myElecPreid = false;
  reco::TrackRef myElecTrk;
  
  // Corrected Cluster energies
  for(int i=0;i<(int)myPFCands.size();i++){
    myHCALenergy += myPFCands[i]->hcalEnergy();
    myECALenergy += myPFCands[i]->ecalEnergy();
  }
  
  if(myleadPFCand.isNonnull()){
    if (myleadPFCand->mva_e_pi()==1) {
      myElecPreid = true;
    }
    math::XYZPointF myElecTrkEcalPos = myleadPFCand->positionAtECALEntrance();
    myElecTrk = myleadPFCand->trackRef();//Electron candidate
    
    if(myElecTrk.isNonnull()) {
      if ((myHCALenergy+myECALenergy)>0.)
	myEmfrac = myECALenergy/(myHCALenergy+myECALenergy);
      myPFTau.setemFraction((float)myEmfrac);
      myPFTau.sethcalTotOverPLead((float)myHCALenergy/(float)myElecTrk->p());
      
      /* Obsolete: If using PFBlock need to use CaloTowers in future!!!!!!
	 myPFTau.sethcalTotOverPLead((float)myHCALenergy/(float)myElecTrk->p());
	 myPFTau.sethcalMaxOverPLead((float)myMaximumHCALPFClusterE/(float)myElecTrk->p());
	 myPFTau.sethcal3x3OverPLead((float)myHCALenergy3x3/(float)myElecTrk->p());
	 myPFTau.setecalStripSumEOverPLead((float)myStripClusterE/(float)myElecTrk->p());
	 myPFTau.setmaximumHCALPFClusterEt(myMaximumHCALPFClusterEt);
      */
      
      // Here use the corrected energies fromPFCandidate (to be changed - can be confusing)
      myPFTau.sethcal3x3OverPLead((float)myHCALenergy/(float)myElecTrk->p());
      myPFTau.setecalStripSumEOverPLead((float)myECALenergy/(float)myElecTrk->p());
      
      myPFTau.setelectronPreIDDecision(myElecPreid);
      if (myElecTrk.isNonnull()) myPFTau.setelectronPreIDTrack(myElecTrk);
      
      // These need to be filled!
      //myPFTau.setbremsRecoveryEOverPLead(my...);
      //myPFTau.setelectronPreIDOutput(my...);
      
    }  
  }
  /* End elecron rejection */
  
  
  return myPFTau;  
}
