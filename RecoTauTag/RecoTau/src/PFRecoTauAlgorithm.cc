#include "RecoTauTag/RecoTau/interface/PFRecoTauAlgorithm.h"

Tau PFRecoTauAlgorithm::tag(const PFIsolatedTauTagInfo& myTagInfo,const Vertex& myPV){
  PFJetRef myPFJet=myTagInfo.pfjetRef();  // catch a ref to the initial PFJet  
  Tau myTau(numeric_limits<int>::quiet_NaN(),myPFJet->p4());   // create the Tau
   
  myTau.setpfjetRef(myPFJet);
  
  myTau.setalternatLorentzVect(myTagInfo.alternatLorentzVect());

  //Setting the SelectedPFCands
  PFCandidateRefVector myPFCands = myTagInfo.PFCands();
  myTau.setSelectedPFCands(myPFCands);  
  
  math::XYZTLorentzVector myPFCandsInvariantMass(0.,0.,0.,0.);
  if((int)(myPFCands.size())!=0){
    for(int i=0;i<(int)myPFCands.size();i++) myPFCandsInvariantMass+=myPFCands[i]->p4();
  }
  myTau.setInvariantMass(myPFCandsInvariantMass.mass());

  //Setting the PFChargedHadrCands
  PFCandidateRefVector myPFChargedHadrCands = myTagInfo.PFChargedHadrCands();
  myTau.setSelectedPFChargedHadrCands(myPFChargedHadrCands);  
  
  //Setting the PFNeutrHadrCands
  PFCandidateRefVector myPFNeutrHadrCands = myTagInfo.PFNeutrHadrCands();
  myTau.setSelectedPFNeutrHadrCands(myPFNeutrHadrCands);  
  
  //Setting the PFGammaCands
  PFCandidateRefVector myPFGammaCands = myTagInfo.PFGammaCands();
  myTau.setSelectedPFGammaCands(myPFGammaCands);  
  
  PFTauElementsOperators myPFTauElementsOperators(myTau);
  TFormula myMatchingConeSizeTFormula=myPFTauElementsOperators.computeConeSizeTFormula(MatchingConeSizeFormula_,"Matching cone size");
  double myMatchingConeSize=myPFTauElementsOperators.computeConeSize(myMatchingConeSizeTFormula,MatchingConeVariableSize_min_,MatchingConeVariableSize_max_);
  PFCandidateRef myleadPFCand=myPFTauElementsOperators.leadPFCand(MatchingConeMetric_,myMatchingConeSize,LeadChargedHadrCand_minPt_);
  double myTau_refInnerPosition_x=0.;
  double myTau_refInnerPosition_y=0.;
  double myTau_refInnerPosition_z=0.;
  if(myleadPFCand.isNonnull()){
    myTau.setleadPFChargedHadrCand(myleadPFCand);
    //Setting the HCalEnergy from the LeadHadron
    myTau.setMaximumHcalEnergy((*myleadPFCand).energy());
    if ((*myleadPFCand).blockRef()->elements().size()!=0){
      for (OwnVector<PFBlockElement>::const_iterator iPFBlock=(*myleadPFCand).blockRef()->elements().begin();iPFBlock!=(*myleadPFCand).blockRef()->elements().end();iPFBlock++){
	if ((*iPFBlock).type()==PFRecTrack_codenumber && ROOT::Math::VectorUtil::DeltaR((*myleadPFCand).momentum(),(*iPFBlock).trackRef()->momentum())<0.001){
	  TrackRef myleadPFCand_rectk=(*iPFBlock).trackRef();
	  if(myleadPFCand_rectk.isNonnull()){
	    if(TransientTrackBuilder_!=0){ 
	      const TransientTrack myleadPFCand_rectransienttk=TransientTrackBuilder_->build(&(*myleadPFCand_rectk));
	      GlobalVector myPFJetdir((*myPFJet).px(),(*myPFJet).py(),(*myPFJet).pz());
	      myTau.setleadPFChargedHadrCandsignedSipt(IPTools::signedTransverseImpactParameter(myleadPFCand_rectransienttk,myPFJetdir,myPV).second.significance());
	    }
	    if((*myleadPFCand_rectk).innerOk()){
	      myTau_refInnerPosition_x=(*myleadPFCand_rectk).innerPosition().x(); 
	      myTau_refInnerPosition_y=(*myleadPFCand_rectk).innerPosition().y(); 
	      myTau_refInnerPosition_z=(*myleadPFCand_rectk).innerPosition().z(); 
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
    mySignalPFChargedHadrCands=myPFTauElementsOperators.PFChargedHadrCandsInCone((*myleadPFCand).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,ChargedHadrCand_minPt_);
    myTau.setSignalPFChargedHadrCands(mySignalPFChargedHadrCands);
    mySignalPFNeutrHadrCands=myPFTauElementsOperators.PFNeutrHadrCandsInCone((*myleadPFCand).momentum(),HCALSignalConeMetric_,myHCALSignalConeSize,NeutrHadrCand_minPt_);
    myTau.setSignalPFNeutrHadrCands(mySignalPFNeutrHadrCands);
    mySignalPFGammaCands=myPFTauElementsOperators.PFGammaCandsInCone((*myleadPFCand).momentum(),ECALSignalConeMetric_,myECALSignalConeSize,GammaCand_minPt_);
    myTau.setSignalPFGammaCands(mySignalPFGammaCands);
    
    //Setting the mass with only Signal PFCand's
    math::XYZTLorentzVector mySignalPFCandsInvariantMass(0.,0.,0.,0.);
    if((int)(mySignalPFChargedHadrCands.size())!=0){
      int mySignalPFChargedHadrCands_qsum=0;       
      for(int i=0;i<(int)mySignalPFChargedHadrCands.size();i++){
	mySignalPFChargedHadrCands_qsum+=mySignalPFChargedHadrCands[i]->charge();
	mySignalPFCandsInvariantMass+=mySignalPFChargedHadrCands[i]->p4();
	mySignalPFCands.push_back(mySignalPFChargedHadrCands[i]);
      }
      myTau.setCharge(mySignalPFChargedHadrCands_qsum);    
    }
    for(int i=0;i<(int)mySignalPFNeutrHadrCands.size();i++){
      mySignalPFCandsInvariantMass+=mySignalPFNeutrHadrCands[i]->p4();
      mySignalPFCands.push_back(mySignalPFNeutrHadrCands[i]);
    }
    for(int i=0;i<(int)mySignalPFGammaCands.size();i++){
      mySignalPFCandsInvariantMass+=mySignalPFGammaCands[i]->p4();
      mySignalPFCands.push_back(mySignalPFGammaCands[i]);
    }
    myTau.setSignalElementsInvariantMass(mySignalPFCandsInvariantMass.mass());
    myTau.setSignalPFCands(mySignalPFCands);
    
    PFCandidateRefVector myIsolPFChargedHadrCands,myIsolPFNeutrHadrCands,myIsolPFGammaCands,myIsolPFCands;
    myIsolPFChargedHadrCands=myPFTauElementsOperators.PFChargedHadrCandsInAnnulus((*myleadPFCand).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,ChargedHadrCand_minPt_);
    myTau.setIsolationPFChargedHadrCands(myIsolPFChargedHadrCands);
    myIsolPFNeutrHadrCands=myPFTauElementsOperators.PFNeutrHadrCandsInAnnulus((*myleadPFCand).momentum(),HCALSignalConeMetric_,myHCALSignalConeSize,HCALIsolConeMetric_,myHCALIsolConeSize,NeutrHadrCand_minPt_);
    myTau.setIsolationPFNeutrHadrCands(myIsolPFNeutrHadrCands);
    myIsolPFGammaCands=myPFTauElementsOperators.PFGammaCandsInAnnulus((*myleadPFCand).momentum(),ECALSignalConeMetric_,myECALSignalConeSize,ECALIsolConeMetric_,myECALIsolConeSize,GammaCand_minPt_);  
    myTau.setIsolationPFGammaCands(myIsolPFGammaCands);

    float myIsolPFChargedHadrCands_Ptsum=0.;
    float myIsolPFGammaCands_Etsum=0.;
    for(int i=0;i<(int)myIsolPFChargedHadrCands.size();i++){
      myIsolPFChargedHadrCands_Ptsum+=myIsolPFChargedHadrCands[i]->pt();
      myIsolPFCands.push_back(myIsolPFChargedHadrCands[i]);
    }
    for(int i=0;i<(int)myIsolPFNeutrHadrCands.size();i++)myIsolPFCands.push_back(myIsolPFNeutrHadrCands[i]);
    for(int i=0;i<(int)myIsolPFGammaCands.size();i++){
      myIsolPFGammaCands_Etsum+=myIsolPFGammaCands[i]->et();
      myIsolPFCands.push_back(myIsolPFGammaCands[i]);
    } 
    myTau.setIsolationPFCands(myIsolPFCands);
     
    //Setting sum of the pT of isolation Annulus charged hadron PFCand's
    myTau.setSumPtIsolation(myIsolPFChargedHadrCands_Ptsum);
    //Setting sum of the ET of isolation Annulus gamma PFCand's
    myTau.setEMIsolation(myIsolPFGammaCands_Etsum); 
  }
  myTau.setVertex(math::XYZPoint(myTau_refInnerPosition_x,myTau_refInnerPosition_y,myTau_refInnerPosition_z));
  
  //Setting the EmOverCharged energy
  myTau.setEmEnergyFraction((myPFJet->chargedEmEnergy() + myPFJet->neutralEmEnergy())/ (myPFJet->chargedHadronEnergy()  +myPFJet->neutralHadronEnergy()+myPFJet->chargedEmEnergy() + myPFJet->neutralEmEnergy() ));
  
  return myTau;  
}
