#include "RecoTauTag/RecoTau/interface/CaloRecoTauAlgorithm.h"

Tau CaloRecoTauAlgorithm::tag(const CombinedTauTagInfo& myTagInfo,const Vertex& myPV){
  CaloJetRef myCaloJet=myTagInfo.isolatedtautaginfoRef()->jet().castTo<CaloJetRef>(); // catch a ref to the initial CaloJet  
  Tau myTau(numeric_limits<int>::quiet_NaN(),myCaloJet->p4()); //create the Tau with the modified Lorentz-vector
  
  myTau.setcalojetRef(myCaloJet);

  math::XYZTLorentzVector myalternatLorentzVect(myTagInfo.alternatrecJet_HepLV().getX(),myTagInfo.alternatrecJet_HepLV().getY(),myTagInfo.alternatrecJet_HepLV().getZ(),myTagInfo.alternatrecJet_HepLV().getT());
  myTau.setalternatLorentzVect(myalternatLorentzVect);

  //Setting the SelectedTracks
  TrackRefVector myTks = myTagInfo.selectedTks();
  myTau.setSelectedTracks(myTks);  
  
  math::XYZTLorentzVector myTksInvariantMass(0.,0.,0.,0.);
  if((int)(myTks.size())!=0){
    for(int i=0;i<(int)myTks.size();i++){
      math::XYZTLorentzVector mychargedpicand_fromTk_LorentzVect(myTks[i]->momentum().x(),myTks[i]->momentum().y(),myTks[i]->momentum().z(),sqrt(pow((double)myTks[i]->momentum().r(),2)+pow(chargedpi_mass,2)));
      myTksInvariantMass+=mychargedpicand_fromTk_LorentzVect;
    }
  }
  myTau.setInvariantMass(myTksInvariantMass.mass());

  CaloTauElementsOperators myCaloTauElementsOperators(myTau);
  TFormula myMatchingConeSizeTFormula=myCaloTauElementsOperators.computeConeSizeTFormula(MatchingConeSizeFormula_,"Matching cone size");
  double myMatchingConeSize=myCaloTauElementsOperators.computeConeSize(myMatchingConeSizeTFormula,MatchingConeVariableSize_min_,MatchingConeVariableSize_max_);
  TrackRef myleadTk=myCaloTauElementsOperators.leadTk(MatchingConeMetric_,myMatchingConeSize,LeadTrack_minPt_);
  double myTau_refInnerPosition_x=0.;
  double myTau_refInnerPosition_y=0.;
  double myTau_refInnerPosition_z=0.;
  if(myleadTk.isNonnull()){
    myTau.setleadTrack(myleadTk);
    if(TransientTrackBuilder_!=0){ 
      SignedTransverseImpactParameter myleadTk_signediptMeasure;
      const TransientTrack myleadTransientTk=TransientTrackBuilder_->build(&(*myleadTk));
      GlobalVector myCaloJetdir((*myCaloJet).px(),(*myCaloJet).py(),(*myCaloJet).pz());
      myTau.setleadTracksignedSipt(myleadTk_signediptMeasure.apply(myleadTransientTk,myCaloJetdir,myPV).second.significance());
    }
    if((*myleadTk).innerOk()){
      myTau_refInnerPosition_x=(*myleadTk).innerPosition().x(); 
      myTau_refInnerPosition_y=(*myleadTk).innerPosition().y(); 
      myTau_refInnerPosition_z=(*myleadTk).innerPosition().z(); 
    }
    TFormula myTrackerSignalConeSizeTFormula=myCaloTauElementsOperators.computeConeSizeTFormula(TrackerSignalConeSizeFormula_,"Tracker signal cone size");
    double myTrackerSignalConeSize=myCaloTauElementsOperators.computeConeSize(myTrackerSignalConeSizeTFormula,TrackerSignalConeVariableSize_min_,TrackerSignalConeVariableSize_max_);
    TFormula myTrackerIsolConeSizeTFormula=myCaloTauElementsOperators.computeConeSizeTFormula(TrackerIsolConeSizeFormula_,"Tracker isolation cone size");
    double myTrackerIsolConeSize=myCaloTauElementsOperators.computeConeSize(myTrackerIsolConeSizeTFormula,TrackerIsolConeVariableSize_min_,TrackerIsolConeVariableSize_max_);     	
    
    TrackRefVector mySignalTks;
    mySignalTks=myCaloTauElementsOperators.tracksInCone((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,Track_minPt_);
    myTau.setSignalTracks(mySignalTks);
    
    //Setting the mass with only Signal Tracks
    math::XYZTLorentzVector mySignalTksInvariantMass(0.,0.,0.,0.);
    if((int)(mySignalTks.size())!=0){
      int mySignalTks_qsum=0;       
      for(int i=0;i<(int)mySignalTks.size();i++){
	mySignalTks_qsum+=mySignalTks[i]->charge();
	math::XYZTLorentzVector mychargedpicand_fromTk_LorentzVect(mySignalTks[i]->momentum().x(),mySignalTks[i]->momentum().y(),mySignalTks[i]->momentum().z(),sqrt(pow((double)mySignalTks[i]->momentum().r(),2)+pow(chargedpi_mass,2)));
	mySignalTksInvariantMass+=mychargedpicand_fromTk_LorentzVect;
      }
      myTau.setCharge(mySignalTks_qsum);    
    }
    myTau.setSignalElementsInvariantMass(mySignalTksInvariantMass.mass());
    
    TrackRefVector myIsolTks;
    myIsolTks=myCaloTauElementsOperators.tracksInAnnulus((*myleadTk).momentum(),TrackerSignalConeMetric_,myTrackerSignalConeSize,TrackerIsolConeMetric_,myTrackerIsolConeSize,Track_minPt_);
    myTau.setIsolationTracks(myIsolTks);
    
    float myIsolTks_Ptsum=0.;
    for(int i=0;i<(int)myIsolTks.size();i++) myIsolTks_Ptsum+=myIsolTks[i]->pt();
    //Setting sum of the pT of isolation Annulus Tracks
    myTau.setSumPtIsolation(myIsolTks_Ptsum);
    
    //Setting sum of the ET of isolation Annulus neutral ECAL BasicClusters
    myTau.setEMIsolation(myTagInfo.isolneutralEtsum());
  }
  myTau.setVertex(math::XYZPoint(myTau_refInnerPosition_x,myTau_refInnerPosition_y,myTau_refInnerPosition_z));
    
  //Setting the max HCalEnergy
  // ***access jet constituents
  const std::vector<CaloTowerRef> myCaloJet_ECALHCALTowers=(*myCaloJet).getConstituents();
  double mymaxEtHCALtower_Et=NAN; 
  for(unsigned int iTower=0;iTower<myCaloJet_ECALHCALTowers.size();iTower++){
    // select max Et HCAL tower
    if((*myCaloJet_ECALHCALTowers[iTower]).hadEt()>=mymaxEtHCALtower_Et) mymaxEtHCALtower_Et=(*myCaloJet_ECALHCALTowers[iTower]).hadEt();
  }
  myTau.setMaximumHcalEnergy(mymaxEtHCALtower_Et);

  //Setting the EmEnergyFraction    
  myTau.setEmEnergyFraction(myTagInfo.neutralE_o_TksEneutralE());
  
  //Setting the number of EcalClusters
  myTau.setNumberOfEcalClusters(myTagInfo.neutralECALClus_number());
  
  return myTau;  
}
