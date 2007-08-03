#include "RecoTauTag/RecoTau/interface/PFRecoTauAlgorithm.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include <Math/GenVector/VectorUtil.h>

Tau PFRecoTauAlgorithm::tag(const PFIsolatedTauTagInfo& myTagInfo){
  //Takes the jet
  //  const Jet & jet = * (myTagInfo.jet());
  PFJetRef myJet = myTagInfo.pfjetRef();
  //Takes the LeadPFChargedHadrCand
  float z_PV = 0;
  TrackRef myLeadTk;
  PFCandidateRef leadPFChargedHadron = (myTagInfo).leadPFChargedHadrCand(MatchingConeSize_, LeadCand_minPt_);
  if(leadPFChargedHadron.isNonnull()){
    if (leadPFChargedHadron->blockRef()->elements().size()!=0){
      for (OwnVector<PFBlockElement>::const_iterator iPFBlock=leadPFChargedHadron->blockRef()->elements().begin();
	   iPFBlock!=leadPFChargedHadron->blockRef()->elements().end();iPFBlock++){
	if ((*iPFBlock).type()==1 &&
	    ROOT::Math::VectorUtil::DeltaR(leadPFChargedHadron->momentum(),(*iPFBlock).trackRef()->momentum())<0.001){
	  myLeadTk =(*iPFBlock).trackRef();
	  if(myLeadTk.isNonnull()){
	    z_PV = myLeadTk->dz();
	  }
	}
      }
    }
  }
  
  math::XYZPoint  vtx = math::XYZPoint( 0, 0, z_PV );
  //create the Tau
  
  Tau myTau(myJet->charge(),myJet->p4(),vtx);
  
  myTau.setpfjetRef(myJet);
  
  //Setting the EmOverCharged energy
  myTau.setEmEnergyFraction((myJet->chargedEmEnergy() + myJet->neutralEmEnergy())/ (myJet->chargedHadronEnergy()  +myJet->neutralHadronEnergy()+myJet->chargedEmEnergy() + myJet->neutralEmEnergy() ));
  
  //Setting the PFCands
  PFCandidateRefVector myPFCands = myTagInfo.PFCands();
  myTau.setSelectedPFCands(myPFCands);  
  
  //Setting the PFChargedHadrCands
  PFCandidateRefVector myPFChargedHadrCands = myTagInfo.PFChargedHadrCands();
  myTau.setSelectedPFChargedHadrCands(myPFChargedHadrCands);  
  
  //Setting the PFNeutrHadrCands
  PFCandidateRefVector myPFNeutrHadrCands = myTagInfo.PFNeutrHadrCands();
  myTau.setSelectedPFNeutrHadrCands(myPFNeutrHadrCands);  
  
  //Setting the PFGammaCands
  PFCandidateRefVector myPFGammaCands = myTagInfo.PFGammaCands();
  myTau.setSelectedPFGammaCands(myPFGammaCands);  
  
  //Setting the LeadPFChargedHadrCands
  if(!leadPFChargedHadron){
  }else{
    math::XYZVector leadPFCand_XYZVector=(*leadPFChargedHadron).momentum() ;
    myTau.setleadPFChargedHadrCand(leadPFChargedHadron);
    //Setting the HCalEnergy from the LeadHadron
    myTau.setMaximumHcalEnergy(leadPFChargedHadron->energy());
        
    //Setting the SignalPFChargedHadrCands
    PFCandidateRefVector mySignalPFChargedHadrCands = 
      myTagInfo.PFChargedHadrCandsInCone(leadPFCand_XYZVector, TrackerSignalConeSize_,Candidates_minPt_);
    myTau.setSignalPFChargedHadrCands(mySignalPFChargedHadrCands); 
    
    //Setting charge
    if((int)(mySignalPFChargedHadrCands.size())!=0){
      int myCharge=0;       
      for(int i=0;i<(int)mySignalPFChargedHadrCands.size();i++) myCharge+=mySignalPFChargedHadrCands[i]->charge();
      myTau.setCharge(myCharge);
    }
    
    //Setting the IsolationBandPFChargedHadrCands
    PFCandidateRefVector myIsolationPFChargedHadrCands = 
      myTagInfo.PFChargedHadrCandsInBand(leadPFCand_XYZVector, TrackerSignalConeSize_,TrackerIsolConeSize_,Candidates_minPt_);
    myTau.setIsolationPFChargedHadrCands(myIsolationPFChargedHadrCands);  
        
    //Setting the SignalPFNeutrHadrCands
    PFCandidateRefVector mySignalPFNeutrHadrCands = 
      myTagInfo.PFNeutrHadrCandsInCone(leadPFCand_XYZVector, ECALSignalConeSize_ ,Candidates_minPt_);
    myTau.setSignalPFNeutrHadrCands(mySignalPFNeutrHadrCands);  
    
    //Setting the IsolationPFNeutrHadrCands
    PFCandidateRefVector myIsolationPFNeutrHadrCands = 
      myTagInfo.PFNeutrHadrCandsInBand(leadPFCand_XYZVector, ECALSignalConeSize_, ECALIsolConeSize_,Candidates_minPt_);
    myTau.setIsolationPFNeutrHadrCands(myIsolationPFNeutrHadrCands);  
    
    //Setting the SignalPFGammaCands
    PFCandidateRefVector mySignalPFGammaCands = 
      myTagInfo.PFGammaCandsInCone(leadPFCand_XYZVector, ECALSignalConeSize_ ,Candidates_minPt_);
    myTau.setSignalPFGammaCands(mySignalPFGammaCands);  
    
    //Setting the IsolationPFGammaCands
    PFCandidateRefVector myIsolationPFGammaCands = 
      myTagInfo.PFGammaCandsInBand(leadPFCand_XYZVector, ECALSignalConeSize_ , ECALIsolConeSize_ ,Candidates_minPt_);
    myTau.setIsolationPFGammaCands(myIsolationPFGammaCands);  
    
    //setting the mass with only Signal objects:
    math::XYZTLorentzVector signalCandidate;
    for(int i=0;i<(int)mySignalPFGammaCands.size();i++) signalCandidate+=mySignalPFGammaCands[i]->p4();
    for(int i=0;i<(int)mySignalPFNeutrHadrCands.size();i++) signalCandidate+=mySignalPFNeutrHadrCands[i]->p4();
    for(int i=0;i<(int)mySignalPFChargedHadrCands.size();i++) signalCandidate+=mySignalPFChargedHadrCands[i]->p4();
    myTau.setInvariantMass(signalCandidate.mass());
    
    //Setting sum of the pT of isolation Annulus charged hadrons
    float mySumPt=0.;
    for(int i=0;i<(int)myIsolationPFChargedHadrCands.size();i++) mySumPt+=myIsolationPFChargedHadrCands[i]->pt();
    myTau.setSumPtIsolation(mySumPt);  
    
    //Setting sum of the E_T of isolation Annulus gamma candidates
    float mySumEt=0.;
    for(int i=0;i<(int)myIsolationPFGammaCands.size();i++) mySumEt+=myIsolationPFGammaCands[i]->pt();
    myTau.setEMIsolation(mySumEt);
  }    
  return myTau;  
}
