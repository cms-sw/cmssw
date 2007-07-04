#include "RecoTauTag/RecoTau/interface/PFRecoTauAlgorithm.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

Tau PFRecoTauAlgorithm::tag(const PFIsolatedTauTagInfo& myTagInfo)
{
  //Takes the jet
  //  const Jet & jet = * (myTagInfo.jet());
  PFJetRef myJet = myTagInfo.pfjetRef();
    //Takes the LeadChargedHadron
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
	
    //Setting the EmOverCharged energy
    myTau.setEmEnergyFraction((myJet->chargedEmEnergy() + myJet->neutralEmEnergy())/ (myJet->chargedHadronEnergy()  +myJet->neutralHadronEnergy()+myJet->chargedEmEnergy() + myJet->neutralEmEnergy() ));
    
    //Setting the ChargedHadrons
    PFCandidateRefVector myChargedHadrons = myTagInfo.PFChargedHadrCands();
    myTau.setSelectedChargedHadrons(myChargedHadrons);  
    
    //Setting the NeutralHadrons
    PFCandidateRefVector myNeutralHadrons = myTagInfo.PFNeutrHadrCands();
    myTau.setSelectedNeutralHadrons(myNeutralHadrons);  
    
    //Setting the GammaCandidates
    PFCandidateRefVector myGammaCandidates = myTagInfo.PFGammaCands();
    myTau.setSelectedGammaCandidates(myGammaCandidates);  
    
    //Setting the LeadChargedHadrons
    if(!leadPFChargedHadron){
    }else{
      math::XYZVector leadPFCand_XYZVector=(*leadPFChargedHadron).momentum() ;
      myTau.setLeadingChargedHadron(leadPFChargedHadron);
      
      //Setting the HCalEnergy from the LeadHadron
      myTau.setMaximumHcalEnergy(leadPFChargedHadron->energy());
      

      //Setting the SignalChargedHadrons
      PFCandidateRefVector mySignalChargedHadrons = 
	myTagInfo.PFChargedHadrCandsInCone(leadPFCand_XYZVector, TrackerSignalConeSize_,Candidates_minPt_);
      myTau.setSignalChargedHadrons(mySignalChargedHadrons); 
      
      
      //Setting charge
      int myCharge = 0.;
      for(int i=0; i<mySignalChargedHadrons.size();i++)
	{
	  myCharge = myCharge + mySignalChargedHadrons[i]->charge();
	}
      myTau.setCharge(myCharge);
    
      //Setting the IsolationBandChargedHadrons
      PFCandidateRefVector myIsolationChargedHadrons = 
	myTagInfo.PFChargedHadrCandsInBand(leadPFCand_XYZVector, TrackerSignalConeSize_,TrackerIsolConeSize_,Candidates_minPt_);
      myTau.setIsolationChargedHadrons(myIsolationChargedHadrons);  
      

      //Setting the SignalNeutralHadrons
      PFCandidateRefVector mySignalNeutralHadrons = 
	myTagInfo.PFNeutrHadrCandsInCone(leadPFCand_XYZVector, ECALSignalConeSize_ ,Candidates_minPt_);
      myTau.setSignalNeutralHadrons(mySignalNeutralHadrons);  
      
      //Setting the IsolationNeutralHadrons
      PFCandidateRefVector myIsolationNeutralHadrons = 
	myTagInfo.PFNeutrHadrCandsInBand(leadPFCand_XYZVector, ECALSignalConeSize_, ECALIsolConeSize_,Candidates_minPt_);
      myTau.setIsolationNeutralHadrons(myIsolationNeutralHadrons);  
      
      //Setting the SignalGammaCandidates
      PFCandidateRefVector mySignalGammaCandidates = 
	myTagInfo.PFGammaCandsInCone(leadPFCand_XYZVector, ECALSignalConeSize_ ,Candidates_minPt_);
      myTau.setSignalGammaCandidates(mySignalGammaCandidates);  
      
      //Setting the IsolationGammaCandidates
      PFCandidateRefVector myIsolationGammaCandidates = 
	myTagInfo.PFGammaCandsInBand(leadPFCand_XYZVector, ECALSignalConeSize_ , ECALIsolConeSize_ ,Candidates_minPt_);
      myTau.setIsolationGammaCandidates(myIsolationGammaCandidates);  

      //setting the mass with only Signal objects:
      math::XYZTLorentzVector signalCandidate;
      for(int i=0;i<mySignalGammaCandidates.size();i++)
	{
	  signalCandidate = signalCandidate + mySignalGammaCandidates[i]->p4();
	}
      for(int i=0;i<mySignalNeutralHadrons.size();i++)
	{
	  signalCandidate = signalCandidate + mySignalNeutralHadrons[i]->p4();
	}
      for(int i=0;i<mySignalChargedHadrons.size();i++)
	{
	  signalCandidate = signalCandidate + mySignalChargedHadrons[i]->p4();
	}
      myTau.setInvariantMass(signalCandidate.mass());
      
      //Setting sum of the pT of isolation Annulus charged hadrons
      float mySumPt=0.;
      for(int i=0; i<myIsolationChargedHadrons.size();i++)
	{
	  mySumPt= mySumPt + myIsolationChargedHadrons[i]->pt();
	}
      myTau.setSumPtIsolation(mySumPt);  
      
      //Setting sum of the E_T of isolation Annulus gamma candidates
      float mySumEt=0.;
      for(int i=0; i<myIsolationGammaCandidates.size();i++)
	{
	  mySumEt= mySumEt + myIsolationGammaCandidates[i]->pt();
	}
      myTau.setEMIsolation(mySumEt);
      
    


/*
setLeadTkTIP(const Measurement1D& myIP)  { transverseIp_leadTk_ = myIP;}
setLeadTk3DIP(const Measurement1D& myIP)  {  ip3D_leadTk_=myIP;}
*/
 
  }    
  return myTau;
  
}
