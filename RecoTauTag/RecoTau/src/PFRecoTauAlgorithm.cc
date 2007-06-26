#include "RecoTauTag/RecoTau/interface/PFRecoTauAlgorithm.h"
#include "DataFormats/Math/interface/Point3D.h"
Tau PFRecoTauAlgorithm::tag(const PFIsolatedTauTagInfo& myTagInfo)
{
  //Takes the jet
  //  const Jet & jet = * (myTagInfo.jet());
  PFJetRef myJet = myTagInfo.pfjetRef();

  //Takes the LeadChargedHadron
  float z_PV = 0;
  TrackRef myLeadTk;
  PFCandidateRef myLeadPFChargedHadronCand = (myTagInfo).leadPFChargedHadrCand(MatchingConeSize_, LeadCand_minPt_);
  if (myLeadPFChargedHadronCand->blockRef()->elements().size()!=0){
    for (OwnVector<PFBlockElement>::const_iterator iPFBlock=myLeadPFChargedHadronCand->blockRef()->elements().begin();
	 iPFBlock!=myLeadPFChargedHadronCand->blockRef()->elements().end();iPFBlock++){
      if ((*iPFBlock).type()==1 &&
	  ROOT::Math::VectorUtil::DeltaR(myLeadPFChargedHadronCand->momentum(),(*iPFBlock).trackRef()->momentum())<0.001){
	myLeadTk =(*iPFBlock).trackRef();
	z_PV = myLeadTk->dz();
      }
    }
  }
  //  cout <<"Vertex "<<myLeadTk->vz()<<endl;
  //  cout <<"Z_imp "<<myLeadTk->dz()<<endl;
  math::XYZPoint  vtx = math::XYZPoint( 0, 0, z_PV );
  //create the Tau
  
  Tau myTau(myJet->charge(),myJet->p4(),vtx);

  //setting the mass
  myTau.setInvariantMass(myTau.mass());

  //Setting the EmOverHcal energy
  myTau.setEmOverHadronEnergy((myJet->chargedEmEnergy() + myJet->neutralEmEnergy())/ (myJet->chargedHadronEnergy()  +myJet->neutralHadronEnergy()) );

  //Setting the LeadChargedHadrons
  PFCandidateRef leadPFChargedHadron = myTagInfo.leadPFChargedHadrCand(MatchingConeSize_, LeadCand_minPt_);
  math::XYZVector leadPFCand_XYZVector=(*leadPFChargedHadron).momentum() ;
  myTau.setLeadingChargedHadron(leadPFChargedHadron);

  //Setting the HCalEnergy from the LeadHadron
  myTau.setMaximumHcalTowerEnergy(leadPFChargedHadron->energy());

  //Setting the ChargedHadrons
  PFCandidateRefVector myChargedHadrons = myTagInfo.PFChargedHadrCands();
  myTau.setSelectedChargedHadrons(myChargedHadrons);  

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

  //Setting the IsolationChargedHadrons
  PFCandidateRefVector myIsolationChargedHadrons = 
    myTagInfo.PFChargedHadrCandsInCone(leadPFCand_XYZVector, TrackerIsolConeSize_,Candidates_minPt_);
  myTau.setIsolationChargedHadrons(myIsolationChargedHadrons);  

  //Setting the NeutralHadrons
  PFCandidateRefVector myNeutralHadrons = myTagInfo.PFNeutrHadrCands();
  myTau.setSelectedNeutralHadrons(myNeutralHadrons);  

  //Setting the SignalNeutralHadrons
  PFCandidateRefVector mySignalNeutralHadrons = 
    myTagInfo.PFNeutrHadrCandsInCone(leadPFCand_XYZVector, ECALSignalConeSize_ ,Candidates_minPt_);
  myTau.setSignalNeutralHadrons(mySignalNeutralHadrons);  

  //Setting the IsolationNeutralHadrons
  PFCandidateRefVector myIsolationNeutralHadrons = 
    myTagInfo.PFNeutrHadrCandsInCone(leadPFCand_XYZVector, ECALIsolConeSize_,Candidates_minPt_);
  myTau.setIsolationNeutralHadrons(myIsolationNeutralHadrons);  
  
  //Setting the GammaCandidates
  PFCandidateRefVector myGammaCandidates = myTagInfo.PFGammaCands();
  myTau.setSelectedGammaCandidates(myGammaCandidates);  

  //Setting the SignalGammaCandidates
  PFCandidateRefVector mySignalGammaCandidates = 
    myTagInfo.PFGammaCandsInCone(leadPFCand_XYZVector, ECALSignalConeSize_ ,Candidates_minPt_);
  myTau.setSignalGammaCandidates(mySignalGammaCandidates);  

  //Setting the IsolationGammaCandidates
  PFCandidateRefVector myIsolationGammaCandidates = 
    myTagInfo.PFGammaCandsInCone(leadPFCand_XYZVector, ECALIsolConeSize_ ,Candidates_minPt_);
  myTau.setIsolationGammaCandidates(myIsolationGammaCandidates);  


 
    
  return myTau;
  
}
