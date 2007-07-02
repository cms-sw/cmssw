#include "RecoTauTag/RecoTau/interface/CaloRecoTauAlgorithm.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

Tau CaloRecoTauAlgorithm::tag(const CombinedTauTagInfo& myTagInfo)
{
  //Takes the jet
  const Jet & jet = * (myTagInfo.jet());
  //Takes the LeadChargedHadron
  float z_PV = 0;
  TrackRef myLeadTk;

  /*
    myLeadTk = 
  if(myLeadTk.isNonnull()){
    cout <<"Vertex "<<myLeadTk->vz()<<endl;
    cout <<"Z_imp "<<myLeadTk->dz()<<endl;
  }
  */
  math::XYZPoint  vtx = math::XYZPoint( 0, 0, z_PV );
  
  //create the Tau  
  Tau myTau(jet.charge(),jet.p4(),vtx);
	
    //Setting the EmOverHcal energy

    
    //Setting the SelectedTracks
    TrackRefVector mySelectedTracks = myTagInfo.selectedTks();
    myTau.setSelectedTracks(mySelectedTracks);  
    
    
    //Setting the LeadChargedHadrons
      
      //Setting the HCalEnergy from the LeadHadron

      

      //Setting the SignalTracks
      
      
      //Setting charge
      myTau.setCharge(myTagInfo.signalTks_qsum());
    
      


      //setting the mass with only Signal objects:
      
      //Setting sum of the pT of isolation Annulus charged hadrons
      
      //Setting sum of the E_T of isolation Annulus gamma candidates
    


/*
setLeadTkTIP(const Measurement1D& myIP)  { transverseIp_leadTk_ = myIP;}
setLeadTk3DIP(const Measurement1D& myIP)  {  ip3D_leadTk_=myIP;}
*/
 

  return myTau;
  
}
