#include "RecoTauTag/RecoTau/interface/CaloRecoTauAlgorithm.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <Math/GenVector/VectorUtil.h>

Tau CaloRecoTauAlgorithm::tag(const CombinedTauTagInfo& myTagInfo)
{
  //Takes the jet
  const Jet & jet = * (myTagInfo.jet());
  //Takes the LeadChargedHadron
  float z_PV = 0;
  TrackRef myLeadTk = (myTagInfo.isolatedtautaginfoRef())->leadingSignalTrack(MatchingConeSize_, LeadCand_minPt_);
  if(myLeadTk.isNonnull()){
    cout <<"Vertex "<<myLeadTk->vz()<<endl;
    cout <<"Z_imp "<<myLeadTk->dz()<<endl;
  }
  
  math::XYZPoint  vtx = math::XYZPoint( 0, 0, z_PV );
  
  //create the Tau  
  Tau myTau(jet.charge(),jet.p4(),vtx);

  //Setting the SelectedTracks
  TrackRefVector mySelectedTracks = myTagInfo.selectedTks();
  myTau.setSelectedTracks(mySelectedTracks);  
  
  //Setting the LeadingTrack
  myTau.setLeadingTrack(myLeadTk);
  if(myLeadTk.isNonnull())
    {
      //Setting the SignalTracks
      TrackRefVector signalTracks = 
	(myTagInfo.isolatedtautaginfoRef())->tracksInCone(myLeadTk->momentum(), TrackerSignalConeSize_,Tracks_minPt_ );

      myTau.setSignalTracks(signalTracks);
      
      //Setting charge
      int charge = 0.;
      for( int i=0;i<signalTracks.size();i++){
	charge = charge+ signalTracks[i]->charge();
      }

      myTau.setCharge(charge);

      //Setting the IsolationTracks
      TrackRefVector isolationTracks = 
	(myTagInfo.isolatedtautaginfoRef())->tracksInCone(myLeadTk->momentum(), TrackerIsolConeSize_,Tracks_minPt_ );
      TrackRefVector isolationBandTracks;
      for(int i=0;i<isolationTracks.size();i++){
	const math::XYZVector trackMomentum = isolationTracks[i]->momentum();
	const math::XYZVector myVector = myLeadTk->momentum();
	float deltaR = ROOT::Math::VectorUtil::DeltaR(myVector, trackMomentum);
	if(deltaR > TrackerSignalConeSize_)
	  isolationBandTracks.push_back(isolationTracks[i]);
      }

      //Setting sum of the pT of isolation Annulus charged hadrons
      float ptSum =0;
      for(int i=0;i<isolationBandTracks.size();i++){
	ptSum = ptSum + isolationBandTracks[i]->pt();
      }
      myTau.setSumPtIsolation(ptSum);
      
      //Setting sum of the E_T of isolation Annulus gamma candidates


      //setting the mass with only Signal objects:
      

      myTau.setIsolationTracks(isolationBandTracks);
      
      //Setting the max HCalEnergy

      //Setting the EmOverHcal energy
      
      //Setting the number of EcalClusters

/*
setLeadTkTIP(const Measurement1D& myIP)  { transverseIp_leadTk_ = myIP;}
setLeadTk3DIP(const Measurement1D& myIP)  {  ip3D_leadTk_=myIP;}
*/
    }

  return myTau;
  
}
