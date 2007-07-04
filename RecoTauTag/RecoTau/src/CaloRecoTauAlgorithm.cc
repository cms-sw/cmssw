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
    z_PV = myLeadTk->vz();
    //    cout <<"Vertex "<<myLeadTk->vz()<<endl;
    //    cout <<"Z_imp "<<myLeadTk->dz()<<endl;
  }
  
  math::XYZPoint  vtx = math::XYZPoint( 0, 0, z_PV );
  math::XYZTLorentzVector myVec(myTagInfo.alternatrecJet_HepLV().getX(),myTagInfo.alternatrecJet_HepLV().getY(),myTagInfo.alternatrecJet_HepLV().getZ(),myTagInfo.alternatrecJet_HepLV().getT());

  //create the Tau  with the modified 4 Vector
  Tau myTau(jet.charge(),myVec,vtx);

  //Setting the mass
  myTau.setInvariantMass(myTagInfo.alternatrecJet_HepLV().m());

  //Setting the SelectedTracks
  TrackRefVector mySelectedTracks = myTagInfo.selectedTks();
  myTau.setSelectedTracks(mySelectedTracks);  
  
  //Setting the LeadingTrack
  myTau.setLeadingTrack(myLeadTk);
  if(myLeadTk.isNonnull())
    {
      //Setting the SignalTracks
      TrackRefVector signalTracks = myTagInfo.signalTks();
      myTau.setSignalTracks(signalTracks);

      myTau.setCharge(myTagInfo.signalTks_qsum());

      //Setting the IsolationTracks
      TrackRefVector isolationTracks = myTagInfo.isolTks();
	/*
	(myTagInfo.isolatedtautaginfoRef())->tracksInCone(myLeadTk->momentum(), TrackerIsolConeSize_,Tracks_minPt_ );
      TrackRefVector isolationBandTracks;
      for(int i=0;i<isolationTracks.size();i++){
	const math::XYZVector trackMomentum = isolationTracks[i]->momentum();
	const math::XYZVector myVector = myLeadTk->momentum();
	float deltaR = ROOT::Math::VectorUtil::DeltaR(myVector, trackMomentum);
	if(deltaR > TrackerSignalConeSize_)
	  isolationBandTracks.push_back(isolationTracks[i]);
      }
	*/
      myTau.setIsolationTracks(isolationTracks);
      
      //Setting sum of the pT of isolation Annulus charged hadrons
      float ptSum =0;
      for(int i=0;i<isolationTracks.size();i++){
	ptSum = ptSum + isolationTracks[i]->pt();
      }
      myTau.setSumPtIsolation(ptSum);

      
      //Setting sum of the E_T of isolation Annulus gamma candidates
      myTau.setEMIsolation(myTagInfo.isolneutralEtsum());
      
      //setting the mass with only Signal objects:
      myTau.setInvariantMass(myTagInfo.alternatrecJet_HepLV().m());
      
      //Setting the max HCalEnergy

      const CaloJet* cj = dynamic_cast<CaloJet*>(const_cast<Jet*>(&jet));
       // access jet constituents
     const std::vector<CaloTowerRef>  TauJetTowers = cj->getConstituents();
     // access towers
     double THmax = -10.; 
     for(int iTower = 0; iTower < TauJetTowers.size(); iTower++) {
       CaloTowerRef theTower = TauJetTowers[iTower];
       // select max Et HCAL tower
       if(theTower->hadEt() >= THmax) {
	 THmax    = theTower->hadEt();
       }
     }

     myTau.setMaximumHcalEnergy(THmax);
 //Setting the EmEnergyFraction

     myTau.setEmEnergyFraction(myTagInfo.neutralE_o_TksEneutralE());
          

     
      //Setting the number of EcalClusters
      myTau.setNumberOfEcalClusters(myTagInfo.neutralECALClus_number());
      
      //LeadTk TIP
      myTau.setSignificanceLeadTkTIP(myTagInfo.leadTk_signedip3D_significance());

    }

  return myTau;
  
}
