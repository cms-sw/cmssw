#include "RecoTauTag/RecoTau/interface/PFRecoTauTagInfoAlgorithm.h"

PFTauTagInfo PFRecoTauTagInfoAlgorithm::tag(const PFJetRef& thePFJet,const TrackRefVector& theTracks,const Vertex& thePV){
  PFTauTagInfo resultExtended;
  resultExtended.setpfjetRef(thePFJet);

  math::XYZTLorentzVector alternatLorentzVect;
  alternatLorentzVect.SetPx(0.);
  alternatLorentzVect.SetPy(0.);
  alternatLorentzVect.SetPz(0.);
  alternatLorentzVect.SetE(0.);
  for (PFCandidateRefVector::const_iterator iGammaCand=resultExtended.PFGammaCands().begin();iGammaCand!=resultExtended.PFGammaCands().end();iGammaCand++) alternatLorentzVect+=(**iGammaCand).p4();
  for (PFCandidateRefVector::const_iterator iChargedHadrCand=resultExtended.PFChargedHadrCands().begin();iChargedHadrCand!=resultExtended.PFChargedHadrCands().end();iChargedHadrCand++) alternatLorentzVect+=(**iChargedHadrCand).p4();  
  resultExtended.setalternatLorentzVect(alternatLorentzVect);

  PFCandidateRefVector thePFCands;
  for (CandidateBaseRefVector ::const_iterator iConstit=(*thePFJet).getJetConstituents().begin();iConstit!=(*thePFJet).getJetConstituents().end();iConstit++) 
    thePFCands.push_back((*iConstit).castTo<PFCandidateRef>());

  PFCandidateRefVector thefilteredPFChargedHadrCands=filteredPFChargedHadrCands(thePFCands,ChargedHadrCand_tkminPt_,ChargedHadrCand_tkminPixelHitsn_,ChargedHadrCand_tkminTrackerHitsn_,ChargedHadrCand_tkmaxipt_,ChargedHadrCand_tkmaxChi2_,ChargedHadrCand_tktorefpointDZ_,UsePVconstraint_,thePV.z());
  PFCandidateRefVector thefilteredPFNeutrHadrCands=filteredPFNeutrHadrCands(thePFCands,NeutrHadrCand_HcalclusminE_);
  PFCandidateRefVector thefilteredPFGammaCands=filteredPFGammaCands(thePFCands,GammaCand_EcalclusminE_);
  PFCandidateRefVector thefilteredPFCands;
  for (PFCandidateRefVector::const_iterator iPFCand=thefilteredPFChargedHadrCands.begin();iPFCand!=thefilteredPFChargedHadrCands.end();iPFCand++) thefilteredPFCands.push_back(*iPFCand);
  for (PFCandidateRefVector::const_iterator iPFCand=thefilteredPFNeutrHadrCands.begin();iPFCand!=thefilteredPFNeutrHadrCands.end();iPFCand++) thefilteredPFCands.push_back(*iPFCand);
  for (PFCandidateRefVector::const_iterator iPFCand=thefilteredPFGammaCands.begin();iPFCand!=thefilteredPFGammaCands.end();iPFCand++) thefilteredPFCands.push_back(*iPFCand);

  resultExtended.setPFChargedHadrCands(thefilteredPFChargedHadrCands);
  resultExtended.setPFNeutrHadrCands(thefilteredPFNeutrHadrCands);
  resultExtended.setPFGammaCands(thefilteredPFGammaCands);
  resultExtended.setPFCands(thefilteredPFCands);

  resultExtended.setTracks(filteredTracks(theTracks,tkminPt_,tkminPixelHitsn_,tkminTrackerHitsn_,tkmaxipt_,tkmaxChi2_,tktorefpointDZ_,UsePVconstraint_,thePV.z()));

  return resultExtended; 
}
PFCandidateRefVector PFRecoTauTagInfoAlgorithm::filteredPFChargedHadrCands(PFCandidateRefVector thePFCands,double ChargedHadrCand_tkminPt,int ChargedHadrCand_tkminPixelHitsn,int ChargedHadrCand_tkminTrackerHitsn,double ChargedHadrCand_tkmaxipt,double ChargedHadrCand_tkmaxChi2,double ChargedHadrCand_tktorefpointDZ,bool UsePVconstraint,double PVtx_Z){
  PFCandidateRefVector filteredPFChargedHadrCands;
  for(PFCandidateRefVector::const_iterator iPFCand=thePFCands.begin();iPFCand!=thePFCands.end();iPFCand++){
    if ((**iPFCand).particleId()==PFChargedHadrCand_codenumber_){
      // *** Whether the charged hadron candidate will be selected or not depends on its rec. tk properties. 
      TrackRef PFChargedHadrCand_rectk;
      if ((**iPFCand).block()->elements().size()!=0){
	for (OwnVector<PFBlockElement>::const_iterator iPFBlock=(**iPFCand).block()->elements().begin();iPFBlock!=(**iPFCand).block()->elements().end();iPFBlock++){
	  if ((*iPFBlock).type()==PFRecTrack_codenumber_ && ROOT::Math::VectorUtil::DeltaR((**iPFCand).momentum(),(*iPFBlock).trackRef()->momentum())<0.001){
	    PFChargedHadrCand_rectk=(*iPFBlock).trackRef();
	  }
	}
      }else continue;
      if (!PFChargedHadrCand_rectk)continue;
      if ((*PFChargedHadrCand_rectk).pt()>=ChargedHadrCand_tkminPt &&
	  (*PFChargedHadrCand_rectk).normalizedChi2()<=ChargedHadrCand_tkmaxChi2 &&
	  fabs((*PFChargedHadrCand_rectk).d0())<=ChargedHadrCand_tkmaxipt &&
	  (*PFChargedHadrCand_rectk).recHitsSize()>=(unsigned int)ChargedHadrCand_tkminTrackerHitsn &&
	  (*PFChargedHadrCand_rectk).hitPattern().numberOfValidPixelHits()>=ChargedHadrCand_tkminPixelHitsn){
	if (UsePVconstraint){
	  if (fabs((*PFChargedHadrCand_rectk).dz()-PVtx_Z)<=ChargedHadrCand_tktorefpointDZ){
	    filteredPFChargedHadrCands.push_back(*iPFCand);
	  }
	}else{
	  filteredPFChargedHadrCands.push_back(*iPFCand);
	}
      }
    }
  }
  return filteredPFChargedHadrCands;
}
PFCandidateRefVector PFRecoTauTagInfoAlgorithm::filteredPFNeutrHadrCands(PFCandidateRefVector thePFCands,double NeutrHadrCand_HcalclusminE){
  PFCandidateRefVector filteredPFNeutrHadrCands;
  for(PFCandidateRefVector::const_iterator iPFCand=thePFCands.begin();iPFCand!=thePFCands.end();iPFCand++){
    if ((**iPFCand).particleId()==PFNeutrHadrCand_codenumber_){
      // *** Whether the neutral hadron candidate will be selected or not depends on its rec. HCAL cluster properties. 
      if ((**iPFCand).energy()>=NeutrHadrCand_HcalclusminE){
	filteredPFNeutrHadrCands.push_back(*iPFCand);
      }
    }
  }
  return filteredPFNeutrHadrCands;
}
PFCandidateRefVector PFRecoTauTagInfoAlgorithm::filteredPFGammaCands(PFCandidateRefVector thePFCands,double GammaCand_EcalclusminE){
  PFCandidateRefVector filteredPFGammaCands;
  for(PFCandidateRefVector::const_iterator iPFCand=thePFCands.begin();iPFCand!=thePFCands.end();iPFCand++){
    if ((**iPFCand).particleId()==PFGammaCand_codenumber_){
      // *** Whether the gamma candidate will be selected or not depends on its rec. ECAL cluster properties. 
      if ((**iPFCand).energy()>=GammaCand_EcalclusminE){
	filteredPFGammaCands.push_back(*iPFCand);
      }
    }
  }
  return filteredPFGammaCands;
}
TrackRefVector PFRecoTauTagInfoAlgorithm::filteredTracks(TrackRefVector theTracks,double tkminPt,int tkminPixelHitsn,int tkminTrackerHitsn,double tkmaxipt,double tkmaxChi2,double tktorefpointDZ,bool UsePVconstraint,double PVtx_Z){
  TrackRefVector filteredTracks;
  for(TrackRefVector::const_iterator iTk=theTracks.begin();iTk!=theTracks.end();iTk++){
    if ((**iTk).pt()>=tkminPt &&
	(**iTk).normalizedChi2()<=tkmaxChi2 &&
	fabs((**iTk).d0())<=tkmaxipt &&
	(**iTk).recHitsSize()>=(unsigned int)tkminTrackerHitsn &&
	(**iTk).hitPattern().numberOfValidPixelHits()>=tkminPixelHitsn){
      if (UsePVconstraint){
	if (fabs((**iTk).dz()-PVtx_Z)<=tktorefpointDZ){
	  filteredTracks.push_back(*iTk);
	}
      }else{
	filteredTracks.push_back(*iTk);
      }
    }
  }
  return filteredTracks;
}

