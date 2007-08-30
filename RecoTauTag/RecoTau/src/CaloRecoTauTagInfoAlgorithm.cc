#include "RecoTauTag/RecoTau/interface/CaloRecoTauTagInfoAlgorithm.h"

TauTagInfo CaloRecoTauTagInfoAlgorithm::tag(const CaloJetRef& theCaloJet,const TrackRefVector& theTracks,const Vertex& thePV){
  TauTagInfo resultExtended;
  resultExtended.setcalojetRef(theCaloJet);

  math::XYZTLorentzVector alternatLorentzVect;
  alternatLorentzVect.SetPx(0.);
  alternatLorentzVect.SetPy(0.);
  alternatLorentzVect.SetPz(0.);
  alternatLorentzVect.SetE(0.);
  resultExtended.setalternatLorentzVect(alternatLorentzVect);

  resultExtended.setTracks(filteredTracks(theTracks,tkminPt_,tkminPixelHitsn_,tkminTrackerHitsn_,tkmaxipt_,tkmaxChi2_,tktorefpointDZ_,UsePVconstraint_,thePV.z()));

  return resultExtended; 
}
TrackRefVector CaloRecoTauTagInfoAlgorithm::filteredTracks(TrackRefVector theTracks,double tkminPt,int tkminPixelHitsn,int tkminTrackerHitsn,double tkmaxipt,double tkmaxChi2,double tktorefpointDZ,bool UsePVconstraint,double PVtx_Z){
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



