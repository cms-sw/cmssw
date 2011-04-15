// $Id: ConversionTools.cc,v 1.1 2010/06/08 20:17:28 bendavid Exp $

#include <TMath.h>
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

using namespace edm;
using namespace reco;


//--------------------------------------------------------------------------------------------------
bool ConversionTools::isGoodConversion(const Conversion &conv, const math::XYZPoint &beamspot, float lxyMin, float probMin, uint nHitsBeforeVtxMax)
{
  
  //Check if a given conversion candidate passes the conversion selection cuts
  
  const reco::Vertex &vtx = conv.conversionVertex();

  //vertex validity
  if (!vtx.isValid()) return false;

  //fit probability
  if (TMath::Prob( vtx.chi2(),  vtx.ndof() )>probMin) return false;

  //compute transverse decay length
  math::XYZVector mom(conv.refittedPairMomentum()); 
  double dbsx = vtx.x() - beamspot.x();
  double dbsy = vtx.y() - beamspot.y();
  double lxy = (mom.x()*dbsx + mom.y()*dbsy)/mom.rho();

  //transverse decay length  
  if ( lxy<lxyMin )
    return false;
    
  //loop through daughters to check nhitsbeforevtx
  for (std::vector<uint8_t>::const_iterator it = conv.nHitsBeforeVtx().begin(); it!=conv.nHitsBeforeVtx().end(); ++it) {
    if ( (*it)>nHitsBeforeVtxMax ) return false;
  }
  
  return true;
}

//--------------------------------------------------------------------------------------------------
bool ConversionTools::matchesConversion(const reco::GsfElectron &ele, const reco::Conversion &conv, bool allowCkfMatch)
{

  edm::RefToBase<reco::Track> gsfref(ele.gsfTrack());
  edm::RefToBase<reco::Track> ckfref(ele.closestCtfTrackRef());

  const std::vector<edm::RefToBase<reco::Track> > &convTracks = conv.tracks();
  for (std::vector<edm::RefToBase<reco::Track> >::const_iterator it=convTracks.begin(); it!=convTracks.end(); ++it) {
    if ( gsfref.isNonnull() && gsfref==*it) return true;
    else if ( allowCkfMatch && ckfref.isNonnull() && ckfref==*it ) return true;
  }

  return false;
}

//--------------------------------------------------------------------------------------------------
bool ConversionTools::matchesConversion(const reco::SuperCluster &sc, const reco::Conversion &conv, float dRMax, float dEtaMax, float dPhiMax) {
  math::XYZVector mom(conv.refittedPairMomentum());
  
  math::XYZPoint scpos(sc.position());
  math::XYZPoint cvtx(conv.conversionVertex().position());


  math::XYZVector cscvector = scpos - cvtx;
  float dR = reco::deltaR(mom,cscvector);
  float dEta = mom.eta() - cscvector.eta();
  float dPhi = reco::deltaPhi(mom.phi(),cscvector.phi());

  if (dR>dRMax) return false;
  if (dEta>dEtaMax) return false;
  if (dPhi>dPhiMax) return false;

  return true;

}


//--------------------------------------------------------------------------------------------------
bool ConversionTools::matchesConversion(const edm::RefToBase<reco::Track> &trk, const reco::Conversion &conv)
{

  if (trk.isNull()) return false;

  const std::vector<edm::RefToBase<reco::Track> > &convTracks = conv.tracks();
  for (std::vector<edm::RefToBase<reco::Track> >::const_iterator it=convTracks.begin(); it!=convTracks.end(); ++it) {
    if (trk==*it) return true;
  }

  return false;
}

//--------------------------------------------------------------------------------------------------
bool ConversionTools::matchesConversion(const reco::TrackRef &trk, const reco::Conversion &conv)
{
  return matchesConversion(edm::RefToBase<reco::Track>(trk), conv);
}

//--------------------------------------------------------------------------------------------------
bool ConversionTools::matchesConversion(const reco::GsfTrackRef &trk, const reco::Conversion &conv)
{
  return matchesConversion(edm::RefToBase<reco::Track>(trk), conv);
}


//--------------------------------------------------------------------------------------------------
bool ConversionTools::hasMatchedConversion(const reco::GsfElectron &ele,
                                                  const edm::Handle<reco::ConversionCollection> &convCol,
                                                  const math::XYZPoint &beamspot, bool allowCkfMatch, float lxyMin, float probMin, uint nHitsBeforeVtxMax)
{
  //check if a given electron candidate matches to at least one conversion candidate in the
  //collection which also passes the selection cuts, optionally match with the closestckf track in
  //in addition to just the gsf track (enabled in default arguments)
  
  for (ConversionCollection::const_iterator it = convCol->begin(); it!=convCol->end(); ++it) {
    if (!matchesConversion(ele, *it, allowCkfMatch)) continue;
    if (!isGoodConversion(*it,beamspot,lxyMin,probMin,nHitsBeforeVtxMax)) continue;
   
    return true;
  }
  
  return false;
  
}

//--------------------------------------------------------------------------------------------------
bool ConversionTools::hasMatchedConversion(const reco::TrackRef &trk,
                                                  const edm::Handle<reco::ConversionCollection> &convCol,
                                                  const math::XYZPoint &beamspot, float lxyMin, float probMin, uint nHitsBeforeVtxMax)
{
  //check if a given track matches to at least one conversion candidate in the
  //collection which also passes the selection cuts
  
  if (trk.isNull()) return false;
  
  for (ConversionCollection::const_iterator it = convCol->begin(); it!=convCol->end(); ++it) {
    if (!matchesConversion(trk, *it)) continue;
    if (!isGoodConversion(*it,beamspot,lxyMin,probMin,nHitsBeforeVtxMax)) continue;
   
    return true;
  }
  
  return false;
  
}

//--------------------------------------------------------------------------------------------------
bool ConversionTools::hasMatchedConversion(const reco::SuperCluster &sc,
                  const edm::Handle<reco::ConversionCollection> &convCol,
                  const math::XYZPoint &beamspot, float dRMax, float dEtaMax, float dPhiMax, float lxyMin, float probMin, uint nHitsBeforeVtxMax)
{
  ConversionRef match;
  
  for (ConversionCollection::const_iterator it = convCol->begin(); it!=convCol->end(); ++it) {
    if (!matchesConversion(sc, *it)) continue;
    if (!isGoodConversion(*it,beamspot,lxyMin,probMin,nHitsBeforeVtxMax)) continue;
   
    return true;
  }
  
  return false;

}


//--------------------------------------------------------------------------------------------------
reco::ConversionRef ConversionTools::matchedConversion(const reco::GsfElectron &ele,
                                                  const edm::Handle<reco::ConversionCollection> &convCol,
                                                  const math::XYZPoint &beamspot, bool allowCkfMatch, float lxyMin, float probMin, uint nHitsBeforeVtxMax)
{
  //check if a given electron candidate matches to at least one conversion candidate in the
  //collection which also passes the selection cuts, optionally match with the closestckf track in
  //in addition to just the gsf track (enabled in default arguments)
  
  ConversionRef match;
  
  double minRho = 999.;
  for (ConversionCollection::const_iterator it = convCol->begin(); it!=convCol->end(); ++it) {
    float rho = it->conversionVertex().position().rho();
    if (rho>minRho) continue;
    if (!matchesConversion(ele, *it, allowCkfMatch)) continue;
    if (!isGoodConversion(*it,beamspot,lxyMin,probMin,nHitsBeforeVtxMax)) continue;
   
    minRho = rho;
    match = ConversionRef(convCol,it-convCol->begin());
  }
  
  return match;
  
}

//--------------------------------------------------------------------------------------------------
reco::ConversionRef ConversionTools::matchedConversion(const reco::TrackRef &trk,
                                                  const edm::Handle<reco::ConversionCollection> &convCol,
                                                  const math::XYZPoint &beamspot, float lxyMin, float probMin, uint nHitsBeforeVtxMax)
{
  //check if a given track matches to at least one conversion candidate in the
  //collection which also passes the selection cuts
  
  ConversionRef match;

  if (trk.isNull()) return match;
  
  double minRho = 999.;
  for (ConversionCollection::const_iterator it = convCol->begin(); it!=convCol->end(); ++it) {
    float rho = it->conversionVertex().position().rho();
    if (rho>minRho) continue;
    if (!matchesConversion(trk, *it)) continue;
    if (!isGoodConversion(*it,beamspot,lxyMin,probMin,nHitsBeforeVtxMax)) continue;
   
    minRho = rho;
    match = ConversionRef(convCol,it-convCol->begin());
  }
  
  return match;
  
}

//--------------------------------------------------------------------------------------------------
reco::ConversionRef ConversionTools::matchedConversion(const reco::SuperCluster &sc,
                  const edm::Handle<reco::ConversionCollection> &convCol,
                  const math::XYZPoint &beamspot, float dRMax, float dEtaMax, float dPhiMax, float lxyMin, float probMin, uint nHitsBeforeVtxMax)
{
  ConversionRef match;
  
  double minRho = 999.;
  for (ConversionCollection::const_iterator it = convCol->begin(); it!=convCol->end(); ++it) {
    float rho = it->conversionVertex().position().rho();
    if (rho>minRho) continue;
    if (!matchesConversion(sc, *it, dRMax,dEtaMax,dPhiMax)) continue;
    if (!isGoodConversion(*it,beamspot,lxyMin,probMin,nHitsBeforeVtxMax)) continue;
   
    minRho = rho;
    match = ConversionRef(convCol,it-convCol->begin());
  }
  
  return match;

}

//--------------------------------------------------------------------------------------------------
bool ConversionTools::hasMatchedPromptElectron(const reco::SuperClusterRef &sc, const edm::Handle<reco::GsfElectronCollection> &eleCol,
                   const edm::Handle<reco::ConversionCollection> &convCol, const math::XYZPoint &beamspot, float lxyMin, float probMin, uint nHitsBeforeVtxMax)
{

  if (sc.isNull()) return false;
  
  for (GsfElectronCollection::const_iterator it = eleCol->begin(); it!=eleCol->end(); ++it) {
    //match electron to supercluster
    if (it->superCluster()!=sc) continue;

    //check expected inner hits
    if (it->gsfTrack()->trackerExpectedHitsInner().numberOfHits()>0) continue;

    //check if electron is matching to a conversion
    if (hasMatchedConversion(*it,convCol,beamspot,lxyMin,probMin,nHitsBeforeVtxMax)) continue;
   
   
    return true;
  }
  
  return false;


}


//--------------------------------------------------------------------------------------------------
reco::GsfElectronRef ConversionTools::matchedPromptElectron(const reco::SuperClusterRef &sc, const edm::Handle<reco::GsfElectronCollection> &eleCol,
                   const edm::Handle<reco::ConversionCollection> &convCol, const math::XYZPoint &beamspot, float lxyMin, float probMin, uint nHitsBeforeVtxMax)
{

  GsfElectronRef match;

  if (sc.isNull()) return match;
  
  for (GsfElectronCollection::const_iterator it = eleCol->begin(); it!=eleCol->end(); ++it) {
    //match electron to supercluster
    if (it->superCluster()!=sc) continue;

    //check expected inner hits
    if (it->gsfTrack()->trackerExpectedHitsInner().numberOfHits()>0) continue;

    //check if electron is matching to a conversion
    if (hasMatchedConversion(*it,convCol,beamspot,lxyMin,probMin,nHitsBeforeVtxMax)) continue;
   
   
    match = GsfElectronRef(eleCol,it-eleCol->begin());
  }
  
  return match;


}
