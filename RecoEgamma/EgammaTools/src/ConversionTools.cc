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
bool ConversionTools::isGoodConversion(const Conversion &conv, const math::XYZPoint &beamspot, float lxyMin, float probMin, unsigned int nHitsBeforeVtxMax)
{
  
  //Check if a given conversion candidate passes the conversion selection cuts
  
  const reco::Vertex &vtx = conv.conversionVertex();

  //vertex validity
  if (!vtx.isValid()) return false;

  //fit probability
  if (TMath::Prob( vtx.chi2(),  vtx.ndof() )<probMin) return false;

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
bool ConversionTools::matchesConversion(const reco::GsfElectron &ele, const reco::Conversion &conv, bool allowCkfMatch, bool allowAmbiguousGsfMatch)
{

  //check if a given GsfElectron matches a given conversion (no quality cuts applied)
  //matching is always attempted through the gsf track ref, and optionally attempted through the
  //closest ctf track ref

  const std::vector<edm::RefToBase<reco::Track> > &convTracks = conv.tracks();
  for (std::vector<edm::RefToBase<reco::Track> >::const_iterator it=convTracks.begin(); it!=convTracks.end(); ++it) {
    if ( ele.reco::GsfElectron::gsfTrack().isNonnull() && ele.reco::GsfElectron::gsfTrack().id()==it->id() && ele.reco::GsfElectron::gsfTrack().key()==it->key()) return true;
    else if ( allowCkfMatch && ele.reco::GsfElectron::closestCtfTrackRef().isNonnull() && ele.reco::GsfElectron::closestCtfTrackRef().id()==it->id() && ele.reco::GsfElectron::closestCtfTrackRef().key()==it->key() ) return true;
    if (allowAmbiguousGsfMatch) {
      for (reco::GsfTrackRefVector::const_iterator tk = ele.ambiguousGsfTracksBegin(); tk!=ele.ambiguousGsfTracksEnd(); ++tk) {
        if (tk->isNonnull() && tk->id()==it->id() && tk->key()==it->key()) return true;
      }
    }
  }

  return false;
}

//--------------------------------------------------------------------------------------------------
bool ConversionTools::matchesConversion(const reco::SuperCluster &sc, const reco::Conversion &conv, float dRMax, float dEtaMax, float dPhiMax) {

  //check if a given SuperCluster matches a given conversion (no quality cuts applied)
  //matching is geometric between conversion momentum and vector joining conversion vertex
  //to supercluster position


  math::XYZVector mom(conv.refittedPairMomentum());
  
  const math::XYZPoint& scpos(sc.position());
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

  //check if given track matches given conversion (matching by ref)

  if (trk.isNull()) return false;

  const std::vector<edm::RefToBase<reco::Track> > &convTracks = conv.tracks();
  for (std::vector<edm::RefToBase<reco::Track> >::const_iterator it=convTracks.begin(); it!=convTracks.end(); ++it) {
    if (trk.id()==it->id() && trk.key()==it->key()) return true;
  }

  return false;
}

//--------------------------------------------------------------------------------------------------
bool ConversionTools::matchesConversion(const reco::TrackRef &trk, const reco::Conversion &conv)
{

  //check if given track matches given conversion (matching by ref)

  if (trk.isNull()) return false;

  const std::vector<edm::RefToBase<reco::Track> > &convTracks = conv.tracks();
  for (std::vector<edm::RefToBase<reco::Track> >::const_iterator it=convTracks.begin(); it!=convTracks.end(); ++it) {
    if (trk.id()==it->id() && trk.key()==it->key()) return true;
  }

  return false;
}

//--------------------------------------------------------------------------------------------------
bool ConversionTools::matchesConversion(const reco::GsfTrackRef &trk, const reco::Conversion &conv)
{

  //check if given track matches given conversion (matching by ref)

  if (trk.isNull()) return false;

  const std::vector<edm::RefToBase<reco::Track> > &convTracks = conv.tracks();
  for (std::vector<edm::RefToBase<reco::Track> >::const_iterator it=convTracks.begin(); it!=convTracks.end(); ++it) {
    if (trk.id()==it->id() && trk.key()==it->key()) return true;
  }

  return false;
}


//--------------------------------------------------------------------------------------------------
bool ConversionTools::hasMatchedConversion(const reco::GsfElectron &ele,
                                                  const reco::ConversionCollection &convCol,
                                                  const math::XYZPoint &beamspot, bool allowCkfMatch, float lxyMin, float probMin, unsigned int nHitsBeforeVtxMax)
{
  //check if a given electron candidate matches to at least one conversion candidate in the
  //collection which also passes the selection cuts, optionally match with the closestckf track in
  //in addition to just the gsf track (enabled in default arguments)
  
  for(auto const& it : convCol) {
    if (!matchesConversion(ele, it, allowCkfMatch)) continue;
    if (!isGoodConversion(it,beamspot,lxyMin,probMin,nHitsBeforeVtxMax)) continue;
   
    return true;
  }
  
  return false;
  
}

//--------------------------------------------------------------------------------------------------
bool ConversionTools::hasMatchedConversion(const reco::TrackRef &trk,
                                                  const reco::ConversionCollection &convCol,
                                                  const math::XYZPoint &beamspot, float lxyMin, float probMin, unsigned int nHitsBeforeVtxMax)
{
  //check if a given track matches to at least one conversion candidate in the
  //collection which also passes the selection cuts
  
  if (trk.isNull()) return false;
  
  for(auto const& it : convCol) {
    if (!matchesConversion(trk, it)) continue;
    if (!isGoodConversion(it,beamspot,lxyMin,probMin,nHitsBeforeVtxMax)) continue;
   
    return true;
  }
  
  return false;
  
}

//--------------------------------------------------------------------------------------------------
bool ConversionTools::hasMatchedConversion(const reco::SuperCluster &sc,
                  const reco::ConversionCollection &convCol,
                  const math::XYZPoint &beamspot, float dRMax, float dEtaMax, float dPhiMax, float lxyMin, float probMin, unsigned int nHitsBeforeVtxMax)
{
  
  //check if a given SuperCluster matches to at least one conversion candidate in the
  //collection which also passes the selection cuts

  for(auto const& it : convCol) {
    if (!matchesConversion(sc, it)) continue;
    if (!isGoodConversion(it,beamspot,lxyMin,probMin,nHitsBeforeVtxMax)) continue;
   
    return true;
  }
  
  return false;

}


//--------------------------------------------------------------------------------------------------
reco::Conversion const* ConversionTools::matchedConversion(const reco::GsfElectron &ele,
                                                  const reco::ConversionCollection &convCol,
                                                  const math::XYZPoint &beamspot, bool allowCkfMatch, float lxyMin, float probMin, unsigned int nHitsBeforeVtxMax)
{
  //check if a given electron candidate matches to at least one conversion candidate in the
  //collection which also passes the selection cuts, optionally match with the closestckf track in
  //in addition to just the gsf track (enabled in default arguments)
  //If multiple conversions are found, returned reference corresponds to minimum
  //conversion radius
  
  reco::Conversion const* match = nullptr;
  
  double minRho = 999.;
  for(auto const& it : convCol) {
    float rho = it.conversionVertex().position().rho();
    if (rho>minRho) continue;
    if (!matchesConversion(ele, it, allowCkfMatch)) continue;
    if (!isGoodConversion(it,beamspot,lxyMin,probMin,nHitsBeforeVtxMax)) continue;
   
    minRho = rho;
    match = &it;
  }
  
  return match;
  
}

//--------------------------------------------------------------------------------------------------
reco::Conversion const* ConversionTools::matchedConversion(const reco::TrackRef &trk,
                                                  const reco::ConversionCollection &convCol,
                                                  const math::XYZPoint &beamspot, float lxyMin, float probMin, unsigned int nHitsBeforeVtxMax)
{
  //check if a given track matches to at least one conversion candidate in the
  //collection which also passes the selection cuts
  //If multiple conversions are found, returned reference corresponds to minimum
  //conversion radius
  
  reco::Conversion const* match = nullptr;

  if (trk.isNull()) return match;
  
  double minRho = 999.;
  for(auto const& it : convCol) {
    float rho = it.conversionVertex().position().rho();
    if (rho>minRho) continue;
    if (!matchesConversion(trk, it)) continue;
    if (!isGoodConversion(it,beamspot,lxyMin,probMin,nHitsBeforeVtxMax)) continue;
   
    minRho = rho;
    match = &it;
  }
  
  return match;
  
}

//--------------------------------------------------------------------------------------------------
reco::Conversion const* ConversionTools::matchedConversion(const reco::SuperCluster &sc,
                  const reco::ConversionCollection &convCol,
                  const math::XYZPoint &beamspot, float dRMax, float dEtaMax, float dPhiMax, float lxyMin, float probMin, unsigned int nHitsBeforeVtxMax)
{

  //check if a given SuperCluster matches to at least one conversion candidate in the
  //collection which also passes the selection cuts
  //If multiple conversions are found, returned reference corresponds to minimum
  //conversion radius

  reco::Conversion const* match = nullptr;
  
  double minRho = 999.;
  for(auto const& it : convCol) {
    float rho = it.conversionVertex().position().rho();
    if (rho>minRho) continue;
    if (!matchesConversion(sc, it, dRMax,dEtaMax,dPhiMax)) continue;
    if (!isGoodConversion(it,beamspot,lxyMin,probMin,nHitsBeforeVtxMax)) continue;
   
    minRho = rho;
    match = &it;
  }
  
  return match;

}

//--------------------------------------------------------------------------------------------------
bool ConversionTools::hasMatchedPromptElectron(const reco::SuperClusterRef &sc, const reco::GsfElectronCollection &eleCol,
                   const reco::ConversionCollection &convCol, const math::XYZPoint &beamspot, bool allowCkfMatch, float lxyMin, float probMin, unsigned int nHitsBeforeVtxMax)
{

  return !(matchedPromptElectron(sc, eleCol, convCol, beamspot,
              allowCkfMatch, lxyMin, probMin, nHitsBeforeVtxMax) == nullptr);
}


//--------------------------------------------------------------------------------------------------
reco::GsfElectron const* ConversionTools::matchedPromptElectron(const reco::SuperClusterRef &sc, const reco::GsfElectronCollection &eleCol,
                   const reco::ConversionCollection &convCol, const math::XYZPoint &beamspot, bool allowCkfMatch, float lxyMin, float probMin, unsigned int nHitsBeforeVtxMax)
{

  //check if a given SuperCluster matches to at least one GsfElectron having zero expected inner hits
  //and not matching any conversion in the collection passing the quality cuts

  reco::GsfElectron const* match = nullptr;

  if (sc.isNull()) return match;
  
  for(auto const& it : eleCol) {
    //match electron to supercluster
    if (it.superCluster()!=sc) continue;

    //check expected inner hits
    if (it.gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) > 0) continue;

    //check if electron is matching to a conversion
    if (hasMatchedConversion(it,convCol,beamspot,allowCkfMatch,lxyMin,probMin,nHitsBeforeVtxMax)) continue;
   
   
    match = &it;
  }
  
  return match;


}
