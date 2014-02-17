//--------------------------------------------------------------------------------------------------
// $Id: ConversionTools.h,v 1.2 2011/05/10 19:27:22 eulisse Exp $
//
// ConversionTools
//
// Utility to match electrons/photons/tracks to conversions and perform various conversion
// selection criteria.
//
// Matching to photons is by geometrical match with the SuperCluster (defined by angle between
// conversion momentum and vector joining conversion vertex to SuperCluster position)
//
// Matching to tracks and electrons is by reference.
//
// Also implemented here is a "conversion-safe electron veto" for photons through the
// matchedPromptElectron and hasMatchedPromptElectron functions
// 
//
// Authors: J.Bendavid
//--------------------------------------------------------------------------------------------------

#ifndef EgammaTools_ConversionTools_h
#define EgammaTools_ConversionTools_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
class ConversionTools
{
  public:
    ConversionTools() {}
                                                      
                                                
    static bool                        isGoodConversion(const reco::Conversion &conv, const math::XYZPoint &beamspot, float lxyMin=2.0, float probMin=1e-6, unsigned int nHitsBeforeVtxMax=1);
    
    static bool                        matchesConversion(const reco::GsfElectron &ele, const reco::Conversion &conv, bool allowCkfMatch=true);
    static bool                        matchesConversion(const reco::SuperCluster &sc, const reco::Conversion &conv, float dRMax = 0.1, float dEtaMax = 999., float dPhiMax = 999.);
    static bool                        matchesConversion(const edm::RefToBase<reco::Track> &trk, const reco::Conversion &conv);
    static bool                        matchesConversion(const reco::TrackRef &trk, const reco::Conversion &conv);
    static bool                        matchesConversion(const reco::GsfTrackRef &trk, const reco::Conversion &conv);


    static bool                        hasMatchedConversion(const reco::GsfElectron &ele,
                                                  const edm::Handle<reco::ConversionCollection> &convCol, const math::XYZPoint &beamspot, bool allowCkfMatch=true, float lxyMin=2.0, float probMin=1e-6, unsigned int nHitsBeforeVtxMax=0);

    static bool                        hasMatchedConversion(const reco::TrackRef &trk,
                                                  const edm::Handle<reco::ConversionCollection> &convCol, const math::XYZPoint &beamspot, float lxyMin=2.0, float probMin=1e-6, unsigned int nHitsBeforeVtxMax=1);

    static bool                        hasMatchedConversion(const reco::SuperCluster &sc,
                                                  const edm::Handle<reco::ConversionCollection> &convCol, const math::XYZPoint &beamspot, float dRMax = 0.1, float dEtaMax = 999., float dPhiMax = 999., float lxyMin=2.0, float probMin=1e-6, unsigned int nHitsBeforeVtxMax=1);


    static reco::ConversionRef         matchedConversion(const reco::GsfElectron &ele,
                                                  const edm::Handle<reco::ConversionCollection> &convCol, const math::XYZPoint &beamspot, bool allowCkfMatch=true, float lxyMin=2.0, float probMin=1e-6, unsigned int nHitsBeforeVtxMax=0);

    static reco::ConversionRef         matchedConversion(const reco::TrackRef &trk,
                                                  const edm::Handle<reco::ConversionCollection> &convCol, const math::XYZPoint &beamspot, float lxyMin=2.0, float probMin=1e-6, unsigned int nHitsBeforeVtxMax=1);

    static reco::ConversionRef         matchedConversion(const reco::SuperCluster &sc,
                                                  const edm::Handle<reco::ConversionCollection> &convCol, const math::XYZPoint &beamspot, float dRMax = 0.1, float dEtaMax = 999., float dPhiMax = 999., float lxyMin=2.0, float probMin=1e-6, unsigned int nHitsBeforeVtxMax=1);

    static bool                        hasMatchedPromptElectron(const reco::SuperClusterRef &sc, const edm::Handle<reco::GsfElectronCollection> &eleCol,
                                                  const edm::Handle<reco::ConversionCollection> &convCol, const math::XYZPoint &beamspot, float lxyMin=2.0, float probMin=1e-6, unsigned int nHitsBeforeVtxMax=0);


    static reco::GsfElectronRef        matchedPromptElectron(const reco::SuperClusterRef &sc, const edm::Handle<reco::GsfElectronCollection> &eleCol,
                                                  const edm::Handle<reco::ConversionCollection> &convCol, const math::XYZPoint &beamspot, float lxyMin=2.0, float probMin=1e-6, unsigned int nHitsBeforeVtxMax=0);

};
#endif
