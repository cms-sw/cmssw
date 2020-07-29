#ifndef RecoEgamma_EgammaElectronAlgos__h
#define RecoEgamma_EgammaElectronAlgos__h

/** \class reco:: .h RecoEgamma/EgammaElectronAlgos/interface/.h
  *
  * Conversion finding and rejection code
  * Uses simple geometric methods to determine whether or not the
  * electron did indeed come from a conversion
  * \author Puneeth Kalavase, University Of California, Santa Barbara
  *
  *
  */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "CommonTools/Utils/interface/KinematicTables.h"
#include "CommonTools/Utils/interface/TrackSpecificColumns.h"

#include <optional>

/*
   Class Looks for oppositely charged track in the
   track collection with the minimum delta cot(theta) between the track
   and the electron's CTF track (if it doesn't exist, we use the electron's
   GSF track). Calculate the dist, dcot, point of conversion and the
   radius of conversion for this pair and fill the ConversionInfo
*/

namespace egamma::conv {

  struct ConversionInfo {
    const float dist;
    const float dcot;
    const float radiusOfConversion;
    // if the partner track is found in the  GSF track collection,
    // this is a ref to the GSF partner track
    const std::optional<int> conversionPartnerCtfTkIdx;
    // if the partner track is found in the  CTF track collection,
    // this is a ref to the CTF partner track
    const std::optional<int> conversionPartnerGsfTkIdx;
    const int deltaMissingHits;
    const int flag;

    // flag 0: Partner track found in the CTF collection using the electron's CTF track
    // flag 1: Partner track found in the CTF collection using the electron's GSF track
    // flag 2: Partner track found in the GSF collection using the electron's CTF track
    // flag 3: Partner track found in the GSF collection using the electron's GSF track
  };

  using TrackTableSpecificColumns = std::tuple<edm::soa::col::Pz,
                                               edm::soa::col::PtError,
                                               edm::soa::col::MissingInnerHits,
                                               edm::soa::col::NumberOfValidHits,
                                               edm::soa::col::Charge,
                                               edm::soa::col::D0>;
  using TrackTable = edm::soa::AddColumns<edm::soa::PtEtaPhiTable, TrackTableSpecificColumns>::type;
  using TrackTableView = edm::soa::ViewFromTable_t<TrackTable>;
  using TrackRowView = TrackTable::const_iterator::value_type;

  // returns the "best" conversion,
  // bField has to be supplied in Tesla
  ConversionInfo findConversion(const reco::GsfElectronCore&,
                                TrackTableView ctfTable,
                                TrackTableView gsfTable,
                                float bFieldAtOrigin,
                                float minFracSharedHits = 0.45);

}  // namespace egamma::conv

#endif
