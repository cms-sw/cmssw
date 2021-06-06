#ifndef RecoEgamma_EgammaElectronAlgos_ConversionFinder_h
#define RecoEgamma_EgammaElectronAlgos_ConversionFinder_h

/** \class reco:: ConversionFinder.h RecoEgamma/EgammaElectronAlgos/interface/ConversionFinder.h
  *
  * Conversion finding and rejection code
  * Uses simple geometric methods to determine whether or not the
  * electron did indeed come from a conversion
  * \author Puneeth Kalavase, University Of California, Santa Barbara
  *
  *
  */

#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "CommonTools/Utils/interface/KinematicTables.h"
#include "CommonTools/Utils/interface/TrackSpecificColumns.h"

#include <iostream>
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
    const float dist = -9999.;
    const float dcot = -9999.;
    const float radiusOfConversion = -9999.;
    // if the partner track is found in the  GSF track collection,
    // this is a ref to the GSF partner track
    const std::optional<int> conversionPartnerCtfTkIdx = std::nullopt;
    // if the partner track is found in the  CTF track collection,
    // this is a ref to the CTF partner track
    const std::optional<int> conversionPartnerGsfTkIdx = std::nullopt;
    const int deltaMissingHits = -9999;
    const int flag = -9999;

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

  std::vector<ConversionInfo> findConversions(const reco::GsfElectronCore& gsfElectron,
                                              TrackTableView ctfTable,
                                              TrackTableView gsfTable,
                                              float bFieldAtOrigin,
                                              float minFracSharedHits);

  //places different cuts on dist, dcot, delmissing hits and arbitration based on R = sqrt(dist*dist + dcot*dcot)
  ConversionInfo findBestConversionMatch(const std::vector<ConversionInfo>& v_convCandidates);

  // returns the "best" conversion,
  // bField has to be supplied in Tesla
  inline ConversionInfo findConversion(const reco::GsfElectronCore& gsfElectron,
                                       TrackTableView ctfTable,
                                       TrackTableView gsfTable,
                                       float bFieldAtOrigin,
                                       float minFracSharedHits = 0.45f) {
    return findBestConversionMatch(findConversions(gsfElectron, ctfTable, gsfTable, bFieldAtOrigin, minFracSharedHits));
  }

}  // namespace egamma::conv

#endif
