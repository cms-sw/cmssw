#include "RecoEgamma/EgammaElectronAlgos/interface/ConversionFinder.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

namespace egamma::conv {

  namespace {

    constexpr float square(float x) { return x * x; };

  }  // namespace

  // configuration parameters
  constexpr float maxRefPtErrorForKfConv = 0.05f;
  constexpr float maxRelGsfPtErrorForKfConv = 0.25f;
  constexpr float maxRelPtDiffForKfConv = 0.2f;
  constexpr float maxRelPtDiffForGsfConv = 0.25f;
  constexpr float dR2Max = square(0.5f);
  constexpr int minNumberOfValidHits = 5;
  constexpr float maxDist2Dcot2 = square(0.05f);
  constexpr int maxDeltaMissingHits = 2;
  constexpr float maxRelGsfPtError = 0.5f;
  constexpr int maxDeltaMissingHitsForKFtoKF = 3;
  constexpr float maxDistOrCotForKFtoKF = 0.02f;

  ConversionInfo getConversionInfo(reco::Track const& el_track, TrackRowView const& track, float bFieldAtOrigin);

  //-----------------------------------------------------------------------------
  std::vector<ConversionInfo> findConversions(const reco::GsfElectronCore& gsfElectron,
                                              TrackTableView ctfTable,
                                              TrackTableView gsfTable,
                                              float bFieldAtOrigin,
                                              float minFracSharedHits) {
    using namespace reco;
    using namespace std;
    using namespace edm;
    using namespace edm::soa::col;

    //get the references to the gsf and ctf tracks that are made
    //by the electron
    const reco::TrackRef eleCtfTk = gsfElectron.ctfTrack();
    const reco::GsfTrackRef& eleGsfTk = gsfElectron.gsfTrack();

    float eleGsfPt = eleGsfTk->pt();
    float eleGsfEta = eleGsfTk->eta();
    float eleGsfPhi = eleGsfTk->phi();

    const bool useEleCtfTrack = eleCtfTk.isNonnull() && gsfElectron.ctfGsfOverlap() > minFracSharedHits;

    std::optional<float> eleCtfPt = std::nullopt;
    std::optional<float> eleCtfEta = std::nullopt;
    std::optional<float> eleCtfPhi = std::nullopt;

    //the electron's CTF track must share at least 45% of the inner hits
    //with the electron's GSF track
    std::optional<int> ctfidx = std::nullopt;
    int gsfidx = static_cast<int>(eleGsfTk.key());

    if (useEleCtfTrack) {
      eleCtfPt = eleCtfTk->pt();
      eleCtfEta = eleCtfTk->eta();
      eleCtfPhi = eleCtfTk->phi();
      ctfidx = static_cast<int>(eleCtfTk.key());
    }

    //these vectors are for those candidate partner tracks that pass our cuts
    std::vector<ConversionInfo> v_candidatePartners;
    //track indices required to make references
    int ctftk_i = 0;
    int gsftk_i = 0;

    //loop over the CTF tracks and try to find the partner track
    for (auto ctftkItr = ctfTable.begin(); ctftkItr != ctfTable.end(); ++ctftkItr, ctftk_i++) {
      if (useEleCtfTrack && ctftk_i == ctfidx.value())
        continue;

      auto ctftk = *ctftkItr;

      //apply quality cuts to remove bad tracks
      if (ctftk.get<PtError>() > maxRefPtErrorForKfConv * ctftk.get<Pt>() ||
          ctftk.get<NumberOfValidHits>() < minNumberOfValidHits)
        continue;

      if (useEleCtfTrack && std::abs(ctftk.get<Pt>() - eleCtfPt.value()) < maxRelPtDiffForKfConv * eleCtfPt.value())
        continue;

      //use the electron's CTF track, if not null, to search for the partner track
      //look only in a cone of 0.5 to save time, and require that the track is opp. sign
      if (useEleCtfTrack && eleCtfTk->charge() + ctftk.get<Charge>() == 0 &&
          deltaR2(eleCtfEta.value(), eleCtfPhi.value(), ctftk.get<Eta>(), ctftk.get<Phi>()) < dR2Max) {
        ConversionInfo convInfo = getConversionInfo(*eleCtfTk, ctftk, bFieldAtOrigin);

        //need to add the track reference information for completeness
        //because the overloaded fnc above does not make a trackRef
        int deltaMissingHits = ctftk.get<MissingInnerHits>() - eleCtfTk->missingInnerHits();

        v_candidatePartners.push_back(
            {convInfo.dist, convInfo.dcot, convInfo.radiusOfConversion, ctftk_i, std::nullopt, deltaMissingHits, 0});

      }  //using the electron's CTF track

      //now we check using the electron's gsf track
      if (eleGsfTk->charge() + ctftk.get<Charge>() == 0 &&
          deltaR2(eleGsfEta, eleGsfPhi, ctftk.get<Eta>(), ctftk.get<Phi>()) < dR2Max &&
          eleGsfTk->ptError() < maxRelGsfPtErrorForKfConv * eleGsfPt) {
        int deltaMissingHits = ctftk.get<MissingInnerHits>() - eleGsfTk->missingInnerHits();

        ConversionInfo convInfo = getConversionInfo(*eleGsfTk, ctftk, bFieldAtOrigin);

        v_candidatePartners.push_back(
            {convInfo.dist, convInfo.dcot, convInfo.radiusOfConversion, ctftk_i, std::nullopt, deltaMissingHits, 1});
      }  //using the electron's GSF track

    }  //loop over the CTF track collection

    //------------------------------------------------------ Loop over GSF collection ----------------------------------//
    for (auto gsftkItr = gsfTable.begin(); gsftkItr != gsfTable.end(); ++gsftkItr, gsftk_i++) {
      //reject the electron's own gsfTrack
      if (gsfidx == gsftk_i)
        continue;

      auto gsftk = *gsftkItr;

      //apply quality cuts to remove bad tracks
      if (gsftk.get<PtError>() > maxRelGsfPtError * gsftk.get<Pt>() ||
          gsftk.get<NumberOfValidHits>() < minNumberOfValidHits)
        continue;

      if (std::abs(gsftk.get<Pt>() - eleGsfPt) < maxRelPtDiffForGsfConv * eleGsfPt)
        continue;

      //try using the electron's CTF track first if it exists
      //look only in a cone of 0.5 around the electron's track
      //require opposite sign
      if (useEleCtfTrack && eleCtfTk->charge() + gsftk.get<Charge>() == 0 &&
          deltaR2(eleCtfEta.value(), eleCtfPhi.value(), gsftk.get<Eta>(), gsftk.get<Phi>()) < dR2Max) {
        int deltaMissingHits = gsftk.get<MissingInnerHits>() - eleCtfTk->missingInnerHits();

        ConversionInfo convInfo = getConversionInfo(*eleCtfTk, gsftk, bFieldAtOrigin);
        //fill the Ref info
        v_candidatePartners.push_back(
            {convInfo.dist, convInfo.dcot, convInfo.radiusOfConversion, std::nullopt, gsftk_i, deltaMissingHits, 2});
      }

      //use the electron's gsf track
      if (eleGsfTk->charge() + gsftk.get<Charge>() == 0 &&
          deltaR2(eleGsfEta, eleGsfPhi, gsftk.get<Eta>(), gsftk.get<Phi>()) < dR2Max &&
          (eleGsfTk->ptError() < maxRelGsfPtError * eleGsfPt)) {
        ConversionInfo convInfo = getConversionInfo(*eleGsfTk, gsftk, bFieldAtOrigin);
        //fill the Ref info

        int deltaMissingHits = gsftk.get<MissingInnerHits>() - eleGsfTk->missingInnerHits();

        v_candidatePartners.push_back(
            {convInfo.dist, convInfo.dcot, convInfo.radiusOfConversion, std::nullopt, gsftk_i, deltaMissingHits, 3});
      }
    }  //loop over the gsf track collection

    return v_candidatePartners;
  }

  //-------------------------------------------------------------------------------------
  ConversionInfo getConversionInfo(reco::Track const& ele, TrackRowView const& track, float bFieldAtOrigin) {
    using namespace edm::soa::col;

    //now calculate the conversion related information
    float rEl = 100.f * ele.pt() / (-0.3f * bFieldAtOrigin * ele.charge());
    float xEl = -1.f * (rEl - ele.d0()) * sin(ele.phi());
    float yEl = (rEl - ele.d0()) * cos(ele.phi());
    rEl = std::abs(rEl);

    float rCand = 100.f * track.get<Pt>() / (-0.3f * bFieldAtOrigin * track.get<Charge>());
    float xCand = -1.f * (rCand - track.get<D0>()) * sin(track.get<Phi>());
    float yCand = (rCand - track.get<D0>()) * cos(track.get<Phi>());
    rCand = std::abs(rCand);

    float d = sqrt(pow(xEl - xCand, 2) + pow(yEl - yCand, 2));
    float dist = d - (rEl + rCand);

    // this is equivalent to `1/tan(theta_1) - 1/tan(theta_2)` but requires less trigonometry
    float dcot = ele.pz() / ele.pt() - track.get<Pz>() / track.get<Pt>();

    //get the point of conversion
    float xa1 = xEl + (xCand - xEl) * rEl / d;
    float xa2 = xCand + (xEl - xCand) * rCand / d;
    float ya1 = yEl + (yCand - yEl) * rEl / d;
    float ya2 = yCand + (yEl - yCand) * rCand / d;

    float x = .5f * (xa1 + xa2);
    float y = .5f * (ya1 + ya2);
    float rconv = sqrt(pow(x, 2) + pow(y, 2));
    // The z-position of the conversion is unused, but here is how it could be computed if needed:
    // float z = ele.dz() + rEl * ele.pz() * std::acos(1 - pow(rconv, 2) / (2. * pow(rEl, 2))) / ele.pt();

    //now assign a sign to the radius of conversion
    float tempsign = ele.px() * x + ele.py() * y;
    tempsign = tempsign / std::abs(tempsign);
    rconv = tempsign * rconv;

    //return an instance of ConversionInfo, but with a NULL track refs
    return {dist, dcot, rconv};
  }

  //------------------------------------------------------------------------------------

  //takes in a vector of candidate conversion partners
  //and arbitrates between them returning the one with the
  //smallest R=sqrt(dist*dist + dcot*dcot)
  ConversionInfo const& arbitrateConversionPartnersbyR(const std::vector<ConversionInfo>& convCandidates) {
    ConversionInfo const* closestConversion = &convCandidates.front();

    if (convCandidates.size() == 1)
      return *closestConversion;

    float R = pow(closestConversion->dist, 2) + pow(closestConversion->dcot, 2);

    for (auto const& temp : convCandidates) {
      float temp_R = pow(temp.dist, 2) + pow(temp.dcot, 2);
      if (temp_R < R) {
        R = temp_R;
        closestConversion = &temp;
      }
    }

    return *closestConversion;
  }

  //------------------------------------------------------------------------------------
  ConversionInfo findBestConversionMatch(const std::vector<ConversionInfo>& v_convCandidates) {
    using namespace std;

    if (v_convCandidates.empty())
      return {};

    if (v_convCandidates.size() == 1)
      return v_convCandidates.at(0);

    vector<ConversionInfo> v_0;
    vector<ConversionInfo> v_1;
    vector<ConversionInfo> v_2;
    vector<ConversionInfo> v_3;
    //loop over the candidates

    for (auto const& temp : v_convCandidates) {
      if (temp.radiusOfConversion <= -2)
        continue;

      if (temp.flag == 0) {
        if ((std::abs(temp.dist) < maxDistOrCotForKFtoKF && std::abs(temp.dcot) < maxDistOrCotForKFtoKF &&
             temp.deltaMissingHits < maxDeltaMissingHitsForKFtoKF) ||
            (pow(temp.dist, 2) + pow(temp.dcot, 2) < maxDist2Dcot2 && temp.deltaMissingHits < maxDeltaMissingHits))
          v_0.push_back(temp);
      }

      if (temp.flag == 1) {
        if (square(temp.dist) + square(temp.dcot) < maxDist2Dcot2 && temp.deltaMissingHits < maxDeltaMissingHits)
          v_1.push_back(temp);
      }
      if (temp.flag == 2) {
        if (square(temp.dist) + square(temp.dcot) < maxDist2Dcot2 && temp.deltaMissingHits < maxDeltaMissingHits)
          v_2.push_back(temp);
      }
      if (temp.flag == 3) {
        if (square(temp.dist) + square(temp.dcot) < maxDist2Dcot2 && temp.deltaMissingHits < maxDeltaMissingHits)
          v_3.push_back(temp);
      }

    }  //candidate conversion loop

    //now do some arbitration

    //give preference to conversion partners found in the CTF collection
    //using the electron's CTF track
    if (!v_0.empty())
      return arbitrateConversionPartnersbyR(v_0);

    if (!v_1.empty())
      return arbitrateConversionPartnersbyR(v_1);

    if (!v_2.empty())
      return arbitrateConversionPartnersbyR(v_2);

    if (!v_3.empty())
      return arbitrateConversionPartnersbyR(v_3);

    //if we get here, we didn't find a candidate conversion partner that
    //satisfied even the loose selections
    //return the the closest partner by R
    return arbitrateConversionPartnersbyR(v_convCandidates);
  }

}  // namespace egamma::conv
