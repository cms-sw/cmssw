#include "RecoEgamma/EgammaElectronAlgos/interface/ConversionFinder.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TMath.h"

namespace egamma {

  typedef math::XYZTLorentzVector LorentzVector;

  //places different cuts on dist, dcot, delmissing hits and arbitration based on R = sqrt(dist*dist + dcot*dcot)
  ConversionInfo findBestConversionMatch(const std::vector<ConversionInfo>& v_convCandidates);

  ConversionInfo getConversionInfo(const reco::Track* el_track,
                                   LorentzVector const& cand_p4,
                                   double cand_d0,
                                   int cand_charge,
                                   const double bFieldAtOrigin);

  //-----------------------------------------------------------------------------
  std::vector<ConversionInfo> getConversionInfos(const reco::GsfElectronCore& gsfElectron,
                                                 edm::soa::TrackTableView ctfTable,
                                                 edm::soa::TrackTableView gsfTable,
                                                 const double bFieldAtOrigin,
                                                 const double minFracSharedHits) {
    using namespace reco;
    using namespace std;
    using namespace edm;
    using namespace edm::soa::col;

    //get the references to the gsf and ctf tracks that are made
    //by the electron
    const reco::TrackRef el_ctftrack = gsfElectron.ctfTrack();
    const reco::GsfTrackRef& el_gsftrack = gsfElectron.gsfTrack();

    float eleGsfPt = el_gsftrack->pt();

    //make p4s for the electron's tracks for use later
    LorentzVector el_ctftrack_p4;
    if (el_ctftrack.isNonnull() && gsfElectron.ctfGsfOverlap() > minFracSharedHits)
      el_ctftrack_p4 = LorentzVector(el_ctftrack->px(), el_ctftrack->py(), el_ctftrack->pz(), el_ctftrack->p());
    LorentzVector el_gsftrack_p4(el_gsftrack->px(), el_gsftrack->py(), el_gsftrack->pz(), el_gsftrack->p());

    //the electron's CTF track must share at least 45% of the inner hits
    //with the electron's GSF track
    int ctfidx = -999.;
    int gsfidx = -999.;
    if (el_ctftrack.isNonnull() && gsfElectron.ctfGsfOverlap() > minFracSharedHits)
      ctfidx = static_cast<int>(el_ctftrack.key());

    gsfidx = static_cast<int>(el_gsftrack.key());

    //these vectors are for those candidate partner tracks that pass our cuts
    vector<ConversionInfo> v_candidatePartners;
    //track indices required to make references
    int ctftk_i = 0;
    int gsftk_i = 0;

    //loop over the CTF tracks and try to find the partner track
    for (auto ctftkItr = ctfTable.begin(); ctftkItr != ctfTable.end(); ++ctftkItr, ctftk_i++) {
      if (ctftk_i == ctfidx)
        continue;

      auto ctftk = *ctftkItr;

      //candidate track's p4
      LorentzVector ctftk_p4 = LorentzVector(ctftk.get<Px>(), ctftk.get<Py>(), ctftk.get<Pz>(), ctftk.get<P>());

      //apply quality cuts to remove bad tracks
      if (ctftk.get<PtError>() / ctftk.get<Pt>() > 0.05)
        continue;
      if (ctftk.get<NumberOfValidHits>() < 5)
        continue;

      if (el_ctftrack.isNonnull() && gsfElectron.ctfGsfOverlap() > minFracSharedHits &&
          std::abs(ctftk_p4.Pt() - el_ctftrack->pt()) / el_ctftrack->pt() < 0.2)
        continue;

      //use the electron's CTF track, if not null, to search for the partner track
      //look only in a cone of 0.5 to save time, and require that the track is opp. sign
      if (el_ctftrack.isNonnull() && gsfElectron.ctfGsfOverlap() > minFracSharedHits &&
          deltaR(el_ctftrack_p4, ctftk_p4) < 0.5 && (el_ctftrack->charge() + ctftk.get<Charge>() == 0)) {
        ConversionInfo convInfo = getConversionInfo(
            (const reco::Track*)(el_ctftrack.get()), ctftk_p4, ctftk.get<D0>(), ctftk.get<Charge>(), bFieldAtOrigin);

        //need to add the track reference information for completeness
        //because the overloaded fnc above does not make a trackRef
        int deltaMissingHits = ctftk.get<MissingInnerHits>() -
                               el_ctftrack->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);

        v_candidatePartners.push_back({convInfo.dist,
                                       convInfo.dcot,
                                       convInfo.radiusOfConversion,
                                       convInfo.pointOfConversion,
                                       ctftk_i,
                                       std::nullopt,
                                       deltaMissingHits,
                                       0});

      }  //using the electron's CTF track

      //now we check using the electron's gsf track
      if (deltaR(el_gsftrack_p4, ctftk_p4) < 0.5 && (el_gsftrack->charge() + ctftk.get<Charge>() == 0) &&
          el_gsftrack->ptError() / eleGsfPt < 0.25) {
        int deltaMissingHits = ctftk.get<MissingInnerHits>() -
                               el_gsftrack->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);

        ConversionInfo convInfo = getConversionInfo(
            (const reco::Track*)(el_gsftrack.get()), ctftk_p4, ctftk.get<D0>(), ctftk.get<Charge>(), bFieldAtOrigin);

        v_candidatePartners.push_back({convInfo.dist,
                                       convInfo.dcot,
                                       convInfo.radiusOfConversion,
                                       convInfo.pointOfConversion,
                                       ctftk_i,
                                       std::nullopt,
                                       deltaMissingHits,
                                       1});
      }  //using the electron's GSF track

    }  //loop over the CTF track collection

    //------------------------------------------------------ Loop over GSF collection ----------------------------------//
    for (auto gsftkItr = gsfTable.begin(); gsftkItr != gsfTable.end(); ++gsftkItr, gsftk_i++) {
      //reject the electron's own gsfTrack
      if (gsfidx == gsftk_i)
        continue;

      auto gsftk = *gsftkItr;

      LorentzVector gsftk_p4 = LorentzVector(gsftk.get<Px>(), gsftk.get<Py>(), gsftk.get<Pz>(), gsftk.get<P>());

      //apply quality cuts to remove bad tracks
      if (gsftk.get<PtError>() / gsftk.get<Pt>() > 0.5)
        continue;
      if (gsftk.get<NumberOfValidHits>() < 5)
        continue;

      if (std::abs(gsftk.get<Pt>() - eleGsfPt) / eleGsfPt < 0.25)
        continue;

      //try using the electron's CTF track first if it exists
      //look only in a cone of 0.5 around the electron's track
      //require opposite sign
      if (el_ctftrack.isNonnull() && gsfElectron.ctfGsfOverlap() > minFracSharedHits &&
          deltaR(el_ctftrack_p4, gsftk_p4) < 0.5 && (el_ctftrack->charge() + gsftk.get<Charge>() == 0)) {
        int deltaMissingHits = gsftk.get<MissingInnerHits>() -
                               el_ctftrack->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);

        ConversionInfo convInfo = getConversionInfo(
            (const reco::Track*)(el_ctftrack.get()), gsftk_p4, gsftk.get<D0>(), gsftk.get<Charge>(), bFieldAtOrigin);
        //fill the Ref info
        v_candidatePartners.push_back({convInfo.dist,
                                       convInfo.dcot,
                                       convInfo.radiusOfConversion,
                                       convInfo.pointOfConversion,
                                       std::nullopt,
                                       gsftk_i,
                                       deltaMissingHits,
                                       2});
      }

      //use the electron's gsf track
      if (deltaR(el_gsftrack_p4, gsftk_p4) < 0.5 && (el_gsftrack->charge() + gsftk.get<Charge>() == 0) &&
          (el_gsftrack->ptError() / el_gsftrack_p4.pt() < 0.5)) {
        ConversionInfo convInfo = getConversionInfo(
            (const reco::Track*)(el_gsftrack.get()), gsftk_p4, gsftk.get<D0>(), gsftk.get<Charge>(), bFieldAtOrigin);
        //fill the Ref info

        int deltaMissingHits = gsftk.get<MissingInnerHits>() -
                               el_gsftrack->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);

        v_candidatePartners.push_back({convInfo.dist,
                                       convInfo.dcot,
                                       convInfo.radiusOfConversion,
                                       convInfo.pointOfConversion,
                                       std::nullopt,
                                       gsftk_i,
                                       deltaMissingHits,
                                       3});
      }
    }  //loop over the gsf track collection

    return v_candidatePartners;
  }

  //-----------------------------------------------------------------------------
  ConversionInfo findConversion(const reco::GsfElectronCore& gsfElectron,
                                edm::soa::TrackTableView ctfTable,
                                edm::soa::TrackTableView gsfTable,
                                const double bFieldAtOrigin,
                                const double minFracSharedHits) {
    std::vector<ConversionInfo> temp =
        getConversionInfos(gsfElectron, ctfTable, gsfTable, bFieldAtOrigin, minFracSharedHits);
    return findBestConversionMatch(temp);
  }

  //-------------------------------------------------------------------------------------
  ConversionInfo getConversionInfo(const reco::Track* el_track,
                                   LorentzVector const& cand_p4,
                                   double cand_d0,
                                   int cand_charge,
                                   const double bFieldAtOrigin) {
    using namespace reco;

    //now calculate the conversion related information
    LorentzVector el_tk_p4(el_track->px(), el_track->py(), el_track->pz(), el_track->p());
    double elCurvature = -0.3 * bFieldAtOrigin * (el_track->charge() / el_tk_p4.pt()) / 100.;
    double rEl = std::abs(1. / elCurvature);
    double xEl = -1 * (1. / elCurvature - el_track->d0()) * sin(el_tk_p4.phi());
    double yEl = (1. / elCurvature - el_track->d0()) * cos(el_tk_p4.phi());

    double candCurvature = -0.3 * bFieldAtOrigin * (cand_charge / cand_p4.pt()) / 100.;
    double rCand = std::abs(1. / candCurvature);
    double xCand = -1 * (1. / candCurvature - cand_d0) * sin(cand_p4.phi());
    double yCand = (1. / candCurvature - cand_d0) * cos(cand_p4.phi());

    double d = sqrt(pow(xEl - xCand, 2) + pow(yEl - yCand, 2));
    double dist = d - (rEl + rCand);
    double dcot = 1. / tan(el_tk_p4.theta()) - 1. / tan(cand_p4.theta());

    //get the point of conversion
    double xa1 = xEl + (xCand - xEl) * rEl / d;
    double xa2 = xCand + (xEl - xCand) * rCand / d;
    double ya1 = yEl + (yCand - yEl) * rEl / d;
    double ya2 = yCand + (yEl - yCand) * rCand / d;

    double x = .5 * (xa1 + xa2);
    double y = .5 * (ya1 + ya2);
    double rconv = sqrt(pow(x, 2) + pow(y, 2));
    double z =
        el_track->dz() + rEl * el_track->pz() * TMath::ACos(1 - pow(rconv, 2) / (2. * pow(rEl, 2))) / el_track->pt();

    math::XYZPoint convPoint(x, y, z);

    //now assign a sign to the radius of conversion
    float tempsign = el_track->px() * x + el_track->py() * y;
    tempsign = tempsign / std::abs(tempsign);
    rconv = tempsign * rconv;

    //return an instance of ConversionInfo, but with a NULL track refs
    return ConversionInfo{dist, dcot, rconv, convPoint, std::nullopt, std::nullopt, -9999, -9999};
  }

  //------------------------------------------------------------------------------------

  //takes in a vector of candidate conversion partners
  //and arbitrates between them returning the one with the
  //smallest R=sqrt(dist*dist + dcot*dcot)
  ConversionInfo arbitrateConversionPartnersbyR(const std::vector<ConversionInfo>& v_convCandidates) {
    if (v_convCandidates.size() == 1)
      return v_convCandidates.at(0);

    double R = sqrt(pow(v_convCandidates.at(0).dist, 2) + pow(v_convCandidates.at(0).dcot, 2));

    int iArbitrated = 0;
    int i = 0;

    for (auto const& temp : v_convCandidates) {
      double temp_R = sqrt(pow(temp.dist, 2) + pow(temp.dcot, 2));
      if (temp_R < R) {
        R = temp_R;
        iArbitrated = i;
      }
      ++i;
    }

    return v_convCandidates.at(iArbitrated);
  }

  //------------------------------------------------------------------------------------
  ConversionInfo findBestConversionMatch(const std::vector<ConversionInfo>& v_convCandidates) {
    using namespace std;

    if (v_convCandidates.empty())
      return ConversionInfo{
          -9999., -9999., -9999., math::XYZPoint(-9999., -9999., -9999), std::nullopt, std::nullopt, -9999, -9999};

    if (v_convCandidates.size() == 1)
      return v_convCandidates.at(0);

    vector<ConversionInfo> v_0;
    vector<ConversionInfo> v_1;
    vector<ConversionInfo> v_2;
    vector<ConversionInfo> v_3;
    //loop over the candidates
    for (unsigned int i = 1; i < v_convCandidates.size(); i++) {
      ConversionInfo temp = v_convCandidates.at(i);

      if (temp.flag == 0) {
        bool isConv = false;
        if (std::abs(temp.dist) < 0.02 && std::abs(temp.dcot) < 0.02 && temp.deltaMissingHits < 3 &&
            temp.radiusOfConversion > -2)
          isConv = true;
        if (sqrt(pow(temp.dist, 2) + pow(temp.dcot, 2)) < 0.05 && temp.deltaMissingHits < 2 &&
            temp.radiusOfConversion > -2)
          isConv = true;

        if (isConv)
          v_0.push_back(temp);
      }

      if (temp.flag == 1) {
        if (sqrt(pow(temp.dist, 2) + pow(temp.dcot, 2)) < 0.05 && temp.deltaMissingHits < 2 &&
            temp.radiusOfConversion > -2)
          v_1.push_back(temp);
      }
      if (temp.flag == 2) {
        if (sqrt(pow(temp.dist, 2) + pow(temp.dcot * temp.dcot, 2)) < 0.05 && temp.deltaMissingHits < 2 &&
            temp.radiusOfConversion > -2)
          v_2.push_back(temp);
      }
      if (temp.flag == 3) {
        if (sqrt(temp.dist * temp.dist + temp.dcot * temp.dcot) < 0.05 && temp.deltaMissingHits < 2 &&
            temp.radiusOfConversion > -2)
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

  //------------------------------------------------------------------------------------

  //------------------------------------------------------------------------------------
  // Exists here for backwards compatibility only. Provides only the dist and dcot
  std::pair<double, double> getConversionInfo(LorentzVector trk1_p4,
                                              int trk1_q,
                                              float trk1_d0,
                                              LorentzVector trk2_p4,
                                              int trk2_q,
                                              float trk2_d0,
                                              float bFieldAtOrigin) {
    double tk1Curvature = -0.3 * bFieldAtOrigin * (trk1_q / trk1_p4.pt()) / 100.;
    double rTk1 = std::abs(1. / tk1Curvature);
    double xTk1 = -1. * (1. / tk1Curvature - trk1_d0) * sin(trk1_p4.phi());
    double yTk1 = (1. / tk1Curvature - trk1_d0) * cos(trk1_p4.phi());

    double tk2Curvature = -0.3 * bFieldAtOrigin * (trk2_q / trk2_p4.pt()) / 100.;
    double rTk2 = std::abs(1. / tk2Curvature);
    double xTk2 = -1. * (1. / tk2Curvature - trk2_d0) * sin(trk2_p4.phi());
    double yTk2 = (1. / tk2Curvature - trk2_d0) * cos(trk2_p4.phi());

    double dist = sqrt(pow(xTk1 - xTk2, 2) + pow(yTk1 - yTk2, 2));
    dist = dist - (rTk1 + rTk2);

    double dcot = 1. / tan(trk1_p4.theta()) - 1. / tan(trk2_p4.theta());

    return {dist, dcot};
  }

}  // namespace egamma

//-------------------------------------------------------------------------------------
