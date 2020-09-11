// -*- C++ -*-
//
//
/**\class L1TkElectronTrackMatchAlgo 

 Description: Algorithm to match L1EGamma oject with L1Track candidates

 Implementation:
     [Notes on implementation]
*/

// system include files
#include <memory>
#include <cmath>

#include "DataFormats/Math/interface/deltaPhi.h"
#include "L1Trigger/L1TTrackMatch/interface/L1TkElectronTrackMatchAlgo.h"
namespace L1TkElectronTrackMatchAlgo {

  float constexpr max_eb_eta = 1.479;
  float constexpr max_eb_z = 315.4;
  float constexpr eb_rperp = 129.0;
  // ------------ match EGamma and Track
  void doMatch(l1t::EGammaBxCollection::const_iterator egIter,
               const edm::Ptr<L1TTTrackType>& pTrk,
               double& dph,
               double& dr,
               double& deta) {
    GlobalPoint egPos = L1TkElectronTrackMatchAlgo::calorimeterPosition(egIter->phi(), egIter->eta(), egIter->energy());
    dph = L1TkElectronTrackMatchAlgo::deltaPhi(egPos, pTrk);
    dr = L1TkElectronTrackMatchAlgo::deltaR(egPos, pTrk);
    deta = L1TkElectronTrackMatchAlgo::deltaEta(egPos, pTrk);
  }
  // ------------ match EGamma and Track
  void doMatchClusterET(l1t::EGammaBxCollection::const_iterator egIter,
                        const edm::Ptr<L1TTTrackType>& pTrk,
                        double& dph,
                        double& dr,
                        double& deta) {
    GlobalPoint egPos = L1TkElectronTrackMatchAlgo::calorimeterPosition(egIter->phi(), egIter->eta(), egIter->energy());
    dph = L1TkElectronTrackMatchAlgo::deltaPhiClusterET(egIter, pTrk);
    dr = L1TkElectronTrackMatchAlgo::deltaR(egPos, pTrk);
    deta = L1TkElectronTrackMatchAlgo::deltaEta(egPos, pTrk);
  }
  // ------------ match EGamma and Track
  void doMatch(const GlobalPoint& epos, const edm::Ptr<L1TTTrackType>& pTrk, double& dph, double& dr, double& deta) {
    dph = L1TkElectronTrackMatchAlgo::deltaPhi(epos, pTrk);
    dr = L1TkElectronTrackMatchAlgo::deltaR(epos, pTrk);
    deta = L1TkElectronTrackMatchAlgo::deltaEta(epos, pTrk);
  }
  // --------------- calculate deltaR between Track and EGamma object
  double deltaPhi(const GlobalPoint& epos, const edm::Ptr<L1TTTrackType>& pTrk) {
    double er = epos.perp();
    double curv = pTrk->rInv();

    double dphi_curv = (asin(er * curv / (2.0)));
    double trk_phi_ecal = reco::deltaPhi(pTrk->momentum().phi(), dphi_curv);

    double dphi = reco::deltaPhi(trk_phi_ecal, epos.phi());
    return dphi;
  }
  // --------------- use cluster et to extrapolate tracks
  double deltaPhiClusterET(l1t::EGammaBxCollection::const_iterator egIter, const edm::Ptr<L1TTTrackType>& pTrk) {
    GlobalPoint epos = L1TkElectronTrackMatchAlgo::calorimeterPosition(egIter->phi(), egIter->eta(), egIter->energy());
    double er = epos.perp();
    double et = egIter->et();
    double pt = pTrk->momentum().perp();
    double curv = pTrk->rInv();

    double dphi_curv = (asin(er * curv * pt / (2.0 * et)));
    double trk_phi_ecal = reco::deltaPhi(pTrk->momentum().phi(), dphi_curv);

    double dphi = reco::deltaPhi(trk_phi_ecal, epos.phi());
    return dphi;
  }
  // --------------- calculate deltaPhi between Track and EGamma object
  double deltaR(const GlobalPoint& epos, const edm::Ptr<L1TTTrackType>& pTrk) {
    //double dPhi = fabs(reco::deltaPhi(epos.phi(), pTrk->momentum().phi()));
    double dPhi = L1TkElectronTrackMatchAlgo::deltaPhi(epos, pTrk);
    double dEta = deltaEta(epos, pTrk);
    return sqrt(dPhi * dPhi + dEta * dEta);
  }
  // --------------- calculate deltaEta between Track and EGamma object
  double deltaEta(const GlobalPoint& epos, const edm::Ptr<L1TTTrackType>& pTrk) {
    double corr_eta = 999.0;
    double er = epos.perp();
    double ez = epos.z();
    double z0 = pTrk->POCA().z();
    double theta = 0.0;
    if (ez - z0 >= 0)
      theta = atan(er / fabs(ez - z0));
    else
      theta = M_PI - atan(er / fabs(ez - z0));
    corr_eta = -1.0 * log(tan(theta / 2.0));
    double deleta = (corr_eta - pTrk->momentum().eta());
    return deleta;
  }
  // -------------- get Calorimeter position
  GlobalPoint calorimeterPosition(double phi, double eta, double e) {
    double x = 0.;
    double y = 0.;
    double z = 0.;
    double depth = 0.89 * (7.7 + log(e));
    double theta = 2 * atan(exp(-1 * eta));
    double r = 0;
    if (fabs(eta) > max_eb_eta) {
      double ecalZ = max_eb_z * fabs(eta) / eta;

      r = ecalZ / cos(2 * atan(exp(-1 * eta))) + depth;
      x = r * cos(phi) * sin(theta);
      y = r * sin(phi) * sin(theta);
      z = r * cos(theta);
    } else {
      double zface = sqrt(cos(theta) * cos(theta) / (1 - cos(theta) * cos(theta)) * eb_rperp * eb_rperp);
      r = sqrt(eb_rperp * eb_rperp + zface * zface) + depth;
      x = r * cos(phi) * sin(theta);
      y = r * sin(phi) * sin(theta);
      z = r * cos(theta);
    }
    GlobalPoint pos(x, y, z);
    return pos;
  }

}  // namespace L1TkElectronTrackMatchAlgo
