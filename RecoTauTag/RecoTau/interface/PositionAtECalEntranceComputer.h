#ifndef RecoTauTag_RecoTau_PositionAtECalEntranceComputer_h
#define RecoTauTag_RecoTau_PositionAtECalEntranceComputer_h

/** \class PositionAtECalEntranceComputer
 *
 * Extrapolate particle (charged or neutral) to ECAL entrance,
 * in order to compute the distance of the tau to ECAL cracks and/or dead ECAL channels
 *
 * \authors Fabio Colombo,
 *          Christian Veelken
 *
 *
 *
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

class MagneticField;
class IdealMagneticFieldRecord;

class PositionAtECalEntranceComputer {
public:
  PositionAtECalEntranceComputer(edm::ConsumesCollector&&, bool isPhase2 = false);
  PositionAtECalEntranceComputer(edm::ConsumesCollector&, bool isPhase2 = false);
  ~PositionAtECalEntranceComputer();

  void beginEvent(const edm::EventSetup&);

  //To do: it seems to more practical to put this to the ES
  reco::Candidate::Point operator()(const reco::Candidate* particle, bool& success) const;

private:
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bField_esToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeo_esToken_;
  double bField_z_;
  bool isPhase2_;
  hgcal::RecHitTools recHitTools_;
  float hgcalFace_z_;
  static constexpr float ecalBarrelEndcapEtaBorder_ = 1.479;
  static constexpr float hgcalHfEtaBorder_ = 3.0;
};

#endif  // RecoTauTag_RecoTau_PositionAtECalEntranceComputer_h
