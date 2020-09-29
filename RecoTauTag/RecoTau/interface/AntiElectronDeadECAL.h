#ifndef RecoTauTag_RecoTau_AntiElectronDeadECAL_h
#define RecoTauTag_RecoTau_AntiElectronDeadECAL_h

/** \class AntiElectronDeadECAL
 *
 * Flag tau candidates reconstructed near dead ECAL channels,
 * in order to reduce e -> tau fakes not rejected by anti-e MVA discriminator
 *
 * The motivation for this flag is this presentation:
 *   https://indico.cern.ch/getFile.py/access?contribId=0&resId=0&materialId=slides&confId=177223
 *
 * Code adapted from:
 *   RecoTauTag/RecoTau/plugins/PFRecoTauDiscriminationAgainstElectronDeadECAL.cc
 *
 * \authors Lauri Andreas Wendland,
 *          Christian Veelken
 *
 *
 *
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoTauTag/RecoTau/interface/PositionAtECalEntranceComputer.h"

#include <vector>
#include <string>

class EcalChannelStatusRcd;
class CaloGeometryRecord;
class IdealGeometryRecord;

class AntiElectronDeadECAL {
public:
  explicit AntiElectronDeadECAL(const edm::ParameterSet&, edm::ConsumesCollector&&);
  ~AntiElectronDeadECAL();

  void beginEvent(const edm::EventSetup&);

  bool operator()(const reco::Candidate* tau) const;

private:
  const unsigned minStatus_;
  const double dR2_;
  const bool extrapolateToECalEntrance_;
  const int verbosity_;

  PositionAtECalEntranceComputer positionAtECalEntrance_;

  void updateBadTowers(const edm::EventSetup&);

  struct TowerInfo {
    TowerInfo(uint32_t id, unsigned nBad, unsigned maxStatus, double eta, double phi)
        : id_(id), nBad_(nBad), maxStatus_(maxStatus), eta_(eta), phi_(phi) {}
    uint32_t id_;
    unsigned nBad_;
    unsigned maxStatus_;
    double eta_;
    double phi_;
  };

  std::vector<TowerInfo> badTowers_;
  static const uint16_t statusMask_ = 0x1F;

  edm::ESWatcher<EcalChannelStatusRcd> channelStatusWatcher_;
  edm::ESWatcher<CaloGeometryRecord> caloGeometryWatcher_;
  edm::ESWatcher<IdealGeometryRecord> idealGeometryWatcher_;

  bool isFirstEvent_;
};

#endif  // RecoTauTag_RecoTau_AntiElectronDeadECAL_h
