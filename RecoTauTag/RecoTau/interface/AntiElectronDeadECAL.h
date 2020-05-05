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
#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoTauTag/RecoTau/interface/PositionAtECalEntrance.h"

#include <vector>
#include <string>

class AntiElectronDeadECAL {
public:
  explicit AntiElectronDeadECAL(const edm::ParameterSet&);
  ~AntiElectronDeadECAL();

  void beginEvent(const edm::EventSetup&);

  bool operator()(const reco::Candidate* tau) const;

private:
  unsigned minStatus_;
  double dR_;
  int verbosity_;
  bool extrapolateToECalEntrance_;

  PositionAtECalEntrance positionAtECalEntrance_;

  void updateBadTowers(const edm::EventSetup&);

  struct towerInfo {
    towerInfo(uint32_t id, unsigned nBad, unsigned maxStatus, double eta, double phi)
        : id_(id), nBad_(nBad), maxStatus_(maxStatus), eta_(eta), phi_(phi) {}
    uint32_t id_;
    unsigned nBad_;
    unsigned maxStatus_;
    double eta_;
    double phi_;
  };
  typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PolarLorentzVector;

  std::vector<towerInfo> badTowers_;
  static const uint16_t statusMask_ = 0x1F;

  uint32_t channelStatusId_cache_;
  uint32_t caloGeometryId_cache_;
  uint32_t idealGeometryId_cache_;
  bool isFirstEvent_;
};

#endif  // RecoTauTag_RecoTau_AntiElectronDeadECAL_h
