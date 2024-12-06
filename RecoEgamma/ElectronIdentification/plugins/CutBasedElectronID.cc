#include "CutBasedElectronID.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <algorithm>

class CutBasedElectronIDVersionStrategyV00_V01Base : public CutBasedElectronIDVersionStrategyBase {
public:
  explicit CutBasedElectronIDVersionStrategyV00_V01Base(bool iQualityTight, edm::ParameterSet const& iCuts)
      : cuts_{iCuts}, qualityIsTight_{iQualityTight} {}
  double cicSelection(const reco::GsfElectron& electron, const reco::VertexCollection* v) const final;
  bool cicNeedsVertices() const final { return false; }

private:
  int classify(const reco::GsfElectron&) const;
  //used in cicSelection
  virtual double calcSigmaee(const reco::GsfElectron& electron) const = 0;
  edm::ParameterSet cuts_;
  bool qualityIsTight_;
};

class CutBasedElectronIDVersionStrategyV00 : public CutBasedElectronIDVersionStrategyV00_V01Base {
public:
  explicit CutBasedElectronIDVersionStrategyV00(bool iQualityTight, edm::ParameterSet const& iCuts)
      : CutBasedElectronIDVersionStrategyV00_V01Base(iQualityTight, iCuts) {}
  double robustSigmaee(const reco::GsfElectron& electron) const override;

private:
  virtual double calcSigmaee(const reco::GsfElectron& electron) const {
    double sigmaee = electron.sigmaEtaEta();  //sqrt(vCov[0]);
    if (electron.isEE()) {
      double eta = fabs(electron.superCluster()->eta());
      sigmaee = sigmaee - 0.02 * (fabs(eta) - 2.3);  //correct sigmaetaeta dependence on eta in endcap
    }
    //The following is reproducing a bug from the original code
    if (electron.isEB()) {
      sigmaee = electron.sigmaIetaIeta();  //sqrt(vCov[0]);
    }
    return sigmaee;
  }
};

class CutBasedElectronIDVersionStrategyV01 : public CutBasedElectronIDVersionStrategyV00_V01Base {
public:
  CutBasedElectronIDVersionStrategyV01(bool iQualityTight, edm::ParameterSet const& iCuts)
      : CutBasedElectronIDVersionStrategyV00_V01Base(iQualityTight, iCuts) {}

private:
  double calcSigmaee(const reco::GsfElectron& electron) const override {
    return electron.sigmaIetaIeta();  //sqrt(vLocCov[0]);
  }
};

class CutBasedElectronIDVersionStrategyV02 : public CutBasedElectronIDVersionStrategyBase {
public:
  CutBasedElectronIDVersionStrategyV02(edm::ParameterSet const& iCuts) : cuts_(iCuts) {}
  double cicSelection(const reco::GsfElectron& electron, const reco::VertexCollection* v) const override;
  bool cicNeedsVertices() const final { return true; }

private:
  int classify(const reco::GsfElectron&) const;
  edm::ParameterSet cuts_;
};

namespace {
  //These exist to guarantee we can't get the order wrong in constructor calls
  enum class WantBinning { true_, false_ };
  enum class NewCategories { true_, false_ };
}  // namespace

class CutBasedElectronIDVersionStrategyV03_V06Base : public CutBasedElectronIDVersionStrategyBase {
public:
  CutBasedElectronIDVersionStrategyV03_V06Base(std::string const& iType, NewCategories newCategories) {
    if (iType == "classbased") {
      newCategories_ = (newCategories == NewCategories::true_);
    }
  }
  bool cicNeedsVertices() const final { return true; }

protected:
  int classify(const reco::GsfElectron&) const;

private:
  bool newCategories_ = false;
};

class CutBasedElectronIDVersionStrategyV03 : public CutBasedElectronIDVersionStrategyV03_V06Base {
public:
  CutBasedElectronIDVersionStrategyV03(std::string const& iType,
                                       NewCategories newCategories,
                                       WantBinning wantBinning,
                                       edm::ParameterSet const& iCuts)
      : CutBasedElectronIDVersionStrategyV03_V06Base(iType, newCategories),
        cuts_{iCuts},
        wantBinning_(wantBinning == WantBinning::true_) {}
  double cicSelection(const reco::GsfElectron& electron, const reco::VertexCollection* v) const override;

  double robustIp(const reco::GsfElectron& electron,
                  edm::Handle<reco::BeamSpot> pBeamSpot,
                  const reco::VertexCollection*) const override;
  bool robustNeedsBeamSpot() const override;
  bool robustNeedsVertices() const override;

private:
  edm::ParameterSet cuts_;
  bool wantBinning_;
};

class CutBasedElectronIDVersionStrategyV04 : public CutBasedElectronIDVersionStrategyV03 {
public:
  CutBasedElectronIDVersionStrategyV04(std::string const& iType,
                                       NewCategories newCategories,
                                       WantBinning wantBinning,
                                       edm::ParameterSet const& iCuts)
      : CutBasedElectronIDVersionStrategyV03(iType, newCategories, wantBinning, iCuts) {}
  Iso robustIso(const reco::GsfElectron& electron) const override;
};
class CutBasedElectronIDVersionStrategyV05 : public CutBasedElectronIDVersionStrategyV04 {
public:
  CutBasedElectronIDVersionStrategyV05(std::string const& iType,
                                       NewCategories newCategories,
                                       WantBinning wantBinning,
                                       edm::ParameterSet const& iCuts)
      : CutBasedElectronIDVersionStrategyV04(iType, newCategories, wantBinning, iCuts) {}
  double robustIp(const reco::GsfElectron& electron,
                  edm::Handle<reco::BeamSpot> pBeamSpot,
                  const reco::VertexCollection*) const final;
  bool robustNeedsBeamSpot() const final;
  bool robustNeedsVertices() const final;
};

class CutBasedElectronIDVersionStrategyV06 : public CutBasedElectronIDVersionStrategyV03_V06Base {
public:
  CutBasedElectronIDVersionStrategyV06(std::string const& iType,
                                       NewCategories newCategories,
                                       edm::ParameterSet const& iCuts)
      : CutBasedElectronIDVersionStrategyV03_V06Base(iType, newCategories), cuts_(iCuts) {}
  double cicSelection(const reco::GsfElectron& electron, const reco::VertexCollection* v) const override;

private:
  bool compute_cut(double x, double et, double cut_min, double cut_max, bool gtn = false) const;

  edm::ParameterSet cuts_;
};

using CutBasedElectronIDVersionStrategyDefault = CutBasedElectronIDVersionStrategyV06;

namespace {
  double getSigmaee(const reco::GsfElectron& electron) {
    double sigmaee = electron.sigmaIetaIeta();  //sqrt(vLocCov[0]);
    if (electron.isEB()) {
      sigmaee = electron.sigmaIetaIeta();  //sqrt(vCov[0]);
    }
    return sigmaee;
  }

  double get_ip(reco::VertexCollection const& vtxC, const reco::GsfElectron& electron) {
    double ip = 0;
    if (!vtxC.empty()) {
      auto const& vtx = vtxC[0];
      ip = fabs(electron.gsfTrack()->dxy(math::XYZPoint(vtx.x(), vtx.y(), vtx.z())));
    } else
      ip = fabs(electron.gsfTrack()->dxy());
    return ip;
  }

  NewCategories newCategories(std::string const& iType, edm::ParameterSet const& conf) {
    if (iType == "classbased") {
      return conf.getParameter<bool>("additionalCategories") ? NewCategories::true_ : NewCategories::false_;
    }
    return NewCategories::false_;
  }

  WantBinning wantBinning(std::string const& iType, edm::ParameterSet const& conf) {
    if (iType == "classbased") {
      return conf.getParameter<bool>("etBinning") ? WantBinning::true_ : WantBinning::false_;
    }
    return WantBinning::false_;
  }
  std::unique_ptr<CutBasedElectronIDVersionStrategyBase> makeStrategy(const std::string& iVersion,
                                                                      const std::string& iType,
                                                                      const std::string& iQuality,
                                                                      edm::ParameterSet const& iPSet,
                                                                      edm::ParameterSet const& cuts) {
    if (iVersion == "V00") {
      return std::make_unique<CutBasedElectronIDVersionStrategyV00>(iQuality == "tight", cuts);
    } else if (iVersion == "V01") {
      return std::make_unique<CutBasedElectronIDVersionStrategyV01>(iQuality == "tight", cuts);
    } else if (iVersion == "V02") {
      return std::make_unique<CutBasedElectronIDVersionStrategyV02>(cuts);
    } else if (iVersion == "V03") {
      return std::make_unique<CutBasedElectronIDVersionStrategyV03>(
          iType, newCategories(iType, iPSet), wantBinning(iType, iPSet), cuts);
    } else if (iVersion == "V04") {
      return std::make_unique<CutBasedElectronIDVersionStrategyV04>(
          iType, newCategories(iType, iPSet), wantBinning(iType, iPSet), cuts);
    } else if (iVersion == "V05") {
      return std::make_unique<CutBasedElectronIDVersionStrategyV05>(
          iType, newCategories(iType, iPSet), wantBinning(iType, iPSet), cuts);
    } else if (iVersion == "V06") {
      return std::make_unique<CutBasedElectronIDVersionStrategyV06>(iType, newCategories(iType, iPSet), cuts);
    } else {
      return std::make_unique<CutBasedElectronIDVersionStrategyDefault>(iType, newCategories(iType, iPSet), cuts);
    }
  }
}  // namespace

CutBasedElectronID::CutBasedElectronID(const edm::ParameterSet& conf, edm::ConsumesCollector& iC) {
  type_ = conf.getParameter<std::string>("electronIDType");
  quality_ = conf.getParameter<std::string>("electronQuality");
  auto version = conf.getParameter<std::string>("electronVersion");

  edm::ParameterSet cuts;
  if (type_ == "robust" || type_ == "classbased") {
    std::string stringCut = type_ + quality_ + "EleIDCuts" + version;
    cuts = conf.getParameter<edm::ParameterSet>(stringCut);
  } else {
    throw cms::Exception("Configuration")
        << "Invalid electronType parameter in CutBasedElectronID: must be robust or classbased\n";
  }

  if (type_ == "robust") {
    barrelCuts_ = cuts.getParameter<std::vector<double> >("barrel");
    endcapCuts_ = cuts.getParameter<std::vector<double> >("endcap");
  }

  versionStrategy_ = makeStrategy(version, type_, quality_, conf, cuts);
  if ((type_ == "classbased" and versionStrategy_->cicNeedsVertices())) {
    verticesCollection_ =
        iC.consumes<std::vector<reco::Vertex> >(conf.getParameter<edm::InputTag>("verticesCollection"));
  } else if (type_ == "robust") {
    //The use of 'verticesCollection' for two different data items was in the old code. This explicitly
    // shows the uses were mutually exclusive
    assert(not(versionStrategy_->robustNeedsVertices() and versionStrategy_->robustNeedsBeamSpot()));
    if (versionStrategy_->robustNeedsVertices()) {
      verticesCollection_ =
          iC.consumes<std::vector<reco::Vertex> >(conf.getParameter<edm::InputTag>("verticesCollection"));
    }
    if (versionStrategy_->robustNeedsBeamSpot()) {
      beamSpot_ = iC.consumes(conf.getParameter<edm::InputTag>("verticesCollection"));
    }
  }
}

double CutBasedElectronID::result(const reco::GsfElectron* electron,
                                  const edm::Event& e,
                                  const edm::EventSetup& es) const {
  if (type_ == "classbased")
    return cicSelection(electron, e, es);
  else if (type_ == "robust")
    return robustSelection(electron, e, es);

  return 0;
}

int CutBasedElectronIDVersionStrategyV00_V01Base::classify(const reco::GsfElectron& electron) const {
  double eOverP = electron.eSuperClusterOverP();
  double fBrem = electron.fbrem();

  int cat = -1;
  if ((electron.isEB() && fBrem < 0.06) || (electron.isEE() && fBrem < 0.1))
    cat = 1;
  else if (eOverP < 1.2 && eOverP > 0.8)
    cat = 0;
  else
    cat = 2;

  return cat;
}

int CutBasedElectronIDVersionStrategyV02::classify(const reco::GsfElectron& electron) const {
  double eOverP = electron.eSuperClusterOverP();
  double fBrem = electron.fbrem();

  int cat = -1;
  if (electron.isEB()) {  // BARREL
    if (fBrem < 0.12)
      cat = 1;
    else if (eOverP < 1.2 && eOverP > 0.9)
      cat = 0;
    else
      cat = 2;
  } else {  // ENDCAP
    if (fBrem < 0.2)
      cat = 1;
    else if (eOverP < 1.22 && eOverP > 0.82)
      cat = 0;
    else
      cat = 2;
  }

  return cat;
}

int CutBasedElectronIDVersionStrategyV03_V06Base::classify(const reco::GsfElectron& electron) const {
  double eta = fabs(electron.superCluster()->eta());
  double eOverP = electron.eSuperClusterOverP();
  double fBrem = electron.fbrem();

  int cat = -1;
  if (electron.isEB()) {
    if ((fBrem >= 0.12) and (eOverP > 0.9) and (eOverP < 1.2))
      cat = 0;
    else if (((eta > .445 and eta < .45) or (eta > .79 and eta < .81) or (eta > 1.137 and eta < 1.157) or
              (eta > 1.47285 and eta < 1.4744)) and
             newCategories_)
      cat = 6;
    else if (electron.trackerDrivenSeed() and !electron.ecalDrivenSeed() and newCategories_)
      cat = 8;
    else if (fBrem < 0.12)
      cat = 1;
    else
      cat = 2;
  } else {
    if ((fBrem >= 0.2) and (eOverP > 0.82) and (eOverP < 1.22))
      cat = 3;
    else if (eta > 1.5 and eta < 1.58 and newCategories_)
      cat = 7;
    else if (electron.trackerDrivenSeed() and !electron.ecalDrivenSeed() and newCategories_)
      cat = 8;
    else if (fBrem < 0.2)
      cat = 4;
    else
      cat = 5;
  }

  return cat;
}

double CutBasedElectronIDVersionStrategyV00_V01Base::cicSelection(const reco::GsfElectron& electron,
                                                                  const reco::VertexCollection* v) const {
  int cat = classify(electron);
  int eb = electron.isEB() ? 0 : 1;

  // LOOSE and TIGHT Selections
  double eOverP = electron.eSuperClusterOverP();
  double fBrem = electron.fbrem();
  if ((eOverP < 0.8) && (fBrem < 0.2))
    return 0.;

  auto cut = cuts_.getParameter<std::vector<double> >("hOverE");
  double hOverE = electron.hadronicOverEm();
  if (hOverE > cut[cat + 4 * eb])
    return 0.;

  cut = cuts_.getParameter<std::vector<double> >("sigmaEtaEta");
  auto sigmaee = calcSigmaee(electron);
  if (sigmaee > cut[cat + 4 * eb])
    return 0.;

  cut = cuts_.getParameter<std::vector<double> >("deltaPhiIn");
  double deltaPhiIn = electron.deltaPhiSuperClusterTrackAtVtx();
  if (eOverP < 1.5) {
    if (fabs(deltaPhiIn) > cut[cat + 4 * eb])
      return 0.;
  } else {
    if (fabs(deltaPhiIn) > cut[3 + 4 * eb])
      return 0.;
  }

  cut = cuts_.getParameter<std::vector<double> >("deltaEtaIn");
  double deltaEtaIn = electron.deltaEtaSuperClusterTrackAtVtx();
  if (fabs(deltaEtaIn) > cut[cat + 4 * eb])
    return 0.;

  cut = cuts_.getParameter<std::vector<double> >("eSeedOverPin");
  double eSeedOverPin = electron.eSeedClusterOverP();
  if (eSeedOverPin < cut[cat + 4 * eb])
    return 0.;

  if (qualityIsTight_) {
    if (eOverP < 0.9 * (1 - fBrem))
      return 0.;
  }
  return 1.;
}

double CutBasedElectronIDVersionStrategyV02::cicSelection(const reco::GsfElectron& electron,
                                                          const reco::VertexCollection* vertices) const {
  double result = 0.;

  int bin = 0;
  double scTheta = (2 * atan(exp(-electron.superCluster()->eta())));
  double scEt = electron.superCluster()->energy() * sin(scTheta);

  if (scEt < 20.)
    bin = 2;
  else if (scEt > 30.)
    bin = 0;
  else
    bin = 1;

  double eSeedOverPin = electron.eSeedClusterOverP();
  double fBrem = electron.fbrem();
  if (fBrem > 0)
    eSeedOverPin = eSeedOverPin + fBrem;

  double tkIso = electron.dr03TkSumPt();
  double ecalIso = electron.dr04EcalRecHitSumEt();
  double hcalIso = electron.dr04HcalTowerSumEt();
  if (bin != 2) {
    tkIso = tkIso * pow(40. / scEt, 2);
    ecalIso = ecalIso * pow(40. / scEt, 2);
    hcalIso = hcalIso * pow(40. / scEt, 2);
  }

  int cat = classify(electron);
  int eb = electron.isEB() ? 0 : 1;

  std::vector<double> cutTk = cuts_.getParameter<std::vector<double> >("cutisotk");
  std::vector<double> cutEcal = cuts_.getParameter<std::vector<double> >("cutisoecal");
  std::vector<double> cutHcal = cuts_.getParameter<std::vector<double> >("cutisohcal");
  if ((tkIso > cutTk[cat + 3 * eb + bin * 6]) || (ecalIso > cutEcal[cat + 3 * eb + bin * 6]) ||
      (hcalIso > cutHcal[cat + 3 * eb + bin * 6]))
    result = 0.;
  else
    result = 2.;

  if (fBrem > -2) {
    std::vector<double> cuthoe = cuts_.getParameter<std::vector<double> >("cuthoe");
    std::vector<double> cutsee = cuts_.getParameter<std::vector<double> >("cutsee");
    std::vector<double> cutdphi = cuts_.getParameter<std::vector<double> >("cutdphiin");
    std::vector<double> cutdeta = cuts_.getParameter<std::vector<double> >("cutdetain");
    std::vector<double> cuteopin = cuts_.getParameter<std::vector<double> >("cuteseedopcor");
    std::vector<double> cutet = cuts_.getParameter<std::vector<double> >("cutet");
    std::vector<double> cutip = cuts_.getParameter<std::vector<double> >("cutip");
    std::vector<double> cutmishits = cuts_.getParameter<std::vector<double> >("cutmishits");

    double sigmaee = getSigmaee(electron);
    double ip = get_ip(*vertices, electron);

    double hOverE = electron.hadronicOverEm();
    double deltaPhiIn = electron.deltaPhiSuperClusterTrackAtVtx();
    double deltaEtaIn = electron.deltaEtaSuperClusterTrackAtVtx();
    double eSeedOverPin = electron.eSeedClusterOverP();
    int mishits = electron.gsfTrack()->missingInnerHits();
    if ((hOverE < cuthoe[cat + 3 * eb + bin * 6]) and (sigmaee < cutsee[cat + 3 * eb + bin * 6]) and
        (fabs(deltaPhiIn) < cutdphi[cat + 3 * eb + bin * 6]) and
        (fabs(deltaEtaIn) < cutdeta[cat + 3 * eb + bin * 6]) and (eSeedOverPin > cuteopin[cat + 3 * eb + bin * 6]) and
        (ip < cutip[cat + 3 * eb + bin * 6]) and (mishits < cutmishits[cat + 3 * eb + bin * 6]))
      result = result + 1.;
  }
  return result;
}

double CutBasedElectronIDVersionStrategyV03::cicSelection(const reco::GsfElectron& electron,
                                                          const reco::VertexCollection* vertices) const {
  double result = 0.;

  int bin = 0;

  double scTheta = (2 * atan(exp(-electron.superCluster()->eta())));
  double scEt = electron.superCluster()->energy() * sin(scTheta);
  if (wantBinning_) {
    if (scEt < 20.)
      bin = 2;
    else if (scEt > 30.)
      bin = 0;
    else
      bin = 1;
  }

  double fBrem = electron.fbrem();
  double eSeedOverPin = electron.eSeedClusterOverP();
  if (fBrem > 0)
    eSeedOverPin = eSeedOverPin + fBrem;

  double tkIso = electron.dr03TkSumPt();
  double ecalIso = electron.dr04EcalRecHitSumEt();
  double hcalIso = electron.dr04HcalTowerSumEt();
  float iso_sum = tkIso + ecalIso + hcalIso;
  float iso_sum_corrected = iso_sum * pow(40. / scEt, 2);

  int cat = classify(electron);
  std::vector<double> cutIsoSum = cuts_.getParameter<std::vector<double> >("cutiso_sum");
  std::vector<double> cutIsoSumCorr = cuts_.getParameter<std::vector<double> >("cutiso_sumoet");
  if ((iso_sum < cutIsoSum[cat + bin * 9]) and (iso_sum_corrected < cutIsoSumCorr[cat + bin * 9]))
    result += 2.;

  if (fBrem > -2) {
    std::vector<double> cuthoe = cuts_.getParameter<std::vector<double> >("cuthoe");
    std::vector<double> cutsee = cuts_.getParameter<std::vector<double> >("cutsee");
    std::vector<double> cutdphi = cuts_.getParameter<std::vector<double> >("cutdphiin");
    std::vector<double> cutdeta = cuts_.getParameter<std::vector<double> >("cutdetain");
    std::vector<double> cuteopin = cuts_.getParameter<std::vector<double> >("cuteseedopcor");
    std::vector<double> cutet = cuts_.getParameter<std::vector<double> >("cutet");

    double hOverE = electron.hadronicOverEm();
    double sigmaee = getSigmaee(electron);
    double deltaPhiIn = electron.deltaPhiSuperClusterTrackAtVtx();
    double deltaEtaIn = electron.deltaEtaSuperClusterTrackAtVtx();
    double eSeedOverPin = electron.eSeedClusterOverP();
    if ((hOverE < cuthoe[cat + bin * 9]) and (sigmaee < cutsee[cat + bin * 9]) and
        (fabs(deltaPhiIn) < cutdphi[cat + bin * 9]) and (fabs(deltaEtaIn) < cutdeta[cat + bin * 9]) and
        (eSeedOverPin > cuteopin[cat + bin * 9]) and (scEt > cutet[cat + bin * 9]))
      result += 1.;
  }

  std::vector<double> cutip = cuts_.getParameter<std::vector<double> >("cutip_gsf");
  double ip = get_ip(*vertices, electron);
  if (ip < cutip[cat + bin * 9])
    result += 8;

  std::vector<double> cutmishits = cuts_.getParameter<std::vector<double> >("cutfmishits");
  std::vector<double> cutdcotdist = cuts_.getParameter<std::vector<double> >("cutdcotdist");

  float dist = (electron.convDist() == -9999. ? 9999 : electron.convDist());
  float dcot = (electron.convDcot() == -9999. ? 9999 : electron.convDcot());

  float dcotdistcomb = ((0.04 - std::max(fabs(dist), fabs(dcot))) > 0 ? (0.04 - std::max(fabs(dist), fabs(dcot))) : 0);

  int mishits = electron.gsfTrack()->missingInnerHits();
  if ((mishits < cutmishits[cat + bin * 9]) and (dcotdistcomb < cutdcotdist[cat + bin * 9]))
    result += 4;

  return result;
}

double CutBasedElectronIDVersionStrategyV06::cicSelection(const reco::GsfElectron& electron,
                                                          const reco::VertexCollection* vertices) const {
  std::vector<double> cutIsoSum = cuts_.getParameter<std::vector<double> >("cutiso_sum");
  std::vector<double> cutIsoSumCorr = cuts_.getParameter<std::vector<double> >("cutiso_sumoet");
  std::vector<double> cuthoe = cuts_.getParameter<std::vector<double> >("cuthoe");
  std::vector<double> cutsee = cuts_.getParameter<std::vector<double> >("cutsee");
  std::vector<double> cutdphi = cuts_.getParameter<std::vector<double> >("cutdphiin");
  std::vector<double> cutdeta = cuts_.getParameter<std::vector<double> >("cutdetain");
  std::vector<double> cuteopin = cuts_.getParameter<std::vector<double> >("cuteseedopcor");
  std::vector<double> cutmishits = cuts_.getParameter<std::vector<double> >("cutfmishits");
  std::vector<double> cutdcotdist = cuts_.getParameter<std::vector<double> >("cutdcotdist");
  std::vector<double> cutip = cuts_.getParameter<std::vector<double> >("cutip_gsf");
  std::vector<double> cutIsoSumCorrl = cuts_.getParameter<std::vector<double> >("cutiso_sumoetl");
  std::vector<double> cuthoel = cuts_.getParameter<std::vector<double> >("cuthoel");
  std::vector<double> cutseel = cuts_.getParameter<std::vector<double> >("cutseel");
  std::vector<double> cutdphil = cuts_.getParameter<std::vector<double> >("cutdphiinl");
  std::vector<double> cutdetal = cuts_.getParameter<std::vector<double> >("cutdetainl");
  std::vector<double> cutipl = cuts_.getParameter<std::vector<double> >("cutip_gsfl");

  int result = 0;

  const int ncuts = 10;
  std::vector<bool> cut_results(ncuts, false);

  double tkIso = electron.dr03TkSumPt();
  double ecalIso = electron.dr04EcalRecHitSumEt();
  double hcalIso = electron.dr04HcalTowerSumEt();
  float iso_sum = tkIso + ecalIso + hcalIso;
  float scEta = electron.superCluster()->eta();
  if (fabs(scEta) > 1.5)
    iso_sum += (fabs(scEta) - 1.5) * 1.09;

  double scTheta = (2 * atan(exp(-electron.superCluster()->eta())));
  double scEt = electron.superCluster()->energy() * sin(scTheta);
  float iso_sumoet = iso_sum * (40. / scEt);

  double eSeedOverPin = electron.eSeedClusterOverP();
  double fBrem = electron.fbrem();
  float eseedopincor = eSeedOverPin + fBrem;
  if (fBrem < 0)
    eseedopincor = eSeedOverPin;

  float dist = (electron.convDist() == -9999. ? 9999 : electron.convDist());
  float dcot = (electron.convDcot() == -9999. ? 9999 : electron.convDcot());

  float dcotdistcomb = ((0.04 - std::max(fabs(dist), fabs(dcot))) > 0 ? (0.04 - std::max(fabs(dist), fabs(dcot))) : 0);

  int cat = classify(electron);
  for (int cut = 0; cut < ncuts; cut++) {
    switch (cut) {
      case 0: {
        double deltaEtaIn = electron.deltaEtaSuperClusterTrackAtVtx();
        cut_results[cut] = compute_cut(fabs(deltaEtaIn), scEt, cutdetal[cat], cutdeta[cat]);
        break;
      }
      case 1: {
        double deltaPhiIn = electron.deltaPhiSuperClusterTrackAtVtx();
        cut_results[cut] = compute_cut(fabs(deltaPhiIn), scEt, cutdphil[cat], cutdphi[cat]);
        break;
      }
      case 2:
        cut_results[cut] = (eseedopincor > cuteopin[cat]);
        break;
      case 3: {
        double hOverE = electron.hadronicOverEm();
        cut_results[cut] = compute_cut(hOverE, scEt, cuthoel[cat], cuthoe[cat]);
        break;
      }
      case 4: {
        double sigmaee = getSigmaee(electron);
        cut_results[cut] = compute_cut(sigmaee, scEt, cutseel[cat], cutsee[cat]);
        break;
      }
      case 5:
        cut_results[cut] = compute_cut(iso_sumoet, scEt, cutIsoSumCorrl[cat], cutIsoSumCorr[cat]);
        break;
      case 6:
        cut_results[cut] = (iso_sum < cutIsoSum[cat]);
        break;
      case 7: {
        double ip = get_ip(*vertices, electron);
        cut_results[cut] = compute_cut(fabs(ip), scEt, cutipl[cat], cutip[cat]);
        break;
      }
      case 8: {
        int mishits = electron.gsfTrack()->missingInnerHits();
        cut_results[cut] = (mishits < cutmishits[cat]);
        break;
      }
      case 9:
        cut_results[cut] = (dcotdistcomb < cutdcotdist[cat]);
        break;
    }
  }

  // ID part
  if (cut_results[0] && cut_results[1] && cut_results[2] && cut_results[3] && cut_results[4])
    result = result + 1;

  // ISO part
  if (cut_results[5] && cut_results[6])
    result = result + 2;

  // IP part
  if (cut_results[7])
    result = result + 8;

  // Conversion part
  if (cut_results[8] && cut_results[9])
    result = result + 4;

  return result;
}
double CutBasedElectronID::cicSelection(const reco::GsfElectron* electron,
                                        const edm::Event& e,
                                        const edm::EventSetup& es) const {
  return versionStrategy_->cicSelection(*electron,
                                        versionStrategy_->cicNeedsVertices() ? &(e.get(verticesCollection_)) : nullptr);
}

bool CutBasedElectronIDVersionStrategyV06::compute_cut(
    double x, double et, double cut_min, double cut_max, bool gtn) const {
  float et_min = 10;
  float et_max = 40;

  bool accept = false;
  float cut = cut_max;  //  the cut at et=40 GeV

  if (et < et_max) {
    cut = cut_min + (1 / et_min - 1 / et) * (cut_max - cut_min) / (1 / et_min - 1 / et_max);
  }

  if (et < et_min) {
    cut = cut_min;
  }

  if (gtn) {  // useful for e/p cut which is gt
    accept = (x >= cut);
  } else {
    accept = (x <= cut);
  }

  //std::cout << x << " " << cut_min << " " << cut << " " << cut_max << " " << et << " " << accept << std::endl;
  return accept;
}

double CutBasedElectronIDVersionStrategyBase::robustSigmaee(const reco::GsfElectron& electron) const {
  return electron.sigmaIetaIeta();
}

double CutBasedElectronIDVersionStrategyV00::robustSigmaee(const reco::GsfElectron& electron) const {
  double sigmaee = electron.sigmaEtaEta();
  double eta = electron.p4().Eta();
  if (electron.isEE())
    sigmaee = sigmaee - 0.02 * (fabs(eta) - 2.3);  //correct sigmaetaeta dependence on eta in endcap
  return sigmaee;
}

double CutBasedElectronIDVersionStrategyBase::robustIp(const reco::GsfElectron& electron,
                                                       edm::Handle<reco::BeamSpot>,
                                                       const reco::VertexCollection*) const {
  return 0;
}

bool CutBasedElectronIDVersionStrategyBase::robustNeedsBeamSpot() const { return false; }
bool CutBasedElectronIDVersionStrategyBase::robustNeedsVertices() const { return false; }

//V03 and V04
double CutBasedElectronIDVersionStrategyV03::robustIp(const reco::GsfElectron& electron,
                                                      edm::Handle<reco::BeamSpot> pBeamSpot,
                                                      const reco::VertexCollection*) const {
  double ip = 0;
  if (pBeamSpot.isValid()) {
    const reco::BeamSpot* bspot = pBeamSpot.product();
    const math::XYZPoint& bspotPosition = bspot->position();
    ip = fabs(electron.gsfTrack()->dxy(bspotPosition));
  } else
    ip = fabs(electron.gsfTrack()->dxy());

  return ip;
}

bool CutBasedElectronIDVersionStrategyV03::robustNeedsBeamSpot() const { return true; }
bool CutBasedElectronIDVersionStrategyV03::robustNeedsVertices() const { return false; }

double CutBasedElectronIDVersionStrategyV05::robustIp(const reco::GsfElectron& electron,
                                                      edm::Handle<reco::BeamSpot>,
                                                      const reco::VertexCollection* pVtxC) const {
  assert(pVtxC);
  auto const& vtxC = *pVtxC;
  double ip = 0;
  if (!vtxC.empty()) {
    auto const& vtx = vtxC[0];
    ip = fabs(electron.gsfTrack()->dxy(math::XYZPoint(vtx.x(), vtx.y(), vtx.z())));
  } else
    ip = fabs(electron.gsfTrack()->dxy());

  return ip;
}

bool CutBasedElectronIDVersionStrategyV05::robustNeedsBeamSpot() const { return false; }
bool CutBasedElectronIDVersionStrategyV05::robustNeedsVertices() const { return true; }

CutBasedElectronIDVersionStrategyBase::Iso CutBasedElectronIDVersionStrategyBase::robustIso(
    const reco::GsfElectron& electron) const {
  Iso iso;
  iso.ecal = electron.dr04EcalRecHitSumEt();
  iso.hcal = electron.dr04HcalTowerSumEt();
  iso.hcal1 = electron.dr04HcalTowerSumEt(1);
  iso.hcal2 = electron.dr04HcalTowerSumEt(2);
  return iso;
}

//V04 and V05
CutBasedElectronIDVersionStrategyBase::Iso CutBasedElectronIDVersionStrategyV04::robustIso(
    const reco::GsfElectron& electron) const {
  Iso iso;
  iso.ecal = electron.dr03EcalRecHitSumEt();
  iso.hcal = electron.dr03HcalTowerSumEt();
  iso.hcal1 = electron.dr03HcalTowerSumEt(1);
  iso.hcal2 = electron.dr03HcalTowerSumEt(2);
  return iso;
}

double CutBasedElectronID::robustSelection(const reco::GsfElectron* electron,
                                           const edm::Event& e,
                                           const edm::EventSetup& es) const {
  double scTheta = (2 * atan(exp(-electron->superCluster()->eta())));
  double scEt = electron->superCluster()->energy() * sin(scTheta);
  double eOverP = electron->eSuperClusterOverP();
  double hOverE = electron->hadronicOverEm();
  double sigmaee = versionStrategy_->robustSigmaee(*electron);
  double e25Max = electron->e2x5Max();
  double e15 = electron->e1x5();
  double e55 = electron->e5x5();
  double e25Maxoe55 = e25Max / e55;
  double e15oe55 = e15 / e55;
  double deltaPhiIn = electron->deltaPhiSuperClusterTrackAtVtx();
  double deltaEtaIn = electron->deltaEtaSuperClusterTrackAtVtx();

  double ip = versionStrategy_->robustIp(
      *electron,
      versionStrategy_->robustNeedsBeamSpot() ? e.getHandle(beamSpot_) : edm::Handle<reco::BeamSpot>(),
      versionStrategy_->robustNeedsVertices() ? &(e.get(verticesCollection_)) : nullptr);
  int mishits = electron->gsfTrack()->missingInnerHits();
  double tkIso = electron->dr03TkSumPt();
  auto iso = versionStrategy_->robustIso(*electron);
  double ecalIso = iso.ecal;
  double ecalIsoPed = (electron->isEB()) ? std::max(0., ecalIso - 1.) : ecalIso;
  double hcalIso = iso.hcal;
  double hcalIso1 = iso.hcal1;
  double hcalIso2 = iso.hcal2;

  // .....................................................................................
  //in the future, this would be better to change to std::span
  const double* cut = nullptr;
  // ROBUST Selection
  double result = 0;

  // hoe, sigmaEtaEta, dPhiIn, dEtaIn
  if (electron->isEB())
    cut = &barrelCuts_.front();
  else
    cut = &endcapCuts_.front();
  // check isolations: if only isolation passes result = 2
  if (quality_ == "highenergy") {
    if ((tkIso > cut[6] || hcalIso2 > cut[11]) ||
        (electron->isEB() && ((ecalIso + hcalIso1) > cut[7] + cut[8] * scEt)) ||
        (electron->isEE() && (scEt >= 50.) && ((ecalIso + hcalIso1) > cut[7] + cut[8] * (scEt - 50))) ||
        (electron->isEE() && (scEt < 50.) && ((ecalIso + hcalIso1) > cut[9] + cut[10] * (scEt - 50))))
      result = 0;
    else
      result = 2;
  } else {
    if ((tkIso > cut[6]) || (ecalIso > cut[7]) || (hcalIso > cut[8]) || (hcalIso1 > cut[9]) || (hcalIso2 > cut[10]) ||
        (tkIso / electron->p4().Pt() > cut[11]) || (ecalIso / electron->p4().Pt() > cut[12]) ||
        (hcalIso / electron->p4().Pt() > cut[13]) || ((tkIso + ecalIso + hcalIso) > cut[14]) ||
        (((tkIso + ecalIso + hcalIso) / electron->p4().Pt()) > cut[15]) || ((tkIso + ecalIsoPed + hcalIso) > cut[16]) ||
        (((tkIso + ecalIsoPed + hcalIso) / electron->p4().Pt()) > cut[17]))
      result = 0.;
    else
      result = 2.;
  }

  if ((hOverE < cut[0]) && (sigmaee < cut[1]) && (fabs(deltaPhiIn) < cut[2]) && (fabs(deltaEtaIn) < cut[3]) &&
      (e25Maxoe55 > cut[4] && e15oe55 > cut[5]) && (sigmaee >= cut[18]) && (eOverP > cut[19] && eOverP < cut[20])) {
    result = result + 1;
  }

  if (ip > cut[21])
    return result;
  if (mishits > cut[22])  // expected missing hits
    return result;
  // positive cut[23] means to demand a valid hit in 1st layer PXB
  if (cut[23] > 0 &&
      !electron->gsfTrack()->hitPattern().hasValidHitInPixelLayer(PixelSubdetector::SubDetector::PixelBarrel, 1))
    return result;

  // cut[24]: Dist cut[25]: dcot
  float dist = fabs(electron->convDist());
  float dcot = fabs(electron->convDcot());
  bool isConversion = (cut[24] > 99. || cut[25] > 99.) ? false : (dist < cut[24] && dcot < cut[25]);
  if (isConversion)
    return result;

  result += 4;

  return result;
}
