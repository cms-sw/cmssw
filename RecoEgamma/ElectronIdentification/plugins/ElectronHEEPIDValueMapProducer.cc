#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EleTkIsolFromCands.h"

#include <memory>
#include <vector>

using IsolationCalculators = std::vector<std::unique_ptr<EleTkIsolFromCands>>;

//Heavily inspired from ElectronIDValueMapProducer
class ElectronHEEPIDValueMapProducer : public edm::stream::EDProducer<> {
private:
  //helper classes to handle AOD vs MiniAOD
  template <typename T>
  struct DualToken {
    edm::EDGetTokenT<T> aod;
    edm::EDGetTokenT<T> miniAOD;
  };
  class DataFormat {
  public:
    enum Format { AUTO = 0, AOD = 1, MINIAOD = 2 };

  private:
    int data_;

  public:
    DataFormat(int val) : data_(val) {}
    bool tryAOD() const { return data_ == AUTO || data_ == AOD; }
    bool tryMiniAOD() const { return data_ == AUTO || data_ == MINIAOD; }
    int operator()() const { return data_; }
  };

public:
  explicit ElectronHEEPIDValueMapProducer(const edm::ParameterSet&);
  ~ElectronHEEPIDValueMapProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  template <typename T>
  static void writeValueMap(edm::Event& iEvent,
                            const edm::Handle<edm::View<reco::GsfElectron>>& handle,
                            const std::vector<T>& values,
                            const std::string& label);

  static int nrSaturatedCrysIn5x5(const reco::GsfElectron& ele,
                                  edm::Handle<EcalRecHitCollection>& ebHits,
                                  edm::Handle<EcalRecHitCollection>& eeHits,
                                  edm::ESHandle<CaloTopology>& caloTopo);

  static float calTrkIso(const reco::GsfElectron& ele, IsolationCalculators const& isolationCalculators);

  template <typename T>
  void setToken(edm::EDGetTokenT<T>& token, edm::InputTag tag) {
    token = consumes<T>(tag);
  }
  template <typename T>
  void setToken(edm::EDGetTokenT<T>& token, const edm::ParameterSet& iPara, const std::string& tag) {
    token = consumes<T>(iPara.getParameter<edm::InputTag>(tag));
  }
  template <typename T>
  void setToken(std::vector<edm::EDGetTokenT<T>>& tokens, const edm::ParameterSet& iPara, const std::string& tagName) {
    auto tags = iPara.getParameter<std::vector<edm::InputTag>>(tagName);
    for (auto& tag : tags) {
      edm::EDGetTokenT<T> token;
      setToken(token, tag);
      tokens.push_back(token);
    }
  }
  template <typename T>
  void setToken(DualToken<T>& token,
                const edm::ParameterSet& iPara,
                const std::string& tagAOD,
                const std::string& tagMiniAOD,
                DataFormat format) {
    if (format.tryAOD())
      token.aod = consumes<T>(iPara.getParameter<edm::InputTag>(tagAOD));
    if (format.tryMiniAOD())
      token.miniAOD = consumes<T>(iPara.getParameter<edm::InputTag>(tagMiniAOD));
  }
  template <typename T>
  void setToken(std::vector<DualToken<T>>& tokens,
                const edm::ParameterSet& iPara,
                const std::string& tagAOD,
                const std::string& tagMiniAOD,
                DataFormat format) {
    auto tagsAOD = iPara.getParameter<std::vector<edm::InputTag>>(tagAOD);
    auto tagsMiniAOD = iPara.getParameter<std::vector<edm::InputTag>>(tagMiniAOD);
    size_t maxSize = std::max(tagsAOD.size(), tagsMiniAOD.size());
    tokens.clear();
    tokens.resize(maxSize);
    if (format.tryAOD()) {
      for (size_t tagNr = 0; tagNr < tagsAOD.size(); tagNr++) {
        setToken(tokens[tagNr].aod, tagsAOD[tagNr]);
      }
    }
    if (format.tryMiniAOD()) {
      for (size_t tagNr = 0; tagNr < tagsMiniAOD.size(); tagNr++) {
        setToken(tokens[tagNr].miniAOD, tagsMiniAOD[tagNr]);
      }
    }
  }

  template <typename T>
  static edm::Handle<T> getHandle(const edm::Event& iEvent, const DualToken<T>& token) {
    edm::Handle<T> handle;
    if (!token.aod.isUninitialized())
      iEvent.getByToken(token.aod, handle);
    if (!handle.isValid() && !token.miniAOD.isUninitialized())
      iEvent.getByToken(token.miniAOD, handle);
    return handle;
  }

  template <typename T>
  static std::vector<edm::Handle<T>> getHandles(const edm::Event& iEvent, const std::vector<DualToken<T>>& tokens) {
    std::vector<edm::Handle<T>> handles(tokens.size());
    if (tokens.empty())
      return handles;
    if (!tokens[0].aod.isUninitialized())
      iEvent.getByToken(tokens[0].aod, handles[0]);
    bool isAOD = handles[0].isValid();
    if (!isAOD && !tokens[0].miniAOD.isUninitialized())
      iEvent.getByToken(tokens[0].miniAOD, handles[0]);

    for (size_t tokenNr = 1; tokenNr < tokens.size(); tokenNr++) {
      auto token = isAOD ? tokens[tokenNr].aod : tokens[tokenNr].miniAOD;
      if (!token.isUninitialized())
        iEvent.getByToken(token, handles[tokenNr]);
    }
    return handles;
  }

  template <typename T>
  static bool isEventAOD(const edm::Event& iEvent, const DualToken<T>& token) {
    edm::Handle<T> handle;
    if (!token.aod.isUninitialized())
      iEvent.getByToken(token.aod, handle);
    if (handle.isValid())
      return true;
    else
      return false;
  }

  DualToken<EcalRecHitCollection> ebRecHitToken_;
  DualToken<EcalRecHitCollection> eeRecHitToken_;
  DualToken<edm::View<reco::GsfElectron>> eleToken_;
  std::vector<DualToken<pat::PackedCandidateCollection>> candTokens_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::ESGetToken<CaloTopology, CaloTopologyRecord> caloTopoToken_;

  EleTkIsolFromCands::Configuration trkIsoCalcCfg_;
  EleTkIsolFromCands::Configuration trkIso04CalcCfg_;
  bool makeTrkIso04_;
  DataFormat dataFormat_;
  std::vector<EleTkIsolFromCands::PIDVeto> candVetosAOD_;
  std::vector<EleTkIsolFromCands::PIDVeto> candVetosMiniAOD_;

  static const std::string eleTrkPtIsoLabel_;
  static const std::string eleTrkPtIso04Label_;
  static const std::string eleNrSaturateIn5x5Label_;
};

const std::string ElectronHEEPIDValueMapProducer::eleTrkPtIsoLabel_ = "eleTrkPtIso";
const std::string ElectronHEEPIDValueMapProducer::eleTrkPtIso04Label_ = "eleTrkPtIso04";
const std::string ElectronHEEPIDValueMapProducer::eleNrSaturateIn5x5Label_ = "eleNrSaturateIn5x5";

ElectronHEEPIDValueMapProducer::ElectronHEEPIDValueMapProducer(const edm::ParameterSet& iConfig)
    : trkIsoCalcCfg_(iConfig.getParameter<edm::ParameterSet>("trkIsoConfig")),
      trkIso04CalcCfg_(iConfig.getParameter<edm::ParameterSet>("trkIso04Config")),
      makeTrkIso04_(iConfig.getParameter<bool>("makeTrkIso04")),
      dataFormat_(iConfig.getParameter<int>("dataFormat")) {
  setToken(ebRecHitToken_, iConfig, "ebRecHitsAOD", "ebRecHitsMiniAOD", dataFormat_);
  setToken(eeRecHitToken_, iConfig, "eeRecHitsAOD", "eeRecHitsMiniAOD", dataFormat_);
  setToken(eleToken_, iConfig, "elesAOD", "elesMiniAOD", dataFormat_);
  setToken(candTokens_, iConfig, "candsAOD", "candsMiniAOD", dataFormat_);
  setToken(beamSpotToken_, iConfig, "beamSpot");
  caloTopoToken_ = esConsumes();

  auto fillVetos = [](const auto& in, auto& out) {
    std::transform(in.begin(), in.end(), std::back_inserter(out), EleTkIsolFromCands::pidVetoFromStr);
  };

  fillVetos(iConfig.getParameter<std::vector<std::string>>("candVetosAOD"), candVetosAOD_);
  if (candVetosAOD_.size() != iConfig.getParameter<std::vector<edm::InputTag>>("candsAOD").size()) {
    throw cms::Exception("ConfigError") << " Error candVetosAOD should be the same size as candsAOD " << std::endl;
  }

  fillVetos(iConfig.getParameter<std::vector<std::string>>("candVetosMiniAOD"), candVetosMiniAOD_);
  if (candVetosMiniAOD_.size() != iConfig.getParameter<std::vector<edm::InputTag>>("candsMiniAOD").size()) {
    throw cms::Exception("ConfigError") << " Error candVetosMiniAOD should be the same size as candsMiniAOD "
                                        << std::endl;
  }

  produces<edm::ValueMap<float>>(eleTrkPtIsoLabel_);
  if (makeTrkIso04_)
    produces<edm::ValueMap<float>>(eleTrkPtIso04Label_);
  produces<edm::ValueMap<int>>(eleNrSaturateIn5x5Label_);
}

ElectronHEEPIDValueMapProducer::~ElectronHEEPIDValueMapProducer() {}

void ElectronHEEPIDValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto eleHandle = getHandle(iEvent, eleToken_);
  auto ebRecHitHandle = getHandle(iEvent, ebRecHitToken_);
  auto eeRecHitHandle = getHandle(iEvent, eeRecHitToken_);
  auto beamSpotHandle = iEvent.getHandle(beamSpotToken_);

  bool isAOD = isEventAOD(iEvent, eleToken_);
  const auto& candVetos = isAOD ? candVetosAOD_ : candVetosMiniAOD_;

  IsolationCalculators trkIsoCalcs;
  IsolationCalculators trkIso04Calcs;

  {
    int handleNr = 0;
    for (auto const& handle : getHandles(iEvent, candTokens_)) {
      if (handle.isValid()) {
        trkIsoCalcs.emplace_back(std::make_unique<EleTkIsolFromCands>(trkIsoCalcCfg_, *handle, candVetos[handleNr]));
        if (makeTrkIso04_) {
          trkIso04Calcs.emplace_back(
              std::make_unique<EleTkIsolFromCands>(trkIso04CalcCfg_, *handle, candVetos[handleNr]));
        }
      }
      ++handleNr;
    }
  }

  edm::ESHandle<CaloTopology> caloTopoHandle = iSetup.getHandle(caloTopoToken_);

  std::vector<float> eleTrkPtIso;
  std::vector<float> eleTrkPtIso04;
  std::vector<int> eleNrSaturateIn5x5;
  for (auto const& ele : *eleHandle) {
    eleTrkPtIso.push_back(calTrkIso(ele, trkIsoCalcs));
    if (makeTrkIso04_) {
      eleTrkPtIso04.push_back(calTrkIso(ele, trkIso04Calcs));
    }
    eleNrSaturateIn5x5.push_back(nrSaturatedCrysIn5x5(ele, ebRecHitHandle, eeRecHitHandle, caloTopoHandle));
  }

  writeValueMap(iEvent, eleHandle, eleTrkPtIso, eleTrkPtIsoLabel_);
  if (makeTrkIso04_)
    writeValueMap(iEvent, eleHandle, eleTrkPtIso04, eleTrkPtIso04Label_);
  writeValueMap(iEvent, eleHandle, eleNrSaturateIn5x5, eleNrSaturateIn5x5Label_);
}

int ElectronHEEPIDValueMapProducer::nrSaturatedCrysIn5x5(const reco::GsfElectron& ele,
                                                         edm::Handle<EcalRecHitCollection>& ebHits,
                                                         edm::Handle<EcalRecHitCollection>& eeHits,
                                                         edm::ESHandle<CaloTopology>& caloTopo) {
  DetId id = ele.superCluster()->seed()->seed();
  auto recHits = id.subdetId() == EcalBarrel ? ebHits.product() : eeHits.product();
  return noZS::EcalClusterTools::nrSaturatedCrysIn5x5(id, recHits, caloTopo.product());
}

float ElectronHEEPIDValueMapProducer::calTrkIso(const reco::GsfElectron& ele,
                                                IsolationCalculators const& isolationCalculators) {
  if (ele.gsfTrack().isNull()) {
    return std::numeric_limits<float>::max();
  }

  float trkIso = 0.;
  for (auto& calculator : isolationCalculators) {
    trkIso += (*calculator)(*ele.gsfTrack()).ptSum;
  }
  return trkIso;
}

template <typename T>
void ElectronHEEPIDValueMapProducer::writeValueMap(edm::Event& iEvent,
                                                   const edm::Handle<edm::View<reco::GsfElectron>>& handle,
                                                   const std::vector<T>& values,
                                                   const std::string& label) {
  std::unique_ptr<edm::ValueMap<T>> valMap(new edm::ValueMap<T>());
  typename edm::ValueMap<T>::Filler filler(*valMap);
  filler.insert(handle, values.begin(), values.end());
  filler.fill();
  iEvent.put(std::move(valMap), label);
}

void ElectronHEEPIDValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("ebRecHitsAOD", edm::InputTag("reducedEcalRecHitsEB"));
  desc.add<edm::InputTag>("eeRecHitsAOD", edm::InputTag("reducedEcalRecHitsEE"));
  desc.add<std::vector<edm::InputTag>>("candsAOD", {edm::InputTag("packedCandidates")});
  desc.add<std::vector<std::string>>("candVetosAOD", {"none"});
  desc.add<edm::InputTag>("elesAOD", edm::InputTag("gedGsfElectrons"));

  desc.add<edm::InputTag>("ebRecHitsMiniAOD", edm::InputTag("reducedEcalRecHitsEB"));
  desc.add<edm::InputTag>("eeRecHitsMiniAOD", edm::InputTag("reducedEcalRecHitsEE"));
  desc.add<std::vector<edm::InputTag>>("candsMiniAOD", {edm::InputTag("packedCandidates")});
  desc.add<std::vector<std::string>>("candVetosMiniAOD", {"none"});
  desc.add<edm::InputTag>("elesMiniAOD", edm::InputTag("gedGsfElectrons"));
  desc.add<int>("dataFormat", 0);
  desc.add<bool>("makeTrkIso04", false);
  desc.add("trkIsoConfig", EleTkIsolFromCands::pSetDescript());
  desc.add("trkIso04Config", EleTkIsolFromCands::pSetDescript());

  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ElectronHEEPIDValueMapProducer);
