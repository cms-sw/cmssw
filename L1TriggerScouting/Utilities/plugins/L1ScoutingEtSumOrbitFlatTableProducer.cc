#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "DataFormats/NanoAOD/interface/OrbitFlatTable.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "L1TriggerScouting/Utilities/interface/conversion.h"

class L1ScoutingEtSumOrbitFlatTableProducer : public edm::stream::EDProducer<> {
public:
  explicit L1ScoutingEtSumOrbitFlatTableProducer(const edm::ParameterSet&);
  ~L1ScoutingEtSumOrbitFlatTableProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  std::unique_ptr<l1ScoutingRun3::OrbitFlatTable> produceSingle(l1ScoutingRun3::BxSumsOrbitCollection const&) const;
  std::unique_ptr<l1ScoutingRun3::OrbitFlatTable> produceMultiple(l1ScoutingRun3::BxSumsOrbitCollection const&) const;

  edm::EDGetTokenT<l1ScoutingRun3::BxSumsOrbitCollection> src_;

  std::string name_;
  std::string doc_;
  bool singleton_;
  bool writePhysicalValues_;
  bool writeHardwareValues_;
  bool writeHF_;
  bool writeAsym_;
  bool writeMinBias_;
  bool writeTowerCount_;
  bool writeCentrality_;
  int ptPrecision_;
  int phiPrecision_;
};

L1ScoutingEtSumOrbitFlatTableProducer::L1ScoutingEtSumOrbitFlatTableProducer(const edm::ParameterSet& params)
    : src_(consumes<OrbitCollection<l1ScoutingRun3::BxSums>>(params.getParameter<edm::InputTag>("src"))),
      name_(params.getParameter<std::string>("name")),
      doc_(params.getParameter<std::string>("doc")),
      singleton_(params.getParameter<bool>("singleton")),
      writePhysicalValues_(params.getParameter<bool>("writePhysicalValues")),
      writeHardwareValues_(params.getParameter<bool>("writeHardwareValues")),
      writeHF_(params.getParameter<bool>("writeHF")),
      writeAsym_(params.getParameter<bool>("writeAsym")),
      writeMinBias_(params.getParameter<bool>("writeMinBias")),
      writeTowerCount_(params.getParameter<bool>("writeTowerCount")),
      writeCentrality_(params.getParameter<bool>("writeCentrality")),
      ptPrecision_(params.getParameter<int>("ptPrecision")),
      phiPrecision_(params.getParameter<int>("phiPrecision")) {
  if (!writePhysicalValues_ && !writeHardwareValues_) {
    throw cms::Exception("L1ScoutingEtSumOrbitFlatTableProducer")
        << "writePhysicalValues and writeHardwareValues cannot be false at the same time!";
  }
  produces<l1ScoutingRun3::OrbitFlatTable>();
}

void L1ScoutingEtSumOrbitFlatTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src");
  desc.add<std::string>("name");
  desc.add<std::string>("doc");
  desc.add<bool>("singleton", true)
      ->setComment("whether to output as singleton (one EtSum per bx) or not (multiple EtSums per bx)");
  desc.add<bool>("writePhysicalValues", true);
  desc.add<bool>("writeHardwareValues", false);
  desc.add<bool>("writeHF", true);
  desc.add<bool>("writeAsym", true);
  desc.add<bool>("writeMinBias", true);
  desc.add<bool>("writeTowerCount", true);
  desc.add<bool>("writeCentrality", true);
  desc.add<int>("ptPrecision", -1);
  desc.add<int>("phiPrecision", -1);

  descriptions.addDefault(desc);
}

void L1ScoutingEtSumOrbitFlatTableProducer::produce(edm::Event& iEvent, edm::EventSetup const&) {
  edm::Handle<l1ScoutingRun3::BxSumsOrbitCollection> src;
  iEvent.getByToken(src_, src);

  auto out = singleton_ ? produceSingle(*src) : produceMultiple(*src);
  iEvent.put(std::move(out));
}

std::unique_ptr<l1ScoutingRun3::OrbitFlatTable> L1ScoutingEtSumOrbitFlatTableProducer::produceSingle(
    l1ScoutingRun3::BxSumsOrbitCollection const& src) const {
  using namespace l1ScoutingRun3;
  auto out = std::make_unique<l1ScoutingRun3::OrbitFlatTable>(src.bxOffsets(), name_, /*singleton=*/true);
  out->setDoc(doc_);

  unsigned int nobjs = out->size();

  // physical values (float)
  std::vector<float> totalEt(nobjs);
  std::vector<float> totalEtEm(nobjs);
  std::vector<float> missEt(nobjs);
  std::vector<float> missEtPhi(nobjs);
  std::vector<float> missEtHF(nobjs);
  std::vector<float> missEtHFPhi(nobjs);
  std::vector<float> totalHt(nobjs);
  std::vector<float> missHt(nobjs);
  std::vector<float> missHtPhi(nobjs);
  std::vector<float> missHtHF(nobjs);
  std::vector<float> missHtHFPhi(nobjs);
  std::vector<float> asymEt(nobjs);
  std::vector<float> asymHt(nobjs);
  std::vector<float> asymEtHF(nobjs);
  std::vector<float> asymHtHF(nobjs);

  // hardware values (int)
  std::vector<int> hwTotalEt(nobjs);
  std::vector<int> hwTotalEtEm(nobjs);
  std::vector<int> hwMissEt(nobjs);
  std::vector<int> hwMissEtPhi(nobjs);
  std::vector<int> hwMissEtHF(nobjs);
  std::vector<int> hwMissEtHFPhi(nobjs);
  std::vector<int> hwTotalHt(nobjs);
  std::vector<int> hwMissHt(nobjs);
  std::vector<int> hwMissHtPhi(nobjs);
  std::vector<int> hwMissHtHF(nobjs);
  std::vector<int> hwMissHtHFPhi(nobjs);
  std::vector<int> hwAsymEt(nobjs);
  std::vector<int> hwAsymHt(nobjs);
  std::vector<int> hwAsymEtHF(nobjs);
  std::vector<int> hwAsymHtHF(nobjs);

  std::vector<int> minBiasHFP0(nobjs);
  std::vector<int> minBiasHFM0(nobjs);
  std::vector<int> minBiasHFP1(nobjs);
  std::vector<int> minBiasHFM1(nobjs);
  std::vector<int> towerCount(nobjs);
  std::vector<int> centrality(nobjs);

  for (unsigned int i = 0; i < nobjs; i++) {
    const auto& sums = src[i];

    // physical values
    totalEt[i] = demux::fEt(sums.hwTotalEt());
    totalEtEm[i] = demux::fEt(sums.hwTotalEtEm());
    missEt[i] = demux::fEt(sums.hwMissEt());
    missEtPhi[i] = demux::fPhi(sums.hwMissEtPhi());
    missEtHF[i] = demux::fEt(sums.hwMissEtHF());
    missEtHFPhi[i] = demux::fPhi(sums.hwMissEtHFPhi());
    totalHt[i] = demux::fEt(sums.hwTotalHt());
    missHt[i] = demux::fEt(sums.hwMissHt());
    missHtPhi[i] = demux::fPhi(sums.hwMissHtPhi());
    missHtHF[i] = demux::fEt(sums.hwMissHtHF());
    missHtHFPhi[i] = demux::fPhi(sums.hwMissHtHFPhi());
    asymEt[i] = demux::fEt(sums.hwAsymEt());
    asymHt[i] = demux::fEt(sums.hwAsymHt());
    asymEtHF[i] = demux::fEt(sums.hwAsymEtHF());
    asymHtHF[i] = demux::fEt(sums.hwAsymHtHF());

    // hardware values
    hwTotalEt[i] = sums.hwTotalEt();
    hwTotalEtEm[i] = sums.hwTotalEtEm();
    hwMissEt[i] = sums.hwMissEt();
    hwMissEtPhi[i] = sums.hwMissEtPhi();
    hwMissEtHF[i] = sums.hwMissEtHF();
    hwMissEtHFPhi[i] = sums.hwMissEtHFPhi();
    hwTotalHt[i] = sums.hwTotalHt();
    hwMissHt[i] = sums.hwMissHt();
    hwMissHtPhi[i] = sums.hwMissHtPhi();
    hwMissHtHF[i] = sums.hwMissHtHF();
    hwMissHtHFPhi[i] = sums.hwMissHtHFPhi();
    hwAsymEt[i] = sums.hwAsymEt();
    hwAsymHt[i] = sums.hwAsymHt();
    hwAsymEtHF[i] = sums.hwAsymEtHF();
    hwAsymHtHF[i] = sums.hwAsymHtHF();

    minBiasHFP0[i] = sums.minBiasHFP0();
    minBiasHFM0[i] = sums.minBiasHFM0();
    minBiasHFP1[i] = sums.minBiasHFP1();
    minBiasHFM1[i] = sums.minBiasHFM1();
    towerCount[i] = sums.towerCount();
    centrality[i] = sums.centrality();
  }

  // fill table

  if (writePhysicalValues_) {
    out->template addColumn<float>("totalEt", totalEt, "totalEt", ptPrecision_);
    out->template addColumn<float>("totalEtEm", totalEtEm, "totalEtEm", ptPrecision_);
    out->template addColumn<float>("missEt", missEt, "missEt pt", ptPrecision_);
    out->template addColumn<float>("missEtPhi", missEtPhi, "missEt phi", phiPrecision_);
    out->template addColumn<float>("totalHt", totalHt, "totalHt", ptPrecision_);
    out->template addColumn<float>("missHt", missHt, "missHt pt", ptPrecision_);
    out->template addColumn<float>("missHtPhi", missHtPhi, "missHt phi", phiPrecision_);
  }
  if (writeHardwareValues_) {
    out->template addColumn<int>("hwTotalEt", hwTotalEt, "hardware totalEt");
    out->template addColumn<int>("hwTotalEtEm", hwTotalEtEm, "hardware totalEtEm");
    out->template addColumn<int>("hwMissEt", hwMissEt, "hardware missEt pt");
    out->template addColumn<int>("hwMissEtPhi", hwMissEtPhi, "hardware missEt phi");
    out->template addColumn<int>("hwTotalHt", hwTotalHt, "hardware totalHt");
    out->template addColumn<int>("hwMissHt", hwMissHt, "hardware missHt pt");
    out->template addColumn<int>("hwMissHtPhi", hwMissHtPhi, "hardware missHt phi");
  }

  if (writeHF_) {
    if (writePhysicalValues_) {
      out->template addColumn<float>("missEtHF", missEtHF, "missEtHF", ptPrecision_);
      out->template addColumn<float>("missEtHFPhi", missEtHFPhi, "missEtHF phi", phiPrecision_);
      out->template addColumn<float>("missHtHF", missHtHF, "missHtHF pt", ptPrecision_);
      out->template addColumn<float>("missHtHFPhi", missHtHFPhi, "missHtHF phi", phiPrecision_);
    }
    if (writeHardwareValues_) {
      out->template addColumn<int>("hwMissEtHF", hwMissEtHF, "hardware missEtHF");
      out->template addColumn<int>("hwMissEtHFPhi", hwMissEtHFPhi, "hardware missEtHF phi");
      out->template addColumn<int>("hwMissHtHF", hwMissHtHF, "hardware missHtHF");
      out->template addColumn<int>("hwMissHtHFPhi", hwMissHtHFPhi, "hardware missHtHF phi");
    }
  }
  if (writeAsym_) {
    if (writePhysicalValues_) {
      out->template addColumn<float>("asymEt", asymEt, "asymEt", ptPrecision_);
      out->template addColumn<float>("asymHt", asymHt, "asymHt", ptPrecision_);
    }
    if (writeHardwareValues_) {
      out->template addColumn<int>("hwAsymEt", hwAsymEt, "hardware asymEt");
      out->template addColumn<int>("hwAsymHt", hwAsymHt, "hardware asymHt");
    }
  }

  if (writeAsym_ && writeHF_) {
    if (writePhysicalValues_) {
      out->template addColumn<float>("asymEtHF", asymEtHF, "asymEtHF", ptPrecision_);
      out->template addColumn<float>("asymHtHF", asymHtHF, "asymHtHF", ptPrecision_);
    }
    if (writeHardwareValues_) {
      out->template addColumn<int>("hwAsymEtHF", hwAsymEtHF, "asymEtHF");
      out->template addColumn<int>("hwAsymHtHF", hwAsymHtHF, "asymHtHF");
    }
  }

  if (writeMinBias_) {
    out->template addColumn<int>("minBiasHFP0", minBiasHFP0, "minBiasHFP0");
    out->template addColumn<int>("minBiasHFM0", minBiasHFM0, "minBiasHFM0");
    out->template addColumn<int>("minBiasHFP1", minBiasHFP1, "minBiasHFP1");
    out->template addColumn<int>("minBiasHFM1", minBiasHFM1, "minBiasHFM1");
  }

  if (writeTowerCount_) {
    out->template addColumn<int>("towerCount", towerCount, "towerCount");
  }

  if (writeCentrality_) {
    out->template addColumn<int>("centrality", centrality, "centrality");
  }

  return out;
}

std::unique_ptr<l1ScoutingRun3::OrbitFlatTable> L1ScoutingEtSumOrbitFlatTableProducer::produceMultiple(
    l1ScoutingRun3::BxSumsOrbitCollection const& src) const {
  using namespace l1ScoutingRun3;
  // compute number of objects per bx to adjust bxOffsets
  unsigned int nitems = 5;  // totalEt, totalEtEm, missEt, totalHt, missHt
  if (writeHF_)
    nitems += 2;  // missEtHF, missHtHF
  if (writeAsym_)
    nitems += (writeHF_ ? 4 : 2);  // asymEt, asymHt, asymEtHF, asymHtHF
  if (writeMinBias_)
    nitems += 4;  // minBiasHFP0, minBiasHFM0, minBiasHFP1, minBiasHFM1
  if (writeTowerCount_)
    nitems += 1;  // towerCount
  if (writeCentrality_)
    nitems += 1;  // centrality

  // adjust bxOffsets since each bx now contains multiple objects instead of single object
  std::vector<unsigned> offsets(src.bxOffsets());
  for (auto& v : offsets)
    v *= nitems;

  auto out = std::make_unique<l1ScoutingRun3::OrbitFlatTable>(offsets, name_, /*singleton=*/false);
  out->setDoc(doc_);

  unsigned int nobjs = out->size();

  // physical values
  std::vector<float> pt(nobjs);
  std::vector<float> phi(nobjs, 0.);

  // hardware values
  std::vector<int> hwEt(nobjs);
  std::vector<int> hwPhi(nobjs, 0);

  std::vector<int> sumType(nobjs);

  unsigned int i = 0;
  for (const l1ScoutingRun3::BxSums& sums : src) {
    assert(i + nitems <= nobjs && i % nitems == 0);

    // totalEt
    pt[i] = demux::fEt(sums.hwTotalEt());
    hwEt[i] = sums.hwTotalEt();
    sumType[i++] = l1t::EtSum::kTotalEt;
    // totalEtEm
    pt[i] = demux::fEt(sums.hwTotalEtEm());
    hwEt[i] = sums.hwTotalEtEm();
    sumType[i++] = l1t::EtSum::kTotalEtEm;
    // missEt
    pt[i] = demux::fEt(sums.hwMissEt());
    phi[i] = demux::fPhi(sums.hwMissEtPhi());
    hwEt[i] = sums.hwMissEt();
    hwPhi[i] = sums.hwMissEtPhi();
    sumType[i++] = l1t::EtSum::kMissingEt;
    // totalHt
    pt[i] = demux::fEt(sums.hwTotalHt());
    hwEt[i] = sums.hwTotalHt();
    sumType[i++] = l1t::EtSum::kTotalHt;
    // missHt
    pt[i] = demux::fEt(sums.hwMissHt());
    phi[i] = demux::fPhi(sums.hwMissHtPhi());
    hwEt[i] = sums.hwMissHt();
    hwPhi[i] = sums.hwMissHtPhi();
    sumType[i++] = l1t::EtSum::kMissingHt;

    if (writeHF_) {
      // missEtHF
      pt[i] = demux::fEt(sums.hwMissEtHF());
      phi[i] = demux::fPhi(sums.hwMissEtHFPhi());
      hwEt[i] = sums.hwMissEtHF();
      hwPhi[i] = sums.hwMissEtHFPhi();
      sumType[i++] = l1t::EtSum::kMissingEtHF;
      // missHtHF
      pt[i] = demux::fEt(sums.hwMissHtHF());
      phi[i] = demux::fPhi(sums.hwMissHtHFPhi());
      hwEt[i] = sums.hwMissHtHF();
      hwPhi[i] = sums.hwMissHtHFPhi();
      sumType[i++] = l1t::EtSum::kMissingHtHF;
    }

    if (writeAsym_) {
      // asymEt
      pt[i] = demux::fEt(sums.hwAsymEt());
      hwEt[i] = sums.hwAsymEt();
      sumType[i++] = l1t::EtSum::kAsymEt;
      // asymHt
      pt[i] = demux::fEt(sums.hwAsymHt());
      hwEt[i] = sums.hwAsymHt();
      sumType[i++] = l1t::EtSum::kAsymHt;

      if (writeHF_) {
        // asymEtHF
        pt[i] = demux::fEt(sums.hwAsymEtHF());
        hwEt[i] = sums.hwAsymEtHF();
        sumType[i++] = l1t::EtSum::kAsymEtHF;
        // asymHtHF
        pt[i] = demux::fEt(sums.hwAsymHtHF());
        hwEt[i] = sums.hwAsymHtHF();
        sumType[i++] = l1t::EtSum::kAsymHtHF;
      }
    }

    if (writeMinBias_) {
      // minBiasHFP0
      pt[i] = sums.minBiasHFP0();
      hwEt[i] = sums.minBiasHFP0();
      sumType[i++] = l1t::EtSum::kMinBiasHFP0;
      // minBiasHFM0
      pt[i] = sums.minBiasHFM0();
      hwEt[i] = sums.minBiasHFM0();
      sumType[i++] = l1t::EtSum::kMinBiasHFM0;
      // minBiasHFP1
      pt[i] = sums.minBiasHFP1();
      hwEt[i] = sums.minBiasHFP1();
      sumType[i++] = l1t::EtSum::kMinBiasHFP1;
      // minBiasHFM1
      pt[i] = sums.minBiasHFM1();
      hwEt[i] = sums.minBiasHFM1();
      sumType[i++] = l1t::EtSum::kMinBiasHFM1;
    }

    if (writeTowerCount_) {
      // towerCount
      pt[i] = sums.towerCount();
      hwEt[i] = sums.towerCount();
      sumType[i++] = l1t::EtSum::kTowerCount;
    }

    if (writeCentrality_) {
      // centrality
      pt[i] = sums.centrality();
      hwEt[i] = sums.centrality();
      sumType[i++] = l1t::EtSum::kCentrality;
    }
  }

  // fill table

  if (writePhysicalValues_) {
    out->template addColumn<float>("pt", pt, "pt", ptPrecision_);
    out->template addColumn<float>("phi", phi, "phi", phiPrecision_);
  }
  if (writeHardwareValues_) {
    out->template addColumn<int>("hwEt", pt, "hardware Et");
    out->template addColumn<int>("hwPhi", phi, "hardware phi");
  }

  out->template addColumn<int>(
      "etSumType",
      sumType,
      "the type of the EtSum "
      "(https://github.com/cms-sw/cmssw/blob/master/DataFormats/L1Trigger/interface/EtSum.h#L27-L56)");
  return out;
}

DEFINE_FWK_MODULE(L1ScoutingEtSumOrbitFlatTableProducer);
