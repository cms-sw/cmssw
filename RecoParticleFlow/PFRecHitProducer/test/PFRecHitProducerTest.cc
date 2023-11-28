#include <algorithm>
#include <cstdio>
#include <string>
#include <optional>
#include <map>
#include <unordered_map>
#include <utility>
#include <variant>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ParticleFlowReco/interface/CaloRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"

class PFRecHitProducerTest : public DQMEDAnalyzer {
public:
  PFRecHitProducerTest(edm::ParameterSet const& conf);
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void endStream() override;

private:
  // Define generic types for token, handle and collection, such that origin of PFRecHits can be selected at runtime.
  // The idea is to read desired format from config (pfRecHitsType1/2) and construct token accordingly. Then use handle
  // and collection corresponding to token type. Finally, construct GenericPFRecHits from each source.
  // This way, this module can be used to validate any combination of legacy and Alpaka formats.

  using LegacyToken = edm::EDGetTokenT<reco::PFRecHitCollection>;
  using AlpakaToken = edm::EDGetTokenT<reco::PFRecHitHostCollection>;
  using GenericPFRecHitToken = std::variant<LegacyToken, AlpakaToken>;
  using LegacyHandle = edm::Handle<reco::PFRecHitCollection>;
  using AlpakaHandle = edm::Handle<reco::PFRecHitHostCollection>;
  using LegacyCollection = const reco::PFRecHitCollection*;
  using AlpakaCollection = const reco::PFRecHitHostCollection::ConstView*;
  using GenericCollection = std::variant<LegacyCollection, AlpakaCollection>;

  static size_t GenericCollectionSize(const GenericCollection& collection) {
    return std::visit([](auto& c) -> size_t { return c->size(); }, collection);
  };

  std::optional<edm::EDGetTokenT<reco::CaloRecHitHostCollection>> caloRecHitsToken_{};
  GenericPFRecHitToken pfRecHitsTokens_[2];

  void dumpEvent(const edm::Event&, const GenericCollection&, const GenericCollection&);
  int32_t num_events_ = 0, num_errors_ = 0;
  std::map<int32_t, uint32_t> errors_;
  const std::string title_;
  const bool strictCompare_, dumpFirstEvent_, dumpFirstError_;
  MonitorElement *hist_energy_, *hist_time_;

  // Container for PFRecHit, independent of how it was constructed
  struct GenericPFRecHit {
    uint32_t detId;
    int depth;
    PFLayer::Layer layer;
    float time;
    float energy;
    float x, y, z;
    std::vector<uint32_t> neighbours4;  // nearest neighbours
    std::vector<uint32_t> neighbours8;  // non-nearest neighbours

    static GenericPFRecHit Construct(const GenericCollection& collection, size_t i) {  // Select appropriate constructor
      return std::visit([i](auto& c) { return GenericPFRecHit{(*c)[i]}; }, collection);
    };
    GenericPFRecHit(const reco::PFRecHit& pfRecHit);  // Constructor from legacy format
    GenericPFRecHit(
        const reco::PFRecHitHostCollection::ConstView::const_element pfRecHitsAlpaka);  // Constructor from Alpaka SoA

    void print(const char* prefix, size_t idx);
  };
};

PFRecHitProducerTest::PFRecHitProducerTest(const edm::ParameterSet& conf)
    : title_(conf.getUntrackedParameter<std::string>("title")),
      strictCompare_(conf.getUntrackedParameter<bool>("strictCompare")),
      dumpFirstEvent_(conf.getUntrackedParameter<bool>("dumpFirstEvent")),
      dumpFirstError_(conf.getUntrackedParameter<bool>("dumpFirstError")) {
  if (conf.existsAs<edm::InputTag>("caloRecHits"))
    caloRecHitsToken_.emplace(consumes(conf.getUntrackedParameter<edm::InputTag>("caloRecHits")));

  const edm::InputTag input[2] = {conf.getUntrackedParameter<edm::InputTag>("pfRecHitsSource1"),
                                  conf.getUntrackedParameter<edm::InputTag>("pfRecHitsSource2")};
  const std::string type[2] = {conf.getUntrackedParameter<std::string>("pfRecHitsType1"),
                               conf.getUntrackedParameter<std::string>("pfRecHitsType2")};
  for (int i = 0; i < 2; i++) {
    if (type[i] == "legacy")
      pfRecHitsTokens_[i].emplace<LegacyToken>(consumes<LegacyHandle::element_type>(input[i]));
    else if (type[i] == "alpaka")
      pfRecHitsTokens_[i].emplace<AlpakaToken>(consumes<AlpakaHandle::element_type>(input[i]));
    else {
      fprintf(stderr, "Invalid value for PFRecHitProducerTest::pfRecHitsType%d: \"%s\"\n", i + 1, type[i].c_str());
      throw;
    }
  }
}

void PFRecHitProducerTest::endStream() {
  fprintf(stderr,
          "PFRecHitProducerTest%s%s%s has compared %u events and found %u problems: [%u, %u, %u, %u, %u]\n",
          title_.empty() ? "" : "[",
          title_.c_str(),
          title_.empty() ? "" : "]",
          num_events_,
          num_errors_,
          errors_[1],   // different number of PFRecHits
          errors_[2],   // detId not found
          errors_[3],   // depth,layer,time,energy or pos different
          errors_[4],   // different number of neighbours
          errors_[5]);  // neighbours different
}

void PFRecHitProducerTest::analyze(const edm::Event& event, const edm::EventSetup&) {
  GenericCollection pfRecHits[2];
  for (int i = 0; i < 2; i++)
    if (std::holds_alternative<LegacyToken>(pfRecHitsTokens_[i])) {
      pfRecHits[i].emplace<LegacyCollection>(&event.get(std::get<LegacyToken>(pfRecHitsTokens_[i])));
    } else {
      pfRecHits[i].emplace<AlpakaCollection>(&event.get(std::get<AlpakaToken>(pfRecHitsTokens_[i])).const_view());
    }

  int error = 0;
  const size_t n = GenericCollectionSize(pfRecHits[0]);
  if (n != GenericCollectionSize(pfRecHits[1]))
    error = 1;  // different number of PFRecHits
  else {
    std::vector<GenericPFRecHit> first, second;
    std::unordered_map<uint32_t, size_t> detId2Idx;  // for second vector
    first.reserve(n);
    second.reserve(n);
    for (size_t i = 0; i < n; i++) {
      first.emplace_back(GenericPFRecHit::Construct(pfRecHits[0], i));
      second.emplace_back(GenericPFRecHit::Construct(pfRecHits[1], i));
      detId2Idx[second.at(i).detId] = i;
    }
    for (size_t i = 0; i < n && error == 0; i++) {
      error = [&]() {
        const GenericPFRecHit& rh1 = first.at(i);
        if (detId2Idx.find(rh1.detId) == detId2Idx.end())
          return 2;  // detId not found

        const GenericPFRecHit& rh2 = second.at(detId2Idx.at(rh1.detId));
        assert(rh1.detId == rh2.detId);
        if (rh1.depth != rh2.depth || rh1.layer != rh2.layer || rh1.x != rh2.x || rh1.y != rh2.y || rh1.z != rh2.z)
          return 3;  // depth, layer or pos different
        if (strictCompare_ && (rh1.time != rh2.time || rh1.energy != rh2.energy))
          return 3;  // time or energy different
        hist_energy_->Fill(rh1.energy, rh2.energy);
        hist_time_->Fill(rh1.time, rh2.time);

        if (rh1.neighbours4.size() != rh2.neighbours4.size() || rh1.neighbours8.size() != rh2.neighbours8.size())
          return 4;  // different number of neighbours

        for (size_t i = 0; i < rh1.neighbours4.size(); i++)
          if (first.at(rh1.neighbours4[i]).detId != second.at(rh2.neighbours4[i]).detId)
            return 5;  // neighbours4 different
        for (size_t i = 0; i < rh1.neighbours8.size(); i++)
          if (first.at(rh1.neighbours8[i]).detId != second.at(rh2.neighbours8[i]).detId)
            return 5;  // neighbours8 different

        return 0;  // no error
      }();
    }
  }

  if (dumpFirstEvent_ && num_events_ == 0)
    dumpEvent(event, pfRecHits[0], pfRecHits[1]);

  if (error) {
    if (dumpFirstError_ && num_errors_ == 0) {
      printf("Error: %d\n", error);
      dumpEvent(event, pfRecHits[0], pfRecHits[1]);
    }
    num_errors_++;
    errors_[error]++;
  }
  num_events_++;
}

void PFRecHitProducerTest::dumpEvent(const edm::Event& event,
                                     const GenericCollection& pfRecHits1,
                                     const GenericCollection& pfRecHits2) {
  if (caloRecHitsToken_) {
    edm::Handle<reco::CaloRecHitHostCollection> caloRecHits;
    event.getByToken(*caloRecHitsToken_, caloRecHits);
    const reco::CaloRecHitHostCollection::ConstView view = caloRecHits->view();
    printf("Found %d recHits\n", view.metadata().size());
    for (int i = 0; i < view.metadata().size(); i++)
      printf("recHit %4d detId:%u energy:%f time:%f flags:%d\n",
             i,
             view.detId(i),
             view.energy(i),
             view.time(i),
             view.flags(i));
  }

  printf("Found %zd/%zd pfRecHits from first/second origin\n",
         GenericCollectionSize(pfRecHits1),
         GenericCollectionSize(pfRecHits2));
  for (size_t i = 0; i < GenericCollectionSize(pfRecHits1); i++)
    GenericPFRecHit::Construct(pfRecHits1, i).print("First", i);
  for (size_t i = 0; i < GenericCollectionSize(pfRecHits2); i++)
    GenericPFRecHit::Construct(pfRecHits2, i).print("Second", i);
}

void PFRecHitProducerTest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addOptionalUntracked<edm::InputTag>("caloRecHits")
      ->setComment("CaloRecHitSoA, if supplied, it is dumped alongside the PFRecHits");
  desc.addUntracked<edm::InputTag>("pfRecHitsSource1")->setComment("First PFRecHit list for comparison");
  desc.addUntracked<edm::InputTag>("pfRecHitsSource2")->setComment("Second PFRecHit list for comparison");
  desc.ifValue(edm::ParameterDescription<std::string>("pfRecHitsType1", "legacy", false),
               edm::allowedValues<std::string>("legacy", "alpaka"))
      ->setComment("Format of first PFRecHit list (legacy or alpaka)");
  desc.ifValue(edm::ParameterDescription<std::string>("pfRecHitsType2", "alpaka", false),
               edm::allowedValues<std::string>("legacy", "alpaka"))
      ->setComment("Format of second PFRecHit list (legacy or alpaka)");
  desc.addUntracked<std::string>("title", "")->setComment("Module name for printout");
  desc.addUntracked<bool>("dumpFirstEvent", false)
      ->setComment("Dump PFRecHits of first event, regardless of result of comparison");
  desc.addUntracked<bool>("dumpFirstError", false)->setComment("Dump PFRecHits upon first encountered error");
  desc.addUntracked<bool>("strictCompare", false)->setComment("Compare all floats for equality");
  descriptions.addDefault(desc);
}

void PFRecHitProducerTest::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder("ParticleFlow/PFRecHitV");
  hist_energy_ = ibooker.book2D("energy", "energy;Input1;Input2;Entries", 100, 0, 100, 100, 0, 100);
  hist_time_ = ibooker.book2D("time", "time;Input1;Input2;Entries", 100, 0, 100, 100, 0, 100);
}

PFRecHitProducerTest::GenericPFRecHit::GenericPFRecHit(const reco::PFRecHit& pfRecHit)
    : detId(pfRecHit.detId()),
      depth(pfRecHit.depth()),
      layer(pfRecHit.layer()),
      time(pfRecHit.time()),
      energy(pfRecHit.energy()),
      x(pfRecHit.position().x()),
      y(pfRecHit.position().y()),
      z(pfRecHit.position().z()) {
  // Fill neighbours4 and neighbours8, then remove elements of neighbours4 from neighbours8
  // This is necessary, because there can be duplicates in the neighbour lists
  // This procedure correctly accounts for these multiplicities
  reco::PFRecHit::Neighbours pfRecHitNeighbours4 = pfRecHit.neighbours4();
  reco::PFRecHit::Neighbours pfRecHitNeighbours8 = pfRecHit.neighbours8();
  neighbours4.reserve(4);
  neighbours8.reserve(8);
  for (auto p = pfRecHitNeighbours8.begin(); p < pfRecHitNeighbours8.end(); p++)
    neighbours8.emplace_back(*p);
  for (auto p = pfRecHitNeighbours4.begin(); p < pfRecHitNeighbours4.end(); p++) {
    neighbours4.emplace_back(*p);
    auto idx = std::find(neighbours8.begin(), neighbours8.end(), *p);
    std::copy(idx + 1, neighbours8.end(), idx);
  }
  neighbours8.resize(pfRecHitNeighbours8.size() - pfRecHitNeighbours4.size());
}

PFRecHitProducerTest::GenericPFRecHit::GenericPFRecHit(
    const reco::PFRecHitHostCollection::ConstView::const_element pfRecHit)
    : detId(pfRecHit.detId()),
      depth(pfRecHit.depth()),
      layer(pfRecHit.layer()),
      time(pfRecHit.time()),
      energy(pfRecHit.energy()),
      x(pfRecHit.x()),
      y(pfRecHit.y()),
      z(pfRecHit.z()) {
  // Copy first four elements into neighbours4 and last four into neighbours8
  neighbours4.reserve(4);
  neighbours8.reserve(4);
  for (size_t k = 0; k < 4; k++)
    if (pfRecHit.neighbours()(k) != -1)
      neighbours4.emplace_back((uint32_t)pfRecHit.neighbours()(k));
  for (size_t k = 4; k < 8; k++)
    if (pfRecHit.neighbours()(k) != -1)
      neighbours8.emplace_back((uint32_t)pfRecHit.neighbours()(k));
}

void PFRecHitProducerTest::GenericPFRecHit::print(const char* prefix, size_t idx) {
  printf("%s %4lu detId:%u depth:%d layer:%d energy:%f time:%f pos:%f,%f,%f neighbours:%lu+%lu(",
         prefix,
         idx,
         detId,
         depth,
         layer,
         energy,
         time,
         x,
         y,
         z,
         neighbours4.size(),
         neighbours8.size());
  for (uint32_t j = 0; j < neighbours4.size(); j++)
    printf("%s%u", j ? "," : "", neighbours4[j]);
  printf(";");
  for (uint32_t j = 0; j < neighbours8.size(); j++)
    printf("%s%u", j ? "," : "", neighbours8[j]);
  printf(")\n");
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFRecHitProducerTest);
