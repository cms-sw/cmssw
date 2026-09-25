#include "catch2/catch_all.hpp"

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Framework/interface/SourceCoordinator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace {
  struct TransitionSpec {
    edm::InputSource::ItemType type;
    edm::EventID id;
  };

  class OrderedTransitionSource : public edm::InputSource {
  public:
    explicit OrderedTransitionSource(std::vector<TransitionSpec> transitions)
        : edm::InputSource(edm::ParameterSet{}, edm::InputSourceDescription{}), transitions_(std::move(transitions)) {}

  private:
    ItemTypeInfo getNextItemType() final {
      if (nextIndex_ >= transitions_.size()) {
        return ItemTypeInfo::isStop();
      }
      lastTransition_ = transitions_[nextIndex_++];
      return ItemTypeInfo(lastTransition_.type);
    }

    std::shared_ptr<edm::RunAuxiliary> readRunAuxiliary_() final {
      return std::make_shared<edm::RunAuxiliary>(lastTransition_.id.run(), edm::Timestamp(0), edm::Timestamp(10));
    }

    std::shared_ptr<edm::LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() final {
      return std::make_shared<edm::LuminosityBlockAuxiliary>(
          lastTransition_.id.run(), lastTransition_.id.luminosityBlock(), edm::Timestamp(0), edm::Timestamp(10));
    }

    void readEvent_(edm::EventPrincipal&) final {}

    // Allow SourceCoordinator::skipEvents() to be used as a generic "consume transition" action in this test.
    void skip(int) final {}

    std::vector<TransitionSpec> transitions_;
    std::size_t nextIndex_{0};
    TransitionSpec lastTransition_{edm::InputSource::ItemType::IsStop, edm::EventID{}};
  };

  std::unique_ptr<edm::SourceCoordinator> makeCoordinator(std::vector<TransitionSpec> transitions,
                                                          edm::ServiceToken& token) {
    auto coordinator = std::make_unique<edm::SourceCoordinator>(
        edm::SharedResourcesAcquirer{}, std::make_shared<std::recursive_mutex>(), token);
    coordinator->setSource(std::make_unique<OrderedTransitionSource>(std::move(transitions)));
    return coordinator;
  }
}  // namespace

TEST_CASE("SourceCoordinator replays transitions from source", "[Framework]") {
  edm::ServiceToken token;

  SECTION("replays configured transition order") {
    auto coordinator = makeCoordinator({{edm::InputSource::ItemType::IsFile, edm::EventID{0, 0, 0}},
                                        {edm::InputSource::ItemType::IsRun, edm::EventID{1, 0, 0}},
                                        {edm::InputSource::ItemType::IsLumi, edm::EventID{1, 11, 0}},
                                        {edm::InputSource::ItemType::IsEvent, edm::EventID{1, 11, 21}},
                                        {edm::InputSource::ItemType::IsStop, edm::EventID{0, 0, 0}}},
                                       token);

    auto status = coordinator->thread_unsafe_peekNextTransitionType();
    REQUIRE(status.nextTransitionType().itemType() == edm::InputSource::ItemType::IsFile);

    // A second peek without consumption must return the cached transition.
    auto repeated = coordinator->thread_unsafe_peekNextTransitionType();
    REQUIRE(repeated.nextTransitionType().itemType() == edm::InputSource::ItemType::IsFile);

    auto [fileBlock, productRegistry] = coordinator->readFile();
    REQUIRE(fileBlock != nullptr);
    REQUIRE(productRegistry == nullptr);

    status = coordinator->thread_unsafe_peekNextTransitionType();
    REQUIRE(status.nextTransitionType().itemType() == edm::InputSource::ItemType::IsRun);
    REQUIRE(status.runAuxiliary() != nullptr);
    REQUIRE(status.runAuxiliary()->run() == 1);
    coordinator->skipEvents(0);

    status = coordinator->thread_unsafe_peekNextTransitionType();
    REQUIRE(status.nextTransitionType().itemType() == edm::InputSource::ItemType::IsLumi);
    REQUIRE(status.lumiAuxiliary() != nullptr);
    REQUIRE(status.lumiAuxiliary()->run() == 1);
    REQUIRE(status.lumiAuxiliary()->luminosityBlock() == 11);
    coordinator->skipEvents(0);

    status = coordinator->thread_unsafe_peekNextTransitionType();
    REQUIRE(status.nextTransitionType().itemType() == edm::InputSource::ItemType::IsEvent);
    coordinator->skipEvents(0);

    status = coordinator->thread_unsafe_peekNextTransitionType();
    REQUIRE(status.nextTransitionType().itemType() == edm::InputSource::ItemType::IsStop);
  }

  SECTION("skips synchronize transitions") {
    auto coordinator = makeCoordinator({{edm::InputSource::ItemType::IsSynchronize, edm::EventID{0, 0, 0}},
                                        {edm::InputSource::ItemType::IsSynchronize, edm::EventID{0, 0, 0}},
                                        {edm::InputSource::ItemType::IsFile, edm::EventID{0, 0, 0}},
                                        {edm::InputSource::ItemType::IsRun, edm::EventID{2, 0, 0}},
                                        {edm::InputSource::ItemType::IsStop, edm::EventID{0, 0, 0}}},
                                       token);

    auto status = coordinator->thread_unsafe_peekNextTransitionType();
    REQUIRE(status.nextTransitionType().itemType() == edm::InputSource::ItemType::IsFile);
    auto [fileBlock, productRegistry] = coordinator->readFile();
    REQUIRE(fileBlock != nullptr);
    REQUIRE(productRegistry == nullptr);

    status = coordinator->thread_unsafe_peekNextTransitionType();
    REQUIRE(status.nextTransitionType().itemType() == edm::InputSource::ItemType::IsRun);
    REQUIRE(status.runAuxiliary() != nullptr);
    REQUIRE(status.runAuxiliary()->run() == 2);
    coordinator->skipEvents(0);

    status = coordinator->thread_unsafe_peekNextTransitionType();
    REQUIRE(status.nextTransitionType().itemType() == edm::InputSource::ItemType::IsStop);
  }
}
