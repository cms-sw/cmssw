#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace BTagSkimMCCount {
  struct Counters {
    Counters() : nEvents_(0), nAccepted_(0) {}
    mutable std::atomic<unsigned int> nEvents_, nAccepted_;
  };
}  // namespace BTagSkimMCCount

class BTagSkimMC : public edm::stream::EDFilter<edm::GlobalCache<BTagSkimMCCount::Counters> > {
public:
  /// constructor
  explicit BTagSkimMC(const edm::ParameterSet&, const BTagSkimMCCount::Counters* count);

  static std::unique_ptr<BTagSkimMCCount::Counters> initializeGlobalCache(edm::ParameterSet const&) {
    return std::make_unique<BTagSkimMCCount::Counters>();
  }

  bool filter(edm::Event& evt, const edm::EventSetup& es) override;
  void endStream() override;
  static void globalEndJob(const BTagSkimMCCount::Counters* counters);

private:
  bool verbose;
  double overallLumi;

  unsigned int nEvents_;
  unsigned int nAccepted_;
  double pthatMin, pthatMax;
  std::string process_;
};
