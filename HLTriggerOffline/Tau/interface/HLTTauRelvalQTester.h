#include "DQMServices/Components/interface/QualityTester.h"


class HLTTauRelvalQTester : public QualityTester
{
 public:
  HLTTauRelvalQTester(const edm::ParameterSet& ps);
  ~HLTTauRelvalQTester() override;

 protected:

  void analyze(const edm::Event& e, const edm::EventSetup& c) override ;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c) override;
  void endRun(const edm::Run& r, const edm::EventSetup& c) override;
  void endJob() override;

 private:
  edm::EDGetTokenT<std::vector<int> > refMothers_;
  std::vector<int> mothers_;
  bool runQTests;
};
