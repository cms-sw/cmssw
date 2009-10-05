#include "DQMServices/Components/interface/QualityTester.h"


class HLTTauRelvalQTester : public QualityTester
{
 public:
  HLTTauRelvalQTester(const edm::ParameterSet& ps);
  ~HLTTauRelvalQTester();

 protected:

  void analyze(const edm::Event& e, const edm::EventSetup& c) ;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
  void endRun(const edm::Run& r, const edm::EventSetup& c);
  void endJob();

 private:
  edm::InputTag refMothers_;
  std::vector<int> mothers_;
  bool runQTests;
};
