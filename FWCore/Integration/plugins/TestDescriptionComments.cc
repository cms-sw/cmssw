
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <string>

namespace edmtest {

  class TestDescriptionComments : public edm::global::EDAnalyzer<> {
  public:
    explicit TestDescriptionComments(edm::ParameterSet const&) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const final {}
  };

  void TestDescriptionComments::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription nestedDesc;
    nestedDesc.add<int>("x", 1);

    std::vector<edm::ParameterSet> vectorOfPSets;

    edm::ParameterSetDescription desc;
    desc.ifValue(
        edm::ParameterDescription<std::string>("sswitch", "b", true, edm::Comment("test1")),
        "a" >> edm::ParameterDescription<int>(std::string("x"), 100, true, edm::Comment("test2")) or
            "b" >> (edm::ParameterDescription<double>("y1", true, edm::Comment("test3")) and
                    edm::ParameterDescription<double>(std::string("y2"), true, edm::Comment(std::string("test4")))) or
            "c" >> edm::ParameterDescription<std::string>("z", "102", true) or
            "d" >> edm::ParameterDescription<edm::ParameterSetDescription>(
                       "nested1", nestedDesc, true, edm::Comment(std::string("test5"))) or
            "e" >> edm::ParameterDescription<edm::ParameterSetDescription>(
                       std::string("nested2"), nestedDesc, true, edm::Comment(std::string("test6"))) or
            "f" >> edm::ParameterDescription<std::vector<edm::ParameterSet> >(
                       "vpset1", nestedDesc, true, vectorOfPSets, edm::Comment("test7")) or
            "g" >> edm::ParameterDescription<std::vector<edm::ParameterSet> >(
                       std::string("vpset2"), nestedDesc, true, vectorOfPSets, edm::Comment("test8")) or
            "h" >> edm::ParameterDescription<std::vector<edm::ParameterSet> >(
                       "vpset3", nestedDesc, true, edm::Comment("test9")) or
            "i" >> edm::ParameterDescription<std::vector<edm::ParameterSet> >(
                       std::string("vpset4"), nestedDesc, true, edm::Comment("test10")));
    descriptions.add("test", desc);
  }
}  // namespace edmtest
using edmtest::TestDescriptionComments;
DEFINE_FWK_MODULE(TestDescriptionComments);
