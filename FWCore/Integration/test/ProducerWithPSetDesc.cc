#include "FWCore/Integration/test/ProducerWithPSetDesc.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <boost/cstdint.hpp>

#include <vector>
#include <limits>
#include <string>

namespace edmtest {
  ProducerWithPSetDesc::ProducerWithPSetDesc(edm::ParameterSet const& iConfig)
  {
    produces<ThingCollection>();
  }

  ProducerWithPSetDesc::~ProducerWithPSetDesc() { }

  void ProducerWithPSetDesc::produce(edm::Event& e, edm::EventSetup const&) {
    // This serves no purpose, I just put it here so the module does something
    // Probably could just make this method do nothing and it would not
    // affect the test.
    std::auto_ptr<ThingCollection> result(new ThingCollection);  //Empty
    e.put(result);
  }

  void
  ProducerWithPSetDesc::
  fillDescriptions(edm::ConfigurationDescriptions & descriptions) {

    edm::ParameterSetDescription iDesc;

    // Try to exercise the description code by adding all different
    // types of parameters with a large range of values.  Also
    // nested ParameterSets and vectors of them at the end.

    iDesc.add<int>("p_int", 2147483647);
    iDesc.addUntracked<int>("p_int_untracked", -2147483647);
    iDesc.addOptional<int>("p_int_opt", 0);
    iDesc.addOptionalUntracked<int>("p_int_optuntracked", 7);

    std::vector<int> vint;
    iDesc.add<std::vector<int> >("vint1", vint);
    vint.push_back(2147483647);
    iDesc.add<std::vector<int> >("vint2", vint);
    vint.push_back(-2147483647);
    iDesc.add<std::vector<int> >("vint3", vint);
    vint.push_back(0);
    iDesc.add<std::vector<int> >("vint4", vint);

    iDesc.add<unsigned>("uint1", 4294967295U);
    iDesc.addUntracked<unsigned>("uint2", 0);

    std::vector<unsigned> vuint;
    iDesc.add<std::vector<unsigned> >("vuint1", vuint);
    vuint.push_back(4294967295U);
    iDesc.add<std::vector<unsigned> >("vuint2", vuint);
    vuint.push_back(0);
    iDesc.add<std::vector<unsigned> >("vuint3", vuint);
    vuint.push_back(11);
    iDesc.add<std::vector<unsigned> >("vuint4", vuint);

    iDesc.add<boost::int64_t>("int64v1", 9000000000000000000LL);
    iDesc.add<boost::int64_t>("int64v2", -9000000000000000000LL);
    iDesc.add<boost::int64_t>("int64v3", 0);

    std::vector<boost::int64_t> vint64;
    iDesc.add<std::vector<boost::int64_t> >("vint64v1", vint64);
    vint64.push_back(9000000000000000000LL);
    iDesc.add<std::vector<boost::int64_t> >("vint64v2", vint64);
    vint64.push_back(-9000000000000000000LL);
    iDesc.add<std::vector<boost::int64_t> >("vint64v3", vint64);
    vint64.push_back(0);
    iDesc.add<std::vector<boost::int64_t> >("vint64v4", vint64);

    iDesc.add<boost::uint64_t>("uint64v1", 18000000000000000000ULL);
    iDesc.addUntracked<boost::uint64_t>("uint64v2", 0);

    std::vector<boost::uint64_t> vuint64;
    iDesc.add<std::vector<boost::uint64_t> >("vuint64v1", vuint64);
    vuint64.push_back(18000000000000000000ULL);
    iDesc.add<std::vector<boost::uint64_t> >("vuint64v2", vuint64);
    vuint64.push_back(0);
    iDesc.add<std::vector<boost::uint64_t> >("vuint64v3", vuint64);
    vuint64.push_back(11);
    iDesc.add<std::vector<boost::uint64_t> >("vuint64v4", vuint64);

    iDesc.add<double>("doublev1", std::numeric_limits<double>::min());
    iDesc.addUntracked<double>("doublev2", 0.0);
    iDesc.addUntracked<double>("doublev3", 0.3);

    std::vector<double> vdouble;
    iDesc.add<std::vector<double> >("vdoublev1", vdouble);
    // cmsRun will fail with a value this big
    // vdouble.push_back(std::numeric_limits<double>::max());
    // This works though
    vdouble.push_back(1e+300);
    iDesc.add<std::vector<double> >("vdoublev2", vdouble);
    vdouble.push_back(0.0);
    iDesc.add<std::vector<double> >("vdoublev3", vdouble);
    vdouble.push_back(11.0);
    iDesc.add<std::vector<double> >("vdoublev4", vdouble);
    vdouble.push_back(0.3);
    iDesc.add<std::vector<double> >("vdoublev5", vdouble);

    iDesc.add<bool>("boolv1", true);
    iDesc.add<bool>("boolv2", false);

    std::string test("Hello");
    iDesc.add<std::string>("stringv1", test);
    test.clear();
    iDesc.add<std::string>("stringv2", test);

    std::vector<std::string> vstring;
    iDesc.add<std::vector<std::string> >("vstringv1", vstring);
    test = "Hello";
    vstring.push_back(test);
    iDesc.add<std::vector<std::string> >("vstringv2", vstring);
    test = "World";
    vstring.push_back(test);
    iDesc.add<std::vector<std::string> >("vstringv3", vstring);
    test = "";
    vstring.push_back(test);
    iDesc.add<std::vector<std::string> >("vstringv4", vstring);

    edm::EventID eventID(11, 12);
    iDesc.add<edm::EventID>("eventIDv1", eventID);
    edm::EventID eventID2(101, 102);
    iDesc.add<edm::EventID>("eventIDv2", eventID2);

    std::vector<edm::EventID> vEventID;
    iDesc.add<std::vector<edm::EventID> >("vEventIDv1", vEventID);
    edm::EventID eventID3(1000, 1100);
    vEventID.push_back(eventID3);
    iDesc.add<std::vector<edm::EventID> >("vEventIDv2", vEventID);
    edm::EventID eventID4(10000, 11000);
    vEventID.push_back(eventID4);
    iDesc.add<std::vector<edm::EventID> >("vEventIDv3", vEventID);
    edm::EventID eventID5(100000, 110000);
    vEventID.push_back(eventID5);
    iDesc.add<std::vector<edm::EventID> >("vEventIDv4", vEventID);

    edm::LuminosityBlockID luminosityID(11, 12);
    iDesc.add<edm::LuminosityBlockID>("luminosityIDv1", luminosityID);
    edm::LuminosityBlockID luminosityID2(101, 102);
    iDesc.add<edm::LuminosityBlockID>("luminosityIDv2", luminosityID2);

    std::vector<edm::LuminosityBlockID> vLuminosityBlockID;
    iDesc.add<std::vector<edm::LuminosityBlockID> >("vLuminosityBlockIDv1", vLuminosityBlockID);
    edm::LuminosityBlockID luminosityID3(1000, 1100);
    vLuminosityBlockID.push_back(luminosityID3);
    iDesc.add<std::vector<edm::LuminosityBlockID> >("vLuminosityBlockIDv2", vLuminosityBlockID);
    edm::LuminosityBlockID luminosityID4(10000, 11000);
    vLuminosityBlockID.push_back(luminosityID4);
    iDesc.add<std::vector<edm::LuminosityBlockID> >("vLuminosityBlockIDv3", vLuminosityBlockID);
    edm::LuminosityBlockID luminosityID5(100000, 110000);
    vLuminosityBlockID.push_back(luminosityID5);
    iDesc.add<std::vector<edm::LuminosityBlockID> >("vLuminosityBlockIDv4", vLuminosityBlockID);

    edm::LuminosityBlockRange lumiRange(1,1, 9,9);
    iDesc.add<edm::LuminosityBlockRange>("lumiRangev1", lumiRange);
    edm::LuminosityBlockRange lumiRange2(3,4, 1000,1000);
    iDesc.add<edm::LuminosityBlockRange>("lumiRangev2", lumiRange2);

    std::vector<edm::LuminosityBlockRange> vLumiRange;
    iDesc.add<std::vector<edm::LuminosityBlockRange> >("vLumiRangev1", vLumiRange);
    vLumiRange.push_back(lumiRange);
    iDesc.add<std::vector<edm::LuminosityBlockRange> >("vLumiRangev2", vLumiRange);
    vLumiRange.push_back(lumiRange2);
    iDesc.add<std::vector<edm::LuminosityBlockRange> >("vLumiRangev3", vLumiRange);

    edm::EventRange eventRange(1,1, 8,8);
    iDesc.add<edm::EventRange>("eventRangev1", eventRange);
    edm::EventRange eventRange2(3,4, 1001,1002);
    iDesc.add<edm::EventRange>("eventRangev2", eventRange2);

    std::vector<edm::EventRange> vEventRange;
    iDesc.add<std::vector<edm::EventRange> >("vEventRangev1", vEventRange);
    vEventRange.push_back(eventRange);
    iDesc.add<std::vector<edm::EventRange> >("vEventRangev2", vEventRange);
    vEventRange.push_back(eventRange2);
    iDesc.add<std::vector<edm::EventRange> >("vEventRangev3", vEventRange);

    edm::InputTag inputTag("One", "Two", "Three");
    iDesc.add<edm::InputTag>("inputTagv1", inputTag);
    edm::InputTag inputTag2("One", "Two");
    iDesc.add<edm::InputTag>("inputTagv2", inputTag2);
    edm::InputTag inputTag3("One");
    iDesc.add<edm::InputTag>("inputTagv3", inputTag3);
    edm::InputTag inputTag4("One", "", "Three");
    iDesc.add<edm::InputTag>("inputTagv4", inputTag4);

    std::vector<edm::InputTag> vInputTag;
    iDesc.add<std::vector<edm::InputTag> >("vInputTagv1", vInputTag);
    vInputTag.push_back(inputTag);
    iDesc.add<std::vector<edm::InputTag> >("vInputTagv2", vInputTag);
    vInputTag.push_back(inputTag2);
    iDesc.add<std::vector<edm::InputTag> >("vInputTagv3", vInputTag);
    vInputTag.push_back(inputTag3);
    iDesc.add<std::vector<edm::InputTag> >("vInputTagv4", vInputTag);
    vInputTag.push_back(inputTag4);
    iDesc.add<std::vector<edm::InputTag> >("vInputTagv5", vInputTag);

    // For purposes of the test, this just needs to point to any file
    // that exists.  I guess pointing to itself cannot ever fail ...
    edm::FileInPath fileInPath("FWCore/Integration/test/ProducerWithPSetDesc.cc");
    iDesc.add<edm::FileInPath>("fileInPath", fileInPath);

    edm::ParameterSetDescription bar;
    bar.add<unsigned int>("Drinks", 5);
    bar.addUntracked<unsigned int>("uDrinks", 5);
    bar.addOptional<unsigned int>("oDrinks", 5);
    bar.addOptionalUntracked<unsigned int>("ouDrinks", 5);
    iDesc.add("bar", bar);

    edm::ParameterSetDescription barx;
    barx.add<unsigned int>("Drinks", 5);
    barx.addUntracked<unsigned int>("uDrinks", 5);
    barx.addOptional<unsigned int>("oDrinks", 5);
    barx.addOptionalUntracked<unsigned int>("ouDrinks", 5);
    std::vector<edm::ParameterSetDescription> bars;
    bars.push_back(barx);
    bars.push_back(barx);
    iDesc.add("bars",bars);

    // Alternate way to add a ParameterSetDescription
    edm::ParameterDescription* parDescription;
    parDescription = iDesc.addOptional("subpset", edm::ParameterSetDescription());
    edm::ParameterSetDescription* subPsetDescription =
      parDescription->parameterSetDescription();

    subPsetDescription->add<int>("xvalue", 11);
    subPsetDescription->addUntracked<edm::ParameterSetDescription>("bar", bar);

    descriptions.add("testProducerWithPsetDesc", iDesc);
  }
}
using edmtest::ProducerWithPSetDesc;
DEFINE_FWK_MODULE(ProducerWithPSetDesc);

