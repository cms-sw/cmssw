#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include "CondTools/RunInfo/interface/LHCInfoPerFillPopConSourceHandler.h"
#include "CondTools/RunInfo/interface/TestLHCInfoPerFillPopConSourceHandler.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

struct LHCInfoPerFillPopConSourceHandlerProtectedAccessor : public LHCInfoPerFillPopConSourceHandler {
  using LHCInfoPerFillPopConSourceHandler::addPayloadToBuffer;
  using LHCInfoPerFillPopConSourceHandler::m_fillPayload;
  using LHCInfoPerFillPopConSourceHandler::m_timestampToLumiid;
  using LHCInfoPerFillPopConSourceHandler::m_tmpBuffer;

  LHCInfoPerFillPopConSourceHandlerProtectedAccessor(edm::ParameterSet const& pset)
      : LHCInfoPerFillPopConSourceHandler(pset) {}
};

// Helper function to create a default ParameterSet
edm::ParameterSet createDefaultPSet() {
  edm::ParameterSet pset;
  pset.addUntrackedParameter<bool>("debug", true);
  pset.addUntrackedParameter<std::string>("startTime", "2023-01-01 00:00:00");
  pset.addUntrackedParameter<std::string>("endTime", "2023-12-31 23:59:59");
  pset.addUntrackedParameter<std::string>("name", "LHCInfoPerFillPopConSourceHandler");
  pset.addUntrackedParameter<std::string>("connectionString", "");
  pset.addUntrackedParameter<std::string>("ecalConnectionString", "");
  pset.addUntrackedParameter<std::string>("authenticationPath", "");
  pset.addUntrackedParameter<std::string>("omsBaseUrl", "");
  pset.addUntrackedParameter<double>("minEnergy", 450.0);
  pset.addUntrackedParameter<double>("maxEnergy", 8000.0);
  pset.addUntrackedParameter<bool>("throwOnInvalid", true);
  return pset;
}

TEST_CASE("LHCInfoPerFillPopConSourceHandler.isPayloadValid works", "[isPayloadValid]") {
  //generate test for both endFill and duringFill modes
  bool endFillMode = GENERATE(true, false);
  edm::ParameterSet pset = createDefaultPSet();
  pset.addUntrackedParameter<bool>("endFill", endFillMode);
  LHCInfoPerFillPopConSourceHandlerProtectedAccessor handler(pset);

  LHCInfoPerFill payload;

  SECTION("Energy within range is valid") {
    payload.setEnergy(6500.0);
    CHECK(handler.isPayloadValid(payload) == true);
  }

  SECTION("Energy at lower bound is valid") {
    payload.setEnergy(450.0);
    CHECK(handler.isPayloadValid(payload) == true);
  }

  SECTION("Energy at upper bound is valid") {
    payload.setEnergy(8000.0);
    CHECK(handler.isPayloadValid(payload) == true);
  }

  SECTION("Energy below range is invalid") {
    payload.setEnergy(400.0);
    CHECK(handler.isPayloadValid(payload) == false);
  }

  SECTION("Energy above range is invalid") {
    payload.setEnergy(8500.0);
    CHECK(handler.isPayloadValid(payload) == false);
  }
}

TEST_CASE("LHCInfoPerFillPopConSourceHandler.addPayloadToBuffer works", "[addPayloadToBuffer]") {
  //generate test for both endFill and duringFill modes
  bool endFillMode = GENERATE(true, false);
  edm::ParameterSet pset = createDefaultPSet();
  pset.addUntrackedParameter<bool>("endFill", endFillMode);
  LHCInfoPerFillPopConSourceHandlerProtectedAccessor handler(pset);

  // Create a mock OMSServiceResultRef
  boost::property_tree::ptree mockRow;

  mockRow.put("start_time", "2023-06-01 12:00:00");
  mockRow.put("delivered_lumi", 100.0f);
  mockRow.put("recorded_lumi", 80.0f);
  mockRow.put("run_number", "3500");
  mockRow.put("lumisection_number", "150");
  cond::OMSServiceResultRef mockResultRef(&mockRow);

  // Set the m_fillPayload before calling addPayloadToBuffer
  handler.m_fillPayload = std::make_unique<LHCInfoPerFill>();
  handler.m_fillPayload->setFillNumber(1234);
  handler.m_fillPayload->setEnergy(6500.0);

  // Call the addPayloadToBuffer method
  handler.addPayloadToBuffer(mockResultRef);

  //verify that the payload was added to the tmpBuffer
  REQUIRE(handler.m_tmpBuffer.empty() == false);
  CHECK(handler.m_tmpBuffer.size() == 1);

  SECTION("addPayloadToBuffer adds correct payload to buffer") {
    auto& addedPayload = handler.m_tmpBuffer.front().second;
    CHECK(addedPayload->delivLumi() == 100.0f);
    CHECK(addedPayload->recLumi() == 80.0f);
    CHECK(addedPayload->fillNumber() == 1234);
    CHECK(addedPayload->energy() == 6500.0);
  }

  SECTION("addPayloadToBuffer adds correct IOV to buffer") {
    auto addedIov = handler.m_tmpBuffer.front().first;
    CHECK(addedIov == cond::time::from_boost(boost::posix_time::time_from_string("2023-06-01 12:00:00")));
  }

  if (!endFillMode) {
    SECTION("addPayloadToBuffer updates timestampToLumiid map in duringFill mode") {
      CAPTURE(endFillMode);
      REQUIRE(handler.m_timestampToLumiid.empty() == false);
      CHECK(handler.m_timestampToLumiid.size() == 1);
      auto it = handler.m_timestampToLumiid.begin();
      CHECK(it->first == cond::time::from_boost(boost::posix_time::time_from_string("2023-06-01 12:00:00")));
      CHECK(it->second == cond::time::lumiTime(3500, 150));
    }
  }
}

TEST_CASE("LHCInfoPerFillPopConSourceHandler.getNewObjects fills IOVs correctly: endFill mode", "[populate]") {
  std::cout << "\nTEST: getNewObjects fills IOVs correctly: endfill mode" << std::endl;
  edm::setStandAloneMessageThreshold(edm::messagelogger::ELinfo);

  bool endFillMode = true;
  edm::ParameterSet pset = createDefaultPSet();
  pset.addUntrackedParameter<bool>("endFill", endFillMode);
  TestLHCInfoPerFillPopConSourceHandler handler(pset);

  // Set up mock data
  std::vector<cond::OMSServiceResultRef> mockOmsFills;
  std::map<unsigned short /*fillNr*/, cond::OMSServiceResult> mockLumiData;

  // Populate mock data
  std::vector<unsigned short> fillNumbers = {1000, 1001, 1002};
  std::vector<float> energies = {6500.0, 6600.0, 6700.0};
  std::vector<boost::posix_time::ptime> fillStartTimes /* force formatting */ = {
      boost::posix_time::time_from_string("2023-06-01 11:30:00"),
      boost::posix_time::time_from_string("2023-06-01 12:30:00"),
      boost::posix_time::time_from_string("2023-06-01 13:30:00")};
  std::vector<boost::posix_time::ptime> fillStableBeamBeginTimes = {
      boost::posix_time::time_from_string("2023-06-01 12:00:00"),
      boost::posix_time::time_from_string("2023-06-01 13:00:00"),
      boost::posix_time::time_from_string("2023-06-01 14:00:00")};
  std::vector<boost::posix_time::ptime> fillEndTimes /* force formatting */ = {
      boost::posix_time::time_from_string("2023-06-01 12:30:00"),
      boost::posix_time::time_from_string("2023-06-01 13:29:00"),
      boost::posix_time::time_from_string("2023-06-01 14:30:00")};

  for (size_t i = 0; i < fillNumbers.size(); ++i) {
    auto fillPayload = std::make_shared<LHCInfoPerFill>();
    fillPayload->setFillNumber(fillNumbers[i]);
    fillPayload->setEnergy(energies[i]);
    fillPayload->setCreationTime(cond::time::from_boost(fillStartTimes[i]));
    fillPayload->setBeginTime(cond::time::from_boost(fillStableBeamBeginTimes[i]));
    fillPayload->setEndTime(cond::time::from_boost(fillEndTimes[i]));
    handler.mockOmsFills.emplace_back(std::make_pair(cond::time::from_boost(fillStartTimes[i]), fillPayload));
  }

  // populate json mockLumiData for each fill
  // OMSServiceResult cannot be initialized directly from vector of OMSServiceResultRef
  // Instead we need to prepare the data in json format and parse it using OMSServiceResult::parseData
  std::vector<std::string> lumiJsons = {
      R"delimiter(
{
  "data": [
    {
      "id": "354496_113",
      "type": "lumisections",
      "attributes": {
        "beams_stable": true,
        "start_time": "2023-06-01T12:00:00Z",
        "run_number": 354496,
        "recorded_lumi": 0,
        "delivered_lumi": 0,
        "lumisection_number": 113
      }
    },
    {
      "id": "354496_114",
      "type": "lumisections",
      "attributes": {
        "beams_stable": true,
        "start_time": "2023-06-01T12:00:00Z",
        "run_number": 354496,
        "recorded_lumi": 1,
        "delivered_lumi": 1,
        "lumisection_number": 114
      }
    },
    {
      "id": "354496_115",
      "type": "lumisections",
      "attributes": {
        "beams_stable": true,
        "start_time": "2023-06-01T12:00:46Z",
        "run_number": 354496,
        "recorded_lumi": 123,
        "delivered_lumi": 123,
        "lumisection_number": 115
      }
    }
  ]
}
)delimiter",
      R"delimiter(
{
  "data": [
    {
      "id": "354496_113",
      "type": "lumisections",
      "attributes": {
        "beams_stable": true,
        "start_time": "2023-06-01T13:00:00Z",
        "run_number": 354496,
        "recorded_lumi": 0,
        "delivered_lumi": 0,
        "lumisection_number": 200
      }
    },
    {
      "id": "354496_114",
      "type": "lumisections",
      "attributes": {
        "beams_stable": true,
        "start_time": "2023-06-01T13:00:23Z",
        "run_number": 354496,
        "recorded_lumi": 0,
        "delivered_lumi": 0,
        "lumisection_number": 201
      }
    }
  ]
}
)delimiter",
      R"delimiter(
{
  "data": [
    {
      "id": "354496_113",
      "type": "lumisections",
      "attributes": {
        "beams_stable": true,
        "start_time": "2023-06-01T14:00:00Z",
        "run_number": 354497,
        "recorded_lumi": 0,
        "delivered_lumi": 0,
        "lumisection_number": 100
      }
    },
    {
      "id": "354496_114",
      "type": "lumisections",
      "attributes": {
        "beams_stable": true,
        "start_time": "2023-06-01T14:00:23Z",
        "run_number": 354497,
        "recorded_lumi": 0,
        "delivered_lumi": 0,
        "lumisection_number": 101
      }
    }
  ]
}
)delimiter"};

  std::cout << "lumiJsons size: " << lumiJsons.size() << std::endl;

  for (size_t i = 0; i < fillNumbers.size(); ++i) {
    handler.mockLumiData[fillNumbers[i]] = cond::OMSServiceResult();
    handler.mockLumiData[fillNumbers[i]].parseData(lumiJsons[i]);
  }

  // Run the populate function
  REQUIRE_NOTHROW(handler.getNewObjects());
  REQUIRE(handler.iovs().empty() == false);

  REQUIRE(handler.iovs().size() == 8);

  // Check that IOVs were populated correctly
  auto it = handler.iovs().begin();
  auto end = handler.iovs().end();

  auto to_since = [](const char* ts) { return cond::time::from_boost(boost::posix_time::time_from_string(ts)); };

  // Helper lambda to check and advance iterator safely
  auto check_iov = [&](decltype(it)& iter,
                       auto expected_since,
                       auto expected_fill,
                       auto expected_energy,
                       auto expected_delivLumi,
                       auto expected_recLumi) {
    REQUIRE(iter != end);
    std::cout << "IOV since: " << iter->first << ", Fill: " << iter->second->fillNumber()
              << ", Energy: " << iter->second->energy() << ", delivLumi: " << iter->second->delivLumi()
              << ", recLumi: " << iter->second->recLumi() << std::endl;
    CHECK(iter->first == (unsigned long long)expected_since);
    CHECK(iter->second->fillNumber() == expected_fill);
    CHECK(iter->second->energy() == expected_energy);
    CHECK(iter->second->delivLumi() == expected_delivLumi);
    CHECK(iter->second->recLumi() == expected_recLumi);
    ++iter;
  };

  check_iov(it, 1, 0, 0.0, 0.0f, 0.0f);                                          // empty payload added to empty tag
  check_iov(it, to_since("2023-06-01 12:00:00"), 1000, 6500.0, 0.0f, 0.0f);      // first payload of SB of fill 1000
  check_iov(it, to_since("2023-06-01 12:00:46"), 1000, 6500.0, 123.0f, 123.0f);  // last payload of SB of fill 1000
  check_iov(it, to_since("2023-06-01 12:30:00"), 0, 0.0, 0.0f, 0.0f);            // empty payload between fills
  check_iov(it, to_since("2023-06-01 13:00:00"), 1001, 6600.0, 0.0f, 0.0f);
  // only one payload of fill 1001 because of same data
  check_iov(it, to_since("2023-06-01 13:29:00"), 0, 0.0, 0.0f, 0.0f);
  check_iov(it, to_since("2023-06-01 14:00:00"), 1002, 6700.0, 0.0f, 0.0f);
  // only one payload of fill 1002 because of same data
  check_iov(it, to_since("2023-06-01 14:30:00"), 0, 0.0, 0.0f, 0.0f);
}

TEST_CASE("LHCInfoPerFillPopConSourceHandler.getNewObjects fills IOVs correctly in duringFill mode", "[populate]") {
  std::cout << "\nTEST: getNewObjects fills IOVs correctly in duringFill mode" << std::endl;
  edm::setStandAloneMessageThreshold(edm::messagelogger::ELinfo);

  bool endFillMode = false;
  edm::ParameterSet pset = createDefaultPSet();
  pset.addUntrackedParameter<bool>("endFill", endFillMode);
  TestLHCInfoPerFillPopConSourceHandler handler(pset);

  // Set up mock data
  std::vector<cond::OMSServiceResultRef> mockOmsFills;
  std::map<unsigned short /*fillNr*/, cond::OMSServiceResult> mockLumiData;
  //set mock execution time
  handler.mockExecutionTime = boost::posix_time::time_from_string("2023-06-01 12:10:00");

  // Populate mock data
  std::vector<unsigned short> fillNumbers = {1000};
  std::vector<float> energies = {GENERATE(450.0, 6600.0, 8000.0)};
  std::vector<boost::posix_time::ptime> fillStartTimes = {boost::posix_time::time_from_string("2023-06-01 11:30:00")};
  std::vector<boost::posix_time::ptime> fillStableBeamBeginTimes = {
      boost::posix_time::time_from_string("2023-06-01 12:00:00")};
  std::vector<boost::posix_time::ptime> fillEndTimes = {boost::posix_time::time_from_string("2023-06-01 12:30:00")};

  for (size_t i = 0; i < fillNumbers.size(); ++i) {
    auto fillPayload = std::make_shared<LHCInfoPerFill>();
    fillPayload->setFillNumber(fillNumbers[i]);
    fillPayload->setEnergy(energies[i]);
    fillPayload->setCreationTime(cond::time::from_boost(fillStartTimes[i]));
    fillPayload->setBeginTime(cond::time::from_boost(fillStableBeamBeginTimes[i]));
    fillPayload->setEndTime(0LL);
    handler.mockOmsFills.emplace_back(std::make_pair(cond::time::from_boost(fillStartTimes[i]), fillPayload));
  }

  std::vector<std::string> lumiJsons = {
      R"delimiter(
{
  "data": [
    {
      "id": "354496_113",
      "type": "lumisections",
      "attributes": {
        "beams_stable": true,
        "start_time": "2023-06-01T12:00:00Z",
        "run_number": 354496,
        "recorded_lumi": 0,
        "delivered_lumi": 0,
        "lumisection_number": 113
      }
    },
    {
      "id": "354496_114",
      "type": "lumisections",
      "attributes": {
        "beams_stable": true,
        "start_time": "2023-06-01T12:00:23Z",
        "run_number": 354496,
        "recorded_lumi": 123,
        "delivered_lumi": 123,
        "lumisection_number": 114
      }
    }
  ]
}
)delimiter"};

  for (size_t i = 0; i < fillNumbers.size(); ++i) {
    handler.mockLumiData[fillNumbers[i]] = cond::OMSServiceResult();
    handler.mockLumiData[fillNumbers[i]].parseData(lumiJsons[i]);
  }

  // Run the populate function
  REQUIRE_NOTHROW(handler.getNewObjects());
  REQUIRE(handler.iovs().empty() == false);

  REQUIRE(handler.iovs().size() == 1);
  // Check that IOVs were populated correctly

  auto it = handler.iovs().begin();
  auto end = handler.iovs().end();

  // Helper lambda to check and advance iterator safely
  auto check_iov =
      [&](decltype(it)& iter, auto expected_fill, auto expected_energy, auto expected_delivLumi, auto expected_recLumi) {
        REQUIRE(iter != end);
        std::cout << "IOV since: " << iter->first << ", Fill: " << iter->second->fillNumber()
                  << ", Energy: " << iter->second->energy() << ", delivLumi: " << iter->second->delivLumi()
                  << ", recLumi: " << iter->second->recLumi() << std::endl;
        // we don't check IOV since in duringFill mode
        CHECK(iter->second->fillNumber() == expected_fill);
        CHECK(iter->second->energy() == expected_energy);
        CHECK(iter->second->delivLumi() == expected_delivLumi);
        CHECK(iter->second->recLumi() == expected_recLumi);
        ++iter;
      };

  check_iov(it, 1000, energies.front(), 123.0f, 123.0f);
}

TEST_CASE(
    "LHCInfoPerFillPopConSourceHandler.getNewObjects doesn't upload payloads with invalid energy in duringFill mode",
    "[populate]") {
  std::cout << "\nTEST: getNewObjects doesn't upload payloads with invalid energy in duringFill mode" << std::endl;
  edm::setStandAloneMessageThreshold(edm::messagelogger::ELinfo);

  bool endFillMode = false;
  edm::ParameterSet pset = createDefaultPSet();
  pset.addUntrackedParameter<bool>("endFill", endFillMode);
  TestLHCInfoPerFillPopConSourceHandler handler(pset);

  // Set up mock data
  std::vector<cond::OMSServiceResultRef> mockOmsFills;
  std::map<unsigned short /*fillNr*/, cond::OMSServiceResult> mockLumiData;
  //set mock execution time
  handler.mockExecutionTime = boost::posix_time::time_from_string("2023-06-01 12:10:00");

  // Populate mock data
  std::vector<unsigned short> fillNumbers = {1000};
  std::vector<float> energies = {GENERATE(-6800., -1., 0., 449.9, 8000.1)};  // invalid energy
  std::vector<boost::posix_time::ptime> fillStartTimes = {boost::posix_time::time_from_string("2023-06-01 11:30:00")};
  std::vector<boost::posix_time::ptime> fillStableBeamBeginTimes = {
      boost::posix_time::time_from_string("2023-06-01 12:00:00")};

  for (size_t i = 0; i < fillNumbers.size(); ++i) {
    auto fillPayload = std::make_shared<LHCInfoPerFill>();
    fillPayload->setFillNumber(fillNumbers[i]);
    fillPayload->setEnergy(energies[i]);
    fillPayload->setCreationTime(cond::time::from_boost(fillStartTimes[i]));
    fillPayload->setBeginTime(cond::time::from_boost(fillStableBeamBeginTimes[i]));
    fillPayload->setEndTime(0LL);
    handler.mockOmsFills.emplace_back(std::make_pair(cond::time::from_boost(fillStartTimes[i]), fillPayload));
  }

  std::vector<std::string> lumiJsons = {
      R"delimiter(
{
  "data": [
    {
      "id": "354496_113",
      "type": "lumisections",
      "attributes": {
        "beams_stable": true,
        "start_time": "2023-06-01T12:00:00Z",
        "run_number": 354496,
        "recorded_lumi": 0,
        "delivered_lumi": 0,
        "lumisection_number": 113
      }
    },
    {
      "id": "354496_114",
      "type": "lumisections",
      "attributes": {
        "beams_stable": true,
        "start_time": "2023-06-01T12:00:23Z",
        "run_number": 354496,
        "recorded_lumi": 123,
        "delivered_lumi": 123,
        "lumisection_number": 114
      }
    }
  ]
}
)delimiter"};

  for (size_t i = 0; i < fillNumbers.size(); ++i) {
    handler.mockLumiData[fillNumbers[i]] = cond::OMSServiceResult();
    handler.mockLumiData[fillNumbers[i]].parseData(lumiJsons[i]);
  }

  // Run the populate function
  REQUIRE_THROWS(handler.getNewObjects());
  // test that the exception message contains "Invalid energy"
  try {
    handler.getNewObjects();
  } catch (const cms::Exception& e) {
    std::string what = e.what();
    CHECK(what.find("Skipping upload of payload with invalid values: Fill = 1000, Energy = ") != std::string::npos);
  }
}

TEST_CASE("LHCInfoPerFillPopConSourceHandler.getNewObjects during fill mode: doesn't upload payloads for ended fills",
          "[populate]") {
  std::cout << "\nTEST: getNewObjects during fill mode: doesn't upload payloads for ended fills" << std::endl;
  edm::setStandAloneMessageThreshold(edm::messagelogger::ELinfo);

  bool endFillMode = false;
  edm::ParameterSet pset = createDefaultPSet();
  pset.addUntrackedParameter<bool>("endFill", endFillMode);
  TestLHCInfoPerFillPopConSourceHandler handler(pset);

  // Set up mock data
  std::vector<cond::OMSServiceResultRef> mockOmsFills;
  std::map<unsigned short /*fillNr*/, cond::OMSServiceResult> mockLumiData;
  //set mock execution time
  handler.mockExecutionTime = boost::posix_time::time_from_string("2023-06-01 12:30:01");

  // Populate mock data
  std::vector<unsigned short> fillNumbers = {1000};
  std::vector<float> energies = {6800};
  std::vector<boost::posix_time::ptime> fillStartTimes = {boost::posix_time::time_from_string("2023-06-01 11:30:00")};
  std::vector<boost::posix_time::ptime> fillStableBeamBeginTimes = {
      boost::posix_time::time_from_string("2023-06-01 12:00:00")};
  std::vector<boost::posix_time::ptime> fillEndTimes = {boost::posix_time::time_from_string("2023-06-01 12:30:00")};

  for (size_t i = 0; i < fillNumbers.size(); ++i) {
    auto fillPayload = std::make_shared<LHCInfoPerFill>();
    fillPayload->setFillNumber(fillNumbers[i]);
    fillPayload->setEnergy(energies[i]);
    fillPayload->setCreationTime(cond::time::from_boost(fillStartTimes[i]));
    fillPayload->setBeginTime(cond::time::from_boost(fillStableBeamBeginTimes[i]));
    fillPayload->setEndTime(cond::time::from_boost(fillEndTimes[i]));
    handler.mockOmsFills.emplace_back(std::make_pair(cond::time::from_boost(fillStartTimes[i]), fillPayload));
  }

  std::vector<std::string> lumiJsons = {
      R"delimiter(
{
  "data": [
    {
      "id": "354496_113",
      "type": "lumisections",
      "attributes": {
        "beams_stable": true,
        "start_time": "2023-06-01T12:00:00Z",
        "run_number": 354496,
        "recorded_lumi": 0,
        "delivered_lumi": 0,
        "lumisection_number": 113
      }
    },
    {
      "id": "354496_114",
      "type": "lumisections",
      "attributes": {
        "beams_stable": true,
        "start_time": "2023-06-01T12:00:23Z",
        "run_number": 354496,
        "recorded_lumi": 123,
        "delivered_lumi": 123,
        "lumisection_number": 114
      }
    }
  ]
}
)delimiter"};

  for (size_t i = 0; i < fillNumbers.size(); ++i) {
    handler.mockLumiData[fillNumbers[i]] = cond::OMSServiceResult();
    handler.mockLumiData[fillNumbers[i]].parseData(lumiJsons[i]);
  }

  handler.getNewObjects();
  CHECK(handler.iovs().size() == 0);
}

TEST_CASE(
    "LHCInfoPerFillPopConSourceHandler.getNewObjects during fill mode: doesn't upload payloads for fills with no "
    "lumisections",
    "[populate]") {
  std::cout << "\nTEST: getNewObjects during fill mode: doesn't upload payloads for fills with no lumisections"
            << std::endl;
  edm::setStandAloneMessageThreshold(edm::messagelogger::ELinfo);

  bool endFillMode = false;
  edm::ParameterSet pset = createDefaultPSet();
  pset.addUntrackedParameter<bool>("endFill", endFillMode);
  TestLHCInfoPerFillPopConSourceHandler handler(pset);

  // Set up mock data
  std::vector<cond::OMSServiceResultRef> mockOmsFills;
  std::map<unsigned short /*fillNr*/, cond::OMSServiceResult> mockLumiData;
  //set mock execution time
  handler.mockExecutionTime = boost::posix_time::time_from_string("2023-06-01 12:00:01");

  // Populate mock data
  std::vector<unsigned short> fillNumbers = {1000};
  std::vector<float> energies = {6800};
  std::vector<boost::posix_time::ptime> fillStartTimes = {boost::posix_time::time_from_string("2023-06-01 11:30:00")};
  std::vector<boost::posix_time::ptime> fillStableBeamBeginTimes = {
      boost::posix_time::time_from_string("2023-06-01 12:00:00")};
  // no need for fillEndTimes, end time of the fill is set to 0

  for (size_t i = 0; i < fillNumbers.size(); ++i) {
    auto fillPayload = std::make_shared<LHCInfoPerFill>();
    fillPayload->setFillNumber(fillNumbers[i]);
    fillPayload->setEnergy(energies[i]);
    fillPayload->setCreationTime(cond::time::from_boost(fillStartTimes[i]));
    fillPayload->setBeginTime(cond::time::from_boost(fillStableBeamBeginTimes[i]));
    fillPayload->setEndTime(0LL);
    handler.mockOmsFills.emplace_back(std::make_pair(cond::time::from_boost(fillStartTimes[i]), fillPayload));
  }

  std::vector<std::string> lumiJsons = {
      R"delimiter(
{
  "data": [
  ]
}
)delimiter"};

  for (size_t i = 0; i < fillNumbers.size(); ++i) {
    handler.mockLumiData[fillNumbers[i]] = cond::OMSServiceResult();
    handler.mockLumiData[fillNumbers[i]].parseData(lumiJsons[i]);
  }

  REQUIRE_NOTHROW(handler.getNewObjects());
  CHECK(handler.iovs().size() == 0);
}
