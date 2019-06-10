#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"

class TestRPCDetID: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestRPCDetID);
  CPPUNIT_TEST(test_region);
  CPPUNIT_TEST(test_ring);
  CPPUNIT_TEST(test_station);
  CPPUNIT_TEST(test_sector);
  CPPUNIT_TEST(test_layer);
  CPPUNIT_TEST(test_subsector);
  CPPUNIT_TEST(test_roll);
  CPPUNIT_TEST_SUITE_END();

public:
  TestRPCDetID() {}
  ~TestRPCDetID() {}
  void setUp();
  void tearDown();

  void test_region();
  void test_ring();
  void test_station();
  void test_sector();
  void test_layer();
  void test_subsector();
  void test_roll();

private:
  std::vector<uint32_t> rpcdetids_;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestRPCDetID);


void TestRPCDetID::setUp()
{
  rpcdetids_ = {
      637588132, 637575332, 637587748, 637579780, 637588134, 637575334, 637587750, 637579782,
      637636840, 637649256, 637637224, 637649640, 637636842, 637649258, 637637226, 637649642,
      637608004, 637620420, 637608388, 637620804, 637608006, 637620422, 637608390, 637620806
  };
}

void TestRPCDetID::tearDown()
{}

void TestRPCDetID::test_region()
{
  auto get_rpc_region = [](uint32_t id) { return (static_cast<int>((id >> 0) & 0X3) + (-1)); };

  for (const auto& id : rpcdetids_) {
    int region = (RPCDetId(id)).region();
    CPPUNIT_ASSERT_EQUAL(get_rpc_region(id), region);
  }
}

void TestRPCDetID::test_ring()
{
  auto get_rpc_ring = [](uint32_t id) { return (static_cast<int>((id >> 2) & 0X7) + (1)); };

  for (const auto& id : rpcdetids_) {
    int ring = (RPCDetId(id)).ring();
    CPPUNIT_ASSERT_EQUAL(get_rpc_ring(id), ring);
  }
}

void TestRPCDetID::test_station()
{
  auto get_rpc_station = [](uint32_t id) { return (static_cast<int>((id >> 5) & 0X3) + (1)); };

  for (const auto& id : rpcdetids_) {
    int station = (RPCDetId(id)).station();
    CPPUNIT_ASSERT_EQUAL(get_rpc_station(id), station);
  }
}

void TestRPCDetID::test_sector()
{
  auto get_rpc_sector = [](uint32_t id) { return (static_cast<int>((id >> 7) & 0XF) + (1)); };

  for (const auto& id : rpcdetids_) {
    int sector = (RPCDetId(id)).sector();
    CPPUNIT_ASSERT_EQUAL(get_rpc_sector(id), sector);
  }
}

void TestRPCDetID::test_layer()
{
  auto get_rpc_layer = [](uint32_t id) { return (static_cast<int>((id >> 11) & 0X1) + (1)); };

  for (const auto& id : rpcdetids_) {
    int layer = (RPCDetId(id)).layer();
    CPPUNIT_ASSERT_EQUAL(get_rpc_layer(id), layer);
  }
}

void TestRPCDetID::test_subsector()
{
  auto get_rpc_subsector = [](uint32_t id) { return (static_cast<int>((id >> 12) & 0X7) + (1)); };

  for (const auto& id : rpcdetids_) {
    int subsector = (RPCDetId(id)).subsector();
    CPPUNIT_ASSERT_EQUAL(get_rpc_subsector(id), subsector);
  }
}

void TestRPCDetID::test_roll()
{
  auto get_rpc_roll = [](uint32_t id) { return (static_cast<int>((id >> 15) & 0X7) + (0)); };

  for (const auto& id : rpcdetids_) {
    int roll = (RPCDetId(id)).roll();
    CPPUNIT_ASSERT_EQUAL(get_rpc_roll(id), roll);
  }
}

