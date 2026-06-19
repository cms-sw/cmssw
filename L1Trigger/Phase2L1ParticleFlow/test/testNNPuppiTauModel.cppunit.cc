//Test of the externally-built NNPuppiTauModel hls4ml emulator: model
//loading (incl. multiple versions/instances in the same process, the
//scenario that used to clash on un-namespaced weight symbols, see
//cms-sw/cmssw#49632) and a basic predict() sanity check.

#include "ap_fixed.h"
#include "hls4ml/emulator.h"

#include "cppunit/extensions/HelperMacros.h"
#include <array>
#include <utility>
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"

namespace {
  typedef ap_fixed<16, 10> input_t;
  typedef ap_fixed<16, 6> result_t;
  constexpr int kNInputs = 80;
}  // namespace

class test_NNPuppiTauModel : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(test_NNPuppiTauModel);
  CPPUNIT_TEST(doModelLoad);
  CPPUNIT_TEST(doMultiModelLoad);
  CPPUNIT_TEST(doPredictSanity);
  CPPUNIT_TEST_SUITE_END();

public:
  void doModelLoad();
  void doMultiModelLoad();
  void doPredictSanity();
};

CPPUNIT_TEST_SUITE_REGISTRATION(test_NNPuppiTauModel);

void test_NNPuppiTauModel::doModelLoad() {
  auto loader = hls4mlEmulator::ModelLoader("NNPuppiTauModel_v1");
  auto model = loader.load_model();
}

void test_NNPuppiTauModel::doMultiModelLoad() {
  auto loader_a = hls4mlEmulator::ModelLoader("NNPuppiTauModel_v1");
  auto loader_b = hls4mlEmulator::ModelLoader("NNPuppiTauModel_v1");
  auto model_a = loader_a.load_model();
  auto model_b = loader_b.load_model();
}

void test_NNPuppiTauModel::doPredictSanity() {
  auto loader = hls4mlEmulator::ModelLoader("NNPuppiTauModel_v1");
  auto model = loader.load_model();

  // All-zero input: not physically representative, but exercises the full
  // dense/relu/sigmoid layer sequence and checks the returned values land in
  // the expected output ranges.
  input_t input[kNInputs];
  for (int i = 0; i < kNInputs; ++i)
    input[i] = input_t(0);

  typedef std::pair<std::array<result_t, 1>, std::array<result_t, 1>> pairtype;
  pairtype result;
  model->prepare_input(input);
  model->predict();
  model->read_result(&result);

  // nn_id (result.second) is the output of a sigmoid layer: must lie in [0, 1].
  double nn_id = double(result.second[0]);
  CPPUNIT_ASSERT(nn_id >= 0. && nn_id <= 1.);
}
