#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include <cppunit/extensions/HelperMacros.h>
#include <boost/filesystem.hpp>

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

// Reference: https://www.tensorflow.org/api_docs/cc

std::string cmsswPath(std::string path)
{
    if (path.size() > 0 && path.substr(0, 1) != "/")
    {
        path = "/" + path;
    }

    std::string base = std::string(std::getenv("CMSSW_BASE"));
    std::string releaseBase = std::string(std::getenv("CMSSW_RELEASE_BASE"));

    return (boost::filesystem::exists(base.c_str()) ? base : releaseBase) + path;
}

class TestTensorFlow: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestTensorFlow);
  CPPUNIT_TEST(test_loading);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown();

  void test_loading();

private:
  std::vector<float> x_test_0;
  std::vector<float> x_test_1;
  std::vector<float> x_test_2;
  std::vector<float> x_test_3;
  std::vector<float> x_test_4;
  std::vector<float> x_test_5;
  std::vector<float> x_test_6;
  std::vector<float> x_test_7;
  std::vector<float> x_test_8;
  std::vector<float> x_test_9;

  std::vector<float> y_test_0;
  std::vector<float> y_test_1;
  std::vector<float> y_test_2;
  std::vector<float> y_test_3;
  std::vector<float> y_test_4;
  std::vector<float> y_test_5;
  std::vector<float> y_test_6;
  std::vector<float> y_test_7;
  std::vector<float> y_test_8;
  std::vector<float> y_test_9;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestTensorFlow);


void TestTensorFlow::setUp()
{
  x_test_0 = {  52.,    0.,    0.,    0., -109.,    0.,    0., -138.,  -89.,    0.,    0.,  179.,
                15.,    0.,    0.,    0.,    9.,    0.,    0.,   10.,    9.,    0.,    0.,   16.,
               -10.,    0.,    0.,    0.,    1.,  -35.,    4.,    0.,    0.,    0.,    5.,    6.};
  x_test_1 = { 125.,    0.,    0.,    4.,   76.,    0.,    0.,   26.,   92.,  178.,  -48.,  218.,
                21.,    0.,    0.,   19.,   19.,    0.,    0.,   18.,   18.,   21.,   19.,   22.,
                -5.,    0.,    0.,    2.,    2.,  -23.,    5.,    0.,    0.,   -6.,   -5.,   -6.};
  x_test_2 = {  84.,    0.,   -8.,  -21.,  -24.,    0.,    0.,  -26.,  -26.,    0.,    3.,  112.,
                12.,    0.,   11.,   11.,   11.,    0.,    0.,   12.,   11.,    0.,   12.,   12.,
                -3.,    0.,    0.,    0.,    0.,   -7.,    5.,    0.,   -5.,    5.,    6.,   -6.};
  x_test_3 = {   0.,   27.,    0.,  -36.,  -40.,   16.,    0.,  -44.,  -44.,    0.,    0.,    0.,
                 0.,   52.,    0.,   53.,   52.,   52.,    0.,   48.,   52.,    0.,    0.,    0.,
                 0.,   -4.,    0.,    0.,    0.,    0.,    0.,   -5.,    0.,    4.,    6.,    0.};
  x_test_4 = {-318.,    0.,  -13.,    0.,    0.,    0.,    0., -191.,    0., -404.,   -5.,    0.,
                35.,    0.,   34.,    0.,    0.,    0.,    0.,   34.,    0.,   37.,   34.,    0.,
                 9.,    0.,   -2.,    0.,    0.,    0.,   -5.,    0.,    5.,    0.,    0.,    0.};
  x_test_5 = {   0.,    0.,    0.,   20.,  -40., -184.,    0.,    0.,  -52.,    0.,    0.,    0.,
                 0.,    0.,    0.,   53.,   52.,   56.,    0.,    0.,   52.,    0.,    0.,    0.,
                 0.,    0.,    0.,   -3.,   -3.,    0.,    0.,    0.,    0.,   -5.,   -6.,    0.};
  x_test_6 = {   0.,    0.,    0.,   -6.,   -7.,    0.,    0.,   -2.,  -14.,   36.,    6.,    0.,
                 0.,    0.,   27.,   26.,   27.,    0.,    0.,   27.,   27.,   25.,   29.,    0.,
                 0.,    0.,    0.,   -1.,   -1.,    0.,    0.,    0.,   -5.,    6.,    6.,    0.};
  x_test_7 = {-367.,    0.,   10.,    3.,  -69.,    0.,    0.,  -32.,  -87.,    0.,  -16.,    0.,
                31.,    0.,   29.,   28.,   28.,    0.,    0.,   28.,   28.,    0.,   30.,    0.,
                 8.,    0.,    2.,   -2.,   -2.,    0.,    5.,    0.,   -6.,    6.,    6.,    0.};
  x_test_8 = {   0.,    0.,    8.,   20.,   76.,    0.,    0.,   45.,  107.,  392.,   12.,    0.,
                 0.,    0.,   22.,   21.,   21.,    0.,    0.,   22.,   21.,   26.,   23.,    0.,
                 0.,    0.,    0.,    2.,    3.,    0.,    0.,    0.,   -6.,    6.,    6.,    0.};
  x_test_9 = {   0.,   36.,    0.,  -28.,  -28.,   16.,    0.,  -20.,  -20.,    0.,    0.,    0.,
                 0.,   50.,    0.,   50.,   50.,   52.,    0.,   52.,   52.,    0.,    0.,    0.,
                 0.,   -4.,    0.,   -1.,    0.,    0.,    0.,   -6.,    0.,   -6.,   -6.,    0.};

  y_test_0 = {  4.6249207e+01,  9.2568684e-05};
  y_test_1 = {  3.4682186e+01,  1.6448584e-04};
  y_test_2 = {  1.5179890e+01,  9.9294585e-01};
  y_test_3 = {  1.1889083e+01,  8.7512171e-01};
  y_test_4 = { -3.7088535e+01,  3.7408798e-05};
  y_test_5 = { -3.8221695e+01,  4.3776014e-05};
  y_test_6 = {  3.5971706e+00,  9.9732649e-01};
  y_test_7 = { -3.7217415e+01,  3.6537862e-05};
  y_test_8 = {  3.8733337e+01,  1.8750130e-04};
  y_test_9 = {  1.2629346e+01,  8.0939955e-01};
}

void TestTensorFlow::tearDown()
{}

void TestTensorFlow::test_loading()
{
  std::string pbFile = cmsswPath("/src/L1Trigger/L1TMuonEndCap/data/emtfpp_tf_graphs/model_graph.26.pb");

  // load the graph
  tensorflow::setLogging();
  tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef(pbFile);
  CPPUNIT_ASSERT(graphDef != nullptr);

  // create a new session and add the graphDef
  tensorflow::Session* session = tensorflow::createSession(graphDef);
  CPPUNIT_ASSERT(session != nullptr);

  // prepare inputs
  tensorflow::Tensor input(tensorflow::DT_FLOAT, { 1, 36 });
  float* d = input.flat<float>().data();
  std::copy(x_test_0.begin(), x_test_0.end(), d);

  // prepare outputs
  std::vector<tensorflow::Tensor> outputs;
  float output_check_0 = y_test_0.at(0);
  float output_check_1 = y_test_0.at(1);

  // session run
  tensorflow::Status status = session->Run({ { "input_1", input } }, { "regr/BiasAdd", "discr/Sigmoid" }, {}, &outputs);  // inputs, fetch_outputs, run_outputs, &outputs
  if (!status.ok())
  {
      std::cout << status.ToString() << std::endl;
      CPPUNIT_ASSERT(false);
  }

  // check the output
  CPPUNIT_ASSERT(outputs.size() == 2);
  //std::cout << "output 0: " << outputs[0].DebugString() << std::endl;
  //std::cout << "output 1: " << outputs[1].DebugString() << std::endl;

  auto almost_equal = [](auto a, auto b) {
    auto relative_difference = std::abs((a - b) / std::min(a, b));
    return (relative_difference < 1e-5);
  };
  CPPUNIT_ASSERT(almost_equal(outputs[0].matrix<float>()(0, 0), output_check_0) );
  CPPUNIT_ASSERT(almost_equal(outputs[1].matrix<float>()(0, 0), output_check_1) );

  // repeat
  for (size_t i=1; i<10; i++) {
    if (i == 1) {
      std::copy(x_test_1.begin(), x_test_1.end(), d);
      output_check_0 = y_test_1.at(0);
      output_check_1 = y_test_1.at(1);
    } else if (i == 2) {
      std::copy(x_test_2.begin(), x_test_2.end(), d);
      output_check_0 = y_test_2.at(0);
      output_check_1 = y_test_2.at(1);
    } else if (i == 3) {
      std::copy(x_test_3.begin(), x_test_3.end(), d);
      output_check_0 = y_test_3.at(0);
      output_check_1 = y_test_3.at(1);
    } else if (i == 4) {
      std::copy(x_test_4.begin(), x_test_4.end(), d);
      output_check_0 = y_test_4.at(0);
      output_check_1 = y_test_4.at(1);
    } else if (i == 5) {
      std::copy(x_test_5.begin(), x_test_5.end(), d);
      output_check_0 = y_test_5.at(0);
      output_check_1 = y_test_5.at(1);
    } else if (i == 6) {
      std::copy(x_test_6.begin(), x_test_6.end(), d);
      output_check_0 = y_test_6.at(0);
      output_check_1 = y_test_6.at(1);
    } else if (i == 7) {
      std::copy(x_test_7.begin(), x_test_7.end(), d);
      output_check_0 = y_test_7.at(0);
      output_check_1 = y_test_7.at(1);
    } else if (i == 8) {
      std::copy(x_test_8.begin(), x_test_8.end(), d);
      output_check_0 = y_test_8.at(0);
      output_check_1 = y_test_8.at(1);
    } else if (i == 9) {
      std::copy(x_test_9.begin(), x_test_9.end(), d);
      output_check_0 = y_test_9.at(0);
      output_check_1 = y_test_9.at(1);
    }

    tensorflow::Status status = session->Run({ { "input_1", input } }, { "regr/BiasAdd", "discr/Sigmoid" }, {}, &outputs);
    CPPUNIT_ASSERT(status.ok());
    CPPUNIT_ASSERT(outputs.size() == 2);
    //std::cout << "output 0: " << outputs[0].DebugString() << std::endl;
    //std::cout << "output 1: " << outputs[1].DebugString() << std::endl;
    CPPUNIT_ASSERT(almost_equal(outputs[0].matrix<float>()(0, 0), output_check_0) );
    CPPUNIT_ASSERT(almost_equal(outputs[1].matrix<float>()(0, 0), output_check_1) );
  }

  // cleanup
  CPPUNIT_ASSERT(tensorflow::closeSession(session));
  delete graphDef;
}
