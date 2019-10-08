/*
 * HelloWorld test of the TensorFlow interface.
 * Based on TensorFlow C++ API 1.3.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include <boost/filesystem.hpp>
#include <cppunit/extensions/HelperMacros.h>
#include <stdexcept>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"

using tensorflow::GraphDef;
using tensorflow::RunOptions;
using tensorflow::SavedModelBundle;
using tensorflow::Session;
using tensorflow::SessionOptions;
using tensorflow::Status;
using tensorflow::Tensor;

std::string cmsswPath(std::string path) {
  if (path.size() > 0 && path.substr(0, 1) != "/") {
    path = "/" + path;
  }

  std::string base = std::string(std::getenv("CMSSW_BASE"));
  std::string releaseBase = std::string(std::getenv("CMSSW_RELEASE_BASE"));

  return (boost::filesystem::exists(base.c_str()) ? base : releaseBase) + path;
}

class testSession : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSession);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string dataPath;

  void setUp();
  void tearDown();
  void checkAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSession);

void testSession::setUp() {
  dataPath =
      cmsswPath("/test/" + std::string(std::getenv("SCRAM_ARCH")) + "/" + boost::filesystem::unique_path().string());

  // create the graph
  std::string testPath = cmsswPath("/src/PhysicsTools/TensorFlow/test");
  std::string cmd = "python " + testPath + "/creategraph.py " + dataPath;
  std::array<char, 128> buffer;
  std::string result;
  std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (!feof(pipe.get())) {
    if (fgets(buffer.data(), 128, pipe.get()) != NULL) {
      result += buffer.data();
    }
  }
  std::cout << std::endl << result << std::endl;
}

void testSession::tearDown() {
  if (boost::filesystem::exists(dataPath)) {
    boost::filesystem::remove_all(dataPath);
  }
}

void testSession::checkAll() {
  std::string modelDir = dataPath + "/simplegraph";

  // object to load and run the graph / session
  Status status;
  SessionOptions sessionOptions;
  RunOptions runOptions;
  SavedModelBundle bundle;

  // load everything
  status = LoadSavedModel(sessionOptions, runOptions, modelDir, {"serve"}, &bundle);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    return;
  }

  // fetch the session
  Session* session = bundle.session.release();

  // prepare inputs
  Tensor input(tensorflow::DT_FLOAT, {1, 10});
  float* d = input.flat<float>().data();
  for (size_t i = 0; i < 10; i++, d++) {
    *d = float(i);
  }
  Tensor scale(tensorflow::DT_FLOAT, {});
  scale.scalar<float>()() = 1.0;

  // prepare outputs
  std::vector<tensorflow::Tensor> outputs;

  // session run
  status = session->Run({{"input", input}, {"scale", scale}}, {"output"}, {}, &outputs);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    return;
  }

  // log the output tensor
  std::cout << outputs[0].DebugString() << std::endl;

  // close the session
  status = session->Close();
  if (!status.ok()) {
    std::cerr << "error while closing session" << std::endl;
  }
  delete session;
}
