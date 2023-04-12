/*
 * Tests visible device interface 
 * For more info,
 * https://github.com/tensorflow/tensorflow/blob/3bc73f5e2ac437b1d9d559751af789c8c965a7f9/tensorflow/core/framework/device_attributes.proto
 * https://stackoverflow.com/questions/74110853/how-to-check-if-tensorflow-is-using-the-cpu-with-the-c-api\
 *
 * Author: Davide Valsecchi
 */

#include <stdexcept>
#include <cppunit/extensions/HelperMacros.h>

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "tensorflow/core/framework/device_attributes.pb.h"

#include "testBase.h"

class testVisibleDevices : public testBase {
  CPPUNIT_TEST_SUITE(testVisibleDevices);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string pyScript() const override;
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testVisibleDevices);

std::string testVisibleDevices::pyScript() const { return "createconstantgraph.py"; }

void testVisibleDevices::test() {
  std::string pbFile = dataPath_ + "/constantgraph.pb";
  tensorflow::Backend backend = tensorflow::Backend::cpu;
  tensorflow::Options options{backend};

  // load the graph
  tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef(pbFile);
  CPPUNIT_ASSERT(graphDef != nullptr);

  // create a new session and add the graphDef
  tensorflow::Session* session = tensorflow::createSession(graphDef, options);
  CPPUNIT_ASSERT(session != nullptr);

  // check for exception
  CPPUNIT_ASSERT_THROW(tensorflow::createSession(nullptr, options), cms::Exception);

  std::vector<tensorflow::DeviceAttributes> response;
  tensorflow::Status status = session->ListDevices(&response);
  CPPUNIT_ASSERT(status.ok());

  // If a single device is found, we assume that it's the CPU.
  // You can check that name if you want to make sure that this is the case
  std::cout << "Available devices: " << response.size() << std::endl;
  CPPUNIT_ASSERT(response.size() == 1);

  for (unsigned int i = 0; i < response.size(); ++i) {
    std::cout << i << " " << response[i].name() << " type: " << response[i].device_type() << std::endl;
  }

  // cleanup
  CPPUNIT_ASSERT(tensorflow::closeSession(session));
  CPPUNIT_ASSERT(session == nullptr);
  delete graphDef;
}
