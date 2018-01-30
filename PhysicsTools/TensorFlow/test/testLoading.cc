/*
 * Graph and session loading tests.
 * Based on TensorFlow C++ API 1.3.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include <boost/filesystem.hpp>
#include <cppunit/extensions/HelperMacros.h>
#include <stdexcept>

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

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

class testLoading : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(testLoading);
    CPPUNIT_TEST(checkAll);
    CPPUNIT_TEST_SUITE_END();

public:
    std::string dataPath;

    void setUp();
    void tearDown();
    void checkAll();

};

CPPUNIT_TEST_SUITE_REGISTRATION(testLoading);

void testLoading::setUp()
{
    dataPath = cmsswPath("/test/" + std::string(getenv("SCRAM_ARCH"))
        + "/" + boost::filesystem::unique_path().string());

    // create the graph
    std::string testPath = cmsswPath("/src/PhysicsTools/TensorFlow/test");
    std::string cmd = "python " + testPath + "/creategraph.py " + dataPath;
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe)
    {
        throw std::runtime_error("popen() failed!");
    }
    while (!feof(pipe.get()))
    {
        if (fgets(buffer.data(), 128, pipe.get()) != NULL)
        {
            result += buffer.data();
        }
    }
    std::cout << std::endl
              << result << std::endl;
}

void testLoading::tearDown()
{
    if (boost::filesystem::exists(dataPath))
    {
        boost::filesystem::remove_all(dataPath);
    }
}

void testLoading::checkAll()
{
    std::string exportDir = dataPath + "/simplegraph";

    // load the graph
    tensorflow::setLogging();
    tensorflow::MetaGraphDef* metaGraph = tensorflow::loadMetaGraph(exportDir);
    CPPUNIT_ASSERT(metaGraph != nullptr);

    // create a new, empty session
    tensorflow::Session* session1 = tensorflow::createSession();
    CPPUNIT_ASSERT(session1 != nullptr);

    // create a new session, using the meta graph
    tensorflow::Session* session2 = tensorflow::createSession(metaGraph, exportDir);
    CPPUNIT_ASSERT(session2 != nullptr);

    // example evaluation
    tensorflow::Tensor input(tensorflow::DT_FLOAT, { 1, 10 });
    float* d = input.flat<float>().data();
    for (size_t i = 0; i < 10; i++, d++)
    {
        *d = float(i);
    }
    tensorflow::Tensor scale(tensorflow::DT_FLOAT, {});
    scale.scalar<float>()() = 1.0;

    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status status = session2->Run({ { "input", input }, { "scale", scale } },
        { "output" }, {}, &outputs);
    if (!status.ok())
    {
        std::cout << status.ToString() << std::endl;
        CPPUNIT_ASSERT(false);
    }

    // check the output
    CPPUNIT_ASSERT(outputs.size() == 1);
    std::cout << outputs[0].DebugString() << std::endl;
    CPPUNIT_ASSERT(outputs[0].matrix<float>()(0, 0) == 46.);

    // run again using the convenience helper
    outputs.clear();
    tensorflow::run(session2, { { "input", input }, { "scale", scale } }, { "output" }, &outputs);
    CPPUNIT_ASSERT(outputs.size() == 1);
    std::cout << outputs[0].DebugString() << std::endl;
    CPPUNIT_ASSERT(outputs[0].matrix<float>()(0, 0) == 46.);

    // check for exception
    CPPUNIT_ASSERT_THROW(tensorflow::run(session2, { { "foo", input } }, { "output" }, &outputs),
        cms::Exception);

    // cleanup
    CPPUNIT_ASSERT(tensorflow::closeSession(session1));
    CPPUNIT_ASSERT(tensorflow::closeSession(session2));
    delete metaGraph;
}
