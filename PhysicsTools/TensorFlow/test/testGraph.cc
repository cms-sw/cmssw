/*
 * Test of the TensorFlow graph interface.
 * Based on TensorFlow C API 1.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include <boost/filesystem.hpp>
#include <cppunit/extensions/HelperMacros.h>
#include <stdexcept>

#include "PhysicsTools/TensorFlow/interface/Graph.h"

#include "FWCore/Utilities/interface/Exception.h"

class testGraph : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(testGraph);
    CPPUNIT_TEST(checkAll);
    CPPUNIT_TEST_SUITE_END();

public:
    std::string dataPath;

    void setUp();
    void tearDown();
    void checkAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testGraph);

void testGraph::setUp()
{
    dataPath = std::string(getenv("CMSSW_BASE")) + "/test/" + std::string(getenv("SCRAM_ARCH"))
        + "/" + boost::filesystem::unique_path().string();

    // create the graph
    std::string testPath = std::string(getenv("CMSSW_BASE")) + "/src/PhysicsTools/TensorFlow/test";
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

void testGraph::tearDown()
{
    if (boost::filesystem::exists(dataPath))
    {
        boost::filesystem::remove_all(dataPath);
    }
}

void testGraph::checkAll()
{
    // create an empty graph
    tf::Graph g;
    CPPUNIT_ASSERT(g.empty());

    // initialize it
    g.init(dataPath + "/simplegraph");
    CPPUNIT_ASSERT(!g.empty());

    // get a tensorflow operation object
    TF_Operation* op = g.getTFOperation("output");
    CPPUNIT_ASSERT(op != nullptr);

    // reset it again
    g.reset();
    CPPUNIT_ASSERT(g.empty());

    // operation should not be allowed anymore
    CPPUNIT_ASSERT_THROW(g.getTFOperation("output"), cms::Exception);
}
