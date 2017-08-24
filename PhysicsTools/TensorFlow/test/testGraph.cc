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
#include "PhysicsTools/TensorFlow/interface/Tensor.h"
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
    //
    // graph creation and evaluation tests
    //

    // create the graph
    tf::Graph g(dataPath + "/simplegraph");

    // define the input
    tf::Shape xShape[] = {2, 10};
    tf::Tensor* x = new tf::Tensor(2, xShape);
    g.defineInput(x, "input");

    tf::Tensor* s = new tf::Tensor(0, 0);
    g.defineInput(s, "scale");

    // define the output
    tf::Tensor* y = new tf::Tensor();
    g.defineOutput(y, "output");

    // set values for both batches
    std::vector<float> values0 = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    x->setVector<float>(1, 0, values0);

    std::vector<float> values1 = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18 };
    x->setVector<float>(1, 1, values1);

    // do not scale at all
    *s->getPtr<float>() = 1.0;

    // evaluate
    g.eval();

    // check the output, should be 46 and 91 for the simplegraph
    CPPUNIT_ASSERT(*y->getPtr<float>(0, 0) == 46);
    CPPUNIT_ASSERT(*y->getPtr<float>(1, 0) == 91);


    //
    // API tests
    //

    CPPUNIT_ASSERT(!g.empty());

    g.reset();
    CPPUNIT_ASSERT(g.empty());

    g.init(dataPath + "/simplegraph");
    CPPUNIT_ASSERT(!g.empty());
    CPPUNIT_ASSERT(g.nInputs() == 0);
    CPPUNIT_ASSERT(g.nOutputs() == 0);

    tf::GraphIO* xIO = g.defineInput(x, "input");
    tf::GraphIO* sIO = g.defineInput(s, "scale");
    tf::GraphIO* yIO = g.defineOutput(y, "output");
    CPPUNIT_ASSERT(g.nInputs() == 2);
    CPPUNIT_ASSERT(g.nOutputs() == 1);
    CPPUNIT_ASSERT(g.hasInput(x, "input"));
    CPPUNIT_ASSERT(g.hasInput(s, "scale"));
    CPPUNIT_ASSERT(g.hasOutput(y, "output"));
    CPPUNIT_ASSERT(g.hasInput(xIO));
    CPPUNIT_ASSERT(g.hasInput(sIO));
    CPPUNIT_ASSERT(g.hasOutput(yIO));

    CPPUNIT_ASSERT_THROW(g.defineInput(x, "input"), cms::Exception);

    g.removeInput(x, "input");
    g.removeInput(s, "scale");
    g.removeOutput(y, "output");
    CPPUNIT_ASSERT(g.nInputs() == 0);
    CPPUNIT_ASSERT(g.nOutputs() == 0);

    xIO = g.defineInput(x, "input");
    sIO = g.defineInput(s, "scale");
    yIO = g.defineOutput(y, "output");
    g.removeInput(xIO);
    g.removeInput(sIO);
    g.removeOutput(yIO);
    CPPUNIT_ASSERT(g.nInputs() == 0);
    CPPUNIT_ASSERT(g.nOutputs() == 0);

    g.defineInput(x, "input");
    g.defineInput(s, "scale");
    g.defineOutput(y, "output");
    CPPUNIT_ASSERT(g.nInputs() == 2);
    CPPUNIT_ASSERT(g.nOutputs() == 1);

    g.eval();
    CPPUNIT_ASSERT(*y->getPtr<float>(0, 0) == 46);
    CPPUNIT_ASSERT(*y->getPtr<float>(1, 0) == 91);

    for (size_t i = 0; i < 100; i++) g.eval();
    CPPUNIT_ASSERT(*y->getPtr<float>(0, 0) == 46);
    CPPUNIT_ASSERT(*y->getPtr<float>(1, 0) == 91);


    // cleanup
    delete x;
    delete s;
    delete y;
}
