/*
 * Test of the TensorFlow session interface.
 * Based on TensorFlow C API 1.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include <boost/filesystem.hpp>
#include <cppunit/extensions/HelperMacros.h>
#include <stdexcept>

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "FWCore/Utilities/interface/Exception.h"

class testSession : public CppUnit::TestFixture
{
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

void testSession::setUp()
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

void testSession::tearDown()
{
    if (boost::filesystem::exists(dataPath))
    {
        boost::filesystem::remove_all(dataPath);
    }
}

void testSession::checkAll()
{
    //
    // graph / session interplay
    //

    // create an empty session
    tf::Session s;
    CPPUNIT_ASSERT(s.empty());

    // create a graph and initialize the session
    tf::Graph g(dataPath + "/simplegraph");
    s.init(&g);
    CPPUNIT_ASSERT(!s.empty());

    // reset and initialize again
    s.reset();
    CPPUNIT_ASSERT(s.empty());
    s.init(&g);
    CPPUNIT_ASSERT(!s.empty());


    //
    // stateful evaluation
    //

    // there should be not in/outputs
    CPPUNIT_ASSERT(s.nInputs() == 0);
    CPPUNIT_ASSERT(s.nOutputs() == 0);

    // add inputs
    tf::Shape xShape[] = { 2, 10 };
    tf::Tensor* x = new tf::Tensor(2, xShape);
    tf::IO* xIO = s.addInput(x, "input");
    tf::Tensor* c = new tf::Tensor(0, 0);
    tf::IO* cIO = s.addInput(c, "scale");
    CPPUNIT_ASSERT(s.nInputs() == 2);
    CPPUNIT_ASSERT(s.hasInput(x, "input"));
    CPPUNIT_ASSERT(s.hasInput(c, "scale"));
    CPPUNIT_ASSERT(s.hasInput(xIO));
    CPPUNIT_ASSERT(s.hasInput(cIO));

    // add outputs
    tf::Tensor* y = new tf::Tensor();
    tf::IO* yIO = s.addOutput(y, "output");
    CPPUNIT_ASSERT(s.nOutputs() == 1);
    CPPUNIT_ASSERT(s.hasOutput(y, "output"));
    CPPUNIT_ASSERT(s.hasOutput(yIO));

    // check duplicates
    CPPUNIT_ASSERT_THROW(s.addInput(x, "input"), cms::Exception);

    // remove and add in/outputs again
    s.removeInput(x, "input");
    s.removeInput(c, "scale");
    s.removeOutput(y, "output");
    CPPUNIT_ASSERT(s.nInputs() == 0);
    CPPUNIT_ASSERT(s.nOutputs() == 0);
    s.addInput(x, "input");
    s.addInput(c, "scale");
    s.addOutput(y, "output");
    CPPUNIT_ASSERT(s.nInputs() == 2);
    CPPUNIT_ASSERT(s.nOutputs() == 1);

    // set values for both batches
    std::vector<float> values0 = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    x->setVector<float>(1, 0, values0);

    std::vector<float> values1 = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18 };
    x->setVector<float>(1, 1, values1);

    // do not scale at all
    *c->getPtr<float>() = 1.0;

    // run
    // the output should be 46 and 91 for the simplegraph
    s.run();
    CPPUNIT_ASSERT(*y->getPtr<float>(0, 0) == 46);
    CPPUNIT_ASSERT(*y->getPtr<float>(1, 0) == 91);

    // repeated evaluation
    for (size_t i = 0; i < 100; i++)
    {
        s.run();
    }
    CPPUNIT_ASSERT(*y->getPtr<float>(0, 0) == 46);
    CPPUNIT_ASSERT(*y->getPtr<float>(1, 0) == 91);

    // reset again
    s.reset();
    CPPUNIT_ASSERT(s.empty());
    CPPUNIT_ASSERT(s.nInputs() == 0);
    CPPUNIT_ASSERT(s.nOutputs() == 0);


    //
    // stateless evaluation
    //

    // initialize again
    s.init(&g);
    CPPUNIT_ASSERT(!s.empty());

    // define input and output objects
    tf::IO* xIn = s.createIO(x, "input");
    tf::IO* cIn = s.createIO(c, "scale");
    tf::IO* yOut = s.createIO(y, "output");
    tf::IOs inputs = { xIn, cIn };
    tf::IOs outputs = { yOut };
    s.run(inputs, outputs);
    CPPUNIT_ASSERT(*y->getPtr<float>(0, 0) == 46);
    CPPUNIT_ASSERT(*y->getPtr<float>(1, 0) == 91);

    // reset the scale and run again
    *c->getPtr<float>() = 2.0;
    s.run(inputs, outputs);
    CPPUNIT_ASSERT(*y->getPtr<float>(0, 0) == 92);
    CPPUNIT_ASSERT(*y->getPtr<float>(1, 0) == 182);

    // inputs can also be passed by value
    *c->getPtr<float>() = 3.0;
    s.run({ xIn, cIn }, outputs);
    CPPUNIT_ASSERT(*y->getPtr<float>(0, 0) == 138);
    CPPUNIT_ASSERT(*y->getPtr<float>(1, 0) == 273);


    // cleanup
    delete x;
    delete c;
    delete y;
    delete xIn;
    delete cIn;
    delete yOut;
}
