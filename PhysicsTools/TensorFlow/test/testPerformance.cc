/*
 * Test of the TensorFlow interface performance.
 * Based on TensorFlow C API 1.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include <boost/filesystem.hpp>
#include <cppunit/extensions/HelperMacros.h>
#include <stdexcept>
#include <sys/time.h>

#include "PhysicsTools/TensorFlow/interface/Graph.h"
#include "PhysicsTools/TensorFlow/interface/Tensor.h"

class testPerformance : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(testPerformance);
    CPPUNIT_TEST(checkAll);
    CPPUNIT_TEST_SUITE_END();

public:
    std::string dataPath;

    void setUp();
    void tearDown();
    void checkAll();

    static long int getTime(timeval* t)
    {
        gettimeofday(t, NULL);
        return t->tv_sec * 1000 + t->tv_usec / 1000;
    }
};

CPPUNIT_TEST_SUITE_REGISTRATION(testPerformance);

void testPerformance::setUp()
{
    dataPath = std::string(getenv("CMSSW_BASE")) + "/test/" + std::string(getenv("SCRAM_ARCH"))
        + "/" + boost::filesystem::unique_path().string();

    // create the graph
    std::string testPath = std::string(getenv("CMSSW_BASE")) + "/src/PhysicsTools/TensorFlow/test";
    std::string cmd = "python " + testPath + "/createlargegraph.py " + dataPath;
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

void testPerformance::tearDown()
{
    if (boost::filesystem::exists(dataPath))
    {
        boost::filesystem::remove_all(dataPath);
    }
}

void testPerformance::checkAll()
{
    struct timeval t;

    // load and initialize the graph
    tf::Graph* g = new tf::Graph(dataPath + "/largegraph");

    // create tensors
    tf::Tensor* x = new tf::Tensor();
    tf::Tensor* y = new tf::Tensor();
    g->defineOutput(y, "output");

    // do the testing various different batch sizes
    int n = 1000;
    int batchSizes[] = { 1, 10, 100, 1000 };
    for (size_t i = 0; i < 4; i++)
    {
        std::cout << "run " << n << " evaluations for batch size " << batchSizes[i] << std::endl;

        // init the input tensor and define it in the graph
        tf::Shape xShape[] = { batchSizes[i], 100 };
        x->init(2, xShape);
        g->defineInput(x, "input");

        // set identical values per batch
        std::vector<float> values;
        for (float v = 0; v < 100; v++)
        {
            values.push_back(v);
        }
        for (int b = 0; b < batchSizes[i]; b++)
        {
            x->setVector<float>(1, b, values);
        }

        // measure the evaluation time with n repititions
        long int t0 = getTime(&t);
        for (int j = 0; j < n; j++)
        {
            g->eval();
        }
        long int t1 = getTime(&t);
        std::cout << "-> " << (t1 - t0) / (float)n << " ms per batch" << std::endl
                  << std::endl;

        // remove the input tensor, it will re-redefined with a new batch size in the next iteration
        g->removeInput(x, "input");
    }

    // cleanup
    delete x;
    delete y;
    delete g;
}
