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

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

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

    // do the testing for various different batch sizes
    int n = 1000;
    int batchSizes[] = { 1, 10, 100, 1000 };

    // load the graph
    tf::Graph g(dataPath + "/largegraph");

    // create tensors
    tf::Tensor* x = new tf::Tensor();
    tf::Tensor* y = new tf::Tensor();


    //
    // stateful evaluation with multiple threads
    //

    std::cout << "test multi-threaded stateful run calls" << std::endl << std::endl;

    tf::Session s(&g, true);

    // add the output
    s.addOutput(y, "output");

    for (size_t i = 0; i < 4; i++)
    {
        std::cout << "run " << n << " evaluations for batch size " << batchSizes[i] << std::endl;

        // init the input tensor and add it to the graph
        tf::Shape xShape[] = { batchSizes[i], 100 };
        x->init(2, xShape);
        s.addInput(x, "input");

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

        // measure the run time with n repititions
        long int t0 = getTime(&t);
        for (int j = 0; j < n; j++)
        {
            s.run();
        }
        long int t1 = getTime(&t);
        std::cout << "-> " << (t1 - t0) / (float)n << " ms per batch" << std::endl
                  << std::endl;

        // remove the input tensor, it will re-redefined with a new batch size in the next iteration
        s.removeInput(x, "input");
    }


    //
    // stateless evaluation with multiple threads
    //

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "test single-threaded stateless run calls" << std::endl << std::endl;

    // reset and add the graph again
    s.reset();
    s.init(&g);

    // create the inputs
    tf::IO* xIn = s.createIO(x, "input");
    tf::IOs inputs = { xIn };

    // create the outputs
    tf::IO* yOut = s.createIO(y, "output");
    tf::IOs outputs = { yOut };

    // do the testing for various different batch sizes
    for (size_t i = 0; i < 4; i++)
    {
        std::cout << "run " << n << " evaluations for batch size " << batchSizes[i] << std::endl;

        // init the input tensor
        tf::Shape xShape[] = { batchSizes[i], 100 };
        x->init(2, xShape);

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

        // measure the run time with n repititions
        long int t0 = getTime(&t);
        for (int j = 0; j < n; j++)
        {
            s.run(inputs, outputs);
        }
        long int t1 = getTime(&t);
        std::cout << "-> " << (t1 - t0) / (float)n << " ms per batch" << std::endl
                  << std::endl;
    }


    // cleanup
    delete x;
    delete y;
    delete yOut;
}
