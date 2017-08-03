/*
 * Performance test of the TensorFlow interface.
 *
 * Graph (from bin/data/largegraph, created by test/createlargegraph.py):
 *   Input:
 *     - op name = "input", op index = 0, shape = (batch, 100)
 *   Hidden layers:
 *     - n = 5, units = 200, activation = elu
 *   Output:
 *     - op name = "output", op index = 0, shape = (batch, 10)
 *
 * Usage:
 *   > test_tfperf
 *
 * Author:
 *   Marcel Rieger
 */

#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <sys/time.h>

#include "PhysicsTools/TensorFlow/interface/Tensor.h"
#include "PhysicsTools/TensorFlow/interface/Graph.h"

long int getTime(timeval* t)
{
    gettimeofday(t, NULL);
    return t->tv_sec * 1000 + t->tv_usec / 1000;
}

int main(int argc, char* argv[])
{
    std::cout << std::endl << "test tf::Graph::eval performance" << std::endl;

    struct timeval t;

    // get the file containing the graph
    std::string cmsswBase = std::string(getenv("CMSSW_BASE"));
    std::string dataDir = cmsswBase + "/src/PhysicsTools/TensorFlow/bin/data";
    std::string graphDir = dataDir + "/largegraph";

    // load and initialize the graph
    tf::Graph* g = new tf::Graph(graphDir);

    // create tensors
    tf::Tensor* x = new tf::Tensor();
    tf::Tensor* y = new tf::Tensor();
    g->defineOutput(y, "output");

    // do the testing various different batch sizes
    int n = 1000;
    int batchSizes[] = {1, 10, 100, 1000};
    for (size_t i = 0; i < 4; i++)
    {
        std::cout << "run " << n << " evaluations for batch size " << batchSizes[i] << std::endl;

        // init the input tensor and define it in the graph
        tf::Shape xShape[] = {batchSizes[i], 100};
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
        std::cout << "-> " << (t1 - t0) / (float)n << " ms per batch" << std::endl << std::endl;

        // remove the input tensor, it will re-redefined with a new batch size in the next iteration
        g->removeInput(x, "input");
    }

    // cleanup
    delete x;
    delete y;
    delete g;

    std::cout << std::endl << "done" << std::endl;

    return 0;
}
