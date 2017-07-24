/*
 * Simple performance test of the tensorflow interface.
 *
 * Graph (from bin/data/largegraph, created by test/createlargegraph.py):
 *   Input:
 *     - name = "input:0", shape = (batch, 100)
 *   Hidden layers:
 *     - n = 5, units = 200, activation = elu
 *   Output:
 *     - name = "output:0", shape = (batch, 10)
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

#include "PhysicsTools/TensorflowInterface/interface/Tensor.h"
#include "PhysicsTools/TensorflowInterface/interface/Graph.h"

long int getTime(timeval* t)
{
    gettimeofday(t, NULL);
    return t->tv_sec * 1000 + t->tv_usec / 1000;
}

int main(int argc, char* argv[])
{
    std::cout << std::endl << "test performance" << std::endl;

    struct timeval t;

    // get the file containing the graph
    std::string cmsswBase = std::string(getenv("CMSSW_BASE"));
    std::string dataDir = cmsswBase + "/src/PhysicsTools/TensorflowInterface/bin/data";
    std::string graphFile = dataDir + "/largegraph";
    std::cout << "load graph " << graphFile << std::endl;

    // load and initialize the graph
    tf::Graph g(graphFile);

    tf::Tensor* x = g.defineInput(new tf::Tensor("input:0"));
    tf::Tensor* y = g.defineOutput(new tf::Tensor("output:0"));

    // do the testing various different batch sizes
    int n = 1000;
    int batchSizes[] = {1, 10, 100, 1000};
    for (size_t i = 0; i < 4; i++)
    {
        std::cout << "run " << n << " evaluations for batch size " << batchSizes[i] << std::endl;

        // update tensors
        tf::Shape xShape[] = {batchSizes[i], 100};
        x->setArray(2, xShape);

        for (int j = 0; j < x->getShape(0); j++)
        {
            for (int k = 0; k < x->getShape(1); k++)
            {
                x->setValue<float>(j, k, (float)k);
            }
        }

        long int t0 = getTime(&t);
        for (int j = 0; j < n; j++)
        {
            g.eval();
        }
        long int t1 = getTime(&t);
        std::cout << "-> " << (t1 - t0) / (float)n << " ms per batch" << std::endl << std::endl;
    }

    // cleanup
    delete x;
    delete y;

    std::cout << std::endl << "done" << std::endl;

    return 0;
}
