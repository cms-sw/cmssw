/*
 * Test of the TensorFlow graph interface.
 *
 * Graph (from bin/data/simplegraph, created by test/creategraph.py):
 *   Input:
 *     - op = "input", op index = 0, shape = (batch, 10)
 *   Output:
 *     - op name = "output", op index = 0, shape = (batch, 1)
 *
 * Usage:
 *   > test_tfgraph
 *
 * Author:
 *   Marcel Rieger
 */

#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>

#include <PhysicsTools/TensorFlow/interface/Tensor.h>
#include <PhysicsTools/TensorFlow/interface/Graph.h>

void test(bool success, const std::string& msg)
{
    if (success)
    {
        std::cout << "    " << msg << std::endl;
    }
    else
    {
        throw std::runtime_error("test failed: " + msg);
    }
}

int main()
{
    std::cout << std::endl << "test tf::Graph" << std::endl;

    // get the directory that contains the simple SavedModel
    std::string cmsswBase = std::string(getenv("CMSSW_BASE"));
    std::string graphDir = cmsswBase + "/src/PhysicsTools/TensorFlow/bin/data/simplegraph";

    //
    // graph creation and evaluation tests
    //

    // create the graph
    tf::Graph g(graphDir);

    // define the input
    tf::Shape xShape[] = {2, 10};
    tf::Tensor* x = new tf::Tensor(2, xShape);
    g.defineInput(x, "input");

    // define the output
    tf::Tensor* y = new tf::Tensor();
    g.defineOutput(y, "output");

    // set values for both batches
    std::vector<float> values0 = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    x->setVector<float>(1, 0, values0);

    std::vector<float> values1 = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18 };
    x->setVector<float>(1, 1, values1);

    // evaluate
    g.eval();

    // check the output, should be 46 and 91 for the simplegraph
    test(*y->getPtr<float>(0, 0) == 46, "output of batch 0 should be 46");
    test(*y->getPtr<float>(1, 0) == 91, "output of batch 1 should be 91");


    //
    // API tests
    //

    bool catched;

    test(!g.empty(), "graph should not be empty");

    g.reset();
    test(g.empty(), "graph should be empty");

    g.init(graphDir);
    test(!g.empty(), "graph should not be empty");

    test(g.nInputs() == 0, "graph should have 0 inputs");
    test(g.nOutputs() == 0, "graph should have 0 outputs");

    tf::GraphIO* xIO = g.defineInput(x, "input");
    tf::GraphIO* yIO = g.defineOutput(y, "output");
    test(g.nInputs() == 1, "graph should have 1 input");
    test(g.nOutputs() == 1, "graph should have 1 output");

    test(g.hasInput(x, "input"), "graph should have x as an input");
    test(g.hasOutput(y, "output"), "graph should have y as an output");

    test(g.hasInput(xIO), "graph should have x as an input");
    test(g.hasOutput(yIO), "graph should have y as an output");

    catched = false;
    try { g.defineInput(x, "input"); }
    catch (...) { catched = true; }
    test(catched, "graph should detect duplicate inputs");

    g.removeInput(x, "input");
    g.removeOutput(y, "output");
    test(g.nInputs() == 0, "graph should have 0 inputs");
    test(g.nOutputs() == 0, "graph should have 0 outputs");

    xIO = g.defineInput(x, "input");
    yIO = g.defineOutput(y, "output");
    g.removeInput(xIO);
    g.removeOutput(yIO);
    test(g.nInputs() == 0, "graph should have 0 inputs");
    test(g.nOutputs() == 0, "graph should have 0 outputs");

    xIO = g.defineInput(x, "input");
    yIO = g.defineOutput(y, "output");
    test(g.nInputs() == 1, "graph should have 1 input");
    test(g.nOutputs() == 1, "graph should have 1 output");

    g.eval();
    test(*y->getPtr<float>(0, 0) == 46, "output of batch 0 should be 46");
    test(*y->getPtr<float>(1, 0) == 91, "output of batch 1 should be 91");

    for (size_t i = 0; i < 100; i++) g.eval();
    test(*y->getPtr<float>(0, 0) == 46, "output of batch 0 should be 46");
    test(*y->getPtr<float>(1, 0) == 91, "output of batch 1 should be 91");


    // cleanup
    delete x;
    delete y;

    return 0;
}
