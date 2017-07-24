/*
 * Simple test of the tensorflow graph interface.
 *
 * Graph (from bin/data/simplegraph, created by test/creategraph.py):
 *   Input:
 *     - name = "input:0", shape = (batch, 10)
 *   Output:
 *     - name = "output:0", shape = (batch, 1)
 *
 * Usage:
 *   > test_tfgraph
 *
 * Author:
 *   Marcel Rieger
 */

#include <iostream>
#include <string>
#include <vector>

#include "PhysicsTools/TensorflowInterface/interface/Graph.h"
#include "PhysicsTools/TensorflowInterface/interface/Tensor.h"

int main(int argc, char* argv[])
{
    std::cout << std::endl
              << "test tf::Graph" << std::endl;

    // get the file containing the graph
    std::string cmsswBase = std::string(getenv("CMSSW_BASE"));
    std::string dataDir = cmsswBase + "/src/PhysicsTools/TensorflowInterface/bin/data";
    std::string graphFile = dataDir + "/simplegraph";
    std::cout << "load graph " << graphFile << std::endl;

    //
    // object definitions
    //

    // load and initialize the graph
    tf::Graph g(graphFile);

    // prepare input and output tensors
    tf::Shape xShape[] = { 1, 10 };
    tf::Tensor* x = g.defineInput(new tf::Tensor("input:0", 2, xShape));
    tf::Tensor* y = g.defineOutput(new tf::Tensor("output:0"));

    //
    // evaluation
    //

    // fill a single batch of the input tensor with consecutive numbers
    std::vector<float> v = { 0., 1., 2., 3., 4., 5., 6., 7., 8., 9. };
    x->setVector<float>(1, 0, v); // axis: 1, axis 0 (= batch dim) value: 0, vector: v
    // this is identical to
    // for (int i = 0; i < x->getShape(1); i++)
    // {
    //     x->setValue<float>(0, i, (float)i);
    // }

    // evaluation call
    // this does not return anything but changes the output tensor(s) in place which is faster
    g.eval();

    // some outputs
    std::cout << "rank : " << y->getRank() << std::endl;
    std::cout << "shape: ";
    for (int i = 0; i < y->getRank(); i++)
    {
        std::cout << y->getShape(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "value: " << y->getValue<float>(0, 0) << std::endl;

    //
    // cleanup
    //

    // remove tensors manually (you type 'new', you type 'delete')
    delete x;
    delete y;

    std::cout << std::endl << "done" << std::endl;

    return 0;
}
