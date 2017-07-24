/*
 * Simple test of the tensorflow tensor interface.
 *
 * Usage:
 *   > test_tftensor
 *
 * Author:
 *   Marcel Rieger
 */

#include <iostream>
#include <string>
#include <vector>

#include "PhysicsTools/TensorflowInterface/interface/PythonInterface.h"
#include "PhysicsTools/TensorflowInterface/interface/Tensor.h"

void test(bool success, const std::string& msg)
{
    if (!success)
    {
        throw std::runtime_error(msg);
    }
}

int main(int argc, char* argv[])
{
    std::cout << std::endl << "test tf::Tensor" << std::endl;

    // unlike graphs, tensors do not have their own python interface,
    // so start one here so that numpy works
    tf::PythonInterface p;


    //
    // simple tests
    //

    tf::Tensor* t = new tf::Tensor("myTensor");
    test(t->isEmpty(), "tensor should be empty");
    test(t->getName() == "myTensor", "wrong tensor name");
    t->setName("myTensor2");
    test(t->getName() == "myTensor2", "wrong tensor name");
    test(t->getRank() == -1, "tensor should have no rank");
    test(t->getShape() == 0, "tensor should have no shape");
    test(t->getShape(0) == -1, "tensor should have no shape");
    delete t;


    //
    // value tests
    //

    tf::Shape shape[] = {2, 3, 4};
    t = new tf::Tensor("myTensor", 3, shape);
    test(t->getRank() == 3, "tensor should have rank 3");
    test(t->getShape() != 0, "tensor should have a shape");
    test(t->getShape(0) == 2, "tensor dim 0 should size 2");
    test(t->getShape(1) == 3, "tensor dim 1 should size 3");
    test(t->getShape(2) == 4, "tensor dim 2 should size 4");
    test(t->getValue<float>(0, 0, 0) == 0., "initial value at 0,0,0 should be 0.");
    t->setValue<float>(0, 0, 0, 17.);
    test(*((float*)(t->getPtr(0, 0, 0))) == 17., "pointer value at 0,0,0 should be 17.");
    test(t->getValue<float>(0, 0, 0) == 17., "value at 0,0,0 should be 17.");
    std::vector<float> values = t->getVector<float>(2, 0, 0);
    test(values[0] == 17., "first vector value should be 17.");
    test(values[1] == 0., "second vector value should be 0.");
    test(t->getVector<float>(-1, 0, 0)[0] == 17., "first vector value should be 17.");
    std::vector<float> v = { 1., 2., 3. };
    tf::Shape pos[] = { 0, 0 };
    t->setVectorAtPos<float>(-2, pos, v);
    test(t->getValue<float>(0, 0, 0) == 1., "value at 0,0,0 should be 1.");
    test(t->getValue<float>(0, 2, 0) == 3., "value at 0,2,0 should be 3.");
    t->setVector<float>(1, 1, 3, v);
    test(t->getValue<float>(1, 2, 3) == 3., "value at 1,2,3 should be 3.");
    delete t;

    std::cout << std::endl << "done" << std::endl;

    return 0;
}
