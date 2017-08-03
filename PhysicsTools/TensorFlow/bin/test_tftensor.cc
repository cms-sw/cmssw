/*
 * Test of the TensorFlow tensor interface.
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

#include "PhysicsTools/TensorFlow/interface/Tensor.h"

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

int main(int argc, char* argv[])
{
    std::cout << std::endl << "test tf::Tensor" << std::endl;

    bool catched;

    //
    // test static functions
    //

    tf::Shape shape[3] = { 2, 3, 4 };
    test(tf::Tensor::getTensorSize(3, shape, TF_FLOAT) == 2 * 3 * 4 * sizeof(float),
        "size should be 96");


    //
    // test empty tensor
    //

    tf::Tensor* emptyTensor = new tf::Tensor();

    test(emptyTensor->empty(), "empty tensor should be empty");

    test(emptyTensor->getDataType() == tf::NO_DATATYPE, "empty tensor should have no datatype");

    test(emptyTensor->getRank() == -1, "empty tensor should have no rank");

    test(emptyTensor->getAxis(0) == -1, "empty tensor should not support axis lookup");

    test(emptyTensor->getShape(0) == -1, "empty tensor should have no shape");

    test(emptyTensor->getData() == 0, "empty tensor should have no data");

    catched = false;
    try { emptyTensor->getIndex(0); }
    catch (...) { catched = true; }
    test(catched, "empty tensor should not support indexing");

    catched = false;
    try { emptyTensor->getPtrAtPos<float>(0); }
    catch (...) { catched = true; }
    test(catched, "empty tensor should not support pointer lookup");

    delete emptyTensor;


    //
    // test tensor tensor
    //

    tf::Tensor* scalar = new tf::Tensor(0, 0);

    test(!scalar->empty(), "scalar should not be empty");

    scalar->reset();
    test(scalar->empty(), "scalar should be empty");

    scalar->init(0, 0);
    test(!scalar->empty(), "scalar should not be empty");

    test(scalar->getDataType() == TF_FLOAT, "scalar should have float datatype");

    test(scalar->getRank() == 0, "scalar should have rank 0");

    test(scalar->getAxis(0) == -1, "scalar should not support axis lookup");

    test(scalar->getShape(0) == 0, "scalar should have no shape");

    test(scalar->getData() != 0, "scalar should have data");

    test(scalar->getIndex(0) == 0, "scalar index should be 0");

    float* ptr0 = scalar->getPtr<float>();
    *ptr0 = 123;
    test(*scalar->getPtr<float>() == 123., "scalar value should be 123");

    catched = false;
    try { scalar->getPtr<float>(0); }
    catch (...) { catched = true; }
    test(catched, "scalar should not support pointer lookup with different rank");

    catched = false;
    try { scalar->getPtrVectorAtPos<float>(0, 0); }
    catch (...) { catched = true; }
    test(catched, "scalar should not support pointer vector lookup");

    delete scalar;


    //
    // test vector tensor
    //

    tf::Shape shape1[1] = { 5 };
    tf::Tensor* vector = new tf::Tensor(1, shape1, TF_DOUBLE);

    test(!vector->empty(), "vector should not be empty");

    test(vector->getDataType() == TF_DOUBLE, "vector should have double datatype");

    test(vector->getRank() == 1, "vector should have rank 1");

    test(vector->getAxis(0) == 0, "vector axis should be 0");

    test(vector->getAxis(-1) == 0, "vector axis -1 should wrap to 0");

    test(vector->getShape(0) == 5, "vector axis 0 should have size 5");

    test(vector->getData() != 0, "vector should have data");

    tf::Shape pos1[1] = { 3 };
    test(vector->getIndex(pos1) == 3, "vector index should be 3");

    double* ptr1 = vector->getPtr<double>(3);
    *ptr1 = 123;
    test(*vector->getPtr<double>(3) == 123., "vector value 3 should be 123");

    catched = false;
    try { vector->getPtr<double>(0, 1); }
    catch (...) { catched = true; }
    test(catched, "vector should not support pointer lookup with different rank");

    std::vector<double*> vec1 = vector->getPtrVector<double>();
    test(vec1.size() == 5, "vector pointer vector should have size 5");

    test(*vec1[3] == 123., "vector pointer vector element 3 should be 123");

    std::vector<double> vec1New = { 5, 6, 7, 8 };
    catched = false;
    try { vector->setVector<double>(vec1New); }
    catch (...) { catched = true; }
    test(catched, "vector assignment should fail with wrong size");

    vec1New.push_back(9);
    vector->setVector<double>(vec1New);
    test(*vec1[3] == 8., "vector pointer vector element 3 should be 8");

    delete vector;


    //
    // test matrix tensor
    //

    tf::Shape shape2[2] = { 5, 8 };
    tf::Tensor* matrix = new tf::Tensor(2, shape2);

    test(!matrix->empty(), "matrix should not be empty");

    test(matrix->getDataType() == TF_FLOAT, "matrix should have float datatype");

    test(matrix->getRank() == 2, "matrix should have rank 2");

    test(matrix->getAxis(0) == 0, "matrix axis should be 0");

    test(matrix->getAxis(-1) == 1, "matrix axis -1 should wrap to 1");

    test(matrix->getShape(-1) == 8, "matrix axis -1 should have size 8");

    test(matrix->getData() != 0, "matrix should have data");

    tf::Shape pos2[2] = { 3, 4 };
    test(matrix->getIndex(pos2) == 28, "matrix index should be 28");

    float* ptr2 = matrix->getPtr<float>(3, 4);
    *ptr2 = 123;
    test(*matrix->getPtr<float>(3, 4) == 123., "matrix value 3,4 should be 123");

    catched = false;
    try { matrix->getPtr<float>(0); }
    catch (...) { catched = true; }
    test(catched, "matrix should not support pointer lookup with different rank");

    std::vector<float*> vec2 = matrix->getPtrVector<float>(1, 3);
    test(vec2.size() == 8, "matrix pointer vector of axis 1 should have size 8");

    test(*vec2[4] == 123., "matrix pointer vector element 3,4 should be 123");

    std::vector<float> vec2New = { 5, 6, 7, 8, 9, 10, 11 };
    catched = false;
    try { matrix->setVector<float>(1, 3, vec2New); }
    catch (...) { catched = true; }
    test(catched, "matrix vector assignment should fail with wrong size");

    vec2New.push_back(12);
    matrix->setVector<float>(1, 3, vec2New);
    test(*vec2[4] == 9., "matrix pointer vector element 3,4 should be 9");

    delete matrix;


    //
    // test rank 3 tensor
    //

    tf::Shape shape3[3] = { 5, 8, 15 };
    tf::Tensor* tensor3 = new tf::Tensor(3, shape3);

    test(!tensor3->empty(), "tensor3 should not be empty");

    test(tensor3->getDataType() == TF_FLOAT, "tensor3 should have float datatype");

    test(tensor3->getRank() == 3, "tensor3 should have rank 3");

    test(tensor3->getAxis(0) == 0, "tensor3 axis should be 0");

    test(tensor3->getAxis(-1) == 2, "tensor3 axis -1 should wrap to 2");

    test(tensor3->getShape(-1) == 15, "tensor3 axis -1 should have size 15");

    test(tensor3->getData() != 0, "tensor3 should have data");

    tf::Shape pos3[3] = { 3, 4, 6 };
    test(tensor3->getIndex(pos3) == 426, "tensor3 index should be 426");

    float* ptr3 = tensor3->getPtr<float>(3, 4, 10);
    *ptr3 = 123;
    test(*tensor3->getPtr<float>(3, 4, 10) == 123., "tensor3 value 3,4,10 should be 123");

    catched = false;
    try { tensor3->getPtr<float>(0); }
    catch (...) { catched = true; }
    test(catched, "tensor3 should not support pointer lookup with different rank");

    std::vector<float*> vec3 = tensor3->getPtrVector<float>(2, 3, 4);
    test(vec3.size() == 15, "tensor3 pointer vector of axis 2 should have size 15");

    test(*vec3[10] == 123., "tensor3 pointer vector element 3,4,10 should be 123");

    std::vector<float> vec3New = { 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
    catched = false;
    try { tensor3->setVector<float>(2, 3, 4, vec3New); }
    catch (...) { catched = true; }
    test(catched, "tensor3 vector assignment should fail with wrong size");

    vec3New.push_back(19);
    tensor3->setVector<float>(2, 3, 4, vec3New);
    test(*vec3[10] == 15., "tensor3 pointer vector element 3,4,10 should be 15");

    delete tensor3;


    std::cout << std::endl << "done" << std::endl;

    return 0;
}
