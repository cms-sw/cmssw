/*
 * Test of the TensorFlow tensor interface.
 * Based on TensorFlow C API 1.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include <cppunit/extensions/HelperMacros.h>

#include "PhysicsTools/TensorFlow/interface/Tensor.h"
#include "FWCore/Utilities/interface/Exception.h"

class testTensor : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(testTensor);
    CPPUNIT_TEST(checkAll);
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp()
    {
    }

    void tearDown()
    {
    }

    void checkAll()
    {
        checkEmptyTensor();
        checkScalarTensor();
        checkVectorTensor();
        checkMatrixTensor();
        checkShape3Tensor();
    }

    void checkEmptyTensor();
    void checkScalarTensor();
    void checkVectorTensor();
    void checkMatrixTensor();
    void checkShape3Tensor();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testTensor);

void testTensor::checkEmptyTensor()
{
    tf::Tensor* emptyTensor = new tf::Tensor();

    CPPUNIT_ASSERT(emptyTensor->empty());
    CPPUNIT_ASSERT(emptyTensor->getDataType() == tf::NO_DATATYPE);
    CPPUNIT_ASSERT(emptyTensor->getRank() == -1);
    CPPUNIT_ASSERT(emptyTensor->getAxis(0) == -1);
    CPPUNIT_ASSERT(emptyTensor->getShape(0) == -1);
    CPPUNIT_ASSERT(emptyTensor->getData() == 0);
    CPPUNIT_ASSERT_THROW(emptyTensor->getIndex(0), cms::Exception);
    CPPUNIT_ASSERT_THROW(emptyTensor->getPtrAtPos<float>(0), cms::Exception);

    delete emptyTensor;
}

void testTensor::checkScalarTensor()
{
    tf::Tensor* scalar = new tf::Tensor(0, 0);

    CPPUNIT_ASSERT(!scalar->empty());

    scalar->reset();
    CPPUNIT_ASSERT(scalar->empty());

    scalar->init(0, 0);
    CPPUNIT_ASSERT(!scalar->empty());
    CPPUNIT_ASSERT(scalar->getDataType() == TF_FLOAT);
    CPPUNIT_ASSERT(scalar->getRank() == 0);
    CPPUNIT_ASSERT(scalar->getAxis(0) == -1);
    CPPUNIT_ASSERT(scalar->getShape(0) == 0);
    CPPUNIT_ASSERT(scalar->getData() != 0);
    CPPUNIT_ASSERT(scalar->getIndex(0) == 0);

    float* ptr0 = scalar->getPtr<float>();
    *ptr0 = 123;
    CPPUNIT_ASSERT(*scalar->getPtr<float>() == 123.);

    CPPUNIT_ASSERT_THROW(scalar->getPtr<float>(0), cms::Exception);
    CPPUNIT_ASSERT_THROW(scalar->getPtrVectorAtPos<float>(0, 0), cms::Exception);

    scalar->fillValuesAtPos<float>((float)8., -1, 0);
    CPPUNIT_ASSERT(*scalar->getPtr<float>() == 8.);

    delete scalar;
}

void testTensor::checkVectorTensor()
{
    tf::Shape shape1[1] = { 5 };
    tf::Tensor* vector = new tf::Tensor(1, shape1, TF_DOUBLE);

    CPPUNIT_ASSERT(!vector->empty());
    CPPUNIT_ASSERT(vector->getDataType() == TF_DOUBLE);
    CPPUNIT_ASSERT(vector->getRank() == 1);
    CPPUNIT_ASSERT(vector->getAxis(0) == 0);
    CPPUNIT_ASSERT(vector->getAxis(-1) == 0);
    CPPUNIT_ASSERT(vector->getShape(0) == 5);
    CPPUNIT_ASSERT(vector->getData() != 0);

    tf::Shape pos1[1] = { 3 };
    CPPUNIT_ASSERT(vector->getIndex(pos1) == 3);

    double* ptr1 = vector->getPtr<double>(3);
    *ptr1 = 123;
    CPPUNIT_ASSERT(*vector->getPtr<double>(3) == 123.);

    CPPUNIT_ASSERT_THROW(vector->getPtr<double>(0, 1), cms::Exception);

    std::vector<double*> vec1 = vector->getPtrVector<double>();
    CPPUNIT_ASSERT(vec1.size() == 5);
    CPPUNIT_ASSERT(*vec1[3] == 123.);

    std::vector<double> vec1New = { 5, 6, 7, 8 };
    CPPUNIT_ASSERT_THROW(vector->setVector<double>(vec1New), cms::Exception);

    vec1New.push_back(9);
    vector->setVector<double>(vec1New);
    CPPUNIT_ASSERT(*vec1[3] == 8.);

    vector->fillValues<double>(0, -1, 2);
    CPPUNIT_ASSERT(*vector->getPtr<double>(1) == 6.);
    CPPUNIT_ASSERT(*vector->getPtr<double>(2) == 0.);
    CPPUNIT_ASSERT(*vector->getPtr<double>(4) == 0.);

    delete vector;
}

void testTensor::checkMatrixTensor()
{
    tf::Shape shape2[2] = { 5, 8 };
    tf::Tensor* matrix = new tf::Tensor(2, shape2);

    CPPUNIT_ASSERT(!matrix->empty());
    CPPUNIT_ASSERT(matrix->getDataType() == TF_FLOAT);
    CPPUNIT_ASSERT(matrix->getRank() == 2);
    CPPUNIT_ASSERT(matrix->getAxis(0) == 0);
    CPPUNIT_ASSERT(matrix->getAxis(-1) == 1);
    CPPUNIT_ASSERT(matrix->getShape(-1) == 8);
    CPPUNIT_ASSERT(matrix->getData() != 0);

    tf::Shape pos2[2] = { 3, 4 };
    CPPUNIT_ASSERT(matrix->getIndex(pos2) == 28);

    float* ptr2 = matrix->getPtr<float>(3, 4);
    *ptr2 = 123;
    CPPUNIT_ASSERT(*matrix->getPtr<float>(3, 4) == 123.);

    CPPUNIT_ASSERT_THROW(matrix->getPtr<float>(0), cms::Exception);

    std::vector<float*> vec2 = matrix->getPtrVector<float>(1, 3);
    CPPUNIT_ASSERT(vec2.size() == 8);
    CPPUNIT_ASSERT(*vec2[4] == 123.);

    std::vector<float> vec2New = { 5, 6, 7, 8, 9, 10, 11 };
    CPPUNIT_ASSERT_THROW(matrix->setVector<float>(1, 3, vec2New), cms::Exception);

    vec2New.push_back(12);
    matrix->setVector<float>(1, 3, vec2New);
    CPPUNIT_ASSERT(*vec2[4] == 9.);

    matrix->fillValues<float>(0, -1, 3, 4);
    CPPUNIT_ASSERT(*matrix->getPtr<float>(3, 3) == 8.);
    CPPUNIT_ASSERT(*matrix->getPtr<float>(3, 4) == 0.);
    CPPUNIT_ASSERT(*matrix->getPtr<float>(4, 7) == 0.);

    delete matrix;
}

void testTensor::checkShape3Tensor()
{
    tf::Shape shape3[3] = { 5, 8, 15 };
    tf::Tensor* tensor3 = new tf::Tensor(3, shape3);

    CPPUNIT_ASSERT(!tensor3->empty());
    CPPUNIT_ASSERT(tensor3->getDataType() == TF_FLOAT);
    CPPUNIT_ASSERT(tensor3->getRank() == 3);
    CPPUNIT_ASSERT(tensor3->getAxis(0) == 0);
    CPPUNIT_ASSERT(tensor3->getAxis(-1) == 2);
    CPPUNIT_ASSERT(tensor3->getShape(-1) == 15);
    CPPUNIT_ASSERT(tensor3->getData() != 0);

    tf::Shape pos3[3] = { 3, 4, 6 };
    CPPUNIT_ASSERT(tensor3->getIndex(pos3) == 426);

    float* ptr3 = tensor3->getPtr<float>(3, 4, 10);
    *ptr3 = 123;
    CPPUNIT_ASSERT(*tensor3->getPtr<float>(3, 4, 10) == 123.);

    CPPUNIT_ASSERT_THROW(tensor3->getPtr<float>(0), cms::Exception);

    std::vector<float*> vec3 = tensor3->getPtrVector<float>(2, 3, 4);
    CPPUNIT_ASSERT(vec3.size() == 15);
    CPPUNIT_ASSERT(*vec3[10] == 123.);

    std::vector<float> vec3New = { 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
    CPPUNIT_ASSERT_THROW(tensor3->setVector<float>(2, 3, 4, vec3New), cms::Exception);

    vec3New.push_back(19);
    tensor3->setVector<float>(2, 3, 4, vec3New);
    CPPUNIT_ASSERT(*vec3[10] == 15.);

    tensor3->fillValues<float>(123, -1, 3, 4, 6);
    CPPUNIT_ASSERT(*tensor3->getPtr<float>(3, 4, 5) == 10.);
    CPPUNIT_ASSERT(*tensor3->getPtr<float>(3, 4, 6) == 123.);
    CPPUNIT_ASSERT(*tensor3->getPtr<float>(3, 4, 7) == 123.);
    CPPUNIT_ASSERT(*tensor3->getPtr<float>(3, 4, 8) == 123.);
    CPPUNIT_ASSERT(*tensor3->getPtr<float>(4, 7, 14) == 123.);

    tensor3->fillValues<float>(0, 2, 3, 4, 6);
    CPPUNIT_ASSERT(*tensor3->getPtr<float>(3, 4, 5) == 10.);
    CPPUNIT_ASSERT(*tensor3->getPtr<float>(3, 4, 6) == 0.);
    CPPUNIT_ASSERT(*tensor3->getPtr<float>(3, 4, 7) == 0.);
    CPPUNIT_ASSERT(*tensor3->getPtr<float>(3, 4, 8) == 123.);
    CPPUNIT_ASSERT(*tensor3->getPtr<float>(4, 7, 14) == 123.);

    delete tensor3;
}
