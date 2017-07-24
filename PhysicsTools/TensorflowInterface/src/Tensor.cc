/*
 * Generic Tensorflow tensor representation.
 *
 * Author:
 *   Marcel Rieger
 */

#include "PhysicsTools/TensorflowInterface/interface/Tensor.h"

namespace tf
{

Tensor::Tensor(const std::string& name)
    : name(name)
    , array(0)
{
    init(-1, 0);
}

Tensor::Tensor(int rank, Shape* shape, int typenum)
    : name("")
    , array(0)
{
    init(rank, shape, typenum);
}

Tensor::Tensor(const std::string& name, int rank, Shape* shape, int typenum)
    : name(name)
    , array(0)
{
    init(rank, shape, typenum);
}

Tensor::~Tensor()
{
    if (!isEmpty())
    {
        Py_DECREF(array);
    }
}

void Tensor::init(int rank, Shape* shape, int typenum)
{
    if (PyArray_API == NULL)
    {
        import_array();
    }

    setArray(rank, shape, typenum);
}

std::string Tensor::getName() const
{
    return name;
}

void Tensor::setName(const std::string& name)
{
    this->name = name;
}

bool Tensor::isEmpty() const
{
    return !array;
}

void Tensor::setArray(PyArrayObject* array)
{
    if (!isEmpty())
    {
        Py_XDECREF(this->array);
        this->array = 0;
    }

    this->array = array;
}

void Tensor::setArray(PyObject* array)
{
    setArray((PyArrayObject*)array);
}

void Tensor::setArray(int rank, Shape* shape, int typenum)
{
    if (rank >= 0)
    {
        setArray((PyArrayObject*)PyArray_ZEROS(rank, shape, typenum, 0));
    }
}

int Tensor::getRank() const
{
    return isEmpty() ? -1 : PyArray_NDIM(array);
}

const Shape* Tensor::getShape() const
{
    return isEmpty() ? 0 : PyArray_SHAPE(array);
}

Shape Tensor::getShape(int axis) const
{
    return isEmpty() ? -1 : PyArray_DIM(array, axis);
}

int Tensor::getAxis(int axis) const
{
    const int rank = getRank();
    if (axis < 0)
    {
        axis = rank + axis;
    }
    if (axis >= rank)
    {
        throw std::runtime_error("axis " + std::to_string(axis) + " invalid for rank "
            + std::to_string(rank));
    }
    return axis;
}

void* Tensor::getPtrAtPos(Shape* pos)
{
    return isEmpty() ? 0 : PyArray_GetPtr(array, pos);
}

void* Tensor::getPtr()
{
    return getPtrAtPos(0);
}

void* Tensor::getPtr(Shape i)
{
    Shape pos[1] = {i};
    return getPtrAtPos(pos);
}

void* Tensor::getPtr(Shape i, Shape j)
{
    Shape pos[2] = {i, j};
    return getPtrAtPos(pos);
}

void* Tensor::getPtr(Shape i, Shape j, Shape k)
{
    Shape pos[3] = {i, j, k};
    return getPtrAtPos(pos);
}

void* Tensor::getPtr(Shape i, Shape j, Shape k, Shape l)
{
    Shape pos[4] = {i, j, k, l};
    return getPtrAtPos(pos);
}

void* Tensor::getPtr(Shape i, Shape j, Shape k, Shape l, Shape m)
{
    Shape pos[5] = {i, j, k, l, m};
    return getPtrAtPos(pos);
}

} // namespace tf
