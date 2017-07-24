/*
 * Generic Tensorflow tensor representation.
 *
 * Author:
 *   Marcel Rieger
 */

#ifndef PHYSICSTOOLS_TENSORFLOWINTERFACE_TENSOR_H
#define PHYSICSTOOLS_TENSORFLOWINTERFACE_TENSOR_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Python.h"
#include "numpy/arrayobject.h"

namespace tf
{

typedef npy_intp Shape;

class Tensor
{
public:
    Tensor(const std::string& name);
    Tensor(int rank, Shape* shape, int typenum = NPY_FLOAT);
    Tensor(const std::string& name, int rank, Shape* shape, int typenum = NPY_FLOAT);
    virtual ~Tensor();

    std::string getName() const;
    void setName(const std::string& name);

    bool isEmpty() const;

    inline PyArrayObject* getArray()
    {
        return array;
    }
    void setArray(PyArrayObject* array); // steals reference
    void setArray(PyObject* array); // steals reference
    void setArray(int rank, Shape* shape, int typenum = NPY_FLOAT);

    int getRank() const;
    const Shape* getShape() const;
    Shape getShape(int axis) const;

    int getAxis(int axis) const;

    void* getPtrAtPos(Shape* pos);
    void* getPtr();
    void* getPtr(Shape i);
    void* getPtr(Shape i, Shape j);
    void* getPtr(Shape i, Shape j, Shape k);
    void* getPtr(Shape i, Shape j, Shape k, Shape l);
    void* getPtr(Shape i, Shape j, Shape k, Shape l, Shape m);

    template <typename T>
    T getValueAtPos(Shape* pos);
    template <typename T>
    T getValue();
    template <typename T>
    T getValue(Shape i);
    template <typename T>
    T getValue(Shape i, Shape j);
    template <typename T>
    T getValue(Shape i, Shape j, Shape k);
    template <typename T>
    T getValue(Shape i, Shape j, Shape k, Shape l);
    template <typename T>
    T getValue(Shape i, Shape j, Shape k, Shape l, Shape m);

    template <typename T>
    void setValueAtPos(Shape* pos, T value);
    template <typename T>
    void setValue(T value);
    template <typename T>
    void setValue(Shape i, T value);
    template <typename T>
    void setValue(Shape i, Shape j, T value);
    template <typename T>
    void setValue(Shape i, Shape j, Shape k, T value);
    template <typename T>
    void setValue(Shape i, Shape j, Shape k, Shape l, T value);
    template <typename T>
    void setValue(Shape i, Shape j, Shape k, Shape l, Shape m, T value);

    template <typename T>
    std::vector<T> getVectorAtPos(int axis, Shape* pos);
    template <typename T>
    std::vector<T> getVector(); // rank 1
    template <typename T>
    std::vector<T> getVector(int axis, Shape a); // rank 2
    template <typename T>
    std::vector<T> getVector(int axis, Shape a, Shape b); // rank 3
    template <typename T>
    std::vector<T> getVector(int axis, Shape a, Shape b, Shape c); // rank 4
    template <typename T>
    std::vector<T> getVector(int axis, Shape a, Shape b, Shape c, Shape d); // rank 5

    template <typename T>
    void setVectorAtPos(int axis, Shape* pos, std::vector<T> v);
    template <typename T>
    void setVector(std::vector<T> v); // rank 1
    template <typename T>
    void setVector(int axis, Shape a, std::vector<T> v); // rank 2
    template <typename T>
    void setVector(int axis, Shape a, Shape b, std::vector<T> v); // rank 3
    template <typename T>
    void setVector(int axis, Shape a, Shape b, Shape c, std::vector<T> v); // rank 4
    template <typename T>
    void setVector(int axis, Shape a, Shape b, Shape c, Shape d, std::vector<T> v); // rank 5

private:
    void init(int rank, Shape* shape, int typenum = NPY_FLOAT);

    std::string name;
    PyArrayObject* array;
};

template <typename T>
T Tensor::getValueAtPos(Shape* pos)
{
    return *((T*)(getPtrAtPos(pos)));
}

template <typename T>
T Tensor::getValue()
{
    return *((T*)(getPtrAtPos(0)));
}

template <typename T>
T Tensor::getValue(Shape i)
{
    return *((T*)(getPtr(i)));
}

template <typename T>
T Tensor::getValue(Shape i, Shape j)
{
    return *((T*)(getPtr(i, j)));
}

template <typename T>
T Tensor::getValue(Shape i, Shape j, Shape k)
{
    return *((T*)(getPtr(i, j, k)));
}

template <typename T>
T Tensor::getValue(Shape i, Shape j, Shape k, Shape l)
{
    return *((T*)(getPtr(i, j, k, l)));
}

template <typename T>
T Tensor::getValue(Shape i, Shape j, Shape k, Shape l, Shape m)
{
    return *((T*)(getPtr(i, j, k, l, m)));
}

template <typename T>
void Tensor::setValueAtPos(Shape* pos, T value)
{
    *((T*)(getPtrAtPos(pos))) = value;
}

template <typename T>
void Tensor::setValue(T value)
{
    *((T*)(getPtr())) = value;
}

template <typename T>
void Tensor::setValue(Shape i, T value)
{
    *((T*)(getPtr(i))) = value;
}

template <typename T>
void Tensor::setValue(Shape i, Shape j, T value)
{
    *((T*)(getPtr(i, j))) = value;
}

template <typename T>
void Tensor::setValue(Shape i, Shape j, Shape k, T value)
{
    *((T*)(getPtr(i, j, k))) = value;
}

template <typename T>
void Tensor::setValue(Shape i, Shape j, Shape k, Shape l, T value)
{
    *((T*)(getPtr(i, j, k, l))) = value;
}

template <typename T>
void Tensor::setValue(Shape i, Shape j, Shape k, Shape l, Shape m, T value)
{
    *((T*)(getPtr(i, j, k, l, m))) = value;
}

template <typename T>
std::vector<T> Tensor::getVectorAtPos(int axis, Shape* pos)
{
    axis = getAxis(axis);
    const int rank = getRank();

    Shape pos2[rank];
    for (int i = 0; i < rank; i++)
    {
        if (i < axis)
        {
            pos2[i] = pos[i];
        }
        else if (i == axis)
        {
            pos2[i] = axis;
        }
        else
        {
            pos2[i] = pos[i - 1];
        }
    }

    std::vector<T> v;
    for (Shape i = 0; i < getShape(axis); i++)
    {
        pos2[axis] = i;
        v.push_back(*((T*)(getPtrAtPos(pos2))));
    }
    return v;
}

template <typename T>
std::vector<T> Tensor::getVector()
{
    return getVectorAtPos<T>(0, 0);
}

template <typename T>
std::vector<T> Tensor::getVector(int axis, Shape a)
{
    Shape pos[1] = { a };
    return getVectorAtPos<T>(axis, pos);
}

template <typename T>
std::vector<T> Tensor::getVector(int axis, Shape a, Shape b)
{
    Shape pos[2] = { a, b };
    return getVectorAtPos<T>(axis, pos);
}

template <typename T>
std::vector<T> Tensor::getVector(int axis, Shape a, Shape b, Shape c)
{
    Shape pos[3] = { a, b, c };
    return getVectorAtPos<T>(axis, pos);
}

template <typename T>
std::vector<T> Tensor::getVector(int axis, Shape a, Shape b, Shape c, Shape d)
{
    Shape pos[4] = { a, b, c, d };
    return getVectorAtPos<T>(axis, pos);
}

template <typename T>
void Tensor::setVectorAtPos(int axis, Shape* pos, std::vector<T> v)
{
    axis = getAxis(axis);
    const int rank = getRank();

    Shape pos2[rank];
    for (int i = 0; i < rank; i++)
    {
        if (i < axis)
        {
            pos2[i] = pos[i];
        }
        else if (i == axis)
        {
            pos2[i] = axis;
        }
        else
        {
            pos2[i] = pos[i - 1];
        }
    }

    if (getShape(axis) != (Shape)v.size())
    {
        throw std::runtime_error("invalid vector size of " + std::to_string(v.size()) + " for "
            "axis shape " + std::to_string(getShape(axis)));
    }

    for (Shape i = 0; i < getShape(axis); i++)
    {
        pos2[axis] = i;
        setValueAtPos<T>(pos2, v[i]);
    }
}

template <typename T>
void Tensor::setVector(std::vector<T> v)
{
    setVectorAtPos<T>(0, 0, v);
}

template <typename T>
void Tensor::setVector(int axis, Shape a, std::vector<T> v)
{
    Shape pos[1] = { a };
    setVectorAtPos<T>(axis, pos, v);
}

template <typename T>
void Tensor::setVector(int axis, Shape a, Shape b, std::vector<T> v)
{
    Shape pos[2] = { a, b };
    setVectorAtPos<T>(axis, pos, v);
}

template <typename T>
void Tensor::setVector(int axis, Shape a, Shape b, Shape c, std::vector<T> v)
{
    Shape pos[3] = { a, b, c };
    setVectorAtPos<T>(axis, pos, v);
}

template <typename T>
void Tensor::setVector(int axis, Shape a, Shape b, Shape c, Shape d, std::vector<T> v)
{
    Shape pos[4] = { a, b, c, d };
    setVectorAtPos<T>(axis, pos, v);
}

} // namepace tf

#endif // PHYSICSTOOLS_TENSORFLOWINTERFACE_TENSOR_H
