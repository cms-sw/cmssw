/*
 * TensorFlow tensor interface.
 * Based on TensorFlow C API 1.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#ifndef PHYSICSTOOLS_TENSORFLOW_TENSOR_H
#define PHYSICSTOOLS_TENSORFLOW_TENSOR_H

#include <vector>

#include "tensorflow/c/c_api.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace tf
{

// typedefs
typedef TF_DataType DataType;
typedef int64_t Shape;

// constants
const DataType NO_DATATYPE = TF_RESOURCE;

// the Tensor class
class Tensor
{
public:
    // default constructor
    Tensor();

    // disable implicit copy constructor
    Tensor(const Tensor& t) = delete;

    // constructor that initializes the internal tensorflow tensor object
    Tensor(int rank, Shape* shape, DataType dtype = TF_FLOAT);

    // destructor
    ~Tensor();

    // static function to return the memory size of a tensor defined by its rank, shape and datatype
    static size_t getTensorSize(int rank, Shape* shape, DataType dtype);

    // inializes the internal tensorflow tensor object with an existing one, takes ownership
    void init(TF_Tensor* t);

    // inializes the internal tensorflow tensor object by creating a new one
    inline void init(int rank, Shape* shape, DataType dtype = TF_FLOAT)
    {
        init(TF_AllocateTensor(dtype, shape, rank, getTensorSize(rank, shape, dtype)));
    }

    // resets the internal tensorflow tensor object
    void reset();

    // returns the pointer to the tensorflow tensor object
    inline TF_Tensor* getTFTensor()
    {
        return tf_tensor_;
    }

    // returns true if the internal tensorflow tensor object is not initalized yet, false otherwise
    inline bool empty() const
    {
        return tf_tensor_ == nullptr;
    }

    // returns the datatype or NO_DATATYPE when empty
    inline DataType getDataType() const
    {
        return empty() ? NO_DATATYPE : TF_TensorType(tf_tensor_);
    }

    // returns the tensor tank or -1 when empty
    inline int getRank() const
    {
        return empty() ? -1 : TF_NumDims(tf_tensor_);
    }

    // performs sanity checks and returns a positive axis number for any (also negative) axis
    int getAxis(int axis) const;

    // returns the shape of an axis or -1 when empty
    inline Shape getShape(int axis) const
    {
        return empty() ? -1 : TF_Dim(tf_tensor_, getAxis(axis));
    }

    // returns the index in a one dimensional array given a coordinate with multi-dimensional shape
    Shape getIndex(Shape* pos) const;

    // returns a pointer to the data object or nullptr when empty
    inline void* getData()
    {
        return empty() ? nullptr : TF_TensorData(tf_tensor_);
    }

    // returns the pointer to the data at an arbitrary position
    template <typename T>
    inline T* getPtrAtPos(Shape* pos)
    {
        T* ptr = static_cast<T*>(getData());
        ptr += getIndex(pos);
        return ptr;
    }

    // returns the pointer to a scalar
    template <typename T>
    inline T* getPtr()
    {
        assertRank(0);
        return static_cast<T*>(getData());
    }

    // returns the pointer to an element in a rank 1 tensor
    template <typename T>
    inline T* getPtr(Shape i)
    {
        assertRank(1);
        Shape pos[1] = { i };
        return getPtrAtPos<T>(pos);
    }

    // returns the pointer to an element in a rank 2 tensor
    template <typename T>
    inline T* getPtr(Shape i, Shape j)
    {
        assertRank(2);
        Shape pos[2] = { i, j };
        return getPtrAtPos<T>(pos);
    }

    // returns the pointer to an element in a rank 3 tensor
    template <typename T>
    inline T* getPtr(Shape i, Shape j, Shape k)
    {
        assertRank(3);
        Shape pos[3] = { i, j, k };
        return getPtrAtPos<T>(pos);
    }

    // returns the pointer to an element in a rank 4 tensor
    template <typename T>
    inline T* getPtr(Shape i, Shape j, Shape k, Shape l)
    {
        assertRank(4);
        Shape pos[4] = { i, j, k, l };
        return getPtrAtPos<T>(pos);
    }

    // determines the value pointers of a tensor along an axis for a fixed position on the other
    // axes
    template <typename T>
    void getPtrVectorAtPos(int axis, Shape* pos, std::vector<T*>& v);

    // returns the value pointers of a tensor along an axis for a fixed position on the other axes
    template <typename T>
    inline std::vector<T*> getPtrVectorAtPos(int axis, Shape* pos)
    {
        std::vector<T*> v;
        getPtrVectorAtPos(axis, pos, v);
        return v;
    }

    // determines the value pointers of a rank 1 tensor
    template <typename T>
    inline void getPtrVector(std::vector<T*>& v)
    {
        assertRank(1);
        getPtrVectorAtPos<T>(0, 0, v);
    }

    // returns the value pointers of a rank 1 tensor
    template <typename T>
    inline std::vector<T*> getPtrVector()
    {
        std::vector<T*> v;
        getPtrVector(v);
        return v;
    }

    // determines the value pointers of a rank 2 tensor along an axis for a fixed position of the
    // other axis
    template <typename T>
    inline void getPtrVector(int axis, Shape a, std::vector<T*>& v)
    {
        assertRank(2);
        Shape pos[1] = { a };
        getPtrVectorAtPos<T>(axis, pos, v);
    }

    // returns the value pointers of a rank 2 tensor along an axis for a fixed position of the other
    // axis
    template <typename T>
    inline std::vector<T*> getPtrVector(int axis, Shape a)
    {
        std::vector<T*> v;
        getPtrVector<T>(axis, a, v);
        return v;
    }

    // determines the value pointers of a rank 3 tensor along an axis for a fixed position of the
    // other axes
    template <typename T>
    inline void getPtrVector(int axis, Shape a, Shape b, std::vector<T*>& v)
    {
        assertRank(3);
        Shape pos[2] = { a, b };
        getPtrVectorAtPos<T>(axis, pos, v);
    }

    // returns the value pointers of a rank 3 tensor along an axis for a fixed position of the other
    // axes
    template <typename T>
    inline std::vector<T*> getPtrVector(int axis, Shape a, Shape b)
    {
        std::vector<T*> v;
        getPtrVector<T>(axis, a, b, v);
        return v;
    }

    // determines the value pointers of a rank 4 tensor along an axis for a fixed position of the
    // other axes
    template <typename T>
    inline void getPtrVector(int axis, Shape a, Shape b, Shape c, std::vector<T*>& v)
    {
        assertRank(4);
        Shape pos[3] = { a, b, c };
        getPtrVectorAtPos<T>(axis, pos, v);
    }

    // returns the value pointers of a rank 4 tensor along an axis for a fixed position of the other
    // axes
    template <typename T>
    inline std::vector<T*> getPtrVector(int axis, Shape a, Shape b, Shape c)
    {
        std::vector<T*> v;
        getPtrVector<T>(axis, a, b, c, v);
        return v;
    }

    // sets the values of a tensor along an axis for a fixed position on the other axes
    template <typename T>
    void setVectorAtPos(int axis, Shape* pos, std::vector<T>& v);

    // sets the values of a rank 1 tensor
    template <typename T>
    inline void setVector(std::vector<T>& v)
    {
        assertRank(1);
        setVectorAtPos<T>(0, 0, v);
    }

    // sets the values of a rank 2 tensor along an axis for a fixed position on the other axis
    template <typename T>
    inline void setVector(int axis, Shape a, std::vector<T>& v)
    {
        assertRank(2);
        Shape pos[1] = { a };
        setVectorAtPos<T>(axis, pos, v);
    }

    // sets the values of a rank 3 tensor along an axis for a fixed position on the other axes
    template <typename T>
    inline void setVector(int axis, Shape a, Shape b, std::vector<T>& v)
    {
        assertRank(3);
        Shape pos[2] = { a, b };
        setVectorAtPos<T>(axis, pos, v);
    }

    // sets the values of a rank 4 tensor along an axis for a fixed position on the other axes
    template <typename T>
    inline void setVector(int axis, Shape a, Shape b, Shape c, std::vector<T>& v)
    {
        assertRank(4);
        Shape pos[3] = { a, b, c };
        setVectorAtPos<T>(axis, pos, v);
    }

    // sets n elements to a constant value starting from a certain position, negative n means all
    template <typename T>
    void fillValuesAtPos(T v, int n, Shape* pos);

    // sets n elements of a rank 1 tensor to a constant value starting from a certain position,
    // negative n means all
    template <typename T>
    inline void fillValues(T v, int n, Shape i)
    {
        assertRank(1);
        Shape pos[1] = { i };
        fillValuesAtPos<T>(v, n, pos);
    }

    // sets n elements of a rank 2 tensor to a constant value starting from a certain position,
    // negative n means all
    template <typename T>
    inline void fillValues(T v, int n, Shape i, Shape j)
    {
        assertRank(2);
        Shape pos[2] = { i, j };
        fillValuesAtPos<T>(v, n, pos);
    }

    // sets n elements of a rank 3 tensor to a constant value starting from a certain position,
    // negative n means all
    template <typename T>
    inline void fillValues(T v, int n, Shape i, Shape j, Shape k)
    {
        assertRank(3);
        Shape pos[3] = { i, j, k };
        fillValuesAtPos<T>(v, n, pos);
    }

    // sets n elements of a rank 4 tensor to a constant value starting from a certain position,
    // negative n means all
    template <typename T>
    inline void fillValues(T v, int n, Shape i, Shape j, Shape k, Shape l)
    {
        assertRank(4);
        Shape pos[4] = { i, j, k, l };
        fillValuesAtPos<T>(v, n, pos);
    }

private:
    // the internal tensorflow tensor object
    TF_Tensor* tf_tensor_;

    // array of cumulative shape products to accelerate indexing
    int64_t* prod_;

    // compares a passed rank to the current rank and throws an exception when they do not match
    inline void assertRank(int rank) const
    {
        if (getRank() != rank)
        {
            throw cms::Exception("InvalidRank") << "invalid rank to perform operation: "
                << getRank() << " (expected " << rank << ")";
        }
    }
};

template <typename T>
void Tensor::getPtrVectorAtPos(int axis, Shape* pos, std::vector<T*>& v)
{
    const int rank = getRank();

    // special treatment of scalars
    if (rank == 0)
    {
        throw cms::Exception("InvalidRank") << "vectors cannot be extracted from scalars";
    }

    axis = getAxis(axis);

    // create a position array that is altered on the requested axis when looping
    Shape pos2[rank];
    for (int i = 0; i < rank; i++)
    {
        if (i < axis)
        {
            pos2[i] = pos[i];
        }
        else if (i == axis)
        {
            pos2[i] = -1;
        }
        else
        {
            pos2[i] = pos[i - 1];
        }
    }

    // start looping
    v.clear();
    for (Shape i = 0, s = getShape(axis); i < s; i++)
    {
        // alter the position array for the request axis
        pos2[axis] = i;

        // simply collect the pointers
        v.push_back(getPtrAtPos<T>(pos2));
    }
}

template <typename T>
void Tensor::setVectorAtPos(int axis, Shape* pos, std::vector<T>& v)
{
    // special treatment of scalars
    if (getRank() == 0)
    {
        throw cms::Exception("InvalidRank") << "vectors cannot be inserted into scalars";
    }

    // get the pointer vector
    std::vector<T*> ptrs;
    getPtrVectorAtPos<T>(axis, pos, ptrs);

    // sanity check
    const int len = getShape(axis);
    if ((int64_t)v.size() != len)
    {
        throw cms::Exception("InvalidDimension") << "invalid vector size: " << v.size()
            << " (should be " << len << ")";
    }

    // assign the passed values
    for (int i = 0; i < len; i++)
    {
        *ptrs[i] = v[i];
    }
}

template <typename T>
void Tensor::fillValuesAtPos(T v, int n, Shape* pos)
{
    // special treatment of scalars
    if (getRank() == 0)
    {
        *getPtr<T>() = v;
        return;
    }

    // get the maximum number of elements to fill
    int nElements = getShape(0) * prod_[0] - getIndex(pos);

    // limit by n
    if (n >= 0 && n < nElements)
    {
        nElements = n;
    }

    // set the values
    // here we exploit that the values we want to set are stored contiguously in the memory, so it
    // is most performant to get the first pointer and use pointer arithmetic to iterate
    if (nElements > 0)
    {
        T* ptr = getPtrAtPos<T>(pos);
        for (int i = 0; i < nElements; i++, ptr++)
        {
            *ptr = v;
        }
    }
}

} // namepace tf

#endif // PHYSICSTOOLS_TENSORFLOW_TENSOR_H
