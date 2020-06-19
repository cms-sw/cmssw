/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5SLICE_TRAITS_HPP
#define H5SLICE_TRAITS_HPP

#include <cstdlib>
#include <vector>

#include "H5Utils.hpp"

namespace HighFive {

class DataSet;
class Group;
class DataSpace;
class DataType;
class Selection;

template <typename Derivate>
class SliceTraits;

class ElementSet {
  public:
    explicit ElementSet(const std::vector<std::size_t>& element_ids);

  private:
    std::vector<std::size_t> _ids;

    template <typename Derivate>
    friend class SliceTraits;
};

template <typename Derivate>
class SliceTraits {
  public:
    ///
    /// select a region in the current Slice/Dataset of 'count' points at
    /// 'offset' separated by 'stride'. If strides are not provided they will
    /// default to 1 in all dimensions.
    /// vector offset and count have to be from the same dimension
    ///
    Selection select(const std::vector<size_t>& offset,
                     const std::vector<size_t>& count,
                     const std::vector<size_t>& stride = std::vector<size_t>())
        const;

    ///
    /// select a set of columns in the last dimension of this dataset.
    /// The column indices must be smaller than the dimension size.
    ///
    Selection select(const std::vector<size_t>& columns) const;

    ///
    /// select a region in the current Slice/Dataset out of a list of elements
    ///
    Selection select(const ElementSet& elements) const;

    ///
    /// Read the entire dataset into a buffer
    /// An exception is raised is if the numbers of dimension of the buffer and
    /// of the dataset are different
    ///
    /// The array type can be a N-pointer or a N-vector. For plain pointers
    /// not dimensionality checking will be performed, it is the user's
    /// reponsibility to ensure that the right amount of space has been
    /// allocated.
    template <typename T>
    void read(T& array) const;

    ///
    /// Read the entire dataset into a raw buffer
    ///
    /// No dimensionality checks will be performed, it is the user's
    /// reponsibility to ensure that the right amount of space has been
    /// allocated.
    template <typename T>
    void read(T* array) const;

    ///
    /// Write the integrality N-dimension buffer to this dataset
    /// An exception is raised is if the numbers of dimension of the buffer and
    /// of the dataset are different
    ///
    /// The array type can be a N-pointer or a N-vector ( e.g int** integer two
    /// dimensional array )
    template <typename T>
    void write(const T& buffer);

    ///
    /// Write from a raw buffer into this dataset
    ///
    /// No dimensionality checks will be performed, it is the user's
    /// reponsibility to ensure that the buffer holds the right amount of
    /// elements. For n-dimensional matrices the buffer layout follows H5
    /// default conventions.
    template <typename T>
    void write(const T* buffer);

  private:
    typedef Derivate derivate_type;
};
}

#endif // H5SLICE_TRAITS_HPP
