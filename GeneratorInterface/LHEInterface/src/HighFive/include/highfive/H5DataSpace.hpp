/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5DATASPACE_HPP
#define H5DATASPACE_HPP

#include <vector>
#include <cstdint>

#ifdef H5_USE_BOOST
#include <boost/multi_array.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#endif

#include "H5Object.hpp"

namespace HighFive {

class File;
class DataSet;

class DataSpace : public Object {
  public:

    static const size_t UNLIMITED = SIZE_MAX;

    /// dataspace type
    enum DataspaceType {
        datascape_scalar,
        datascape_null
        // simple dataspace are handle directly from their dimensions
    };

    /// create a dataspace of N-dimensions
    /// Each dimension is configured this way
    ///  size(dim1) = vec[0]
    ///  size(dim2) = vec[1]
    ///  etc...
    explicit DataSpace(const std::vector<size_t>& dims);

    /// Create a dataspace from an iterator pair
    template <typename IT>
    DataSpace(const IT begin, const IT end);

    /// \brief Create a resizable N-dimensional dataspace
    /// \params dims Initial size of dataspace
    /// \params maxdims Maximum size of the dataspace
    explicit DataSpace(const std::vector<size_t>& dims,
                       const std::vector<size_t>& maxdims);

    ///
    /// \brief DataSpace create a dataspace of a single dimension and of size
    /// dim1
    /// \param dim1
    ///
    explicit DataSpace(size_t dim1);

    ///
    /// \brief DataSpace create a scalar dataspace or a null dataset
    ///
    explicit DataSpace(DataspaceType dtype);

    /// Create a new DataSpace
    ///  with a different id available for modifications
    DataSpace clone() const;

    ///
    /// \brief getNumberDimensions
    /// \return the number of dimensions in the current dataspace
    ///
    size_t getNumberDimensions() const;

    /// \brief getDimensions
    /// \return return a vector of N-element, each element is the size of the
    /// associated dataset dimension
    std::vector<size_t> getDimensions() const;

    /// \brief getMaxDimensions
    /// \return return a vector of N-element, each element is the size of the
    /// associated dataset maximum dimension
    std::vector<size_t> getMaxDimensions() const;

    /// Create a dataspace matching a single element of a basic type
    ///  supported type are integrals (int,long), floating points (float,double)
    ///  and std::string
    template <typename ScalarValue>
    static DataSpace From(const ScalarValue& scalar_value);

    /// Create a dataspace matching the container dimensions and size
    /// Supported Containers are:
    ///  - vector of fundamental types
    ///  - vector of std::string
    ///  - boost::multi_array
    template <typename Value>
    static DataSpace From(const std::vector<Value>& vec);


#ifdef H5_USE_BOOST
    template <typename Value, std::size_t Dims>
    static DataSpace From(const boost::multi_array<Value, Dims>& container);

    template <typename Value>
    static DataSpace From(const boost::numeric::ublas::matrix<Value>& mat);
#endif

  protected:
    explicit DataSpace();

    friend class Attribute;
    friend class File;
    friend class DataSet;
};
}

#include "bits/H5Dataspace_misc.hpp"

#endif // H5DATASPACE_HPP
