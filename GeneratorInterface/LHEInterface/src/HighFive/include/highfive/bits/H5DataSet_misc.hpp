/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5DATASET_MISC_HPP
#define H5DATASET_MISC_HPP

#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>

#ifdef H5_USE_BOOST
#include <boost/multi_array.hpp>
#endif

#include <H5Dpublic.h>
#include <H5Ppublic.h>

#include "../H5DataSet.hpp"
#include "../H5DataSpace.hpp"
#include "../H5DataType.hpp"

#include "H5Slice_traits_misc.hpp"
#include "H5Utils.hpp"

namespace HighFive {

inline DataSet::DataSet() {}

inline size_t DataSet::getStorageSize() const {
    return H5Dget_storage_size(_hid);
}

inline DataType DataSet::getDataType() const {
    DataType res;
    res._hid = H5Dget_type(_hid);
    return res;
}

inline DataSpace DataSet::getSpace() const {
    DataSpace space;
    if ((space._hid = H5Dget_space(_hid)) < 0) {
        HDF5ErrMapper::ToException<DataSetException>(
            "Unable to get DataSpace out of DataSet");
    }
    return space;
}

inline DataSpace DataSet::getMemSpace() const { return getSpace(); }

inline size_t DataSet::getOffset() const {
    haddr_t addr = H5Dget_offset(_hid);
    if (addr == HADDR_UNDEF) {
        HDF5ErrMapper::ToException<DataSetException>(
            "Cannot get offset of DataSet.");
    }
    return addr;
}

inline void DataSet::resize(const std::vector<size_t>& dims) {

    const size_t numDimensions = getSpace().getDimensions().size();
    if (dims.size() != numDimensions) {
        HDF5ErrMapper::ToException<DataSetException>(
            "Invalid dataspace dimensions, got " + std::to_string(dims.size()) +
            " expected " + std::to_string(numDimensions));
    }

    std::vector<hsize_t> real_dims(dims.begin(), dims.end());

    if (H5Dset_extent(getId(), real_dims.data()) < 0) {
        HDF5ErrMapper::ToException<DataSetException>(
            "Could not resize dataset.");
    }
}

} // namespace HighFive

#endif // H5DATASET_MISC_HPP
