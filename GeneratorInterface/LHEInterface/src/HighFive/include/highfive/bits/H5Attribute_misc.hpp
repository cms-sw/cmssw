/*
 *  Copyright (c), 2017, Ali Can Demiralp <ali.demiralp@rwth-aachen.de>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5ATTRIBUTE_MISC_HPP
#define H5ATTRIBUTE_MISC_HPP

#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>

#ifdef H5_USE_BOOST
#include <boost/multi_array.hpp>
#endif

#include <H5Apublic.h>
#include <H5Ppublic.h>

#include "../H5Attribute.hpp"
#include "../H5DataSpace.hpp"
#include "../H5DataType.hpp"

#include "H5Converter_misc.hpp"
#include "H5Utils.hpp"

namespace HighFive {

inline Attribute::Attribute() {}

inline size_t Attribute::getStorageSize() const {
    return H5Aget_storage_size(_hid);
}

inline DataType Attribute::getDataType() const {
    DataType res;
    res._hid = H5Aget_type(_hid);
    return res;
}

inline DataSpace Attribute::getSpace() const {
    DataSpace space;
    if ((space._hid = H5Aget_space(_hid)) < 0) {
        HDF5ErrMapper::ToException<AttributeException>(
            "Unable to get DataSpace out of Attribute");
    }
    return space;
}

inline DataSpace Attribute::getMemSpace() const { return getSpace(); }

template <typename T>
inline void Attribute::read(T& array) const {
    typedef typename std::remove_const<T>::type type_no_const;

    type_no_const& nocv_array = const_cast<type_no_const&>(array);

    const size_t dim_array = details::array_dims<type_no_const>::value;
    DataSpace space = getSpace();
    DataSpace mem_space = getMemSpace();

    if (!details::checkDimensions(mem_space, dim_array)) {
        std::ostringstream ss;
        ss << "Impossible to read attribute of dimensions "
           << mem_space.getNumberDimensions() << " into arrays of dimensions "
           << dim_array;
        throw DataSpaceException(ss.str());
    }

    // Create mem datatype
    const AtomicType<typename details::type_of_array<type_no_const>::type>
        array_datatype;

    // Apply pre read convertions
    details::data_converter<type_no_const> converter(nocv_array, mem_space);

    if (H5Aread(getId(), array_datatype.getId(),
                static_cast<void*>(converter.transform_read(nocv_array))) < 0) {
        HDF5ErrMapper::ToException<AttributeException>(
            "Error during HDF5 Read: ");
    }

    // re-arrange results
    converter.process_result(nocv_array);
}

template <typename T>
inline void Attribute::write(const T& buffer) {
    typedef typename std::remove_const<T>::type type_no_const;

    type_no_const& nocv_buffer = const_cast<type_no_const&>(buffer);

    const size_t dim_buffer = details::array_dims<type_no_const>::value;
    DataSpace space = getSpace();
    DataSpace mem_space = getMemSpace();

    if (!details::checkDimensions(mem_space, dim_buffer)) {
        std::ostringstream ss;
        ss << "Impossible to write buffer of dimensions " << dim_buffer
           << " into attribute of dimensions "
           << mem_space.getNumberDimensions();
        throw DataSpaceException(ss.str());
    }

    const AtomicType<typename details::type_of_array<type_no_const>::type>
        array_datatype;

    // Apply pre write convertions
    details::data_converter<type_no_const> converter(nocv_buffer, mem_space);

    if (H5Awrite(getId(), array_datatype.getId(),
                 static_cast<const void*>(
                     converter.transform_write(nocv_buffer))) < 0) {
        HDF5ErrMapper::ToException<DataSetException>(
            "Error during HDF5 Write: ");
    }
}
}

#endif // H5ATTRIBUTE_MISC_HPP
