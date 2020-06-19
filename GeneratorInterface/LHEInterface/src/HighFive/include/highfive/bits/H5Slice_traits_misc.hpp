/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5SLICE_TRAITS_MISC_HPP
#define H5SLICE_TRAITS_MISC_HPP

#include "H5Slice_traits.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>

#ifdef H5_USE_BOOST
#include <boost/multi_array.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#endif

#include <H5Dpublic.h>
#include <H5Ppublic.h>

#include "../H5DataSpace.hpp"
#include "../H5DataType.hpp"
#include "../H5Selection.hpp"

#include "H5Converter_misc.hpp"

namespace HighFive {

namespace details {

// map the correct reference to the dataset depending of the layout
// dataset -> itself
// subselection -> parent dataset
inline const DataSet& get_dataset(const Selection* ptr) {
    return ptr->getDataset();
}

inline const DataSet& get_dataset(const DataSet* ptr) { return *ptr; }

// map the correct memspace identifier depending of the layout
// dataset -> entire memspace
// selection -> resolve space id
inline hid_t get_memspace_id(const Selection* ptr) {
    return ptr->getMemSpace().getId();
}

inline hid_t get_memspace_id(const DataSet* ptr) {
    (void)ptr;
    return H5S_ALL;
}
}

inline ElementSet::ElementSet(const std::vector<std::size_t>& element_ids)
    : _ids(element_ids) {}

template <typename Derivate>
inline Selection
SliceTraits<Derivate>::select(const std::vector<size_t>& offset,
                              const std::vector<size_t>& count,
                              const std::vector<size_t>& stride) const {
    // hsize_t type convertion
    // TODO : normalize hsize_t type in HighFive namespace
    std::vector<hsize_t> offset_local(offset.size());
    std::vector<hsize_t> count_local(count.size());
    std::vector<hsize_t> stride_local(stride.size());
    std::copy(offset.begin(), offset.end(), offset_local.begin());
    std::copy(count.begin(), count.end(), count_local.begin());
    std::copy(stride.begin(), stride.end(), stride_local.begin());

    DataSpace space = static_cast<const Derivate*>(this)->getSpace().clone();
    if (H5Sselect_hyperslab(space.getId(), H5S_SELECT_SET, offset_local.data(),
                            stride.empty() ? NULL : stride_local.data(),
                            count_local.data(), NULL) < 0) {
        HDF5ErrMapper::ToException<DataSpaceException>(
            "Unable to select hyperslap");
    }

    return Selection(DataSpace(count), space,
                     details::get_dataset(static_cast<const Derivate*>(this)));
}

template <typename Derivate>
inline Selection
SliceTraits<Derivate>::select(const std::vector<size_t>& columns) const {

    const DataSpace& space = static_cast<const Derivate*>(this)->getSpace();
    const DataSet& dataset =
        details::get_dataset(static_cast<const Derivate*>(this));
    std::vector<size_t> dims = space.getDimensions();
    std::vector<hsize_t> counts(dims.size());
    std::copy(dims.begin(), dims.end(), counts.begin());
    counts[dims.size() - 1] = 1;
    std::vector<hsize_t> offsets(dims.size(), 0);

    H5Sselect_none(space.getId());
    for (std::vector<size_t>::const_iterator i = columns.begin();
         i != columns.end(); ++i) {

        offsets[offsets.size() - 1] = *i;
        if (H5Sselect_hyperslab(space.getId(), H5S_SELECT_OR, offsets.data(),
                                0, counts.data(), 0) < 0) {
            HDF5ErrMapper::ToException<DataSpaceException>(
                "Unable to select hyperslap");
        }
    }

    dims[dims.size() - 1] = columns.size();
    return Selection(DataSpace(dims), space, dataset);
}

template <typename Derivate>
inline Selection
SliceTraits<Derivate>::select(const ElementSet& elements) const {
    hsize_t* data = NULL;
    const std::size_t length = elements._ids.size();
    std::vector<hsize_t> raw_elements;

    // optimised at compile time
    // switch for data conversion on 32bits platforms
    if (std::is_same<std::size_t, hsize_t>::value) {
        data = (hsize_t*)(&(elements._ids[0]));
    } else {
        raw_elements.resize(length);
        std::copy(elements._ids.begin(), elements._ids.end(),
                  raw_elements.begin());
        data = &(raw_elements[0]);
    }

    DataSpace space = static_cast<const Derivate*>(this)->getSpace().clone();
    if (H5Sselect_elements(space.getId(), H5S_SELECT_SET, length, data) < 0) {
        HDF5ErrMapper::ToException<DataSpaceException>(
            "Unable to select elements");
    }

    return Selection(DataSpace(length), space,
                     details::get_dataset(static_cast<const Derivate*>(this)));
}

template <typename Derivate>
template <typename T>
inline void SliceTraits<Derivate>::read(T& array) const {
    typedef typename std::remove_const<T>::type type_no_const;

    type_no_const& nocv_array = const_cast<type_no_const&>(array);

    const size_t dim_array = details::array_dims<type_no_const>::value;
    DataSpace space = static_cast<const Derivate*>(this)->getSpace();
    DataSpace mem_space = static_cast<const Derivate*>(this)->getMemSpace();

    if (!details::checkDimensions(mem_space, dim_array)) {
        std::ostringstream ss;
        ss << "Impossible to read DataSet of dimensions "
           << mem_space.getNumberDimensions() << " into arrays of dimensions "
           << dim_array;
        throw DataSpaceException(ss.str());
    }

    // Create mem datatype
    const AtomicType<typename details::type_of_array<type_no_const>::type>
        array_datatype;

    // Apply pre read convertions
    details::data_converter<type_no_const> converter(nocv_array, mem_space);

    if (H5Dread(
            details::get_dataset(static_cast<const Derivate*>(this)).getId(),
            array_datatype.getId(),
            details::get_memspace_id((static_cast<const Derivate*>(this))),
            space.getId(), H5P_DEFAULT,
            static_cast<void*>(converter.transform_read(nocv_array))) < 0) {
        HDF5ErrMapper::ToException<DataSetException>(
            "Error during HDF5 Read: ");
    }

    // re-arrange results
    converter.process_result(array);
}

template <typename Derivate>
template <typename T>
inline void SliceTraits<Derivate>::read(T* array) const {

    DataSpace space = static_cast<const Derivate*>(this)->getSpace();
    DataSpace mem_space = static_cast<const Derivate*>(this)->getMemSpace();

    // Create mem datatype
    const AtomicType<typename details::type_of_array<T>::type> array_datatype;

    if (H5Dread(
            details::get_dataset(static_cast<const Derivate*>(this)).getId(),
            array_datatype.getId(),
            details::get_memspace_id((static_cast<const Derivate*>(this))),
            space.getId(), H5P_DEFAULT,
            static_cast<void*>(array)) < 0) {
        HDF5ErrMapper::ToException<DataSetException>(
            "Error during HDF5 Read: ");
    }
}

template <typename Derivate>
template <typename T>
inline void SliceTraits<Derivate>::write(const T& buffer) {
    typedef typename std::remove_const<T>::type type_no_const;

    type_no_const& nocv_buffer = const_cast<type_no_const&>(buffer);

    const size_t dim_buffer = details::array_dims<type_no_const>::value;
    DataSpace space = static_cast<const Derivate*>(this)->getSpace();
    DataSpace mem_space = static_cast<const Derivate*>(this)->getMemSpace();

    if (!details::checkDimensions(mem_space, dim_buffer)) {
        std::ostringstream ss;
        ss << "Impossible to write buffer of dimensions " << dim_buffer
           << " into dataset of dimensions " << mem_space.getNumberDimensions();
        throw DataSpaceException(ss.str());
    }

    const AtomicType<typename details::type_of_array<type_no_const>::type>
        array_datatype;

    // Apply pre write convertions
    details::data_converter<type_no_const> converter(nocv_buffer, mem_space);

    if (H5Dwrite(details::get_dataset(static_cast<Derivate*>(this)).getId(),
                 array_datatype.getId(),
                 details::get_memspace_id((static_cast<Derivate*>(this))),
                 space.getId(), H5P_DEFAULT,
                 static_cast<const void*>(
                     converter.transform_write(nocv_buffer))) < 0) {
        HDF5ErrMapper::ToException<DataSetException>(
            "Error during HDF5 Write: ");
    }
}

template <typename Derivate>
template <typename T>
inline void SliceTraits<Derivate>::write(const T* buffer) {

    DataSpace space = static_cast<const Derivate*>(this)->getSpace();
    DataSpace mem_space = static_cast<const Derivate*>(this)->getMemSpace();

    const AtomicType<typename details::type_of_array<T>::type> array_datatype;

    if (H5Dwrite(details::get_dataset(static_cast<Derivate*>(this)).getId(),
                 array_datatype.getId(),
                 details::get_memspace_id((static_cast<Derivate*>(this))),
                 space.getId(), H5P_DEFAULT,
                 static_cast<const void*>(buffer)) < 0) {
        HDF5ErrMapper::ToException<DataSetException>(
            "Error during HDF5 Write: ");
    }
}
}

#endif // H5SLICE_TRAITS_MISC_HPP
