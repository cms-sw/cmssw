/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5ANNOTATE_TRAITS_MISC_HPP
#define H5ANNOTATE_TRAITS_MISC_HPP

#include "H5Annotate_traits.hpp"
#include "H5Iterables_misc.hpp"

#include <string>
#include <vector>

#include "../H5Attribute.hpp"
#include "../H5DataSpace.hpp"
#include "../H5DataType.hpp"
#include "../H5Exception.hpp"

#include <H5Apublic.h>

namespace HighFive {

template <typename Derivate>
inline Attribute
AnnotateTraits<Derivate>::createAttribute(const std::string& attribute_name,
                                          const DataSpace& space,
                                          const DataType& dtype) {
    Attribute attribute;
    if ((attribute._hid = H5Acreate2(
             static_cast<Derivate*>(this)->getId(), attribute_name.c_str(),
             dtype._hid, space._hid, H5P_DEFAULT, H5P_DEFAULT)) < 0) {
        HDF5ErrMapper::ToException<AttributeException>(
            std::string("Unable to create the attribute \"") + attribute_name +
            "\":");
    }
    return attribute;
}

template <typename Derivate>
template <typename Type>
inline Attribute
AnnotateTraits<Derivate>::createAttribute(const std::string& attribute_name,
                                          const DataSpace& space) {
    return createAttribute(attribute_name, space, AtomicType<Type>());
}

template <typename Derivate>
inline Attribute AnnotateTraits<Derivate>::getAttribute(
    const std::string& attribute_name) const {
    Attribute attribute;
    if ((attribute._hid = H5Aopen(static_cast<const Derivate*>(this)->getId(),
                                  attribute_name.c_str(), H5P_DEFAULT)) < 0) {
        HDF5ErrMapper::ToException<AttributeException>(
            std::string("Unable to open the attribute \"") + attribute_name +
            "\":");
    }
    return attribute;
}

template <typename Derivate>
inline size_t AnnotateTraits<Derivate>::getNumberAttributes() const {
    int res = H5Aget_num_attrs(static_cast<const Derivate*>(this)->getId());
    if (res < 0) {
        HDF5ErrMapper::ToException<AttributeException>(std::string(
            "Unable to count attributes in existing group or file"));
    }
    return res;
}

template <typename Derivate>
inline std::vector<std::string>
AnnotateTraits<Derivate>::listAttributeNames() const {

    std::vector<std::string> names;
    details::HighFiveIterateData iterateData(names);

    size_t num_objs = getNumberAttributes();
    names.reserve(num_objs);

    if (H5Aiterate2(static_cast<const Derivate*>(this)->getId(), H5_INDEX_NAME,
                    H5_ITER_INC, NULL,
                    &details::internal_high_five_iterate<H5A_info_t>,
                    static_cast<void*>(&iterateData)) < 0) {
        HDF5ErrMapper::ToException<AttributeException>(
            std::string("Unable to list attributes in group"));
    }

    return names;
}

template <typename Derivate>
inline bool
AnnotateTraits<Derivate>::hasAttribute(const std::string& attr_name) const {
    int res = H5Aexists(static_cast<const Derivate*>(this)->getId(),
                        attr_name.c_str());
    if (res < 0) {
        HDF5ErrMapper::ToException<AttributeException>(
            std::string("Unable to check for attribute in group"));
    }
    return res;
}
}

#endif // H5ANNOTATE_TRAITS_MISC_HPP
