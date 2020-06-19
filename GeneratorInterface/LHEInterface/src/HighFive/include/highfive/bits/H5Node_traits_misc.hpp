/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5NODE_TRAITS_MISC_HPP
#define H5NODE_TRAITS_MISC_HPP

#include "H5Iterables_misc.hpp"
#include "H5Node_traits.hpp"

#include <string>
#include <vector>

#include "../H5Attribute.hpp"
#include "../H5DataSet.hpp"
#include "../H5DataSpace.hpp"
#include "../H5DataType.hpp"
#include "../H5Exception.hpp"
#include "../H5Group.hpp"

#include <H5Apublic.h>
#include <H5Dpublic.h>
#include <H5Fpublic.h>
#include <H5Gpublic.h>
#include <H5Ppublic.h>
#include <H5Tpublic.h>

namespace HighFive {

  template <typename Derivate>
  inline DataSet NodeTraits<Derivate>::createDataSet(const std::string& dataset_name,
                                                     const DataSpace& space,
                                                     const DataType& dtype,
                                                     const DataSetCreateProps& createProps) {
    DataSet set;
    if ((set._hid = H5Dcreate2(static_cast<Derivate*>(this)->getId(),
                               dataset_name.c_str(),
                               dtype._hid,
                               space._hid,
                               H5P_DEFAULT,
                               createProps.getId(),
                               H5P_DEFAULT)) < 0) {
      HDF5ErrMapper::ToException<DataSetException>(std::string("Unable to create the dataset \"") + dataset_name +
                                                   "\":");
    }
    return set;
  }

  template <typename Derivate>
  template <typename Type>
  inline DataSet NodeTraits<Derivate>::createDataSet(const std::string& dataset_name,
                                                     const DataSpace& space,
                                                     const DataSetCreateProps& createProps) {
    return createDataSet(dataset_name, space, AtomicType<Type>(), createProps);
  }

  template <typename Derivate>
  inline DataSet NodeTraits<Derivate>::getDataSet(const std::string& dataset_name) const {
    DataSet set;
    if ((set._hid = H5Dopen2(static_cast<const Derivate*>(this)->getId(), dataset_name.c_str(), H5P_DEFAULT)) < 0) {
      HDF5ErrMapper::ToException<DataSetException>(std::string("Unable to open the dataset \"") + dataset_name + "\":");
    }
    return set;
  }

  template <typename Derivate>
  inline Group NodeTraits<Derivate>::createGroup(const std::string& group_name) {
    Group group;
    if ((group._hid = H5Gcreate2(
             static_cast<Derivate*>(this)->getId(), group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) < 0) {
      HDF5ErrMapper::ToException<GroupException>(std::string("Unable to create the group \"") + group_name + "\":");
    }
    return group;
  }

  template <typename Derivate>
  inline Group NodeTraits<Derivate>::getGroup(const std::string& group_name) const {
    Group group;
    if ((group._hid = H5Gopen2(static_cast<const Derivate*>(this)->getId(), group_name.c_str(), H5P_DEFAULT)) < 0) {
      HDF5ErrMapper::ToException<GroupException>(std::string("Unable to open the group \"") + group_name + "\":");
    }
    return group;
  }

  template <typename Derivate>
  inline size_t NodeTraits<Derivate>::getNumberObjects() const {
    hsize_t res;
    if (H5Gget_num_objs(static_cast<const Derivate*>(this)->getId(), &res) < 0) {
      HDF5ErrMapper::ToException<GroupException>(std::string("Unable to count objects in existing group or file"));
    }
    return res;
  }

  template <typename Derivate>
  inline std::string NodeTraits<Derivate>::getObjectName(size_t index) const {
    const ssize_t maxLength = 1023;
    char buffer[maxLength + 1];
    ssize_t length = H5Lget_name_by_idx(static_cast<const Derivate*>(this)->getId(),
                                        ".",
                                        H5_INDEX_NAME,
                                        H5_ITER_INC,
                                        index,
                                        buffer,
                                        maxLength,
                                        H5P_DEFAULT);
    if (length < 0)
      HDF5ErrMapper::ToException<GroupException>("Error accessing object name");
    if (length <= maxLength)
      return std::string(buffer, length);
    std::vector<char> bigBuffer(length + 1, 0);
    H5Lget_name_by_idx(static_cast<const Derivate*>(this)->getId(),
                       ".",
                       H5_INDEX_NAME,
                       H5_ITER_INC,
                       index,
                       bigBuffer.data(),
                       length,
                       H5P_DEFAULT);
    return std::string(bigBuffer.data(), length);
  }

  template <typename Derivate>
  inline std::vector<std::string> NodeTraits<Derivate>::listObjectNames() const {
    std::vector<std::string> names;
    details::HighFiveIterateData iterateData(names);

    size_t num_objs = getNumberObjects();
    names.reserve(num_objs);

    if (H5Literate(static_cast<const Derivate*>(this)->getId(),
                   H5_INDEX_NAME,
                   H5_ITER_INC,
                   NULL,
                   &details::internal_high_five_iterate<H5L_info_t>,
                   static_cast<void*>(&iterateData)) < 0) {
      HDF5ErrMapper::ToException<GroupException>(std::string("Unable to list objects in group"));
    }

    return names;
  }

  template <typename Derivate>
  inline bool NodeTraits<Derivate>::exist(const std::string& node_name) const {
    htri_t val = H5Lexists(static_cast<const Derivate*>(this)->getId(), node_name.c_str(), H5P_DEFAULT);
    if (val < 0) {
      HDF5ErrMapper::ToException<GroupException>(std::string("Invalid link for exist() "));
    }

    return (val > 0);
  }

}  // namespace HighFive

#endif  // H5NODE_TRAITS_MISC_HPP
