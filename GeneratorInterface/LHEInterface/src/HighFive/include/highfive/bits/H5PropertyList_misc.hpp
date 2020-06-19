/*
 *  Copyright (c), 2017-2018, Adrien Devresse <adrien.devresse@epfl.ch>
 *                            Juan Hernando <juan.hernando@epfl.ch>
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5PROPERTY_LIST_MISC_HPP
#define H5PROPERTY_LIST_MISC_HPP

#include <H5Ppublic.h>

#include "../H5PropertyList.hpp"

namespace HighFive {

  inline Properties::Properties(Type type) : _type(type), _hid(H5P_DEFAULT) {}

#ifdef H5_USE_CXX11
  inline Properties::Properties(Properties&& other) : _type(other._type), _hid(other._hid) { other._hid = H5P_DEFAULT; }

  inline Properties& Properties::operator=(Properties&& other) {
    _type = other._type;
    // This code handles self-assigment without ifs
    const auto hid = other._hid;
    other._hid = H5P_DEFAULT;
    _hid = hid;
    return *this;
  }
#endif

  inline Properties::~Properties() {
    // H5P_DEFAULT and H5I_INVALID_HID are not the same Ensuring that ~Object
    if (_hid != H5P_DEFAULT)
      H5Pclose(_hid);
  }

  template <typename Property>
  inline void Properties::add(const Property& property) {
    if (_hid == H5P_DEFAULT) {
      hid_t type;
      // The HP5_XXX are macros with function calls
      switch (_type) {
        case FILE_ACCESS: {
          type = H5P_FILE_ACCESS;
          break;
        }
        case DATASET_CREATE: {
          type = H5P_DATASET_CREATE;
          break;
        }
        default:
          HDF5ErrMapper::ToException<PropertyException>(std::string("Unsupported property list type"));
      }
      if ((_hid = H5Pcreate(type)) < 0) {
        HDF5ErrMapper::ToException<PropertyException>(std::string("Unable to create property list"));
      }
    }

    property.apply(_hid);
  }

  inline void Chunking::apply(const hid_t hid) const {
    if (H5Pset_chunk(hid, _dims.size(), _dims.data()) < 0) {
      HDF5ErrMapper::ToException<PropertyException>("Error setting chunk property");
    }
  }

  inline void Deflate::apply(const hid_t hid) const {
    if (!H5Zfilter_avail(H5Z_FILTER_DEFLATE)) {
      HDF5ErrMapper::ToException<PropertyException>("Error setting deflate property");
    }

    if (H5Pset_deflate(hid, _level) < 0) {
      HDF5ErrMapper::ToException<PropertyException>("Error setting deflate property");
    }
  }

  inline void Shuffle::apply(const hid_t hid) const {
    if (!H5Zfilter_avail(H5Z_FILTER_SHUFFLE)) {
      HDF5ErrMapper::ToException<PropertyException>("Error setting shuffle property");
    }

    if (H5Pset_shuffle(hid) < 0) {
      HDF5ErrMapper::ToException<PropertyException>("Error setting shuffle property");
    }
  }
}  // namespace HighFive
#endif  // H5PROPERTY_LIST_HPP
