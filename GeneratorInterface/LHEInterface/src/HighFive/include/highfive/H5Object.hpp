/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5OBJECT_HPP
#define H5OBJECT_HPP

#include <H5Ipublic.h>

namespace HighFive {

template <typename Derivate>
class NodeTraits;
template <typename Derivate>
class AnnotateTraits;

class Object {
  public:
    // decrease reference counter
    ~Object();

    ///
    /// \brief isValid
    /// \return true if current Object is a valid HDF5Object
    ///
    bool isValid() const;

    ///
    /// \brief getId
    /// \return iternal HDF5 id to the object
    ///  provided for C API compatibility
    ///
    hid_t getId() const;

  protected:
    // empty constructor
    Object();

    // copy constructor, increase reference counter
    Object(const Object& other);

    Object& operator=(const Object& other);

    hid_t _hid;

  private:
    template <typename Derivate>
    friend class NodeTraits;
    template <typename Derivate>
    friend class AnnotateTraits;
};
}

#include "bits/H5Object_misc.hpp"

#endif // H5OBJECT_HPP
