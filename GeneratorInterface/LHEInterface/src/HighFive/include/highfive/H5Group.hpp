/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef HIGHFIVE_H5GROUP_HPP
#define HIGHFIVE_H5GROUP_HPP

#include "H5Object.hpp"
#include "bits/H5Annotate_traits.hpp"
#include "bits/H5Node_traits.hpp"

namespace HighFive {

class File;

class Group : public Object,
              public NodeTraits<Group>,
              public AnnotateTraits<Group> {
  public:
    Group();

    friend class File;
};
}

#include "bits/H5Group_misc.hpp"

#endif // HIGHFIVE_H5GROUP_HPP
