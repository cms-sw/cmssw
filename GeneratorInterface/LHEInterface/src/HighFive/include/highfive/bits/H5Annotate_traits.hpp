/*
 *  Copyright (c), 2017, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE_1_0.txt or copy at
 *          http://www.boost.org/LICENSE_1_0.txt)
 *
 */
#ifndef H5ANNOTATE_TRAITS_HPP
#define H5ANNOTATE_TRAITS_HPP

#include <string>

namespace HighFive {

class Attribute;
class DataSet;
class Group;
class DataSpace;
class DataType;

template <typename Derivate>
class AnnotateTraits {
  public:
    ///
    /// \brief create a new attribute with the name attribute_name
    /// \param attribute_name
    /// \return the attribute object
    ///
    Attribute createAttribute(const std::string& attribute_name,
                              const DataSpace& space, const DataType& type);

    ///
    /// \brief createDataSet create a new dataset in the current file with a
    /// size specified by space
    /// \param dataset_name identifier of the dataset
    /// \param space Associated DataSpace, see \ref DataSpace for more
    /// informations
    /// \return DataSet Object
    ///
    ///
    ///
    template <typename Type>
    Attribute createAttribute(const std::string& attribute_name,
                              const DataSpace& space);

    ///
    /// \brief open an existing attribute with the name attribute_name
    /// \param attribute_name
    /// \return the attribute object
    ///
    Attribute getAttribute(const std::string& attribute_name) const;

    ///
    /// \brief return the number of attributes of the node / group
    /// \return number of attributes
    size_t getNumberAttributes() const;

    ///
    /// \brief list all attribute name of the node / group
    /// \return number of attributes
    std::vector<std::string> listAttributeNames() const;

    ///
    /// \brief checks an attribute exists
    /// \return number of attributes
    bool hasAttribute(const std::string& attr_name) const;

  private:
    typedef Derivate derivate_type;
};
}

#include "H5Annotate_traits_misc.hpp"

#endif // H5ANNOTATE_TRAITS_HPP
