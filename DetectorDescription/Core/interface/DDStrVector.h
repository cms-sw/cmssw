#ifndef DETECTOR_DESCRIPTION_CORE_DDSTRVECTOR_H
#define DETECTOR_DESCRIPTION_CORE_DDSTRVECTOR_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDName.h"

class DDStrVector;

//! output operator for printing ...
std::ostream& operator<<(std::ostream& o, const DDStrVector& cons);

//! a named constant corresponding to the DDL-XML tag <Constant> and <ConstantsStrVector>
class DDStrVector : public DDBase<DDName, std::unique_ptr<std::vector<std::string>>> {
public:
  //! size type for the size of the stored values
  using size_t = std::vector<std::string>::size_type;

  //! value type of the managed object
  using value_type = std::vector<std::string>;

  //! an uninitialized constant; one can assign an initialized constant to make it valid
  DDStrVector();

  //! a refenrence to a constant
  DDStrVector(const DDName& name);

  //! creation of a new named constant; if it already existed with the given name, it's overwritten with new values
  DDStrVector(const DDName& name, std::unique_ptr<std::vector<std::string>> value);

  //! the size of the array of values
  size_t size() const { return rep().size(); }

  //! the stored values
  const value_type& values() const { return rep(); }

  //! returns the value on position pos; does not check boundaries!
  std::string operator[](size_t pos) const { return rep()[pos]; }

  //! return the first stored value; does not check boundaries!
  std::string value() const { return rep()[0]; }

  //! read-only iterator pointing to the begin of the stored values
  value_type::const_iterator vectorBegin() const { return rep().begin(); }

  //! read-only iterator poining one place after the stored values
  value_type::const_iterator vectorEnd() const { return rep().end(); }

  //! convert to a double
  operator std::string() const { return rep()[0]; }

  //! convert to a std::vector<double>
  operator std::vector<std::string>() const { return rep(); }
};

#endif
