#ifndef DDString_h
#define DDString_h

#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include <string>
#include <iostream>
#include <memory>

class DDString;


//! output operator for printing ...
std::ostream & operator<<(std::ostream & o, const DDString & cons);

//! a named constant corresponding to the DDL-XML tag <Constant> and <ConstantsVector>
class DDString : public DDBase<DDName, std::unique_ptr<std::string> >
{
public:
   //! an uninitialized constant; one can assign an initialized constant to make it valid
   DDString();
   
   //! a refenrence to a constant
   DDString(const DDName & name);
   
   //! creation of a new named constant; if it already existed with the given name, it's overwritten with new values
   DDString(const DDName & name, std::unique_ptr<std::string> value);
      
   //! return the first stored value; does not check boundaries!
   const std::string & value() const { return rep(); }
      
   //! convert to a std::string
   operator std::string() const { return rep(); }
};

#endif
