#ifndef DDString_h
#define DDString_h

#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include <string>
#include <iostream>

class DDString;
using std::string;

//! output operator for printing ...
ostream & operator<<(ostream & o, const DDString & cons);

//! a named constant corresponding to the DDL-XML tag <Constant> and <ConstantsVector>
class DDString : public DDBase<DDName, string * >
{
public:
   //! an uninitialized constant; one can assign an initialized constant to make it valid
   DDString();
   
   //! a refenrence to a constant
   DDString(const DDName & name);
   
   //! creation of a new named constant; if it already existed with the given name, it's overwritten with new values
   DDString(const DDName & name, string* value);
      
   //! return the first stored value; does not check boundaries!
   const string & value() const { return rep(); }
      
   //! convert to a string
   operator string() const { return rep(); }
};

#endif
