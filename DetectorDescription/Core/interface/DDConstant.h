#ifndef DDConstant_h
#define DDConstant_h

#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include <vector>
#include <iostream>

class DDConstant;

//! output operator for printing ...
std::ostream & operator<<(std::ostream & o, const DDConstant & cons);

//! a named constant corresponding to the DDL-XML tag <Constant> and <ConstantsVector>
class DDConstant : public DDBase<DDName, double * >
{
public:
   //! an uninitialized constant; one can assign an initialized constant to make it valid
   DDConstant();
   
   //! a refenrence to a constant
   DDConstant(const DDName & name);
   
   //! creation of a new named constant; if it already existed with the given name, it's overwritten with new values
   DDConstant(const DDName & name, double* value);
   
   //! creates all DDConstants from the variables of the ClhepEvaluator
   static void createConstantsFromEvaluator();
      
   //! return the first stored value; does not check boundaries!
   double value() const { return rep(); }
      
   //! convert to a double
   operator double() const { return rep(); }
};

//! std::maps the XML naming convention, i.e. <Numeric name='foo' value='4711'/> -> DDNumeric 
typedef DDConstant DDNumeric;

#endif
