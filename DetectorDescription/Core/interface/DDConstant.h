#ifndef DDConstant_h
#define DDConstant_h

#include <iostream>
#include <vector>
#include <memory>

#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDName.h"

class DDConstant;
class ClhepEvaluator;

//! output operator for printing ...
std::ostream & operator<<(std::ostream & o, const DDConstant & cons);

//! a named constant corresponding to the DDL-XML tag <Constant> and <ConstantsVector>
class DDConstant : public DDBase<DDName, std::unique_ptr<double> >
{
public:
   //! an uninitialized constant; one can assign an initialized constant to make it valid
   DDConstant();
   
   //! a refenrence to a constant
   DDConstant(const DDName & name);
   
   //! creation of a new named constant; if it already existed with the given name, it's overwritten with new values
   DDConstant(const DDName & name, std::unique_ptr<double> value);
   
   //! creates all DDConstants from the variables of the ClhepEvaluator
   static void createConstantsFromEvaluator(ClhepEvaluator&);
      
   //! return the first stored value; does not check boundaries!
   double value() const { return rep(); }
      
   //! convert to a double
   operator double() const { return rep(); }
};

//! std::maps the XML naming convention, i.e. <Numeric name='foo' value='4711'/> -> DDNumeric 
using DDNumeric = DDConstant;

#endif
