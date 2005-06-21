#ifndef DDStrVector_h
#define DDStrVector_h

#include "DetectorDescription/DDCore/interface/DDBase.h"
#include "DetectorDescription/DDCore/interface/DDName.h"
#include <vector>
#include <iostream>

class DDStrVector;

//! output operator for printing ...
ostream & operator<<(ostream & o, const DDStrVector & cons);

//typedef std::vector<double> dd_constant_type;

//! a named constant corresponding to the DDL-XML tag <Constant> and <ConstantsStrVector>
class DDStrVector : public DDBase<DDName, vector<string>* >
{
public:

   //! size type for the size of the stored values
   typedef vector<string>::size_type size_t;
   
   //! iterator for read-only acces to stored values
   //typedef vector<string>::const_iterator const_iterator;
   
   //! value type of the managed object
   typedef vector<string> value_type;
   
   //! an uninitialized constant; one can assign an initialized constant to make it valid
   DDStrVector();
   
   //! a refenrence to a constant
   DDStrVector(const DDName & name);
   
   //! creation of a new named constant; if it already existed with the given name, it's overwritten with new values
   DDStrVector(const DDName & name, vector<string>* value);
      
   //! the size of the array of values 
   size_t size() const { return rep().size(); }
   
   //! the stored values
   const value_type & values() const { return rep(); }
   
   //! returns the value on position pos; does not check boundaries!
   string operator[](size_t pos) const { return rep()[pos]; }
   
   //! return the first stored value; does not check boundaries!
   string value() const { return rep()[0]; }
   
   //! read-only iterator pointing to the begin of the stored values
   value_type::const_iterator vectorBegin() const { return rep().begin(); }
   
   //! read-only iterator poining one place after the stored values
   value_type::const_iterator vectorEnd() const { return rep().end(); }
   
   //! convert to a double
   operator string() const { return rep()[0]; }
   
   //! convert to a vector<double>
   operator vector<string>() const { return rep(); }
   
};
#endif
