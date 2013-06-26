#ifndef DDVector_h
#define DDVector_h

#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include <vector>
#include <iostream>

class DDVector;

//! output operator for printing ...
std::ostream & operator<<(std::ostream & o, const DDVector & cons);

//typedef std::vector<double> dd_constant_type;

//! a named constant corresponding to the DDL-XML tag <Constant> and <ConstantsVector>
class DDVector : public DDBase<DDName, std::vector<double>* >
{
public:

   //! size type for the size of the stored values
   typedef std::vector<double>::size_type size_t;
   
   //! iterator for read-only acces to stored values
   //typedef std::vector<double>::const_iterator const_iterator;
   
   //! value type of the managed object
   typedef std::vector<double> value_type;
   
   //! an uninitialized constant; one can assign an initialized constant to make it valid
   DDVector();
   
   //! a refenrence to a constant
   DDVector(const DDName & name);
   
   //! creation of a new named constant; if it already existed with the given name, it's overwritten with new values
   DDVector(const DDName & name, std::vector<double>* value);
      
   //! the size of the array of values 
   size_t size() const { return rep().size(); }
   
   //! the stored values
   const value_type & values() const { return rep(); }
   
   //! returns the value on position pos; does not check boundaries!
   double operator[](size_t pos) const { return rep()[pos]; }
   
   //! return the first stored value; does not check boundaries!
   double value() const { return rep()[0]; }
   
   //! read-only iterator pointing to the begin of the stored values
   value_type::const_iterator vectorBegin() const { return rep().begin(); }
   
   //! read-only iterator poining one place after the stored values
   value_type::const_iterator vectorEnd() const { return rep().end(); }
   
   //! convert to a double
   operator double() const { return rep()[0]; }
   
   //! convert to a std::vector<double>
   operator std::vector<double>() const { return rep(); }
   
   //! convert to a std::vector<int> (expensive!)
   operator std::vector<int>() const;
};
#endif
