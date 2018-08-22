#ifndef DD_DDMap_h
#define DD_DDMap_h

#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "DetectorDescription/Core/interface/DDReadMapType.h"
#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDName.h"

class DDMap;

//! output operator for printing ...
std::ostream & operator<<(std::ostream & o, const DDMap & cons);

//! simply a std::map<std::string,double> supporting an addional operator[] const
using dd_map_type = ReadMapType<double>;

//! a named constant corresponding to the DDL-XML tag <Constant> and <ConstantsVector>
class DDMap : public DDBase<DDName, std::unique_ptr<dd_map_type> >
{
public:
   //! the type of the managed object
   using value_type = dd_map_type;
   
   //! size type for the size of the stored values
   using size_t = dd_map_type::size_type;
      
   //! an uninitialized constant; one can assign an initialized constant to make it valid
   DDMap();
   
   //! a refenrence to a constant
   DDMap(const DDName & name);
   
   //! creation of a new named constant; if it already existed with the given name, it's overwritten with new values
   DDMap(const DDName & name, std::unique_ptr<dd_map_type> value);
      
   //! the size of the array of values 
   size_t size() const { return rep().size(); }
   
   //! the stored values
   const dd_map_type & values() const { return rep(); }
   
   //! returns the value on position pos; does not check boundaries!
   const double & operator[](const std::string & name) const {
     const dd_map_type & r(rep()); 
     return r[name];
   } 
   
   //! read-only iterator pointing to the begin of the stored values
   value_type::const_iterator mapBegin() const { return rep().begin(); }
   
   //! read-only iterator poining one place after the stored values
   value_type::const_iterator mapEnd() const { return rep().end(); }
   
};
#endif // DD_DDMap_h
