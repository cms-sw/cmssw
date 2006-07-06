#ifndef Common_ParameterSetID_h
#define Common_ParameterSetID_h

/*----------------------------------------------------------------------
  
ParameterSetID: A globally unique identifier for each collection of
tracked parameters. Two ParameterSet objects will have equal
ParameterSetIDs if they contain the same set of tracked parameters.

We calculate the ParameterSetID from the names and values of the
tracked parameters within a ParameterSet, currently using the MD5
algorithm.

$Id: ParameterSetID.h,v 1.1.2.1 2006/06/27 21:05:17 paterno Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/Hash.h"
#include "DataFormats/Common/interface/HashedTypes.h"

namespace edm 
{
  typedef Hash<ParameterSetType> ParameterSetID;

//   class  ParameterSetID 
//   {
//   public:

//     // We use the string representation of the 16-byte MD5 digest, for
//     // now. This can be improved.
//     typedef std::string    value_t;

//     ParameterSetID();
//     explicit ParameterSetID(const value_t& v);

//     bool isValid() const;

//     bool operator==(ParameterSetID const& rh) const;
//     bool operator!=(ParameterSetID const& rh) const;

//     bool operator<(ParameterSetID const& rh) const;
//     bool operator>(ParameterSetID const& rh) const;

//     std::ostream& print(std::ostream& os) const;

//   private:
//     value_t   value_;
//   };

  
//   inline
//   std::ostream&
//   operator<<(std::ostream& os, ParameterSetID const& id) 
//   {
//     id.print(os);
//     return os;
//   }
}
#endif
