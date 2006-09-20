#ifndef Common_ParameterSetID_h
#define Common_ParameterSetID_h

/*----------------------------------------------------------------------
  
ParameterSetID: A globally unique identifier for each collection of
tracked parameters. Two ParameterSet objects will have equal
ParameterSetIDs if they contain the same set of tracked parameters.

We calculate the ParameterSetID from the names and values of the
tracked parameters within a ParameterSet, currently using the MD5
algorithm.

$Id: ParameterSetID.h,v 1.2 2006/07/06 18:34:05 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/Hash.h"
#include "DataFormats/Common/interface/HashedTypes.h"

namespace edm 
{
  typedef Hash<ParameterSetType> ParameterSetID;

}
#endif
