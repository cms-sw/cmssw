#ifndef DataFormats_Provenance_ParameterSetID_h
#define DataFormats_Provenance_ParameterSetID_h

/*----------------------------------------------------------------------
  
ParameterSetID: A globally unique identifier for each collection of
tracked parameters. Two ParameterSet objects will have equal
ParameterSetIDs if they contain the same set of tracked parameters.

We calculate the ParameterSetID from the names and values of the
tracked parameters within a ParameterSet, currently using the MD5
algorithm.

$Id: ParameterSetID.h,v 1.1 2007/03/04 04:48:09 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/Hash.h"
#include "DataFormats/Provenance/interface/HashedTypes.h"

namespace edm 
{
  typedef Hash<ParameterSetType> ParameterSetID;

}
#endif
