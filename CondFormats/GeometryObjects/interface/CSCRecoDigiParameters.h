#ifndef CSCObjects_CSCRecoDigiParameters_h
#define CSCObjects_CSCRecoDigiParameters_h

/** \class CSCRecoDigiParameters
 *
 *  Build the CSCGeometry from the DDD description.
 *
 *  \author Tim Cox
 *
 *  Michael Case (MEC) One per Chamber Type.
 */

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <vector>

class CSCRecoDigiParameters {

public:
  CSCRecoDigiParameters() { } 
  ~CSCRecoDigiParameters() { }

  std::vector<int> pUserParOffset; // where the fupars for a ch. type start in the fupars blob.
  std::vector<int> pUserParSize; // size of the fupars.  if known, then both this and the above can go.
  std::vector<int> pChamberType;
  std::vector<float> pfupars;   // user parameters

  COND_SERIALIZABLE;
};

#endif

