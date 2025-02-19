#ifndef MuScleFitProvenance_h
#define MuScleFitProvenance_h

#include <TObject.h>

/**
 * This class is used to store some provenance information about the tree.
 */

class MuScleFitProvenance : public TObject
{
 public:
  MuScleFitProvenance()
  {}

  MuScleFitProvenance(const int inputMuonType) :
    muonType(inputMuonType)
  {}

  int muonType;

  ClassDef(MuScleFitProvenance, 1)
};
ClassImp(MuScleFitProvenance)

#endif