#ifndef Alignment_CommonAlignmentAlgorithm_AlignableDataIORoot_h
#define Alignment_CommonAlignmentAlgorithm_AlignableDataIORoot_h

#include <map>

#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignableDataIO.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORootBase.h"

/// concrete class for ROOT based IO of Alignable positions 

class AlignableDataIORoot : public AlignmentIORootBase, public AlignableDataIO
{

  friend class AlignmentIORoot;

  private:
  /// constructor 
  AlignableDataIORoot(PosType p); 

  /// open IO 
  int open(const char* filename, int iteration, bool writemode)
  { newopen=true; return openRoot(filename,iteration,writemode); }

  /// close IO 
  int close(void){ return closeRoot(); }

  /// write absolute positions 
  int writeAbsRaw(const AlignableAbsData &ad);
  /// read absolute positions 
  AlignableAbsData readAbsRaw(Alignable* ali,int& ierr);
  /// write relative positions 
  int writeRelRaw(const AlignableRelData &ad);
  /// read relative positions 
  AlignableRelData readRelRaw(Alignable* ali,int& ierr);

  int findEntry(align::ID, align::StructureType);
  void createBranches(void);
  void setBranchAddresses(void);

  // data members

  /// root tree contents 
  align::StructureType ObjId;
  //unsigned int Id;
  align::ID Id;
  Double_t Pos[3];
  Double_t Rot[9];
  UInt_t numDeformationValues_;
  enum {kMaxNumPar = 20}; // slighly above 'two bowed surfaces' limit
  Float_t deformationValues_[kMaxNumPar];

  bool newopen;
  typedef std::map< std::pair<align::ID, align::StructureType>, int > treemaptype;
  treemaptype treemap;

};



#endif
