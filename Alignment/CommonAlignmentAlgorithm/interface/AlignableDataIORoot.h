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
  int open(const char* filename, int iteration, bool writemode) override
  { newopen=true; return openRoot(filename,iteration,writemode); }

  /// close IO 
  int close(void) override{ return closeRoot(); }

  /// write absolute positions 
  int writeAbsRaw(const AlignableAbsData &ad) override;
  /// read absolute positions 
  AlignableAbsData readAbsRaw(Alignable* ali,int& ierr) override;
  /// write relative positions 
  int writeRelRaw(const AlignableRelData &ad) override;
  /// read relative positions 
  AlignableRelData readRelRaw(Alignable* ali,int& ierr) override;

  int findEntry(align::ID, align::StructureType);
  void createBranches(void) override;
  void setBranchAddresses(void) override;

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
