#ifndef Alignment_CommonAlignmentAlgorithm_AlignableDataIORoot_h
#define Alignment_CommonAlignmentAlgorithm_AlignableDataIORoot_h

#include <map>

#include "TTree.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignableDataIO.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORootBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignableData.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"

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

  int findEntry(unsigned int detId,int comp);
  void createBranches(void);
  void setBranchAddresses(void);

  // data members

  /// root tree contents 
  int ObjId;
  //unsigned int Id;
  int Id;
  double Pos[3];
  double Rot[9];

  bool newopen;
  typedef  std::map< std::pair<int,int> , int > treemaptype;
  treemaptype treemap;

};



#endif
