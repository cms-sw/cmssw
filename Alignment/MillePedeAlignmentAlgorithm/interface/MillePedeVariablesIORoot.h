#ifndef MILLEPEDEVARIABLESIOROOT_H
#define MILLEPEDEVARIABLESIOROOT_H

/// \class MillePedeVariablesIORoot
///
/// ROOT based IO of MillePedeVariables
///
///  \author    : Gero Flucke
///  date       : November 2006
///  $Revision: 1.4 $
///  $Date: 2010/10/26 20:49:42 $
///  (last update by $Author: flucke $)


#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORootBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentUserVariablesIO.h"

#include <vector>

// ROOT types:
#include "Rtypes.h"

class Alignable;
class AlignmentUserVariables;


//__________________________________________________________________________________________________

class MillePedeVariablesIORoot : public AlignmentIORootBase, public AlignmentUserVariablesIO
{
  public:
  MillePedeVariablesIORoot();
  virtual ~MillePedeVariablesIORoot() {}

  /** write user variables */
  void writeMillePedeVariables(const std::vector<Alignable*> &alivec, const char *filename,
			       int iter, bool validCheck, int &ierr);

  /** read user variables (not that their memory is owned by this class!) */
  std::vector<AlignmentUserVariables*> readMillePedeVariables
    (const std::vector<Alignable*> &alivec, const char *filename, int iter, int &ierr);

 protected:

  /** write MillePedeVariables attached to AlignmentParameters of one Alignable */
  virtual int writeOne(Alignable *ali); // inherited from AlignmentUserVariablesIO

  /** read MillePedeVariables belonging to one Alignable */
  virtual AlignmentUserVariables* readOne(Alignable *ali, int &ierr);
  // previous inherited from AlignmentUserVariablesIO

  /** open IO */  // inherited from AlignmentUserVariablesIO
  virtual int open(const char *filename, int iteration, bool writemode) 
    { return this->openRoot(filename, iteration, writemode);}

  /** close IO */
  virtual int close() {return this->closeRoot();} // inherited from AlignmentUserVariablesIO

  /// create root branches 
  virtual void createBranches();      // inherited from AlignmentIORootBase
  /// set root branche addresses 
  virtual void setBranchAddresses();  // inherited from AlignmentIORootBase

 private:
  // variables for ROOT tree
  enum {kMaxNumPar = 20}; // slighly above 'two bowed surfaces' limit

  unsigned int myId;
  int          myObjId;

  unsigned int myNumPar;
  Byte_t       myIsValid[kMaxNumPar];
  Float_t      myDiffBefore[kMaxNumPar];
  Float_t      myGlobalCor[kMaxNumPar];
  Float_t      myPreSigma[kMaxNumPar];
  Float_t      myParameter[kMaxNumPar];
  Float_t      mySigma[kMaxNumPar];
  UInt_t       myHitsX;
  UInt_t       myHitsY;
  UInt_t       myLabel;
};

#endif
