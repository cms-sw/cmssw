#ifndef MILLEPEDEVARIABLESIOROOT_H
#define MILLEPEDEVARIABLESIOROOT_H

/// \class MillePedeVariablesIORoot
///
/// ROOT based IO of MillePedeVariables
///
///  \author    : Gero Flucke
///  date       : November 2006
///  $Revision: 1.3 $
///  $Date: 2007/03/16 17:03:02 $
///  (last update by $Author: flucke $)


#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORootBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentUserVariablesIO.h"

#include <string>
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
  ~MillePedeVariablesIORoot() override {}

  /** write user variables */
  void writeMillePedeVariables(const std::vector<Alignable*> &alivec, const char *filename,
			       int iter, bool validCheck, int &ierr);

  /** read user variables (not that their memory is owned by this class!) */
  std::vector<AlignmentUserVariables*> readMillePedeVariables
    (const std::vector<Alignable*> &alivec, const char *filename, int iter, int &ierr);

 protected:

  /** write MillePedeVariables attached to AlignmentParameters of one Alignable */
  int writeOne(Alignable *ali) override; // inherited from AlignmentUserVariablesIO

  /** read MillePedeVariables belonging to one Alignable */
  AlignmentUserVariables* readOne(Alignable *ali, int &ierr) override;
  // previous inherited from AlignmentUserVariablesIO

  /** open IO */  // inherited from AlignmentUserVariablesIO
  int open(const char *filename, int iteration, bool writemode) override 
    { return this->openRoot(filename, iteration, writemode);}

  /** close IO */
  int close() override {return this->closeRoot();} // inherited from AlignmentUserVariablesIO

  /// create root branches 
  void createBranches() override;      // inherited from AlignmentIORootBase
  /// set root branche addresses 
  void setBranchAddresses() override;  // inherited from AlignmentIORootBase

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
  std::string  myName;
  std::string* myNamePtr;	// needed for ROOT IO
};

#endif
