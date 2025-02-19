#ifndef AlignmentUserVariablesIO_H
#define AlignmentUserVariablesIO_H

#include "Alignment/CommonAlignment/interface/Utilities.h"

/// \class AlignmentUserVariablesIO
///
/// Abstract base class for I/O of AlignmentUserVariables.
/// Note that it is the caller's responsibility to delete objects created during reading.
///
///  $Date: 2007/10/08 14:38:15 $
///  $Revision: 1.5 $
///  $Author: cklae $ (at least last update...)

class AlignmentUserVariables;

class AlignmentUserVariablesIO 
{

  protected:

  virtual ~AlignmentUserVariablesIO() {}

  /** open IO */
  virtual int open(const char* filename, int iteration, bool writemode) =0;

  /** close IO */
  virtual int close(void) =0;

  /** write AlignmentUserVariables of one Alignable */
  virtual int writeOne(Alignable* ali) =0;

  /** read AlignmentUserVariables of one Alignable,
      object should be created and has to be deleted */
  virtual AlignmentUserVariables* readOne(Alignable* ali, int& ierr) =0;

  /** write AlignmentUserVariables of many Alignables */
  int write(const align::Alignables& alivec, bool validCheck);

  /** read AlignmentUserVariables of many Alignables (using readOne, so take care of memory!) */
  std::vector<AlignmentUserVariables*> read(const align::Alignables& alivec, int& ierr);

};

#endif
