#ifndef AlignmentUserVariablesIO_H
#define AlignmentUserVariablesIO_H

#include <vector>

/// \class AlignmentUserVariablesIO
///
/// Abstract base class for I/O of AlignmentUserVariables.
/// Note that it is the caller's responsibility to delete objects created during reading.
///
///  $Date: 2006/12/12 08:55:44 $
///  $Revision: 1.3 $
///  $Author: flucke $ (at least last update...)

class Alignable;
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
  int write(const std::vector<Alignable*>& alivec, bool validCheck);

  /** read AlignmentUserVariables of many Alignables (using readOne, so take care of memory!) */
  std::vector<AlignmentUserVariables*> read(const std::vector<Alignable*>& alivec, int& ierr);

};

#endif
