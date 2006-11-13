#ifndef AlignmentUserVariablesIO_H
#define AlignmentUserVariablesIO_H

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"

#include<vector>

/** abstract base class for I/O of AlignmentUserVariables; 
 *  NOTE: New objects AlignmentUserVariables are created by the read
 *  methods. They are deleted when the destructor of this class
 *  is called.
 */

class AlignmentUserVariablesIO 
{

  protected:

  virtual ~AlignmentUserVariablesIO();

  /** open IO */
  virtual int open(const char* filename, int iteration, bool writemode) =0;

  /** close IO */
  virtual int close(void) =0;

  /** write AlignmentUserVariables of one Alignable */
  virtual int writeOne(Alignable* ali) =0;

  /** read AlignmentUserVariables of one Alignable */
  virtual AlignmentUserVariables* readOne(Alignable* ali, int& ierr) =0;

  /** write AlignmentUserVariables of many Alignables */
  int write(const std::vector<Alignable*>& alivec, bool validCheck);

  /** read AlignmentUserVariables of many Alignables */
  std::vector<AlignmentUserVariables*> read(const std::vector<Alignable*>& alivec, 
    int& ierr);

  // data members

  std::vector<AlignmentUserVariables*> theReadUserVariables;

};

#endif
