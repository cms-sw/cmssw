#ifndef SELECTIONUSERVARIABLES_H
#define SELECTIONUSERVARIABLES_H

/**
 * \class SelectionUserVariables
 *
 * Ugly class to "missuse" AlignmentParameters::userVariables() to transfer information
 * about other parameter selections then just '1' (keep) or '0' (ignore) to the alignment
 * algorithm.
 *
 *  \author    : Gero Flucke
 *  date       : November 2006
 *  $Revision: 1.2 $
 *  $Date: 2007/10/08 14:38:15 $
 *  (last update by $Author: cklae $)
 */

#include "Alignment/CommonAlignment/interface/AlignmentUserVariables.h"

class SelectionUserVariables : public AlignmentUserVariables 
{
 public:
  explicit SelectionUserVariables(const std::vector<char> &sel) : myFullSelection(sel) {}
  virtual ~SelectionUserVariables() {}
  virtual SelectionUserVariables* clone() const { return new SelectionUserVariables(*this);}

  const std::vector<char>& fullSelection() const {return myFullSelection;}

 private:
  std::vector<char>  myFullSelection;
};

#endif
