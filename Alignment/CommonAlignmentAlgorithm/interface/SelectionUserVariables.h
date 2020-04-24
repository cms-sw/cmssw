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
 *  $Revision: 1.1 $
 *  $Date: 2006/11/30 10:08:26 $
 *  (last update by $Author: flucke $)
 */

#include "Alignment/CommonAlignment/interface/AlignmentUserVariables.h"

class SelectionUserVariables : public AlignmentUserVariables 
{
 public:
  explicit SelectionUserVariables(const std::vector<char> &sel) : myFullSelection(sel) {}
  ~SelectionUserVariables() override {}
  SelectionUserVariables* clone() const override { return new SelectionUserVariables(*this);}

  const std::vector<char>& fullSelection() const {return myFullSelection;}

 private:
  std::vector<char>  myFullSelection;
};

#endif
