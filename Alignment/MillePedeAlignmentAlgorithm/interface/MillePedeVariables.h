#ifndef MILLEPEDEVARIABLES_H
#define MILLEPEDEVARIABLES_H

/**
 * \class MillePedeVariables
 *
 * container for millepede specific variables attached to AlignmentParameters
 *
 *  \author    : Gero Flucke
 *  date       : November 2006
 *  $Revision: 1.2 $
 *  $Date: 2006/11/07 10:45:09 $
 *  (last update by $Author: flucke $)
 */

#include "Alignment/CommonAlignment/interface/AlignmentUserVariables.h"

#include <vector>

class MillePedeVariables : public AlignmentUserVariables {
 public:
  
  /** constructor */
  explicit MillePedeVariables(unsigned int nParams);
  /** destructor */
  virtual ~MillePedeVariables() {}
  /** clone method (using copy constructor) */
  virtual MillePedeVariables* clone() const { return new MillePedeVariables(*this);}

  /// set default values for all data concerning nParam (false if nParam out of range)
  bool setAllDefault(unsigned int nParam);
  /// number of parameters
  unsigned int size() const {return myIsValid.size();}

  /// get valid flag array
  const std::vector<bool>& isValid() const { return myIsValid;}
  /// get valid flag array for changing it
  std::vector<bool>& isValid() { return myIsValid;}

  /// get array of differences to start value
  const std::vector<float>& diffBefore() const {return myDiffBefore;}
  /// get array of differences to start value for changing it
  std::vector<float>& diffBefore() {return myDiffBefore;}

  /// get global correlation array
  const std::vector<float>& globalCor() const {return myGlobalCor;}
  /// get global correlation array for changing it
  std::vector<float>& globalCor() {return myGlobalCor;}

  /// get array of presigmas (<= 0: means fixed)
  const std::vector<float>& preSigma() const {return myPreSigma;}
  /// get array of presigmas (<= 0: means fixed) for changing it
  std::vector<float>& preSigma() {return myPreSigma;}

  /// true if parameter is fixed
  bool isFixed(unsigned int nParam) const;

 private:
  std::vector<bool>  myIsValid;
  std::vector<float> myDiffBefore;
  std::vector<float> myGlobalCor;
  std::vector<float> myPreSigma; /// <= 0 means fixed
};

#endif
