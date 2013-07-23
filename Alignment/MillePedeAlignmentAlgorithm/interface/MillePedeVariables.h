#ifndef MILLEPEDEVARIABLES_H
#define MILLEPEDEVARIABLES_H

/**
 * \class MillePedeVariables
 *
 * container for millepede specific variables attached to AlignmentParameters
 *
 *  \author    : Gero Flucke
 *  date       : November 2006
 *  $Revision: 1.3 $
 *  $Date: 2007/03/16 17:03:02 $
 *  (last update by $Author: flucke $)
 */

#include "Alignment/CommonAlignment/interface/AlignmentUserVariables.h"

#include <vector>

class MillePedeVariables : public AlignmentUserVariables {
 public:
  
  /** constructor */
  MillePedeVariables(unsigned int nParams, unsigned int label);
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

  /// get array of sigmas
  const std::vector<float>& sigma() const {return mySigma;}
  /// get array of sigmas for changing it
  std::vector<float>& sigma() {return mySigma;}

  /// get array of parameters
  const std::vector<float>& parameter() const {return myParameter;}
  /// get array of parameters for changing it
  std::vector<float>& parameter() {return myParameter;}

  /// get alignable label as used by pede
  unsigned int label() const {return myLabel;}
  /// set alignable label as used by pede
  void setLabel(unsigned int label) { myLabel = label;}

  /// get number of hits for x-measurement
  unsigned int hitsX() const {return myHitsX;}
  /// increase hits for x-measurement
  void increaseHitsX(unsigned int add = 1) { myHitsX += add;}
  void setHitsX(unsigned int hitsX) { myHitsX = hitsX;}

  /// get number of hits for y-measurement
  unsigned int hitsY() const {return myHitsY;}
  /// increase hits for y-measurement
  void increaseHitsY(unsigned int add = 1) { myHitsY += add;}
  void setHitsY(unsigned int hitsY) { myHitsY = hitsY;}

  /// true if parameter is fixed
  bool isFixed(unsigned int nParam) const;

 private:
  MillePedeVariables() {} // make unusable default constructor

  std::vector<bool>  myIsValid;
  std::vector<float> myDiffBefore;
  std::vector<float> myGlobalCor;
  std::vector<float> myPreSigma; /// <= 0 means fixed
  std::vector<float> myParameter;
  std::vector<float> mySigma;
  unsigned int       myHitsX;
  unsigned int       myHitsY;
  unsigned int       myLabel;
};

#endif
