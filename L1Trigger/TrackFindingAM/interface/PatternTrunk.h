#ifndef _PATTERNTRUNK_H_
#define _PATTERNTRUNK_H_

#include <map>
#include <vector>
#include <cmath>
#include <cstring>
#ifdef __APPLE__
#include <boost/mpl/and.hpp> // needed for boost v1.35
#endif
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include "GradedPattern.h"

using namespace std;

/**
   \brief A PatternTrunk can contain one low definition pattern and all the associated full definitions patterns (all high definition patterns contnained have the same low resolution version). Used to store patterns and compute variable resolution patterns : we keep the low resolution versions and compute the DC bits values from the high resolution patterns.
**/

class PatternTrunk{
 public:
  /**
     \brief Constructor
     \param p The low definition pattern (will be copied)
  **/
  PatternTrunk(Pattern* p);

  /**
     \brief Default constructor
  **/
  PatternTrunk();
  ~PatternTrunk();
  /**
     \brief Add a new full definition pattern to the structure
     Increment the low definition pattern grade. If the FD pattern is already contained, increments the grade.
  **/
  void addFDPattern(Pattern* p);

  /**
     \brief Add a new full definition pattern to the structure and update the average pt of this pattern
     Increment the low definition pattern grade. If the FD pattern is already contained, increments the grade.
  **/
  void addFDPattern(Pattern* p, float pt);


  /**
     \brief Get the list of all the full definition patterns
     \return A vector of pointers on the Patterns (each Pattern is a copy)
  **/
  vector<GradedPattern*> getFDPatterns();
 /**
     \brief Get the low definition pattern
     \return A pointer on a copy of the Pattern
  **/
  GradedPattern* getLDPattern();
 /**
     \brief Get the average PT of the tracks that have created the low definition pattern
     \return The PT in GeV/c
  **/
  float getLDPatternPT();
  /**
     \brief Get the number of FD patterns
     \return The number of FD patterns stored in the PatternTrunk
  **/
  int getFDPatternNumber();

  /**
     \brief Set the DC bits of the LD patterns. All FD patterns are removed.
     \param r The number of DC bits used between FD and LD
  **/
  void computeAdaptativePattern(short r);

  /**
     \brief Link the low definition patterns to the detector structure
     \param d The detector
     \param sec The ladders in the sector
     \param modules The modules in the sector (one vector per ladder)
  **/  
  void link(Detector& d, const vector< vector<int> >& sec, const vector<map<int, vector<int> > >& modules);

  /**
     \brief Change the DC bits of the LDP to take the parameter's DC bits into account (used while merging banks)
     \param p A new pattern
  **/
  void updateDCBits(GradedPattern* p);

  /**
     \brief Returns a copy of the active pattern
     \param active_threshold The minimum number of active super strips to activate the pattern
     \return A pointer on the copy
  **/
  GradedPattern* getActivePattern(int active_threshold);

  /**
     \brief Check if the high resolution pattern is already in the bank when DC bits are activated
     \param hp The attern to check
     \result True if the pattern is already in tha bank, false otherwise
   **/
  bool checkPattern(Pattern* hp);

 private:
  GradedPattern* lowDefPattern;
  map<string, GradedPattern*> fullDefPatterns;

  /**
    \brief Compute the DC bits. We use Gray Code to encode the positions :
    0 : 000
    1 : 001
    2 : 011
    3 : 010
    4 : 110
    5 : 111
    6 : 101
    7 : 100
    The values used for the Don't Care bits are :
    0 : 0
    1 : 1
    2 : X (don't care)
    3 : Unused
    For example if the DC bits are 1X0 it means positions 4 and 7.
    \param v List of DC bits, used for recursivity
    \param values List of used strips at full resolution (00110000 if the third and fourth strips are used at full def)
    \param size The number of strips in the previous array. Depends on the number of DC bits used (2 for 1 DC bit, 4 for 2 DC bits, 8 for 3 DC bits, ...)
    \param reverse Used for recursivity. Usefull to know if we are on the first or the second half of the array
  **/
  void computeDCBits(vector<int> &v, bool* values, int size, int reverse);
  void deleteFDPatterns();

  friend class boost::serialization::access;
  
  template<class Archive> void save(Archive & ar, const unsigned int version) const{
    ar << lowDefPattern;;
    ar << fullDefPatterns;
  }
  
  template<class Archive> void load(Archive & ar, const unsigned int version){
    ar >> lowDefPattern;;
    ar >> fullDefPatterns;
  }
  
  BOOST_SERIALIZATION_SPLIT_MEMBER()

};
#endif
