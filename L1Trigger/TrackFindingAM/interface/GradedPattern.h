#ifndef _GRADEDPATTERN_H_
#define _GRADEDPATTERN_H_

#include "Pattern.h"

#include <boost/serialization/base_object.hpp>

using namespace std;

/**
   \brief A Pattern wih a grade (the number of tracks corresponding to this pattern).
   Can also contain (optionnaly) the average Pt of the tracks
**/

class GradedPattern : public Pattern{
 public:
  /**
     \brief Constructor
  **/
  GradedPattern();
  /**
     \brief Copy Constructor
  **/
  GradedPattern(const Pattern& p);
  /**
     \brief Get the grade of the Pattern
     \return The number of tracks having generated the pattern
  **/
  int getGrade() const;
  /**
     \brief Get the average Pt of the tracks having generated the pattern (if used)
     \return The average Pt
  **/
  float getAveragePt() const;
  /**
     Increment the grade (tracks occurences + 1)
  **/
  void increment();
  /**
     Increment the grade (tracks occurences + 1) and add a Pt value to the average Pt
     @param pt The Pt value of the last track
  **/
  void increment(float pt);
  /**
     \brief Allows to compare 2 patterns on their grade
     \param gp The second pattern
     \return -1 if the pattern has a lower grade
  **/
  int operator<(const GradedPattern& gp);

 private:
  int grade;
  float averagePt;

  friend class boost::serialization::access;
  
  template<class Archive> void save(Archive & ar, const unsigned int version) const{
    ar << boost::serialization::base_object<Pattern>(*this);
    ar << grade;
    ar << averagePt;
  }
  
  template<class Archive> void load(Archive & ar, const unsigned int version){
    ar >> boost::serialization::base_object<Pattern>(*this);
    ar >> grade;
    ar >> averagePt;
  }
  
  BOOST_SERIALIZATION_SPLIT_MEMBER()

};
#endif
