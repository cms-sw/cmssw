/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors: 
 *   Maciej Wróbel (wroblisko@gmail.com)
 *   Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#ifndef CounterChecker_h
#define CounterChecker_h

#include <map>
#include <string>
#include <vector>
#include <iostream>

#include "DataFormats/TotemRawData/interface/TotemFramePosition.h"
#include "DataFormats/TotemRawData/interface/TotemRawToDigiStatus.h"

//----------------------------------------------------------------------------------------------------

/**
 *\brief Class for finding the most popular both EC and BC counter, and filling the conversion
 * status 'wrong EC/BC number' for frames with different value.
 */
class CounterChecker 
{
 public:
  typedef unsigned short word;
  typedef std::map<word, std::vector<TotemFramePosition> > CounterMap;
  enum CheckerType {BCChecker, ECChecker};
  
 private:
  class Comparer
  {
    public:
      typedef unsigned short word;
      bool operator()(const std::pair<word, std::vector<TotemFramePosition> > &a, const std::pair<word, std::vector<TotemFramePosition> > &b)
      {
        return a.second.size() < b.second.size();
      }
  };  

  /// counter value -> list of frames with this value
  CounterMap relationMap;

  /// EC or BC counter checker
  CheckerType type;

  /// the name of this check, used in error messages
  std::string name;

  /// minimal required number of frames to search for the most frequent one
  unsigned int min;
  
  /// the most frequent value is accepted only provided its sub-sample size is greater than the
  /// specified fraction of the full sample
  double fraction;
  
  /// if >= 3, the information about wrong counter for each frame will be shown
  unsigned int verbosity;
  
 public:
  /**
   * \param t: CounterChecker::ECCounter or CounterChecker::BCCounter. On that, depends whether
   * checker will fill wrong EC or BC rogress error.
   * \param name: name
   * \param min: minimal required number of frames to search for the most frequent one
   */
  CounterChecker(CheckerType _type = CounterChecker::BCChecker, const std::string &_name="",
    unsigned int _min=0, double _fraction=0., unsigned int _verbosity=0) : type(_type),
    name(_name), min(_min), fraction(_fraction), verbosity(_verbosity) {}

  /// add new value to map, counter takes value of EC or BC number
  void Fill(word counter, TotemFramePosition fr);

  /// summarizes and fill the status (wrong EC and BC progress error for some frames)
  void Analyze(TotemRawToDigiStatus &status, bool error, std::ostream &es);
};


#endif
