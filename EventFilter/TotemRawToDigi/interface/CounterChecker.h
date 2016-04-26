/****************************************************************************
 *
 * This is a part of the TOTEM offline software.
 * Authors: 
 *   Maciej Wróbel (wroblisko@gmail.com)
 *   Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#ifndef EventFilter_TotemRawToDigi_CounterChecker
#define EventFilter_TotemRawToDigi_CounterChecker

#include <map>
#include <string>
#include <vector>
#include <iostream>

#include "CondFormats/TotemReadoutObjects/interface/TotemFramePosition.h"

#include "DataFormats/TotemDigi/interface/TotemVFATStatus.h"

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
    template<typename T>
    void Analyze(T &status, bool error, std::ostream &es);
  
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
  
    /// level of verbosity
    unsigned int verbosity;
};

//-------------------------------------------------------------------------------------------------

template<typename T>
void CounterChecker::Analyze(T &status, bool error, std::ostream &es) 
{
  word mostFrequentCounter = 0;
  word mostFrequentSize = 0;
  unsigned int totalFrames = 0;

  // finding the most frequent counter
  for (CounterMap::iterator iter = relationMap.begin(); iter != relationMap.end(); iter++)
  {
    unsigned int iterSize = iter->second.size();
    totalFrames += iterSize;

    if (iterSize > mostFrequentSize)
    {
      mostFrequentCounter = iter->first;
      mostFrequentSize = iter->second.size();
    }
  }

  if (totalFrames < min)
  {
    if (verbosity > 0)
      es << "Too few frames to determine the most frequent " << name << " value.";

    return;
  }

  // if there are too few frames with the most frequent value
  if ((float)mostFrequentSize/(float)totalFrames < fraction)
  {
    if (verbosity > 0)
      es << "  The most frequent " << name << " value is doubtful - variance is too high.";

    return;
  }

  for (CounterMap::iterator iter = relationMap.begin(); iter != relationMap.end(); iter++)
  {
    if (iter->first != mostFrequentCounter)
    {
      for (std::vector<TotemFramePosition>::iterator fr = iter->second.begin(); fr !=  iter->second.end(); fr++)
      {
        if (error)
        {
          if (type == ECChecker) 
            status[*fr].status.setECProgressError();
          if (type == BCChecker) 
            status[*fr].status.setBCProgressError();    
        }

        if (verbosity > 0)
          es << "  Frame at " << *fr << ": " << name << " number " << iter->first
            << " is different from the most frequent one " << mostFrequentCounter << std::endl;
      }
    }
  }
}

#endif
