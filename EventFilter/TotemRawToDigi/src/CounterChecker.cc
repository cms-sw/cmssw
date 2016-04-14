/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors: 
 *   Maciej Wróbel (wroblisko@gmail.com)
 *   Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#include "EventFilter/TotemRawToDigi/interface/CounterChecker.h"

//-------------------------------------------------------------------------------------------------

using namespace std;

//-------------------------------------------------------------------------------------------------

void CounterChecker::Fill(word counter, TotemFramePosition fr)
{
  pair<CounterMap::iterator, bool> ret;
  
  vector<TotemFramePosition> list;
  list.push_back(fr);
  ret = relationMap.insert(pair<word, vector<TotemFramePosition> >(counter, list));
  if (ret.second == false)
    relationMap[counter].push_back(fr);
}

//-------------------------------------------------------------------------------------------------

void CounterChecker::Analyze(map<TotemFramePosition, TotemVFATStatus> &status, bool error, ostream &es) 
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
      es << "Too few frames to determine the most frequent " << name << " value.";
      return;
  }

  // if there are too few frames with the most frequent value
  if ((float)mostFrequentSize/(float)totalFrames < fraction)
  {
    es << "  The most frequent " << name <<
        " value is doubtful - variance is too high.";
    return;
  }

  for (CounterMap::iterator iter = relationMap.begin(); iter != relationMap.end(); iter++)
  {
    if (iter->first != mostFrequentCounter)
    {
      for (vector<TotemFramePosition>::iterator fr = iter->second.begin(); fr !=  iter->second.end(); fr++)
      {
        if (error)
        {
          if (type == ECChecker) 
            status[*fr].setECProgressError();
          if (type == BCChecker) 
            status[*fr].setBCProgressError();    
        }

        es << "  Frame at " << *fr << ": " << name << " number " <<
            iter->first << " is different from the most frequent one " <<
            mostFrequentCounter << endl;
      }
    }
  }
}
