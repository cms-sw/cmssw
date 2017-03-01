/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors: 
 *   Maciej Wróbel (wroblisko@gmail.com)
 *   Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#include "EventFilter/CTPPSRawToDigi/interface/CounterChecker.h"

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
