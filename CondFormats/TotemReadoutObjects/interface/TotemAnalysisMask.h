/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Maciej Wróbel (wroblisko@gmail.com)
*   Jan Kašpar (jan.kaspar@cern.ch)
*
****************************************************************************/

#ifndef _TotemAnalysisMask_h_
#define _TotemAnalysisMask_h_

#include "CondFormats/TotemReadoutObjects/interface/TotemSymbId.h"

#include <set>
#include <map>

//----------------------------------------------------------------------------------------------------

/**
 *\brief Contains data on masked channels of a VFAT.
 */
class TotemVFATAnalysisMask
{
  public:
    TotemVFATAnalysisMask() : fullMask(false) {}

    /// whether all channels of the VFAT shall be masked
    bool fullMask;

    /// list of channels to be masked
    std::set<unsigned char> maskedChannels;
};

//----------------------------------------------------------------------------------------------------

/**
 *\brief Channel-mask mapping.
 **/
class TotemAnalysisMask
{
  public:
    std::map<TotemSymbID, TotemVFATAnalysisMask> analysisMask;

    void Insert(const TotemSymbID &sid, const TotemVFATAnalysisMask &vam);
};

#endif
