/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Maciej Wróbel (wroblisko@gmail.com)
*   Jan Kašpar (jan.kaspar@cern.ch)
*
****************************************************************************/

#ifndef CondFormats_CTPPSReadoutObjects_TotemDAQMapping
#define CondFormats_CTPPSReadoutObjects_TotemDAQMapping

#include "CondFormats/CTPPSReadoutObjects/interface/TotemFramePosition.h"

#include "CondFormats/CTPPSReadoutObjects/interface/TotemSymbId.h"

#include <map>

//----------------------------------------------------------------------------------------------------

/**
 *\brief Contains mappind data related to a VFAT.
 */
class TotemVFATInfo
{
  public:
    /// the symbolic id
    TotemSymbID symbolicID;

    /// the hardware ID (16 bit)
    unsigned int hwID;
    
    friend std::ostream& operator << (std::ostream& s, const TotemVFATInfo &fp);
};

//----------------------------------------------------------------------------------------------------

/**
 *\brief The mapping between FramePosition and VFATInfo.
 */
class TotemDAQMapping
{
  public:
    std::map<TotemFramePosition, TotemVFATInfo> VFATMapping;
    
    void insert(const TotemFramePosition &fp, const TotemVFATInfo &vi);
};

#endif
