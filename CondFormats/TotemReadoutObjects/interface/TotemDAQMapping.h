/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Maciej Wróbel (wroblisko@gmail.com)
*   Jan Kašpar (jan.kaspar@cern.ch)
*
****************************************************************************/

#ifndef _TotemDAQMapping_h_
#define _TotemDAQMapping_h_

#include "DataFormats/TotemRawData/interface/TotemFramePosition.h"

#include "CondFormats/TotemReadoutObjects/interface/TotemSymbId.h"

#include <map>

//----------------------------------------------------------------------------------------------------

/**
 *\brief Contains mappind data related to a VFAT.
 */
class TotemVFATInfo
{
  public:
    /// is data of coincidence-chip VFAT
    enum {data, CC} type;

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
    
    void Insert(const TotemFramePosition &fp, const TotemVFATInfo &vi);
};

#endif
