/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com) 
*    
****************************************************************************/


#include "EventFilter/CTPPSRawToDigi/interface/VFATFrameCollection.h"

//----------------------------------------------------------------------------------------------------
    
const VFATFrame* VFATFrameCollection::GetFrameByIndexID(TotemFramePosition index, unsigned int ID)
{
  const VFATFrame* returnframe = GetFrameByIndex(index);
  if (returnframe == NULL)
    return NULL;
  return (returnframe->getChipID() == (ID & 0xFFF)) ? returnframe : NULL;
}
