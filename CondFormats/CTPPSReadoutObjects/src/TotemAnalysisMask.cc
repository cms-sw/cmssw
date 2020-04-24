/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Maciej Wróbel (wroblisko@gmail.com)
*   Jan Kašpar (jan.kaspar@cern.ch)
*
****************************************************************************/

#include "FWCore/Utilities/interface/typelookup.h"

#include "CondFormats/CTPPSReadoutObjects/interface/TotemAnalysisMask.h"

//----------------------------------------------------------------------------------------------------

void TotemAnalysisMask::insert(const TotemSymbID &sid, const TotemVFATAnalysisMask &vam)
{
  analysisMask[sid] = vam;
}

//----------------------------------------------------------------------------------------------------

TYPELOOKUP_DATA_REG(TotemAnalysisMask);
