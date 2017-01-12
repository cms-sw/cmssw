/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Maciej Wróbel (wroblisko@gmail.com)
*
****************************************************************************/

#include "FWCore/Utilities/interface/typelookup.h"

#include "CondFormats/TotemReadoutObjects/interface/TotemDAQMapping.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

std::ostream& operator << (std::ostream& s, const TotemVFATInfo &vi)
{
  if (vi.type == TotemVFATInfo::data)
    s << "type=data, ";
  else
    s << "type=  CC, ";

  s << vi.symbolicID << ", hw id=0x" << hex << vi.hwID << dec;

  return s;
}

//----------------------------------------------------------------------------------------------------

void TotemDAQMapping::insert(const TotemFramePosition &fp, const TotemVFATInfo &vi)
{
  auto it = VFATMapping.find(fp);  
  if (it != VFATMapping.end())
  {
    cerr << "WARNING in DAQMapping::Insert > Overwriting entry at " << fp << ". Previous: " << endl 
      << "    " << VFATMapping[fp] << "," << endl << "  new: " << endl << "    " << vi << ". " << endl;
  }

  VFATMapping[fp] = vi;
}

//----------------------------------------------------------------------------------------------------

TYPELOOKUP_DATA_REG(TotemDAQMapping);
