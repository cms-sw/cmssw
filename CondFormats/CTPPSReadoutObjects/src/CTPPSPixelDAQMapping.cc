/****************************************************************************
 *
 * 
 * Authors: 
 * F.Ferro ferro@ge.infn.it
 *
 ****************************************************************************/

#include "FWCore/Utilities/interface/typelookup.h"

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelDAQMapping.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

std::set<unsigned int> CTPPSPixelDAQMapping::fedIds() const {
  std::set<unsigned int> fedSet;
  for (const auto &p : ROCMapping){
    fedSet.insert(p.first.getFEDId() );
  }
  return fedSet;
}

std::ostream& operator << (std::ostream& s, const CTPPSPixelROCInfo &vi)
{
  s << "ID="<< vi.iD << "  ROC=" << vi.roc;

  return s;
}

//----------------------------------------------------------------------------------------------------

void CTPPSPixelDAQMapping::insert(const CTPPSPixelFramePosition &fp, const CTPPSPixelROCInfo &vi)
{
  auto it = ROCMapping.find(fp);  
  if (it != ROCMapping.end())
    {
      edm::LogError("RPix") << "WARNING in DAQMapping::insert > Overwriting entry at " << fp << ". Previous: " 
	   << "    " << ROCMapping[fp] << ","  << "  new: "  << "    " << vi << ". ";
    }

  ROCMapping[fp] = vi;
}
