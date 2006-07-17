#include "DataFormats/EcalDigi/interface/EBDataFrame.h"



EBDataFrame::EBDataFrame() : EcalDataFrame()
{
}

EBDataFrame::EBDataFrame(const EBDetId& id) : 
  EcalDataFrame(), id_(id)

{
  // TODO : test id for EcalBarrel or EcalEndcap
}

std::ostream& operator<<(std::ostream& s, const EBDataFrame& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i=0; i<digi.size(); i++) 
    s << "  " << digi.sample(i) << std::endl;
  return s;
}
  
