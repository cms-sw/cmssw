#include "DataFormats/EcalDigi/interface/EEDataFrame.h"

EEDataFrame::EEDataFrame() : EcalDataFrame()
{
}

EEDataFrame::EEDataFrame(const EEDetId& id) : 
  EcalDataFrame(), id_(id)
{
  // TODO : test id for EcalBarrel or EcalEndcap
}

std::ostream& operator<<(std::ostream& s, const EEDataFrame& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i=0; i<digi.size(); i++) 
    s << "  " << digi.sample(i) << std::endl;
  return s;
}
