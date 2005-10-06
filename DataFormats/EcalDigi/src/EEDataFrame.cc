#include "DataFormats/EcalDigi/interface/EEDataFrame.h"



EEDataFrame::EEDataFrame() : id_(0), 
			     size_(0),
			     data_(MAXSAMPLES)
{
}

EEDataFrame::EEDataFrame(const EEDetId& id) : 
  id_(id), 
  size_(0),
  data_(MAXSAMPLES)
{
  // TODO : test id for EcalBarrel or EcalEndcap
}

void EEDataFrame::setSize(int size) {
  if (size>MAXSAMPLES) size_=MAXSAMPLES;
  else if (size<=0) size_=0;
  else size_=size;
}

std::ostream& operator<<(std::ostream& s, const EEDataFrame& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i=0; i<digi.size(); i++) 
    s << "  " << digi.sample(i) << std::endl;
  return s;
}
