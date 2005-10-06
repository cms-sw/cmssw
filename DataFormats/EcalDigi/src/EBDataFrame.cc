#include "DataFormats/EcalDigi/interface/EBDataFrame.h"



EBDataFrame::EBDataFrame() : id_(0), 
			     size_(0),
			     data_(MAXSAMPLES)
{
}

EBDataFrame::EBDataFrame(const EBDetId& id) : 
  id_(id), 
  size_(0),
  data_(MAXSAMPLES)
{
  // TODO : test id for EcalBarrel or EcalEndcap
}

void EBDataFrame::setSize(int size) {
  if (size>MAXSAMPLES) size_=MAXSAMPLES;
  else if (size<=0) size_=0;
  else size_=size;
}

std::ostream& operator<<(std::ostream& s, const EBDataFrame& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i=0; i<digi.size(); i++) 
    s << "  " << digi.sample(i) << std::endl;
  return s;
}
  
