#include "DataFormats/EcalDigi/interface/ESDataFrame.h"

ESDataFrame::ESDataFrame() : id_(0), 
			     size_(0),
			     data_(MAXSAMPLES)
{
}

ESDataFrame::ESDataFrame(const ESDetId& id) : 
  id_(id), 
  size_(0),
  data_(MAXSAMPLES)
{
}

void ESDataFrame::setSize(const int& size) {
  if (size > MAXSAMPLES) size_ = MAXSAMPLES;
  else if (size <= 0) size_=0;
  else size_ = size;
}

std::ostream& operator<<(std::ostream& s, const ESDataFrame& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i=0; i<digi.size(); i++) 
    s << "  " << digi.sample(i) << std::endl;
  return s;
}
