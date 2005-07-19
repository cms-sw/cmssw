#include "DataFormats/HcalDigi/interface/HFDataFrame.h"

HFDataFrame::HFDataFrame() : id_(0), 
				 size_(0),
				 hcalPresamples_(0) {
}

HFDataFrame::HFDataFrame(const HcalDetId& id) : 
  id_(id), 
  size_(0),
  hcalPresamples_(0) {
  // TODO : test id for HcalForward
}

void HFDataFrame::setSize(int size) {
  if (size>MAXSAMPLES) size_=MAXSAMPLES;
  else if (size<=0) size_=0;
  else size_=size;
}
void HFDataFrame::setPresamples(int ps) {
  if (ps>size_) hcalPresamples_=size_;
  else hcalPresamples_=ps;
}
void HFDataFrame::setReadoutIds(const HcalElectronicsId& eid) {
  electronicsId_=eid;
}
std::ostream& operator<<(std::ostream& s, const HFDataFrame& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i=0; i<digi.size(); i++) 
    s << "  " << digi.sample(i) << std::endl;
  return s;
}
