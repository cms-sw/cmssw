#include "CalibFormats/CaloObjects/interface/CaloSamples.h"

CaloSamples::CaloSamples() : id_(), size_(0), presamples_(0) {
  for (int i=0; i<MAXSAMPLES; i++) data_[i]=0;
}

CaloSamples::CaloSamples(const DetId& id, int size) : id_(id), size_(size), presamples_(0) {
  for (int i=0; i<MAXSAMPLES; i++) data_[i]=0;
}

void CaloSamples::setPresamples(int pre) {
  presamples_=pre;
}

CaloSamples& CaloSamples::scale(double value) {
  for (int i=0; i<MAXSAMPLES; i++) data_[i]*=value;
  return (*this);
}

CaloSamples& CaloSamples::operator+=(double value) {
  for (int i=0; i<MAXSAMPLES; i++) data_[i]+=value;
  return (*this);
}

bool 
CaloSamples::isBlank() const // are the samples blank (zero?)
{
   return ( 0.0 != data_[5] ||
	    0.0 != data_[6] ||
	    0.0 != data_[4] ||
	    0.0 != data_[7] ||
	    0.0 != data_[3] ||
	    0.0 != data_[8] ||
	    0.0 != data_[2] ||
	    0.0 != data_[9] ||
	    0.0 != data_[1] ||
	    0.0 != data_[0]    ) ;
}

void 
CaloSamples::setBlank() // keep id, presamples, size but zero out data
{
   std::fill( data_ , data_ + MAXSAMPLES, (double)0.0 ) ;
}

std::ostream& operator<<(std::ostream& s, const CaloSamples& samples) {
  s << "DetId=" << samples.id().rawId();
  s << ", "<<  samples.size() << "samples" << std::endl;
  for (int i=0; i<samples.size(); i++)
    s << i << ":" << samples[i] << std::endl;
  return s;
}
