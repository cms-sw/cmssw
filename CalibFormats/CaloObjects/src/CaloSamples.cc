#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <math.h>
#include <iostream>

CaloSamples::CaloSamples() : id_(), size_(0), presamples_(0), preciseSize_(0), precisePresamples_(0) { setBlank() ; }

CaloSamples::CaloSamples(const DetId& id, int size) :
   id_          ( id   ) , 
   size_        ( size ) , 
   presamples_  ( 0    ) ,
   deltaTprecise_ (0.0f) ,
   preciseSize_(0), 
   precisePresamples_(0) { setBlank() ; }

CaloSamples::CaloSamples(const DetId& id, int size, int presize) :
   id_          ( id   ) , 
   size_        ( size ) , 
   presamples_  ( 0    ) ,
   deltaTprecise_ (0.0f) ,
   preciseSize_(presize), 
   precisePresamples_(0) { setBlank() ; }

// add option to set these later.
void CaloSamples::resetPrecise() {
  preciseData_.resize(preciseSize_,0);
}

void CaloSamples::setPresamples( int pre ) {
   presamples_ = pre ;
}

CaloSamples& CaloSamples::scale( double value ) {
   for (int i=0; i<MAXSAMPLES; i++) data_[i]*=value;
   for (std::vector<float>::iterator j=preciseData_.begin() ; j!=preciseData_.end(); j++)
     (*j)*=value;
   return (*this);
}

CaloSamples& CaloSamples::operator+=(double value) {  
   for (int i=0; i<MAXSAMPLES; i++) data_[i]+=value;
   for (std::vector<float>::iterator j=preciseData_.begin() ; j!=preciseData_.end(); j++)
     (*j)+=value*deltaTprecise_/25.0; // note that the scale is conserved!
  return (*this);
}

CaloSamples& CaloSamples::operator+=(const CaloSamples & other) {
  if(size_ != other.size_ ||
     presamples_ != other.presamples_ ||
     preciseSize_ != other.preciseSize_)  {
    edm::LogError("CaloHitResponse") << "Mismatched calo signals "; 
  }
  int i;
  for(i = 0; i < size_; ++i) {
    data_[i] += other.data_[i];
  }
  if ( preciseData_.size() == 0 && other.preciseData_.size() > 0 ) resetPrecise();
  if ( other.preciseData_.size() > 0 ) {
    for(i = 0; i < preciseSize_; ++i) {
      preciseData_[i] += other.preciseData_[i];
    }
  }
  return *this;
}

CaloSamples &
CaloSamples::offsetTime(double offset)
{
  double data[MAXSAMPLES];
  for( int i ( 0 ) ; i != MAXSAMPLES ; ++i )
  {
    double t = i*25. - offset;
    int firstbin = floor(t/25.);
    double f = t/25. - firstbin;
    int nextbin = firstbin + 1;
    double v1 = (firstbin < 0 || firstbin >= MAXSAMPLES) ? 0. : data_[firstbin];
    double v2 = (nextbin < 0  || nextbin  >= MAXSAMPLES) ? 0. : data_[nextbin];
    data[i] = (v1*(1.-f)+v2*f);
  }
  for( int i ( 0 ) ; i != MAXSAMPLES ; ++i )
  {
    data_[i] = data[i];
  }
  return (*this);
}

bool 
CaloSamples::isBlank() const // are the samples blank (zero?)
{
   for( int i ( 0 ) ; i != MAXSAMPLES ; ++i )
   {
      if( 1.e-6 < fabs( data_[i] ) ) return false ;
   }
   return true ;
}

void 
CaloSamples::setBlank() // keep id, presamples, size but zero out data
{
   std::fill( data_ , data_ + MAXSAMPLES, (double)0.0 ) ;
   std::fill( preciseData_.begin() , preciseData_.end(), (double)0.0 ) ;
}

std::ostream& operator<<(std::ostream& s, const CaloSamples& samples) {
  s << "DetId=" << samples.id();
  // print out every so many precise samples
  float preciseStep = samples.preciseSize()/samples.size();
  s << ", "<<  samples.size() << " samples";
  if (preciseStep > 0) 
    s << ", " << samples.preciseSize() << " preciseSamples" 
      << ", " << preciseStep << " precise step";
  s << '\n';
  for (int i=0; i<samples.size(); i++) {
    s << i << ":" << samples[i];
    int precisei = i*preciseStep;
    if(precisei < samples.preciseSize()) {
      s << " " << samples.preciseAt(precisei) ;
    }
    s << std::endl;
  }
  return s;
}
