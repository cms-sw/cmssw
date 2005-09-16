#include <cmath>
#include <iostream>

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/StripDigi.h"

SiStripCluster::SiStripCluster( unsigned int detid, const StripDigiRange& range) :
  detId_(detid), firstStrip_(range.first->strip())
{
  amplitudes_.reserve( range.second - range.first);
  int sumx = 0;
  int sumx2 = 0;
  int suma = 0;

  int lastStrip = -1;;
  for (StripDigiIter i=range.first; i!=range.second; i++) {

    /// check if digis consecutive
    if (lastStrip>0 && i->strip() != lastStrip + 1) {
      for (int j=0; j < i->strip()-(lastStrip+1); j++) {
	amplitudes_.push_back( 0);
      }
    }
    lastStrip = i->strip();

    short amp = i->adc();       // FIXME: gain correction here
    amplitudes_.push_back( amp);
    sumx += i->strip()*amp;
    sumx2 += i->strip()*i->strip()*amp;
    suma += amp;
  }
  // strip centers are offset by half pitch w.r.t. strip numbers,
  // so one has to add 0.5 to get the correct barycenter position
  float temp_barycenter = sumx / static_cast<float>(suma);
  barycenter_ =  temp_barycenter + 0.5;
  barycenter_error_ = sumx2 / static_cast<float>(suma) - (temp_barycenter * temp_barycenter);
  // error is given squared
//   barycenter_error_ = sqrt(barycenter_error_);
  if ( amplitudes_.size() == 1 ) {
    // FIXME: if only one amplitude forms the cluster, set the error to a strip
    barycenter_error_ = 1.0;
  }
  if ( barycenter_error_ == 0 ) {
    std::cout << "still zero error" << std::endl;
  }
}

