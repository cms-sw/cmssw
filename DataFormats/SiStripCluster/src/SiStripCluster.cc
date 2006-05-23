#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/StripDigi.h"

SiStripCluster::SiStripCluster( unsigned int detid, const StripDigiRange& range) :
  detId_(detid), firstStrip_(range.first->strip())
{
  amplitudes_.reserve( range.second - range.first);
  int sumx = 0;
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
    suma += amp;
  }
  // strip centers are offcet by half pitch w.r.t. strip numbers,
  // so one has to add 0.5 to get the correct barycenter position
  barycenter_ = sumx / static_cast<float>(suma) + 0.5;
}

