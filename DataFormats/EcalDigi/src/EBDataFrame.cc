#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<iostream>

float EBDataFrame::spikeEstimator() const
{
        if ( size() != 10 ) {
                edm::LogError("InvalidNumberOfSamples") << "This method only applies to signals sampled 10 times ("
                        << size() << " samples found)";
                return 10.;
        }
        // skip faulty channels
        if ( sample(5).adc() == 0 ) return 10.;
        size_t imax = 0;
        int maxAdc = 0;
        for ( int i = 0; i < size(); ++i ) {
                if ( sample(i).adc() > maxAdc ) {
                        imax = i;
                        maxAdc = sample(i).adc();
                }
        }
        // skip early signals
        if ( imax < 4 ) return 10.;
        float ped = 1./3. * (sample(0).adc() + sample(1).adc() + sample(2).adc());
        return 0.18*(sample(4).adc()-ped)/(sample(5).adc()-ped) + (sample(6).adc()-ped)/(sample(5).adc()-ped);
}


std::ostream& operator<<(std::ostream& s, const EBDataFrame& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i=0; i<digi.size(); i++) 
    s << "  " << digi[i] << std::endl;
  return s;
}
