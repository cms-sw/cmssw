#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<iostream>

float EBDataFrame::spikeEstimator() const
{
        if ( size() != 10 ) {
                edm::LogError("InvalidNumberOfSamples") << "This method only applies to signals sampled 10 times ("
                        << size() << " samples found)";
        }
        return 0.18*sample(4).adc()/sample(5).adc() + sample(6).adc()/sample(5).adc();
}


std::ostream& operator<<(std::ostream& s, const EBDataFrame& digi) {
  s << digi.id() << " " << digi.size() << " samples " << std::endl;
  for (int i=0; i<digi.size(); i++) 
    s << "  " << digi[i] << std::endl;
  return s;
}
