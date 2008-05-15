#ifndef TruncatedAverageDeDxEstimator_h
#define TruncatedAverageDeDxEstimator_h

#include <numeric>

class TruncatedAverageDeDxEstimator: public BaseDeDxEstimator
{
public: 
 TruncatedAverageDeDxEstimator(float fraction): m_fraction(fraction) {}

 virtual Measurement1D  dedx(std::vector<Measurement1D> ChargeMeasurements){
    if(ChargeMeasurements.size()<=0) return 0;

    std::sort(ChargeMeasurements.begin(), ChargeMeasurements.end(), LessFunc() );

    int     nTrunc = int( ChargeMeasurements.size()*m_fraction);
    double sumdedx = 0;
    for(unsigned int i=0;i + nTrunc <  ChargeMeasurements.size() ; i++){
       sumdedx+=ChargeMeasurements[i].value();
    } 
    return Measurement1D( sumdedx/(ChargeMeasurements.size()-nTrunc) , 0 );
 } 

private:
 float m_fraction;

};

#endif
