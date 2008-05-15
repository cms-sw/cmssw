#ifndef MedianDeDxEstimator_h
#define MedianDeDxEstimator_h


class MedianDeDxEstimator: public BaseDeDxEstimator
{
public: 
 MedianDeDxEstimator(float expo) {}

 virtual Measurement1D  dedx(std::vector<Measurement1D> ChargeMeasurements){

    if(ChargeMeasurements.size()<=0)return 0;

    std::sort(ChargeMeasurements.begin(), ChargeMeasurements.end(), LessFunc() );
    return Measurement1D( ChargeMeasurements[ChargeMeasurements.size()/2].value() , 0 ); 
 }

 
};

#endif
