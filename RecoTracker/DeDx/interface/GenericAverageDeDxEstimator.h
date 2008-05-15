#ifndef GenericAverageDeDxEstimator_h
#define GenericAverageDeDxEstimator_h


class GenericAverageDeDxEstimator: public BaseDeDxEstimator
{
public: 
 GenericAverageDeDxEstimator(float expo): m_expo(expo) {}

 virtual Measurement1D  dedx(std::vector<Measurement1D> ChargeMeasurements){ 
    size_t n = ChargeMeasurements.size();
    if(ChargeMeasurements.size()<=0)return 0;

    double result = 0;
    for(size_t i = 0; i< n; i ++){
       result += pow(ChargeMeasurements[i].value(),m_expo);
    }

    return Measurement1D( result , 0 );
 } 

private:
 float m_expo;

};

#endif
