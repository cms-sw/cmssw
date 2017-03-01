#ifndef TrackReco_DeDxData_h
#define TrackReco_DeDxData_h

#include "DataFormats/Common/interface/ValueMap.h"

namespace reco
{

class DeDxData
{

public:
    DeDxData();
    DeDxData(float val, float er, unsigned int num);
    virtual ~DeDxData();
    float dEdx() const;
    float dEdxError() const;
    int numberOfSaturatedMeasurements() const;
    unsigned int numberOfMeasurements() const;

private:
    float value_;
    float error_;
    unsigned int numberOfMeasurements_;
};



//Association Track -> float estimator
typedef std::vector<reco::DeDxData>   DeDxDataCollection;
typedef edm::ValueMap<reco::DeDxData> DeDxDataValueMap;


// //Association Track -> float estimator
//typedef  edm::AssociationVector<reco::TrackRefProd,std::vector<DeDxData> >  DeDxDataCollection;
//typedef  DeDxDataCollection::value_type DeDxData;
//typedef  edm::Ref<DeDxDataCollection> DeDxDataRef;
//typedef  edm::RefProd<DeDxDataCollection> DeDxDataRefProd;
//typedef  edm::RefVector<DeDxDataCollection> DeDxDataRefVector;

} // namespace reco
#endif

