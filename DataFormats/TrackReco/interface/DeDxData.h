#ifndef TrackReco_DeDxData_h
#define TrackReco_DeDxData_h

#include "DataFormats/Common/interface/ValueMap.h"

namespace reco {
  namespace io_v1 {

    class DeDxData {
    public:
      DeDxData();
      DeDxData(float val, int nsat, unsigned int num);
      DeDxData(float val, float er, int sat, unsigned int num);
      virtual ~DeDxData();
      float dEdx() const;
      float dEdxError() const;
      int numberOfSaturatedMeasurements() const;
      unsigned int numberOfMeasurements() const;

    private:
      float value_;
      float error_;
      unsigned int numberOfMeasurements_;
      int numberOfSatMeasurements_;
    };

  }  // namespace io_v1
  using DeDxData = io_v1::DeDxData;

  //Association Track -> float estimator
  typedef std::vector<DeDxData> DeDxDataCollection;
  typedef edm::ValueMap<DeDxData> DeDxDataValueMap;

  // //Association Track -> float estimator
  //typedef  edm::AssociationVector<reco::TrackRefProd,std::vector<DeDxData> >  DeDxDataCollection;
  //typedef  DeDxDataCollection::value_type DeDxData;
  //typedef  edm::Ref<DeDxDataCollection> DeDxDataRef;
  //typedef  edm::RefProd<DeDxDataCollection> DeDxDataRefProd;
  //typedef  edm::RefVector<DeDxDataCollection> DeDxDataRefVector;

}  // namespace reco
#endif
