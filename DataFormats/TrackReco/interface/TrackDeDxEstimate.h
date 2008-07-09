#ifndef TrackReco_TrackDeDxEstimate_h
#define TrackReco_TrackDeDxEstimate_h
//#include "DataFormats/TrackReco/interface/TrackFwd.h"
//#include "DataFormats/Common/interface/AssociationVector.h"
//#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
//#include <vector>


namespace reco {

  class TrackDeDxEstimate{

    TrackDeDxEstimate();
    TrackDeDxEstimate(float val, float er, unsigned int num );
    ~TrackDeDxEstimate();

    float dEdx() const;
    float dEdxError() const;
    unsigned int numberOfMeasurements() const;


  private:

    float value_;
    float error_;
    unsigned int numberOfMeasurements_;

  }

// //Association Track -> float estimator
// typedef  edm::AssociationVector<reco::TrackRefProd,std::vector<Measurement1D> >  TrackDeDxEstimateCollection;
// typedef  TrackDeDxEstimateCollection::value_type TrackDeDxEstimate;
// typedef  edm::Ref<TrackDeDxEstimateCollection> TrackDeDxEstimateRef;
// typedef  edm::RefProd<TrackDeDxEstimateCollection> TrackDeDxEstimateRefProd;
// typedef  edm::RefVector<TrackDeDxEstimateCollection> TrackDeDxEstimateRefVector;
}
#endif
