#ifndef RECOCALOTOOLS_SELECTORS_CALOCONESELECTOR_H
#define RECOCALOTOOLS_SELECTORS_CALOCONESELECTOR_H 1

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include <memory>
#include <functional>

/** \class CaloConeSelector
  *  
  * \author J. Mans - Minnesota
  */

template <class T>
class CaloConeSelector {
public:
  CaloConeSelector(double dR, const CaloGeometry* geom)
      : geom_(geom), deltaR_(dR), detector_(DetId::Detector(0)), subdet_(0) {}

  CaloConeSelector(double dR, const CaloGeometry* geom, DetId::Detector detector, int subdet = 0)
      : geom_(geom), deltaR_(dR), detector_(detector), subdet_(subdet) {}

  void inline selectCallback(double eta,
                             double phi,
                             const edm::SortedCollection<T>& inputCollection,
                             std::function<void(const T&)> callback) {
    GlobalPoint p(GlobalPoint::Cylindrical(1, phi, tanh(eta)));
    return selectCallback(p, inputCollection, callback);
  }

  void inline selectCallback(const GlobalPoint& p,
                             const edm::SortedCollection<T>& inputCollection,
                             std::function<void(const T&)> callback) {
    // TODO: handle default setting of detector_ (loops over subdet)
    // TODO: heuristics of when it is better to loop over inputCollection instead (small # hits)
    for (int subdet = subdet_; subdet <= 7 && (subdet_ == 0 || subdet_ == subdet); subdet++) {
      const CaloSubdetectorGeometry* sdg = geom_->getSubdetectorGeometry(detector_, subdet);
      if (sdg != nullptr) {
        // get the list of detids within range (from geometry)
        CaloSubdetectorGeometry::DetIdSet dis = sdg->getCells(p, deltaR_);
        // loop over detids...
        typename edm::SortedCollection<T>::const_iterator j, je = inputCollection.end();

        for (CaloSubdetectorGeometry::DetIdSet::iterator i = dis.begin(); i != dis.end(); i++) {
          if (i->subdetId() != subdet)
            continue;  // possible for HCAL where the same geometry object handles all the detectors
          j = inputCollection.find(*i);
          if (j != je)
            callback(*j);
        }
      }
    }
  }

private:
  const CaloGeometry* geom_;
  double deltaR_;
  DetId::Detector detector_;
  int subdet_;
};

#endif
