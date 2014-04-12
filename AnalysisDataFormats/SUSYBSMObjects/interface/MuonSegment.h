#ifndef MuonSegment_H
#define MuonSegment_H
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <vector>

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"

namespace susybsm {

 class MuonSegment
 {
 public:
   MuonSegment() {};
   //bool isDT()  const { return DT;}
   //bool isCSC() const { return CSC;}

   //void setDT(const bool type) {DT = type;}
   //void setCSC(const bool type) {CSC = type;}

   void setDTSegmentRef(const DTRecSegment4DRef segment) { DTSegmentRef_ = segment;}
   void setCSCSegmentRef(const CSCSegmentRef segment) { CSCSegmentRef_ = segment;}

   void setGP(const GlobalPoint point) { gp=point;}

   GlobalPoint getGP() const {return gp;}

   DTRecSegment4DRef getDTSegmentRef() const {return DTSegmentRef_;}
   CSCSegmentRef getCSCSegmentRef() const {return CSCSegmentRef_;}

 private:
   GlobalPoint gp;

   DTRecSegment4DRef DTSegmentRef_;
   CSCSegmentRef CSCSegmentRef_;
 };

  typedef  std::vector<MuonSegment> MuonSegmentCollection;
  typedef  edm::Ref<MuonSegmentCollection> MuonSegmentRef;
  typedef  edm::RefProd<MuonSegmentCollection> MuonSegmentRefProd;
  typedef  edm::RefVector<MuonSegmentCollection> MuonSegmentRefVector;
}

#endif
