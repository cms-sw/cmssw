#include "Utilities/General/interface/precomputed_value_sort.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/DetSorting.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <iterator>

using namespace geomsort;

typedef Surface::PositionType PositionType;
typedef Surface::RotationType RotationType;

// A fake Det class

class MyDet : public GeomDet {
public:
  MyDet(const PositionType& pos) : GeomDet(new BoundPlane(pos, RotationType(), new RectangularPlaneBounds(0, 0, 0))) {}

  virtual DetId geographicalId() const { return DetId(); }
  std::vector<const GeomDet*> components() const override { return std::vector<const GeomDet*>(); }

  /// Which subdetector
  SubDetector subDetector() const override { return GeomDetEnumerators::DT; }
};

// A simple helper for printing the object
struct dump {
  void operator()(const GeomDet& o) {
    edm::LogVerbatim("CommonTopologies") << o.position() << " R  : " << o.position().perp()
                                         << " Phi: " << o.position().phi() << " Z  : " << o.position().z();
  }
  void operator()(const GeomDet* o) { operator()(*o); }
};

int main() {
  //
  // Example of sorting a vector of objects store by value.
  //

  // Fill the vector to be sorted
  std::vector<MyDet> v;
  v.emplace_back(MyDet(PositionType(2, 1, 1)));
  v.emplace_back(MyDet(PositionType(1, 1, 2)));
  v.emplace_back(MyDet(PositionType(1, 2, 3)));
  v.emplace_back(MyDet(PositionType(2, 2, 4)));

  edm::LogVerbatim("CommonTopologies") << "Original  vector: ";
  for_each(v.begin(), v.end(), dump());

  edm::LogVerbatim("CommonTopologies") << "Sort in R       : ";
  // Here we sort in R
  precomputed_value_sort(v.begin(), v.end(), DetR());
  for_each(v.begin(), v.end(), dump());

  edm::LogVerbatim("CommonTopologies") << "Sort in phi     : ";
  // Here we sort in phi
  precomputed_value_sort(v.begin(), v.end(), DetPhi());
  for_each(v.begin(), v.end(), dump());

  edm::LogVerbatim("CommonTopologies") << "Sort in z       : ";
  // Here we sort in Z
  precomputed_value_sort(v.begin(), v.end(), DetZ());
  for_each(v.begin(), v.end(), dump());

  //
  // Now do the same with a vector of pointers
  //
  edm::LogVerbatim("CommonTopologies") << std::endl << "Again with pointers";

  std::vector<const MyDet*> vp;
  for (std::vector<MyDet>::const_iterator i = v.begin(); i != v.end(); i++) {
    vp.emplace_back(&(*i));
  }

  edm::LogVerbatim("CommonTopologies") << "Sort in R       : ";
  // Here we sort in R
  precomputed_value_sort(vp.begin(), vp.end(), DetR());
  for_each(vp.begin(), vp.end(), dump());

  edm::LogVerbatim("CommonTopologies") << "Sort in phi     : ";
  // Here we sort in phi
  precomputed_value_sort(vp.begin(), vp.end(), DetPhi());
  for_each(vp.begin(), vp.end(), dump());

  edm::LogVerbatim("CommonTopologies") << "Sort in z       : ";
  // Here we sort in Z
  precomputed_value_sort(vp.begin(), vp.end(), DetZ());
  for_each(vp.begin(), vp.end(), dump());

  return 0;
}
