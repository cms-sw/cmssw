#ifndef MTDGeometryBuilder_MTDParametersFromDD_h
#define MTDGeometryBuilder_MTDParametersFromDD_h

#include <vector>
#include "CondFormats/GeometryObjects/interface/PMTDParameters.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DDCompactView;
namespace cms {
  class DDCompactView;
}
class PMTDParameters;

class MTDParametersFromDD {
public:
  MTDParametersFromDD() {}
  virtual ~MTDParametersFromDD() {}

  bool build(const DDCompactView*, PMTDParameters&);
  bool build(const cms::DDCompactView*, PMTDParameters&);

private:
  void putOne(int, std::vector<int>&, PMTDParameters&);
};

#endif
