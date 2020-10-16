#ifndef MuonNUmbering_MuonOffsetFromDD_h
#define MuonNUmbering_MuonOffsetFromDD_h

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>
#include <vector>

class DDFilteredView;
class MuonOffsetMap;

class MuonOffsetFromDD {
public:
  MuonOffsetFromDD() = default;
  virtual ~MuonOffsetFromDD() {}

  bool build(const DDCompactView*, MuonOffsetMap&);
  bool build(const cms::DDCompactView*, MuonOffsetMap&);

private:
  bool buildParameters(const MuonOffsetMap&);
  int getNumber(const std::string&, const DDsvalues_type&);
  static constexpr int nset_ = 51;
};

#endif
