#ifndef Geometry_MuonNUmbering_MuonOffsetFromDD_h
#define Geometry_MuonNUmbering_MuonOffsetFromDD_h

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
  MuonOffsetFromDD(std::vector<std::string> names);

  bool build(const DDCompactView*, MuonOffsetMap&);
  bool build(const cms::DDCompactView*, MuonOffsetMap&);

private:
  bool debugParameters(const MuonOffsetMap&);
  int getNumber(const std::string&, const DDsvalues_type&);
  const std::vector<std::string> specpars_;
  const unsigned int nset_;
};

#endif
