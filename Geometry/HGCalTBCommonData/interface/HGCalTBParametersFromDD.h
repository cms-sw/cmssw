#ifndef HGCalCommonData_HGCalTBParametersFromDD_h
#define HGCalCommonData_HGCalTBParametersFromDD_h

#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

class HGCalTBParameters;

class HGCalTBParametersFromDD {
public:
  HGCalTBParametersFromDD() = default;
  virtual ~HGCalTBParametersFromDD() = default;

  bool build(const DDCompactView* cpv,
             HGCalTBParameters& php,
             const std::string& name,
             const std::string& namew,
             const std::string& namec,
             const std::string& namet);
  bool build(const cms::DDCompactView* cpv,
             HGCalTBParameters& php,
             const std::string& name,
             const std::string& namew,
             const std::string& namec,
             const std::string& namet,
             const std::string& name2);

private:
  double getDDDValue(const char* s, const DDsvalues_type& sv);
  std::vector<double> getDDDArray(const char* s, const DDsvalues_type& sv);
  constexpr static double tan30deg_ = 0.5773502693;
};

#endif
