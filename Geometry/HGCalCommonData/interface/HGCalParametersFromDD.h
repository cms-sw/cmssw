#ifndef HGCalCommonData_HGCalParametersFromDD_h
#define HGCalCommonData_HGCalParametersFromDD_h

#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

class HGCalParameters;

class HGCalParametersFromDD {
public:
  HGCalParametersFromDD() = default;
  virtual ~HGCalParametersFromDD() = default;

  bool build(const DDCompactView* cpv,
             HGCalParameters& php,
             const std::string& name,
             const std::string& namew,
             const std::string& namec,
             const std::string& namet);
  bool build(const cms::DDCompactView* cpv,
             HGCalParameters& php,
             const std::string& name,
             const std::string& namew,
             const std::string& namec,
             const std::string& namet,
             const std::string& name2);

private:
  void getCellPosition(HGCalParameters& php, int type);
  double getDDDValue(const char* s, const DDsvalues_type& sv);
  std::vector<double> getDDDArray(const char* s, const DDsvalues_type& sv);
  constexpr static double tan30deg_ = 0.5773502693;
};

#endif
