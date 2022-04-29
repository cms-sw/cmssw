#ifndef Geometry_HGCalCommonData_HGCalCassette_h
#define Geometry_HGCalCommonData_HGCalCassette_h

#include <cmath>
#include <cstdint>
#include <vector>

class HGCalCassette {
public:
  HGCalCassette(int cassette, const std::vector<double>& shifts) { setParameter(cassette, shifts); }
  HGCalCassette() {}

  void setParameter(int cassette, const std::vector<double>& shifts);
  std::pair<double, double> getShift(int layer, int zside, int cassette);

private:
  const std::vector<int> positEE_ = {2, 1, 0, 5, 4, 3};
  const std::vector<int> positHE_ = {5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6};
  int cassette_;
  bool typeHE_;
  std::vector<double> shifts_;
};

#endif
