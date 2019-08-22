#ifndef RecoLocalCalo_HcalRecAlgos_PedestalSub_h
#define RecoLocalCalo_HcalRecAlgos_PedestalSub_h 1

#include <typeinfo>
#include <vector>

class PedestalSub {
public:
  PedestalSub();
  ~PedestalSub();

  void calculate(const std::vector<double>& inputCharge,
                 const std::vector<double>& inputPedestal,
                 const std::vector<double>& inputNoise,
                 std::vector<double>& corrCharge,
                 int soi,
                 int nSample) const;

  double getCorrection(const std::vector<double>& inputCharge,
                       const std::vector<double>& inputPedestal,
                       const std::vector<double>& inputNoise,
                       int soi,
                       int nSample) const;
};

#endif
