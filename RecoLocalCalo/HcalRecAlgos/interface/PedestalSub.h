#ifndef RecoLocalCalo_HcalRecAlgos_PedestalSub_h
#define RecoLocalCalo_HcalRecAlgos_PedestalSub_h 1

#include <typeinfo>
#include <vector>

class PedestalSub
{
 public:

  PedestalSub();
  ~PedestalSub();
  
  void init(int runCond, float threshold, float quantile);
  
  void calculate(const std::vector<double> & inputCharge, const std::vector<double> & inputPedestal, std::vector<double> & corrCharge) const;
  
  double getCorrection(const std::vector<double> & inputCharge, const std::vector<double> & inputPedestal) const;

  
 private:
  float fThreshold;
  float fQuantile;
  float fCondition;
  
};

#endif 
