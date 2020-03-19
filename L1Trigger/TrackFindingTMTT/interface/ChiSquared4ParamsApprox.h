#ifndef __CHI_SQUARED_4_PARAMS_APPROX__
#define __CHI_SQUARED_4_PARAMS_APPROX__
 
#include "L1Trigger/TrackFindingTMTT/interface/L1ChiSquared.h"

namespace TMTT {
 
class ChiSquared4ParamsApprox : public L1ChiSquared{
 
public:
    ChiSquared4ParamsApprox(const Settings* settings, const uint nPar);
 
    ~ChiSquared4ParamsApprox(){}
 
protected:
    std::vector<double> seed(const L1track3D& l1track3D);
    std::vector<double> residuals(std::vector<double> x);
    Matrix<double> D(std::vector<double> x);
    Matrix<double> Vinv();
    std::map<std::string, double> convertParams(std::vector<double> x);
 
private:
    std::vector<double> mapToVec(std::map<std::string, double> x);
    std::map<std::string, double> vecToMap(std::vector<double> x);
};

}
 
#endif
