#ifndef CMGTools_TTHAnalysis_DistributionRemapper_h
#define CMGTools_TTHAnalysis_DistributionRemapper_h

#include <vector>
#include <Math/Interpolator.h>
struct TH1;

class DistributionRemapper {
    public:
        DistributionRemapper() : interp_(0) {} // for persistency
        DistributionRemapper(const TH1 *source, const TH1 *target) ;
        ~DistributionRemapper() ;
        double operator()(double x) const { return Eval(x); }
        double Eval(double x) const ;
    private:
        void init() const;

        double xmin_, ymin_, xmax_, ymax_;
        std::vector<double> x_, y_;
        mutable ROOT::Math::Interpolator *interp_;  //! not to be serialized    
};

#endif
