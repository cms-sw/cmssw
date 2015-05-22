#ifndef TTHAnalysis_plotter_smearer_h
#define TTHAnalysis_plotter_smearer_h
#include <TRandom3.h>
#include <cmath>


extern TRandom *gSmearer_;
inline double smearMC(double x, double mu, double sigma) { 
    return x + gSmearer_->Gaus(mu,sigma); 
}
inline double logSmearMC(double x, double mu, double sigma) { 
    return std::exp(std::log(x) + gSmearer_->Gaus(mu,sigma)); 
}
inline double shiftMC(double x, double delta) { 
    return x + delta; 
}
inline double scaleShiftMC(double x, double scale, double shift) { 
    return x*scale + shift; 
}

#endif
