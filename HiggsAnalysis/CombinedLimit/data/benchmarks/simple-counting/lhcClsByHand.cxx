/// Compile me with the command
/// $ gcc -lstdc++ $(root-config --cflags --ldflags --libs) -lm lhcClsByHand.cxx -o lhcClsByHand.exe
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <TRandom3.h>

/// Observed events
int    Nobs = 4;
/// Expected background
double B0    = 3.26;
/// Log-normal systematic uncertainty
double kappa = 1.2353;
/// Is the uncertainty on the background or on the signal? 
bool uncertainty_on_b = true;

double nll(int Nobs, double r, double S, double S_In) {
    // Poisson term
    double Nexp = (uncertainty_on_b ? r + B0*pow(kappa,S) : r*pow(kappa,S) + B0);
    double logPois = (Nobs == 0 ? 0 : Nobs*log(Nexp)) - Nexp; // the log(N!) cancels anyway with the denominator
    double logGaus = -0.5*(S - S_In)*(S - S_In);
    return - (logPois + logGaus);
}

double minimizeFixedR(int Nobs, double r, double &S, double S_In, double tolerance=0.001) {
    S = S_In; double S_step = 0.5; 
    double y0 = nll(Nobs, r, S, S_In); 
    double y1u = nll(Nobs, r, S + 0.5*S_step, S_In);
    double y1d = nll(Nobs, r, S - 0.5*S_step, S_In);
    while (S_step > tolerance) {
        if (y1d > y0 && y1u > y0) {
            S_step *= 0.5;
            y1u = nll(Nobs, r, S + 0.5*S_step, S_In);
            y1d = nll(Nobs, r, S - 0.5*S_step, S_In);
        } else if (y1d < y0) { 
            y1u = y0;
            y0  = y1d; 
            S -= 0.5*S_step;
            y1d = nll(Nobs, r, S - 0.5*S_step, S_In);
        } else {
            y1d = y0;
            y0  = y1u;
            S += 0.5*S_step;
            y1u = nll(Nobs, r, S + 0.5*S_step, S_In);
        }
        //std::cout << "     for r = " << r << ", S = " << S << " +/- " << S_step << ", y0 = " << y0 << std::endl;
    }
    //std::cout << " for r = " << r << ", the minimum is " << y0 << " at S = " << S << std::endl;
    return y0;
}

double minimizeFloatR(int Nobs, double &r, double &S, double S_In, double tolerance=0.001) {
    double y0 = minimizeFixedR(Nobs, r, S, S_In);
    double r_step = 2.0, r_max = r;
    double ru = r + 0.5*r_step; if (ru > r_max) ru = r_max;
    double rd = r - 0.5*r_step; if (rd <     0) rd = 0;
    double y1u = minimizeFixedR(Nobs, ru, S, S_In);
    double y1d = minimizeFixedR(Nobs, rd, S, S_In);
    while (r_step > tolerance) {
        assert(r >= 0 && r <= r_max);
        if (y1d > y0 && y1u > y0) {
            //std::cout << "  decrease step size" << std::endl;
            r_step *= 0.5;
            ru = r + 0.5*r_step; if (ru > r_max) ru = r_max;
            rd = r - 0.5*r_step; if (ru <     0) rd = 0;
            y1u = minimizeFixedR(Nobs, ru, S, S_In);
            y1d = minimizeFixedR(Nobs, rd, S, S_In);
        } else if (y1d < y0) {
            y1u = y0;  ru  = r;
            if (rd == 0) {
                //std::cout << "  hit low edge" << std::endl;
                r = r_step = 0.5*r;
                y0 = minimizeFixedR(Nobs, r, S, S_In);
            } else {
                //std::cout << "  step down"  << std::endl;
                y0  = y1d; r = rd;
                rd  = r - r_step; if (rd <     0) rd = 0;
                y1d = minimizeFixedR(Nobs, rd, S, S_In);;
            }
        } else /*if (y1u < y0)*/ {
            y1d = y0; rd = r;
            if (ru == r_max) {
                //std::cout << "  hit high edge" << std::endl;
                r_step = 0.5*(r_max - r);
                r      = 0.5*(r_max + r);
                y0 = minimizeFixedR(Nobs, r, S, S_In);
            } else {
                //std::cout << "  step up"  << std::endl;
                y0  = y1u; r = ru;
                ru  = r + r_step; if (ru > r_max) ru = r_max;
                y1u = minimizeFixedR(Nobs, ru, S, S_In);;
            }
        }
    }
    return y0;
}


void toyByHand(double r0 = 1.0, int toys=500, int seed=1) {
    TRandom *rand = new TRandom3(seed);
    std::cout << "Computing LHC-CLs by hand" << std::endl;
    std::cout << "Observed events: " << Nobs << std::endl;
    std::cout << "Expected background: " << B0 << std::endl;
    std::cout << "Log-normal uncertainty on " << (uncertainty_on_b ? "background" : "signal") << ", kappa: " << kappa << std::endl;
    std::cout << "Computing p-values for the hypothesis of signal r = " << r0 << std::endl;

    double r = r0; 
    double S = 0; 
    double S_In = 0;
    double nll_S = minimizeFixedR(Nobs, r0, S, S_In);
    double S_fitS = S;
    std::cout << "Minimum for fixed r = " << r0 <<": theta = " << S_fitS << std::endl;
    double nll_B = minimizeFixedR(Nobs, 0.0, S, S_In);
    double S_fitB = S; // this is trivial, no need to fit
    std::cout << "Minimum for fixed r = 0.0: theta = " << S_fitB << std::endl;
    r = r0;
    double nll_F = minimizeFloatR(Nobs, r, S, S_In);
    std::cout << "Minimum for floating r: theta = " << S << ", r = " << r << std::endl;

    double qData = (nll_S - nll_F);
    std::cout << "Test statistics on data: " << qData << std::endl;

    // Compute CLsb
    int Nabove = 0, Nbelow = 0;
    for (int i = 0; i < toys; ++i) {
        // Reset nuisances
        S = S_fitS;
        r = r0;
        S_In = rand->Gaus(S, 1.0);
        int Ntoy = rand->Poisson(uncertainty_on_b ? r+B0*pow(kappa,S) : r*pow(kappa,S)+B0);
        double toyNll_S = minimizeFixedR(Ntoy, r0, S, S_In);
        double toyNll_F = minimizeFloatR(Ntoy,  r, S, S_In);
        double qToy = (toyNll_S - toyNll_F);
        //std::cout << "q toy (" << i << ") = " << qToy << std::endl;
        if (qToy <= qData) Nbelow++; else Nabove++;
    }
    double CLsb = Nabove/double(Nabove+Nbelow), CLsbError = sqrt(CLsb*(1.-CLsb)/double(Nabove+Nbelow));
    std::cout << "CLsb = " << Nabove << "/" << Nabove+Nbelow << " = " << CLsb << " +/- " << CLsbError << std::endl;

    // Compute CLb
    int Nabove2 = 0, Nbelow2 = 0;
    for (int i = 0; i < toys/4; ++i) {
        // Reset nuisances
        S = S_fitB;
        r = 0.0;
        S_In = rand->Gaus(S, 1.0);
        int Ntoy = rand->Poisson(uncertainty_on_b ? r+B0*pow(kappa,S) : r*pow(kappa,S)+B0);
        r = r0;
        double toyNll_S = minimizeFixedR(Ntoy,  r0, S, S_In);
        double toyNll_F = minimizeFloatR(Ntoy,   r, S, S_In);
        double qToy = (toyNll_S - toyNll_F);
        //std::cout << "q toy (" << i << ") = " << qToy << std::endl;
        if (qToy <= qData) Nbelow2++; else Nabove2++;
    }
    double CLb = Nabove2/double(Nabove2+Nbelow2), CLbError = sqrt(CLb*(1.-CLb)/double(Nabove2+Nbelow2));
    std::cout << "CLb  = " << Nabove2 << "/" << Nabove2+Nbelow2 << " = " << CLb << " +/- " << CLbError << std::endl;

    std::cout << "CLs  = "<< CLsb/CLb << " +/- " << hypot(CLsbError/CLb, CLsb*CLbError/CLb/CLb) << std::endl;
}

int main(int argc, char **argv) {
   toyByHand(atof(argc > 1 ? argv[1] : "1.0"),
             atoi(argc > 2 ? argv[2] : "500"), 
             atoi(argc > 3 ? argv[3] : "1"));
   return 0; 
}
