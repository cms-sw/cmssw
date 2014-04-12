#include "SeedPtFunction.h"
SeedPtFunction::SeedPtFunction(){
  
  
}

SeedPtFunction::~SeedPtFunction(){
 
}

Double_t SeedPtFunction::fitf(Double_t *x, Double_t *par) {
         Double_t theX = 10. + x[0];
         Double_t fitval =  par[0]
                         + (par[1]/  theX )
                         + (par[2]/ ( theX*theX ) )
                         + (par[3]/ ( theX*theX*theX ) ) ;
         return fitval;
}

Double_t SeedPtFunction::fitf2(Double_t *x, Double_t *par) {
         Double_t fitval =  par[0]
                         + (par[1] * x[0] )
                         + (par[2] / x[0] ) ;
         return fitval;
}

Double_t SeedPtFunction::linear(Double_t *x, Double_t *par) {
         Double_t fitval2 =  par[0]
                          + (par[1]* x[0] )
                          + (par[2]* x[0]*x[0]  )
                          + (par[3]* x[0]*x[0]*x[0] );
         return fitval2;
}

Double_t SeedPtFunction::fgaus(Double_t *x, Double_t *par) {

     Double_t gs_Value = TMath::Gaus(x[0],par[1],par[2]) ;
     Double_t fitV = par[0]*gs_Value ;
     return fitV;
}

// sigma : sigma of the data set w.r.t mean
// deviation : the deviation of data and mean/prefit value
bool SeedPtFunction::DataRejection(double sigma, double deviation, int N_data ) {

    bool reject = false ;
    /// gaussian probability for data point
    double p_gaus = 0.0;
    double k = 0.0;
    for (int i=0; i != 10000; i++ ) {
        k += ( deviation*0.0001) ;
        double n1 = 1.0/ (sigma*sqrt(2.0*3.14159)) ;
        double x2 = (-1.0*k*k)/(2.0*sigma*sigma) ;
        double gaus1 = n1*exp(x2);
        p_gaus += (gaus1*deviation*0.0001);
    }
    /// expected number outside the deviation of the distribution
    double nExpected = (1.0-(p_gaus*2.0))*(N_data*1.0);

    if ( nExpected < 0.99 ) reject = true;

    return reject;
}
