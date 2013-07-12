#ifndef CalibMuon_DTCalibration_DTResidualFitter_h
#define CalibMuon_DTCalibration_DTResidualFitter_h

/*
 *  $Date: 2010/11/18 21:38:41 $
 *  $Revision: 1.2 $
 *  \author A. Vilela Pereira
 */

class TH1F;

struct DTResidualFitResult {
public:
   DTResidualFitResult(double mean, double meanErr, double sigma, double sigmaErr): fitMean(mean), 
                                                                                    fitMeanError(meanErr),
                                                                                    fitSigma(sigma),
                                                                                    fitSigmaError(sigmaErr) {} 

   double fitMean;
   double fitMeanError;
   double fitSigma;
   double fitSigmaError;
};

class DTResidualFitter {
public:
   DTResidualFitter(bool debug = false);
   ~DTResidualFitter();

   DTResidualFitResult fitResiduals(TH1F& histo, int nSigmas = 1);

private:
   bool debug_;
};
#endif
