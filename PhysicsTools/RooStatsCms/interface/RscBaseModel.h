#ifndef __RscBaseModel__
#define __RscBaseModel__

/// RscBaseModel: defines a distribution for a component of the analysis

/**
\class RscBaseModel
$Revision: 1.2 $
$Date: 2009/02/17 14:19:17 $
\author G. Schott (gregory.Schott<at>cern.ch), Danilo Piparo - Universitaet Karlsruhe

The distribution is defined in the datacard. An example for a Gaussian 
distribution might be:
\verbatim
[test_sig_var1]
    model = gauss
    test_sig_var1_mean = 100 C
    test_sig_var1_sigma = 30 C
\endverbatim

And for a flat distribution:
\verbatim
[test_bkg_var1]
    comment = bkg model for variable var1
    model = flat
\endverbatim


At the moment the supported distributions are:
- Gaussian (model name \e gauss , variables mean-sigma)
- Double Gaussian (model name \e dblgauss , variables mean1-sigma1-mean2-sigma2-frac)
- Sum of 4 Gaussian (model name \e fourGaussians , variables mean1-sigma1-mean2-sigma2-mean3-sigma3-mean4-sigma4-frac1-frac2-frac3)
- Breit-Wigner (model name \e BreitWigner , variables mean1-sigma1)
- Exponential (model name \e Exponential , variables slope )
- Polynomial, 7th degree (model name \e poly7, variables coef_0,coef_1,.. assumed to be 0 if not set)
- Flat distribution (poly 0), (model name \e flat , no parameters needed )
- Histogram (model name \e histo , no parameters needed )
- Counting experiment (model name \e yieldonly , no parameters needed)

This last option allows to use a TH1 to describe the distribution: very useful 
in the case where no parametrisation is at our disposal.\n
To use this feature:\n
\verbatim
[qqhtt_bkg_mh]
    model = histo
    qqhtt_bkg_mh_fileName = ptdr_qqhtt_histos.root
    qqhtt_bkg_mh_dataName = qqhtt_bkg_histo

\endverbatim
It is also possible to consider a mere number counting experiment. In this case 
the model name will be \e yieldonly. This allows what we call an \b "hybrid 
\b approach" to combination, i.e. treating in the same way number counting 
experiments and situation where a continuous distribution is at our disposal.
**/


#include <iostream>

#include "RooCategory.h"
#include "RooStringVar.h"

#include "PhysicsTools/RooStatsCms/interface/RscAbsPdfBuilder.h"

class RscBaseModel : public RscAbsPdfBuilder {

public:
  RscBaseModel(TString theName, RooRealVar& theVar, RooArgSet* discVars=0);
 
  ~RscBaseModel();
  
  /// The discriminating variable
  RooRealVar* x; // discriminating variable

  // gauss
  /// Gaussian / BreitWigner distribution parameter
  RooRealVar* mean;
  /// Gaussian distribution parameter
  RooRealVar* sigma;

  // dblgauss
  /// Double gaussian distribution parameter
  RooRealVar* mean1;
  /// Double gaussian distribution parameter
  RooRealVar* mean2;
  /// Double gaussian distribution parameter
  RooRealVar* sigma1;
  /// Double gaussian distribution parameter
  RooRealVar* sigma2;
  /// Double gaussian distribution parameter
  RooRealVar* frac;

  /// 4 gaussian distribution parameter
  RooRealVar* mean3;
  /// 4 gaussian distribution parameter
  RooRealVar* mean4;
  /// 4 gaussian distribution parameter
  RooRealVar* sigma3;
  /// 4 gaussian distribution parameter
  RooRealVar* sigma4;
  /// 4 gaussian distribution parameter
  RooRealVar* frac1;
  /// 4 gaussian distribution parameter
  RooRealVar* frac2;
  /// 4 gaussian distribution parameter
  RooRealVar* frac3;

  // exponential
  /// Exponential distribution parameter
  RooRealVar* slope;

  // Poly7
  /// Polynomial distribution parameter
  RooRealVar* coef_0;
  /// Polynomial distribution parameter
  RooRealVar* coef_1;
  /// Polynomial distribution parameter
  RooRealVar* coef_2;
  /// Polynomial distribution parameter
  RooRealVar* coef_3;
  /// Polynomial distribution parameter
  RooRealVar* coef_4;
  /// Polynomial distribution parameter
  RooRealVar* coef_5;
  /// Polynomial distribution parameter
  RooRealVar* coef_6;
  /// Polynomial distribution parameter
  RooRealVar* coef_7;

  // BifurGaus
  RooRealVar* sigmaL;
  RooRealVar* sigmaR;

  // CBShape+Gaussian
  RooRealVar* m0;
  RooRealVar* alpha;
  RooRealVar* n;
  RooRealVar* gmean;
  RooRealVar* gsigma;

  // BreitWigner
  /// Gaussian / BreitWigner distribution parameter
  RooRealVar* width;

  // histo
  /// Root file name of the stored histogram
  RooStringVar fileName;
  /// Name of the histogram
  RooStringVar dataName;

  RooCategory model;

  void readDataCard();

  void writeDataCard(ostream& out);
  
private:
  
  /// Name of the model  
  TString _name;

  void buildPdf();

  #ifndef SWIG
  //ClassDef(RscBaseModel,1)
  #endif /*SWIG */
};

#endif
