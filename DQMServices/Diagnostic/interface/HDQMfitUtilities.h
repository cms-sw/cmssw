#ifndef DQMServices_HDQMfitUtilities_H
#define DQMServices_HDQMfitUtilities_H

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TF1.h"
#include "TH1.h"
#include "TMath.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/**
   @ class HDQMfitUtilities
   @ fit Landau distributions to historic monitoring elements
   @ fits from Susy's analysis
   (DQM/SiStripHistoricInfoClient/test/TrendsWithFits)
*/

namespace HDQMUtil {
  double langaufun(double *x, double *par);
  int32_t langaupro(double *params, double &maxx, double &FWHM);
  double Gauss(double *x, double *par);
}  // namespace HDQMUtil

class HDQMfitUtilities {
public:
  HDQMfitUtilities();
  ~HDQMfitUtilities();

  void init();
  double doLanGaussFit(MonitorElement *ME) { return doLanGaussFit(ME->getTH1F()); }
  double doLanGaussFit(TH1F *);

  double doGaussFit(MonitorElement *ME) { return doGaussFit(ME->getTH1F()); }
  double doGaussFit(TH1F *);

  double getLanGaussPar(std::string s);
  double getLanGaussParErr(std::string s);
  double getLanGaussConv(std::string s);

  double getGaussPar(std::string s);
  double getGaussParErr(std::string s);

  double getFitChi() { return chi2GausS; }
  int getFitnDof() { return nDofGausS; }

private:
  double pLanGausS[4], epLanGausS[4];
  double pGausS[3], epGausS[3];
  double pLanConv[2];
  double chi2GausS;
  int32_t nDofGausS;
  TF1 *langausFit;
  TF1 *gausFit;
};

#endif  // DQM_SiStripHistoricInfoClient_HDQMfitUtilities_H
