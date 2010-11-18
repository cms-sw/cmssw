
/*
 *  Fits core distribution to single gaussian; iterates once.  
 *
 *  $Date: 2010/11/17 17:54:23 $
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTResidualFitter.h"

#include "TH1F.h"
#include "TString.h"

DTResidualFitter::DTResidualFitter() {}

DTResidualFitter::~DTResidualFitter() {}

DTResidualFitResult DTResidualFitter::fitResiduals(TH1F const& histo, int nSigmas){
  
   float minFit = histo.GetMean() - histo.GetRMS();
   float maxFit = histo.GetMean() + histo.GetRMS();

   TString funcName(histo.GetName());
   funcName += "_gaus"
   TF1* fitFunc = new TF1(funcName,"gaus",minFit,maxFit);

   histo.Fit(fitFunc,"RQ");

   minFit = fitFunc->GetParameter(1) - nSigmas*fitFunc->GetParameter(2);
   maxFit = fitFunc->GetParameter(1) + nSigmas*fitFunc->GetParameter(2);
   fitFunc->SetRange(minFit,maxFit);
   histo.Fit(fitFunc,"RQ");

   return DTResidualFitResult( fitFunc->GetParameter(1),
                               fitFunc->GetParError(1),
                               fitFunc->GetParameter(2),
                               fitFunc->GetParError(2) ); 
} 
