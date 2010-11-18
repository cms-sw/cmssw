
/*
 *  Fits core distribution to single gaussian; iterates once.  
 *
 *  $Date: 2010/11/18 21:00:11 $
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTResidualFitter.h"

#include "TH1F.h"
#include "TF1.h"
#include "TString.h"

DTResidualFitter::DTResidualFitter() {}

DTResidualFitter::~DTResidualFitter() {}

DTResidualFitResult DTResidualFitter::fitResiduals(TH1F& histo, int nSigmas){
  
   float minFit = histo.GetMean() - histo.GetRMS();
   float maxFit = histo.GetMean() + histo.GetRMS();

   TString funcName = TString(histo.GetName()) + "_gaus";
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
