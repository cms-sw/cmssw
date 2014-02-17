
/*
 *  Fits core distribution to single gaussian; iterates once.  
 *
 *  $Date: 2010/11/19 14:02:09 $
 *  $Revision: 1.3 $
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/interface/DTResidualFitter.h"

#include "TH1F.h"
#include "TF1.h"
#include "TString.h"

DTResidualFitter::DTResidualFitter(bool debug):debug_(debug) {}

DTResidualFitter::~DTResidualFitter() {}

DTResidualFitResult DTResidualFitter::fitResiduals(TH1F& histo, int nSigmas){
  
   TString option("R");
   if(!debug_) option += "Q";
 
   float minFit = histo.GetMean() - histo.GetRMS();
   float maxFit = histo.GetMean() + histo.GetRMS();

   TString funcName = TString(histo.GetName()) + "_gaus";
   TF1* fitFunc = new TF1(funcName,"gaus",minFit,maxFit);

   histo.Fit(fitFunc,option);

   minFit = fitFunc->GetParameter(1) - nSigmas*fitFunc->GetParameter(2);
   maxFit = fitFunc->GetParameter(1) + nSigmas*fitFunc->GetParameter(2);
   fitFunc->SetRange(minFit,maxFit);
   histo.Fit(fitFunc,option);

   return DTResidualFitResult( fitFunc->GetParameter(1),
                               fitFunc->GetParError(1),
                               fitFunc->GetParameter(2),
                               fitFunc->GetParError(2) ); 
} 
