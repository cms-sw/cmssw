/****************************************************************************
 * File AcceptanceTableHelper.cc
 *
 * Access to acceptance tables stored in root histograms
 *
 * Author: Dmitry Zaborov
 *
 * Version: $Id: AcceptanceTableHelper.cc,v 1.1 2008/11/25 17:34:15 beaudett Exp $
 ***************************************************************************/

#include "FastSimulation/ForwardDetectors/plugins/AcceptanceTableHelper.h"

#include <iostream>
#include <math.h>

/** Read from root file <f> acceptance tables named <basename> and <basename>_hight */

void AcceptanceTableHelper::Init(TFile& f, const std::string basename)
{
  // ... read table for low t
  TH3F *h = (TH3F*)f.Get(basename.c_str());

  if (h != NULL)
  {
    h_log10t_log10Xi_Phi = (TH3F*)h->Clone();
    std::cout << "Read ";
    h_log10t_log10Xi_Phi->SetDirectory(0); // secure it from deleting if the file is eventually closed
    h_log10t_log10Xi_Phi->Print();
  } else {
    std::cout << "Warning: could not get acceptance table " << basename << std::endl;
  }
  
  // ... read table for high t
  std::string name2 = basename+"_hight";
  h = (TH3F*)f.Get(name2.c_str());

  if (h != NULL)
  {
    h_t_log10Xi_Phi = (TH3F*)h->Clone();
    h_t_log10Xi_Phi->SetDirectory(0); // secure it from deleting if the file is eventually closed
    std::cout << "Read ";
    h_t_log10Xi_Phi->Print();
  } else {
    std::cout << "Warning: could not get acceptance table " << name2 << std::endl;
  }
}

/** Return acceptance for given t, xi, phi */

float AcceptanceTableHelper::GetAcceptance(float t, float xi, float phi) {

  float log10t  = log10(-t);
  float log10Xi = log10(xi);

  float acc = 0;

  if ((h_log10t_log10Xi_Phi != NULL)				  // if table exists
       && (log10t < h_log10t_log10Xi_Phi->GetXaxis()->GetXmax())) // and t within table range
  {
  
    float log10tMin = h_log10t_log10Xi_Phi->GetXaxis()->GetXmin();
    if (log10t < log10tMin) log10t = log10tMin; // very small t should go to the lowest t bin
    
    acc = h_log10t_log10Xi_Phi->GetBinContent(h_log10t_log10Xi_Phi->FindBin(log10t, log10Xi, phi));

  } else if (h_t_log10Xi_Phi != NULL) { // if table exists for high t

     acc = h_t_log10Xi_Phi->GetBinContent(h_t_log10Xi_Phi->FindBin(-t, log10Xi, phi));
  }
  
  return acc;
}
