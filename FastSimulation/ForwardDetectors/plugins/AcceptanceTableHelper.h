#ifndef FastSim_ProtonTaggerAcceptanceHelper_H
#define FastSim_ProtonTaggerAcceptanceHelper_H

/// Access to acceptance tables stored in ROOT file

/**
 * Helper class to get actual values from acceptance tables stored as TH3F.
 * The class can provide acceptance values for a near-beam detector 
 * or a combination (e.g. 420+220).
 *
 * A class instance is initialized from a root file with TH3F-type histograms of acceptance:
 * for  low t: acceptance as function of log10(-t), log10(xi) and phi;
 * for high t: acceptance as function of -t, log10(xi) and phi.
 *
 * The acceptance can be extracted from tables for given t,xi,phi with a dedicated method.
 *
 * Author: Dmitry Zaborov
 */

// Version: $Id: AcceptanceTableHelper.h,v 1.1 2007/09/28 08:39:53 dzaborov Exp $

#include "TFile.h"
#include "TH3F.h"

#include <string>

class AcceptanceTableHelper
{
public:

  /// Default constructor
  AcceptanceTableHelper() : h_log10t_log10Xi_Phi(nullptr),h_t_log10Xi_Phi(nullptr) {;}
  
  /// Delete acceptance histograms
  ~AcceptanceTableHelper() { 
    if (h_log10t_log10Xi_Phi) delete h_log10t_log10Xi_Phi;
    if (h_t_log10Xi_Phi) delete h_t_log10Xi_Phi;
  }

  /// Get acceptance tables from root file
  void Init(TFile&, const std::string);

  /// Acceptance as a function of t, xi and phi
  float GetAcceptance(float, float, float);
  
private:

  /// Table for  low t: acceptance as a function of log10(t), log10(Xi) and Phi
  TH3F *h_log10t_log10Xi_Phi;

  /// Table for high t: acceptance as a function of -t, log10(Xi) and Phi
  TH3F *h_t_log10Xi_Phi;

};

#endif
