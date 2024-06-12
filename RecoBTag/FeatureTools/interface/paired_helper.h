/*#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#endif*/
//#include "external/ExRootAnalysis/ExRootTreeReader.h"
//#include "external/ExRootAnalysis/ExRootResult.h"
#include <cstdlib>
#include <iostream>
#include <functional>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include "TH1F.h"
#include "TH2F.h"
#include "TClonesArray.h"
#include "TTree.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <TROOT.h>
#include <TLorentzVector.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TMath.h>
#include <Rtypes.h>
#include <TString.h>
#include <TRandom.h>
#include <TRandom3.h>
#include "TParticle.h"
#include <vector>

const double pi = TMath::Pi();

double tDeltaPhi(double phi1, double phi2) { return TVector2::Phi_mpi_pi(phi1 - phi2); }

double deltaR(double eta1, double phi1, double eta2, double phi2) {
  double deta = eta1 - eta2;
  double dphi = tDeltaPhi(phi1, phi2);
  return std::hypot(deta, dphi);
}

double shiftpi(double phi, double shift, double lim) {
  if (shift == 0)
    return phi;
  if (shift > 0) {
    if (phi < lim)
      return phi + shift;
  } else if (phi > lim) {
    return phi + shift;
  }
  return phi;
}

bool inEllipse(double jet1_eta, double jet1_phi, double jet2_eta, double jet2_phi, double cand_eta, double cand_phi) {
  float eta1 = static_cast<float>(jet1_eta);
  float phi1 = static_cast<float>(jet1_phi);
  float eta2 = static_cast<float>(jet2_eta);
  float phi2 = static_cast<float>(jet2_phi);
  float eta0 = static_cast<float>(cand_eta);
  float phi0 = static_cast<float>(cand_phi);
  float semimajoradd = 1.0;
  float djet, semimajor, focus, eta_center, phi_center, eta_f1, phi_f1, eta_f2, phi_f2, f1dist, f2dist,
      distsum;  //, phi_m1, phi_m2;
  double semiminor;

  djet = deltaR(eta1, phi1, eta2, phi2);
  semiminor = 1.5;
  semimajor = std::max({semiminor, double(djet / 2 + semimajoradd)});
  focus = pow(pow(semimajor, 2) - pow(semiminor, 2), 0.5);  // Distance of focus to center

  eta_center = (eta1 + eta2) / 2;
  phi_center = TVector2::Phi_mpi_pi(phi1 + TVector2::Phi_mpi_pi(phi2 - phi1) / 2);

  //focus 1
  eta_f1 = eta_center + focus / (djet / 2) * (eta1 - eta_center);
  phi_f1 = TVector2::Phi_mpi_pi(phi_center + focus / (djet / 2) * TVector2::Phi_mpi_pi(phi1 - phi_center));

  //focus 2
  eta_f2 = eta_center + focus / (djet / 2) * (eta2 - eta_center);
  phi_f2 = TVector2::Phi_mpi_pi(phi_center + focus / (djet / 2) * TVector2::Phi_mpi_pi(phi2 - phi_center));

  // Two ends of major axis. This is necesssary to make sure that the point p is not in between the foci on the wrong side of the phi axis
  // phi_m1 = TVector2::Phi_mpi_pi(phi_center + semimajor/(djet/2)  *TVector2::Phi_mpi_pi(phi1-phi_center));
  // phi_m2 = TVector2::Phi_mpi_pi(phi_center + semimajor/(djet/2)  *TVector2::Phi_mpi_pi(phi2-phi_center));

  double shift = 0, lim = 0;
  // if (phi_center > phi_m1 && phi_center > phi_m2) shift = 2*pi;
  // if (phi_center < phi_m1 && phi_center < phi_m2) shift = -2*pi;

  if (phi_center >= 0) {
    shift = 2 * pi;
    lim = phi_center - pi;
  } else {
    shift = -2 * pi;
    lim = phi_center + pi;
  }

  // if (abs(phi1-phi2) > 3.4) cout  << "(" << eta1 << "," << phi1 << "), " << "(" << eta2 << "," << phi2 << "), " << "(" << eta_f1 << "," << phi_f1 << "), "<< "(" << eta_f2 << "," << phi_f2 << "), " << "(" << eta_center << "," << phi_center << ")" << endl;
  float phi0_s = shiftpi(phi0, shift, lim);

  f1dist = std::hypot(eta0 - eta_f1, phi0_s - shiftpi(phi_f1, shift, lim));
  f2dist = std::hypot(eta0 - eta_f2, phi0_s - shiftpi(phi_f2, shift, lim));
  distsum = f1dist + f2dist;

  //if in ellipse, the sum of the distances will be less than 2*semimajor
  if (distsum < 2 * semimajor)
    return true;
  else
    return false;
}