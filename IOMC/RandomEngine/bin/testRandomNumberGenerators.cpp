#include<iostream>
#include<iomanip>

#include "TROOT.h"
#include "TFile.h"
#include "TH1D.h"

#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/JamesRandom.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandPoisson.h"
#include "CLHEP/Random/RandExponential.h"
#include "CLHEP/Random/RandGamma.h"
#include "CLHEP/Random/RandLandau.h"

#include "G4Poisson.hh"
#include "Randomize.hh"

#include "TRandom3.h"

int main() {

  //  edm::MessageDrop::instance()->debugEnabled = false;

  int seed1 = 123456789;
  unsigned int seed2 = 234567891;

  double randomNumber;

  int ncycle = 1000000000;
  int nfreq = 10000000;

  // define histograms for distributions:

  TH1D* histoCLHEPflat        = new TH1D("histoCLHEPflat","CLHEP::RandFlat",100,0.,1.);
  TH1D* histoCLHEPgaussQ      = new TH1D("histoCLHEPgaussQ","CLHEP::RandGaussQ",200,-10.,10.);
  TH1D* histoCLHEPpoissonQ    = new TH1D("histoCLHEPpoissonQ","CLHEP::RandPoissonQ",40,0.,20.);
  TH1D* histoCLHEPpoisson     = new TH1D("histoCLHEPpoisson","CLHEP::RandPoisson",40,0.,20.);
  TH1D* histoCLHEPexponential = new TH1D("histoCLHEPexponential","CLHEP::RandExponential",200,0.,20.);
  TH1D* histoCLHEPgamma       = new TH1D("histoCLHEPgamma","CLHEP::RandGamma",200,0.,20.);
  TH1D* histoCLHEPlandau      = new TH1D("histoCLHEPlandau","CLHEP::RandLandau",400,-10.,390.);

  TH1D* histoROOTrndm         = new TH1D("histoROOTrndm","TRandom3::Rndm",100,0.,1.);
  TH1D* histoROOTgaus         = new TH1D("histoROOTgaus","TRandom3::Gaus",200,-10.,10.);

  TH1D* histoG4Poisson        = new TH1D("histoG4Poisson","G4Poisson",40,0.,20.);
  TH1D* histoG4UniformRand    = new TH1D("histoG4UniformRand","G4UniformRand",100,0.,1.);

  // CLHEP HepJamesRandom engine

  CLHEP::HepJamesRandom engine(seed1);

  // Test of CLHEP flat distribution
  
  CLHEP::RandFlat theFlat(engine);
  
  for (int i = 0; i < ncycle; ++i) {
    randomNumber = theFlat.fire();
    histoCLHEPflat->Fill(randomNumber);
    if (i%nfreq == 1) std::cout << "Flat random number #i           " << std::setw(12) << i << " = " << std::fixed << std::setw(22) << std::setprecision(14) << randomNumber << std::endl;
  }

  // Test of CLHEP GaussQ distribution

  CLHEP::RandGaussQ theGaussQ(engine);

  for (int i = 0; i < ncycle; ++i) {
    randomNumber = theGaussQ.fire();
    histoCLHEPgaussQ->Fill(randomNumber);
    if (i%nfreq == 1) std::cout << "GaussQ random number #i         " << std::setw(12) << i << " = " << std::fixed << std::setw(22) << std::setprecision(14) << randomNumber << std::endl;
  }

  // Test of CLHEP PoissonQ distribution

  CLHEP::RandPoissonQ thePoissonQ(engine);

  for (int i = 0; i < ncycle; ++i) {
    randomNumber = thePoissonQ.fire();
    histoCLHEPpoissonQ->Fill(randomNumber);
    if (i%nfreq == 1) std::cout << "PoissonQ random number #i       " << std::setw(12) << i << " = " << std::fixed << std::setw(22) << std::setprecision(14) << randomNumber << std::endl;
  }

  // Test of CLHEP Poisson distribution

  CLHEP::RandPoisson thePoisson(engine);

  for (int i = 0; i < ncycle; ++i) {
    randomNumber = thePoisson.fire();
    histoCLHEPpoisson->Fill(randomNumber);
    if (i%nfreq == 1) std::cout << "Poisson random number #i        " << std::setw(12) << i << " = " << std::fixed << std::setw(22) << std::setprecision(14) << randomNumber << std::endl;
  }

  // Test of CLHEP Exponential distribution

  CLHEP::RandExponential theExponential(engine);

  for (int i = 0; i < ncycle; ++i) {
    randomNumber = theExponential.fire();
    histoCLHEPexponential->Fill(randomNumber);
    if (i%nfreq == 1) std::cout << "Exponential random number #i    " << std::setw(12) << i << " = " << std::fixed << std::setw(22) << std::setprecision(14) << randomNumber << std::endl;
  }

  // Test of CLHEP Gamma distribution

  CLHEP::RandGamma theGamma(engine);

  for (int i = 0; i < ncycle; ++i) {
    randomNumber = theGamma.fire();
    histoCLHEPgamma->Fill(randomNumber);
    if (i%nfreq == 1) std::cout << "Gamma random number #i          " << std::setw(12) << i << " = " << std::fixed << std::setw(22) << std::setprecision(14) << randomNumber << std::endl;
  }

  // Test of CLHEP Landau distribution

  CLHEP::RandLandau theLandau(engine);

  for (int i = 0; i < ncycle; ++i) {
    randomNumber = theLandau.fire();
    histoCLHEPlandau->Fill(randomNumber);
    if (i%nfreq == 1) std::cout << "Landau random number #i         " << std::setw(12) << i << " = " << std::fixed << std::setw(22) << std::setprecision(14) << randomNumber << std::endl;
  }

  TRandom3* rootEngine = new TRandom3(seed2); 

  // Test of TRandom3 flat
  
  for (int i = 0; i < ncycle; ++i) {
    randomNumber = rootEngine->Rndm();
    histoROOTrndm->Fill(randomNumber);
    if (i%nfreq == 1) std::cout << "TRandom3 flat random number #i  " << std::setw(12) << i << " = " << std::fixed << std::setw(22) << std::setprecision(14) << randomNumber << std::endl;
  }
  
  // Test of TRandom3 Gauss
  
  for (int i = 0; i < ncycle; ++i) {
    randomNumber = rootEngine->Gaus();
    histoROOTgaus->Fill(randomNumber);
    if (i%nfreq == 1) std::cout << "TRandom3 Gauss random number #i " << std::setw(12) << i << " = " << std::fixed << std::setw(22) << std::setprecision(14) << randomNumber << std::endl;
  }

  // Test of G4Poisson

  int ave = 1;
  for (int i = 0; i < ncycle; ++i) {
    randomNumber = G4Poisson(ave);
    histoG4Poisson->Fill(randomNumber);
    if (i%nfreq == 1) std::cout << "G4Poisson random number #i " << std::setw(12) << i << " = " << std::fixed << std::setw(22) << std::setprecision(14) << randomNumber << std::endl;              
  }

  // Test of G4UniformRand

  for (int i = 0; i < ncycle; ++i) {
    randomNumber = G4UniformRand();
    histoG4UniformRand->Fill(randomNumber);
    if (i%nfreq == 1) std::cout << "G4UniformRand random number #i " << std::setw(12) << i << " = " << std::fixed << std::setw(22) << std::setprecision(14) << randomNumber << std::endl;              
  }

  // Write out files 
 
  TFile * file = new TFile("testRandomNumberGenerators.root","RECREATE");
  histoCLHEPflat->Write();
  histoCLHEPgaussQ->Write();
  histoCLHEPpoissonQ->Write();
  histoCLHEPpoisson->Write();
  histoCLHEPexponential->Write();
  histoCLHEPgamma->Write();
  histoCLHEPlandau->Write();
  histoROOTrndm->Write();
  histoROOTgaus->Write();
  histoG4Poisson->Write();
  histoG4UniformRand->Write();
  file->Close();
 
  return 0;

} 
