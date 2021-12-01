/**
 
  \brief Hydjet2 main class
  \details class to read Hydjet2 particles arrays 
  \author Wouf (Wouf@mail.cern.ch)
  \version 2.4.3
  \date NOV 2021
  \copyright GNU Public License.
*/

#ifndef HYDJET2_H
#define HYDJET2_H

#include <TMath.h>
#include <TRandom.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "InitialParams.h"

class Hydjet2 {

public:
  Hydjet2(InitialParamsHydjet_t &); ///< Constructor of Hydjet2 with input parameters structure (InitialParams.h) as input parameter
  ~Hydjet2();

  void GenerateEvent(Double_t &); ///< Generate one event

  //ref-qualified getters

  std::vector<Int_t> &GetPdg() & { return pdg; }                               ///< Getter (lvalue) for output information: pdg encodings
  std::vector<Int_t> &GetMpdg() & { return Mpdg; }                             ///< Getter (lvalue) for output information: pdg encodings for mother hadrons
  std::vector<Int_t> &GetType() & { return type; }                             ///< Getter (lvalue) for output information: type of particle (0-from hydro or decays, 1 -from jets)
  std::vector<Int_t> &GetPythiaStatus() & { return pythiaStatus; }             ///< Getter (lvalue) for output information: pythia status code
  std::vector<Double_t> &GetPx() & { return Px; }                              ///< Getter (lvalue) for output information: x-hadron momentum component,[GeV/c]
  std::vector<Double_t> &GetPy() & { return Py; }                              ///< Getter (lvalue) for output information: y-hadron momentum component,[GeV/c]
  std::vector<Double_t> &GetPz() & { return Pz; }                              ///< Getter (lvalue) for output information: z-hadron momentum component,[GeV/c]
  std::vector<Double_t> &GetE() & { return E; }                                ///< Getter (lvalue) for output information: hadron total energy,[GeV]
  std::vector<Double_t> &GetX() & { return X; }                                ///< Getter (lvalue) for output information: x-hadron coordinate component,[fm]
  std::vector<Double_t> &GetY() & { return Y; }                                ///< Getter (lvalue) for output information: y-hadron coordinate component,[fm]
  std::vector<Double_t> &GetZ() & { return Z; }                                ///< Getter (lvalue) for output information: z-hadron coordinate component,[fm]
  std::vector<Double_t> &GetT() & { return T; }                                ///< Getter (lvalue) for output information: hadron time,[fm/c]
  std::vector<Int_t> &GetIndex() & { return Index; }                           ///< Getter (lvalue) for output information: particle index in the secondaries tree
  std::vector<Int_t> &GetMotherIndex() & { return MotherIndex; }               ///< Getter (lvalue) for output information: mother index
  std::vector<Int_t> &GetNDaughters() & { return NDaughters; }                 ///< Getter (lvalue) for output information: number of daughters
  std::vector<Int_t> &GetFirstDaughterIndex() & { return FirstDaughterIndex; } ///< Getter (lvalue) for output information: first daughter index
  std::vector<Int_t> &GetLastDaughterIndex() & { return LastDaughterIndex; }   ///< Getter (lvalue) for output information: last daughter index
  std::vector<Int_t> &GetiJet() & { return iJet; }                             ///< Getter (lvalue) for output information: subevent number (0 - for hydro, number of origin jet for others)
  std::vector<Int_t> &GetFinal() & { return ifFinal; }                         ///< Getter (lvalue) for output information: if the partical is final(=0 no, this particle has decayed; 1= yes, final state particle)
  Int_t &GetNtot() & { return Ntot; }                                          ///< Getter (lvalue) for output information: generated value of total event multiplicity (Ntot=Nhyd+Npyt)
  Int_t &GetNpyt() & { return Npyt; }                                          ///< Getter (lvalue) for output information: generated multiplicity of "hard" jet-induced particles
  Int_t &GetNhyd() & { return Nhyd; }                                          ///< Getter (lvalue) for output information: generated multiplicity of "soft" hydro-induced particles
  Int_t &GetNjet() & { return Njet; }                                          ///< Getter (lvalue) for output information: generated number of hard parton-parton scatterings with pt>fPtmin
  Int_t &GetNbcol() & { return Nbcol; }                                        ///< Getter (lvalue) for output information: mean number of binary NN sub-collisions at given "Bgen"
  Int_t &GetNpart() & { return Npart; }                                        ///< Getter (lvalue) for output information: mean number of nucleons-participants at given "Bgen"
  Double_t &GetBgen() & { return Bgen; }                                       ///< Getter (lvalue) for output information: generated value of impact parameter in units of nucleus radius RA
  Double_t &GetSigin() & { return Sigin; }                                     ///< Getter (lvalue) for output information: total inelastic NN cross section at given "fSqrtS" (in mb)
  Double_t &GetSigjet() & { return Sigjet; }                                   ///< Getter (lvalue) for output information: hard scattering NN cross section at given "fSqrtS" & "fPtmin" (in mb)
  Double_t &GetPsiv3() & { return v3psi; }                                     ///< Getter (lvalue) for output information: angle for third Fourier harmonic of azimuthal particle distribution
  Int_t &GetNev() & { return nev; }                                            ///< Getter (lvalue) for output information: requested number of events
  Bool_t &IsEmpty() & { return emptyEvent; }                                   ///< Getter (lvalue) for output information: if true - the event is empty
  std::vector<Int_t> &GetVersion() & { return Version; }                       ///< Getter (lvalue) for output information: version information

  std::vector<Int_t> GetPdg() && { return std::move(pdg); }                               ///< Getter (rvalue) for output information: pdg encodings
  std::vector<Int_t> GetMpdg() && { return std::move(Mpdg); }                             ///< Getter (rvalue) for output information: pdg encodings for mother hadrons
  std::vector<Int_t> GetType() && { return std::move(type); }                             ///< Getter (rvalue) for output information: type of particle (0-from hydro or decays, 1 -from jets)
  std::vector<Int_t> GetPythiaStatus() && { return std::move(pythiaStatus); }             ///< Getter (rvalue) for output information: pythia status code
  std::vector<Double_t> GetPx() && { return std::move(Px); }                              ///< Getter (rvalue) for output information: x-hadron momentum component,[GeV/c]
  std::vector<Double_t> GetPy() && { return std::move(Py); }                              ///< Getter (rvalue) for output information: y-hadron momentum component,[GeV/c]
  std::vector<Double_t> GetPz() && { return std::move(Pz); }                              ///< Getter (rvalue) for output information: z-hadron momentum component,[GeV/c]
  std::vector<Double_t> GetE() && { return std::move(E); }                                ///< Getter (rvalue) for output information: hadron total energy,[GeV]
  std::vector<Double_t> GetX() && { return std::move(X); }                                ///< Getter (rvalue) for output information: x-hadron coordinate component,[fm]
  std::vector<Double_t> GetY() && { return std::move(Y); }                                ///< Getter (rvalue) for output information: y-hadron coordinate component,[fm]
  std::vector<Double_t> GetZ() && { return std::move(Z); }                                ///< Getter (rvalue) for output information: z-hadron coordinate component,[fm]
  std::vector<Double_t> GetT() && { return std::move(T); }                                ///< Getter (rvalue) for output information: hadron time,[fm/c]
  std::vector<Int_t> GetIndex() && { return std::move(Index); }                           ///< Getter (rvalue) for output information: particle index in the secondaries tree
  std::vector<Int_t> GetMotherIndex() && { return std::move(MotherIndex); }               ///< Getter (rvalue) for output information: mother index
  std::vector<Int_t> GetNDaughters() && { return std::move(NDaughters); }                 ///< Getter (rvalue) for output information: number of daughters
  std::vector<Int_t> GetFirstDaughterIndex() && { return std::move(FirstDaughterIndex); } ///< Getter (rvalue) for output information: first daughter index
  std::vector<Int_t> GetLastDaughterIndex() && { return std::move(LastDaughterIndex); }   ///< Getter (rvalue) for output information: last daughter index
  std::vector<Int_t> GetiJet() && { return std::move(iJet); }                             ///< Getter (rvalue) for output information: subevent number (0 - for hydro, number of origin jet for others)
  std::vector<Int_t> GetFinal() && { return std::move(ifFinal); }                         ///< Getter (rvalue) for output information: if the partical is final(=0 no, this particle has decayed; 1= yes, final state particle)
  Int_t GetNtot() && { return std::move(Ntot); }                                          ///< Getter (rvalue) for output information: generated value of total event multiplicity (Ntot=Nhyd+Npyt)
  Int_t GetNpyt() && { return std::move(Npyt); }                                          ///< Getter (rvalue) for output information: generated multiplicity of "hard" jet-induced particles
  Int_t GetNhyd() && { return std::move(Nhyd); }                                          ///< Getter (rvalue) for output information: generated multiplicity of "soft" hydro-induced particles
  Int_t GetNjet() && { return std::move(Njet); }                                          ///< Getter (rvalue) for output information: generated number of hard parton-parton scatterings with pt>fPtmin
  Int_t GetNbcol() && { return std::move(Nbcol); }                                        ///< Getter (rvalue) for output information: mean number of binary NN sub-collisions at given "Bgen"
  Int_t GetNpart() && { return std::move(Npart); }                                        ///< Getter (rvalue) for output information: mean number of nucleons-participants at given "Bgen"
  Double_t GetBgen() && { return std::move(Bgen); }                                       ///< Getter (rvalue) for output information: generated value of impact parameter in units of nucleus radius RA
  Double_t GetSigin() && { return std::move(Sigin); }                                     ///< Getter (rvalue) for output information: total inelastic NN cross section at given "fSqrtS" (in mb)
  Double_t GetSigjet() && { return std::move(Sigjet); }                                   ///< Getter (rvalue) for output information: hard scattering NN cross section at given "fSqrtS"  &  &  "fPtmin" (in mb)
  Double_t GetPsiv3() && { return std::move(v3psi); }                                     ///< Getter (rvalue) for output information: angle for third Fourier harmonic of azimuthal particle distribution
  Int_t GetNev() && { return std::move(nev); }                                            ///< Getter (rvalue) for output information: requested number of events
  Bool_t IsEmpty() && { return std::move(emptyEvent); }                                   ///< Getter (rvalue) for output information: if true - the event is empty
  std::vector<Int_t> GetVersion() && { return std::move(Version); }                       ///< Getter (rvalue) for output information: version information

private:
  int cm = 1, va, vb, vc;

  std::vector<Int_t> Version = std::vector<Int_t>{0, 0, 0};
  clock_t start;
  Bool_t emptyEvent = false;
  Bool_t FirstEv;

  ///define event characteristics:
  Int_t nev;

  ///total event multiplicity, number of produced hadrons in hard part/soft part
  Int_t Ntot = 0, Npyt = 0, Nhyd = 0;

  /// number of jets, number of binary collisions, number of participants :
  //Int_t Njet = null, Nbcol = null, Npart = null;
  Int_t Njet, Nbcol, Npart;

  ///impact parameter
  Double_t Bgen, Sigin, Sigjet, v3psi;

  //define hadron characteristic vectors
  std::vector<Int_t> pdg;                ///< pdg encodings
  std::vector<Int_t> Mpdg;               ///< pdg encodings for mother hadrons
  std::vector<Int_t> type;               ///< type of particle: 0-from hydro or decays, 1 -from jets
  std::vector<Int_t> pythiaStatus;       ///< pythia status code
  std::vector<Double_t> Px;              ///< x-hadron momentum component,[GeV/c]
  std::vector<Double_t> Py;              ///< y-hadron momentum component,[GeV/c]
  std::vector<Double_t> Pz;              ///< z-hadron momentum component,[GeV/c]
  std::vector<Double_t> E;               ///< hadron total energy,[GeV]
  std::vector<Double_t> X;               ///< x-hadron coordinate component,[fm]
  std::vector<Double_t> Y;               ///< y-hadron coordinate component,[fm]
  std::vector<Double_t> Z;               ///< z-hadron coordinate component,[fm]
  std::vector<Double_t> T;               ///< hadron time,[fm/c]
  std::vector<Int_t> Index;              ///< particle index in the secondaries tree
  std::vector<Int_t> MotherIndex;        ///< mother index
  std::vector<Int_t> NDaughters;         ///< number of daughters
  std::vector<Int_t> FirstDaughterIndex; ///< first daughter index
  std::vector<Int_t> LastDaughterIndex;  ///< last daughter index
  std::vector<Int_t> iJet;               ///< subevent number
  std::vector<Int_t> ifFinal;            ///< (=0 no, this particle has decayed; 1= yes, final state particle)
};
#endif
