#ifndef EvtGenTestAnalyzer_H
#define EvtGenTestAnalyzer_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations
class TFile;
class TH1D;

class EvtGenTestAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  //
  explicit EvtGenTestAnalyzer(const edm::ParameterSet&);
  ~EvtGenTestAnalyzer() override = default;  // no need to delete ROOT stuff
                                             // as it'll be deleted upon closing TFile

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override;
  void endJob() override;

private:
  //
  std::string fOutputFileName;
  edm::EDGetTokenT<edm::HepMCProduct> tokenHepMC_;

  TFile* fOutputFile;
  TH1D* hGeneralId;
  TH1D* hIdPhiDaugs;
  TH1D* hIdJpsiMot;
  TH1D* hnB;
  TH1D* hnBz;
  TH1D* hnBzb;
  TH1D* hPtRadPho;
  TH1D* hPhiRadPho;
  TH1D* hEtaRadPho;
  TH1D* hnJpsi;
  TH1D* hMinvb;
  TH1D* hPtbs;
  TH1D* hPbs;
  TH1D* hPhibs;
  TH1D* hEtabs;
  TH1D* hPtmu;
  TH1D* hPmu;
  TH1D* hPhimu;
  TH1D* hEtamu;
  TH1D* htbJpsiKs;
  TH1D* htbbarJpsiKs;
  TH1D* htbPlus;
  TH1D* htbsUnmix;
  TH1D* htbsMix;
  TH1D* htbUnmix;
  TH1D* htbMix;
  TH1D* htbMixPlus;
  TH1D* htbMixMinus;
  TH1D* hmumuMassSqr;
  TH1D* hmumuMassSqrPlus;
  TH1D* hmumuMassSqrMinus;
  TH1D* hIdBsDaugs;
  TH1D* hIdBDaugs;
  TH1D* hCosTheta1;
  TH1D* hCosTheta2;
  TH1D* hPhi1;
  TH1D* hPhi2;
  TH1D* hCosThetaLambda;

  std::ofstream* decayed;
  std::ofstream* undecayed;
  int nevent, nbs;
};

#endif
