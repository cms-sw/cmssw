#ifndef RecoParticleFlow_Benchmark_PFBenchmarkAna_h
#define RecoParticleFlow_Benchmark_PFBenchmarkAna_h

#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <string>

#include <TH1F.h>
#include <TH2F.h>
#include <TFile.h>

class DQMStore; // CMSSW_2_X_X
//class DaqMonitorBEInterface; // CMSSW_1_X_X

class PFBenchmarkAna {
public:

  PFBenchmarkAna();
  virtual ~PFBenchmarkAna();

  void setup(DQMStore *DQM = NULL); // CMSSW_2_X_X
  //void setup(DaqMonitorBEInterface *DQM = NULL); // CMSSW_1_X_X
  template <typename RecoCandidate, typename GenCandidate, template <typename> class VectorT, template <typename> class VectorU>
  void fill(const VectorT<RecoCandidate> *, const VectorU<GenCandidate> *, bool PlotAgainstReco = true);
  void write(std::string Filename);

private:

  TFile *file_;

  TH1F *hDeltaEt;
  TH2F *hDeltaEtvsEt;
  TH2F *hDeltaEtOverEtvsEt;
  TH2F *hDeltaEtvsEta;
  TH2F *hDeltaEtOverEtvsEta;
  TH2F *hDeltaEtvsPhi;
  TH2F *hDeltaEtOverEtvsPhi;
  TH2F *hDeltaEtvsDeltaR;
  TH2F *hDeltaEtOverEtvsDeltaR;

  TH1F *hDeltaEta;
  TH2F *hDeltaEtavsEt;
  TH2F *hDeltaEtaOverEtavsEt; // ms: propose remove
  TH2F *hDeltaEtavsEta;
  TH2F *hDeltaEtaOverEtavsEta; // ms: propose remove
  TH2F *hDeltaEtavsPhi; // ms: propose remove
  TH2F *hDeltaEtaOverEtavsPhi; // ms: propose remove

  TH1F *hDeltaPhi;
  TH2F *hDeltaPhivsEt;
  TH2F *hDeltaPhiOverPhivsEt; // ms: propose remove
  TH2F *hDeltaPhivsEta;
  TH2F *hDeltaPhiOverPhivsEta; // ms: propose remove
  TH2F *hDeltaPhivsPhi; // ms: propose remove
  TH2F *hDeltaPhiOverPhivsPhi; // ms: propose remove

  TH1F *hDeltaR;
  TH2F *hDeltaRvsEt;
  TH2F *hDeltaRvsEta;
  TH2F *hDeltaRvsPhi; // ms: propose remove

protected:

  //DQMStore *dbe_;
  PFBenchmarkAlgo *algo_;

};

template <typename RecoCandidate, typename GenCandidate, template <typename> class VectorT, template <typename> class VectorU>
void PFBenchmarkAna::fill(const VectorT<RecoCandidate> *RecoCollection, const VectorU<GenCandidate> *GenCollection, bool PlotAgainstReco) {

  // loop over reco particles
  for (unsigned int i = 0; i < RecoCollection->size(); i++) {

    // generate histograms comparing the reco and truth candidate (truth = closest in delta-R)
    const RecoCandidate *particle = &(*RecoCollection)[i];
    const GenCandidate *gen_particle = algo_->matchByDeltaR(particle,GenCollection);

    // get the quantities to place on the denominator and/or divide by
    double et, eta, phi;
    if (PlotAgainstReco) { et = particle->et(); eta = particle->eta(); phi = particle->phi(); }
    else { et = gen_particle->et(); eta = gen_particle->eta(); phi = gen_particle->phi(); }
    
    // get the delta quantities
    double deltaEt = algo_->deltaEt(particle,gen_particle);
    double deltaR = algo_->deltaR(particle,gen_particle);
    double deltaEta = algo_->deltaEta(particle,gen_particle);
    double deltaPhi = algo_->deltaPhi(particle,gen_particle);
    
    // fill histograms
    hDeltaEt->Fill(deltaEt);
    hDeltaEtvsEt->Fill(et,deltaEt);
    hDeltaEtOverEtvsEt->Fill(et,deltaEt/et);
    hDeltaEtvsEta->Fill(eta,deltaEt);
    hDeltaEtOverEtvsEta->Fill(eta,deltaEt/et);
    hDeltaEtvsPhi->Fill(phi,deltaEt);
    hDeltaEtOverEtvsPhi->Fill(phi,deltaEt/et);
    hDeltaEtvsDeltaR->Fill(deltaR,deltaEt);
    hDeltaEtOverEtvsDeltaR->Fill(deltaR,deltaEt/et);
    
    hDeltaEta->Fill(deltaEta);
    hDeltaEtavsEt->Fill(et,deltaEta/eta);
    hDeltaEtaOverEtavsEt->Fill(et,deltaEta/eta);
    hDeltaEtavsEta->Fill(eta,deltaEta);
    hDeltaEtaOverEtavsEta->Fill(eta,deltaEta/eta);
    hDeltaEtavsPhi->Fill(phi,deltaEta);
    hDeltaEtaOverEtavsPhi->Fill(phi,deltaEta/eta);
    
    hDeltaPhi->Fill(deltaPhi);
    hDeltaPhivsEt->Fill(et,deltaPhi);
    hDeltaPhiOverPhivsEt->Fill(et,deltaPhi/phi);
    hDeltaPhivsEta->Fill(eta,deltaPhi);
    hDeltaPhiOverPhivsEta->Fill(eta,deltaPhi/phi);
    hDeltaPhivsPhi->Fill(phi,deltaPhi);
    hDeltaPhiOverPhivsPhi->Fill(phi,deltaPhi/phi);

    hDeltaR->Fill(deltaR);
    hDeltaRvsEt->Fill(et,deltaR);
    hDeltaRvsEta->Fill(eta,deltaR);

  }

}


#endif // RecoParticleFlow_Benchmark_PFBenchmarkAna_h
