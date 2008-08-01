#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAna.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// CMSSW_2_X_X
#include "DQMServices/Core/interface/DQMStore.h"

// preprocessor macro for booking 1d histos with DQMStore -or- bare Root
#define BOOK1D(name,title,nbinsx,lowx,highx) \
  h##name = DQM ? DQM->book1D(#name,#title,nbinsx,lowx,highx)->getTH1F() \
    : new TH1F(#name,#title,nbinsx,lowx,highx)

// preprocessor macro for booking 2d histos with DQMStore -or- bare Root
#define BOOK2D(name,title,nbinsx,lowx,highx,nbinsy,lowy,highy) \
  h##name = DQM ? DQM->book2D(#name,#title,nbinsx,lowx,highx,nbinsy,lowy,highy)->getTH2F() \
    : new TH2F(#name,#title,nbinsx,lowx,highx,nbinsy,lowy,highy)

/*
// CMSSW_1_X_X
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "DQMServices/CoreROOT/interface/CollateMERoot.h"

// preprocessor macro for booking 1d histos with DaqMonitorBEInterface -or- bare Root
#define BOOK1D(name,title,nbinsx,lowx,highx) \
  if (DQM) { \
    MonitorElement *me = DQM->book1D(#name,#title,nbinsx,lowx,highx); \
    MonitorElementT<TNamed> *ob = dynamic_cast<MonitorElementT<TNamed>* >(me); \
    if (ob) h##name = dynamic_cast<TH1F *>(ob->operator->()); \
  } else h##name = new TH1F(#name,#title,nbinsx,lowx,highx)

// preprocessor macro for booking 2d histos with DaqMonitorBEInterface -or- bare Root
#define BOOK2D(name,title,nbinsx,lowx,highx,nbinsy,lowy,highy) \
  if (DQM) { \
    MonitorElement *me = DQM->book2D(#name,#title,nbinsx,lowx,highx,nbinsy,lowy,highy); \
    MonitorElementT<TNamed> *ob = dynamic_cast<MonitorElementT<TNamed>* >(me); \
    if (ob) h##name = dynamic_cast<TH2F *>(ob->operator->()); \
  } else h##name = new TH2F(#name,#title,nbinsx,lowx,highx,nbinsy,lowy,highy)
*/

// all versions OK
// preprocesor macro for setting axis titles
#define SETAXES(name,xtitle,ytitle) \
  h##name->GetXaxis()->SetTitle(xtitle); h##name->GetYaxis()->SetTitle(ytitle)

PFBenchmarkAna::PFBenchmarkAna() {}

PFBenchmarkAna::~PFBenchmarkAna() {}

void PFBenchmarkAna::setup(DQMStore *DQM) { // CMSSW_2_X_X
//void PFBenchmarkAna::setup(DaqMonitorBEInterface *DQM) { // CMSSW_1_X_X

  // use bare Root if no DQM (FWLite applications)
  if (!DQM) file_ = new TFile();

  // Book Histograms

  // delta et quantities
  BOOK1D(DeltaEt,DeltaEt,1000,-100,100);
  BOOK2D(DeltaEtvsEt,DeltaEtvsEt,1000,0,1000,1000,-100,100);
  BOOK2D(DeltaEtOverEtvsEt,DeltaEtOverEtvsEt,1000,0,1000,100,-1,1);
  BOOK2D(DeltaEtvsEta,DeltaEtvsEta,200,-5,5,1000,-100,100);
  BOOK2D(DeltaEtOverEtvsEta,DeltaEtOverEtvsEta,200,-5,5,100,-1,1);
  BOOK2D(DeltaEtvsPhi,DeltaEtvsPhi,200,-M_PI,M_PI,1000,-100,100);
  BOOK2D(DeltaEtOverEtvsPhi,DeltaEtOverEtvsPhi,200,-M_PI,M_PI,100,-1,1);
  BOOK2D(DeltaEtvsDeltaR,DeltaEtvsDeltaR,100,0,1,1000,-100,100);
  BOOK2D(DeltaEtOverEtvsDeltaR,DeltaEtOverEtvsDeltaR,100,0,1,100,-1,1);

  // delta eta quantities
  BOOK1D(DeltaEta,DeltaEta,100,-3,3);
  BOOK2D(DeltaEtavsEt,DeltaEtavsEt,1000,0,1000,100,-3,3);
  BOOK2D(DeltaEtaOverEtavsEt,DeltaEtaOverEtavsEt,1000,0,1000,100,-1,1); // ms: propose remove
  BOOK2D(DeltaEtavsEta,DeltaEtavsEta,200,-5,5,100,-3,3);
  BOOK2D(DeltaEtaOverEtavsEta,DeltaEtaOverEtvsEta,200,-5,5,100,-1,1); // ms: propose remove
  BOOK2D(DeltaEtavsPhi,DeltaEtavsPhi,200,-M_PI,M_PI,200,-M_PI,M_PI); // ms: propose remove
  BOOK2D(DeltaEtaOverEtavsPhi,DeltaEtaOverEtavsPhi,200,-M_PI,M_PI,100,-1,1); // ms: propose remove

  // delta phi quantities
  BOOK1D(DeltaPhi,DeltaPhi,100,-M_PI_2,M_PI_2);
  BOOK2D(DeltaPhivsEt,DeltaPhivsEt,1000,0,1000,100,-M_PI_2,M_PI_2);
  BOOK2D(DeltaPhiOverPhivsEt,DeltaPhiOverPhivsEt,1000,0,1000,100,-1,1); // ms: propose remove
  BOOK2D(DeltaPhivsEta,DeltaPhivsEta,200,-5,5,100,-M_PI_2,M_PI_2);
  BOOK2D(DeltaPhiOverPhivsEta,DeltaPhiOverPhivsEta,200,-5,5,100,-1,1); // ms: propose remove
  BOOK2D(DeltaPhivsPhi,DeltaPhivsPhi,200,-M_PI,M_PI,200,-M_PI,M_PI); // ms: propose remove
  BOOK2D(DeltaPhiOverPhivsPhi,DeltaPhiOverPhivsPhi,200,-M_PI,M_PI,100,-1,1); // ms: propose remove

  // delta R quantities
  BOOK1D(DeltaR,DeltaR,100,0,1);
  BOOK2D(DeltaRvsEt,DeltaRvsEt,1000,0,1000,100,0,1);
  BOOK2D(DeltaRvsEta,DeltaRvsEta,200,-5,5,100,0,1);
  BOOK2D(DeltaRvsPhi,DeltaRvsPhi,200,-M_PI,M_PI,100,0,1); // ms: propose remove

  // number of truth particles found within given cone radius of reco
  //BOOK2D(NumInConeVsConeSize,NumInConeVsConeSize,100,0,1,25,0,25);

  // Set Axis Titles

  // delta et quantities
  SETAXES(DeltaEt,"#DeltaEt","Events");
  SETAXES(DeltaEtvsEt,"Et","#DeltaEt");
  SETAXES(DeltaEtOverEtvsEt,"Et","#DeltaEt/Et");
  SETAXES(DeltaEtvsEta,"Eta","#DeltaEt");
  SETAXES(DeltaEtOverEtvsEta,"Eta","#DeltaEt/Et");
  SETAXES(DeltaEtvsPhi,"Phi","#DeltaEt");
  SETAXES(DeltaEtOverEtvsPhi,"Phi","#DeltaEt/Et");
  SETAXES(DeltaEtvsDeltaR,"#DeltaR","#DeltaEt");
  SETAXES(DeltaEtOverEtvsDeltaR,"#DeltaR","#DeltaEt/Et");

  // delta eta quantities
  SETAXES(DeltaEta,"#DeltaEta","Events");
  SETAXES(DeltaEtavsEt,"Et","#DeltaEta");
  SETAXES(DeltaEtavsEta,"Eta","#DeltaEta");
  SETAXES(DeltaEtaOverEtavsEt,"Et","#DeltaEta/Eta");
  SETAXES(DeltaEtaOverEtavsEta,"Eta","#DeltaEta/Eta");
  SETAXES(DeltaEtavsPhi,"Phi","#DeltaEta");
  SETAXES(DeltaEtaOverEtavsPhi,"Phi","#DeltaEta/Eta");

  // delta phi quantities
  SETAXES(DeltaPhi,"#DeltaPhi","Events");
  SETAXES(DeltaPhivsEt,"Et","#DeltaPhi");
  SETAXES(DeltaPhivsEta,"Eta","#DeltaPhi");
  SETAXES(DeltaPhiOverPhivsEt,"Et","#DeltaPhi/Phi");
  SETAXES(DeltaPhiOverPhivsEta,"Eta","#DeltaPhi/Phi");
  SETAXES(DeltaPhivsPhi,"Phi","#DeltaEta");
  SETAXES(DeltaPhiOverPhivsPhi,"Phi","#DeltaPhi/Phi");

  // delta R quantities
  SETAXES(DeltaR,"#DeltaR","Events");
  SETAXES(DeltaRvsEt,"Et","#DeltaR");
  SETAXES(DeltaRvsEta,"Eta","#DeltaR");
  SETAXES(DeltaRvsPhi,"Phi","#DeltaR");
  
}


void PFBenchmarkAna::fill(const edm::View<reco::Candidate> *RecoCollection, const edm::View<reco::Candidate> *GenCollection, bool PlotAgainstReco) {

  // loop over reco particles
  for (unsigned int i = 0; i < RecoCollection->size(); i++) {

    // generate histograms comparing the reco and truth candidate (truth = closest in delta-R)
    const reco::Candidate *particle = &(*RecoCollection)[i];
    const reco::Candidate *gen_particle = algo_->matchByDeltaR(particle,GenCollection);

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

void PFBenchmarkAna::write(std::string Filename) {

  if (Filename.size() != 0 && file_)
    file_->Write(Filename.c_str());

}
