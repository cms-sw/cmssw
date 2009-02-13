#include "RecoParticleFlow/Benchmark/interface/GenericBenchmark.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// CMSSW_2_X_X
#include "DQMServices/Core/interface/DQMStore.h"

// preprocessor macro for booking 1d histos with DQMStore -or- bare Root
#define BOOK1D(name,title,nbinsx,lowx,highx) \
  h##name = DQM ? DQM->book1D(#name,title,nbinsx,lowx,highx)->getTH1F() \
    : new TH1F(#name,title,nbinsx,lowx,highx)

// preprocessor macro for booking 2d histos with DQMStore -or- bare Root
#define BOOK2D(name,title,nbinsx,lowx,highx,nbinsy,lowy,highy) \
  h##name = DQM ? DQM->book2D(#name,title,nbinsx,lowx,highx,nbinsy,lowy,highy)->getTH2F() \
    : new TH2F(#name,title,nbinsx,lowx,highx,nbinsy,lowy,highy)






// all versions OK
// preprocesor macro for setting axis titles

#define SETAXES(name,xtitle,ytitle) \
  h##name->GetXaxis()->SetTitle(xtitle); h##name->GetYaxis()->SetTitle(ytitle)

#define ET (PlotAgainstReco_)?"reconstructed E_{T}":"generated E_{T}"
#define ETA (PlotAgainstReco_)?"reconstructed #eta":"generated #eta"
#define PHI (PlotAgainstReco_)?"reconstructed #phi":"generated #phi"



GenericBenchmark::GenericBenchmark() {}

GenericBenchmark::~GenericBenchmark() {}

void GenericBenchmark::setup(DQMStore *DQM, bool PlotAgainstReco_) { 

  // CMSSW_2_X_X
  // use bare Root if no DQM (FWLite applications)
  if (!DQM) file_ = new TFile();

  // Book Histograms

  // delta et quantities
  BOOK1D(DeltaEt,"#DeltaE_{T}",1000,-60,40);
  BOOK2D(DeltaEtvsEt,"#DeltaE_{T} vs E_{T}",1000,0,1000,1000,-100,100);
  BOOK2D(DeltaEtOverEtvsEt,"#DeltaE_{T}/E_{T} vsE_{T}",1000,0,1000,100,-1,1);
  BOOK2D(DeltaEtvsEta,"#DeltaE_{T} vs #eta",200,-5,5,1000,-100,100);
  BOOK2D(DeltaEtOverEtvsEta,"#DeltaE_{T}/E_{T} vs #eta",200,-5,5,100,-1,1);
  BOOK2D(DeltaEtvsPhi,"#DeltaE_{T} vs #phi",200,-M_PI,M_PI,1000,-100,100);
  BOOK2D(DeltaEtOverEtvsPhi,"#DeltaE_{T}/E_{T} vs #Phi",200,-M_PI,M_PI,100,-1,1);
  BOOK2D(DeltaEtvsDeltaR,"#DeltaE_{T} vs #DeltaR",100,0,1,1000,-100,100);
  BOOK2D(DeltaEtOverEtvsDeltaR,"#DeltaE_{T}/E_{T} vs #DeltaR",100,0,1,100,-1,1);

  // delta eta quantities
  BOOK1D(DeltaEta,"#Delta#eta",100,-0.2,0.2);
  BOOK2D(DeltaEtavsEt,"#Delta#eta vs E_{T}",250,0,500,1000,-0.5,0.5);
  BOOK2D(DeltaEtaOverEtavsEt,"#Delta#eta/#eta vs E_(T}",1000,0,1000,100,-1,1); // ms: propose remove
  BOOK2D(DeltaEtavsEta,"#Delta#eta vs #eta",200,-5,5,100,-3,3);
  BOOK2D(DeltaEtaOverEtavsEta,"EDelta#eta/#eta vs #eta",200,-5,5,100,-1,1); // ms: propose remove
  BOOK2D(DeltaEtavsPhi,"#Delta#eta vs #phi",200,-M_PI,M_PI,200,-M_PI,M_PI); // ms: propose remove
  BOOK2D(DeltaEtaOverEtavsPhi,"#Delta#eta/#eta vs #phi",200,-M_PI,M_PI,100,-1,1); // ms: propose remove

  // delta phi quantities
  BOOK1D(DeltaPhi,"#Delta#phi",100,-0.2,0.2);
  BOOK2D(DeltaPhivsEt,"#Delta#phi vs E_{T}",250,0,500,1000,-0.5,0.5);
  BOOK2D(DeltaPhiOverPhivsEt,"#Delta#phi/#phi vs E_{T}",1000,0,1000,100,-1,1); // ms: propose remove
  BOOK2D(DeltaPhivsEta,"#Delta#phi vs #eta",200,-5,5,100,-M_PI_2,M_PI_2);
  BOOK2D(DeltaPhiOverPhivsEta,"#Delta#phi/#phi vs #eta",200,-5,5,100,-1,1); // ms: propose remove
  BOOK2D(DeltaPhivsPhi,"#Delta#phi vs #phi",200,-M_PI,M_PI,200,-M_PI,M_PI); // ms: propose remove
  BOOK2D(DeltaPhiOverPhivsPhi,"#Delta#phi/#phi vs #phi",200,-M_PI,M_PI,100,-1,1); // ms: propose remove

  // delta R quantities
  BOOK1D(DeltaR,"#DeltaR",100,0,1);
  BOOK2D(DeltaRvsEt,"#DeltaR vs E_{T}",1000,0,1000,100,0,1);
  BOOK2D(DeltaRvsEta,"#DeltaR vs #eta",200,-5,5,100,0,1);
  BOOK2D(DeltaRvsPhi,"#DeltaR vs #phi",200,-M_PI,M_PI,100,0,1); // ms: propose remove

  // number of truth particles found within given cone radius of reco
  //BOOK2D(NumInConeVsConeSize,NumInConeVsConeSize,100,0,1,25,0,25);

  // Set Axis Titles
 
  // delta et quantities
  SETAXES(DeltaEt,"#DeltaE_{T}","Events");
  SETAXES(DeltaEtvsEt,ET,"#DeltaE_{T}");
  SETAXES(DeltaEtOverEtvsEt,ET,"#DeltaE_{T}/E_{T}");
  SETAXES(DeltaEtvsEta,ETA,"#DeltaE_{T}");
  SETAXES(DeltaEtOverEtvsEta,ETA,"#DeltaE_{T}/E_{T}");
  SETAXES(DeltaEtvsPhi,PHI,"#DeltaE_{T}");
  SETAXES(DeltaEtOverEtvsPhi,PHI,"#DeltaE_{T}/E_{T}");
  SETAXES(DeltaEtvsDeltaR,"#DeltaR","#DeltaE_{T}");
  SETAXES(DeltaEtOverEtvsDeltaR,"#DeltaR","#DeltaE_{T}/E_{T}");

  // delta eta quantities
  SETAXES(DeltaEta,"#Delta#eta","Events");
  SETAXES(DeltaEtavsEt,ET,"#Delta#eta");
  SETAXES(DeltaEtavsEta,ETA,"#Delta#eta");
  SETAXES(DeltaEtaOverEtavsEt,ET,"#Delta#eta/#eta");
  SETAXES(DeltaEtaOverEtavsEta,ETA,"#Delta#eta/#eta");
  SETAXES(DeltaEtavsPhi,PHI,"#Delta#eta");
  SETAXES(DeltaEtaOverEtavsPhi,PHI,"#Delta#eta/#eta");

  // delta phi quantities
  SETAXES(DeltaPhi,"#Delta#phi","Events");
  SETAXES(DeltaPhivsEt,ET,"#Delta#phi");
  SETAXES(DeltaPhivsEta,ETA,"#Delta#phi");
  SETAXES(DeltaPhiOverPhivsEt,ET,"#Delta#phi/#phi");
  SETAXES(DeltaPhiOverPhivsEta,ETA,"#Delta#phi/#phi");
  SETAXES(DeltaPhivsPhi,PHI,"#Delta#phi");
  SETAXES(DeltaPhiOverPhivsPhi,PHI,"#Delta#phi/#phi");

  // delta R quantities
  SETAXES(DeltaR,"#DeltaR","Events");
  SETAXES(DeltaRvsEt,ET,"#DeltaR");
  SETAXES(DeltaRvsEta,ETA,"#DeltaR");
  SETAXES(DeltaRvsPhi,PHI,"#DeltaR");
  
}


void GenericBenchmark::fill(const edm::View<reco::Candidate> *RecoCollection, const edm::View<reco::Candidate> *GenCollection, bool PlotAgainstReco, bool onlyTwoJets, double recPt_cut, double maxEta_cut, double deltaR_cut) {

  // loop over reco particles
  for (unsigned int i = 0; i < RecoCollection->size(); i++) {
    
    // generate histograms comparing the reco and truth candidate (truth = closest in delta-R)
    const reco::Candidate *particle = &(*RecoCollection)[i];
    const reco::Candidate *gen_particle = algo_->matchByDeltaR(particle,GenCollection);

   if  ((particle!=NULL) &&  (gen_particle!=NULL)){
      //   std::cout << RecoCollection->size() << " " << GenCollection->size() <<particle  <<gen_particle << std::endl;   

    // Count the number of jets with a larger energy
    unsigned highJets = 0;
    for(unsigned j=0; j<RecoCollection->size(); j++) { 
      const reco::Candidate *otherParticle = &(*RecoCollection)[j];
      if ( j != i && otherParticle->pt() > particle->pt() ) highJets++;
    }
    if ( onlyTwoJets && highJets > 1 ) continue;
		
    //skip reconstructed PFJets with p_t < recPt_cut
    if (particle->pt() < recPt_cut and recPt_cut != -1.)
      continue;
    //skip if PFJet within eta>maxEta_cut
    if (fabs(particle->eta())>maxEta_cut and maxEta_cut != -1.)
      continue;

    // get the quantities to place on the denominator and/or divide by
    double et, eta, phi;
    if (PlotAgainstReco) { et = particle->et(); eta = particle->eta(); phi = particle->phi(); }
    else { et = gen_particle->et(); eta = gen_particle->eta(); phi = gen_particle->phi(); }
    
    // get the delta quantities
    double deltaEt = algo_->deltaEt(particle,gen_particle);
    double deltaR = algo_->deltaR(particle,gen_particle);
    double deltaEta = algo_->deltaEta(particle,gen_particle);
    double deltaPhi = algo_->deltaPhi(particle,gen_particle);
   
    //TODO implement variable Cut:
     if (fabs(deltaR)>deltaR_cut and deltaR_cut != -1.)
       continue;

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
    hDeltaEtavsEt->Fill(et,deltaEta);
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

}

void GenericBenchmark::write(std::string Filename) {

  if (Filename.size() != 0 && file_)
    file_->Write(Filename.c_str());

}
