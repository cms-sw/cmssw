#include "RecoParticleFlow/Benchmark/interface/GenericBenchmark.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// CMSSW_2_X_X
#include "DQMServices/Core/interface/DQMStore.h"


//Colin: it seems a bit strange to use the preprocessor for that kind of 
//thing. Looks like all these macros could be replaced by plain functions.

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

#define ET (PlotAgainstReco_)?"reconstructed E_{T} [GeV]":"generated E_{T} [GeV]"
#define ETA (PlotAgainstReco_)?"reconstructed #eta":"generated #eta"
#define PHI (PlotAgainstReco_)?"reconstructed #phi (rad)":"generated #phi (rad)"

using namespace std;

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
  BOOK2D(DeltaEtavsEta,"#Delta#eta vs #eta",200,-5,5,100,-3,3);

  // delta phi quantities
  BOOK1D(DeltaPhi,"#Delta#phi",100,-0.2,0.2);
  BOOK2D(DeltaPhivsEt,"#Delta#phi vs E_{T}",250,0,500,1000,-0.5,0.5);
  BOOK2D(DeltaPhivsEta,"#Delta#phi vs #eta",200,-5,5,100,-M_PI_2,M_PI_2);

  // delta R quantities
  BOOK1D(DeltaR,"#DeltaR",100,0,1);
  BOOK2D(DeltaRvsEt,"#DeltaR vs E_{T}",1000,0,1000,100,0,1);
  BOOK2D(DeltaRvsEta,"#DeltaR vs #eta",200,-5,5,100,0,1);


  // seen and gen distributions, for efficiency computation
  BOOK1D(EtaSeen,"seen #eta",100,-5,5);
  BOOK1D(PhiSeen,"seen #phi",100,-3.5,3.5);
  BOOK1D(EtSeen,"seen E_{T}",1000,0,1000);
  
  BOOK1D(EtaGen,"generated #eta",100,-5,5);
  BOOK1D(PhiGen,"generated #phi",100,-3.5,3.5);
  BOOK1D(EtGen,"generated E_{T}",1000,0,1000);
  

  // number of truth particles found within given cone radius of reco
  //BOOK2D(NumInConeVsConeSize,NumInConeVsConeSize,100,0,1,25,0,25);

  // Set Axis Titles
 
  // delta et quantities
  SETAXES(DeltaEt,"#DeltaE_{T} [GeV]","");
  SETAXES(DeltaEtvsEt,ET,"#DeltaE_{T} [GeV]");
  SETAXES(DeltaEtOverEtvsEt,ET,"#DeltaE_{T}/E_{T}");
  SETAXES(DeltaEtvsEta,ETA,"#DeltaE_{T} [GeV]");
  SETAXES(DeltaEtOverEtvsEta,ETA,"#DeltaE_{T}/E_{T}");
  SETAXES(DeltaEtvsPhi,PHI,"#DeltaE_{T} [GeV]");
  SETAXES(DeltaEtOverEtvsPhi,PHI,"#DeltaE_{T}/E_{T}");
  SETAXES(DeltaEtvsDeltaR,"#DeltaR","#DeltaE_{T} [GeV]");
  SETAXES(DeltaEtOverEtvsDeltaR,"#DeltaR","#DeltaE_{T}/E_{T}");

  // delta eta quantities
  SETAXES(DeltaEta,"#Delta#eta","Events");
  SETAXES(DeltaEtavsEt,ET,"#Delta#eta");
  SETAXES(DeltaEtavsEta,ETA,"#Delta#eta");

  // delta phi quantities
  SETAXES(DeltaPhi,"#Delta#phi [rad]","Events");
  SETAXES(DeltaPhivsEt,ET,"#Delta#phi [rad]");
  SETAXES(DeltaPhivsEta,ETA,"#Delta#phi [rad]");

  // delta R quantities
  SETAXES(DeltaR,"#DeltaR","Events");
  SETAXES(DeltaRvsEt,ET,"#DeltaR");
  SETAXES(DeltaRvsEta,ETA,"#DeltaR");
  
  SETAXES(EtaSeen,"","seen #eta");
  SETAXES(PhiSeen,"","seen #phi [rad]");
  SETAXES(EtSeen,"","seen E_{T}");

  SETAXES(EtaGen,"","generated #eta");
  SETAXES(PhiGen,"","generated #phi [rad]");
  SETAXES(EtGen,"","generated E_{T}");


}


void GenericBenchmark::fill(const edm::View<reco::Candidate> *RecoCollection, 
			    const edm::View<reco::Candidate> *GenCollection, 
			    bool PlotAgainstReco, bool onlyTwoJets, 
			    double recPt_cut, 
			    double maxEta_cut, 
			    double deltaR_cut) {

  // loop over reco particles
  for (unsigned int i = 0; i < RecoCollection->size(); i++) {
    
    // generate histograms comparing the reco and truth candidate (truth = closest in delta-R)
    const reco::Candidate *particle = &(*RecoCollection)[i];

    assert( particle!=NULL ); 
    if( !accepted(particle, recPt_cut, maxEta_cut)) continue;

    const reco::Candidate *gen_particle = algo_->matchByDeltaR(particle,
							       GenCollection);
    if(gen_particle==NULL) continue; 


    // Count the number of jets with a larger energy
    if( onlyTwoJets ) {
      unsigned highJets = 0;
      for(unsigned j=0; j<RecoCollection->size(); j++) { 
	const reco::Candidate *otherParticle = &(*RecoCollection)[j];
	if ( j != i && otherParticle->pt() > particle->pt() ) highJets++;
      }
      if ( highJets > 1 ) continue;
    }		

    // get the quantities to place on the denominator and/or divide by
    double et = gen_particle->et();
    double eta = gen_particle->eta();
    double phi = gen_particle->phi();
    
    if (PlotAgainstReco) { 
      et = particle->et(); 
      eta = particle->eta(); 
      phi = particle->phi(); 
    }

    
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
    hDeltaEtavsEta->Fill(eta,deltaEta);
    
    hDeltaPhi->Fill(deltaPhi);
    hDeltaPhivsEt->Fill(et,deltaPhi);
    hDeltaPhivsEta->Fill(eta,deltaPhi);

    hDeltaR->Fill(deltaR);
    hDeltaRvsEt->Fill(et,deltaR);
    hDeltaRvsEta->Fill(eta,deltaR);
  }


  // loop over gen particles
  
//   cout<<"Reco size = "<<RecoCollection->size()<<", ";
//   cout<<"Gen size = "<<GenCollection->size()<<endl;

  for (unsigned int i = 0; i < GenCollection->size(); i++) {

    const reco::Candidate *gen_particle = &(*GenCollection)[i]; 

    if( !accepted(gen_particle, recPt_cut, maxEta_cut)) {
      continue;
    }

    hEtaGen->Fill(gen_particle->eta() );
    hPhiGen->Fill(gen_particle->phi() );
    hEtGen->Fill(gen_particle->et() );

    const reco::Candidate *rec_particle = algo_->matchByDeltaR(gen_particle,
							       RecoCollection);
    if(! rec_particle) continue; // no match
    // must make a cut on delta R 

    hEtaSeen->Fill(gen_particle->eta() );
    hPhiSeen->Fill(gen_particle->phi() );
    hEtSeen->Fill(gen_particle->et() );

  }

}

bool GenericBenchmark::accepted(const reco::Candidate* particle,
				double ptCut,
				double etaCut ) const {
 
  //skip reconstructed PFJets with p_t < recPt_cut
  if (particle->pt() < ptCut and ptCut != -1.)
    return false;

  if (fabs(particle->eta())>etaCut and etaCut != -1.)
    return false;

  //accepted!
  return true;
 
}


void GenericBenchmark::write(std::string Filename) {

  if (Filename.size() != 0 && file_)
    file_->Write(Filename.c_str());

}
