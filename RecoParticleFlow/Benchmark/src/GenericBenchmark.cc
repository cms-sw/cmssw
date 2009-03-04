#include "RecoParticleFlow/Benchmark/interface/GenericBenchmark.h"
#include "RecoParticleFlow/Benchmark/interface/BenchmarkTree.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// CMSSW_2_X_X
#include "DQMServices/Core/interface/DQMStore.h"

#include <TROOT.h>
#include <TFile.h>

//Colin: it seems a bit strange to use the preprocessor for that kind of 
//thing. Looks like all these macros could be replaced by plain functions.

// preprocessor macro for booking 1d histos with DQMStore -or- bare Root
#define BOOK1D(name,title,nbinsx,lowx,highx)				\
  h##name = DQM ? DQM->book1D(#name,title,nbinsx,lowx,highx)->getTH1F() \
    : new TH1F(#name,title,nbinsx,lowx,highx)

// preprocessor macro for booking 2d histos with DQMStore -or- bare Root
#define BOOK2D(name,title,nbinsx,lowx,highx,nbinsy,lowy,highy)		\
  h##name = DQM ? DQM->book2D(#name,title,nbinsx,lowx,highx,nbinsy,lowy,highy)->getTH2F() \
    : new TH2F(#name,title,nbinsx,lowx,highx,nbinsy,lowy,highy)


// all versions OK
// preprocesor macro for setting axis titles

#define SETAXES(name,xtitle,ytitle)					\
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

  int nbinsEt = 1000;
  float minEt = 0;
  float maxEt = 1000;

  float minDeltaEt = -100;
  float maxDeltaEt = 50;

  int nbinsEta = 200;
  float minEta = -5;
  float maxEta = 5;

  int nbinsDeltaEta = 1000;
  float minDeltaEta = -0.5;
  float maxDeltaEta = 0.5;

  int nbinsDeltaPhi = 1000;
  float minDeltaPhi = -0.5;
  float maxDeltaPhi = 0.5;


  // delta et quantities
  BOOK1D(DeltaEt,"#DeltaE_{T}", nbinsEt, minDeltaEt, maxDeltaEt);
  BOOK2D(DeltaEtvsEt,"#DeltaE_{T} vs E_{T}",
	 nbinsEt, minEt, maxEt,
	 nbinsEt,minDeltaEt, maxDeltaEt);
  BOOK2D(DeltaEtOverEtvsEt,"#DeltaE_{T}/E_{T} vsE_{T}",
	 nbinsEt, minEt, maxEt,
	 nbinsEt,-1,1);
  BOOK2D(DeltaEtvsEta,"#DeltaE_{T} vs #eta",
	 nbinsEta, minEta, maxEta,
	 nbinsEt,minDeltaEt, maxDeltaEt);
  BOOK2D(DeltaEtOverEtvsEta,"#DeltaE_{T}/E_{T} vs #eta",
	 nbinsEta, minEta, maxEta,
	 100,-1,1);
  BOOK2D(DeltaEtvsPhi,"#DeltaE_{T} vs #phi",
	 200,-M_PI,M_PI,
	 nbinsEt,minDeltaEt, maxDeltaEt);
  BOOK2D(DeltaEtOverEtvsPhi,"#DeltaE_{T}/E_{T} vs #Phi",
	 200,-M_PI,M_PI,
	 100,-1,1);
  BOOK2D(DeltaEtvsDeltaR,"#DeltaE_{T} vs #DeltaR",
	 100,0,1,
	 nbinsEt,minDeltaEt, maxDeltaEt);
  BOOK2D(DeltaEtOverEtvsDeltaR,"#DeltaE_{T}/E_{T} vs #DeltaR",
	 100,0,1,
	 100,-1,1);

  // delta eta quantities
  BOOK1D(DeltaEta,"#Delta#eta",nbinsDeltaEta,minDeltaEta,maxDeltaEta);
  BOOK2D(DeltaEtavsEt,"#Delta#eta vs E_{T}",
	 nbinsEt, minEt, maxEt,
	 nbinsDeltaEta,minDeltaEta,maxDeltaEta);
  BOOK2D(DeltaEtavsEta,"#Delta#eta vs #eta",
	 nbinsEta, minEta, maxEta,
	 nbinsDeltaEta,minDeltaEta,maxDeltaEta);

  // delta phi quantities
  BOOK1D(DeltaPhi,"#Delta#phi",nbinsDeltaPhi,minDeltaPhi,maxDeltaPhi);
  BOOK2D(DeltaPhivsEt,"#Delta#phi vs E_{T}",
	 nbinsEt, minEt, maxEt,
	 nbinsDeltaPhi,minDeltaPhi,maxDeltaPhi);
  BOOK2D(DeltaPhivsEta,"#Delta#phi vs #eta",
	 nbinsEta, minEta, maxEta,
	 nbinsDeltaPhi,minDeltaPhi,maxDeltaPhi);

  // delta R quantities
  BOOK1D(DeltaR,"#DeltaR",100,0,1);
  BOOK2D(DeltaRvsEt,"#DeltaR vs E_{T}",
	 nbinsEt, minEt, maxEt,
	 100,0,1);
  BOOK2D(DeltaRvsEta,"#DeltaR vs #eta",
	 nbinsEta, minEta, maxEta,
	 100,0,1);

  BOOK1D(NRec,"Number of reconstructed objects",20,0,20);

  // seen and gen distributions, for efficiency computation
  BOOK1D(EtaSeen,"seen #eta",100,-5,5);
  BOOK1D(PhiSeen,"seen #phi",100,-3.5,3.5);
  BOOK1D(EtSeen,"seen E_{T}",nbinsEt, minEt, maxEt);
  
  BOOK1D(EtaGen,"generated #eta",100,-5,5);
  BOOK1D(PhiGen,"generated #phi",100,-3.5,3.5);
  BOOK1D(EtGen,"generated E_{T}",nbinsEt, minEt, maxEt);
  
  BOOK1D(NGen,"Number of generated objects",20,0,20);

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

  SETAXES(NRec,"Number of Rec Objects","");
  
  SETAXES(EtaSeen,"seen #eta","");
  SETAXES(PhiSeen,"seen #phi [rad]","");
  SETAXES(EtSeen,"seen E_{T} [GeV]","");

  SETAXES(EtaGen,"generated #eta","");
  SETAXES(PhiGen,"generated #phi [rad]","");
  SETAXES(EtGen,"generated E_{T} [GeV]","");

  SETAXES(NGen,"Number of Gen Objects","");
  
  TDirectory* oldpwd = gDirectory;


  TIter next( gROOT->GetListOfFiles() );
  while ( TFile *file = (TFile *)next() )
    cout<<"file "<<file->GetName()<<endl;


  cout<<"DQM subdir"<<endl;
  cout<< DQM->pwd().c_str()<<endl;

  DQM->cd( DQM->pwd() );

  cout<<"current dir"<<endl;
  gDirectory->pwd();
  
  

  oldpwd->cd();
  //gDirectory->pwd();


  //   tree_ = new BenchmarkTree("genericBenchmark", "Generic Benchmark TTree");
}


void GenericBenchmark::fill(const edm::View<reco::Candidate> *RecoCollection, 
			    const edm::View<reco::Candidate> *GenCollection, 
			    bool startFromGen, 
			    bool PlotAgainstReco, 
			    bool onlyTwoJets, 
			    double recPt_cut, 
			    double minEta_cut, 
			    double maxEta_cut, 
			    double deltaR_cut) {

  // loop over reco particles

  if( !startFromGen) {
    int nRec = 0;
    for (unsigned int i = 0; i < RecoCollection->size(); i++) {
      
      // generate histograms comparing the reco and truth candidate (truth = closest in delta-R)
      const reco::Candidate *particle = &(*RecoCollection)[i];
      
      assert( particle!=NULL ); 
      if( !accepted(particle, recPt_cut, 
		    minEta_cut, maxEta_cut)) continue;

    
      // Count the number of jets with a larger energy
      if( onlyTwoJets ) {
	unsigned highJets = 0;
	for(unsigned j=0; j<RecoCollection->size(); j++) { 
	  const reco::Candidate *otherParticle = &(*RecoCollection)[j];
	  if ( j != i && otherParticle->pt() > particle->pt() ) highJets++;
	}
	if ( highJets > 1 ) continue;
      }		
      nRec++;
      
      const reco::Candidate *gen_particle = algo_->matchByDeltaR(particle,
								 GenCollection);
      if(gen_particle==NULL) continue; 



      // fill histograms
      fillHistos( gen_particle, particle, deltaR_cut, PlotAgainstReco);
    }

    hNRec->Fill(nRec);
  }

  // loop over gen particles
  
  //   cout<<"Reco size = "<<RecoCollection->size()<<", ";
  //   cout<<"Gen size = "<<GenCollection->size()<<endl;

  int nGen = 0;
  for (unsigned int i = 0; i < GenCollection->size(); i++) {

    const reco::Candidate *gen_particle = &(*GenCollection)[i]; 

    if( !accepted(gen_particle, recPt_cut, minEta_cut, maxEta_cut)) {
      continue;
    }

    hEtaGen->Fill(gen_particle->eta() );
    hPhiGen->Fill(gen_particle->phi() );
    hEtGen->Fill(gen_particle->et() );

    const reco::Candidate *rec_particle = algo_->matchByDeltaR(gen_particle,
							       RecoCollection);
    nGen++;
    if(! rec_particle) continue; // no match
    // must make a cut on delta R 

    hEtaSeen->Fill(gen_particle->eta() );
    hPhiSeen->Fill(gen_particle->phi() );
    hEtSeen->Fill(gen_particle->et() );

    if( startFromGen ) 
      fillHistos( gen_particle, rec_particle, deltaR_cut, PlotAgainstReco);

    
  }
  hNGen->Fill(nGen);

}

bool GenericBenchmark::accepted(const reco::Candidate* particle,
				double ptCut,
				double minEtaCut,
				double maxEtaCut ) const {
 
  //skip reconstructed PFJets with p_t < recPt_cut
  if (particle->pt() < ptCut and ptCut != -1.)
    return false;

  if (fabs(particle->eta())>maxEtaCut and maxEtaCut > 0)
    return false;
  if (fabs(particle->eta())<minEtaCut and minEtaCut > 0)
    return false;

  //accepted!
  return true;
 
}


void GenericBenchmark::fillHistos( const reco::Candidate* genParticle,
				   const reco::Candidate* recParticle,
				   double deltaR_cut,
				   bool plotAgainstReco ) {


  
  // get the quantities to place on the denominator and/or divide by
  double et = genParticle->et();
  double eta = genParticle->eta();
  double phi = genParticle->phi();
  
  if (plotAgainstReco) { 
    et = recParticle->et(); 
    eta = recParticle->eta(); 
    phi = recParticle->phi(); 
  }
  
    
  // get the delta quantities
  double deltaEt = algo_->deltaEt(recParticle,genParticle);
  double deltaR = algo_->deltaR(recParticle,genParticle);
  double deltaEta = algo_->deltaEta(recParticle,genParticle);
  double deltaPhi = algo_->deltaPhi(recParticle,genParticle);
   
  //TODO implement variable Cut:
  if (fabs(deltaR)>deltaR_cut and deltaR_cut != -1.)
    return;


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

  BenchmarkTreeEntry entry;
  entry.deltaEt = deltaEt;
  entry.deltaEta = deltaEta;
  entry.et = et;
  entry.eta = eta;
    
  //     tree_->Fill(entry);

}

void GenericBenchmark::write(std::string Filename) {

  if (Filename.size() != 0 && file_)
    file_->Write(Filename.c_str());

}
