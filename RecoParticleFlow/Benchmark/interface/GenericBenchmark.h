#ifndef RecoParticleFlow_Benchmark_GenericBenchmark_h
#define RecoParticleFlow_Benchmark_GenericBenchmark_h

//COLIN: necessary?
#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"


#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/MET.h"

#include <string>

#include <TH1F.h>
#include <TH2F.h>
#include <TFile.h>

class DQMStore; // CMSSW_2_X_X

class BenchmarkTree;

class GenericBenchmark{

 public:

  GenericBenchmark();
  virtual ~GenericBenchmark();

  void setup(DQMStore *DQM = NULL, 
	     bool PlotAgainstReco_=true, 
	     float minDeltaEt = -100., float maxDeltaEt = 50., 
	     float minDeltaPhi = -0.5, float maxDeltaPhi = 0.5,
	     bool doMetPlots=false);

  template< typename C>
  void fill(const C *RecoCollection, 
	    const C *GenCollection,
	    bool startFromGen=false, 
	    bool PlotAgainstReco =true, 
	    bool onlyTwoJets = false, 
	    double recPt_cut = -1., 
	    double minEta_cut = -1., 
	    double maxEta_cut = -1., 
	    double deltaR_cut = -1.);


  void write(std::string Filename);

  void setfile(TFile *file);

 private:
  
  bool accepted(const reco::Candidate* particle,
		double ptCut,
		double minEtaCut,
		double maxEtaCut ) const;

  void fillHistos( const reco::Candidate* genParticle,
		   const reco::Candidate* recParticle,
		   double deltaR_cut,
		   bool plotAgainstReco);

  TFile *file_;

  TH1F *hDeltaEt;
  TH1F *hDeltaEx;
  TH1F *hDeltaEy;
  TH2F *hDeltaEtvsEt;
  TH2F *hDeltaEtOverEtvsEt;
  TH2F *hDeltaEtvsEta;
  TH2F *hDeltaEtOverEtvsEta;
  TH2F *hDeltaEtvsPhi;
  TH2F *hDeltaEtOverEtvsPhi;
  TH2F *hDeltaEtvsDeltaR;
  TH2F *hDeltaEtOverEtvsDeltaR;

  TH2F *hEtRecvsEt;
  TH2F *hEtRecOverTrueEtvsTrueEt;

  TH1F *hDeltaEta;
  TH2F *hDeltaEtavsEt;
  TH2F *hDeltaEtavsEta;

  TH1F *hDeltaPhi;
  TH2F *hDeltaPhivsEt;
  TH2F *hDeltaPhivsEta;

  TH1F *hDeltaR;
  TH2F *hDeltaRvsEt;
  TH2F *hDeltaRvsEta;

  TH1F *hNRec;


  TH1F *hEtGen;
  TH1F *hEtaGen;
  TH1F *hPhiGen;

  TH1F *hNGen;

  TH1F *hEtSeen;
  TH1F *hEtaSeen;
  TH1F *hPhiSeen;

  TH1F *hEtRec;
  TH1F *hExRec;
  TH1F *hEyRec;
  TH1F *hPhiRec;

  TH1F *hSumEt;
  TH1F *hTrueSumEt;
  TH2F *hDeltaSetvsSet;
  TH2F *hDeltaMexvsSet;
  TH2F *hDeltaSetOverSetvsSet;
  TH2F *hRecSetvsTrueSet;
  TH2F *hRecSetOverTrueSetvsTrueSet;
  TH2F *hTrueMexvsTrueSet;

  BenchmarkTree*  tree_;

  bool doMetPlots_;

 protected:

  DQMStore *dbe_;
  PFBenchmarkAlgo *algo_;

};

template< typename C>
void GenericBenchmark::fill(const C *RecoCollection, 
			    const C *GenCollection,
			    bool startFromGen, 
			    bool PlotAgainstReco, 
			    bool onlyTwoJets, 
			    double recPt_cut, 
			    double minEta_cut, 
			    double maxEta_cut, 
			    double deltaR_cut) {

  //if (doMetPlots_)
  //{
  //  const reco::MET* met1=static_cast<const reco::MET*>(&((*RecoCollection)[0]));
  //  if (met1!=NULL) std::cout << "FL: met1.sumEt() = " << (*met1).sumEt() << std::endl;
  //}

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

    hPhiRec->Fill(rec_particle->phi() );
    hEtRec->Fill(rec_particle->et() );
    hExRec->Fill(rec_particle->px() );
    hEyRec->Fill(rec_particle->py() );

    hEtRecvsEt->Fill(gen_particle->et(),rec_particle->et());
    hEtRecOverTrueEtvsTrueEt->Fill(gen_particle->et(),rec_particle->et()/gen_particle->et());

    if( startFromGen ) 
      fillHistos( gen_particle, rec_particle, deltaR_cut, PlotAgainstReco);

    
  }
  hNGen->Fill(nGen);

}


#endif // RecoParticleFlow_Benchmark_GenericBenchmark_h
