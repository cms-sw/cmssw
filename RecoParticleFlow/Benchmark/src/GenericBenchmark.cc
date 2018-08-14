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

GenericBenchmark::~GenericBenchmark() noexcept(false) {}

void GenericBenchmark::setup(DQMStore *DQM, bool PlotAgainstReco_, float minDeltaEt, float maxDeltaEt,
			     float minDeltaPhi, float maxDeltaPhi, bool doMetPlots) {

  //std::cout << "minDeltaPhi = " << minDeltaPhi << std::endl;

  // CMSSW_2_X_X
  // use bare Root if no DQM (FWLite applications)
  //if (!DQM)
  //{
  //  file_ = new TFile("pfmetBenchmark.root", "recreate");
  //  cout << "Info: DQM is not available to provide data storage service. Using TFile to save histograms. "<<endl;
  //}
  // Book Histograms

  //std::cout << "FL : pwd = ";
  //gDirectory->pwd();
  //std::cout << std::endl;

  int nbinsEt = 1000;
  float minEt = 0;
  float maxEt = 1000;

  //float minDeltaEt = -100;
  //float maxDeltaEt = 50;

  int nbinsEta = 200;
  float minEta = -5;
  float maxEta = 5;

  int nbinsDeltaEta = 1000;
  float minDeltaEta = -0.5;
  float maxDeltaEta = 0.5;

  int nbinsDeltaPhi = 1000;
  //float minDeltaPhi = -0.5;
  //float maxDeltaPhi = 0.5;


  // delta et quantities
  BOOK1D(DeltaEt,"#DeltaE_{T}", nbinsEt, minDeltaEt, maxDeltaEt);
  BOOK1D(DeltaEx,"#DeltaE_{X}", nbinsEt, minDeltaEt, maxDeltaEt);
  BOOK1D(DeltaEy,"#DeltaE_{Y}", nbinsEt, minDeltaEt, maxDeltaEt);
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
  BOOK2D(EtvsEtaSeen,"seen E_{T} vs eta",100,-5,5,200, 0, 200);
  BOOK2D(EtvsPhiSeen,"seen E_{T} vs seen #phi",100,-3.5,3.5,200, 0, 200);  

  BOOK1D(PhiRec,"Rec #phi",100,-3.5,3.5);
  BOOK1D(EtRec,"Rec E_{T}",nbinsEt, minEt, maxEt);
  BOOK1D(ExRec,"Rec E_{X}",nbinsEt, -maxEt, maxEt);
  BOOK1D(EyRec,"Rec E_{Y}",nbinsEt, -maxEt, maxEt);

  BOOK2D(EtRecvsEt,"Rec E_{T} vs E_{T}",
	 nbinsEt, minEt, maxEt,
	 nbinsEt, minEt, maxEt);
  BOOK2D(EtRecOverTrueEtvsTrueEt,"Rec E_{T} / E_{T} vs E_{T}",
	 nbinsEt, minEt, maxEt,
	 1000, 0., 100.);

  BOOK1D(EtaGen,"generated #eta",100,-5,5);
  BOOK1D(PhiGen,"generated #phi",100,-3.5,3.5);
  BOOK1D(EtGen,"generated E_{T}",nbinsEt, minEt, maxEt);
  BOOK2D(EtvsEtaGen,"generated E_{T} vs generated #eta",100,-5,5,200, 0, 200);  
  BOOK2D(EtvsPhiGen,"generated E_{T} vs generated #phi",100,-3.5,3.5, 200, 0, 200);  

  BOOK1D(NGen,"Number of generated objects",20,0,20);

  if (doMetPlots)
  {
    BOOK1D(SumEt,"SumEt", 1000, 0., 3000.);
    BOOK1D(TrueSumEt,"TrueSumEt", 1000, 0., 3000.);
    BOOK2D(DeltaSetvsSet,"#DeltaSEt vs trueSEt",
	   3000, 0., 3000.,
	   1000,-1000., 1000.);
    BOOK2D(DeltaMexvsSet,"#DeltaMEX vs trueSEt",
	   3000, 0., 3000.,
	   1000,-400., 400.);
    BOOK2D(DeltaSetOverSetvsSet,"#DeltaSetOverSet vs trueSet",
	   3000, 0., 3000.,
	   1000,-1., 1.);
    BOOK2D(RecSetvsTrueSet,"Set vs trueSet",
	   3000, 0., 3000.,
	   3000,0., 3000.);
    BOOK2D(RecSetOverTrueSetvsTrueSet,"Set/trueSet vs trueSet",
	   3000, 0., 3000.,
	   500,0., 2.);
    BOOK2D(TrueMexvsTrueSet,"trueMex vs trueSet",
	   3000,0., 3000.,
	   nbinsEt, -maxEt, maxEt);
  }
  // number of truth particles found within given cone radius of reco
  //BOOK2D(NumInConeVsConeSize,NumInConeVsConeSize,100,0,1,25,0,25);

  // Set Axis Titles
 
  // delta et quantities
  SETAXES(DeltaEt,"#DeltaE_{T} [GeV]","");
  SETAXES(DeltaEx,"#DeltaE_{X} [GeV]","");
  SETAXES(DeltaEy,"#DeltaE_{Y} [GeV]","");
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
  SETAXES(EtvsEtaSeen,"seen #eta","seen E_{T}");
  SETAXES(EtvsPhiSeen,"seen #phi [rad]","seen E_{T}");

  SETAXES(PhiRec,"#phi [rad]","");
  SETAXES(EtRec,"E_{T} [GeV]","");
  SETAXES(ExRec,"E_{X} [GeV]","");
  SETAXES(EyRec,"E_{Y} [GeV]","");
  SETAXES(EtRecvsEt,ET,"Rec E_{T} [GeV]");
  SETAXES(EtRecOverTrueEtvsTrueEt,ET,"Rec E_{T} / E_{T} [GeV]");

  SETAXES(EtaGen,"generated #eta","");
  SETAXES(PhiGen,"generated #phi [rad]","");
  SETAXES(EtGen,"generated E_{T} [GeV]","");
  SETAXES(EtvsPhiGen,"generated #phi [rad]","generated E_{T} [GeV]");
  SETAXES(EtvsEtaGen,"generated #eta","generated E_{T} [GeV]");

  SETAXES(NGen,"Number of Gen Objects","");
  
  if (doMetPlots)
  {
    SETAXES(SumEt,"SumEt [GeV]","");
    SETAXES(TrueSumEt,"TrueSumEt [GeV]","");
    SETAXES(DeltaSetvsSet,"TrueSumEt","#DeltaSumEt [GeV]");
    SETAXES(DeltaMexvsSet,"TrueSumEt","#DeltaMEX [GeV]");
    SETAXES(DeltaSetOverSetvsSet,"TrueSumEt","#DeltaSumEt/trueSumEt");
    SETAXES(RecSetvsTrueSet,"TrueSumEt","SumEt");
    SETAXES(RecSetOverTrueSetvsTrueSet,"TrueSumEt","SumEt/trueSumEt");
    SETAXES(TrueMexvsTrueSet,"TrueSumEt","TrueMEX");
  }

  TDirectory* oldpwd = gDirectory;


  TIter next( gROOT->GetListOfFiles() );

  const bool debug=false;

  while ( TFile *file = (TFile *)next() )
  {
    if (debug) cout<<"file "<<file->GetName()<<endl;
  }
  if (DQM)
  {
    cout<<"DQM subdir"<<endl;
    cout<< DQM->pwd().c_str()<<endl;

    DQM->cd( DQM->pwd() );
  }

  if (debug)
  {
    cout<<"current dir"<<endl;
    gDirectory->pwd();
  }
  
  oldpwd->cd();
  //gDirectory->pwd();

  doMetPlots_=doMetPlots;

  //   tree_ = new BenchmarkTree("genericBenchmark", "Generic Benchmark TTree");
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
  //std::cout << "FL : et = " << et << std::endl;
  //std::cout << "FL : eta = " << eta << std::endl;
  //std::cout << "FL : phi = " << phi << std::endl;
  //std::cout << "FL : rec et = " << recParticle->et() << std::endl;
  //std::cout << "FL : rec eta = " << recParticle->eta() << std::endl;
  //std::cout << "FL : rec phi = " <<recParticle-> phi() << std::endl;

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
   
  //std::cout << "FL :deltaR_cut = " << deltaR_cut << std::endl;
  //std::cout << "FL :deltaR = " << deltaR << std::endl;

  if (deltaR>deltaR_cut && deltaR_cut != -1.)
    return;  

  hDeltaEt->Fill(deltaEt);
  hDeltaEx->Fill(recParticle->px()-genParticle->px());
  hDeltaEy->Fill(recParticle->py()-genParticle->py());
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

  if (doMetPlots_)
  {
    const reco::MET* met1=static_cast<const reco::MET*>(genParticle);
    const reco::MET* met2=static_cast<const reco::MET*>(recParticle);
    if (met1!=nullptr && met2!=nullptr)
    {
      //std::cout << "FL: met1.sumEt() = " << (*met1).sumEt() << std::endl;
      hTrueSumEt->Fill((*met1).sumEt());
      hSumEt->Fill((*met2).sumEt());
      hDeltaSetvsSet->Fill((*met1).sumEt(),(*met2).sumEt()-(*met1).sumEt());
      hDeltaMexvsSet->Fill((*met1).sumEt(),recParticle->px()-genParticle->px());
      hDeltaMexvsSet->Fill((*met1).sumEt(),recParticle->py()-genParticle->py());
      if ((*met1).sumEt()>0.01) hDeltaSetOverSetvsSet->Fill((*met1).sumEt(),((*met2).sumEt()-(*met1).sumEt())/(*met1).sumEt());
      hRecSetvsTrueSet->Fill((*met1).sumEt(),(*met2).sumEt());
      hRecSetOverTrueSetvsTrueSet->Fill((*met1).sumEt(),(*met2).sumEt()/((*met1).sumEt()));
      hTrueMexvsTrueSet->Fill((*met1).sumEt(),(*met1).px());
      hTrueMexvsTrueSet->Fill((*met1).sumEt(),(*met1).py());
    }
    else
    {
      std::cout << "Warning : reco::MET* == NULL" << std::endl;
    }
  }

  //     tree_->Fill(entry);

}

void GenericBenchmark::write(std::string Filename) {

  if (!Filename.empty() && file_)
    file_->Write(Filename.c_str());

}

void GenericBenchmark::setfile(TFile *file)
{
  file_=file;
}
