#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include <vector>
#include "TH1.h"

class zPdfUnc : public edm::EDAnalyzer {
public:
  zPdfUnc(const edm::ParameterSet& pset);
private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
  virtual void endJob() override;
  edm::EDGetTokenT<reco::GenParticleCollection> genToken_;
  edm::EDGetTokenT<std::vector<double> > pdfweightsToken_;
  unsigned int pdfmember_ , nbinsMass_, nbinsPt_, nbinsAng_ ;
  double massMax_, ptMax_, angMax_;
  double  accPtMin_,accMassMin_,accMassMax_, accEtaMin_, accEtaMax_,  accMassMinDenominator_ ;
  TH1F *h_nZ_;
  TH1F *h_mZMC_, *h_ptZMC_, *h_phiZMC_, *h_thetaZMC_, *h_etaZMC_, *h_rapidityZMC_;
  TH1F *hardpt, *softpt, * hardeta, *softeta;
  TH1F * h_weight_histo;
  bool isMCatNLO_;
  double nAcc_, nAccReW_,  nBothMuHasZHasGrandMa_;
  std::string filename_;

};

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "HepMC/WeightContainer.h"
#include "HepMC/GenEvent.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <cmath>
#include <iostream>
#include <fstream>

using namespace std;
using namespace reco;
using namespace edm;

zPdfUnc::zPdfUnc(const ParameterSet& pset) :
  genToken_(consumes<GenParticleCollection>(pset.getParameter<InputTag>("genParticles"))),
  pdfweightsToken_(consumes<std::vector<double> >(pset.getParameter<InputTag>("pdfweights"))),
  pdfmember_(pset.getUntrackedParameter<unsigned int>("pdfmember")),
  nbinsMass_(pset.getUntrackedParameter<unsigned int>("nbinsMass")),
  nbinsPt_(pset.getUntrackedParameter<unsigned int>("nbinsPt")),
  nbinsAng_(pset.getUntrackedParameter<unsigned int>("nbinsAng")),
  massMax_(pset.getUntrackedParameter<double>("massMax")),
  ptMax_(pset.getUntrackedParameter<double>("ptMax")),
  angMax_(pset.getUntrackedParameter<double>("angMax")),
  accPtMin_(pset.getUntrackedParameter<double>("accPtMin")),
  accMassMin_(pset.getUntrackedParameter<double>("accMassMin")),
  accMassMax_(pset.getUntrackedParameter<double>("accMassMax")),
  accEtaMin_(pset.getUntrackedParameter<double>("accEtaMin")),
  accEtaMax_(pset.getUntrackedParameter<double>("accEtaMax")),
  accMassMinDenominator_(pset.getUntrackedParameter<double>("accMassMinDenominator")),
  isMCatNLO_(pset.getUntrackedParameter<bool>("isMCatNLO")),
  filename_(pset.getUntrackedParameter<std::string>("outfilename")) {
  cout << ">>> Z Histogrammer constructor" << endl;
  Service<TFileService> fs;

  TFileDirectory ZMCHisto = fs->mkdir( "ZMCHisto" );
  h_nZ_ = ZMCHisto.make<TH1F>("ZNumber", "number of Z particles", 11, -0.5, 10.5);
  h_weight_histo  = ZMCHisto.make<TH1F>("weight_histo","weight_histo",20,-10,10);

  h_mZMC_ = ZMCHisto.make<TH1F>("ZMCMass", "Z MC mass (GeV/c^{2})", nbinsMass_,  0, massMax_);
  h_ptZMC_ = ZMCHisto.make<TH1F>("ZMCPt", "Z MC p_{t} (GeV/c)", nbinsPt_, 0, ptMax_);
 hardpt = ZMCHisto.make<TH1F>("hardpt", "hard muon p_{t} (GeV/c)", nbinsPt_, 0, ptMax_);
 softpt = ZMCHisto.make<TH1F>("softpt", "soft muon p_{t} (GeV/c)", nbinsPt_, 0, ptMax_);


  h_phiZMC_ = ZMCHisto.make<TH1F>("ZMCPhi", "Z MC #phi", nbinsAng_,  -angMax_, angMax_);
  h_thetaZMC_ = ZMCHisto.make<TH1F>("ZMCTheta", "Z MC #theta", nbinsAng_,  0, angMax_);
  h_etaZMC_ = ZMCHisto.make<TH1F>("ZMCEta", "Z MC #eta", nbinsAng_,  -angMax_, angMax_);
  h_rapidityZMC_ = ZMCHisto.make<TH1F>("ZMCRapidity", "Z MC y", nbinsAng_,  -angMax_, angMax_);

  hardeta = ZMCHisto.make<TH1F>("hard muon eta", "hard muon #eta", nbinsAng_,  -angMax_, angMax_);
  softeta = ZMCHisto.make<TH1F>("soft muon eta", "soft muon #eta", nbinsAng_,  -angMax_, angMax_);
  nAcc_=0.;
  nAccReW_ =0;
  nBothMuHasZHasGrandMa_=0;

}

void zPdfUnc::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  cout << ">>> Z Histogrammer analyze" << endl;

  Handle<GenParticleCollection> gen;
  Handle< vector<double> > pdfweights;

  event.getByToken(genToken_, gen);
  event.getByToken(pdfweightsToken_, pdfweights );



   // get weight and fill it to histogram
 std::vector<double> weights = (*pdfweights);

  unsigned int nmembers = weights.size();


  double weight =  weights[pdfmember_];

  h_weight_histo->Fill(weight);

  cout << "found nmember: " << nmembers << endl;
  cout << "taken the member n " << pdfmember_ << " weight= " << weight <<   endl;



  std::vector<GenParticle> muons;


    for(unsigned int i = 0; i < gen->size(); ++i){
      const GenParticle & muMC  = (*gen)[i];
      // filling only muons coming form Z
      if (abs(muMC.pdgId())==13 &&  muMC.status()==1  && muMC.numberOfMothers()>0) {
	if (muMC.mother()->numberOfMothers()> 0 ){
	  cout << "I'm getting a muon \n"
	       << "with " << "muMC.numberOfMothers()  " <<  muMC.numberOfMothers() << "\n the first mother has pdgId " << muMC.mother()->pdgId()
	       << "with " << "muMC.mother()->numberOfMothers()  " <<  muMC.mother()->numberOfMothers()<< "\n the first grandma has pdgId " << muMC.mother()->mother()->pdgId()<<endl;
	  if (muMC.mother()->mother()->pdgId() ==23 ) muons.push_back(muMC);
	}
      }
    }

  cout << "finally I selected " << muons.size() << " muons" << endl;

// if there are at least two muons,
   // calculate invarant mass of first two and fill it into histogram

  double inv_mass = 0.0;
   double Zpt_ = 0.0;
   double Zeta_ = 0.0;
   double Ztheta_ = 0.0;
   double Zphi_ = 0.0;
   double Zrapidity_ = 0.0;

   if(muons.size()>1) {
     if (muons[0].mother()->mother()->pdgId()==23 && muons[1].mother()->mother()->pdgId()==23) nBothMuHasZHasGrandMa_ ++;
     math::XYZTLorentzVector tot_momentum(muons[0].p4());
     math::XYZTLorentzVector mom2(muons[1].p4());
     tot_momentum += mom2;
     inv_mass = sqrt(tot_momentum.mass2());

   Zpt_=tot_momentum.pt();
     Zeta_ = tot_momentum.eta();
     Ztheta_ = tot_momentum.theta();
     Zphi_ = tot_momentum.phi();
    Zrapidity_ = tot_momentum.Rapidity();



     // IMPORTANT: use the weight of the event ...

    double weight_sign = weight;
      //double weight_sign = 1.    ;
    // fill the denominator numbers only if mass>accMassMinDenominator and go on in that case
    if (inv_mass > accMassMinDenominator_ ) {
      h_mZMC_->Fill(inv_mass,weight_sign);
      h_ptZMC_->Fill(Zpt_,weight_sign);
      h_etaZMC_->Fill(Zeta_,weight_sign);
      h_thetaZMC_->Fill(Ztheta_,weight_sign);
      h_phiZMC_->Fill(Zphi_,weight_sign);
      h_rapidityZMC_-> Fill (Zrapidity_,weight_sign );

      double pt1 = muons[0].pt();
      double pt2 = muons[1].pt();
      double eta1 = muons[0].eta();
      double eta2 = muons[1].eta();
      if(pt1>pt2) {
	hardpt->Fill(pt1,weight_sign);
	softpt->Fill(pt2,weight_sign);
	hardeta->Fill(eta1,weight_sign);
	softeta->Fill(eta2,weight_sign);
      } else {
	hardpt->Fill(pt2,weight_sign);
	softpt->Fill(pt1,weight_sign);
	hardeta->Fill(eta2,weight_sign);
	softeta->Fill(eta1,weight_sign);
      }

      //evaluating the geometric acceptance
      if ( pt1 >= accPtMin_  && pt2 >= accPtMin_ &&  fabs(eta1)>= accEtaMin_  && fabs(eta2) >= accEtaMin_ && fabs(eta1)<= accEtaMax_  && fabs(eta2) <= accEtaMax_ && inv_mass>= accMassMin_ && inv_mass<= accMassMax_)  {
	nAcc_ ++;
	nAccReW_ +=weight;
      }
    }
   }
}


void zPdfUnc::endJob() {
  cout << " number of events accepted :" << nAcc_ << endl;
  cout << " number of events accepted reweigthed :" << nAccReW_ << endl;
  double nev = h_mZMC_->GetEntries();
  double nev_weigthed = h_mZMC_->Integral( ) ;
  cout << " number of total events :" << nev   << endl;
    cout << " number of total weighted events :" << nev_weigthed << endl;
 cout << " number of cases in which BothMuHasZHasGrandMa :" << nBothMuHasZHasGrandMa_  << endl;
  double eff = (double)nAcc_ / (double) h_mZMC_->GetEntries();
 double eff_rew = (double)nAccReW_ / (double) h_mZMC_->Integral(0,nbinsMass_ +1 );
  double err = sqrt( eff * (1. - eff) / (double) h_mZMC_->GetEntries() );
  double err_rew = sqrt( eff_rew * (1. - eff_rew) / (double) h_mZMC_->Integral(0,nbinsMass_ +1 ) );
  cout << " geometric acceptance: " << eff << "+/-" << err << endl;
  cout << " geometric acceptance reweighted: " << eff_rew << "+/-" << err_rew << endl;

  ofstream myfile;
  myfile.open(filename_.c_str(), std::ios::app);
  myfile<< eff << " "<< eff_rew << " " << nev << " " << nev_weigthed << endl ;
  myfile.close();
}
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(zPdfUnc);

