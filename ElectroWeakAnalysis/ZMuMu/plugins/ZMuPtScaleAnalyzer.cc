#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "TH1.h"
#include "TRandom3.h"


class ZMuPtScaleAnalyzer : public edm::EDAnalyzer {
public:
  ZMuPtScaleAnalyzer(const edm::ParameterSet& pset);
private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
  virtual void endJob() override;
  edm::EDGetTokenT<reco::GenParticleCollection>   genToken_;
  unsigned int nbinsMass_, nbinsPt_, nbinsAng_;
  double massMax_, ptMax_, angMax_;
  double  accPtMin_,accMassMin_,accMassMax_, accMassMinDen_,accMassMaxDen_, accEtaMin_, accEtaMax_, ptScale_;
  TH1F *h_nZ_, *h_mZ_, *h_ptZ_, *h_phiZ_, *h_thetaZ_, *h_etaZ_, *h_rapidityZ_;
  TH1F *h_mZMC_, *h_ptZMC_, *h_phiZMC_, *h_thetaZMC_, *h_etaZMC_, *h_rapidityZMC_;
  TH1F *hardpt, *softpt, * hardeta, *softeta;
  unsigned int nAcc_,nAccPtScaleP_,nAccPtScaleN_, nAccPtScaleSmearedFlat_ , nAccPtScaleSmearedGaus_,  nBothMuHasZHasGrandMa_;
 int  muPdgStatus_;
};

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
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

using namespace std;
using namespace reco;
using namespace edm;

ZMuPtScaleAnalyzer::ZMuPtScaleAnalyzer(const ParameterSet& pset) :
  genToken_(consumes<GenParticleCollection>(pset.getParameter<InputTag>("genParticles"))),
  nbinsMass_(pset.getUntrackedParameter<unsigned int>("nbinsMass")),
  nbinsPt_(pset.getUntrackedParameter<unsigned int>("nbinsPt")),
  nbinsAng_(pset.getUntrackedParameter<unsigned int>("nbinsAng")),
  massMax_(pset.getUntrackedParameter<double>("massMax")),
  ptMax_(pset.getUntrackedParameter<double>("ptMax")),
  angMax_(pset.getUntrackedParameter<double>("angMax")),
  accPtMin_(pset.getUntrackedParameter<double>("accPtMin")),
  accMassMin_(pset.getUntrackedParameter<double>("accMassMin")),
  accMassMax_(pset.getUntrackedParameter<double>("accMassMax")),
  accMassMinDen_(pset.getUntrackedParameter<double>("accMassMinDen")),
  accMassMaxDen_(pset.getUntrackedParameter<double>("accMassMaxDen")),
  accEtaMin_(pset.getUntrackedParameter<double>("accEtaMin")),
  accEtaMax_(pset.getUntrackedParameter<double>("accEtaMax")),
  ptScale_(pset.getUntrackedParameter<double>("ptScale")),
  muPdgStatus_(pset.getUntrackedParameter<int>("muPdgStatus")){

  cout << ">>> Z Histogrammer constructor" << endl;
  Service<TFileService> fs;
  TFileDirectory ZMCHisto = fs->mkdir( "ZMCHisto" );

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
  nAcc_=0;
  nAccPtScaleP_=0;
  nAccPtScaleN_=0;
  nAccPtScaleSmearedFlat_=0;
  nAccPtScaleSmearedGaus_=0;
  nBothMuHasZHasGrandMa_=0;}



void ZMuPtScaleAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  cout << ">>> Z HistogrammerZLONLOHistogrammer.cc analyze" << endl;

  Handle<GenParticleCollection> gen;
  Handle<double> weights;

  event.getByToken(genToken_, gen);




   // get weight and fill it to histogram




  std::vector<GenParticle> muons;

  double mZGen = -100;

    for(unsigned int i = 0; i < gen->size(); ++i){
      const GenParticle & muMC  = (*gen)[i];
      // filliScaledPng only muons coming form Z
      if (abs(muMC.pdgId())==13 &&  muMC.status()==muPdgStatus_  && muMC.numberOfMothers()>0) {

	  cout << "I'm getting a muon \n"
	       << "with " << "muMC.numberOfMothers()  " <<  muMC.numberOfMothers() << "\n the first mother has pdgId " << muMC.mother()->pdgId()
	       << "with " << "muMC.mother()->numberOfMothers()  " <<  muMC.mother()->numberOfMothers()<< "\n the first grandma has pdgId " << muMC.mother()->mother()->pdgId()<<endl;
	       cout << "with  muMC.eta() " <<  muMC.eta()<<endl;
	       muons.push_back(muMC);
      }
	  // introducing here the gen mass cut......................
	       /*
   if (muPdgStatus_ ==1) {
	    mZGen = muMC.mother()->mother()->mass();
	    if (muMC.mother()->mother()->pdgId() ==23  && mZGen>accMassMinDen_ && mZGen<accMassMaxDen_ ) muons.push_back(muMC);}
	}
          if (muPdgStatus_ ==3) {
	     mZGen = muMC.mother()->mass();
	    if (muMC.mother()->pdgId() ==23  && mZGen>accMassMinDen_ && mZGen<accMassMaxDen_ ) muons.push_back(muMC);}
	}
*/


      const GenParticle & zMC  = (*gen)[i];
            if (zMC.pdgId()==23 &&  zMC.status()==3 &&zMC.numberOfDaughters()>1  ) {
        mZGen = zMC.mass();
	cout << "I'm selecting a Z MC with mass " << mZGen << endl;
	  if(mZGen>accMassMinDen_ && mZGen<accMassMaxDen_ ) h_mZMC_->Fill(mZGen);



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




      double weight_sign = 1.    ;

      //h_mZMC_->Fill(inv_mass);
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
   if ( pt1 >= accPtMin_  && pt2 >= accPtMin_ &&  fabs(eta1)>= accEtaMin_  && fabs(eta2) >= accEtaMin_ && fabs(eta1)<= accEtaMax_  && fabs(eta2) <= accEtaMax_ && inv_mass>= accMassMin_ && inv_mass<= accMassMax_) nAcc_++;


   cout << "pt1" <<  pt1 << endl;

     // scaling the muon pt
   double  pt1ScaledP = pt1* ( 1. + ptScale_);
  cout << "pt1 ScaledP of " <<  ( 1. + ptScale_) << endl;
   cout << "pt1ScaledP" <<  pt1ScaledP << endl;

   double  pt2ScaledP = pt2 * ( 1. + ptScale_);


   //evaluating the geometric acceptance
   if ( pt1ScaledP >= accPtMin_  && pt2ScaledP >= accPtMin_ &&  fabs(eta1)>= accEtaMin_  && fabs(eta2) >= accEtaMin_ && fabs(eta1)<= accEtaMax_  && fabs(eta2) <= accEtaMax_ && inv_mass>= accMassMin_ && inv_mass<= accMassMax_)
nAccPtScaleP_++;




     // scaling the muon pt
   double  pt1ScaledN = pt1* ( 1. - ptScale_);
   double  pt2ScaledN = pt2 * ( 1. - ptScale_);


   //evaluating the geometric acceptance
   if ( pt1ScaledN >= accPtMin_  && pt2ScaledN >= accPtMin_ &&  fabs(eta1)>= accEtaMin_  && fabs(eta2) >= accEtaMin_ && fabs(eta1)<= accEtaMax_  && fabs(eta2) <= accEtaMax_ && inv_mass>= accMassMin_ && inv_mass<= accMassMax_)
nAccPtScaleN_++;


  // scaling the muon pt
   TRandom3 f;
   f.SetSeed(123456789);
   double  pt1SmearedFlat = pt1* ( 1. + ptScale_ * f.Uniform() );
   double  pt2SmearedFlat = pt2 * ( 1. + ptScale_ * f.Uniform() ) ;


   //evaluating the geometric acceptance
   if ( pt1SmearedFlat >= accPtMin_  && pt2SmearedFlat >= accPtMin_ &&  fabs(eta1)>= accEtaMin_  && fabs(eta2) >= accEtaMin_ && fabs(eta1)<= accEtaMax_  && fabs(eta2) <= accEtaMax_ && inv_mass>= accMassMin_ && inv_mass<= accMassMax_)
 nAccPtScaleSmearedFlat_++;


// scaling the muon pt
   TRandom3 ff;
   ff.SetSeed(123456789);
   double  pt1SmearedGaus = pt1* ( 1. + ptScale_ * f.Gaus() );
   double  pt2SmearedGaus = pt2 * ( 1. + ptScale_ * f.Gaus() ) ;


   //evaluating the geometric acceptance
   if ( pt1SmearedGaus >= accPtMin_  && pt2SmearedGaus >= accPtMin_ &&  fabs(eta1)>= accEtaMin_  && fabs(eta2) >= accEtaMin_ && fabs(eta1)<= accEtaMax_  && fabs(eta2) <= accEtaMax_ && inv_mass>= accMassMin_ && inv_mass<= accMassMax_)
 nAccPtScaleSmearedGaus_++;




   }}




void ZMuPtScaleAnalyzer::endJob() {
  cout << " number of events accepted :" << nAcc_ << endl;
  cout << " number of total events :" << h_mZMC_->GetEntries()  << endl;
   cout << " number of cases in which BothMuHasZHasGrandMa :" << nBothMuHasZHasGrandMa_  << endl;
  cout << " number of events pt scaled positively accepted :" << nAccPtScaleP_ << endl;

  cout << " number of events pt scaled negatively accepted :" << nAccPtScaleN_ << endl;

cout << " number of events pt scaled smeared flattely accepted :" << nAccPtScaleSmearedFlat_ << endl;

cout << " number of events pt scaled smeared gaussianely accepted :" << nAccPtScaleSmearedGaus_ << endl;


  double eff = (double)nAcc_ / (double) h_mZMC_->GetEntries();
  double err = sqrt( eff * (1. - eff) / (double) h_mZMC_->GetEntries() );
  cout << " geometric acceptance: " << eff << "+/-" << err << endl;

 double effScaledP = (double)nAccPtScaleP_ / (double) h_mZMC_->GetEntries();
  double errScaledP = sqrt( effScaledP * (1. - effScaledP) / (double) h_mZMC_->GetEntries() );
  cout << " geometric acceptance when pt muon is positively scaled: " << effScaledP << "+/-" << errScaledP << endl;

 double effScaledN = (double)nAccPtScaleN_ / (double) h_mZMC_->GetEntries();
  double errScaledN = sqrt( effScaledN * (1. - effScaledN) / (double) h_mZMC_->GetEntries() );
  cout << " geometric acceptance when pt muon is negatively scaled: " << effScaledN << "+/-" << errScaledN << endl;

 double effSmearedFlat = (double) nAccPtScaleSmearedFlat_ / (double) h_mZMC_->GetEntries();
  double errSmearedFlat = sqrt( effSmearedFlat * (1. - effSmearedFlat) / (double) h_mZMC_->GetEntries() );
  cout << " geometric acceptance when pt muon is scaled with a flat smaering: " << effSmearedFlat << "+/-" << errSmearedFlat << endl;

 double effSmearedGaus = (double) nAccPtScaleSmearedGaus_ / (double) h_mZMC_->GetEntries();
  double errSmearedGaus = sqrt( effSmearedGaus * (1. - effSmearedGaus) / (double) h_mZMC_->GetEntries() );
  cout << " geometric acceptance when pt muon is scaled with a gaussian smearing: " << effSmearedGaus << "+/-" << errSmearedGaus << endl;


}
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZMuPtScaleAnalyzer);

