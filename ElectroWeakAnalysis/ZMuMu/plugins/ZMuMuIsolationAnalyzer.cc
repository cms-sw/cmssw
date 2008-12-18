#include "DataFormats/Common/interface/AssociationVector.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "PhysicsTools/Utilities/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/PatCandidates/interface/Isolation.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "TH1.h"
#include "TH2.h"
#include <vector>
#include <string>
#include <iostream>
#include <sstream>

using namespace edm;
using namespace std;
using namespace reco;
using namespace isodeposit;

class ZMuMuIsolationAnalyzer : public edm::EDAnalyzer {
public:
  ZMuMuIsolationAnalyzer(const edm::ParameterSet& pset);
private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  virtual void endJob();
  InputTag src_muons;
  double dRVeto;
  double dRTrk, dREcal, dRHcal;
  double ptThreshold, etEcalThreshold, etHcalThreshold;
  double alpha, beta;
  double pt, eta;

  TH1F * h_IsoZ_tk,* h_IsoW_tk,* h_IsoOther_tk ;
  TH1F * h_IsoZ_ecal,* h_IsoW_ecal,* h_IsoOther_ecal ;
  TH1F * h_IsoZ_hcal,* h_IsoW_hcal,* h_IsoOther_hcal ;
  TH1F * IsoZ,* IsoW,* IsoOther ;
  TH1F * TkrPt,* EcalEt,* HcalEt ;
  TH1F * EcalEtZ, * HcalEtZ;
  enum MuTag { muFromZ, muFromW, muFromOther };
  template<typename T>
  MuTag muTag(const T & mu) const;
  void Deposits(const pat::IsoDeposit* isodep, double dR_max,  TH1F* hist);
  void histo(const TH1F* hist, char* cx, char* cy) const;
};

template<typename T>
ZMuMuIsolationAnalyzer::MuTag ZMuMuIsolationAnalyzer::muTag(const T& mu) const {
  GenParticleRef p = mu.genParticleRef();
  if(p.isNull()) return muFromOther;
  int sizem = p->numberOfMothers();
  if(sizem != 1) return muFromOther;
  const Candidate * moth1 = p->mother();
  if(moth1 == 0) return muFromOther;
  int pdgId1 = moth1->pdgId();
  if(abs(pdgId1)!=13) return muFromOther;
  const Candidate * moth2 = moth1->mother();
  if(moth2 == 0) return muFromOther;
  int pdgId2 = moth2->pdgId();
  if(pdgId2 == 23) return muFromZ;
  if(abs(pdgId2)==24) return muFromW;
  return muFromOther;
}

void ZMuMuIsolationAnalyzer::Deposits(const pat::IsoDeposit* isodep,double dR_max,TH1F* hist){
  for(IsoDeposit::const_iterator it= isodep->begin(); it!= isodep->end(); ++it){
    if(it->dR()<dR_max) {
      double theta= 2*(TMath::ATan(TMath::Exp(-(it->eta() ) ) ) );
      // double theta= 2;
      hist->Fill(it->value()/TMath::Sin(theta));
    }
  }
}

void ZMuMuIsolationAnalyzer::histo(const TH1F* hist,char* cx, char*cy) const{
  hist->GetXaxis()->SetTitle(cx);
  hist->GetYaxis()->SetTitle(cy);
  hist->GetXaxis()->SetTitleOffset(1);
  hist->GetYaxis()->SetTitleOffset(1.2);
  hist->GetXaxis()->SetTitleSize(0.04);
  hist->GetYaxis()->SetTitleSize(0.04);
  hist->GetXaxis()->SetLabelSize(0.03);
  hist->GetYaxis()->SetLabelSize(0.03);
}

ZMuMuIsolationAnalyzer::ZMuMuIsolationAnalyzer(const ParameterSet& pset):
  src_muons(pset.getParameter<InputTag>("src")),
  dRVeto(pset.getUntrackedParameter<double>("veto")),
  dRTrk(pset.getUntrackedParameter<double>("deltaRTrk")),
  dREcal(pset.getUntrackedParameter<double>("deltaREcal")),
  dRHcal(pset.getUntrackedParameter<double>("deltaRHcal")),
  ptThreshold(pset.getUntrackedParameter<double>("ptThreshold")),
  etEcalThreshold(pset.getUntrackedParameter<double>("etEcalThreshold")),
  etHcalThreshold(pset.getUntrackedParameter<double>("etHcalThreshold")),
  alpha(pset.getUntrackedParameter<double>("alpha")),
  beta(pset.getUntrackedParameter<double>("beta")),
  pt(pset.getUntrackedParameter<double>("pt")),
  eta(pset.getUntrackedParameter<double>("eta")) {
  edm::Service<TFileService> fs;
  std::ostringstream str1,str2,str3,str4,str5,str6,str7,str8,str9,str10,n_tracks;
  str1 << "muons from Z with p_{t} > " << ptThreshold << " GeV/c"<<" and #Delta R < "<<dRTrk;
  str2 << "muons from W with p_{t} > " << ptThreshold << " GeV/c"<<" and #Delta R < "<<dRTrk;
  str3 << "muons from Others with p_{t} > " << ptThreshold << " GeV/c"<<" and #Delta R < "<<dRTrk;
  str4 << "muons from Z with p_{t} > " << ptThreshold << " GeV/c"<<" and #Delta R < "<<dRTrk<<" (alpha = "<<alpha<<" , "<<"beta = "<<beta<<" )";
  str5 << "muons from W with p_{t} > " << ptThreshold << " GeV/c"<<" and #Delta R < "<<dRTrk<<" (alpha = "<<alpha<<" , "<<"beta = "<<beta<<" )";
  str6 << "muons from Other with p_{t} > " << ptThreshold << " GeV/c"<<" and #Delta R < "<<dRTrk<<" (alpha = "<<alpha<<" , "<<"beta = "<<beta<<" )";
  n_tracks <<"Number of tracks for muon with p_{t} > " << ptThreshold <<" and #Delta R < "<<dRTrk<< " GeV/c";
  str7<<"Isolation Vs p_{t} with p_{t} > " << ptThreshold << " GeV/c"<<" and #Delta R < "<<dRTrk<<"(Tracker)";
  str8<<"Isolation Vs p_{t} with p_{t} > " << ptThreshold << " GeV/c"<<" and #Delta R < "<<dRTrk<<"(Ecal)";
  str9<<"Isolation Vs p_{t} with p_{t} > " << ptThreshold << " GeV/c"<<" and #Delta R < "<<dRTrk<<"(Hcal)";
  str10<<"Isolation Vs p_{t} with p_{t} > " << ptThreshold << " GeV/c"<<" and #Delta R < "<<dRTrk<<" (alpha = "<<alpha<<" , "<<"beta = "<<beta<<" )";
  h_IsoZ_tk = fs->make<TH1F>("ZIso_Tk",str1.str().c_str(),100,0.,20.);
  h_IsoW_tk = fs->make<TH1F>("WIso_Tk",str2.str().c_str(),100,0.,20.);
  h_IsoOther_tk = fs->make<TH1F>("otherIso_Tk",str3.str().c_str(),100,0.,20.);
  h_IsoZ_ecal = fs->make<TH1F>("ZIso_ecal",str1.str().c_str(),100,0.,20.);
  h_IsoW_ecal = fs->make<TH1F>("WIso_ecal",str2.str().c_str(),100,0.,20.);
  h_IsoOther_ecal = fs->make<TH1F>("otherIso_ecal",str3.str().c_str(),100,0.,20.);
  h_IsoZ_hcal = fs->make<TH1F>("ZIso_hcal",str1.str().c_str(),100,0.,20.);
  h_IsoW_hcal = fs->make<TH1F>("WIso_hcal",str2.str().c_str(),100,0.,20.);
  h_IsoOther_hcal = fs->make<TH1F>("otherIso_hcal",str3.str().c_str(),100,0.,20.);
  IsoZ = fs->make<TH1F>("ZIso",str4.str().c_str(),100,0.,20.);
  IsoW = fs->make<TH1F>("WIso",str5.str().c_str(),100,0.,20.);
  IsoOther = fs->make<TH1F>("otherIso",str6.str().c_str(),100,0.,20.);

  TkrPt = fs->make<TH1F>("TkrPt","IsoDeposit p distribution in the Tracker",100,0.,10.);
  EcalEt = fs->make<TH1F>("EcalEt","IsoDeposit E distribution in the Ecal",100,0.,5.);
  HcalEt = fs->make<TH1F>("HcalEt","IsoDeposit E distribution in the Hcal",100,0.,5.);

  EcalEtZ = fs->make<TH1F>("VetoEcalEt"," #Sigma E_{T} deposited in veto cone in the Ecal",100,0.,10.);
  HcalEtZ = fs->make<TH1F>("VetoHcalEt"," #Sigma E_{T} deposited in veto cone in the Hcal",100,0.,10.);
}

void ZMuMuIsolationAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  Handle<CandidateView> dimuons;
  event.getByLabel(src_muons,dimuons);

  for(unsigned int i=0; i < dimuons->size(); ++ i ) {
    const Candidate & zmm = (* dimuons)[i];
    const Candidate * dau0 = zmm.daughter(0);
    const Candidate * dau1 = zmm.daughter(1);
    const pat::Muon & mu0 = dynamic_cast<const pat::Muon &>(*dau0->masterClone());
    const pat::GenericParticle & mu1 = dynamic_cast<const pat::GenericParticle &>(*dau1->masterClone());

    const pat::IsoDeposit * muTrackIso =mu0.trackerIsoDeposit();
    const pat::IsoDeposit * tkTrackIso =mu1.trackerIsoDeposit();
    const pat::IsoDeposit * muEcalIso =mu0.ecalIsoDeposit();
    const pat::IsoDeposit * tkEcalIso =mu1.ecalIsoDeposit();
    const pat::IsoDeposit * muHcalIso =mu0.hcalIsoDeposit();
    const pat::IsoDeposit * tkHcalIso =mu1.hcalIsoDeposit();

   
    if(mu0.pt() > pt && mu1.pt() > pt && abs(mu0.eta()) < eta && abs(mu1.eta()) < eta){

      Direction muDir = Direction(mu0.eta(),mu0.phi());
      Direction tkDir = Direction(mu1.eta(),mu1.phi());

      IsoDeposit::AbsVetos vetos_mu;
      vetos_mu.push_back(new ConeVeto( muDir, dRVeto ));
      vetos_mu.push_back(new ThresholdVeto( ptThreshold ));

      reco::IsoDeposit::AbsVetos vetos_tk;
      vetos_tk.push_back(new ConeVeto( tkDir, dRVeto ));
      vetos_tk.push_back(new ThresholdVeto( ptThreshold ));
      
      reco::IsoDeposit::AbsVetos vetos_mu_ecal;
      vetos_mu_ecal.push_back(new ConeVeto( muDir, 0. ));
      vetos_mu_ecal.push_back(new ThresholdVeto( etEcalThreshold ));
      
      reco::IsoDeposit::AbsVetos vetos_tk_ecal;
      vetos_tk_ecal.push_back(new ConeVeto( tkDir, 0. ));
      vetos_tk_ecal.push_back(new ThresholdVeto( etEcalThreshold ));
      
      reco::IsoDeposit::AbsVetos vetos_mu_hcal;
      vetos_mu_hcal.push_back(new ConeVeto( muDir, 0. ));
      vetos_mu_hcal.push_back(new ThresholdVeto( etHcalThreshold ));
      
      reco::IsoDeposit::AbsVetos vetos_tk_hcal;
      vetos_tk_hcal.push_back(new ConeVeto( tkDir, 0. ));
      vetos_tk_hcal.push_back(new ThresholdVeto( etHcalThreshold ));
      MuTag tag_mu = muTag(mu0);
      MuTag tag_track = muTag(mu1);
     
      double  Tk_isovalue = TMath::Max(muTrackIso->sumWithin(dRTrk,vetos_mu),tkTrackIso->sumWithin(dRTrk, vetos_tk));
      double  Ecal_isovalue = TMath::Max(muEcalIso->sumWithin(dREcal,vetos_mu_ecal),tkEcalIso->sumWithin(dREcal, vetos_tk_ecal));
      double  Hcal_isovalue = TMath::Max(muHcalIso->sumWithin(dRHcal,vetos_mu_hcal),tkHcalIso->sumWithin(dRHcal, vetos_tk_hcal));
      EcalEtZ->Fill(muEcalIso->candEnergy());
      EcalEtZ->Fill(tkEcalIso->candEnergy());
      HcalEtZ->Fill(muHcalIso->candEnergy());
      HcalEtZ->Fill(tkHcalIso->candEnergy());
      double iso_value = alpha*((0.5*(1+beta)* Ecal_isovalue) + (0.5*(1-beta)*Hcal_isovalue)) +(1-alpha)*Tk_isovalue;
      
      if(tag_mu==muFromZ && tag_track==muFromZ){
	h_IsoZ_tk->Fill(Tk_isovalue);
	h_IsoZ_ecal->Fill(Ecal_isovalue);
	h_IsoZ_hcal->Fill(Hcal_isovalue);
	IsoZ->Fill(iso_value);
	Deposits(muTrackIso,dRTrk,TkrPt);
	Deposits(muEcalIso,dREcal,EcalEt);
	Deposits(muHcalIso,dRHcal,HcalEt);
	Deposits(tkTrackIso,dRTrk,TkrPt);
	Deposits(tkEcalIso,dREcal,EcalEt);
	Deposits(tkHcalIso,dRHcal,HcalEt);
      }
      if(tag_mu==muFromW || tag_track==muFromW){ 
	h_IsoW_tk->Fill(Tk_isovalue);
	h_IsoW_ecal->Fill(Ecal_isovalue);
	h_IsoW_hcal->Fill(Hcal_isovalue);
	IsoW->Fill(iso_value);
	Deposits(muTrackIso,dRTrk,TkrPt);
	Deposits(muEcalIso,dREcal,EcalEt);
	Deposits(muHcalIso,dRHcal,HcalEt);
	Deposits(tkTrackIso,dRTrk,TkrPt);
	Deposits(tkEcalIso,dREcal,EcalEt);
	Deposits(tkHcalIso,dRHcal,HcalEt);
      }
      else{
	h_IsoOther_tk->Fill(Tk_isovalue);
	h_IsoOther_ecal->Fill(Ecal_isovalue);
	h_IsoOther_hcal->Fill(Hcal_isovalue);
	IsoOther->Fill(iso_value);
	Deposits(muTrackIso,dRTrk,TkrPt);
	Deposits(muEcalIso,dREcal,EcalEt);
	Deposits(muHcalIso,dRHcal,HcalEt);
	Deposits(tkTrackIso,dRTrk,TkrPt);
	Deposits(tkEcalIso,dREcal,EcalEt);
	Deposits(tkHcalIso,dRHcal,HcalEt);
      }
    }
  }    

  histo(h_IsoZ_tk,"#Sigma p_{T}","Events");
  histo(h_IsoW_tk,"#Sigma p_{T}","Events");
  histo(h_IsoOther_tk,"#Sigma p_{T}","#Events");
  histo(h_IsoZ_ecal,"#Sigma E_{t}","Events");
  histo(h_IsoW_ecal,"#Sigma E_{t}","Events");
  histo(h_IsoOther_ecal,"#Sigma E_{t}","Events");
  histo(h_IsoZ_hcal,"#Sigma E_{t}","Events");
  histo(h_IsoW_hcal,"#Sigma E_{t}","Events");
  histo(h_IsoOther_hcal,"#Sigma E_{t}","Events");
  histo(TkrPt,"p ","");
  histo(EcalEt,"E ","");
  histo(HcalEt,"E ","");
  histo(HcalEtZ,"E_{T}","");
  histo(EcalEtZ,"E_{T}","");
}
  
void ZMuMuIsolationAnalyzer::endJob() {
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZMuMuIsolationAnalyzer);
