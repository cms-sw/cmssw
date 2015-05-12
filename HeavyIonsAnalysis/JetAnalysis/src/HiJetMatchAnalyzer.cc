// -*- C++ -*-
//
// Package:    HiJetMatchAnalyzer
// Class:      HiJetMatchAnalyzer
//
/**\class HiJetMatchAnalyzer HiJetMatchAnalyzer.cc CmsHi/HiJetMatchAnalyzer/src/HiJetMatchAnalyzer.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Thu Sep  9 10:38:59 EDT 2010
// $Id: HiJetMatchAnalyzer.cc,v 1.2 2012/04/22 19:12:44 yilmaz Exp $
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "HeavyIonsAnalysis/JetAnalysis/interface/RhoGetter.h"
#include "TTree.h"

using namespace std;
using namespace edm;

static const int MAXJETS = 500;

struct etdr{
  double et;
  double dr;
};

class JRA{

public:
  int nref;
  int bin;
  float b;
  float hf;
  float jtpt[MAXJETS];
  float jtrawpt[MAXJETS];
  float refpt[MAXJETS];
  float jteta[MAXJETS];
  float refeta[MAXJETS];
  float jtphi[MAXJETS];
  float refphi[MAXJETS];
  float l2[MAXJETS];
  float l3[MAXJETS];
  float area[MAXJETS];
  float pu[MAXJETS];
  float rho[MAXJETS];

  float weight;
};

struct JRAV{
  int index;
  float jtpt;
  float jtrawpt;
  float refpt;
  float refcorpt;
  float jteta;
  float refeta;
  float jtphi;
  float refphi;
  float l2;
  float l3;
  float area;
  float pu;
  float rho;

};

//
// class declaration
//
bool comparePt(JRAV a, JRAV b) {return a.jtpt > b.jtpt;}

class HiJetMatchAnalyzer : public edm::EDAnalyzer {
public:
  explicit HiJetMatchAnalyzer(const edm::ParameterSet&);
  ~HiJetMatchAnalyzer();


private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  bool selectJet(int i);
  // ----------member data ---------------------------

  bool usePat_;
  bool doMC_;
  bool filterJets_;
  bool diJetsOnly_;
  bool matchDiJets_;
  bool matchPatGen_;
  bool matchNew_;
  bool sortJets_;
  bool correctJets_;
  bool getFastJets_;

  double matchR_;
  double genPtMin_;
  double ptMin_;
  double emfMin_;
  double n90Min_;
  double n90hitMin_;

  edm::InputTag jetTag_;
  edm::InputTag matchTag_;
  std::vector<edm::InputTag> matchTags_;

  JRA jra_;
  std::vector<JRA> jraMatch_;

  TTree* t;

  edm::Handle<edm::GenHIEvent> mc;
  edm::Handle<reco::Centrality> cent;

  edm::Handle<reco::JetView> jets;
  edm::Handle<pat::JetCollection> patjets;

  FactorizedJetCorrector* jetCorrector_;
  edm::ESWatcher<JetCorrectionsRecord> watchJetCorrectionsRecord_;

  std::string tags_;
  std::string levels_;
  std::string algo_;

  edm::Service<TFileService> fs;

  edm::Handle<vector<double> > ktRhos;
  edm::Handle<vector<double> > akRhos;
  bool doFastJets_;

  vector<int> doMatchedFastJets_;
  vector<int> correctMatchedJets_;

  InputTag ktSrc_;
  InputTag akSrc_;


};

bool HiJetMatchAnalyzer::selectJet(int i){
  //const reco::Jet& jet = (*jets)[i];
  if(usePat_){
    const pat::Jet& patjet = (*patjets)[i];
    if(patjet.emEnergyFraction() <= emfMin_) return false;
    if(patjet.jetID().n90Hits <= n90hitMin_) return false;
    if(doMC_){

    }

  }

  return true;
}


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HiJetMatchAnalyzer::HiJetMatchAnalyzer(const edm::ParameterSet& iConfig)

{

  //now do what ever initialization is needed
  matchR_ = iConfig.getUntrackedParameter<double>("matchR",0.25);

  ptMin_ = iConfig.getUntrackedParameter<double>("ptMin",0);
  genPtMin_ = iConfig.getUntrackedParameter<double>("genPtMin",20);
  emfMin_ = iConfig.getUntrackedParameter<double>("emfMin",0.01);
  n90Min_ = iConfig.getUntrackedParameter<double>("n90Min",1);
  n90hitMin_ = iConfig.getUntrackedParameter<double>("n90hitMin",1);

  filterJets_ = iConfig.getUntrackedParameter<bool>("filterJets",true);
  diJetsOnly_ = iConfig.getUntrackedParameter<bool>("diJetsOnly",false);
  matchDiJets_ = iConfig.getUntrackedParameter<bool>("matchDiJets",false);
  matchPatGen_ = iConfig.getUntrackedParameter<bool>("matchPatGen",false);

  matchNew_ = iConfig.getUntrackedParameter<bool>("matchNew",false);

  usePat_ = iConfig.getUntrackedParameter<bool>("usePat",true);
  doMC_ = iConfig.getUntrackedParameter<bool>("doMC",true);

  sortJets_ = iConfig.getUntrackedParameter<bool>("sortJets",true);
  correctJets_ = iConfig.getUntrackedParameter<bool>("correctJets",false);

  correctMatchedJets_ = iConfig.getUntrackedParameter<std::vector<int> >("correctMatchedJets",std::vector<int>(0));
  doMatchedFastJets_ = iConfig.getUntrackedParameter<std::vector<int> >("doMatchedFastJets",std::vector<int>(0));

  jetTag_ = iConfig.getUntrackedParameter<edm::InputTag>("src",edm::InputTag("selectedPatJets"));
  matchTag_ = iConfig.getUntrackedParameter<edm::InputTag>("match",edm::InputTag("selectedPatJets"));
  matchTags_ = iConfig.getUntrackedParameter<std::vector<edm::InputTag> >("matches",std::vector<edm::InputTag>(0));

  ktSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("ktSrc",edm::InputTag("kt4CaloJets"));
  akSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("akSrc",edm::InputTag("ak5CaloJets"));
  doFastJets_ = iConfig.getUntrackedParameter<bool>("doFastJets",false);

  getFastJets_ = iConfig.getUntrackedParameter<bool>("getFastJets",false);

  for(unsigned int i = 0; i < doMatchedFastJets_.size(); ++i){
    getFastJets_ = getFastJets_ || (bool)doMatchedFastJets_[i];
  }


  if(correctJets_){

    levels_ = iConfig.getUntrackedParameter<string>("corrLevels","L2Relative:L3Absolute");

    algo_ = iConfig.getUntrackedParameter<string>("algo","IC5Calo");
    tags_ = "";

    string l[2] = {"L2Relative","L3Absolute"};

    for(int i = 0; i <2; ++i){
      edm::FileInPath fip("CondFormats/JetMETObjects/data/Spring10_"+l[i]+"_"+algo_+".txt");
      tags_ += fip.fullPath();
      if(i < 2 - 1)tags_ +=":";
    }

    jetCorrector_ = new FactorizedJetCorrector(levels_, tags_);
  }




}


HiJetMatchAnalyzer::~HiJetMatchAnalyzer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HiJetMatchAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  iEvent.getByLabel(jetTag_,jets);
  if(usePat_)iEvent.getByLabel(jetTag_,patjets);
  std::vector<JRAV> jraV;

  if(getFastJets_){
    iEvent.getByLabel(edm::InputTag(ktSrc_.label(),"rhos"),ktRhos);
    iEvent.getByLabel(edm::InputTag(akSrc_.label(),"rhos"),akRhos);
  }

  for(unsigned int j = 0 ; j < jets->size(); ++j){
    if(filterJets_ && !selectJet(j)) continue;
    const reco::Jet& jet = (*jets)[j];
    JRAV jv;
    jv.jtpt = jet.pt();
    jv.jteta = jet.eta();
    jv.jtphi = jet.phi();
    jv.jtrawpt = jet.pt();
    jv.area = jet.jetArea();
    jv.pu  = jet.pileup();
    jv.index = j;

    double pt = jet.pt();
    // double ktRho = -1, akRho = -1;
    double akRho = -1;
    if(getFastJets_){
      //ktRho = getRho(jv.jteta,*ktRhos);
      akRho = getRho(jv.jteta,*akRhos);
    }

    jv.rho = akRho;

    if(doFastJets_){
      jv.pu = jet.jetArea()*akRho;
      pt -= jv.pu;
    }
    jv.jtpt = pt;

    if(correctJets_){
      jetCorrector_->setJetEta(jet.eta());
      jetCorrector_->setJetPt(pt);
      //	jetCorrector_->setJetE(jet.energy());
      vector<float> corrs = jetCorrector_->getSubCorrections();
      jv.l2 = corrs[0];
      jv.l3 = corrs[1];
      jv.jtpt = pt*jv.l2*jv.l3;
    }


    if(usePat_){
      const pat::Jet& patjet = (*patjets)[j];

      jv.jtrawpt = patjet.correctedJet("raw").pt();
      jv.jtpt = patjet.pt();

      if(doMC_ && matchPatGen_ && patjet.genJet() != 0){
	if(patjet.genJet()->pt() < genPtMin_) continue;
	jv.refpt = patjet.genJet()->pt();
	jv.refeta = patjet.genJet()->eta();
	jv.refphi = patjet.genJet()->phi();
      }else{
	jv.refpt = -99;
	jv.refeta = -99;
	jv.refphi = -99;
      }
    }
    jraV.push_back(jv);
  }

  if(sortJets_){
    std::sort(jraV.begin(),jraV.end(),comparePt);
  }

  for(unsigned int i = 0; i < jraV.size(); ++i){
    JRAV& jv = jraV[i];
    const reco::Jet& jet = (*jets)[jv.index];

    if(matchNew_){
      for(unsigned int im = 0; im < matchTags_.size(); ++im){
	edm::Handle<reco::JetView> matchedJets;
	iEvent.getByLabel(matchTags_[im],matchedJets);
	jraMatch_[im].jtrawpt[i] = -99;
	jraMatch_[im].jtpt[i] = -99;
	jraMatch_[im].jteta[i] = -99;
	jraMatch_[im].jtphi[i] = -99;
	jraMatch_[im].area[i] = -99;
	jraMatch_[im].l2[i] = -99;
	jraMatch_[im].l3[i] = -99;
	jraMatch_[im].pu[i] = -99;
	for(unsigned int m = 0 ; m < matchedJets->size(); ++m){
	  const reco::Jet& match = (*matchedJets)[m];
	  double dr = reco::deltaR(jet.eta(),jet.phi(),match.eta(),match.phi());
	  if(dr < matchR_ && match.pt() > genPtMin_){
	    jraMatch_[im].jtpt[i] = match.pt();
	    jraMatch_[im].jtrawpt[i] = match.pt();
	    jraMatch_[im].jteta[i] = match.eta();
	    jraMatch_[im].jtphi[i] = match.phi();
	    jraMatch_[im].area[i] = match.jetArea();

	    //double ktRhoM = -1, akRhoM = -1;
	    double akRhoM = -1;
	    if(getFastJets_){
	      //ktRhoM = getRho(jraMatch_[im].jteta[i],*ktRhos);
	      akRhoM = getRho(jraMatch_[im].jteta[i],*akRhos);
	    }

	    jraMatch_[im].rho[i] = akRhoM;
	    double pt = jraMatch_[im].jtpt[i];

	    if((bool)doMatchedFastJets_[im]){
	      jraMatch_[im].pu[i] = jraMatch_[im].area[i]*akRhoM;
	      pt -= jraMatch_[im].pu[i];
	    }

	    jraMatch_[im].jtpt[i] = pt;

	    if((bool)correctMatchedJets_[im]){

	      jetCorrector_->setJetEta(jraMatch_[im].jteta[i]);
	      jetCorrector_->setJetPt(pt);
	      //		jetCorrector_->setJetE(match.energy());
	      vector<float> corrs = jetCorrector_->getSubCorrections();
	      jraMatch_[im].l2[i] = corrs[0];
	      // Should it be re-set for L3???
	      jraMatch_[im].l3[i] = corrs[1];
	      jraMatch_[im].jtpt[i] = pt*jraMatch_[im].l2[i]*jraMatch_[im].l3[i];
	    }

	  }
	}
      }
    }

    jra_.jtpt[i] = jv.jtpt;
    jra_.jteta[i] = jv.jteta;
    jra_.jtphi[i] = jv.jtphi;
    jra_.jtrawpt[i] = jv.jtrawpt;
    jra_.refpt[i] = jv.refpt;
    jra_.refeta[i] = jv.refeta;
    jra_.refphi[i] = jv.refphi;

    jra_.area[i] = jv.area;
    jra_.pu[i] = jv.pu;
    jra_.rho[i] = jv.rho;
    jra_.l2[i] = jv.l2;
    jra_.l3[i] = jv.l3;

  }
  jra_.nref = jraV.size();

  t->Fill();

}

// ------------ method called once each job just before starting event loop  ------------


void
HiJetMatchAnalyzer::beginJob(){
  t= fs->make<TTree>("t","Jet Response Analyzer");
  t->Branch("nref",&jra_.nref,"nref/I");
  t->Branch("jtpt",jra_.jtpt,"jtpt[nref]/F");
  t->Branch("jteta",jra_.jteta,"jteta[nref]/F");
  t->Branch("jtphi",jra_.jtphi,"jtphi[nref]/F");
  t->Branch("jtrawpt",jra_.jtrawpt,"jtrawpt[nref]/F");

  if(correctJets_){
    t->Branch("l2",jra_.l2,"l2[nref]/F");
    t->Branch("l3",jra_.l3,"l3[nref]/F");
  }
  t->Branch("area",jra_.area,"area[nref]/F");
  t->Branch("pu",jra_.pu,"pu[nref]/F");
  t->Branch("rho",jra_.rho,"rho[nref]/F");

  t->Branch("refpt",jra_.refpt,"refpt[nref]/F");
  t->Branch("refcorpt",jra_.refpt,"refcorpt[nref]/F");
  t->Branch("refeta",jra_.refeta,"refeta[nref]/F");
  t->Branch("refphi",jra_.refphi,"refphi[nref]/F");
  t->Branch("weight",&jra_.weight,"weight/F");

  jraMatch_.clear();
  for(unsigned int im = 0; im < matchTags_.size(); ++im){
    JRA jrm;
    jraMatch_.push_back(jrm);
  }

  for(unsigned int im = 0; im < matchTags_.size(); ++im){
    t->Branch(Form("jtpt%d",im),jraMatch_[im].jtpt,Form("jtpt%d[nref]/F",im));
    t->Branch(Form("jteta%d",im),jraMatch_[im].jteta,Form("jteta%d[nref]/F",im));
    t->Branch(Form("jtphi%d",im),jraMatch_[im].jtphi,Form("jtphi%d[nref]/F",im));
    t->Branch(Form("jtrawpt%d",im),jraMatch_[im].jtrawpt,Form("jtrawpt%d[nref]/F",im));

    if((bool)correctMatchedJets_[im]){
      t->Branch(Form("l2_%d",im),jraMatch_[im].l2,Form("l2_%d[nref]/F",im));
      t->Branch(Form("l3_%d",im),jraMatch_[im].l3,Form("l3_%d[nref]/F",im));
    }

    t->Branch(Form("area%d",im),jraMatch_[im].area,Form("area%d[nref]/F",im));
    t->Branch(Form("pu%d",im),jraMatch_[im].pu,Form("pu%d[nref]/F",im));
    t->Branch(Form("rho%d",im),jraMatch_[im].rho,Form("rho%d[nref]/F",im));

  }

}

// ------------ method called once each job just after ending the event loop  ------------
void
HiJetMatchAnalyzer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiJetMatchAnalyzer);
