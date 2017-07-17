/* \class testAnalyzer
 * author: Noli Pasquale
 * version 0.1
 * TriggerEfficiency module
 *
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "TH1.h"
#include <vector>
#include <string>
#include <iostream>
#include <iterator>
using namespace edm;
using namespace std;
using namespace reco;
using namespace pat;

class testAnalyzer : public edm::EDAnalyzer {
public:
  testAnalyzer(const edm::ParameterSet& pset);
private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
  virtual void endJob() override;
  int SingleTrigger_ ,DoubleTrigger_ ,  NoTrigger_ , zmumuIncrement_ ;
  EDGetTokenT<vector<pat::Muon> > selectMuonToken_;
  EDGetTokenT<CandidateView> zMuMuToken_;
  string pathName_;
  int  nbinsEta_;
  double minEta_ , maxEta_;
  int  nbinsPt_;
  double minPt_ , maxPt_;
  int nbinsEtaPt_;
  TH1D *h_pt_distribution_;
  TH1D *h_numberTrigMuon_, *h_numberMuon_ ;
  TH1D *h_numberTrigMuon_ptStudy_, *h_numberMuon_ptStudy_ ;
  TH1D *h_EtaDist_Pt80_;
  vector<double> vectorPt , vectorEta ;
};

testAnalyzer::testAnalyzer(const edm::ParameterSet& pset) :
  selectMuonToken_( consumes<vector<pat::Muon> >( pset.getParameter<InputTag>( "selectMuon" ) ) ),
  zMuMuToken_( consumes<CandidateView>( pset.getParameter<InputTag>( "ZMuMu" ) ) ),
  pathName_( pset.getParameter<string>( "pathName" ) ),
  nbinsEta_( pset.getParameter<int>( "EtaBins" ) ),
  minEta_( pset.getParameter<double>( "minEta" ) ),
  maxEta_( pset.getParameter<double>( "maxEta" ) ),
  nbinsPt_( pset.getParameter<int>( "PtBins" ) ),
  minPt_( pset.getParameter<double>( "minPt" ) ),
  maxPt_( pset.getParameter<double>( "maxPt" ) ),
  nbinsEtaPt_( pset.getParameter<int>( "EtaPt80Bins" ) ){
  SingleTrigger_= 0;
  DoubleTrigger_= 0;
  NoTrigger_= 0;
  zmumuIncrement_=0;
  Service<TFileService> fs;
  h_pt_distribution_ = fs->make<TH1D>("PtResolution ","Pt Resolution",200,-4.,4.);
  h_numberMuon_ = fs->make<TH1D>("Denominatore","Number of Muons vs Eta",nbinsEta_,minEta_,maxEta_);
  h_numberTrigMuon_ = fs->make<TH1D>("NumeratoreTrigMuon","Number of Triggered Muons vs Eta",nbinsEta_ ,minEta_,maxEta_);
  h_numberMuon_ptStudy_ = fs->make<TH1D>("DenominatorePtStudy","Number of Muons vs Pt",nbinsPt_,minPt_,maxPt_);
  h_numberTrigMuon_ptStudy_ = fs->make<TH1D>("NumeratoreTrigMuonPtStudy","Number of Triggered Muons vs Pt",nbinsPt_,minPt_,maxPt_);
  h_EtaDist_Pt80_ = fs->make<TH1D>("EtaDistr","Eta distribution (Pt>80)",nbinsEtaPt_,minEta_,maxEta_);
}

void testAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  Handle<vector<pat::Muon> > selectMuon;
  event.getByToken(selectMuonToken_, selectMuon);
  Handle<CandidateView> zMuMu;
  event.getByToken(zMuMuToken_, zMuMu);
  int zmumuSize = zMuMu->size();
  if(zmumuSize > 0){
    for( int i = 0; i < zmumuSize ; ++i){
      bool singleTrigFlag0 = false;
      bool singleTrigFlag1 = false;
      zmumuIncrement_++;
      const Candidate & zMuMuCand = (*zMuMu)[i];
      CandidateBaseRef dau0 = zMuMuCand.daughter(0)->masterClone();
      CandidateBaseRef dau1 = zMuMuCand.daughter(1)->masterClone();
      const pat::Muon& mu0 = dynamic_cast<const pat::Muon&>(*dau0);//cast in patMuon
      const pat::Muon& mu1 = dynamic_cast<const pat::Muon&>(*dau1);
      const pat::TriggerObjectStandAloneCollection mu0HLTMatches =
	mu0.triggerObjectMatchesByPath( pathName_ );
      const pat::TriggerObjectStandAloneCollection mu1HLTMatches =
	mu1.triggerObjectMatchesByPath( pathName_ );
      double EtaPatMu0 = mu0.eta();
      double EtaPatMu1 = mu1.eta();
      double PtPatMu0 = mu0.pt();
      double PtPatMu1 = mu1.pt();
      h_numberMuon_->Fill(EtaPatMu0);
      h_numberMuon_->Fill(EtaPatMu1);
      h_numberMuon_ptStudy_->Fill(PtPatMu0);
      h_numberMuon_ptStudy_->Fill(PtPatMu1);
      int dimTrig0 = mu0HLTMatches.size();
      int dimTrig1 = mu1HLTMatches.size();
      if(dimTrig0 !=0){
	for(int j = 0; j < dimTrig0 ; ++j){
	singleTrigFlag0 = true;
	h_numberTrigMuon_->Fill(EtaPatMu0);
	h_numberTrigMuon_ptStudy_->Fill(PtPatMu0);
	double PtTrig = mu0HLTMatches[j].pt();
	double PtDif = PtTrig-PtPatMu0;
	h_pt_distribution_->Fill(PtDif);
	}
      }
      else{
	if(PtPatMu0>80) {
	  h_EtaDist_Pt80_->Fill(EtaPatMu0);
	  vectorPt.push_back(PtPatMu0);
	  vectorEta.push_back(EtaPatMu0);
	}
      }
      if(dimTrig1 !=0){
	for(int j = 0; j < dimTrig1 ; ++j){
	  singleTrigFlag1 = true;
	  h_numberTrigMuon_->Fill(EtaPatMu1);
	  h_numberTrigMuon_ptStudy_->Fill(PtPatMu1);
	  double PtTrig = mu0HLTMatches[j].pt();
	  double PtDif = PtTrig-PtPatMu1;
	  h_pt_distribution_->Fill(PtDif);
	}
      }
      else{
	if(PtPatMu0>80) {
	  h_EtaDist_Pt80_->Fill(EtaPatMu1);
	  vectorPt.push_back(PtPatMu0);
	  vectorEta.push_back(EtaPatMu0);
	}
      }

      if(singleTrigFlag0 && singleTrigFlag1)DoubleTrigger_++;
      if(singleTrigFlag0 && !singleTrigFlag1)SingleTrigger_++;
      if(!singleTrigFlag0 && singleTrigFlag1)SingleTrigger_++;
      if(!singleTrigFlag0 && !singleTrigFlag1)NoTrigger_++;

    }//end loop on ZMuMu candidates
  }//end check on ZMuMu

}

void testAnalyzer::endJob() {
  cout<< "DoubleTrigger = " << DoubleTrigger_ << " , SingleTrigger = " << SingleTrigger_ << " , NoTrigger = "<< NoTrigger_ <<" ,zmumuIncrement = "<< zmumuIncrement_ << endl;
  double OneTrig = (double)SingleTrigger_/(double)zmumuIncrement_;
  double DoubleTrig = (double)DoubleTrigger_/(double)zmumuIncrement_;
  cout<< "eps^2 = " <<  DoubleTrig << endl;
  cout<< "2eps(1 - eps) = " << OneTrig << endl;
  int dimVec =vectorPt.size();
  for(int i = 0; i<dimVec; ++i) cout << "Pt = " << vectorPt[i] << " ==> Eta = " << vectorEta[i] << endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(testAnalyzer);

