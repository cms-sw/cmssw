// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"


#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "DataFormats/PatCandidates/interface/LookupTableRecord.h"

#include "FWCore/Utilities/interface/transform.h"

#include "TH1F.h"
#include "TFile.h"
#include "TTree.h"

using namespace std;
using namespace edm;
//
// class decleration
//
class IsolationAnalysis : public edm::EDAnalyzer {
public:
  explicit IsolationAnalysis(const edm::ParameterSet&);
  ~IsolationAnalysis();
   
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

 // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<reco::Candidate> > muonLabel_;
  
  std::vector<edm::EDGetTokenT<edm::ValueMap<double> > > tokensIsoValMuonsIsoDeposit_;
  typedef std::vector< edm::Handle< edm::ValueMap<double> > > IsoValuesIsoDeposit;

  std::vector<edm::EDGetTokenT<edm::ValueMap<float> > > tokensIsoValMuonsCITK_;
  typedef std::vector< edm::Handle< edm::ValueMap<float> > > IsoValuesCITK;
  
  TH1F* muon_chIso_isodeposit;
  TH1F* muon_nhIso_isodeposit;
  TH1F* muon_phIso_isodeposit;
  TH1F* muon_puIso_isodeposit;

  TH1F* muon_chIso_citk;
  TH1F* muon_nhIso_citk;
  TH1F* muon_phIso_citk;
  TH1F* muon_puIso_citk;

};

IsolationAnalysis::IsolationAnalysis(const edm::ParameterSet& iConfig):
    muonLabel_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("muonLabel")))
{
  edm::Service<TFileService> fs;

  muon_chIso_isodeposit = fs->make<TH1F>("muon_chIso_isodeposit","muon_chIso_isodeposit",400,0,4);
  muon_nhIso_isodeposit = fs->make<TH1F>("muon_nhIso_isodeposit","muon_nhIso_isodeposit",400,0,4);
  muon_phIso_isodeposit = fs->make<TH1F>("muon_phIso_isodeposit","muon_phIso_isodeposit",400,0,4);
  muon_puIso_isodeposit = fs->make<TH1F>("muon_puIso_isodeposit","muon_puIso_isodeposit",400,0,4);

  muon_chIso_citk = fs->make<TH1F>("muon_chIso_citk","muon_chIso_citk",400,0,4);
  muon_nhIso_citk = fs->make<TH1F>("muon_nhIso_citk","muon_nhIso_citk",400,0,4);
  muon_phIso_citk = fs->make<TH1F>("muon_phIso_citk","muon_phIso_citk",400,0,4);
  muon_puIso_citk = fs->make<TH1F>("muon_puIso_citk","muon_puIso_citk",400,0,4);

  tokensIsoValMuonsCITK_ = edm::vector_transform(iConfig.getParameter< std::vector<edm::InputTag> >("IsoValMuonsCITK"), [this](edm::InputTag const & tag){return consumes<edm::ValueMap<float> >(tag);});
  tokensIsoValMuonsIsoDeposit_ = edm::vector_transform(iConfig.getParameter< std::vector<edm::InputTag> >("IsoValMuonsIsoDeposit"), [this](edm::InputTag const & tag){return consumes<edm::ValueMap<double> >(tag);});

}

IsolationAnalysis::~IsolationAnalysis()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallopate resources etc.)

}

void IsolationAnalysis::beginJob(){
   //Add event and RUN BRANCHING         
}

void IsolationAnalysis::endJob(){
     
}

void IsolationAnalysis::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup){

    using namespace edm;
    using namespace std;
    using namespace reco;

    Handle<View<reco::Candidate> > muonLabel;
    iEvent.getByToken(muonLabel_, muonLabel);

    unsigned nTypesCITK=4;
    IsoValuesCITK muonIsoValuesCITK(nTypesCITK);

    unsigned nTypesIsoDeposit=4;
    IsoValuesIsoDeposit muonIsoValuesIsoDeposit(nTypesIsoDeposit);

    for (size_t j = 0; j<tokensIsoValMuonsCITK_.size(); ++j) {
      iEvent.getByToken(tokensIsoValMuonsCITK_[j], muonIsoValuesCITK[j]);
    }

    for (size_t j = 0; j<tokensIsoValMuonsIsoDeposit_.size(); ++j) {
      iEvent.getByToken(tokensIsoValMuonsIsoDeposit_[j], muonIsoValuesIsoDeposit[j]);
    }

    for(unsigned imu=0; imu< muonLabel->size() ;++imu) {
      reco::CandidateBaseRef myMuonRef(muonLabel,imu);

      const IsoValuesCITK * myIsoValuesCITK = &muonIsoValuesCITK;
      float chIso_citk =  (*(*myIsoValuesCITK)[0])[myMuonRef];
      float nhIso_citk = (*(*myIsoValuesCITK)[1])[myMuonRef];
      float phIso_citk = (*(*myIsoValuesCITK)[2])[myMuonRef];
      float puIso_citk = (*(*myIsoValuesCITK)[3])[myMuonRef];

      muon_chIso_citk->Fill(chIso_citk);
      muon_nhIso_citk->Fill(nhIso_citk);
      muon_phIso_citk->Fill(phIso_citk);
      muon_puIso_citk->Fill(puIso_citk);

      const IsoValuesIsoDeposit * myIsoValuesIsoDeposit = &muonIsoValuesIsoDeposit;
      double chIso_isodeposit =  (*(*myIsoValuesIsoDeposit)[0])[myMuonRef];
      double nhIso_isodeposit = (*(*myIsoValuesIsoDeposit)[1])[myMuonRef];
      double phIso_isodeposit = (*(*myIsoValuesIsoDeposit)[2])[myMuonRef];
      double puIso_isodeposit = (*(*myIsoValuesIsoDeposit)[3])[myMuonRef];

      muon_chIso_isodeposit->Fill(chIso_isodeposit);
      muon_nhIso_isodeposit->Fill(nhIso_isodeposit);
      muon_phIso_isodeposit->Fill(phIso_isodeposit);
      muon_puIso_isodeposit->Fill(puIso_isodeposit);

    }

}

//define this as a plug-in
DEFINE_FWK_MODULE(IsolationAnalysis);

