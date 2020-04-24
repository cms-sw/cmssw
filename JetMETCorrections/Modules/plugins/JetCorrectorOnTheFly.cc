#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH1F.h"


using namespace edm;
using namespace reco;
using namespace std;

//
// class declaration
//
template<class Jet>
class JetCorrectorOnTheFly : public edm::EDAnalyzer {
public:
  explicit JetCorrectorOnTheFly(const edm::ParameterSet&);
  ~JetCorrectorOnTheFly();

private:
  typedef std::vector<Jet> JetCollection;
  virtual void beginJob() override ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;

  edm::Service<TFileService> fs;
  edm::EDGetTokenT<reco::JetCorrector> mJetCorrector;
  edm::EDGetTokenT<JetCollection> mJetToken;
  edm::EDGetTokenT<reco::VertexCollection> mVertexToken;
  double mMinRawJetPt;
  bool mDebug;
  TH1F *mRawPt,*mCorPt;
};
//
//----------- Class Implementation ------------------------------------------
//
//---------------------------------------------------------------------------
template<class Jet>
JetCorrectorOnTheFly<Jet>::JetCorrectorOnTheFly(const edm::ParameterSet& iConfig)
{
  mJetCorrector      = consumes<reco::JetCorrector>(iConfig.getParameter<edm::InputTag>("JetCorrector"));
  mJetToken       = consumes<JetCollection>(edm::InputTag(iConfig.getParameter<edm::InputTag> ("JetCollectionName")));
  mVertexToken       = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  mMinRawJetPt   = iConfig.getParameter<double> ("MinRawJetPt");
  mDebug         = iConfig.getParameter<bool> ("Debug");
}
//---------------------------------------------------------------------------
template<class Jet>
JetCorrectorOnTheFly<Jet>::~JetCorrectorOnTheFly()
{

}
//---------------------------------------------------------------------------
template<class Jet>
void JetCorrectorOnTheFly<Jet>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<reco::JetCorrector> corrector;
  iEvent.getByToken(mJetCorrector, corrector);
  Handle<JetCollection> jets;
  iEvent.getByToken(mJetToken,jets);

  edm::Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(mVertexToken,recVtxs);
  int NPV(0);
  for(unsigned int ind=0;ind<recVtxs->size();ind++) {
    if (!((*recVtxs)[ind].isFake())) {
      NPV++;
    }
  }
  typename JetCollection::const_iterator i_jet;
  /////////// Loop over all jets and apply correction /////
  for(i_jet = jets->begin(); i_jet != jets->end(); i_jet++) {
    if (i_jet->pt() < mMinRawJetPt) continue;
    //double scale = corrector->correction(i_jet->p4());
    double scale = corrector->correction(*i_jet);
    if (mDebug) {
      std::cout<<"energy = "<<i_jet->energy()<<", "
               <<"eta = "<<i_jet->eta()<<", "
               <<"raw pt = "<<i_jet->pt()<<", "
               <<"NPV = "<<NPV<<", "
               <<"correction = "<<scale<<", "
               <<"cor pt = "<<scale*i_jet->pt()<<endl;
    }
    mRawPt->Fill(i_jet->pt());
    mCorPt->Fill(scale*i_jet->pt());
  }


}
//---------------------------------------------------------------------------
template<class Jet>
void JetCorrectorOnTheFly<Jet>::beginJob()
{
  mRawPt = fs->make<TH1F>("RawJetPt","RawJetPt",1000,0,1000);
  mCorPt = fs->make<TH1F>("CorJetPt","CorJetPt",1000,0,1000);
}
//---------------------------------------------------------------------------
template<class Jet>
void JetCorrectorOnTheFly<Jet>::endJob()
{

}
//---------------------------------------------------------------------------
typedef JetCorrectorOnTheFly<CaloJet> CaloJetCorrectorOnTheFly;
DEFINE_FWK_MODULE(CaloJetCorrectorOnTheFly);

typedef JetCorrectorOnTheFly<PFJet> PFJetCorrectorOnTheFly;
DEFINE_FWK_MODULE(PFJetCorrectorOnTheFly);

typedef JetCorrectorOnTheFly<JPTJet> JPTJetCorrectorOnTheFly;
DEFINE_FWK_MODULE(JPTJetCorrectorOnTheFly);























