// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "EgammaAnalysis/ElectronTools/interface/EGammaMvaEleEstimator.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
//
// class declaration
//

using namespace std;
using namespace reco;
class ElectronIdMVAProducer : public edm::EDFilter {
public:
  explicit ElectronIdMVAProducer(const edm::ParameterSet&);
  ~ElectronIdMVAProducer();
  
private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  
  // ----------member data ---------------------------
  bool verbose_;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> electronToken_;
  edm::EDGetTokenT<double> eventrhoToken_;
  edm::EDGetTokenT<EcalRecHitCollection> reducedEBRecHitCollectionToken_;
  edm::EDGetTokenT<EcalRecHitCollection> reducedEERecHitCollectionToken_;
  
  double _Rho;
  string method_;
  vector<string> mvaWeightFiles_;
  bool Trig_;
  bool NoIP_;
  
  EGammaMvaEleEstimator* mvaID_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ElectronIdMVAProducer::ElectronIdMVAProducer(const edm::ParameterSet& iConfig) {
  verbose_ = iConfig.getUntrackedParameter<bool>("verbose", false);
  vertexToken_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexTag"));
  electronToken_ = consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electronTag"));
  eventrhoToken_ = consumes<double>(edm::InputTag("kt6PFJets", "rho"));
  reducedEBRecHitCollectionToken_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedEBRecHitCollection"));
  reducedEERecHitCollectionToken_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedEERecHitCollection"));
  method_ = iConfig.getParameter<string>("method");
  std::vector<string> fpMvaWeightFiles = iConfig.getParameter<std::vector<std::string> >("mvaWeightFile");
  Trig_ = iConfig.getParameter<bool>("Trig");
  NoIP_ = iConfig.getParameter<bool>("NoIP");
  
  produces<edm::ValueMap<float> >("");
  
  mvaID_ = new EGammaMvaEleEstimator();
  
  EGammaMvaEleEstimator::MVAType type_;
  if(Trig_ && !NoIP_){type_ = EGammaMvaEleEstimator::kTrig;}
  
  if(Trig_ && NoIP_){type_ = EGammaMvaEleEstimator::kTrigNoIP;}
  
  if(!Trig_){type_ = EGammaMvaEleEstimator::kNonTrig;}
  
  bool manualCat_ = true;
  
  string path_mvaWeightFileEleID;
  for(unsigned ifile=0 ; ifile < fpMvaWeightFiles.size() ; ++ifile) {
    path_mvaWeightFileEleID = edm::FileInPath ( fpMvaWeightFiles[ifile].c_str() ).fullPath();
    mvaWeightFiles_.push_back(path_mvaWeightFileEleID);
  }
  
  mvaID_->initialize(method_, type_, manualCat_, mvaWeightFiles_);
  
}


ElectronIdMVAProducer::~ElectronIdMVAProducer()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool ElectronIdMVAProducer::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  
  std::auto_ptr<edm::ValueMap<float> > out(new edm::ValueMap<float>() );
  
  Handle<reco::VertexCollection>  vertexCollection;
  iEvent.getByToken(vertexToken_, vertexCollection);
  
  Vertex dummy;
  const Vertex *pv = &dummy;
  if ( vertexCollection->size() != 0) {
    pv = &*vertexCollection->begin();
  } else { // create a dummy PV
    Vertex::Error e;
    e(0, 0) = 0.0015 * 0.0015;
    e(1, 1) = 0.0015 * 0.0015;
    e(2, 2) = 15. * 15.;
    Vertex::Point p(0, 0, 0);
    dummy = Vertex(p, e, 0, 0, 0);
  }
  
  EcalClusterLazyTools lazyTools(iEvent, iSetup, reducedEBRecHitCollectionToken_, reducedEERecHitCollectionToken_);
  
  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);
  TransientTrackBuilder thebuilder = *(builder.product());
  
  Handle<reco::GsfElectronCollection> egCollection;
  iEvent.getByToken(electronToken_,egCollection);
  const reco::GsfElectronCollection egCandidates = (*egCollection.product());
  
  _Rho=0;
  edm::Handle<double> rhoPtr;
  iEvent.getByToken(eventrhoToken_,rhoPtr);
  _Rho=*rhoPtr;
  
  std::vector<float> values;
  values.reserve(egCollection->size());
  
  for ( reco::GsfElectronCollection::const_iterator egIter = egCandidates.begin(); egIter != egCandidates.end(); ++egIter) {
    
    double mvaVal = -999999;
    if(!NoIP_){
      mvaVal = mvaID_->mvaValue( *egIter, *pv,thebuilder,lazyTools, verbose_);
    }
    if(NoIP_){
      mvaVal = mvaID_->mvaValue( *egIter, *pv, _Rho,/*thebuilder,*/lazyTools, verbose_);
    }
    
    values.push_back( mvaVal );
  }
  
  edm::ValueMap<float>::Filler filler(*out);
  filler.insert(egCollection, values.begin(), values.end() );
  filler.fill();
  
  iEvent.put(out);
  
  return true;
}

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronIdMVAProducer);
