// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/CaloMuon.h"

using namespace std;
using namespace edm;
using namespace reco;


class MuonFilter : public EDFilter {
 public:
  explicit MuonFilter(const edm::ParameterSet& pset);
  ~MuonFilter();
  virtual bool filter(edm::Event& event, const edm::EventSetup& eventSetup);
       
protected:
       
private: 

  edm::InputTag muonTag_;
  edm::InputTag caloMuonTag_;
  bool acceptMuon_;
  bool acceptCalo_;

};


MuonFilter::MuonFilter(const ParameterSet& pset){
  // input tags
  muonTag_         =  pset.getParameter<edm::InputTag>("muonTag");
  caloMuonTag_     =  pset.getParameter<edm::InputTag>("caloMuonTag");
  acceptMuon_      =  pset.getUntrackedParameter<bool>("acceptMuon",true);
  acceptCalo_      =  pset.getUntrackedParameter<bool>("acceptCalo",false);
}

MuonFilter::~MuonFilter(){
}

bool MuonFilter::filter(edm::Event& event, const edm::EventSetup& eventSetup){

  Handle<reco::MuonCollection> muons;
  event.getByLabel(muonTag_,muons);

  Handle<reco::CaloMuonCollection> caloMuons;
  event.getByLabel(caloMuonTag_,caloMuons);

  if (acceptMuon_ && acceptCalo_)  return ( (muons->size() > 0 ) || (caloMuons->size() > 0 ) );
  if (acceptMuon_ && !acceptCalo_)  return ( (muons->size() > 0 ) );
  if (!acceptMuon_ && acceptCalo_)  return ( (caloMuons->size() > 0 ) );

  return false;

}

DEFINE_FWK_MODULE(MuonFilter);
