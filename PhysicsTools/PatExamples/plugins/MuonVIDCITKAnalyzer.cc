#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
/*
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "DataFormats/PatCandidates/interface/LookupTableRecord.h"

#include "FWCore/Utilities/interface/transform.h"

#include "TH1F.h"
#include "TFile.h"
#include "TTree.h"
*/

using namespace std;
using namespace edm;
//
// class decleration
//
class MuonVIDCITKAnalyzer : public edm::EDAnalyzer 
{
public:
  MuonVIDCITKAnalyzer(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&);
   
private:
  edm::EDGetTokenT<edm::View<reco::Candidate> > muonToken_;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  
  // VID
  typedef edm::ValueMap<bool> VIDMap;
  edm::EDGetTokenT<VIDMap> muonLooseVIDToken_;
  edm::EDGetTokenT<VIDMap> muonMediumVIDToken_;
  edm::EDGetTokenT<VIDMap> muonTightVIDToken_;
  edm::EDGetTokenT<VIDMap> muonSoftVIDToken_;
  edm::EDGetTokenT<VIDMap> muonHighPtVIDToken_;

  // CITK
  typedef edm::ValueMap<double> CITKMap;
  edm::EDGetTokenT<CITKMap> muonChIsoCITKToken_;
  edm::EDGetTokenT<CITKMap> muonNhIsoCITKToken_;
  edm::EDGetTokenT<CITKMap> muonPhIsoCITKToken_;
  edm::EDGetTokenT<CITKMap> muonPuIsoCITKToken_;
};

MuonVIDCITKAnalyzer::MuonVIDCITKAnalyzer(const edm::ParameterSet& iConfig)
{
  muonToken_ = consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("muon"));
  vertexToken_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertex"));

  string vidPrefix = "muonVIDs:cutBasedMuonId-MuonPOG-V0-";
  muonLooseVIDToken_  = consumes<VIDMap>(edm::InputTag(vidPrefix+"loose" ));
  muonMediumVIDToken_ = consumes<VIDMap>(edm::InputTag(vidPrefix+"medium"));
  muonTightVIDToken_  = consumes<VIDMap>(edm::InputTag(vidPrefix+"tight" ));
  muonSoftVIDToken_   = consumes<VIDMap>(edm::InputTag(vidPrefix+"soft"  ));
  muonHighPtVIDToken_ = consumes<VIDMap>(edm::InputTag(vidPrefix+"highpt"));

}

void MuonVIDCITKAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& eSetup)
{
  edm::Handle<View<reco::Candidate> > muonHandle;
  event.getByToken(muonToken_, muonHandle);

  edm::Handle<reco::VertexCollection> vertexHandle;
  event.getByToken(vertexToken_, vertexHandle);
  reco::Vertex vertex = vertexHandle->at(0);

  edm::Handle<VIDMap> muonLooseVIDHandle;
  edm::Handle<VIDMap> muonMediumVIDHandle;
  edm::Handle<VIDMap> muonTightVIDHandle;
  edm::Handle<VIDMap> muonSoftVIDHandle;
  edm::Handle<VIDMap> muonHighPtVIDHandle;
  event.getByToken(muonLooseVIDToken_, muonLooseVIDHandle);
  event.getByToken(muonMediumVIDToken_, muonMediumVIDHandle);
  event.getByToken(muonTightVIDToken_, muonTightVIDHandle);
  event.getByToken(muonSoftVIDToken_, muonSoftVIDHandle);
  event.getByToken(muonHighPtVIDToken_, muonHighPtVIDHandle);
  
  for ( size_t i=0, n=muonHandle->size(); i<n; ++i )
  {
    reco::CandidateBaseRef muRef(muonHandle, i);
    const auto& mu = dynamic_cast<const reco::Muon&>(*muRef);

    const bool vidLoose  = (*muonLooseVIDHandle )[muRef];
    const bool vidMedium = (*muonMediumVIDHandle)[muRef];
    const bool vidTight  = (*muonTightVIDHandle )[muRef];
    const bool vidSoft   = (*muonSoftVIDHandle  )[muRef];
    const bool vidHighPt = (*muonHighPtVIDHandle)[muRef];

    const bool isLoose  = muon::isLooseMuon(mu);
    const bool isMedium = muon::isMediumMuon(mu);
    const bool isTight  = muon::isTightMuon(mu, vertex);
    const bool isSoft   = muon::isSoftMuon(mu, vertex);
    const bool isHighPt = muon::isHighPtMuon(mu, vertex);

    assert(vidLoose  == isLoose );
    assert(vidMedium == isMedium);
    assert(vidTight  == isTight );
    assert(vidSoft   == isSoft  );
    assert(vidHighPt == isHighPt);
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuonVIDCITKAnalyzer);
