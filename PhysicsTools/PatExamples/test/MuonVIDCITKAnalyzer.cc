#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH1F.h"
#include "TH2F.h"

#include <sstream>

using namespace std;
using namespace edm;
//
// class decleration
//
class MuonVIDCITKAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources>
{
public:
  MuonVIDCITKAnalyzer(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

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
  typedef edm::ValueMap<float> CITKMap;
  edm::EDGetTokenT<CITKMap> muonChIsoCITKToken_;
  edm::EDGetTokenT<CITKMap> muonNhIsoCITKToken_;
  edm::EDGetTokenT<CITKMap> muonPhIsoCITKToken_;
  edm::EDGetTokenT<CITKMap> muonPuIsoCITKToken_;

  // IsoDeposit
  typedef edm::ValueMap<double> IsoMap;
  edm::EDGetTokenT<IsoMap> muonChIsoIsoDepToken_;
  edm::EDGetTokenT<IsoMap> muonNhIsoIsoDepToken_;
  edm::EDGetTokenT<IsoMap> muonPhIsoIsoDepToken_;
  edm::EDGetTokenT<IsoMap> muonPuIsoIsoDepToken_;

  // Histograms
  TH2F* h2Ch_, * h2Nh_, * h2Ph_, * h2Pu_;
  TH1F* hIsoDiffCh_, * hIsoDiffNh_, * hIsoDiffPh_, * hIsoDiffPu_;
};

MuonVIDCITKAnalyzer::MuonVIDCITKAnalyzer(const edm::ParameterSet& iConfig)
{
  muonToken_ = consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("muon"));
  vertexToken_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertex"));

  string vidPrefix = "muoMuonIDs:cutBasedMuonId-MuonPOG-V0-";
  muonLooseVIDToken_  = consumes<VIDMap>(edm::InputTag(vidPrefix+"loose" ));
  muonMediumVIDToken_ = consumes<VIDMap>(edm::InputTag(vidPrefix+"medium"));
  muonTightVIDToken_  = consumes<VIDMap>(edm::InputTag(vidPrefix+"tight" ));
  muonSoftVIDToken_   = consumes<VIDMap>(edm::InputTag(vidPrefix+"soft"  ));
  muonHighPtVIDToken_ = consumes<VIDMap>(edm::InputTag(vidPrefix+"highpt"));

  muonChIsoCITKToken_ = consumes<CITKMap>(edm::InputTag("muonPFNoPileUpIsolation:h+-DR040-ThresholdVeto000-ConeVeto000"));
  muonNhIsoCITKToken_ = consumes<CITKMap>(edm::InputTag("muonPFNoPileUpIsolation:h0-DR040-ThresholdVeto050-ConeVeto001"));
  muonPhIsoCITKToken_ = consumes<CITKMap>(edm::InputTag("muonPFNoPileUpIsolation:gamma-DR040-ThresholdVeto050-ConeVeto001"));
  muonPuIsoCITKToken_ = consumes<CITKMap>(edm::InputTag("muonPFPileUpIsolation:h+-DR040-ThresholdVeto050-ConeVeto001"));

  muonChIsoIsoDepToken_ = consumes<IsoMap>(edm::InputTag("muPFIsoValueCharged04PAT"));
  muonNhIsoIsoDepToken_ = consumes<IsoMap>(edm::InputTag("muPFIsoValueNeutral04PAT"));
  muonPhIsoIsoDepToken_ = consumes<IsoMap>(edm::InputTag("muPFIsoValueGamma04PAT"));
  muonPuIsoIsoDepToken_ = consumes<IsoMap>(edm::InputTag("muPFIsoValuePU04PAT"));

  usesResource("TFileService");
  edm::Service<TFileService> fs;
  h2Ch_ = fs->make<TH2F>("h2Ch", "ChIso;IsoValue ChIso;CITK ChIso", 100, 0, 10, 100, 0, 10);
  h2Nh_ = fs->make<TH2F>("h2Nh", "NhIso;IsoValue NhIso;CITK NhIso", 100, 0, 10, 100, 0, 10);
  h2Ph_ = fs->make<TH2F>("h2Ph", "PhIso;IsoValue PhIso;CITK PhIso", 100, 0, 10, 100, 0, 10);
  h2Pu_ = fs->make<TH2F>("h2Pu", "PuIso;IsoValue PuIso;CITK PuIso", 100, 0, 10, 100, 0, 10);

  hIsoDiffCh_ = fs->make<TH1F>("hIsoDiffCh", "Diff. ChIso;CITK-IsoValue", 1000, -100, 100);
  hIsoDiffNh_ = fs->make<TH1F>("hIsoDiffNh", "Diff. NhIso;CITK-IsoValue", 1000, -100, 100);
  hIsoDiffPh_ = fs->make<TH1F>("hIsoDiffPh", "Diff. PhIso;CITK-IsoValue", 1000, -100, 100);
  hIsoDiffPu_ = fs->make<TH1F>("hIsoDiffPu", "Diff. PuIso;CITK-IsoValue", 1000, -100, 100);

}

void MuonVIDCITKAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& eSetup)
{
  edm::Handle<edm::View<reco::Candidate> > muonHandle;
  event.getByToken(muonToken_, muonHandle);

  edm::Handle<reco::VertexCollection> vertexHandle;
  event.getByToken(vertexToken_, vertexHandle);
  reco::Vertex vertex = vertexHandle->at(0);

  // VID
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

  // CITK
  edm::Handle<CITKMap> muonChIsoCITKHandle;
  edm::Handle<CITKMap> muonNhIsoCITKHandle;
  edm::Handle<CITKMap> muonPhIsoCITKHandle;
  edm::Handle<CITKMap> muonPuIsoCITKHandle;
  event.getByToken(muonChIsoCITKToken_, muonChIsoCITKHandle);
  event.getByToken(muonNhIsoCITKToken_, muonNhIsoCITKHandle);
  event.getByToken(muonPhIsoCITKToken_, muonPhIsoCITKHandle);
  event.getByToken(muonPuIsoCITKToken_, muonPuIsoCITKHandle);

  // IsoDeposit
  edm::Handle<IsoMap> muonChIsoIsoDepHandle;
  edm::Handle<IsoMap> muonNhIsoIsoDepHandle;
  edm::Handle<IsoMap> muonPhIsoIsoDepHandle;
  edm::Handle<IsoMap> muonPuIsoIsoDepHandle;
  event.getByToken(muonChIsoIsoDepToken_, muonChIsoIsoDepHandle);
  event.getByToken(muonNhIsoIsoDepToken_, muonNhIsoIsoDepHandle);
  event.getByToken(muonPhIsoIsoDepToken_, muonPhIsoIsoDepHandle);
  event.getByToken(muonPuIsoIsoDepToken_, muonPuIsoIsoDepHandle);

  for ( size_t i=0, n=muonHandle->size(); i<n; ++i )
  {
    const auto& mu = dynamic_cast<const pat::Muon&>(muonHandle->at(i));
    const auto& muRef = mu.originalObjectRef();

    stringstream sout;

    // Check standard ID vs VID
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

    if ( vidLoose  != isLoose  ) { sout << " isLoose " << vidLoose  << ' ' << isLoose  << endl; }
    if ( vidMedium != isMedium ) { sout << " isMedium" << vidMedium << ' ' << isMedium << endl; }
    if ( vidTight  != isTight  ) { sout << " isTight " << vidTight  << ' ' << isTight  << endl; }
    if ( vidSoft   != isSoft   ) { sout << " isSoft  " << vidSoft   << ' ' << isSoft   << endl; }
    if ( vidHighPt != isHighPt ) { sout << " isHighPt" << vidHighPt << ' ' << isHighPt << endl; }

    // Check standard IsoDeposit vs CITK
    const double citkChIso = (*muonChIsoCITKHandle)[muRef];
    const double citkNhIso = (*muonNhIsoCITKHandle)[muRef];
    const double citkPhIso = (*muonPhIsoCITKHandle)[muRef];
    const double citkPuIso = (*muonPuIsoCITKHandle)[muRef];

    const double isoDepChIso = (*muonChIsoIsoDepHandle)[muRef];
    const double isoDepNhIso = (*muonNhIsoIsoDepHandle)[muRef];
    const double isoDepPhIso = (*muonPhIsoIsoDepHandle)[muRef];
    const double isoDepPuIso = (*muonPuIsoIsoDepHandle)[muRef];

    const double patChIso = mu.chargedHadronIso();
    const double patNhIso = mu.neutralHadronIso();
    const double patPhIso = mu.photonIso();
    const double patPuIso = mu.puChargedHadronIso();

    h2Ch_->Fill(isoDepChIso, citkChIso);
    h2Nh_->Fill(isoDepNhIso, citkNhIso);
    h2Ph_->Fill(isoDepPhIso, citkPhIso);
    h2Pu_->Fill(isoDepPuIso, citkPuIso);

    hIsoDiffCh_->Fill(citkChIso-isoDepChIso);
    hIsoDiffNh_->Fill(citkNhIso-isoDepNhIso);
    hIsoDiffPh_->Fill(citkPhIso-isoDepPhIso);
    hIsoDiffPu_->Fill(citkPuIso-isoDepPuIso);

    if ( std::abs(citkChIso-isoDepChIso) >= 1e-4 ) { sout << " ChIso citk=" << citkChIso << " isodep=" << isoDepChIso << endl; }
    if ( std::abs(citkNhIso-isoDepNhIso) >= 1e-4 ) { sout << " NhIso citk=" << citkNhIso << " isodep=" << isoDepNhIso << endl; }
    if ( std::abs(citkPhIso-isoDepPhIso) >= 1e-4 ) { sout << " PhIso citk=" << citkPhIso << " isodep=" << isoDepPhIso << endl; }
    if ( std::abs(citkPuIso-isoDepPuIso) >= 1e-4 ) { sout << " PuIso citk=" << citkPuIso << " isodep=" << isoDepPuIso << endl; }

    if ( std::abs(patChIso-isoDepChIso) >= 1e-4 ) { sout << " ChIso pat=" << patChIso << " isodep=" << isoDepChIso << endl; }
    if ( std::abs(patNhIso-isoDepNhIso) >= 1e-4 ) { sout << " NhIso pat=" << patNhIso << " isodep=" << isoDepNhIso << endl; }
    if ( std::abs(patPhIso-isoDepPhIso) >= 1e-4 ) { sout << " PhIso pat=" << patPhIso << " isodep=" << isoDepPhIso << endl; }
    if ( std::abs(patPuIso-isoDepPuIso) >= 1e-4 ) { sout << " PuIso pat=" << patPuIso << " isodep=" << isoDepPuIso << endl; }

    if ( !sout.str().empty() )
    {
      cout << event.id().event() << " mu" << i << " pt=" << muRef->pt() << " eta=" << muRef->eta() << " phi=" << muRef->phi() << endl;
      cout << sout.rdbuf();
    }
  }

}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuonVIDCITKAnalyzer);
