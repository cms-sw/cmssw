#include "DQM/Physics/src/CentralityDQM.h"

#include <memory>

// DQM
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Centrality
#include "DataFormats/VertexReco/interface/Vertex.h"

using namespace edm;
using namespace reco;
using namespace hi;

//
// -- Constructor
//
CentralityDQM::CentralityDQM(const edm::ParameterSet& ps) {
  edm::LogInfo("CentralityDQM") << " Starting CentralityDQM "
                                << "\n";

  centralityTag_ = ps.getParameter<InputTag>("centralitycollection");
  centralityToken = consumes<reco::Centrality>(centralityTag_);

  centralityBinTag_ = (ps.getParameter<edm::InputTag>("centralitybincollection"));
  centralityBinToken = consumes<int>(centralityBinTag_);

  vertexTag_ = ps.getParameter<InputTag>("vertexcollection");
  vertexToken = consumes<std::vector<reco::Vertex> >(vertexTag_);

  eventplaneTag_ = ps.getParameter<InputTag>("eventplanecollection");
  eventplaneToken = consumes<reco::EvtPlaneCollection>(eventplaneTag_);

  // just to initialize
}

//
// -- Destructor
//
CentralityDQM::~CentralityDQM() {
  edm::LogInfo("CentralityDQM") << " Deleting CentralityDQM "
                                << "\n";
}

//
//  -- Book histograms
//
void CentralityDQM::bookHistograms(DQMStore::IBooker& bei, edm::Run const&, edm::EventSetup const&) {
  // void CentralityDQM::bookHistograms(DQMStore* bei){

  bei.setCurrentFolder("Physics/Centrality/");

  h_hiNpix = bei.book1D("h_hiNpix", "h_hiNpix", 750, 0, 75000);
  h_hiNpixelTracks = bei.book1D("h_hiNpixelTracks", "hiNpixelTracks", 500, 0, 5000);
  h_hiNtracks = bei.book1D("h_hiNtracks", "h_hiNtracks", 500, 0, 5000);
  h_hiNtracksPtCut = bei.book1D("h_hiNtracksPtCut", "h_hiNtracksPtCut", 500, 0, 5000);
  h_hiNtracksEtaCut = bei.book1D("h_hiNtracksEtaCut", "h_hiNtracksEtaCut", 500, 0, 5000);
  h_hiNtracksEtaPtCut = bei.book1D("h_hiNtracksEtaPtCut", "h_hiNtracksEtaPtCut", 500, 0, 5000);

  h_hiHF = bei.book1D("h_hiHF", "h_hiHF", 900, 0, 9000);
  h_hiHFplus = bei.book1D("h_hiHFplus", "h_hiHFplus", 900, 0, 9000);
  h_hiHFminus = bei.book1D("h_hiHFminus", "h_hiHFminus", 900, 0, 9000);
  h_hiHFplusEta4 = bei.book1D("h_hiHFplusEta4", "h_hiHFplusEta4", 900, 0, 9000);
  h_hiHFminusEta4 = bei.book1D("h_hiHFminusEta4", "h_hiHFminusEta4", 900, 0, 9000);

  h_hiHFhit = bei.book1D("h_hiHFhit", "h_hiHFhit", 3000, 0, 300000);
  h_hiHFhitPlus = bei.book1D("h_hiHFhitPlus", "h_hiHFhitPlus", 2000, 0, 200000);
  h_hiHFhitMinus = bei.book1D("h_hiHFhitMinus", "h_hiHFhitMinus", 2000, 0, 200000);

  h_hiEB = bei.book1D("h_hiEB", "h_hiEB", 600, 0, 6000);
  h_hiET = bei.book1D("h_hiET", "h_hiET", 600, 0, 6000);
  h_hiEE = bei.book1D("h_hiEE", "h_hiEE", 600, 0, 6000);
  h_hiEEplus = bei.book1D("h_hiEEplus", "h_hiEEplus", 600, 0, 6000);
  h_hiEEminus = bei.book1D("h_hiEEminus", "h_hiEEminus", 600, 0, 6000);
  h_hiZDC = bei.book1D("h_hiZDC", "h_hiZDC", 600, 0, 6000);
  h_hiZDCplus = bei.book1D("h_hiZDCplus", "h_hiZDCplus", 600, 0, 6000);
  h_hiZDCminus = bei.book1D("h_hiZDCminus", "h_hiZDCminus", 600, 0, 6000);

  h_vertex_x = bei.book1D("h_vertex_x", "h_vertex_x", 400, -4, 4);
  h_vertex_y = bei.book1D("h_vertex_y", "h_vertex_y", 400, -4, 4);
  h_vertex_z = bei.book1D("h_vertex_z", "h_vertex_z", 400, -40, 40);

  h_cent_bin = bei.book1D("h_cent_bin", "h_cent_bin", 200, 0, 200);

  Double_t psirange = 4;
  bei.setCurrentFolder("Physics/Centrality/EventPlane/");
  h_ep_HFm2 = bei.book1D("h_ep_HFm2", "h_ep_HFm2", 800, -psirange, psirange);
  h_ep_HFp2 = bei.book1D("h_ep_HFp2", "h_ep_HFp2", 800, -psirange, psirange);
  h_ep_trackmid2 = bei.book1D("h_ep_trackmid2", "h_ep_trackmid2", 800, -psirange, psirange);
  h_ep_trackm2 = bei.book1D("h_ep_trackm2", "h_ep_trackm2", 800, -psirange, psirange);
  h_ep_trackp2 = bei.book1D("h_ep_trackp2", "h_ep_trackp2", 800, -psirange, psirange);
  h_ep_HFm3 = bei.book1D("h_ep_HFm3", "h_ep_HFm3", 800, -psirange, psirange);
  h_ep_HFp3 = bei.book1D("h_ep_HFp3", "h_ep_HFp3", 800, -psirange, psirange);
  h_ep_trackmid3 = bei.book1D("h_ep_trackmid3", "h_ep_trackmid3", 800, -psirange, psirange);
}

//
//  -- Analyze
//
void CentralityDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  edm::Handle<reco::Centrality> cent;
  iEvent.getByToken(centralityToken, cent);  //_centralitytag comes from the cfg

  edm::Handle<int> cbin;
  iEvent.getByToken(centralityBinToken, cbin);

  // as an inputTag and is
  //"hiCentrality"
  edm::Handle<reco::EvtPlaneCollection> ep;
  iEvent.getByToken(eventplaneToken, ep);

  if (cent.isValid()) {
    int hibin = -999;
    if (cbin.isValid())
      hibin = *cbin;

    //  std::cout<<  " ------------------------------------- "  << hibin << std::endl;

    h_cent_bin->Fill(hibin);
    h_hiNpix->Fill(cent->multiplicityPixel());
    h_hiNpixelTracks->Fill(cent->NpixelTracks());
    h_hiNtracks->Fill(cent->Ntracks());  //

    h_hiNtracksPtCut->Fill(cent->NtracksPtCut());
    h_hiNtracksEtaCut->Fill(cent->NtracksEtaCut());
    h_hiNtracksEtaPtCut->Fill(cent->NtracksEtaPtCut());

    h_hiHF->Fill(cent->EtHFtowerSum());
    h_hiHFplus->Fill(cent->EtHFtowerSumPlus());
    h_hiHFminus->Fill(cent->EtHFtowerSumMinus());
    h_hiHFplusEta4->Fill(cent->EtHFtruncatedPlus());
    h_hiHFminusEta4->Fill(cent->EtHFtruncatedMinus());

    h_hiHFhit->Fill(cent->EtHFhitSum());
    h_hiHFhitPlus->Fill(cent->EtHFhitSumPlus());
    h_hiHFhitMinus->Fill(cent->EtHFhitSumMinus());

    h_hiZDC->Fill(cent->zdcSum());
    h_hiZDCplus->Fill(cent->zdcSumPlus());
    h_hiZDCminus->Fill(cent->zdcSumMinus());

    h_hiEEplus->Fill(cent->EtEESumPlus());
    h_hiEEminus->Fill(cent->EtEESumMinus());
    h_hiEE->Fill(cent->EtEESum());
    h_hiEB->Fill(cent->EtEBSum());
    h_hiET->Fill(cent->EtMidRapiditySum());

    edm::Handle<std::vector<reco::Vertex> > vertex;
    iEvent.getByToken(vertexToken, vertex);
    h_vertex_x->Fill(vertex->begin()->x());
    h_vertex_y->Fill(vertex->begin()->y());
    h_vertex_z->Fill(vertex->begin()->z());
  }

  if (ep.isValid()) {
    EvtPlaneCollection::const_iterator rp = ep->begin();

    h_ep_HFm2->Fill(rp[HFm2].angle(0));
    h_ep_HFp2->Fill(rp[HFp2].angle(0));
    h_ep_trackmid2->Fill(rp[trackmid2].angle(0));
    h_ep_trackm2->Fill(rp[trackm2].angle(0));
    h_ep_trackp2->Fill(rp[trackp2].angle(0));

    h_ep_HFm3->Fill(rp[HFm3].angle(0));
    h_ep_HFp3->Fill(rp[HFp3].angle(0));
    h_ep_trackmid3->Fill(rp[trackmid3].angle(0));
  }
}
