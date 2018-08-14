
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/Jet.h"


class TrackingFailureFilter : public edm::global::EDFilter<> {

  public:

    explicit TrackingFailureFilter(const edm::ParameterSet & iConfig);
    ~TrackingFailureFilter() override {}

  private:

    bool filter(edm::StreamID, edm::Event & iEvent, const edm::EventSetup & iSetup) const override;

    edm::EDGetTokenT<edm::View<reco::Jet> > jetSrcToken_;
    edm::EDGetTokenT<std::vector<reco::Track> > trackSrcToken_;
    edm::EDGetTokenT<std::vector<reco::Vertex> > vertexSrcToken_;
    const double dzTrVtxMax_, dxyTrVtxMax_, minSumPtOverHT_;

    const bool taggingMode_, debug_;

};


TrackingFailureFilter::TrackingFailureFilter(const edm::ParameterSet & iConfig)
  : jetSrcToken_          (consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("JetSource")))
  , trackSrcToken_        (consumes<std::vector<reco::Track> >(iConfig.getParameter<edm::InputTag>("TrackSource")))
  , vertexSrcToken_       (consumes<std::vector<reco::Vertex> >(iConfig.getParameter<edm::InputTag>("VertexSource")))
  , dzTrVtxMax_      (iConfig.getParameter<double>("DzTrVtxMax"))
  , dxyTrVtxMax_     (iConfig.getParameter<double>("DxyTrVtxMax"))
  , minSumPtOverHT_  (iConfig.getParameter<double>("MinSumPtOverHT"))
  , taggingMode_     (iConfig.getParameter<bool>("taggingMode"))
  , debug_           (iConfig.getParameter<bool>("debug"))
{

  produces<bool>();
}


bool TrackingFailureFilter::filter(edm::StreamID, edm::Event & iEvent, const edm::EventSetup & iSetup) const {

  edm::Handle<edm::View<reco::Jet> > jets;
  iEvent.getByToken(jetSrcToken_, jets);
  edm::Handle<std::vector<reco::Track> > tracks;
  iEvent.getByToken(trackSrcToken_, tracks);
  edm::Handle<std::vector<reco::Vertex> > vtxs;
  iEvent.getByToken(vertexSrcToken_, vtxs);

  double ht = 0;
  for (edm::View<reco::Jet>::const_iterator j = jets->begin(); j != jets->end(); ++j) {
    ht += j->pt();
  }
  double sumpt = 0;
  if (!vtxs->empty()) {
//    const reco::Vertex * vtx = &((*vtxs)[0]);
    for (std::vector<reco::Track>::const_iterator tr = tracks->begin(); tr != tracks->end(); ++tr) {
      bool associateToPV = false;
      for(int iv=0; iv<(int)vtxs->size(); iv++){
         const reco::Vertex * pervtx = &((*vtxs)[iv]);
         if( fabs(tr->dz(pervtx->position())) <= dzTrVtxMax_ && fabs(tr->dxy(pervtx->position())) <= dxyTrVtxMax_ ){
            associateToPV = true;
         }
      }
//      if (fabs(tr->dz(vtx->position())) > dzTrVtxMax_) continue;
//      if (fabs(tr->dxy(vtx->position())) > dxyTrVtxMax_) continue;
      if( !associateToPV ) continue;
      sumpt += tr->pt();
    }
  }
  const bool pass = (sumpt/ht) > minSumPtOverHT_;

  if( !pass && debug_ )
    edm::LogInfo("TrackingFailureFilter")
              << "TRACKING FAILURE: "
              << iEvent.id().run() << " : " << iEvent.id().luminosityBlock() << " : " << iEvent.id().event()
              << " HT=" << ht
              << " SumPt=" << sumpt;

  iEvent.put(std::make_unique<bool>(pass));

  return taggingMode_ || pass; // return false if filtering and not enough tracks in event

}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackingFailureFilter);
