#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include "CommonTools/RecoAlgos/interface/TrackPUIDMVA.h"

using namespace std;
using namespace edm;

class TrackPUIDMVAProducer : public edm::stream::EDProducer<> {  
 public:
  
  TrackPUIDMVAProducer(const ParameterSet& pset); 

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
  template <class H, class T>
  void fillValueMap(edm::Event& iEvent, const edm::Handle<H>& handle, const std::vector<T>& vec, const std::string& name) const;
  
  void produce(edm::Event& ev, const edm::EventSetup& es) final;

 private:
  static constexpr char mvaName[] = "mtdQualMVA";
  
  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  edm::EDGetTokenT<reco::TrackCollection> tracksMTDToken_;

  edm::EDGetTokenT<edm::ValueMap<float> > btlMatchChi2Token_;
  edm::EDGetTokenT<edm::ValueMap<float> > btlMatchTimeChi2Token_;
  edm::EDGetTokenT<edm::ValueMap<float> > etlMatchChi2Token_;
  edm::EDGetTokenT<edm::ValueMap<float> > etlMatchTimeChi2Token_;
  edm::EDGetTokenT<edm::ValueMap<float> > mtdTimeToken_;
  edm::EDGetTokenT<edm::ValueMap<float> > pathLengthToken_;

  TrackPUIDMVA mva_;    

};

  
TrackPUIDMVAProducer::TrackPUIDMVAProducer(const ParameterSet& iConfig) :
  tracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracksSrc"))),
  tracksMTDToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracksMTDSrc"))),
  btlMatchChi2Token_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("btlMatchChi2Src"))),
  btlMatchTimeChi2Token_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("btlMatchTimeChi2Src"))),
  etlMatchChi2Token_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("etlMatchChi2Src"))),
  etlMatchTimeChi2Token_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("etlMatchTimeChi2Src"))) ,
  mtdTimeToken_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("mtdTimeSrc"))),
  pathLengthToken_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("pathLengthSrc"))),
  mva_(iConfig.getParameter<edm::FileInPath>("trackPUID_mtdQualBDT_weights_file").fullPath())
{  
  produces<edm::ValueMap<float> >(mvaName);
}

// Configuration descriptions
void TrackPUIDMVAProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracksSrc", edm::InputTag("generalTracks"))->
    setComment("Input tracks collection");
  desc.add<edm::InputTag>("tracksMTDSrc", edm::InputTag("trackExtenderWithMTD"))->
    setComment("Input tracks collection for MTD extended tracks");
  desc.add<edm::InputTag>("btlMatchChi2Src", edm::InputTag("trackExtenderWithMTD", "btlMatchChi2"))->
    setComment("BTL Chi2 Matching value Map");
  desc.add<edm::InputTag>("btlMatchTimeChi2Src", edm::InputTag("trackExtenderWithMTD", "btlMatchTimeChi2"))->
    setComment("BTL Chi2 Matching value Map");
  desc.add<edm::InputTag>("etlMatchChi2Src", edm::InputTag("trackExtenderWithMTD", "etlMatchChi2"))->
    setComment("ETL Chi2 Matching value Map");
  desc.add<edm::InputTag>("etlMatchTimeChi2Src", edm::InputTag("trackExtenderWithMTD", "etlMatchTimeChi2"))->
    setComment("ETL Chi2 Matching value Map");
  desc.add<edm::InputTag>("mtdTimeSrc", edm::InputTag("trackExtenderWithMTD", "tmtd"))->
    setComment("MTD TIme value Map");
  desc.add<edm::InputTag>("pathLengthSrc", edm::InputTag("trackExtenderWithMTD", "pathLength"))->
    setComment("MTD PathLength value Map");
  desc.add<edm::FileInPath>("trackPUID_mtdQualBDT_weights_file",edm::FileInPath("CommonTools/RecoAlgos/data/clf4D_MTDquality_bo.xml"))->
    setComment("Track PUID 4D BDT weights");
  descriptions.add("trackPUIDMVAProducer", desc);
}

template <class H, class T>
void TrackPUIDMVAProducer::fillValueMap(edm::Event& iEvent, const edm::Handle<H>& handle, const std::vector<T>& vec, const std::string& name) const {
  auto out = std::make_unique<edm::ValueMap<T>>();
  typename edm::ValueMap<T>::Filler filler(*out);
  filler.insert(handle, vec.begin(), vec.end());
  filler.fill();
  iEvent.put(std::move(out),name);
}

void TrackPUIDMVAProducer::produce( edm::Event& ev, const edm::EventSetup& es ) {
  
  edm::Handle<reco::TrackCollection> tracksH;  
  ev.getByToken(tracksToken_,tracksH);
  const auto& tracks = *tracksH;

  edm::Handle<reco::TrackCollection> tracksMTDH;  
  ev.getByToken(tracksMTDToken_,tracksMTDH);
  

  edm::Handle<edm::ValueMap<float> > btlMatchChi2H;
  edm::Handle<edm::ValueMap<float> > btlMatchTimeChi2H;
  edm::Handle<edm::ValueMap<float> > etlMatchChi2H;
  edm::Handle<edm::ValueMap<float> > etlMatchTimeChi2H;
  edm::Handle<edm::ValueMap<float> > mtdTimeH;
  edm::Handle<edm::ValueMap<float> > pathLengthH;
  
  ev.getByToken(btlMatchChi2Token_, btlMatchChi2H);
  auto btlMatchChi2 = *btlMatchChi2H.product();
  ev.getByToken(btlMatchTimeChi2Token_, btlMatchTimeChi2H);
  auto btlMatchTimeChi2 = *btlMatchTimeChi2H.product();
  ev.getByToken(etlMatchChi2Token_, etlMatchChi2H);
  auto etlMatchChi2 = *etlMatchChi2H.product();
  ev.getByToken(etlMatchTimeChi2Token_, etlMatchTimeChi2H);
  auto etlMatchTimeChi2 = *etlMatchTimeChi2H.product();
  ev.getByToken(pathLengthToken_, pathLengthH);
  auto pathLength = *pathLengthH.product();
  ev.getByToken(mtdTimeToken_, mtdTimeH);
  auto mtdTime = *mtdTimeH.product();
  
  std::vector<float> mvaOutRaw;

  //Loop over tracks collection
  for (unsigned int itrack = 0; itrack<tracks.size(); ++itrack) {
    const reco::Track &track = tracks[itrack];
    const reco::TrackRef trackref(tracksH,itrack);
    const reco::TrackRef mtdTrackref(tracksMTDH,itrack);

    //---training performed only above 0.5 GeV
    if(track.pt() < 0.5)
	mvaOutRaw.push_back(-1.);
    else
	mvaOutRaw.push_back(
			    mva_(trackref, mtdTrackref, btlMatchChi2, btlMatchTimeChi2, etlMatchChi2, etlMatchTimeChi2,
				 mtdTime, pathLength)
			    );
  }
  fillValueMap(ev, tracksH, mvaOutRaw, mvaName);  
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(TrackPUIDMVAProducer);
