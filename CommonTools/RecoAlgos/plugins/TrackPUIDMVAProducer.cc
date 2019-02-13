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
  static constexpr char puId3DmvaName[] = "puId3DMVA";
  static constexpr char puId4DmvaName[] = "puId4DMVA";
  
  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  edm::EDGetTokenT<reco::TrackCollection> tracksMTDToken_;

  edm::EDGetTokenT<edm::ValueMap<float> > btlMatchChi2Token_;
  edm::EDGetTokenT<edm::ValueMap<float> > btlMatchTimeChi2Token_;
  edm::EDGetTokenT<edm::ValueMap<float> > etlMatchChi2Token_;
  edm::EDGetTokenT<edm::ValueMap<float> > etlMatchTimeChi2Token_;
  edm::EDGetTokenT<edm::ValueMap<float> > mtdTimeToken_;
  edm::EDGetTokenT<edm::ValueMap<float> > pathLengthToken_;
  edm::EDGetTokenT<edm::ValueMap<float> > t0PIDToken_;
  edm::EDGetTokenT<edm::ValueMap<float> > sigmat0PIDToken_;

  edm::EDGetTokenT<reco::VertexCollection> vtxsToken_;
  edm::EDGetTokenT<reco::VertexCollection> vtxs4DToken_;
  double maxDz_;

  TrackPUIDMVA mva3D_;
  TrackPUIDMVA mva4D_;    

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
  t0PIDToken_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("t0TOFPIDSrc"))),
  sigmat0PIDToken_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("sigmat0TOFPIDSrc"))),
  vtxsToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vtxsSrc"))),
  vtxs4DToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vtxs4DSrc"))),
  maxDz_(iConfig.getParameter<double>("maxDz")),
  mva3D_(iConfig.getParameter<edm::FileInPath>("trackPUID_3DBDT_weights_file").fullPath(), false),
  mva4D_(iConfig.getParameter<edm::FileInPath>("trackPUID_4DBDT_weights_file").fullPath(), true)
{  
  produces<edm::ValueMap<float> >(puId3DmvaName);
  produces<edm::ValueMap<float> >(puId4DmvaName);
}

// Configuration descriptions
void TrackPUIDMVAProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracksSrc", edm::InputTag("generalTracks"))->
    setComment("Input tracks collection");
  desc.add<edm::InputTag>("tracksMTDSrc", edm::InputTag("trackExtenderWithMTD"))->
    setComment("Input tracks collection for MTD extended tracks");
  desc.add<edm::InputTag>("vtxsSrc", edm::InputTag("offlinePrimaryVertices"))->
    setComment("Input primary vertex collection");
  desc.add<edm::InputTag>("vtxs4DSrc", edm::InputTag("offlinePrimaryVertices4D"))->
    setComment("Input primary vertex 4D collection");
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
  desc.add<edm::InputTag>("t0TOFPIDSrc", edm::InputTag("tofPID", "t0"))->
    setComment("TOFPID T0 value Map");
  desc.add<edm::InputTag>("sigmat0TOFPIDSrc", edm::InputTag("tofPID", "sigmat0"))->
    setComment("TOFPID sigmaT0 value Map");
  desc.add<double>("maxDz", 1.)->
    setComment("Maximum distance in z for track-primary vertex association for particle id [cm]");
  desc.add<edm::FileInPath>("trackPUID_3DBDT_weights_file",edm::FileInPath("CommonTools/RecoAlgos/data/clf3D_dz1cm_bo.xml"))->
    setComment("Track PUID 3D BDT weights");
  desc.add<edm::FileInPath>("trackPUID_4DBDT_weights_file",edm::FileInPath("CommonTools/RecoAlgos/data/clf4D_dz1cm_bo.xml"))->
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
  
  edm::Handle<reco::VertexCollection> vtxsH;  
  ev.getByToken(vtxsToken_,vtxsH);
  const auto& vtxs = *vtxsH;

  edm::Handle<reco::VertexCollection> vtxs4DH;  
  ev.getByToken(vtxs4DToken_,vtxs4DH);
  const auto& vtxs4D = *vtxs4DH;

  edm::Handle<edm::ValueMap<float> > btlMatchChi2H;
  edm::Handle<edm::ValueMap<float> > btlMatchTimeChi2H;
  edm::Handle<edm::ValueMap<float> > etlMatchChi2H;
  edm::Handle<edm::ValueMap<float> > etlMatchTimeChi2H;
  edm::Handle<edm::ValueMap<float> > mtdTimeH;
  edm::Handle<edm::ValueMap<float> > pathLengthH;
  edm::Handle<edm::ValueMap<float> > t0PIDH;
  edm::Handle<edm::ValueMap<float> > sigmat0PIDH;
  
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
  ev.getByToken(t0PIDToken_, t0PIDH);
  auto t0PID = *t0PIDH.product();    
  ev.getByToken(sigmat0PIDToken_, sigmat0PIDH);
  auto sigmat0PID = *sigmat0PIDH.product();    
  
  std::vector<float> puID3DmvaOutRaw;
  std::vector<float> puID4DmvaOutRaw;

  //Loop over tracks collection
  for (unsigned int itrack = 0; itrack<tracks.size(); ++itrack) {
    const reco::Track &track = tracks[itrack];
    const reco::TrackRef trackref(tracksH,itrack);
    const reco::TrackRef mtdTrackref(tracksMTDH,itrack);

    //---training performed only above 0.5 GeV
    if(track.pt() < 0.5)
      {
	puID3DmvaOutRaw.push_back(-1.);
	puID4DmvaOutRaw.push_back(-1.);
      }
    else
      {
	puID3DmvaOutRaw.push_back(vtxs.size()>0  && std::abs(track.dz(vtxs[0].position()))<maxDz_ ? mva3D_(trackref, vtxs[0]) : -1.); 
	puID4DmvaOutRaw.push_back(vtxs4D.size()>0 && std::abs(track.dz(vtxs4D[0].position()))<maxDz_ ? 
				  mva4D_(trackref, mtdTrackref, vtxs4D[0],
					 t0PID, sigmat0PID, btlMatchChi2, btlMatchTimeChi2, etlMatchChi2, etlMatchTimeChi2,
					 mtdTime, pathLength) : -1.);
      }
  }

  fillValueMap(ev, tracksH, puID3DmvaOutRaw, puId3DmvaName);
  fillValueMap(ev, tracksH, puID4DmvaOutRaw, puId4DmvaName);  
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(TrackPUIDMVAProducer);
