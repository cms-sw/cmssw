/// \brief Abstract
/*!
\author Daniele Benedetti
\date November 2010
  PFTrackTransformer transforms all the tracks in the PFRecTracks.
  NOTE the PFRecTrack collection is transient.  
*/

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

class PFTrackProducer : public edm::stream::EDProducer<> {
public:
  ///Constructor
  explicit PFTrackProducer(const edm::ParameterSet&);

private:
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;

  ///Produce the PFRecTrack collection
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;

  ///PFTrackTransformer
  std::unique_ptr<PFTrackTransformer> pfTransformer_;
  std::vector<edm::EDGetTokenT<reco::TrackCollection>> tracksContainers_;
  std::vector<edm::EDGetTokenT<std::vector<Trajectory>>> trajContainers_;
  edm::EDGetTokenT<reco::GsfTrackCollection> gsfTrackLabel_;
  edm::EDGetTokenT<reco::MuonCollection> muonColl_;
  edm::EDGetTokenT<reco::VertexCollection> vtx_h;
  ///TRACK QUALITY
  bool useQuality_;
  reco::TrackBase::TrackQuality trackQuality_;
  bool trajinev_;
  bool gsfinev_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFTrackProducer);

using namespace std;
using namespace edm;
using namespace reco;
PFTrackProducer::PFTrackProducer(const ParameterSet& iConfig)
    : transientTrackToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      magneticFieldToken_(esConsumes<edm::Transition::BeginRun>()),
      pfTransformer_() {
  produces<reco::PFRecTrackCollection>();

  std::vector<InputTag> tags = iConfig.getParameter<vector<InputTag>>("TkColList");
  trajinev_ = iConfig.getParameter<bool>("TrajInEvents");
  tracksContainers_.reserve(tags.size());
  if (trajinev_) {
    trajContainers_.reserve(tags.size());
  }
  for (auto const& tag : tags) {
    tracksContainers_.push_back(consumes<reco::TrackCollection>(tag));
    if (trajinev_) {
      trajContainers_.push_back(consumes<std::vector<Trajectory>>(tag));
    }
  }

  useQuality_ = iConfig.getParameter<bool>("UseQuality");

  gsfinev_ = iConfig.getParameter<bool>("GsfTracksInEvents");
  if (gsfinev_) {
    gsfTrackLabel_ = consumes<reco::GsfTrackCollection>(iConfig.getParameter<InputTag>("GsfTrackModuleLabel"));
  }

  trackQuality_ = reco::TrackBase::qualityByName(iConfig.getParameter<std::string>("TrackQuality"));

  muonColl_ = consumes<reco::MuonCollection>(iConfig.getParameter<InputTag>("MuColl"));

  vtx_h = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("PrimaryVertexLabel"));
}

void PFTrackProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  //create the empty collections
  auto PfTrColl = std::make_unique<reco::PFRecTrackCollection>();

  //read track collection
  Handle<GsfTrackCollection> gsftrackcoll;
  bool foundgsf = false;
  if (gsfinev_) {
    foundgsf = iEvent.getByToken(gsfTrackLabel_, gsftrackcoll);
  }

  //Get PV for STIP calculation, if there is none then take the dummy
  Handle<reco::VertexCollection> vertex;
  iEvent.getByToken(vtx_h, vertex);
  reco::Vertex dummy;
  const reco::Vertex* pv = &dummy;
  if (vertex.isValid()) {
    pv = &*vertex->begin();
  } else {  // create a dummy PV
    reco::Vertex::Error e;
    e(0, 0) = 0.0015 * 0.0015;
    e(1, 1) = 0.0015 * 0.0015;
    e(2, 2) = 15. * 15.;
    reco::Vertex::Point p(0, 0, 0);
    dummy = reco::Vertex(p, e, 0, 0, 0);
  }

  //setup transient track builder
  TransientTrackBuilder const& thebuilder = iSetup.getData(transientTrackToken_);

  // read muon collection
  Handle<reco::MuonCollection> recMuons;
  iEvent.getByToken(muonColl_, recMuons);

  //default value for when trajinev_ is false
  const vector<Trajectory> dummyTj(0);

  for (unsigned int istr = 0; istr < tracksContainers_.size(); istr++) {
    //Track collection
    Handle<reco::TrackCollection> tkRefCollection;
    iEvent.getByToken(tracksContainers_[istr], tkRefCollection);
    reco::TrackCollection Tk = *(tkRefCollection.product());

    //Use a pointer to aoid unnecessary copying of the collection
    const vector<Trajectory>* Tj = &dummyTj;
    if (trajinev_) {
      //Trajectory collection
      Handle<vector<Trajectory>> tjCollection;
      iEvent.getByToken(trajContainers_[istr], tjCollection);

      Tj = tjCollection.product();
    }

    for (unsigned int i = 0; i < Tk.size(); i++) {
      reco::TrackRef trackRef(tkRefCollection, i);

      if (useQuality_ && (!(Tk[i].quality(trackQuality_)))) {
        bool isMuCandidate = false;

        //TrackRef trackRef(tkRefCollection, i);

        if (recMuons.isValid()) {
          for (unsigned j = 0; j < recMuons->size(); j++) {
            reco::MuonRef muonref(recMuons, j);
            if (muonref->track().isNonnull())
              if (muonref->track() == trackRef && muonref->isGlobalMuon()) {
                isMuCandidate = true;
                //cout<<" SAVING TRACK "<<endl;
                break;
              }
          }
        }
        if (!isMuCandidate) {
          continue;
        }
      }

      // find the pre-id kf track
      bool preId = false;
      if (foundgsf) {
        //NOTE: foundgsf is only true if gsftrackcoll is valid
        for (auto const& gsfTrack : *gsftrackcoll) {
          if (gsfTrack.seedRef().isNull())
            continue;
          auto const& seed = *(gsfTrack.extra()->seedRef());
          auto const& ElSeed = dynamic_cast<ElectronSeed const&>(seed);
          if (ElSeed.ctfTrack().isNonnull()) {
            if (ElSeed.ctfTrack() == trackRef) {
              preId = true;
              break;
            }
          }
        }
      }
      if (preId) {
        // Set PFRecTrack of type KF_ElCAND
        reco::PFRecTrack pftrack(trackRef->charge(), reco::PFRecTrack::KF_ELCAND, i, trackRef);

        bool valid = false;
        if (trajinev_) {
          valid = pfTransformer_->addPoints(pftrack, *trackRef, (*Tj)[i]);
        } else {
          Trajectory FakeTraj;
          valid = pfTransformer_->addPoints(pftrack, *trackRef, FakeTraj);
        }
        if (valid) {
          //calculate STIP
          double stip = -999;
          const reco::PFTrajectoryPoint& atECAL = pftrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance);
          if (atECAL.isValid())  //if track extrapolates to ECAL
          {
            GlobalVector direction(pftrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().x(),
                                   pftrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().y(),
                                   pftrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().z());
            stip = IPTools::signedTransverseImpactParameter(thebuilder.build(*trackRef), direction, *pv)
                       .second.significance();
          }
          pftrack.setSTIP(stip);
          PfTrColl->push_back(pftrack);
        }
      } else {
        reco::PFRecTrack pftrack(trackRef->charge(), reco::PFRecTrack::KF, i, trackRef);
        bool valid = false;
        if (trajinev_) {
          valid = pfTransformer_->addPoints(pftrack, *trackRef, (*Tj)[i]);
        } else {
          Trajectory FakeTraj;
          valid = pfTransformer_->addPoints(pftrack, *trackRef, FakeTraj);
        }

        if (valid) {
          double stip = -999;
          const reco::PFTrajectoryPoint& atECAL = pftrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance);
          if (atECAL.isValid()) {
            GlobalVector direction(pftrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().x(),
                                   pftrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().y(),
                                   pftrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().z());
            stip = IPTools::signedTransverseImpactParameter(thebuilder.build(*trackRef), direction, *pv)
                       .second.significance();
          }
          pftrack.setSTIP(stip);
          PfTrColl->push_back(pftrack);
        }
      }
    }
  }
  iEvent.put(std::move(PfTrColl));
}

// ------------ method called once each job just before starting event loop  ------------
void PFTrackProducer::beginRun(const edm::Run& run, const EventSetup& iSetup) {
  auto const& magneticField = iSetup.getData(magneticFieldToken_);
  pfTransformer_ = std::make_unique<PFTrackTransformer>(math::XYZVector(magneticField.inTesla(GlobalPoint(0, 0, 0))));
  if (!trajinev_)
    pfTransformer_->OnlyProp();
}

// ------------ method called once each job just after ending the event loop  ------------
void PFTrackProducer::endRun(const edm::Run& run, const EventSetup& iSetup) { pfTransformer_.reset(); }
