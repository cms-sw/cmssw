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

using namespace std;
using namespace edm;

class TOFPIDProducer : public edm::stream::EDProducer<> {
public:
  TOFPIDProducer(const ParameterSet& pset);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  template <class H, class T>
  void fillValueMap(edm::Event& iEvent,
                    const edm::Handle<H>& handle,
                    const std::vector<T>& vec,
                    const std::string& name) const;

  void produce(edm::Event& ev, const edm::EventSetup& es) final;

private:
  static constexpr char t0Name[] = "t0";
  static constexpr char sigmat0Name[] = "sigmat0";
  static constexpr char t0safeName[] = "t0safe";
  static constexpr char sigmat0safeName[] = "sigmat0safe";
  static constexpr char probPiName[] = "probPi";
  static constexpr char probKName[] = "probK";
  static constexpr char probPName[] = "probP";

  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> t0Token_;
  edm::EDGetTokenT<edm::ValueMap<float>> tmtdToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> sigmat0Token_;
  edm::EDGetTokenT<edm::ValueMap<float>> sigmatmtdToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> pathLengthToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> pToken_;
  edm::EDGetTokenT<reco::VertexCollection> vtxsToken_;
  double vtxMaxSigmaT_;
  double maxDz_;
  double maxDtSignificance_;
  double minProbHeavy_;
  double fixedT0Error_;
};

TOFPIDProducer::TOFPIDProducer(const ParameterSet& iConfig)
    : tracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracksSrc"))),
      t0Token_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("t0Src"))),
      tmtdToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("tmtdSrc"))),
      sigmat0Token_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmat0Src"))),
      sigmatmtdToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmatmtdSrc"))),
      pathLengthToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("pathLengthSrc"))),
      pToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("pSrc"))),
      vtxsToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vtxsSrc"))),
      vtxMaxSigmaT_(iConfig.getParameter<double>("vtxMaxSigmaT")),
      maxDz_(iConfig.getParameter<double>("maxDz")),
      maxDtSignificance_(iConfig.getParameter<double>("maxDtSignificance")),
      minProbHeavy_(iConfig.getParameter<double>("minProbHeavy")),
      fixedT0Error_(iConfig.getParameter<double>("fixedT0Error")) {
  produces<edm::ValueMap<float>>(t0Name);
  produces<edm::ValueMap<float>>(sigmat0Name);
  produces<edm::ValueMap<float>>(t0safeName);
  produces<edm::ValueMap<float>>(sigmat0safeName);
  produces<edm::ValueMap<float>>(probPiName);
  produces<edm::ValueMap<float>>(probKName);
  produces<edm::ValueMap<float>>(probPName);
}

// Configuration descriptions
void TOFPIDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracksSrc", edm::InputTag("generalTracks"))->setComment("Input tracks collection");
  desc.add<edm::InputTag>("t0Src", edm::InputTag("trackExtenderWithMTD:generalTrackt0"))
      ->setComment("Input ValueMap for track time at beamline");
  desc.add<edm::InputTag>("tmtdSrc", edm::InputTag("trackExtenderWithMTD:generalTracktmtd"))
      ->setComment("Input ValueMap for track time at MTD");
  desc.add<edm::InputTag>("sigmat0Src", edm::InputTag("trackExtenderWithMTD:generalTracksigmat0"))
      ->setComment("Input ValueMap for track time uncertainty at beamline");
  desc.add<edm::InputTag>("sigmatmtdSrc", edm::InputTag("trackExtenderWithMTD:generalTracksigmatmtd"))
      ->setComment("Input ValueMap for track time uncertainty at MTD");
  desc.add<edm::InputTag>("pathLengthSrc", edm::InputTag("trackExtenderWithMTD:generalTrackPathLength"))
      ->setComment("Input ValueMap for track path lengh from beamline to MTD");
  desc.add<edm::InputTag>("pSrc", edm::InputTag("trackExtenderWithMTD:generalTrackp"))
      ->setComment("Input ValueMap for track momentum magnitude (normally from refit with MTD hits)");
  desc.add<edm::InputTag>("vtxsSrc", edm::InputTag("unsortedOfflinePrimaryVertices4DwithPID"))
      ->setComment("Input primary vertex collection");
  desc.add<double>("vtxMaxSigmaT", 0.025)
      ->setComment("Maximum primary vertex time uncertainty for use in particle id [ns]");
  desc.add<double>("maxDz", 0.1)
      ->setComment("Maximum distance in z for track-primary vertex association for particle id [cm]");
  desc.add<double>("maxDtSignificance", 5.0)
      ->setComment(
          "Maximum distance in time (normalized by uncertainty) for track-primary vertex association for particle id");
  desc.add<double>("minProbHeavy", 0.75)
      ->setComment("Minimum probability for a particle to be a kaon or proton before reassigning the timestamp");
  desc.add<double>("fixedT0Error", 0.)->setComment("Use a fixed T0 uncertainty [ns]");

  descriptions.add("tofPIDProducer", desc);
}

template <class H, class T>
void TOFPIDProducer::fillValueMap(edm::Event& iEvent,
                                  const edm::Handle<H>& handle,
                                  const std::vector<T>& vec,
                                  const std::string& name) const {
  auto out = std::make_unique<edm::ValueMap<T>>();
  typename edm::ValueMap<T>::Filler filler(*out);
  filler.insert(handle, vec.begin(), vec.end());
  filler.fill();
  iEvent.put(std::move(out), name);
}

void TOFPIDProducer::produce(edm::Event& ev, const edm::EventSetup& es) {
  constexpr double m_k = 0.493677;                                    //[GeV]
  constexpr double m_p = 0.9382720813;                                //[GeV]
  constexpr double c_cm_ns = CLHEP::c_light * CLHEP::ns / CLHEP::cm;  //[cm/ns]
  constexpr double c_inv = 1.0 / c_cm_ns;

  edm::Handle<reco::TrackCollection> tracksH;
  ev.getByToken(tracksToken_, tracksH);
  const auto& tracks = *tracksH;

  edm::Handle<edm::ValueMap<float>> t0H;
  ev.getByToken(t0Token_, t0H);
  const auto& t0In = *t0H;

  edm::Handle<edm::ValueMap<float>> tmtdH;
  ev.getByToken(tmtdToken_, tmtdH);
  const auto& tmtdIn = *tmtdH;

  edm::Handle<edm::ValueMap<float>> sigmat0H;
  ev.getByToken(sigmat0Token_, sigmat0H);
  const auto& sigmat0In = *sigmat0H;

  edm::Handle<edm::ValueMap<float>> sigmatmtdH;
  ev.getByToken(sigmatmtdToken_, sigmatmtdH);
  const auto& sigmatmtdIn = *sigmatmtdH;

  edm::Handle<edm::ValueMap<float>> pathLengthH;
  ev.getByToken(pathLengthToken_, pathLengthH);
  const auto& pathLengthIn = *pathLengthH;

  edm::Handle<edm::ValueMap<float>> pH;
  ev.getByToken(pToken_, pH);
  const auto& pIn = *pH;

  edm::Handle<reco::VertexCollection> vtxsH;
  ev.getByToken(vtxsToken_, vtxsH);
  const auto& vtxs = *vtxsH;

  //output value maps (PID probabilities and recalculated time at beamline)
  std::vector<float> t0OutRaw;
  std::vector<float> sigmat0OutRaw;
  std::vector<float> t0safeOutRaw;
  std::vector<float> sigmat0safeOutRaw;
  std::vector<float> probPiOutRaw;
  std::vector<float> probKOutRaw;
  std::vector<float> probPOutRaw;

  //Do work here
  for (unsigned int itrack = 0; itrack < tracks.size(); ++itrack) {
    const reco::Track& track = tracks[itrack];
    const reco::TrackRef trackref(tracksH, itrack);
    float t0 = t0In[trackref];
    float t0safe = t0;
    float sigmat0safe = sigmat0In[trackref];
    float sigmatmtd = (sigmatmtdIn[trackref] > 0. && fixedT0Error_ > 0.) ? fixedT0Error_ : sigmatmtdIn[trackref];
    float sigmat0 = sigmatmtd;

    float prob_pi = -1.;
    float prob_k = -1.;
    float prob_p = -1.;

    if (sigmat0 > 0.) {
      double rsigmazsq = 1. / track.dzError() / track.dzError();
      double rsigmat = 1. / sigmatmtd;

      //find associated vertex
      int vtxidx = -1;
      int vtxidxmindz = -1;
      int vtxidxminchisq = -1;
      double mindz = maxDz_;
      double minchisq = std::numeric_limits<double>::max();
      //first try based on association weights, but keep track of closest in z and z-t as well
      for (unsigned int ivtx = 0; ivtx < vtxs.size(); ++ivtx) {
        const reco::Vertex& vtx = vtxs[ivtx];
        float w = vtx.trackWeight(trackref);
        if (w > 0.5) {
          vtxidx = ivtx;
          break;
        }
        double dz = std::abs(track.dz(vtx.position()));
        if (dz < mindz) {
          mindz = dz;
          vtxidxmindz = ivtx;
        }
        if (vtx.tError() > 0. && vtx.tError() < vtxMaxSigmaT_) {
          double dt = std::abs(t0 - vtx.t());
          double dtsig = dt * rsigmat;
          double chisq = dz * dz * rsigmazsq + dtsig * dtsig;
          if (dz < maxDz_ && dtsig < maxDtSignificance_ && chisq < minchisq) {
            minchisq = chisq;
            vtxidxminchisq = ivtx;
          }
        }
      }

      //if no vertex found based on association weights, fall back to closest in z or z-t
      if (vtxidx < 0) {
        //if closest vertex in z does not have valid time information, just use it,
        //otherwise use the closest vertex in z-t plane with timing info, with a fallback to the closest in z
        if (vtxidxmindz >= 0 && !(vtxs[vtxidxmindz].tError() > 0. && vtxs[vtxidxmindz].tError() < vtxMaxSigmaT_)) {
          vtxidx = vtxidxmindz;
        } else if (vtxidxminchisq >= 0) {
          vtxidx = vtxidxminchisq;
        } else if (vtxidxmindz >= 0) {
          vtxidx = vtxidxmindz;
        }
      }

      //testing mass hypotheses only possible if there is an associated vertex with time information
      if (vtxidx >= 0 && vtxs[vtxidx].tError() > 0. && vtxs[vtxidx].tError() < vtxMaxSigmaT_) {
        //compute chisq in z-t plane for nominal vertex and mass hypothesis (pion)
        const reco::Vertex& vtxnom = vtxs[vtxidx];
        double dznom = std::abs(track.dz(vtxnom.position()));
        double dtnom = std::abs(t0 - vtxnom.t());
        double dtsignom = dtnom * rsigmat;
        double chisqnom = dznom * dznom * rsigmazsq + dtsignom * dtsignom;

        //recompute t0 for alternate mass hypotheses
        double t0_best = t0;

        //reliable match, revert to raw mtd time uncertainty
        if (dtsignom < maxDtSignificance_) {
          sigmat0safe = sigmatmtd;
        }

        double tmtd = tmtdIn[trackref];
        double pathlength = pathLengthIn[trackref];
        double magp = pIn[trackref];

        double gammasq_k = 1. + magp * magp / m_k / m_k;
        double beta_k = std::sqrt(1. - 1. / gammasq_k);
        double t0_k = tmtd - pathlength / beta_k * c_inv;

        double gammasq_p = 1. + magp * magp / m_p / m_p;
        double beta_p = std::sqrt(1. - 1. / gammasq_p);
        double t0_p = tmtd - pathlength / beta_p * c_inv;

        double chisqmin = chisqnom;

        double chisqmin_pi = chisqnom;
        double chisqmin_k = std::numeric_limits<double>::max();
        double chisqmin_p = std::numeric_limits<double>::max();
        //loop through vertices and check for better matches
        for (const reco::Vertex& vtx : vtxs) {
          if (!(vtx.tError() > 0. && vtx.tError() < vtxMaxSigmaT_)) {
            continue;
          }

          double dz = std::abs(track.dz(vtx.position()));
          if (dz >= maxDz_) {
            continue;
          }

          double chisqdz = dz * dz * rsigmazsq;

          double dt_k = std::abs(t0_k - vtx.t());
          double dtsig_k = dt_k * rsigmat;
          double chisq_k = chisqdz + dtsig_k * dtsig_k;

          if (dtsig_k < maxDtSignificance_ && chisq_k < chisqmin_k) {
            chisqmin_k = chisq_k;
          }

          double dt_p = std::abs(t0_p - vtx.t());
          double dtsig_p = dt_p * rsigmat;
          double chisq_p = chisqdz + dtsig_p * dtsig_p;

          if (dtsig_p < maxDtSignificance_ && chisq_p < chisqmin_p) {
            chisqmin_p = chisq_p;
          }

          if (dtsig_k < maxDtSignificance_ && chisq_k < chisqmin) {
            chisqmin = chisq_k;
            t0_best = t0_k;
            t0safe = t0_k;
            sigmat0safe = sigmatmtd;
          }
          if (dtsig_p < maxDtSignificance_ && chisq_p < chisqmin) {
            chisqmin = chisq_p;
            t0_best = t0_p;
            t0safe = t0_p;
            sigmat0safe = sigmatmtd;
          }
        }

        //compute PID probabilities
        //*TODO* deal with heavier nucleons and/or BSM case here?
        double rawprob_pi = exp(-0.5 * chisqmin_pi);
        double rawprob_k = exp(-0.5 * chisqmin_k);
        double rawprob_p = exp(-0.5 * chisqmin_p);

        double normprob = 1. / (rawprob_pi + rawprob_k + rawprob_p);

        prob_pi = rawprob_pi * normprob;
        prob_k = rawprob_k * normprob;
        prob_p = rawprob_p * normprob;

        double prob_heavy = 1. - prob_pi;

        if (prob_heavy > minProbHeavy_) {
          t0 = t0_best;
        }
      }
    }

    t0OutRaw.push_back(t0);
    sigmat0OutRaw.push_back(sigmat0);
    t0safeOutRaw.push_back(t0safe);
    sigmat0safeOutRaw.push_back(sigmat0safe);
    probPiOutRaw.push_back(prob_pi);
    probKOutRaw.push_back(prob_k);
    probPOutRaw.push_back(prob_p);
  }

  fillValueMap(ev, tracksH, t0OutRaw, t0Name);
  fillValueMap(ev, tracksH, sigmat0OutRaw, sigmat0Name);
  fillValueMap(ev, tracksH, t0safeOutRaw, t0safeName);
  fillValueMap(ev, tracksH, sigmat0safeOutRaw, sigmat0safeName);
  fillValueMap(ev, tracksH, probPiOutRaw, probPiName);
  fillValueMap(ev, tracksH, probKOutRaw, probKName);
  fillValueMap(ev, tracksH, probPOutRaw, probPName);
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(TOFPIDProducer);
