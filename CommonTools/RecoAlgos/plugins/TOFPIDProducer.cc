#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

using namespace std;
using namespace edm;





class TOFPIDProducer : public edm::stream::EDProducer<> {  
  static constexpr char t0Name[] = "t0";
  static constexpr char sigmat0Name[] = "sigmat0";
  static constexpr char probPiName[] = "probPi";
  static constexpr char probKName[] = "probK";
  static constexpr char probPName[] = "probP";
 public:
  
  TOFPIDProducer(const ParameterSet& pset); 

  void produce(edm::Event& ev, const edm::EventSetup& es) final;

 private:
  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  edm::EDGetTokenT<edm::ValueMap<float> > t0Token_;
  edm::EDGetTokenT<edm::ValueMap<float> > tmtdToken_;
  edm::EDGetTokenT<edm::ValueMap<float> > sigmatToken_;
  edm::EDGetTokenT<edm::ValueMap<float> > pathLengthToken_;
  edm::EDGetTokenT<edm::ValueMap<float> > pToken_;
  edm::EDGetTokenT<reco::VertexCollection> vtxsToken_;
  edm::EDGetTokenT<reco::BeamSpot> bsToken_;
};


  
TOFPIDProducer::TOFPIDProducer(const ParameterSet& iConfig) :
  tracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracksSrc"))),
  t0Token_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("t0Src"))),
  tmtdToken_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("tmtdSrc"))),
  sigmatToken_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("sigmatSrc"))),
  pathLengthToken_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("pathLengthSrc"))),
  pToken_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("pSrc"))),
  vtxsToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vtxsSrc"))),
  bsToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotSrc"))) {
  constexpr float maxChi2=25.;
  constexpr float nSigma=3.;
  
  produces<edm::ValueMap<float> >(t0Name);
  produces<edm::ValueMap<float> >(sigmat0Name);
  produces<edm::ValueMap<float> >(probPiName); 
  produces<edm::ValueMap<float> >(probKName);
  produces<edm::ValueMap<float> >(probPName);
}


void TOFPIDProducer::produce( edm::Event& ev,
						      const edm::EventSetup& es ) {  
  
  constexpr double m_k = 0.493677; //[GeV]
  constexpr double m_p = 0.9382720813; //[GeV]
  constexpr double c = 2.99792458e1; //[cm/ns]
  
  edm::Handle<reco::TrackCollection> tracksH;  
  ev.getByToken(tracksToken_,tracksH);
  const auto& tracks = *tracksH;

  edm::Handle<edm::ValueMap<float> > t0H;
  ev.getByToken(t0Token_, t0H);
  const auto &t0In = *t0H;

  edm::Handle<edm::ValueMap<float> > tmtdH;
  ev.getByToken(tmtdToken_, tmtdH);
  const auto &tmtdIn = *tmtdH;
  
  edm::Handle<edm::ValueMap<float> > sigmatH;
  ev.getByToken(sigmatToken_, sigmatH);
  const auto &sigmatIn = *sigmatH;
  
  edm::Handle<edm::ValueMap<float> > pathLengthH;
  ev.getByToken(pathLengthToken_, pathLengthH);
  const auto &pathLengthIn = *pathLengthH;
  
  edm::Handle<edm::ValueMap<float> > pH;
  ev.getByToken(pToken_, pH);
  const auto &pIn = *pH;
  
  edm::Handle<reco::VertexCollection> vtxsH;  
  ev.getByToken(vtxsToken_,vtxsH);
  const auto& vtxs = *vtxsH;
  
  edm::Handle<reco::BeamSpot> bsH;  
  ev.getByToken(bsToken_,bsH);
  const auto& bs = *bsH;

  //output value maps (PID probabilities and recalculated time at beamline)
  auto t0Out = std::make_unique<edm::ValueMap<float>>();
  std::vector<float> t0OutRaw;
  
  auto sigmat0Out = std::make_unique<edm::ValueMap<float>>();
  std::vector<float> sigmat0OutRaw;

  auto probPiOut = std::make_unique<edm::ValueMap<float>>();
  std::vector<float> probPiOutRaw;
  
  auto probKOut = std::make_unique<edm::ValueMap<float>>();
  std::vector<float> probKOutRaw;
  
  auto probPOut = std::make_unique<edm::ValueMap<float>>();
  std::vector<float> probPOutRaw;
  
  //Do work here
  for (unsigned int itrack = 0; itrack<tracks.size(); ++itrack) {
    const reco::Track &track = tracks[itrack];
    const reco::TrackRef trackref(tracksH,itrack);
    float t0 = t0In[trackref];
    float sigmat0 = sigmatIn[trackref];
    
    float prob_pi = -1.;
    float prob_k = -1.;
    float prob_p = -1.;
    
    if (sigmat0>0.) {
      
      double rsigmazsq = 1./track.dzError()/track.dzError();
      double rsigmat0sq = 1./sigmat0/sigmat0;
      
      //find associated vertex
      int vtxidx = -1;
      int vtxidxmindz = -1;
      int vtxidxminchisq = -1;
      double mindz = std::numeric_limits<double>::max();
      double minchisq = std::numeric_limits<double>::max();
      //first try based on association weights, but keep track of closest in z and z-t as well
      for (unsigned int ivtx = 0; ivtx<vtxs.size(); ++ivtx) {
        const reco::Vertex &vtx = vtxs[ivtx];
        float w = vtx.trackWeight(trackref);
        if (w>0.5) {
          vtxidx = ivtx;
          break;
        }
        double dz = std::abs(track.dz(vtx.position()));
        if (dz<mindz) {
          mindz = dz;
          vtxidxmindz = ivtx;
        }
        if (vtx.tError()>0. && vtx.tError()<0.5*sqrt(2.)*sigmat0) {
          double dt = std::abs(t0-vtx.t());
          double chisq = dz*dz*rsigmazsq + dt*dt*rsigmat0sq;
          if (chisq<minchisq) {
            minchisq = chisq;
            vtxidxminchisq = ivtx;
          }
        }
      }
      
      //if no vertex found based on association weights, fall back to closest in z or z-t
      if (vtxidx<0) {
        //if closest vertex in z has time information, use the closest vertex in z-t plane with timing info,
        //otherwise just use the closest in z
        if (vtxs[vtxidxmindz].tError()>0. && vtxs[vtxidxmindz].tError()<0.5*sqrt(2.)*sigmat0) {
          vtxidx = vtxidxminchisq;
        }
        else {
          vtxidx = vtxidxmindz;
        }
      }
      
      //testing mass hypotheses only possible if there is an associated vertex with time information
      if (vtxidx>=0 && vtxs[vtxidx].tError()>0. && vtxs[vtxidx].tError()<0.5*sqrt(2.)*sigmat0) {        
        //compute chisq in z-t plane for nominal vertex and mass hypothesis (pion)
        const reco::Vertex &vtxnom = vtxs[vtxidx];
        double dznom = std::abs(track.dz(vtxnom.position()));
        double dtnom = std::abs(t0 - vtxnom.t());
        double chisqnom = dznom*dznom*rsigmazsq + dtnom*dtnom*rsigmat0sq;
        
        //recompute t0 for alternate mass hypotheses
        float tmtd = tmtdIn[trackref];
        float pathlength = pathLengthIn[trackref];
        float magp = pIn[trackref];
        
        double gammasq_k = 1. + magp*magp/m_k/m_k;
        double beta_k = std::sqrt(1.-1./gammasq_k);
        double t0_k = tmtd - pathlength/beta_k/c;
        
        double gammasq_p = 1. + magp*magp/m_p/m_p;
        double beta_p = std::sqrt(1.-1./gammasq_p);
        double t0_p = tmtd - pathlength/beta_p/c;
        
        double chisqmin = chisqnom;
        
        double chisqmin_pi = chisqnom;
        double chisqmin_k = std::numeric_limits<double>::max();
        double chisqmin_p = std::numeric_limits<double>::max();
        //loop through vertices and check for better matches
        for (const reco::Vertex &vtx : vtxs) {
          if (!(vtx.tError()>0. && vtx.tError()<0.5*sqrt(2.)*sigmat0)) {
            continue;
          }
          
          double dz = std::abs(track.dz(vtx.position()));
          double chisqdz = dz*dz*rsigmazsq;
          
          double dt_k = std::abs(t0_k - vtx.t());
          double chisq_k = chisqdz + dt_k*dt_k*rsigmat0sq;
          
          if (chisq_k<chisqmin_k) {
            chisqmin_k = chisq_k;
          }
          
          double dt_p = std::abs(t0_p - vtx.t());
          double chisq_p = chisqdz + dt_p*dt_p*rsigmat0sq;
          
          if (chisq_p<chisqmin_p) {
            chisqmin_p = chisq_p;
          }
          
          if (chisq_k<chisqmin) {
            chisqmin = chisq_k;
            t0 = t0_k;
          }
          if (chisq_p<chisqmin) {
            chisqmin = chisq_p;
            t0 = t0_p;
          }
          
        }
        
        //compute PID probabilities
        //*TODO* deal with heavier nucleons and/or BSM case here?
        double rawprob_pi = exp(-0.5*chisqmin_pi);
        double rawprob_k = exp(-0.5*chisqmin_k);
        double rawprob_p = exp(-0.5*chisqmin_p);
        
        double normprob = 1./(rawprob_pi + rawprob_k + rawprob_p);
        
        prob_pi = rawprob_pi*normprob;
        prob_k = rawprob_k*normprob;
        prob_p = rawprob_p*normprob;
      }
      
    }
    
    t0OutRaw.push_back(t0);
    sigmat0OutRaw.push_back(sigmat0);
    probPiOutRaw.push_back(prob_pi);
    probKOutRaw.push_back(prob_k);
    probPOutRaw.push_back(prob_p);
  }

  edm::ValueMap<float>::Filler fillert0s(*t0Out);
  fillert0s.insert(tracksH,t0OutRaw.cbegin(),t0OutRaw.cend());
  fillert0s.fill();
  ev.put(std::move(t0Out),t0Name);

  edm::ValueMap<float>::Filler fillersigmat0s(*sigmat0Out);
  fillersigmat0s.insert(tracksH,sigmat0OutRaw.cbegin(),sigmat0OutRaw.cend());
  fillersigmat0s.fill();
  ev.put(std::move(sigmat0Out),sigmat0Name);
  
  edm::ValueMap<float>::Filler fillerprobPis(*probPiOut);
  fillerprobPis.insert(tracksH,probPiOutRaw.cbegin(),probPiOutRaw.cend());
  fillerprobPis.fill();
  ev.put(std::move(probPiOut),probPiName);  
  
  edm::ValueMap<float>::Filler fillerprobKs(*probKOut);
  fillerprobKs.insert(tracksH,probKOutRaw.cbegin(),probKOutRaw.cend());
  fillerprobKs.fill();
  ev.put(std::move(probKOut),probKName);

  edm::ValueMap<float>::Filler fillerprobPs(*probPOut);
  fillerprobPs.insert(tracksH,probPOutRaw.cbegin(),probPOutRaw.cend());
  fillerprobPs.fill();
  ev.put(std::move(probPOut),probPName);  
  
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(TOFPIDProducer);
