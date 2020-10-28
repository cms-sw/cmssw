// Original Author:  Emmanuelle Perez,40 1-A28,+41227671915,
//         Created:  Tue Nov 12 17:03:19 CET 2013
//Modified by Emily MacDonald, 30 Nov 2018

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TCorrelator/interface/TkEtMiss.h"
#include "DataFormats/L1TCorrelator/interface/TkEtMissFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkPrimaryVertex.h"

// detector geometry
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

using namespace l1t;

class L1TrackerEtMissProducer : public edm::EDProducer {
public:
  typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
  typedef std::vector< L1TTTrackType > L1TTTrackCollectionType;

  explicit L1TrackerEtMissProducer(const edm::ParameterSet&);
  ~L1TrackerEtMissProducer();

private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------
  float maxZ0;	// in cm
  float deltaZ;	// in cm
  float maxEta;
  float chi2dofMax;
  float bendChi2Max;
  float minPt; // in GeV
  int nStubsmin;
  int nPSStubsMin;  // minimum number of stubs in PS modules
  float maxPt;	    // in GeV
  int highPtTracks; // saturate or truncate
  bool displaced; //prompt/displaced tracks

  const edm::EDGetTokenT< TkPrimaryVertexCollection > pvToken;
  const edm::EDGetTokenT<std::vector<TTTrack< Ref_Phase2TrackerDigi_ > > > trackToken;
};

//constructor//
L1TrackerEtMissProducer::L1TrackerEtMissProducer(const edm::ParameterSet& iConfig) :
pvToken(consumes<TkPrimaryVertexCollection>(iConfig.getParameter<edm::InputTag>("L1VertexInputTag"))),
trackToken(consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag")))
{
  maxZ0 = (float)iConfig.getParameter<double>("maxZ0");
  deltaZ = (float)iConfig.getParameter<double>("deltaZ");
  chi2dofMax = (float)iConfig.getParameter<double>("chi2dofMax");
  bendChi2Max = (float)iConfig.getParameter<double>("bendChi2Max");
  minPt = (float)iConfig.getParameter<double>("minPt");
  nStubsmin = iConfig.getParameter<int>("nStubsmin");
  nPSStubsMin = iConfig.getParameter<int>("nPSStubsMin");
  maxPt = (float)iConfig.getParameter<double>("maxPt");
  maxEta = (float)iConfig.getParameter<double>("maxEta");
  highPtTracks = iConfig.getParameter<int>("highPtTracks");
  displaced = iConfig.getParameter<bool>("displaced");

  if (displaced) produces<TkEtMissCollection>("L1TrackerEtMissExtended");
  else produces<TkEtMissCollection>("L1TrackerEtMiss");
}

L1TrackerEtMissProducer::~L1TrackerEtMissProducer() { }

void L1TrackerEtMissProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::unique_ptr<TkEtMissCollection> METCollection(new TkEtMissCollection);

  // Tracker Topology
  edm::ESHandle<TrackerTopology> tTopoHandle_;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle_);
  const TrackerTopology* tTopo = tTopoHandle_.product();

  edm::Handle<TkPrimaryVertexCollection> L1VertexHandle;
  iEvent.getByToken(pvToken,L1VertexHandle);

  edm::Handle<L1TTTrackCollectionType> L1TTTrackHandle;
  iEvent.getByToken(trackToken, L1TTTrackHandle);
  L1TTTrackCollectionType::const_iterator trackIter;

  if( !L1VertexHandle.isValid() ) {
    LogError("L1TrackerEtMissProducer")
    << "\nWarning: TkPrimaryVertexCollection not found in the event. Exit\n";
    return;
  }

  if( !L1TTTrackHandle.isValid() ) {
    LogError("L1TrackerEtMissProducer")
    << "\nWarning: L1TTTrackCollection not found in the event. Exit\n";
    return;
  }

  float sumPx = 0;
  float sumPy = 0;
  float etTot = 0;
  double sumPx_PU = 0;
  double sumPy_PU = 0;
  double etTot_PU = 0;
  float zVTX = L1VertexHandle->begin()->zvertex();

  for (trackIter = L1TTTrackHandle->begin(); trackIter != L1TTTrackHandle->end(); ++trackIter) {
    float pt = trackIter->momentum().perp();
    float phi = trackIter->momentum().phi();
    float eta = trackIter->momentum().eta();
    float chi2dof = trackIter->chi2Red();
    float bendChi2 = trackIter->stubPtConsistency();
    float z0  = trackIter->z0();
    std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ > > >  theStubs = trackIter -> getStubRefs() ;
    int nstubs = (int) theStubs.size();

    if (pt < minPt) continue;
    if (fabs(z0) > maxZ0) continue;
    if (fabs(eta) > maxEta) continue;
    if (chi2dof > chi2dofMax) continue;
    if (bendChi2 > bendChi2Max) continue;

    if (maxPt > 0 && pt > maxPt) {
      if (highPtTracks == 0)  continue;	// ignore these very high PT tracks: truncate
      if (highPtTracks == 1)  pt = maxPt; // saturate
    }

    int nPS = 0; // number of stubs in PS modules
    // loop over the stubs
    for (unsigned int istub=0; istub<(unsigned int)theStubs.size(); istub++) {
      DetId detId( theStubs.at(istub)->getDetId() );
      if (detId.det() == DetId::Detector::Tracker) {
        if ( (detId.subdetId() == StripSubdetector::TOB && tTopo->tobLayer(detId) <= 3) || (detId.subdetId() == StripSubdetector::TID && tTopo->tidRing(detId) <= 9) ) nPS++;
      }
    }

    if (nstubs < nStubsmin) continue;
    if (nPS < nPSStubsMin) continue;

    if (!displaced){ // if displaced, deltaZ = 3.0 cm, very loose
      // construct deltaZ cut to be based on track eta
      if      ( fabs(eta)>=0   &&  fabs(eta)<0.7)  deltaZ = 0.4;
      else if ( fabs(eta)>=0.7 &&  fabs(eta)<1.0)  deltaZ = 0.6;
      else if ( fabs(eta)>=1.0 &&  fabs(eta)<1.2)  deltaZ = 0.76;
      else if ( fabs(eta)>=1.2 &&  fabs(eta)<1.6)  deltaZ = 1.0;
      else if ( fabs(eta)>=1.6 &&  fabs(eta)<2.0)  deltaZ = 1.7;
      else if ( fabs(eta)>=2.0 &&  fabs(eta)<=2.4) deltaZ = 2.2;
    }

    if ( fabs(z0 - zVTX) <= deltaZ) {
      sumPx += pt*cos(phi);
      sumPy += pt*sin(phi);
      etTot += pt;
    }
    else {	// PU sums
      sumPx_PU += pt*cos(phi);
      sumPy_PU += pt*sin(phi);
      etTot_PU += pt;
    }
  } // end loop over tracks

  float et = sqrt(sumPx*sumPx + sumPy*sumPy);
  double etmiss_PU = sqrt(sumPx_PU*sumPx_PU + sumPy_PU*sumPy_PU);

  math::XYZTLorentzVector missingEt( -sumPx, -sumPy, 0, et);
  int ibx = 0;
  METCollection->push_back( TkEtMiss( missingEt,
    TkEtMiss::kMET,
    etTot,
    etmiss_PU,
    etTot_PU,
    ibx )
  );

  if (displaced) iEvent.put( std::move(METCollection), "L1TrackerEtMissExtended");
  else iEvent.put( std::move(METCollection), "L1TrackerEtMiss");
} // end producer

void L1TrackerEtMissProducer::beginJob() { }

void L1TrackerEtMissProducer::endJob() { }

DEFINE_FWK_MODULE(L1TrackerEtMissProducer);
