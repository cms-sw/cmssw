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
#include "DataFormats/L1TrackTrigger/interface/L1TkEtMissParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEtMissParticleFwd.h"
#include "DataFormats/L1TVertex/interface/Vertex.h"

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
  float maxZ0;	    // in cm
  float DeltaZ;	    // in cm
  float maxEta;
  float chi2dofMax;
  float bendchi2Max;
  float minPt;	    // in GeV
  int nStubsmin;
  int nStubsPSmin;  // minimum number of stubs in PS modules
  float maxPt;	    // in GeV
  int HighPtTracks; // saturate or truncate

  const edm::EDGetTokenT< VertexCollection > pvToken;
  const edm::EDGetTokenT<std::vector<TTTrack< Ref_Phase2TrackerDigi_ > > > trackToken;
};

///////////////
//constructor//
///////////////
L1TrackerEtMissProducer::L1TrackerEtMissProducer(const edm::ParameterSet& iConfig) :
pvToken(consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("L1VertexInputTag"))),
trackToken(consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag")))
{
  maxZ0 = (float)iConfig.getParameter<double>("maxZ0");
  DeltaZ = (float)iConfig.getParameter<double>("DeltaZ");
  chi2dofMax = (float)iConfig.getParameter<double>("chi2dofMax");
  bendchi2Max = (float)iConfig.getParameter<double>("bendchi2Max");
  minPt = (float)iConfig.getParameter<double>("minPt");
  nStubsmin = iConfig.getParameter<int>("nStubsmin");
  nStubsPSmin = iConfig.getParameter<int>("nStubsPSmin");
  maxPt = (float)iConfig.getParameter<double>("maxPt");
  maxEta = (float)iConfig.getParameter<double>("maxEta");
  HighPtTracks = iConfig.getParameter<int>("HighPtTracks");

  produces<L1TkEtMissParticleCollection>("trkMET");
}

//////////////
//destructor//
//////////////
L1TrackerEtMissProducer::~L1TrackerEtMissProducer() {
}

////////////
//producer//
////////////
void L1TrackerEtMissProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  // Tracker Topology
  edm::ESHandle<TrackerTopology> tTopoHandle_;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle_);
  const TrackerTopology* tTopo = tTopoHandle_.product();

  std::unique_ptr<L1TkEtMissParticleCollection> METCollection(new L1TkEtMissParticleCollection);

  edm::Handle<VertexCollection> L1VertexHandle;
  iEvent.getByToken(pvToken,L1VertexHandle);

  edm::Handle<L1TTTrackCollectionType> L1TTTrackHandle;
  iEvent.getByToken(trackToken, L1TTTrackHandle);
  L1TTTrackCollectionType::const_iterator trackIter;

  if( !L1VertexHandle.isValid() ) {
    LogError("L1TrackerEtMissProducer")<< "\nWarning: VertexCollection not found in the event. Exit"<< std::endl;
    return;
  }

  if( !L1TTTrackHandle.isValid() ) {
    LogError("L1TrackerEtMissProducer")<< "\nWarning: L1TTTrackCollection not found in the event. Exit"<< std::endl;
    return;
  }


  float sumPx = 0;
  float sumPy = 0;
  float etTot = 0;
  double sumPx_PU = 0;
  double sumPy_PU = 0;
  double etTot_PU = 0;

  float zVTX = L1VertexHandle->begin()->z0();

  for (trackIter = L1TTTrackHandle->begin(); trackIter != L1TTTrackHandle->end(); ++trackIter) {
    float pt = trackIter->getMomentum().perp();
    float phi = trackIter->getMomentum().phi();
    float eta = trackIter->getMomentum().eta();
    std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ > > >  theStubs = trackIter -> getStubRefs() ;
    int nstubs = (int) theStubs.size();

    float chi2 = trackIter->getChi2();
    float chi2dof = chi2 / (2*nstubs-4);
    float bendchi2 = trackIter->getStubPtConsistency();
    float z0  = trackIter->getPOCA().z();

    if (pt < minPt) continue;
    if (fabs(z0) > maxZ0) continue;
    if (fabs(eta) > maxEta) continue;
    if (chi2dof > chi2dofMax) continue;
    if (bendchi2 > bendchi2Max) continue;

    if ( maxPt > 0 && pt > maxPt)  {
      if (HighPtTracks == 0)  continue;	// ignore these very high PT tracks: truncate
      if (HighPtTracks == 1)  pt = maxPt; // saturate
    }

    int nPS = 0.;     // number of stubs in PS modules
    // loop over the stubs
    for (unsigned int istub=0; istub<(unsigned int)theStubs.size(); istub++) {
      DetId detId( theStubs.at(istub)->getDetId() );
      if (detId.det() == DetId::Detector::Tracker) {
        if ( (detId.subdetId() == StripSubdetector::TOB && tTopo->tobLayer(detId) <= 3) || (detId.subdetId() == StripSubdetector::TID && tTopo->tidRing(detId) <= 9) ) nPS++;
      }
    }

    if (nstubs < nStubsmin) continue;
    if (nPS < nStubsPSmin) continue;

    // construct deltaZ cut to be based on track eta
    if      ( fabs(eta)>=0   &&  fabs(eta)<0.7)  DeltaZ = 0.4;
    else if ( fabs(eta)>=0.7 &&  fabs(eta)<1.0)  DeltaZ = 0.6;
    else if ( fabs(eta)>=1.0 &&  fabs(eta)<1.2)  DeltaZ = 0.76;
    else if ( fabs(eta)>=1.2 &&  fabs(eta)<1.6)  DeltaZ = 1.0;
    else if ( fabs(eta)>=1.6 &&  fabs(eta)<2.0)  DeltaZ = 1.7;
    else if ( fabs(eta)>=2.0 &&  fabs(eta)<=2.4) DeltaZ = 2.2;

    if ( fabs(z0 - zVTX) <= DeltaZ) {
      sumPx += pt*cos(phi);
      sumPy += pt*sin(phi);
      etTot += pt ;
    }
    else {	// PU sums
      sumPx_PU += pt*cos(phi);
      sumPy_PU += pt*sin(phi);
      etTot_PU += pt ;
    }
  } // end loop over tracks

  float et = sqrt( sumPx*sumPx + sumPy*sumPy );
  double etmiss_PU = sqrt( sumPx_PU*sumPx_PU + sumPy_PU*sumPy_PU );

  math::XYZTLorentzVector missingEt( -sumPx, -sumPy, 0, et);

  int ibx = 0;
  METCollection->push_back( L1TkEtMissParticle( missingEt,
    L1TkEtMissParticle::kMET,
    etTot,
    etmiss_PU,
    etTot_PU,
    ibx ) );

  iEvent.put( std::move(METCollection), "trkMET");
} // end producer

void L1TrackerEtMissProducer::beginJob() {
}

void L1TrackerEtMissProducer::endJob() {
}

DEFINE_FWK_MODULE(L1TrackerEtMissProducer);
