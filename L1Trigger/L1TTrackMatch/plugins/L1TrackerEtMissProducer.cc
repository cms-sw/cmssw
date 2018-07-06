// Original Author:  Emmanuelle Perez,40 1-A28,+41227671915,
//         Created:  Tue Nov 12 17:03:19 CET 2013

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
    float chi2Max;
    float minPt;	    // in GeV
    int nStubsmin;
    int nStubsPSmin;  // minimum number of stubs in PS modules
    float maxPt;	    // in GeV
    int HighPtTracks; // saturate or truncate
    bool doPtComp;
    bool doTightChi2;
    const edm::EDGetTokenT< L1TkPrimaryVertexCollection > pvToken;
    const edm::EDGetTokenT<std::vector<TTTrack< Ref_Phase2TrackerDigi_ > > > trackToken;
};

///////////////
//constructor//
///////////////
L1TrackerEtMissProducer::L1TrackerEtMissProducer(const edm::ParameterSet& iConfig) :
  pvToken(consumes<L1TkPrimaryVertexCollection>(iConfig.getParameter<edm::InputTag>("L1VertexInputTag"))),
  trackToken(consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag")))
{
  maxZ0 = (float)iConfig.getParameter<double>("maxZ0");
  DeltaZ = (float)iConfig.getParameter<double>("DeltaZ");
  chi2Max = (float)iConfig.getParameter<double>("chi2Max");
  minPt = (float)iConfig.getParameter<double>("minPt");
  nStubsmin = iConfig.getParameter<int>("nStubsmin");
  nStubsPSmin = iConfig.getParameter<int>("nStubsPSmin");
  maxPt = (float)iConfig.getParameter<double>("maxPt");
  HighPtTracks = iConfig.getParameter<int>("HighPtTracks");
  doPtComp     = iConfig.getParameter<bool>("doPtComp");
  doTightChi2 = iConfig.getParameter<bool>("doTightChi2");

  produces<L1TkEtMissParticleCollection>("MET");
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

  edm::Handle<L1TkPrimaryVertexCollection> L1VertexHandle;
  iEvent.getByToken(pvToken,L1VertexHandle);
  std::vector<L1TkPrimaryVertex>::const_iterator vtxIter;

  edm::Handle<L1TTTrackCollectionType> L1TTTrackHandle;
  iEvent.getByToken(trackToken, L1TTTrackHandle);
  L1TTTrackCollectionType::const_iterator trackIter;

  if( !L1VertexHandle.isValid() ) {
    LogError("L1TrackerEtMissProducer")<< "\nWarning: L1TkPrimaryVertexCollection not found in the event. Exit"<< std::endl;
    return;
  }

  if( !L1TTTrackHandle.isValid() ) {
    LogError("L1TrackerEtMissProducer")<< "\nWarning: L1TTTrackCollectionType not found in the event. Exit"<< std::endl;
    return;
  }

  int ivtx = 0;
  for (vtxIter = L1VertexHandle->begin(); vtxIter != L1VertexHandle->end(); ++vtxIter) {
    float zVTX = vtxIter -> getZvertex();
    edm::Ref< L1TkPrimaryVertexCollection > vtxRef( L1VertexHandle, ivtx );
    ivtx ++;

    float sumPx = 0;
    float sumPy = 0;
    float etTot = 0;
    double sumPx_PU = 0;
    double sumPy_PU = 0;
    double etTot_PU = 0;

    for (trackIter = L1TTTrackHandle->begin(); trackIter != L1TTTrackHandle->end(); ++trackIter) {
      float pt = trackIter->getMomentum().perp();
      float eta = trackIter->getMomentum().eta();
      float chi2 = trackIter->getChi2();
      float ztr  = trackIter->getPOCA().z();

      if (pt < minPt) continue;
      if (fabs(ztr) > maxZ0 ) continue;
      if (chi2 > chi2Max) continue;

      float pt_rescale = 1;
      if ( maxPt > 0 && pt > maxPt)  {
        if (HighPtTracks == 0)  continue;	// ignore these very high PT tracks.
        if (HighPtTracks == 1)  {
          pt_rescale = maxPt / pt;	// will be used to rescale px and py
          pt = maxPt;     // saturate
        }
      }

      int nstubs = 0;
      float nPS = 0.;     // number of stubs in PS modules
      std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ > > >  theStubs = trackIter -> getStubRefs() ;

      int tmp_trk_nstub = (int) theStubs.size();
      if ( tmp_trk_nstub < 0) continue;
      // loop over the stubs
      for (unsigned int istub=0; istub<(unsigned int)theStubs.size(); istub++) {
        nstubs ++;
        DetId detId( theStubs.at(istub)->getDetId() );
        bool isPS = false;
        if (detId.det() == DetId::Detector::Tracker) {
          if (detId.subdetId() == StripSubdetector::TOB && tTopo->tobLayer(detId) <= 3)        isPS = true;
          else if (detId.subdetId() == StripSubdetector::TID && tTopo->tidRing(detId) <= 9)  isPS = true;
        }
        if (isPS) nPS ++;
      }
      if (nstubs < nStubsmin) continue;
      if (nPS < nStubsPSmin) continue;

      float trk_consistency = trackIter ->getStubPtConsistency();
      float chi2dof = chi2 / (2*nstubs-4);

      if(doPtComp && nstubs==4) {
        if (fabs(eta)<2.2 && trk_consistency>10) continue;
        else if (fabs(eta)>2.2 && chi2dof>5.0) continue;
      }

      if(doTightChi2) {
        if(pt>10.0 && chi2dof>5.0 ) continue;
      }

      if ( fabs(ztr - zVTX) <= DeltaZ) {   // eg DeltaZ = 1 cm
        sumPx += trackIter->getMomentum().x() * pt_rescale ;
        sumPy += trackIter->getMomentum().y() * pt_rescale ;
        etTot += pt ;
      }
      else {	// PU sums
        sumPx_PU += trackIter->getMomentum().x() * pt_rescale ;
        sumPy_PU += trackIter->getMomentum().y() * pt_rescale ;
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
      vtxRef,
      ibx ) );
  } // end loop over vertices

  iEvent.put( std::move(METCollection), "MET");
} // end producer

// ------------ method called once each job just before starting event loop  ------------
void L1TrackerEtMissProducer::beginJob() {
}

// ------------ method called once each job just after ending the event loop  ------------
void L1TrackerEtMissProducer::endJob() {
}

DEFINE_FWK_MODULE(L1TrackerEtMissProducer);
