///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Producer of L1TkJetParticle,                                          //
// Cluster L1 tracks using fastjet                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "L1Trigger/TrackTrigger/interface/StubPtConsistency.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkPrimaryVertex.h"
#include "DataFormats/Math/interface/LorentzVector.h"

// L1 tracks
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

// geometry
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "RecoJets/JetProducers/plugins/VirtualJetProducer.h"
#include <string>
#include "TMath.h"
#include "TH1.h"

using namespace l1t;
using namespace edm;
using namespace std;


//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

class L1TrackerJetProducer : public edm::EDProducer
{
public:

  typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
  typedef std::vector< L1TTTrackType > L1TTTrackCollectionType;

  explicit L1TrackerJetProducer(const edm::ParameterSet&);
  ~L1TrackerJetProducer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  //virtual void beginRun(edm::Run&, edm::EventSetup const&);
  //virtual void endRun(edm::Run&, edm::EventSetup const&);
  //virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
  //virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

  // member data

  // track selection criteria
  float TRK_ZMAX;       // [cm]
  float TRK_CHI2MAX;    // maximum track chi2
  float TRK_PTMIN;      // [GeV]
  float TRK_ETAMAX;     // [rad]
  unsigned int   TRK_NSTUBMIN;   // minimum number of stubs
  int   TRK_NSTUBPSMIN; // minimum number of stubs in PS modules
  int L1Tk_nPar;
  double DeltaZ0Cut;    // save with |L1z-z0| < maxZ0
  double CONESize;      // Use anti-kt with this cone size
  bool doPtComp;
  bool doTightChi2;
  double BendConsistency;
  //need PVtx here
  const edm::EDGetTokenT<std::vector<TTTrack< Ref_Phase2TrackerDigi_ > > > trackToken;
  edm::EDGetTokenT<L1TkPrimaryVertexCollection>PVertexToken;
};

//////////////
// constructor
L1TrackerJetProducer::L1TrackerJetProducer(const edm::ParameterSet& iConfig) :
trackToken(consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag"))),
PVertexToken(consumes<L1TkPrimaryVertexCollection>(iConfig.getParameter<edm::InputTag>("L1PrimaryVertexTag")))
{

  produces<L1TkJetParticleCollection>("L1TrackerJets");
  L1Tk_nPar   =(int)iConfig.getParameter<int>("L1Tk_nPar");
  TRK_ZMAX    = (float)iConfig.getParameter<double>("TRK_ZMAX");
  TRK_CHI2MAX = (float)iConfig.getParameter<double>("TRK_CHI2MAX");
  TRK_PTMIN   = (float)iConfig.getParameter<double>("TRK_PTMIN");
  TRK_ETAMAX  = (float)iConfig.getParameter<double>("TRK_ETAMAX");
  TRK_NSTUBMIN   = (unsigned int)iConfig.getParameter<int>("TRK_NSTUBMIN");
  TRK_NSTUBPSMIN = (int)iConfig.getParameter<int>("TRK_NSTUBPSMIN");
  DeltaZ0Cut =(float)iConfig.getParameter<double>("DeltaZ0Cut");
  CONESize =(float)iConfig.getParameter<double>("CONESize");
  doPtComp     = iConfig.getParameter<bool>("doPtComp");
  doTightChi2 = iConfig.getParameter<bool>("doTightChi2");
  BendConsistency=iConfig.getParameter<double>("BendConsistency");
}

/////////////
// destructor
L1TrackerJetProducer::~L1TrackerJetProducer() {
}

///////////
// producer
void L1TrackerJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // ----------------------------------------------------------------------------------------------
  // output container
  // ----------------------------------------------------------------------------------------------

  std::unique_ptr<L1TkJetParticleCollection> L1TrackerJets(new L1TkJetParticleCollection);


  // ----------------------------------------------------------------------------------------------
  // retrieve input containers
  // ----------------------------------------------------------------------------------------------
  // L1 tracks
  edm::Handle< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > > TTTrackHandle;
  iEvent.getByToken(trackToken, TTTrackHandle);
  std::vector< TTTrack< Ref_Phase2TrackerDigi_ > >::const_iterator iterL1Track;

  // Tracker Topology
  edm::ESHandle<TrackerTopology> tTopoHandle_;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle_);
  const TrackerTopology* tTopo = tTopoHandle_.product();
  ESHandle<TrackerGeometry> tGeomHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(tGeomHandle);

   
  edm::ESHandle<MagneticField> magneticFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);  
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  double mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0,0,0)).z();
  const TrackerGeometry* const theTrackerGeom = tGeomHandle.product();
  
  edm::Handle<L1TkPrimaryVertexCollection >L1TkPrimaryVertexHandle;
  iEvent.getByToken(PVertexToken, L1TkPrimaryVertexHandle);
  fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, CONESize);
  std::vector<fastjet::PseudoJet>  JetInputs;

  unsigned int this_l1track = 0;
  for ( iterL1Track = TTTrackHandle->begin(); iterL1Track != TTTrackHandle->end(); iterL1Track++ ) {
    ++this_l1track;
    if(fabs(iterL1Track->getPOCA(L1Tk_nPar).z())>TRK_ZMAX)continue;
    if(fabs(iterL1Track->getMomentum(L1Tk_nPar).eta())>TRK_ETAMAX)continue;
    if(iterL1Track->getMomentum(L1Tk_nPar).perp()<TRK_PTMIN)continue;
    if(iterL1Track->getChi2()>TRK_CHI2MAX)continue;
    std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ > > >  theStubs = iterL1Track -> getStubRefs() ;	    if(theStubs.size()<TRK_NSTUBMIN)continue;
    float chi2ndof=(iterL1Track->getChi2()/(2*theStubs.size() - L1Tk_nPar));
    float trk_bstubPt=StubPtConsistency::getConsistency(TTTrackHandle->at(this_l1track-1), theTrackerGeom, tTopo,mMagneticFieldStrength,4);//trkPtr->getStubPtConsis
    if(trk_bstubPt>BendConsistency)continue;
    if(doTightChi2 && (iterL1Track->getMomentum(L1Tk_nPar).perp()>20 && chi2ndof>5))continue;
    int tmp_trk_nstubPS = 0;
    for (unsigned int istub=0; istub<(unsigned int)theStubs.size(); istub++) {
      DetId detId( theStubs.at(istub)->getDetId() );
      bool tmp_isPS = false;
      if (detId.det() == DetId::Detector::Tracker) {
        if (detId.subdetId() == StripSubdetector::TOB && tTopo->tobLayer(detId) <= 3)     tmp_isPS = true;
        else if (detId.subdetId() == StripSubdetector::TID && tTopo->tidRing(detId) <= 9) tmp_isPS = true;
      }
      if (tmp_isPS) tmp_trk_nstubPS++;
    }
    if(tmp_trk_nstubPS<TRK_NSTUBPSMIN)continue;
    double DeltaZtoVtx=fabs(L1TkPrimaryVertexHandle->begin()->getZvertex()-iterL1Track->getPOCA(L1Tk_nPar).z());
    if(DeltaZtoVtx>DeltaZ0Cut)continue;

    fastjet::PseudoJet psuedoJet(iterL1Track->getMomentum().x(), iterL1Track->getMomentum().y(), iterL1Track->getMomentum().z(), iterL1Track->getMomentum().mag());
    JetInputs.push_back(psuedoJet);	//input tracks for clustering
    JetInputs.back().set_user_index(this_l1track-1);//save track index in the collection
  }
  fastjet::ClusterSequence cs(JetInputs,jet_def);//define the output jet collection
  std::vector<fastjet::PseudoJet> JetOutputs=fastjet::sorted_by_pt(cs.inclusive_jets(0));//Output Jet ollection pTOrdered

  for (unsigned int ijet=0;ijet<JetOutputs.size();++ijet) {
    math::XYZTLorentzVector jetP4(JetOutputs[ijet].px(),JetOutputs[ijet].py(),JetOutputs[ijet].pz(),JetOutputs[ijet].modp());
    float sumpt=0;
    float avgZ=0;
    std::vector< edm::Ptr< L1TTTrackType > > L1TrackPtrs;
    std::vector<fastjet::PseudoJet> fjConstituents =fastjet::sorted_by_pt(cs.constituents(JetOutputs[ijet]));

    for(unsigned int i=0; i<fjConstituents.size(); ++i){
      auto index =fjConstituents[i].user_index();
      edm::Ptr< L1TTTrackType > trkPtr(TTTrackHandle, index) ;
      L1TrackPtrs.push_back(trkPtr); //L1Tracks in the jet
      sumpt=sumpt+trkPtr->getMomentum().perp();
      avgZ=avgZ+trkPtr->getMomentum().perp()*trkPtr->getPOCA(L1Tk_nPar).z();
    }
    avgZ=avgZ/sumpt;
    edm::Ref< JetBxCollection > jetRef ;
    L1TkJetParticle trkJet(jetP4, jetRef, L1TrackPtrs, avgZ);
    L1TrackerJets->push_back(trkJet);

  }


  iEvent.put( std::move(L1TrackerJets), "L1TrackerJets");

}


// ------------ method called once each job just before starting event loop  ------------
void
L1TrackerJetProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TrackerJetProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
L1TrackerJetProducer::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
L1TrackerJetProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TrackerJetProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TrackerJetProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TrackerJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TrackerJetProducer);
