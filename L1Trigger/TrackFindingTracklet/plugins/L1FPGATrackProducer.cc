//////////////////////////
//  Producer by Anders  //
//     and Emmanuele    //
//    july 2012 @ CU    //
//////////////////////////


#ifndef L1TFPGATRACK_PRDC_H
#define L1TFPGATRACK_PRDC_H

////////////////////
// FRAMEWORK HEADERS
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

///////////////////////
// DATA FORMATS HEADERS
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
//
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
//
#include "L1Trigger/TrackFindingTracklet/interface/slhcevent.h"
#include "L1Trigger/TrackFindingTracklet/interface/L1TBarrel.h"
#include "L1Trigger/TrackFindingTracklet/interface/L1TDisk.h"
#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
//
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
//
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
//
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
//
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"

////////////////////////////
// DETECTOR GEOMETRY HEADERS
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
//
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"

///////////////
// FPGA emulation
#include "L1Trigger/TrackFindingTracklet/interface/Constants.h"
#include "L1Trigger/TrackFindingTracklet/interface/Sector.h"
#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"
#include "L1Trigger/TrackFindingTracklet/interface/CPUTimer.h"
#include "L1Trigger/TrackFindingTracklet/interface/StubVariance.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletCalculator.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculator.h"
#include "L1Trigger/TrackFindingTracklet/interface/Cabling.h"

#include "L1Trigger/TrackFindingTracklet/interface/GlobalHistTruth.h"
#include "L1Trigger/TrackFindingTracklet/interface/HistImp.h"

////////////////
// PHYSICS TOOLS
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
//
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

#include "L1Trigger/TrackTrigger/interface/StubPtConsistency.h"

#include "L1Trigger/TrackFindingTracklet/interface/StubKiller.h"

// L1Trk HLS end-of-job bit range checks etc.
#ifdef USEHYBRID
#ifdef USE_HLS
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFParamsCombCallHLS.h"
#endif
#endif

//////////////
// STD HEADERS
#include <memory>
#include <string>
#include <iostream>
#include <fstream>

//////////////
// NAMESPACES
using namespace edm;


#ifdef IMATH_ROOT
TFile* var_base::h_file_=0;
bool   var_base::use_root = false;
#endif


//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

/////////////////////////////////////
// this class is needed to make a map
// between different types of stubs
struct L1TStubCompare
{
public:
  bool operator()(const L1TStub& x, const L1TStub& y) const {
    if (x.layer() != y.layer()) return (y.layer()>x.layer());
    else {
      if (x.ladder() != y.ladder()) return (y.ladder()>x.ladder());
      else {
	if (x.module() != y.module()) return (y.module()>x.module());
	else {
	  if (x.iz() != y.iz()) return (y.iz()>x.iz());
	  else return (x.iphi()>y.iphi());
	}
      }
    }
  }
};


class L1FPGATrackProducer : public edm::EDProducer
{
public:

  /// Constructor/destructor
  explicit L1FPGATrackProducer(const edm::ParameterSet& iConfig);
  virtual ~L1FPGATrackProducer();

protected:

private:

  int eventnum;

  /// Containers of parameters passed by python configuration file
  edm::ParameterSet config;

  bool readMoreMcTruth_;

  /// File path for configuration files
  edm::FileInPath fitPatternFile;
  edm::FileInPath memoryModulesFile;
  edm::FileInPath processingModulesFile;
  edm::FileInPath wiresFile;

  edm::FileInPath DTCLinkFile;
  edm::FileInPath moduleCablingFile;

  edm::FileInPath DTCLinkLayerDiskFile;

  
  int failscenario_;
  StubKiller* my_stubkiller;

  double phiWindowSF_;

  string asciiEventOutName_;
  std::ofstream asciiEventOut_;

  HistImp* histimp;

  string geometryType_;

  Sector** sectors;
  Cabling cabling;

  unsigned int nHelixPar_;
  bool extended_;
  
  std::map<string,vector<int> > dtclayerdisk;

  edm::ESHandle<TrackerTopology> tTopoHandle;
  edm::ESHandle<TrackerGeometry> tGeomHandle;

  edm::InputTag MCTruthClusterInputTag;
  edm::InputTag MCTruthStubInputTag;
  edm::InputTag TrackingParticleInputTag;
  edm::InputTag TrackingVertexInputTag;
  edm::InputTag simTrackSrc_;
  edm::InputTag simVertexSrc_;
  edm::InputTag ttStubSrc_;
  edm::InputTag bsSrc_;

  const edm::EDGetTokenT< edm::SimTrackContainer > simTrackToken_;
  const edm::EDGetTokenT< edm::SimVertexContainer > simVertexToken_;
  const edm::EDGetTokenT< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > > > ttStubToken_;
  const edm::EDGetTokenT< reco::BeamSpot > bsToken_;

  edm::EDGetTokenT< TTClusterAssociationMap< Ref_Phase2TrackerDigi_ > > ttClusterMCTruthToken_;
  edm::EDGetTokenT< TTStubAssociationMap< Ref_Phase2TrackerDigi_ > > ttStubMCTruthToken_;
  edm::EDGetTokenT< std::vector< TrackingParticle > > TrackingParticleToken_;
  edm::EDGetTokenT< std::vector< TrackingVertex > > TrackingVertexToken_;

  /// ///////////////// ///
  /// MANDATORY METHODS ///
  virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
  virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
  virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );
  virtual void endJob();
};


//////////////
// CONSTRUCTOR
L1FPGATrackProducer::L1FPGATrackProducer(edm::ParameterSet const& iConfig) :
  config(iConfig),
  readMoreMcTruth_(iConfig.getParameter<bool>("readMoreMcTruth")),
  MCTruthClusterInputTag(readMoreMcTruth_ ? config.getParameter<edm::InputTag>("MCTruthClusterInputTag") : edm::InputTag()),
  MCTruthStubInputTag(readMoreMcTruth_ ? config.getParameter<edm::InputTag>("MCTruthStubInputTag") : edm::InputTag()),
  TrackingParticleInputTag(readMoreMcTruth_ ? iConfig.getParameter<edm::InputTag>("TrackingParticleInputTag") : edm::InputTag()),
  TrackingVertexInputTag(readMoreMcTruth_ ? iConfig.getParameter<edm::InputTag>("TrackingVertexInputTag") : edm::InputTag()),
  simTrackSrc_(config.getParameter<edm::InputTag>("SimTrackSource")),
  simVertexSrc_(config.getParameter<edm::InputTag>("SimVertexSource")),
  ttStubSrc_(config.getParameter<edm::InputTag>("TTStubSource")),
  bsSrc_(config.getParameter<edm::InputTag>("BeamSpotSource")),

  simTrackToken_(consumes< edm::SimTrackContainer >(simTrackSrc_)),
  simVertexToken_(consumes< edm::SimVertexContainer >(simVertexSrc_)),
  ttStubToken_(consumes< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > > >(ttStubSrc_)),
  bsToken_(consumes< reco::BeamSpot >(bsSrc_))
{
  if (readMoreMcTruth_) {
      ttClusterMCTruthToken_ = consumes< TTClusterAssociationMap< Ref_Phase2TrackerDigi_ > >(MCTruthClusterInputTag);
      ttStubMCTruthToken_ = consumes< TTStubAssociationMap< Ref_Phase2TrackerDigi_ > >(MCTruthStubInputTag);
      TrackingParticleToken_ = consumes< std::vector< TrackingParticle > >(TrackingParticleInputTag);
      TrackingVertexToken_ = consumes< std::vector< TrackingVertex > >(TrackingVertexInputTag);
  }

  produces< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > >( "Level1TTTracks" ).setBranchAlias("Level1TTTracks");

  failscenario_ = iConfig.getUntrackedParameter<int>("failscenario",0);

  phiWindowSF_ = iConfig.getUntrackedParameter<double>("phiWindowSF",1.0);

  asciiEventOutName_ = iConfig.getUntrackedParameter<string>("asciiFileName","");

  geometryType_ = iConfig.getUntrackedParameter<string>("trackerGeometryType","");

  fitPatternFile = iConfig.getParameter<edm::FileInPath> ("fitPatternFile");
  processingModulesFile = iConfig.getParameter<edm::FileInPath> ("processingModulesFile");
  memoryModulesFile = iConfig.getParameter<edm::FileInPath> ("memoryModulesFile");
  wiresFile = iConfig.getParameter<edm::FileInPath> ("wiresFile");

  DTCLinkFile = iConfig.getParameter<edm::FileInPath> ("DTCLinkFile");
  moduleCablingFile = iConfig.getParameter<edm::FileInPath> ("moduleCablingFile");

  DTCLinkLayerDiskFile = iConfig.getParameter<edm::FileInPath> ("DTCLinkLayerDiskFile");


  // --------------------------------------------------------------------------------
  // get all constants
  // --------------------------------------------------------------------------------


  extended_  = iConfig.getUntrackedParameter<bool>("Extended",false);
  nHelixPar_ = iConfig.getUntrackedParameter<unsigned int>("Hnpar",4);
  hourglassExtended = extended_; 
  nHelixPar = nHelixPar_;
  
  krinvpars = TrackletCalculator::ITC_L1L2.rinv_final.get_K();
  kphi0pars = TrackletCalculator::ITC_L1L2.phi0_final.get_K();
  ktpars    = TrackletCalculator::ITC_L1L2.t_final.get_K();
  kz0pars   = TrackletCalculator::ITC_L1L2.z0_final.get_K();
  kd0pars   = kd0;

  krdisk = kr;
  kzpars = kz;
  krprojshiftdisk = TrackletCalculator::ITC_L1L2.rD_0_final.get_K();

  //those can be made more transparent...
  kphiproj123=kphi0pars*4;
  kphiproj456=kphi0pars/2;
  kzproj=kz;
  kphider=krinvpars*(1<<phiderbitshift);
  kzder=ktpars*(1<<zderbitshift);
  kphiprojdisk=kphi0pars*4.0;
  krprojderdiskshift=krprojderdisk*(1<<rderdiskbitshift);
  krprojderdisk=(1.0/ktpars)/(1<<t2bits);


  eventnum=0;
  if (asciiEventOutName_!="") {
    asciiEventOut_.open(asciiEventOutName_.c_str());
  }

  // adding capability of booking histograms internal to tracklet steps
  if (bookHistos) {
    histimp=new HistImp;
    TH1::AddDirectory(kTRUE); 
    histimp->init();
    histimp->bookLayerResidual();
    histimp->bookDiskResidual();
    histimp->bookTrackletParams();
    histimp->bookSeedEff();

    GlobalHistTruth::histograms()=histimp;
  }


  sectors=new Sector*[NSector];

  if (debug1) {
    cout << "cabling DTC links :     "<<DTCLinkFile.fullPath()<<endl;
    cout << "module cabling :     "<<moduleCablingFile.fullPath()<<endl;
    cout << "DTC link layer disk :     "<<DTCLinkLayerDiskFile.fullPath()<<endl;
  }

  cabling.init(DTCLinkFile.fullPath().c_str(),moduleCablingFile.fullPath().c_str());

  ifstream in(DTCLinkLayerDiskFile.fullPath().c_str());
  string dtc;
  in >> dtc;
  while (in.good()){
    vector<int> tmp;
    dtclayerdisk[dtc]=tmp;
    int layerdisk;
    in >> layerdisk;
    while (layerdisk>0) {
      dtclayerdisk[dtc].push_back(layerdisk);
      in >> layerdisk;
    }
    in >> dtc;
  }
  

  for (unsigned int i=0;i<NSector;i++) {
    sectors[i]=new Sector(i);
  }

  if (debug1) {
    cout << "fit pattern :     "<<fitPatternFile.fullPath()<<endl;
    cout << "process modules : "<<processingModulesFile.fullPath()<<endl;
    cout << "memory modules :  "<<memoryModulesFile.fullPath()<<endl;
    cout << "wires          :  "<<wiresFile.fullPath()<<endl;
  }

  fitpatternfile=fitPatternFile.fullPath();


  if (debug1) cout << "Will read memory modules file"<<endl;
  
  // Read list of memory modules (format: ModuleType: ModuleInstance)
  ifstream inmem(memoryModulesFile.fullPath().c_str());
  assert(inmem.good());

  while (inmem.good()){
    string memType, memName, size;
    inmem >>memType>>memName>>size;
    if (!inmem.good()) continue;
    if (writetrace) {
      cout << "Read memory: "<<memType<<" "<<memName<<endl;
    }
    for (unsigned int i=0;i<NSector;i++) {
      sectors[i]->addMem(memType,memName);
    }

  }


  if (debug1) cout << "Will read processing modules file"<<endl;
  
  // Read list of processing modules (format: ModuleType: ModuleInstance)
  ifstream inproc(processingModulesFile.fullPath().c_str());
  assert(inproc.good());

  while (inproc.good()){
    string procType, procName;
    inproc >>procType>>procName;
    if (!inproc.good()) continue;
    if (writetrace) {
      cout << "Read process: "<<procType<<" "<<procName<<endl;
    }
    for (unsigned int i=0;i<NSector;i++) {
      sectors[i]->addProc(procType,procName);
    }

  }


  if (debug1) cout << "Will read wiring information"<<endl;
  

  // Reading wiring map (format: Mem  input=> ProcModuleWrite.pinX  output=>ProcModuleRead.pinY) 
  ifstream inwire(wiresFile.fullPath().c_str());
  assert(inwire.good());

  while (inwire.good()){
    string line;
    getline(inwire,line);
    if (!inwire.good()) continue;
    if (writetrace) {
      cout << "Line : "<<line<<endl;
    }
    stringstream ss(line);
    string mem,tmp1,procin,tmp2,procout;
    ss>>mem>>tmp1>>procin;
    if (procin=="output=>") {
      procin="";
      ss>>procout;
    }
    else{
      ss>>tmp2>>procout;
    }

    for (unsigned int i=0;i<NSector;i++) {
      sectors[i]->addWire(mem,procin,procout);
    }

  }


}

/////////////
// DESTRUCTOR
L1FPGATrackProducer::~L1FPGATrackProducer()
{
  if (asciiEventOutName_!="") {
    asciiEventOut_.close();
  }

  if (bookHistos) {
    histimp->close();
  }

  delete[] sectors;

}

//////////
// END JOB
void L1FPGATrackProducer::endRun(const edm::Run& run, const edm::EventSetup& iSetup)
{
}

void L1FPGATrackProducer::endJob()
{
#ifdef USEHYBRID
#ifdef USE_HLS
  TMTT::Settings settingsTMTT;
  TMTT::KFParamsCombCallHLS fitterKF(&settingsTMTT, nHelixPar, "KFfitterHLS");
  fitterKF.endJob(); // Check bits ranges of KF HLS.
#endif
#endif
}

////////////
// BEGIN JOB
void L1FPGATrackProducer::beginRun(const edm::Run& run, const edm::EventSetup& iSetup )
{

  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  iSetup.get<TrackerDigiGeometryRecord>().get(tGeomHandle);

  const TrackerTopology* const tTopo = tTopoHandle.product();
  const TrackerGeometry* const theTrackerGeom = tGeomHandle.product();

  // ------------------------------------------------------------------------------------------
  // check killing stubs for stress test

  int failtype = 0;
  if (failscenario_ < 0 || failscenario_ > 5) {
    std::cout << "invalid fail scenario! ignoring input" << std::endl;
  }
  else {
    failtype = failscenario_;
  }

  my_stubkiller = new StubKiller();
  my_stubkiller->initialise(failtype, tTopo, theTrackerGeom);
  
  // ------------------------------------------------------------------------------------------


}

//////////
// PRODUCE
void L1FPGATrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  bool doMyDebug = false;
  if (doMyDebug) std::cout << "start in L1FPGATrackProducer::produce()" << std::endl;

  bool isTilted = true;
  if (geometryType_ == "flat" || geometryType_ == "D10") isTilted = false;

  if (doMyDebug) {
    if (isTilted) std::cout << "assuming the TILTED barrel geometry!" << std::endl;
    else std::cout << "assuming the FLAT barrel geometry!" << std::endl;
  }


  typedef std::map< L1TStub, edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ >  >, L1TStubCompare > stubMapType;

  typedef edm::Ref< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >, TTCluster< Ref_Phase2TrackerDigi_ > > TTClusterRef;

  /// Prepare output
  std::unique_ptr< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > > L1TkTracksForOutput( new std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > );

  stubMapType stubMap;

  /// Geometry handles etc
  edm::ESHandle<TrackerGeometry> geometryHandle;


  /// Set pointers to Stacked Modules
  iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);


  ////////////////////////
  // GET MAGNETIC FIELD //
  edm::ESHandle<MagneticField> magneticFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  double mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0,0,0)).z();


  ////////////
  // GET BS //
  edm::Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByToken( bsToken_, beamSpotHandle );
  math::XYZPoint bsPosition=beamSpotHandle->position();

  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  iSetup.get<TrackerDigiGeometryRecord>().get(tGeomHandle);

  eventnum++;
  SLHCEvent ev;
  ev.setEventNum(eventnum);
  ev.setIPx(bsPosition.x());
  ev.setIPy(bsPosition.y());

  GlobalHistTruth::event()=&ev;

  ///////////////////
  // GET SIMTRACKS //
  edm::Handle<edm::SimTrackContainer>   simTrackHandle;
  edm::Handle<edm::SimVertexContainer>  simVtxHandle;
  iEvent.getByToken( simTrackToken_, simTrackHandle );
  iEvent.getByToken( simVertexToken_, simVtxHandle );

  // tracking particles
  edm::Handle< std::vector< TrackingParticle > > TrackingParticleHandle;
  edm::Handle< std::vector< TrackingVertex > > TrackingVertexHandle;
  if (readMoreMcTruth_) iEvent.getByToken(TrackingParticleToken_, TrackingParticleHandle);
  if (readMoreMcTruth_) iEvent.getByToken(TrackingVertexToken_, TrackingVertexHandle);



  const TrackerTopology* const tTopo = tTopoHandle.product();
  const TrackerGeometry* const theTrackerGeom = tGeomHandle.product();


  ////////////////////////
  // GET THE PRIMITIVES //
  edm::Handle< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > > >    Phase2TrackerDigiTTStubHandle;
  iEvent.getByToken( ttStubToken_,        Phase2TrackerDigiTTStubHandle );

  // must be defined for code to compile, even if it's not used unless readMoreMcTruth_ is true
  map<edm::Ptr< TrackingParticle >, int > translateTP;


  // MC truth association maps
  edm::Handle< TTClusterAssociationMap< Ref_Phase2TrackerDigi_ > > MCTruthTTClusterHandle;
  edm::Handle< TTStubAssociationMap< Ref_Phase2TrackerDigi_ > > MCTruthTTStubHandle;
  if (readMoreMcTruth_) {
  iEvent.getByToken(ttClusterMCTruthToken_, MCTruthTTClusterHandle);
  iEvent.getByToken(ttStubMCTruthToken_, MCTruthTTStubHandle);


  ////////////////////////////////////////////////
  /// LOOP OVER TRACKING PARTICLES & GET SIMTRACKS

  if (doMyDebug) std::cout << "loop over tracking particles" << std::endl;

  int this_tp = 0;
  std::vector< TrackingParticle >::const_iterator iterTP;

  int ntps=1; //count from 1 ; 0 will mean invalid

  for (iterTP = TrackingParticleHandle->begin(); iterTP != TrackingParticleHandle->end(); ++iterTP) {

    edm::Ptr< TrackingParticle > tp_ptr(TrackingParticleHandle, this_tp);
    this_tp++;

    // only keep TPs producing a cluster
    if (MCTruthTTClusterHandle->findTTClusterRefs(tp_ptr).size() < 1) continue;

    if (iterTP->g4Tracks().size()==0) {
      if (doMyDebug) cout << "TP has no g4Track" << endl;
      continue;
    }

    int sim_trackid = ntps;
    int sim_eventid = iterTP->g4Tracks().at(0).eventId().event();
    int sim_type = iterTP->pdgId();
    float sim_pt = iterTP->pt();
    float sim_eta = iterTP->eta();
    float sim_phi = iterTP->phi();

    float vx=iterTP->vertex().x();
    float vy=iterTP->vertex().y();
    float vz=iterTP->vertex().z();

    if (sim_pt<1.0) continue;
    if (fabs(vz)>100.0) continue;
    if (hypot(vx,vy)>50.0) continue;

    if (doMyDebug) std::cout << "adding sim track with eventID trackID type pt eta phi = " << sim_eventid << " " << sim_trackid << " "
			     << sim_type << " " << sim_pt << " " << sim_eta << " " << sim_phi << std::endl;

    ev.addL1SimTrack(sim_eventid, ntps, sim_type, sim_pt, sim_eta, sim_phi,
		     vx, vy, vz);

    translateTP[tp_ptr]=ntps;

    //cout << "translateTP : "<<ntps<<endl;

    ntps++;


  }//end loop over TPs

  } // end if (readMoreMcTruth_)
  

  ////////////////////////////////
  /// COLLECT STUB INFORMATION ///
  ////////////////////////////////

  // loop over stubs
  if (doMyDebug) std::cout << "loop over stubs" << std::endl;

  for (auto gd=theTrackerGeom->dets().begin(); gd != theTrackerGeom->dets().end(); gd++) {

    DetId detid = (*gd)->geographicalId();
    if(detid.subdetId()!=StripSubdetector::TOB && detid.subdetId()!=StripSubdetector::TID ) continue; // only run on OT
    if(!tTopo->isLower(detid) ) continue; // loop on the stacks: choose the lower arbitrarily
    DetId stackDetid = tTopo->stack(detid); // Stub module detid

    if (Phase2TrackerDigiTTStubHandle->find( stackDetid ) == Phase2TrackerDigiTTStubHandle->end() ) continue;

    // Get the DetSets of the Clusters
    edmNew::DetSet< TTStub< Ref_Phase2TrackerDigi_ > > stubs = (*Phase2TrackerDigiTTStubHandle)[ stackDetid ];
    const GeomDetUnit* det0 = theTrackerGeom->idToDetUnit( detid );
    const PixelGeomDetUnit* theGeomDet = dynamic_cast< const PixelGeomDetUnit* >( det0 );
    const PixelTopology* topol = dynamic_cast< const PixelTopology* >( &(theGeomDet->specificTopology()) );

    unsigned int isPSmodule=0;
    if (topol->nrows() == 960) isPSmodule=1;


    // loop over stubs
    for ( auto stubIter = stubs.begin();stubIter != stubs.end();++stubIter ) {
      edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_  > >, TTStub< Ref_Phase2TrackerDigi_  > >
	tempStubPtr = edmNew::makeRefTo( Phase2TrackerDigiTTStubHandle, stubIter );

      vector<int> assocTPs;

      if (readMoreMcTruth_) { 
      for (unsigned int iClus = 0; iClus <= 1; iClus++) { // Loop over both clusters that make up stub.

	const TTClusterRef& ttClusterRef = tempStubPtr->clusterRef(iClus);

	// Now identify all TP's contributing to either cluster in stub.
	vector< edm::Ptr< TrackingParticle > > vecTpPtr = MCTruthTTClusterHandle->findTrackingParticlePtrs(ttClusterRef);

	for (edm::Ptr< TrackingParticle> tpPtr : vecTpPtr) {
	  if (translateTP.find(tpPtr) != translateTP.end()) {
	    //cout << "Lookup translateTP : "<<translateTP.at(tpPtr)<<endl;
	    if (iClus==0) {
	      assocTPs.push_back( translateTP.at(tpPtr) );
	    } else {
	      assocTPs.push_back( -translateTP.at(tpPtr) );
	    }
	    // N.B. Since not all tracking particles are stored in InputData::vTPs_, sometimes no match will be found.
	  } else {
	    assocTPs.push_back(0);
	  }
	}
      }
      } // end if (readMoreMcTruth_)      

      MeasurementPoint coords = tempStubPtr->clusterRef(0)->findAverageLocalCoordinatesCentered();
      LocalPoint clustlp = topol->localPosition(coords);
      GlobalPoint posStub  =  theGeomDet->surface().toGlobal(clustlp);

      int eventID=-1;

      if (readMoreMcTruth_) {
      edm::Ptr< TrackingParticle > my_tp = MCTruthTTStubHandle->findTrackingParticlePtr(tempStubPtr);

      if (my_tp.isNull()) {
	if (doMyDebug) cout << "TP is null pointer" << endl;
      }
      else {
	if (doMyDebug) cout << "TP is NOT null pointer" << endl;
      }
      } // end if (readMoreMcTruth_)
      
      int layer=-999999;
      int ladder=-999999;
      int module=-999999;

      int strip=460;

      if ( detid.subdetId()==StripSubdetector::TOB ) {
	layer  = static_cast<int>(tTopo->layer(detid));
        module = static_cast<int>(tTopo->module(detid));
	ladder = static_cast<int>(tTopo->tobRod(detid));
	if (doMyDebug) cout << "layer = " << layer << " vs " << static_cast<int>(tTopo->tobLayer(detid)) << endl;

	// https://github.com/cms-sw/cmssw/tree/master/Geometry/TrackerNumberingBuilder
	// tobSide = 1: ring- (tilted)
	// tobSide = 2: ring+ (tilted)
	// tobSide = 3: barrel (flat)
	int tobSide = static_cast<int>(tTopo->tobSide(detid));

	if (isTilted) {
	  if (layer==1)
	    {
	      if (tobSide==1) {
		module = static_cast<int>(tTopo->tobRod(detid));
		ladder = static_cast<int>(tTopo->module(detid));
	      }
	      if (tobSide==2) {
		module = 19+static_cast<int>(tTopo->tobRod(detid));
		ladder = static_cast<int>(tTopo->module(detid));
	      }
	      if (tobSide==3) module = 12+static_cast<int>(tTopo->module(detid));
	    }

	  if (layer==2)
	    {
	      if (tobSide==1) {
		module = static_cast<int>(tTopo->tobRod(detid));
		ladder = static_cast<int>(tTopo->module(detid));
	      }
	      if (tobSide==2) {
		module = 23+static_cast<int>(tTopo->tobRod(detid));
		ladder = static_cast<int>(tTopo->module(detid));
	      }
	      if (tobSide==3) module = 12+static_cast<int>(tTopo->module(detid));
	    }

	  if (layer==3)
	    {
	      if (tobSide==1) {
		module = static_cast<int>(tTopo->tobRod(detid));
		ladder = static_cast<int>(tTopo->module(detid));
	      }
	      if (tobSide==2) {
		module = 27+static_cast<int>(tTopo->tobRod(detid));
		ladder = static_cast<int>(tTopo->module(detid));
	      }
	      if (tobSide==3) module = 12+static_cast<int>(tTopo->module(detid));
	    }
	}//end special stuff for tilted barrel
      }
      else if ( detid.subdetId()==StripSubdetector::TID ) {
	layer  = 1000+static_cast<int>(tTopo->tidRing(detid));
	ladder =  static_cast<int>(tTopo->module(detid));
	module = static_cast<int>(tTopo->tidWheel(detid));
	if (doMyDebug) cout << "disk = " << layer << " vs ring = " << static_cast<int>(tTopo->tidRing(detid)) << endl;
      }

      if (doMyDebug) std::cout << "... stub with layer module ladder = " << layer << " " << ladder << " " << module << std::endl;

      // clusters
      if (doMyDebug) std::cout << "... getting clusters for that stub" << std::endl;
      std::vector<bool> innerStack;
      std::vector<int> irphi;
      std::vector<int> iz;
      std::vector<int> iladder;
      std::vector<int> imodule;

      /// Get the Inner and Outer TTCluster
      edm::Ref< edmNew::DetSetVector< TTCluster<Ref_Phase2TrackerDigi_> >, TTCluster<Ref_Phase2TrackerDigi_> > innerCluster = tempStubPtr->clusterRef(0);

      std::vector< int > innerrows= innerCluster->getRows();
      std::vector< int > innercols= innerCluster->getCols();

      for (unsigned int ihit=0;ihit<innerrows.size();ihit++){
	innerStack.push_back(true);
	irphi.push_back(innerrows[ihit]);
	iz.push_back(innercols[ihit]);
	iladder.push_back(ladder);
	imodule.push_back(module);
      }


      edm::Ref< edmNew::DetSetVector< TTCluster<Ref_Phase2TrackerDigi_> >, TTCluster<Ref_Phase2TrackerDigi_> > outerCluster = tempStubPtr->clusterRef(1);

      std::vector< int > outerrows= outerCluster->getRows();
      std::vector< int > outercols= outerCluster->getCols();

      for (unsigned int ihit=0;ihit<outerrows.size();ihit++){
	innerStack.push_back(false);
	irphi.push_back(outerrows[ihit]);
	iz.push_back(outercols[ihit]);
	iladder.push_back(ladder);
	imodule.push_back(module);
      }


      // -----------------------------------------------------
      // check module orientation, if flipped, need to store that information for track fit
      // -----------------------------------------------------

      const DetId innerDetId = innerCluster->getDetId();
      const GeomDetUnit* det_inner = theTrackerGeom->idToDetUnit( innerDetId );
      const PixelGeomDetUnit* theGeomDet_inner = dynamic_cast< const PixelGeomDetUnit* >( det_inner );
      const PixelTopology* topol_inner = dynamic_cast< const PixelTopology* >( &(theGeomDet_inner->specificTopology()) );

      MeasurementPoint coords_inner = innerCluster->findAverageLocalCoordinatesCentered();
      LocalPoint clustlp_inner = topol_inner->localPosition(coords_inner);
      GlobalPoint posStub_inner  =  theGeomDet_inner->surface().toGlobal(clustlp_inner);

      const DetId outerDetId = outerCluster->getDetId();
      const GeomDetUnit* det_outer = theTrackerGeom->idToDetUnit( outerDetId );
      const PixelGeomDetUnit* theGeomDet_outer = dynamic_cast< const PixelGeomDetUnit* >( det_outer );
      const PixelTopology* topol_outer = dynamic_cast< const PixelTopology* >( &(theGeomDet_outer->specificTopology()) );

      MeasurementPoint coords_outer = outerCluster->findAverageLocalCoordinatesCentered();
      LocalPoint clustlp_outer = topol_outer->localPosition(coords_outer);
      GlobalPoint posStub_outer  =  theGeomDet_outer->surface().toGlobal(clustlp_outer);

      unsigned int isFlipped=0;
      if (posStub_outer.mag() < posStub_inner.mag()) isFlipped = 1;

      // -----------------------------------------------------

      // correct sign for stubs in negative endcap
      float stub_bend = tempStubPtr->bendFE();
      float stub_pt = -1;
      if (layer>999 && posStub.z()<0.0) {
	//stub_pt=-stub_pt;
	stub_bend=-stub_bend;
      }
      if (irphi.size()!=0) {
      	strip=irphi[0];
      }

      // ------------------------------------------------------------------------------------------
      // check killing stubs for stress test
      
      const TTStub<Ref_Phase2TrackerDigi_> *mystub = &(*tempStubPtr);
      bool killthis = my_stubkiller->killStub(mystub);
      //bool killthis = false;
      
      // ------------------------------------------------------------------------------------------


      if (tempStubPtr->rawBend() > 100.) {
	if (doMyDebug) std::cout << "... if FE inefficiencies calculated, this stub is thrown out! " << endl;
      }
      else if (killthis) {
	if (doMyDebug) std::cout << "killing this stub!" << std::endl;
      }
      else {
	if (doMyDebug) std::cout << "... add this stub to the event!" << std::endl;
	if (ev.addStub(layer,ladder,module,strip,eventID,assocTPs,stub_pt,stub_bend,
		       posStub.x(),posStub.y(),posStub.z(),
		       innerStack,irphi,iz,iladder,imodule,isPSmodule,isFlipped)) {

	  L1TStub lastStub=ev.lastStub();
	  stubMap[lastStub]=tempStubPtr;
	}
      }

    }
  }

  if (doMyDebug) std::cout << "Will actually do L1 tracking:"<<std::endl;


  //////////////////////////
  // NOW RUN THE L1 tracking

  if (asciiEventOutName_!="") {
    ev.write(asciiEventOut_);
  }

  CPUTimer readTimer;
  CPUTimer cleanTimer;
  CPUTimer addStubTimer;
  CPUTimer VMRouterTimer;
  CPUTimer TETimer;
  CPUTimer TEDTimer;
  CPUTimer TRETimer;
  CPUTimer TCTimer;
  CPUTimer TCDTimer;
  CPUTimer PTTimer;
  CPUTimer PRTimer;
  CPUTimer METimer;
  CPUTimer MCTimer;
  CPUTimer MPTimer;
  CPUTimer MTTimer;
  CPUTimer FTTimer;
  CPUTimer PDTimer;

  if (writeSeeds) {
    ofstream fout("seeds.txt", ofstream::out);
    fout.close();
  }

  bool first=true;

  std::vector<Track*> tracks;

  int selectmu=0;
  L1SimTrack simtrk(0,0,0,0.0,0.0,0.0,0.0,0.0,0.0);

  ofstream outres;
  if (writeResEff) outres.open("trackres.txt");

  ofstream outeff;
  if (writeResEff) outeff.open("trackeff.txt");

  int nlayershit=0;

#include "FPGA.icc"


  int ntracks=0;


  for (unsigned itrack=0; itrack<tracks.size(); itrack++) {
    Track* track=tracks[itrack];

    if (track->duplicate()) continue;

    ntracks++;

    // this is where we create the TTTrack object
    double tmp_rinv     = track->rinv();
    double tmp_phi      = track->phi0();
    double tmp_tanL     = track->tanL();
    double tmp_z0       = track->z0();
    double tmp_d0       = track->d0();
    double tmp_chi2rphi = track->chisqrphi();
    double tmp_chi2rz   = track->chisqrz();
    unsigned int tmp_hit  = track->hitpattern();
    double tmp_Bfield = mMagneticFieldStrength;
   
    TTTrack< Ref_Phase2TrackerDigi_ > aTrack(tmp_rinv, tmp_phi, tmp_tanL, tmp_z0, tmp_d0, tmp_chi2rphi, tmp_chi2rz, 0,0,0, tmp_hit, nHelixPar_, tmp_Bfield);

    unsigned int trksector = track->sector();
    unsigned int trkseed = (unsigned int) abs(track->seed());

    aTrack.setPhiSector(trksector); //tracklet phi sector
    aTrack.setTrackSeedType(trkseed);   //tracklet seed

    vector<L1TStub*> stubptrs = track->stubs();
    vector<L1TStub> stubs;

    if (doMyDebug) {
      cout << "FPGA Track pt, eta, phi, z0, chi2, nstub, rinv = "
	   << track->pt() << " " << track->eta() << " " << track->phi0() << " " << track->z0() << " "
	   << track->chisq() << " " << stubptrs.size() << " " << track->rinv() << endl;
      cout << "INT FPGA Track irinv, iphi0, iz0, it, ichisq = "
	   << track->irinv() << " " << track->iphi0() << " " << track->iz0() << " " << track->it()
	   << " " << track->ichisq() << endl;
    }

    for (unsigned int i=0;i<stubptrs.size();i++){
      stubs.push_back(*(stubptrs[i]));
    }


    stubMapType::const_iterator it;
    for (vector<L1TStub>::const_iterator itstubs = stubs.begin(); itstubs != stubs.end(); itstubs++) {
      it=stubMap.find(*itstubs);
      if (it!=stubMap.end()) {
	aTrack.addStubRef(it->second);
      }
      else{
	cout << "Could not find stub in stub map"<<endl;
	cout << "stub:"<<itstubs->layer()<<" "
	     <<itstubs->ladder()<<" "
	     <<itstubs->module()<<" "
	     <<itstubs->iz()<<" "
	     <<itstubs->iphi()<<endl;

      }
    }


    // pt consistency
    float ptconsistency = StubPtConsistency::getConsistency(aTrack, theTrackerGeom, tTopo,  mMagneticFieldStrength, nHelixPar_);
    aTrack.setStubPtConsistency(ptconsistency);


    // set TTTrack word
    aTrack.setTrackWordBits();
    
    // test track word
    //aTrack.testTrackWordBits();
      
    L1TkTracksForOutput->push_back(aTrack);

  }

  iEvent.put( std::move(L1TkTracksForOutput), "Level1TTTracks");

  for (unsigned int k=0;k<NSector;k++) {
    sectors[k]->clean();
  }
  

} /// End of produce()


// ///////////////////////////
// // DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(L1FPGATrackProducer);

#endif
