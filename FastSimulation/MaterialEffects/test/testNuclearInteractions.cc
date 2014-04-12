// user include files
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "FastSimulation/Particle/interface/ParticleTable.h"
#include "FastSimDataFormats/NuclearInteractions/interface/NUEvent.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <vector>
#include <string>
#include "TFile.h"
#include "TTree.h"
#include "TProcessID.h"

class testNuclearInteractions : public edm::EDProducer {

public :
  explicit testNuclearInteractions(const edm::ParameterSet&);
  ~testNuclearInteractions();

  virtual void produce(edm::Event&, const edm::EventSetup& ) override;
  virtual void beginRun(edm::Run const&,  const edm::EventSetup & ) override;
private:
  
  // See RecoParticleFlow/PFProducer/interface/PFProducer.h
  edm::ParameterSet particleFilter_;
  bool saveNU;
  std::vector<FSimEvent*> mySimEvent;
  NUEvent* nuEvent;
  TTree* nuTree;
  TFile* outFile;
  int ObjectNumber;
  std::string simModuleLabel_;
  // Histograms
  DQMStore * dbe;
  std::vector<MonitorElement*> h0;
  std::vector<MonitorElement*> h1;
  std::vector<MonitorElement*> h2;
  std::vector<MonitorElement*> h3;
  std::vector<MonitorElement*> h4;
  std::vector<MonitorElement*> h5;
  std::vector<MonitorElement*> h6;
  std::vector<MonitorElement*> h7;
  std::vector<MonitorElement*> h8;
  std::vector<MonitorElement*> h9;
  std::vector<MonitorElement*> h10;
  std::vector<MonitorElement*> htmp;
  std::vector<MonitorElement*> totalCharge;

  std::vector< std::vector<MonitorElement*> > h100;
  std::vector< std::vector<MonitorElement*> > h200;
  std::vector< std::vector<MonitorElement*> > h300;
  std::vector< std::vector<double> > trackerRadius;
  std::vector< std::vector<double> > trackerLength;
  std::vector< std::vector<double> > blockTrackerRadius;
  std::vector< std::vector<double> > blockTrackerLength;
  std::vector< std::vector<double> > subTrackerRadius;
  std::vector< std::vector<double> > subTrackerLength;
  std::vector<double> tmpRadius;
  std::vector<double> tmpLength;

  std::vector<unsigned> stoppedPions;
  std::vector<unsigned> interactingPions;
  /*
  std::vector<MonitorElement*> h6;
  std::vector<MonitorElement*> h7;
  std::vector<MonitorElement*> h8;
  std::vector<MonitorElement*> h9;
  std::vector<MonitorElement*> h10;
  std::vector<MonitorElement*> h11;
  std::vector<MonitorElement*> h12;
  */
  int intfull;
  int intfast;

  std::string NUEventFileName;
  std::string outputFileName;

  int totalNEvt;
  int totalNU;
  int maxNU;

};

testNuclearInteractions::testNuclearInteractions(const edm::ParameterSet& p) :
  mySimEvent(2, static_cast<FSimEvent*>(0)),
  h0(2,static_cast<MonitorElement*>(0)),
  h1(2,static_cast<MonitorElement*>(0)),
  h2(2,static_cast<MonitorElement*>(0)),
  h3(2,static_cast<MonitorElement*>(0)),
  h4(2,static_cast<MonitorElement*>(0)),
  h5(2,static_cast<MonitorElement*>(0)),
  h6(2,static_cast<MonitorElement*>(0)),
  h7(2,static_cast<MonitorElement*>(0)),
  h8(2,static_cast<MonitorElement*>(0)),
  h9(2,static_cast<MonitorElement*>(0)),
  h10(2,static_cast<MonitorElement*>(0)),
  htmp(2,static_cast<MonitorElement*>(0)),
  totalCharge(2,static_cast<MonitorElement*>(0)),
  tmpRadius(2,static_cast<double>(0.)),
  tmpLength(2,static_cast<double>(0.)),
  stoppedPions(2,static_cast<unsigned>(0)),
  interactingPions(2,static_cast<unsigned>(0)),
  /*
  h6(2,static_cast<MonitorElement*>(0)),
  h7(2,static_cast<MonitorElement*>(0)),
  h8(2,static_cast<MonitorElement*>(0)),
  h9(2,static_cast<MonitorElement*>(0)),
  h10(2,static_cast<MonitorElement*>(0)),
  h11(2,static_cast<MonitorElement*>(0)),
  h12(2,static_cast<MonitorElement*>(0)),
 */
  intfull(0),
  intfast(0),
  totalNEvt(0),
  totalNU(0)
{
  
  // This producer produce a vector of SimTracks
  produces<edm::SimTrackContainer>();

  // Let's just initialize the SimEvent's
  particleFilter_ = p.getParameter<edm::ParameterSet>
    ( "TestParticleFilter" );   

  // Do we save the nuclear interactions?
  saveNU = p.getParameter<bool>("SaveNuclearInteractions");
  maxNU = p.getParameter<unsigned>("MaxNumberOfNuclearInteractions");
  if ( saveNU ) 
    std::cout << "Nuclear Interactions will be saved ! " << std::endl;

  // For the full sim
  mySimEvent[0] = new FSimEvent(particleFilter_);
  // For the fast sim
  mySimEvent[1] = new FSimEvent(particleFilter_);

  // Where the nuclear interactions are saved;
  NUEventFileName = "none";
  if ( saveNU ) { 

    nuEvent = new NUEvent();
  
    NUEventFileName = 
      p.getUntrackedParameter<std::string>("NUEventFile","NuclearInteractionsTest.root");
    //    std::string outFileName = "NuclearInteractionsTest.root";
    outFile = new TFile(NUEventFileName.c_str(),"recreate");

    // Open the tree
    nuTree = new TTree("NuclearInteractions","");
    nuTree->Branch("nuEvent","NUEvent",&nuEvent,32000,99);

  }

  outputFileName = 
    p.getUntrackedParameter<std::string>("OutputFile","testNuclearInteractions.root");
  // ObjectNumber
  ObjectNumber = -1;
    
  // ... and the histograms
  dbe = edm::Service<DQMStore>().operator->();
  h0[0] = dbe->book2D("radioFull", "Full Tracker radiography", 1000, 0.,320.,1000,0., 150. );
  h0[1] = dbe->book2D("radioFast", "Fast Tracker radiography", 1000, 0.,320.,1000,0., 150. );
  h1[0] = dbe->book1D("vertexFull", "Full Nb of Vertices",20,-0.5,19.5);
  h1[1] = dbe->book1D("vertexFast", "Fast Nb of Vertices",20,-0.5,19.5);
  h2[0] = dbe->book1D("daughterFull", "Full Nb of daughters",20,-0.5,19.5);
  h2[1] = dbe->book1D("daughterFast", "Fast Nb of daughters",20,-0.5,19.5);
  h3[0] = dbe->book1D("ecmFull", "Full centre-of-mass energy",100,0.,10.);
  h3[1] = dbe->book1D("ecmFast", "Fast centre-of-mass energy",100,0.,10.);
  h4[0] = dbe->book1D("FecmFull", "Full c.m. energy fraction",100,0.,2.);
  h4[1] = dbe->book1D("FecmFast", "Fast c.m. energy fraction",100,0.,2.);
  h5[0] = dbe->book1D("FmomFull", "Full momemtum",100,0.,10.);
  h5[1] = dbe->book1D("FmomFast", "Fast momemtum",100,0.,10.);
  h6[0] = dbe->book1D("DeltaEFull4", "Full DeltaE",2000,-1.,4.);
  h6[1] = dbe->book1D("DeltaEFast4", "Fast DetlaE",2000,-1.,4.);
  h7[0] = dbe->book1D("DeltaEFull3", "Full DeltaE 3 daugh",2000,-1.,4.);
  h7[1] = dbe->book1D("DeltaEFast3", "Fast DetlaE 3 daugh",2000,-1.,4.);
  h8[0] = dbe->book1D("DeltaMFull4", "Full DeltaE",2000,-10.,40.);
  h8[1] = dbe->book1D("DeltaMFast4", "Fast DetlaE",2000,-10.,40.);
  h9[0] = dbe->book1D("DeltaMFull3", "Full DeltaE 3 daugh",2000,-10.,40.);
  h9[1] = dbe->book1D("DeltaMFast3", "Fast DetlaE 3 daugh",2000,-10.,40.);
  h10[0] = dbe->book1D("EafterFull", "E(after)/E(before) full",200,0.,4.);
  h10[1] = dbe->book1D("EafterFast", "E(after)/E(before) fast",200,0.,4.);
  /*
  h6[0] = dbe->book2D("radioFullRem1", "Full Tracker radiography", 1000, 0.,320.,1000,0., 150. );
  h6[1] = dbe->book2D("radioFastRem1", "Fast Tracker radiography", 1000, 0.,320.,1000,0., 150. );
  h7[0] = dbe->book2D("radioFullRem2", "Full Tracker radiography", 1000, 0.,320.,1000,0., 150. );
  h7[1] = dbe->book2D("radioFullRem2", "Fast Tracker radiography", 1000, 0.,320.,1000,0., 150. );
  h8[0] = dbe->book2D("radioFullBP", "Full BP radiography", 1000, 0.,320.,1000,0., 150. );
  h8[1] = dbe->book2D("radioFastBP", "Fast BP radiography", 1000, 0.,320.,1000,0., 150. );
  h9[0] = dbe->book2D("radioFullPX", "Full PX radiography", 1000, 0.,320.,1000,0., 150. );
  h9[1] = dbe->book2D("radioFastPX", "Fast PX radiography", 1000, 0.,320.,1000,0., 150. );
  h10[0] = dbe->book2D("radioFullTI", "Full TI radiography", 1000, 0.,320.,1000,0., 150. );
  h10[1] = dbe->book2D("radioFastTI", "Fast TI radiography", 1000, 0.,320.,1000,0., 150. );
  h11[0] = dbe->book2D("radioFullTO", "Full TO radiography", 1000, 0.,320.,1000,0., 150. );
  h11[1] = dbe->book2D("radioFastTO", "Fast TO radiography", 1000, 0.,320.,1000,0., 150. );
  h12[0] = dbe->book2D("radioFullCA", "Full CA radiography", 1000, 0.,320.,1000,0., 150. );
  h12[1] = dbe->book2D("radioFastCA", "Fast CA radiography", 1000, 0.,320.,1000,0., 150. );
  */

  totalCharge[0] = dbe->book1D("ChargeFull", "Total Charge (full)",19,-9.5,9.5);
  totalCharge[1] = dbe->book1D("ChargeFast", "Total Charge (fast)",19,-9.5,9.5);

  // Beam Pipe
  htmp[0] = dbe->book1D("BeamPipeFull", "Full Beam Pipe",120,0.,3.);
  htmp[1] = dbe->book1D("BeamPipeFast", "Fast Beam Pipe",120,0.,3.);
  std::vector<double> tmpRadius = p.getUntrackedParameter<std::vector<double> >("BPCylinderRadius");
  std::vector<double> tmpLength = p.getUntrackedParameter<std::vector<double> >("BPCylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // Beam Pipe (cont'd)
  htmp[0] = dbe->book1D("BPFull", "Full Beam Pipe",120,0.,3.);
  htmp[1] = dbe->book1D("BPFast", "Fast Beam Pipe",120,0.,3.);
  h300.push_back(htmp);
  subTrackerRadius.push_back(tmpRadius);
  subTrackerLength.push_back(tmpLength);

  // PIXB1
  htmp[0] = dbe->book1D("PXB1Full", "Full Pixel Barrel 1",120,0.,3.);
  htmp[1] = dbe->book1D("PXB1Fast", "Fast Pixel Barrel 1",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("PXB1CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("PXB1CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // PIXB2
  htmp[0] = dbe->book1D("PXB2Full", "Full Pixel Barrel 2",120,0.,3.);
  htmp[1] = dbe->book1D("PXB2Fast", "Fast Pixel Barrel 2",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("PXB2CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("PXB2CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // PIXB3
  htmp[0] = dbe->book1D("PXB3Full", "Full Pixel Barrel 3",120,0.,3.);
  htmp[1] = dbe->book1D("PXB3Fast", "Fast Pixel Barrel 3",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("PXB3CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("PXB3CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // PIXB Cables
  htmp[0] = dbe->book1D("PXBCFull", "Full Pixel Barrel Cables",120,0.,3.);
  htmp[1] = dbe->book1D("PXBCFast", "Fast Pixel Barrel Cables",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("PXBCablesCylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("PXBCablesCylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // All Pixel Barrel
  htmp[0] = dbe->book1D("PXBFull", "Full Pixel Barrel",120,0.,3.);
  htmp[1] = dbe->book1D("PXBFast", "Fast Pixel Barrel",120,0.,3.);
  h200.push_back(htmp);
  blockTrackerRadius.push_back(tmpRadius);
  blockTrackerLength.push_back(tmpLength);

  // PIXD1
  htmp[0] = dbe->book1D("PXD1Full", "Full Pixel Disk 1",120,0.,3.);
  htmp[1] = dbe->book1D("PXD1Fast", "Fast Pixel Disk 1",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("PXD1CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("PXD1CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // PIXD2
  htmp[0] = dbe->book1D("PXD2Full", "Full Pixel Disk 2",120,0.,3.);
  htmp[1] = dbe->book1D("PXD2Fast", "Fast Pixel Disk 2",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("PXD2CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("PXD2CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // PIXD Cables
  htmp[0] = dbe->book1D("PXDCFull", "Full Pixel Disk Cables",120,0.,3.);
  htmp[1] = dbe->book1D("PXDCFast", "Fast Pixel Disk Cables",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("PXDCablesCylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("PXDCablesCylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // All Pixel Disks
  htmp[0] = dbe->book1D("PXDFull", "Full Pixel Disk",120,0.,3.);
  htmp[1] = dbe->book1D("PXDFast", "Fast Pixel Disk",120,0.,3.);
  h200.push_back(htmp);
  blockTrackerRadius.push_back(tmpRadius);
  blockTrackerLength.push_back(tmpLength);

  // All Pixel
  htmp[0] = dbe->book1D("PixelFull", "Full Pixel",120,0.,3.);
  htmp[1] = dbe->book1D("PixelFast", "Fast Pixel",120,0.,3.);
  h300.push_back(htmp);
  subTrackerRadius.push_back(tmpRadius);
  subTrackerLength.push_back(tmpLength);

  // TIB1
  htmp[0] = dbe->book1D("TIB1Full", "Full Tracker Inner Barrel 1",120,0.,3.);
  htmp[1] = dbe->book1D("TIB1Fast", "Fast Tracker Inner Barrel 1",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TIB1CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TIB1CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TIB2
  htmp[0] = dbe->book1D("TIB2Full", "Full Tracker Inner Barrel 2",120,0.,3.);
  htmp[1] = dbe->book1D("TIB2Fast", "Fast Tracker Inner Barrel 2",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TIB2CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TIB2CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TIB3
  htmp[0] = dbe->book1D("TIB3Full", "Full Tracker Inner Barrel 3",120,0.,3.);
  htmp[1] = dbe->book1D("TIB3Fast", "Fast Tracker Inner Barrel 3",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TIB3CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TIB3CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TIB4
  htmp[0] = dbe->book1D("TIB4Full", "Full Tracker Inner Barrel 4",120,0.,3.);
  htmp[1] = dbe->book1D("TIB4Fast", "Fast Tracker Inner Barrel 4",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TIB4CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TIB4CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TIB Cables
  htmp[0] = dbe->book1D("TIBCFull", "Full Tracker Inner Barrel Cables",120,0.,3.);
  htmp[1] = dbe->book1D("TIBCFast", "Fast Tracker Inner Barrel Cables",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TIBCablesCylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TIBCablesCylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // All TIB
  htmp[0] = dbe->book1D("TIBFull", "Full Tracker Inner Barrel",120,0.,3.);
  htmp[1] = dbe->book1D("TIBFast", "Fast Tracker Inner Barrel",120,0.,3.);
  h200.push_back(htmp);
  blockTrackerRadius.push_back(tmpRadius);
  blockTrackerLength.push_back(tmpLength);

  // TID1
  htmp[0] = dbe->book1D("TID1Full", "Full Tracker Inner Disk 1",120,0.,3.);
  htmp[1] = dbe->book1D("TID1Fast", "Fast Tracker Inner Disk 1",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TID1CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TID1CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TID2
  htmp[0] = dbe->book1D("TID2Full", "Full Tracker Inner Disk 2",120,0.,3.);
  htmp[1] = dbe->book1D("TID2Fast", "Fast Tracker Inner Disk 2",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TID2CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TID2CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TID3
  htmp[0] = dbe->book1D("TID3Full", "Full Tracker Inner Disk 3",120,0.,3.);
  htmp[1] = dbe->book1D("TID3Fast", "Fast Tracker Inner Disk 3",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TID3CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TID3CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TID Cables
  htmp[0] = dbe->book1D("TIDCFull", "Full Tracker Inner Disk Cables",120,0.,3.);
  htmp[1] = dbe->book1D("TIDCFast", "Fast Tracker Inner Disk Cables",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TIDCablesCylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TIDCablesCylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // All TID
  htmp[0] = dbe->book1D("TIDFull", "Full Tracker Inner Disk",120,0.,3.);
  htmp[1] = dbe->book1D("TIDFast", "Fast Tracker Inner Disk",120,0.,3.);
  h200.push_back(htmp);
  blockTrackerRadius.push_back(tmpRadius);
  blockTrackerLength.push_back(tmpLength);

  // All Inner Tracker
  htmp[0] = dbe->book1D("InnerFull", "Full Inner Tracker",120,0.,3.);
  htmp[1] = dbe->book1D("InnerFast", "Fast Inner Tracker",120,0.,3.);
  h300.push_back(htmp);
  subTrackerRadius.push_back(tmpRadius);
  subTrackerLength.push_back(tmpLength);

  // TOB1
  htmp[0] = dbe->book1D("TOB1Full", "Full Tracker Outer Barrel 1",120,0.,3.);
  htmp[1] = dbe->book1D("TOB1Fast", "Fast Tracker Outer Barrel 1",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TOB1CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TOB1CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB2
  htmp[0] = dbe->book1D("TOB2Full", "Full Tracker Outer Barrel 2",120,0.,3.);
  htmp[1] = dbe->book1D("TOB2Fast", "Fast Tracker Outer Barrel 2",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TOB2CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TOB2CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB3
  htmp[0] = dbe->book1D("TOB3Full", "Full Tracker Outer Barrel 3",120,0.,3.);
  htmp[1] = dbe->book1D("TOB3Fast", "Fast Tracker Outer Barrel 3",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TOB3CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TOB3CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB4
  htmp[0] = dbe->book1D("TOB4Full", "Full Tracker Outer Barrel 4",120,0.,3.);
  htmp[1] = dbe->book1D("TOB4Fast", "Fast Tracker Outer Barrel 4",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TOB4CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TOB4CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB5
  htmp[0] = dbe->book1D("TOB5Full", "Full Tracker Outer Barrel 5",120,0.,3.);
  htmp[1] = dbe->book1D("TOB5Fast", "Fast Tracker Outer Barrel 5",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TOB5CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TOB5CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB6
  htmp[0] = dbe->book1D("TOB6Full", "Full Tracker Outer Barrel 6",120,0.,3.);
  htmp[1] = dbe->book1D("TOB6Fast", "Fast Tracker Outer Barrel 6",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TOB6CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TOB6CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB Cables
  htmp[0] = dbe->book1D("TOBCFull", "Full Tracker Outer Barrel Cables",120,0.,3.);
  htmp[1] = dbe->book1D("TOBCFast", "Fast Tracker Outer Barrel Cables",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TOBCablesCylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TOBCablesCylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // All TOB
  htmp[0] = dbe->book1D("TOBFull", "Full Tracker Outer Barrel",120,0.,3.);
  htmp[1] = dbe->book1D("TOBFast", "Fast Tracker Outer Barrel",120,0.,3.);
  h200.push_back(htmp);
  blockTrackerRadius.push_back(tmpRadius);
  blockTrackerLength.push_back(tmpLength);

  // TEC1
  htmp[0] = dbe->book1D("TEC1Full", "Full Tracker EndCap 1",120,0.,3.);
  htmp[1] = dbe->book1D("TEC1Fast", "Fast Tracker Endcap 1",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC1CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC1CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC2
  htmp[0] = dbe->book1D("TEC2Full", "Full Tracker EndCap 2",120,0.,3.);
  htmp[1] = dbe->book1D("TEC2Fast", "Fast Tracker Endcap 2",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC2CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC2CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC3
  htmp[0] = dbe->book1D("TEC3Full", "Full Tracker EndCap 3",120,0.,3.);
  htmp[1] = dbe->book1D("TEC3Fast", "Fast Tracker Endcap 3",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC3CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC3CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC4
  htmp[0] = dbe->book1D("TEC4Full", "Full Tracker EndCap 4",120,0.,3.);
  htmp[1] = dbe->book1D("TEC4Fast", "Fast Tracker Endcap 4",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC4CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC4CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC5
  htmp[0] = dbe->book1D("TEC5Full", "Full Tracker EndCap 5",120,0.,3.);
  htmp[1] = dbe->book1D("TEC5Fast", "Fast Tracker Endcap 5",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC5CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC5CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC6
  htmp[0] = dbe->book1D("TEC6Full", "Full Tracker EndCap 6",120,0.,3.);
  htmp[1] = dbe->book1D("TEC6Fast", "Fast Tracker Endcap 6",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC6CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC6CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC7
  htmp[0] = dbe->book1D("TEC7Full", "Full Tracker EndCap 7",120,0.,3.);
  htmp[1] = dbe->book1D("TEC7Fast", "Fast Tracker Endcap 7",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC7CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC7CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC8
  htmp[0] = dbe->book1D("TEC8Full", "Full Tracker EndCap 8",120,0.,3.);
  htmp[1] = dbe->book1D("TEC8Fast", "Fast Tracker Endcap 8",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC8CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC8CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC9
  htmp[0] = dbe->book1D("TEC9Full", "Full Tracker EndCap 9",120,0.,3.);
  htmp[1] = dbe->book1D("TEC9Fast", "Fast Tracker Endcap 9",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC9CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC9CylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // All TEC
  htmp[0] = dbe->book1D("TECFull", "Full Tracker EndCap",120,0.,3.);
  htmp[1] = dbe->book1D("TECFast", "Fast Tracker EndCap",120,0.,3.);
  h200.push_back(htmp);
  blockTrackerRadius.push_back(tmpRadius);
  blockTrackerLength.push_back(tmpLength);

  // All Outer 
  htmp[0] = dbe->book1D("OuterFull", "Full Outer Tracker",120,0.,3.);
  htmp[1] = dbe->book1D("OuterFast", "Fast Outer Tracker",120,0.,3.);
  h300.push_back(htmp);
  subTrackerRadius.push_back(tmpRadius);
  subTrackerLength.push_back(tmpLength);

  // Outer Cables
  htmp[0] = dbe->book1D("TECCFull", "Full Tracker Outer Cables",120,0.,3.);
  htmp[1] = dbe->book1D("TECCFast", "Fast Tracker Outer Cables",120,0.,3.);
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TrackerCablesCylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TrackerCablesCylinderLength");
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // All TEC
  htmp[0] = dbe->book1D("CablesFull", "Full Tracker Cables",120,0.,3.);
  htmp[1] = dbe->book1D("CablesFast", "Fast Tracker Cables",120,0.,3.);
  h200.push_back(htmp);
  blockTrackerRadius.push_back(tmpRadius);
  blockTrackerLength.push_back(tmpLength);

  // All 
  htmp[0] = dbe->book1D("TrackerFull", "Full Tracker",120,0.,3.);
  htmp[1] = dbe->book1D("TrackerFast", "Fast Tracker",120,0.,3.);
  h300.push_back(htmp);
  subTrackerRadius.push_back(tmpRadius);
  subTrackerLength.push_back(tmpLength);


 
  //  for ( unsigned hist=0; hist<h100.size(); ++hist ) 
  //    std::cout << "Cylinder " << hist 
  //	      << ", Radius = " << trackerRadius[hist][0] 
  //	      << " "  << trackerRadius[hist][1] 
  //	      << ", Length = " << trackerLength[hist][0] 
  //	      << " " << trackerLength[hist][1] << std::endl;

								
}

testNuclearInteractions::~testNuclearInteractions()
{
  std::cout << "Number of stopped pions : " << stoppedPions[0] << " " << stoppedPions[1] << " " << std::endl;
  std::cout << "Number of interac pions : " << interactingPions[0] << " " << interactingPions[1] << " " << std::endl;

  dbe->save(outputFileName);

  if ( saveNU ) {
 
    outFile->cd();
    // Fill the last (incomplete) nuEvent
    nuTree->Fill();
    // Conclude the writing on disk
    nuTree->Write();
    // Print information
    nuTree->Print();
    // And tidy up everything!
    //  outFile->Close();
    delete nuEvent;
    delete nuTree;
    delete outFile;

  }
  
  //  delete mySimEvent;
}

void testNuclearInteractions::beginRun(edm::Run const&, const edm::EventSetup & es)
{
  // init Particle data table (from Pythia)
  edm::ESHandle < HepPDT::ParticleDataTable > pdt;
  es.getData(pdt);
  if ( !ParticleTable::instance() ) ParticleTable::instance(&(*pdt));
  mySimEvent[0]->initializePdt(&(*pdt));
  mySimEvent[1]->initializePdt(&(*pdt));

}

void
testNuclearInteractions::produce(edm::Event& iEvent, const edm::EventSetup& iSetup )
{

  ++totalNEvt;
  if ( totalNEvt/1000*1000 == totalNEvt ) 
    std::cout << "Number of event analysed/NU "
	      << totalNEvt << " / " << totalNU << std::endl; 

  std::auto_ptr<edm::SimTrackContainer> nuclSimTracks(new edm::SimTrackContainer);

  //  std::cout << "Fill full event " << std::endl;
  edm::Handle<std::vector<SimTrack> > fullSimTracks;
  iEvent.getByLabel("g4SimHits",fullSimTracks);
  edm::Handle<std::vector<SimVertex> > fullSimVertices;
  iEvent.getByLabel("g4SimHits",fullSimVertices);
  mySimEvent[0]->fill( *fullSimTracks, *fullSimVertices );
  
  //  std::cout << "Fill fast event " << std::endl;
  /* */
  //  if ( !saveNU ) { 
    edm::Handle<std::vector<SimTrack> > fastSimTracks;
    iEvent.getByLabel("famosSimHits",fastSimTracks);
    edm::Handle<std::vector<SimVertex> > fastSimVertices;
    iEvent.getByLabel("famosSimHits",fastSimVertices);
    mySimEvent[1]->fill( *fastSimTracks, *fastSimVertices );
  //}
  /* */
  
  //mySimEvent[0]->print();
  XYZTLorentzVector theProtonMomentum(0.,0.,0.,0.939);

  // Save the object number count for a new NUevent
  if ( saveNU ) { 
    if ( ObjectNumber == -1 || nuEvent->nInteractions() == 1000 ) {
      ObjectNumber = TProcessID::GetObjectCount();
      nuEvent->reset();
    }
  }

  for ( unsigned ievt=0; ievt<2; ++ievt ) {

    //    std::cout << "Event number " << ievt << std::endl;
    //    mySimEvent[ievt]->print();

    //    const std::vector<FSimVertex>& fsimVertices = *(mySimEvent[ievt]->vertices() );
    //    if ( !fsimVertices.size() ) continue;
    if ( !mySimEvent[ievt]->nVertices() ) continue; 
    const FSimTrack& thePion = mySimEvent[ievt]->track(0);

    //    h1[ievt]->Fill(fsimVertices.size());
    //    if ( fsimVertices.size() == 1 ) continue;  
    h1[ievt]->Fill(mySimEvent[ievt]->nVertices());
    // Count stopping particles
    if ( mySimEvent[ievt]->nVertices() == 1 ) { 
      if ( thePion.trackerSurfaceMomentum().e() < 1E-10 ) ++stoppedPions[ievt];
    }
    if ( mySimEvent[ievt]->nVertices() == 1 ) continue;  

    const FSimVertex& thePionVertex = mySimEvent[ievt]->vertex(1);

    double zed = thePionVertex.position().z();
    double radius = thePionVertex.position().pt();
    double eta = thePionVertex.position().eta();

    h0[ievt]->Fill(fabs(thePionVertex.position().z()),
		        thePionVertex.position().pt());

    // Pion's number of daughters
    // FSimTrack& thePion = mySimEvent[ievt]->track(0);
 
    unsigned ndaugh = thePionVertex.nDaughters();
    h2[ievt]->Fill(ndaugh);

    // Check for a second vertex
    //    bool theSecondVertex = fsimVertices.size() > 2 ?
    //      mySimEvent[ievt]->vertex(2).parent().id() == 0 : false;
    //    std::cout << "Plusieurs interactions ? " << theSecondVertex << std::endl;
    
    // First and last daughters
    int firstDaughter = -1;
    int lastDaughter = -1;
    if ( thePionVertex.nDaughters() ) { 
      lastDaughter = thePionVertex.daughters()[thePionVertex.nDaughters()-1];
      firstDaughter = thePionVertex.daughters()[0];
    }
    // Reject charged pion/kaon leptonic decays (already simulated in FAMOS)
    //    if ( thePionVertex.nDaughters() == 1 ) { 
    //      const FSimTrack& myDaugh = mySimEvent[ievt]->track(firstDaughter);
    //      if (abs(myDaugh.type()) == 11 || abs(myDaugh.type()) == 13 ) return;
    //    } 
    
    XYZTLorentzVector totMoth = thePion.momentum();
    XYZTLorentzVector totDaugh(0.,0.,0.,0.);
    // double qMoth = thePion.charge();
    // double qDaugh = 0;
    unsigned nleptons=0;
    unsigned nothers=0;
    if(!(firstDaughter<0||lastDaughter<0)) {
      for(int idaugh=firstDaughter;idaugh<=lastDaughter;++idaugh) {
	const FSimTrack& myDaugh = mySimEvent[ievt]->track(idaugh);
	totDaugh += myDaugh.momentum();
	//  qDaugh += myDaugh.charge();
	// Count the leptons
	if ( abs(myDaugh.type()) == 11 || abs(myDaugh.type()) == 13 ) ++nleptons;
	// Count the hadrons
	if ( abs(myDaugh.type()) != 111 && abs(myDaugh.type()) != 211 ) ++nothers;
      }
    }

    // Reject decays (less than one/four daughters, for pions and kaons)
    if ( ( abs(thePion.type()) == 211 && ndaugh == 1) || 
	 ( abs(thePion.type()) == 130 && ndaugh < 4 ) || 
	 ( abs(thePion.type()) == 321 && ndaugh < 4 ) ) { 

      double diffE = (totMoth-totDaugh).E();
      double diffP = std::sqrt((totMoth-totDaugh).Vect().Mag2());
      double diffm = totMoth.M2()-totDaugh.M2();
      // double diffQ = qMoth-qDaugh;
      // Neutral particles (K0L) don't experience dE/dx nor multiple scattering nor B deflection. 
      // E,p are conserved!
      if ( abs(thePion.type()) == 130 && fabs(diffE) < 1E-5 && diffP < 1E-5 ) return; 
      // Low-multiplicity final states with one electron or one muon 
      // usually don't come from an interaction. All pions are taken care
      // of by this cut
      if ( nleptons == 1 ) return;
      // Reserve of tricks in case it does not work
      /*
      BaseParticlePropagator pDaugh(totDaugh,thePion.endVertex().position(),qDaugh);
      double d0 = pDaugh.xyImpactParameter();
      double z0 = pDaugh.zImpactParameter();
      pDaugh.propagateToNominalVertex();
      diffP = std::sqrt((totMoth-pDaugh.Momentum()).Vect().Mag2());
      */
      // Charge kaons may experience dE/dx and multiple scattering -> relax the cuts
      h7[ievt]->Fill(diffE);
      h9[ievt]->Fill(diffm);
      if ( abs(thePion.type()) != 211 &&      // Ignore pions 
	   diffE > -1E-5 && diffE < 0.1 &&    // Small deltaE - to be checked as f(E)
	   nothers == 0 ) return;             // Only pions in the decays
      h6[ievt]->Fill(diffE);
      h8[ievt]->Fill(diffm);
    }  

    if ( ndaugh )
      ++interactingPions[ievt];
    else  
      ++stoppedPions[ievt];

    // Find the daughters, and boost them.
    if(!(firstDaughter<0||lastDaughter<0)) {

      // Compute the boost for the cm frame, and the cm energy.
      XYZTLorentzVector theBoost = thePion.momentum()+theProtonMomentum;
      double ecm = theBoost.mag();
      theBoost /=  theBoost.e();
      if ( ievt == 0 && saveNU && totalNEvt < maxNU) {
	NUEvent::NUInteraction interaction;
	interaction.first = nuEvent->nParticles();
	interaction.last = interaction.first + lastDaughter - firstDaughter;
	nuEvent->addNUInteraction(interaction);
	++totalNU;
      }

      // A few checks
      double qTot = 0.;
      double eBefore = thePion.momentum().E();
      double eAfter = 0.;
      XYZTLorentzVector theTotal(0.,0.,0.,0.);

      // Rotation to bring the collision axis around z
      XYZVector zAxis(0.,0.,1.);
      XYZVector rotationAxis = (theBoost.Vect().Cross(zAxis)).Unit();
      double rotationAngle = std::acos(theBoost.Vect().Unit().Z());
      RawParticle::Rotation rotation(rotationAxis,rotationAngle);

      for(int idaugh=firstDaughter;idaugh<=lastDaughter;++idaugh) {

	// The track
	const FSimTrack& myDaugh = mySimEvent[ievt]->track(idaugh);
	qTot += myDaugh.charge();
	//	std::cout << "Daughter " << idaugh << " " << myDaugh << std::endl;
        RawParticle theMom(myDaugh.momentum());
	eAfter += theMom.E();

	// Boost the track
	theMom.boost(-theBoost.x(),-theBoost.y(),-theBoost.z());
	theTotal = theTotal + theMom.momentum();

	// Rotate ->  along the Z axis
	theMom.rotate(rotation);
	
 	// Save the fully simulated tracks
	if ( ievt == 0 && saveNU && totalNEvt <= maxNU) { 
	  NUEvent::NUParticle particle;
	  particle.px = theMom.px()/ecm;
	  particle.py = theMom.py()/ecm;
	  particle.pz = theMom.pz()/ecm;
	  particle.mass = theMom.mag();
	  particle.id = myDaugh.type();
	  nuEvent->addNUParticle(particle);
	  //	  SimTrack nuclSimTrack(myDaugh.type(),theMom/ecm,-1,-1);
	  //	  nuclSimTracks->push_back(nuclSimTrack);
	}
      }

      // Save some histograms
      h3[ievt]->Fill(ecm);
      h4[ievt]->Fill(theTotal.mag()/ecm);
      h5[ievt]->Fill(sqrt(theTotal.Vect().mag2()));
      h10[ievt]->Fill(eAfter/eBefore);

      // Total charge of daughters
      totalCharge[ievt]->Fill(qTot);

      // Fill the individual layer histograms !
      bool filled = false;
      for ( unsigned hist=0; hist<h100.size() && !filled; ++hist ) {
	if ( radius < trackerRadius[hist][ievt] && 
	     fabs(zed) < trackerLength[hist][ievt] ) {
	    h100[hist][ievt]->Fill(eta);
	    filled = true;
	  }
	}

      // Fill the block histograms !
      filled = false;
      for ( unsigned hist=0; hist<h200.size() && !filled; ++hist ) {
	if ( radius < blockTrackerRadius[hist][ievt] && 
	     fabs(zed) < blockTrackerLength[hist][ievt] ) {
	  h200[hist][ievt]->Fill(eta);
	  filled = true;
	}
      }

      // Fill the cumulative histograms !
      for ( unsigned hist=0; hist<h300.size(); ++hist ) {
	if ( ( radius < subTrackerRadius[hist][ievt] && 
	       fabs(zed) < subTrackerLength[hist][ievt] ) || 
	     ( hist == 2 && 
	       radius < subTrackerRadius[1][ievt] && 
	       fabs(zed) < subTrackerLength[1][ievt] ) ) {
	  h300[hist][ievt]->Fill(eta);
	}
      }
    }
    
    // Save the fully simulated tracks from the nuclear interaction
    if ( ievt == 0 && saveNU && totalNEvt <= maxNU ) {
      //      std::cout << "Saved " << nuclSimTracks->size() 
      //		<< " simTracks in the Event" << std::endl;
      // iEvent.put(nuclSimTracks);

      //      std::cout << "Number of interactions in nuEvent = "
      //		<< nuEvent->nInteractions() << std::endl;
      if ( nuEvent->nInteractions() == 1000 ) { 
        // Reset Event object count to avoid memory overflows
	TProcessID::SetObjectCount(ObjectNumber);
	// Save the nuEvent
	std::cout << "Saved " << nuEvent->nInteractions() 
		  << " Interaction(s) with " << nuEvent->nParticles()
		  << " Particles in the NUEvent " << std::endl;
       	outFile->cd(); 
	nuTree->Fill();
	//	nuTree->Print();

      }

    }

  }


}

//define this as a plug-in

DEFINE_FWK_MODULE(testNuclearInteractions);
