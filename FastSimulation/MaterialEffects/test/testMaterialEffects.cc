// user include files
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "FastSimulation/Particle/interface/ParticleTable.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <vector>
#include <string>

class testMaterialEffects : public DQMEDAnalyzer {
public :
  explicit testMaterialEffects(const edm::ParameterSet&);
  ~testMaterialEffects(){};

  virtual void analyze(const edm::Event&, const edm::EventSetup& ) override;
  virtual void dqmBeginRun(edm::Run const& ,edm::EventSetup const& ) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  
private:
  
  // See RecoParticleFlow/PFProducer/interface/PFProducer.h
  edm::ParameterSet particleFilter_;
  std::vector<FSimEvent*> mySimEvent;
  std::string simModuleLabel_;  
  //  TH2F * h100;
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
  std::vector<MonitorElement*> h11;
  std::vector<MonitorElement*> h12;
  std::vector<MonitorElement*> h13;
  std::vector<MonitorElement*> h14;
  std::vector<MonitorElement*> h15;
  std::vector<MonitorElement*> h16;
  std::vector<MonitorElement*> h17;
  std::vector<MonitorElement*> htmp;

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

  unsigned int nevt;

};

testMaterialEffects::testMaterialEffects(const edm::ParameterSet& p) :
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
  h11(2,static_cast<MonitorElement*>(0)),
  h12(2,static_cast<MonitorElement*>(0)),
  h13(2,static_cast<MonitorElement*>(0)),
  h14(2,static_cast<MonitorElement*>(0)),
  h15(2,static_cast<MonitorElement*>(0)),
  h16(2,static_cast<MonitorElement*>(0)),
  h17(2,static_cast<MonitorElement*>(0)),
  htmp(2,static_cast<MonitorElement*>(0)),
  tmpRadius(2,static_cast<double>(0.)),
  tmpLength(2,static_cast<double>(0.))
{
  
  particleFilter_ = p.getParameter<edm::ParameterSet>
    ( "TestParticleFilter" );   
  // For the full sim
  mySimEvent[0] = new FSimEvent(particleFilter_);
  // For the fast sim
  mySimEvent[1] = new FSimEvent(particleFilter_);

  // Beam Pipe
  std::vector<double> tmpRadius = p.getUntrackedParameter<std::vector<double> >("BPCylinderRadius");
  std::vector<double> tmpLength = p.getUntrackedParameter<std::vector<double> >("BPCylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // Beam Pipe (cont'd)
  blockTrackerRadius.push_back(tmpRadius);
  blockTrackerLength.push_back(tmpLength);

  // Beam Pipe (cont'd)
  subTrackerRadius.push_back(tmpRadius);
  subTrackerLength.push_back(tmpLength);

  // PIXB1
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("PXB1CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("PXB1CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // PIXB2
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("PXB2CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("PXB2CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // PIXB3
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("PXB3CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("PXB3CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // PIXB Cables
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("PXBCablesCylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("PXBCablesCylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // All Pixel Barrel
  blockTrackerRadius.push_back(tmpRadius);
  blockTrackerLength.push_back(tmpLength);

  // PIXD1
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("PXD1CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("PXD1CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // PIXD2
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("PXD2CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("PXD2CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // PIXD Cables
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("PXDCablesCylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("PXDCablesCylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // All Pixel Disks
  blockTrackerRadius.push_back(tmpRadius);
  blockTrackerLength.push_back(tmpLength);

  // All Pixel
  subTrackerRadius.push_back(tmpRadius);
  subTrackerLength.push_back(tmpLength);

  // TIB1
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TIB1CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TIB1CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TIB2
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TIB2CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TIB2CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TIB3
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TIB3CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TIB3CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TIB4
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TIB4CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TIB4CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TIB Cables
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TIBCablesCylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TIBCablesCylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // All TIB
  blockTrackerRadius.push_back(tmpRadius);
  blockTrackerLength.push_back(tmpLength);

  // TID1
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TID1CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TID1CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TID2
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TID2CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TID2CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TID3
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TID3CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TID3CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TID Cables
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TIDCablesCylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TIDCablesCylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // All TID
  blockTrackerRadius.push_back(tmpRadius);
  blockTrackerLength.push_back(tmpLength);

  // All Inner Tracker
  subTrackerRadius.push_back(tmpRadius);
  subTrackerLength.push_back(tmpLength);

  // TOB1
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TOB1CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TOB1CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB2
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TOB2CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TOB2CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB3
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TOB3CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TOB3CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB4
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TOB4CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TOB4CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB5
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TOB5CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TOB5CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB6
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TOB6CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TOB6CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB Cables
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TOBCablesCylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TOBCablesCylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // All TOB
  blockTrackerRadius.push_back(tmpRadius);
  blockTrackerLength.push_back(tmpLength);

  // TEC1
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC1CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC1CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC2
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC2CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC2CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC3
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC3CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC3CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC4
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC4CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC4CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC5
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC5CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC5CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC6
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC6CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC6CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC7
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC7CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC7CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC8
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC8CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC8CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC9
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TEC9CylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TEC9CylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // All TEC
  blockTrackerRadius.push_back(tmpRadius);
  blockTrackerLength.push_back(tmpLength);

  // All Outer 
  subTrackerRadius.push_back(tmpRadius);
  subTrackerLength.push_back(tmpLength);

  // Outer Cables
  tmpRadius = p.getUntrackedParameter<std::vector<double> >("TrackerCablesCylinderRadius");
  tmpLength = p.getUntrackedParameter<std::vector<double> >("TrackerCablesCylinderLength");
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // All TEC
  blockTrackerRadius.push_back(tmpRadius);
  blockTrackerLength.push_back(tmpLength);

  // All 
  subTrackerRadius.push_back(tmpRadius);
  subTrackerLength.push_back(tmpLength);

  // All Patrice : Finer granularity
  subTrackerRadius.push_back(tmpRadius);
  subTrackerLength.push_back(tmpLength);

  nevt=0;
}

void testMaterialEffects::bookHistograms(DQMStore::IBooker & ibooker,
					 edm::Run const & iRun,
					 edm::EventSetup const & iSetup)
{
  ibooker.setCurrentFolder("testMaterialEffects");

  h0[0] = ibooker.book2D("radioFull", "Full Tracker radiography", 1000, 0.,320.,1000,0., 150. );
  h0[1] = ibooker.book2D("radioFast", "Fast Tracker radiography", 1000, 0.,320.,1000,0., 150. );
  h1[0] = ibooker.book1D("etaEFull", "Full Electron eta distribution",54,0.,2.7);
  h1[1] = ibooker.book1D("etaEFast", "Fast Electron eta distribution",54,0.,2.7);
  h2[0] = ibooker.book1D("EgammaFull", "Full Brem energy distribution",600,0.,300.);
  h2[1] = ibooker.book1D("EgammaFast", "Fast Brem energy distribution",600,0.,300.);
  h3[0] = ibooker.book1D("FEgammaFull", "Full Brem energy fraction distribution",1000,0.,1.);
  h3[1] = ibooker.book1D("FEgammaFast", "Fast Brem energy fraction distribution",1000,0.,1.);
  h4[0] = ibooker.book1D("NgammaFull", "Full Brem number",25,-0.5,24.5);
  h4[1] = ibooker.book1D("NgammaFast", "Fast Brem number",25,-0.5,24.5);
  h5[0] = ibooker.book1D("NgammaMinFull", "Full Brem number > Emin",25,-0.5,24.5);
  h5[1] = ibooker.book1D("NgammaMinFast", "Fast Brem number > Emin",25,-0.5,24.5);
  h6[0] = ibooker.book2D("radioFullRem1", "Full Tracker radiography", 1000, 0.,320.,1000,0., 150. );
  h6[1] = ibooker.book2D("radioFastRem1", "Fast Tracker radiography", 1000, 0.,320.,1000,0., 150. );
  h7[0] = ibooker.book2D("radioFullRem2", "Full Tracker radiography", 1000, 0.,320.,1000,0., 150. );
  h7[1] = ibooker.book2D("radioFastRem2", "Fast Tracker radiography", 1000, 0.,320.,1000,0., 150. );
  h8[0] = ibooker.book2D("radioFullBP", "Full BP radiography", 1000, 0.,320.,1000,0., 150. );
  h8[1] = ibooker.book2D("radioFastBP", "Fast BP radiography", 1000, 0.,320.,1000,0., 150. );
  h9[0] = ibooker.book2D("radioFullPX", "Full PX radiography", 1000, 0.,320.,1000,0., 150. );
  h9[1] = ibooker.book2D("radioFastPX", "Fast PX radiography", 1000, 0.,320.,1000,0., 150. );
  h10[0] = ibooker.book2D("radioFullTI", "Full TI radiography", 1000, 0.,320.,1000,0., 150. );
  h10[1] = ibooker.book2D("radioFastTI", "Fast TI radiography", 1000, 0.,320.,1000,0., 150. );
  h11[0] = ibooker.book2D("radioFullTO", "Full TO radiography", 1000, 0.,320.,1000,0., 150. );
  h11[1] = ibooker.book2D("radioFastTO", "Fast TO radiography", 1000, 0.,320.,1000,0., 150. );
  h12[0] = ibooker.book2D("radioFullCA", "Full CA radiography", 1000, 0.,320.,1000,0., 150. );
  h12[1] = ibooker.book2D("radioFastCA", "Fast CA radiography", 1000, 0.,320.,1000,0., 150. );
  h13[0] = ibooker.book1D("TrackerFullR","Full Tracker Radius",300,0.,150.);
  h13[1] = ibooker.book1D("TrackerFastR","Fast Tracker Radius",300,0.,150.);
  h14[0] = ibooker.book1D("TrackerFullR2","Full Tracker Radius 2",800,0.,40.);
  h14[1] = ibooker.book1D("TrackerFastR2","Fast Tracker Radius 2",800,0.,40.);
  h15[0] = ibooker.book1D("HF1Full","Full HF1 region",550,0.,5.5);
  h15[1] = ibooker.book1D("HF1Fast","Fast HF1 region",550,0.,5.5);
  h16[0] = ibooker.book1D("HF2Full","Full HF2 region",550,0.,5.5);
  h16[1] = ibooker.book1D("HF2Fast","Fast HF2 region",550,0.,5.5);
  h17[0] = ibooker.book1D("HF3Full","Full HF3 region",550,0.,5.5);
  h17[1] = ibooker.book1D("HF3Fast","Fast HF3 region",550,0.,5.5);

								
  // Beam Pipe
  htmp[0] = ibooker.book1D("BeamPipeFull", "Full Beam Pipe",220,0.,5.5);
  htmp[1] = ibooker.book1D("BeamPipeFast", "Fast Beam Pipe",220,0.,5.5);
  h100.push_back(htmp);

  // Beam Pipe (cont'd)
  htmp[0] = ibooker.book1D("BPFullDummy", "Full Beam Pipe",220,0.,5.5);
  htmp[1] = ibooker.book1D("BPFastDummy", "Fast Beam Pipe",220,0.,5.5);
  h200.push_back(htmp);

  // Beam Pipe (cont'd)
  htmp[0] = ibooker.book1D("BPFull", "Full Beam Pipe",220,0.,5.5);
  htmp[1] = ibooker.book1D("BPFast", "Fast Beam Pipe",220,0.,5.5);
  h300.push_back(htmp);

  // PIXB1
  htmp[0] = ibooker.book1D("PXB1Full", "Full Pixel Barrel 1",220,0.,5.5);
  htmp[1] = ibooker.book1D("PXB1Fast", "Fast Pixel Barrel 1",220,0.,5.5);
  h100.push_back(htmp);

  // PIXB2
  htmp[0] = ibooker.book1D("PXB2Full", "Full Pixel Barrel 2",220,0.,5.5);
  htmp[1] = ibooker.book1D("PXB2Fast", "Fast Pixel Barrel 2",220,0.,5.5);
  h100.push_back(htmp);

  // PIXB3
  htmp[0] = ibooker.book1D("PXB3Full", "Full Pixel Barrel 3",220,0.,5.5);
  htmp[1] = ibooker.book1D("PXB3Fast", "Fast Pixel Barrel 3",220,0.,5.5);
  h100.push_back(htmp);

  // PIXB Cables
  htmp[0] = ibooker.book1D("PXBCFull", "Full Pixel Barrel Cables",220,0.,5.5);
  htmp[1] = ibooker.book1D("PXBCFast", "Fast Pixel Barrel Cables",220,0.,5.5);
  h100.push_back(htmp);

  // All Pixel Barrel
  htmp[0] = ibooker.book1D("PXBFull", "Full Pixel Barrel",220,0.,5.5);
  htmp[1] = ibooker.book1D("PXBFast", "Fast Pixel Barrel",220,0.,5.5);
  h200.push_back(htmp);

  // PIXD1
  htmp[0] = ibooker.book1D("PXD1Full", "Full Pixel Disk 1",220,0.,5.5);
  htmp[1] = ibooker.book1D("PXD1Fast", "Fast Pixel Disk 1",220,0.,5.5);
  h100.push_back(htmp);

  // PIXD2
  htmp[0] = ibooker.book1D("PXD2Full", "Full Pixel Disk 2",220,0.,5.5);
  htmp[1] = ibooker.book1D("PXD2Fast", "Fast Pixel Disk 2",220,0.,5.5);
  h100.push_back(htmp);

  // PIXD Cables
  htmp[0] = ibooker.book1D("PXDCFull", "Full Pixel Disk Cables",220,0.,5.5);
  htmp[1] = ibooker.book1D("PXDCFast", "Fast Pixel Disk Cables",220,0.,5.5);
  h100.push_back(htmp);

  // All Pixel Disks
  htmp[0] = ibooker.book1D("PXDFull", "Full Pixel Disk",220,0.,5.5);
  htmp[1] = ibooker.book1D("PXDFast", "Fast Pixel Disk",220,0.,5.5);
  h200.push_back(htmp);

  // All Pixel
  htmp[0] = ibooker.book1D("PixelFull", "Full Pixel",220,0.,5.5);
  htmp[1] = ibooker.book1D("PixelFast", "Fast Pixel",220,0.,5.5);
  h300.push_back(htmp);

  // TIB1
  htmp[0] = ibooker.book1D("TIB1Full", "Full Tracker Inner Barrel 1",220,0.,5.5);
  htmp[1] = ibooker.book1D("TIB1Fast", "Fast Tracker Inner Barrel 1",220,0.,5.5);
  h100.push_back(htmp);

  // TIB2
  htmp[0] = ibooker.book1D("TIB2Full", "Full Tracker Inner Barrel 2",220,0.,5.5);
  htmp[1] = ibooker.book1D("TIB2Fast", "Fast Tracker Inner Barrel 2",220,0.,5.5);
  h100.push_back(htmp);

  // TIB3
  htmp[0] = ibooker.book1D("TIB3Full", "Full Tracker Inner Barrel 3",220,0.,5.5);
  htmp[1] = ibooker.book1D("TIB3Fast", "Fast Tracker Inner Barrel 3",220,0.,5.5);
  h100.push_back(htmp);

  // TIB4
  htmp[0] = ibooker.book1D("TIB4Full", "Full Tracker Inner Barrel 4",220,0.,5.5);
  htmp[1] = ibooker.book1D("TIB4Fast", "Fast Tracker Inner Barrel 4",220,0.,5.5);
  h100.push_back(htmp);

  // TIB Cables
  htmp[0] = ibooker.book1D("TIBCFull", "Full Tracker Inner Barrel Cables",220,0.,5.5);
  htmp[1] = ibooker.book1D("TIBCFast", "Fast Tracker Inner Barrel Cables",220,0.,5.5);
  h100.push_back(htmp);

  // All TIB
  htmp[0] = ibooker.book1D("TIBFull", "Full Tracker Inner Barrel",220,0.,5.5);
  htmp[1] = ibooker.book1D("TIBFast", "Fast Tracker Inner Barrel",220,0.,5.5);
  h200.push_back(htmp);

  // TID1
  htmp[0] = ibooker.book1D("TID1Full", "Full Tracker Inner Disk 1",220,0.,5.5);
  htmp[1] = ibooker.book1D("TID1Fast", "Fast Tracker Inner Disk 1",220,0.,5.5);
  h100.push_back(htmp);

  // TID2
  htmp[0] = ibooker.book1D("TID2Full", "Full Tracker Inner Disk 2",220,0.,5.5);
  htmp[1] = ibooker.book1D("TID2Fast", "Fast Tracker Inner Disk 2",220,0.,5.5);
  h100.push_back(htmp);

  // TID3
  htmp[0] = ibooker.book1D("TID3Full", "Full Tracker Inner Disk 3",220,0.,5.5);
  htmp[1] = ibooker.book1D("TID3Fast", "Fast Tracker Inner Disk 3",220,0.,5.5);
  h100.push_back(htmp);

  // TID Cables
  htmp[0] = ibooker.book1D("TIDCFull", "Full Tracker Inner Disk Cables",220,0.,5.5);
  htmp[1] = ibooker.book1D("TIDCFast", "Fast Tracker Inner Disk Cables",220,0.,5.5);
  h100.push_back(htmp);

  // All TID
  htmp[0] = ibooker.book1D("TIDFull", "Full Tracker Inner Disk",220,0.,5.5);
  htmp[1] = ibooker.book1D("TIDFast", "Fast Tracker Inner Disk",220,0.,5.5);
  h200.push_back(htmp);

  // All Inner Tracker
  htmp[0] = ibooker.book1D("InnerFull", "Full Inner Tracker",220,0.,5.5);
  htmp[1] = ibooker.book1D("InnerFast", "Fast Inner Tracker",220,0.,5.5);
  h300.push_back(htmp);

  // TOB1
  htmp[0] = ibooker.book1D("TOB1Full", "Full Tracker Outer Barrel 1",220,0.,5.5);
  htmp[1] = ibooker.book1D("TOB1Fast", "Fast Tracker Outer Barrel 1",220,0.,5.5);
  h100.push_back(htmp);

  // TOB2
  htmp[0] = ibooker.book1D("TOB2Full", "Full Tracker Outer Barrel 2",220,0.,5.5);
  htmp[1] = ibooker.book1D("TOB2Fast", "Fast Tracker Outer Barrel 2",220,0.,5.5);
  h100.push_back(htmp);

  // TOB3
  htmp[0] = ibooker.book1D("TOB3Full", "Full Tracker Outer Barrel 3",220,0.,5.5);
  htmp[1] = ibooker.book1D("TOB3Fast", "Fast Tracker Outer Barrel 3",220,0.,5.5);
  h100.push_back(htmp);

  // TOB4
  htmp[0] = ibooker.book1D("TOB4Full", "Full Tracker Outer Barrel 4",220,0.,5.5);
  htmp[1] = ibooker.book1D("TOB4Fast", "Fast Tracker Outer Barrel 4",220,0.,5.5);
  h100.push_back(htmp);

  // TOB5
  htmp[0] = ibooker.book1D("TOB5Full", "Full Tracker Outer Barrel 5",220,0.,5.5);
  htmp[1] = ibooker.book1D("TOB5Fast", "Fast Tracker Outer Barrel 5",220,0.,5.5);
  h100.push_back(htmp);

  // TOB6
  htmp[0] = ibooker.book1D("TOB6Full", "Full Tracker Outer Barrel 6",220,0.,5.5);
  htmp[1] = ibooker.book1D("TOB6Fast", "Fast Tracker Outer Barrel 6",220,0.,5.5);
  h100.push_back(htmp);

  // TOB Cables
  htmp[0] = ibooker.book1D("TOBCFull", "Full Tracker Outer Barrel Cables",220,0.,5.5);
  htmp[1] = ibooker.book1D("TOBCFast", "Fast Tracker Outer Barrel Cables",220,0.,5.5);
  h100.push_back(htmp);

  // All TOB
  htmp[0] = ibooker.book1D("TOBFull", "Full Tracker Outer Barrel",220,0.,5.5);
  htmp[1] = ibooker.book1D("TOBFast", "Fast Tracker Outer Barrel",220,0.,5.5);
  h200.push_back(htmp);

  // TEC1
  htmp[0] = ibooker.book1D("TEC1Full", "Full Tracker EndCap 1",220,0.,5.5);
  htmp[1] = ibooker.book1D("TEC1Fast", "Fast Tracker Endcap 1",220,0.,5.5);
  h100.push_back(htmp);

  // TEC2
  htmp[0] = ibooker.book1D("TEC2Full", "Full Tracker EndCap 2",220,0.,5.5);
  htmp[1] = ibooker.book1D("TEC2Fast", "Fast Tracker Endcap 2",220,0.,5.5);
  h100.push_back(htmp);

  // TEC3
  htmp[0] = ibooker.book1D("TEC3Full", "Full Tracker EndCap 3",220,0.,5.5);
  htmp[1] = ibooker.book1D("TEC3Fast", "Fast Tracker Endcap 3",220,0.,5.5);
  h100.push_back(htmp);

  // TEC4
  htmp[0] = ibooker.book1D("TEC4Full", "Full Tracker EndCap 4",220,0.,5.5);
  htmp[1] = ibooker.book1D("TEC4Fast", "Fast Tracker Endcap 4",220,0.,5.5);
  h100.push_back(htmp);

  // TEC5
  htmp[0] = ibooker.book1D("TEC5Full", "Full Tracker EndCap 5",220,0.,5.5);
  htmp[1] = ibooker.book1D("TEC5Fast", "Fast Tracker Endcap 5",220,0.,5.5);
  h100.push_back(htmp);

  // TEC6
  htmp[0] = ibooker.book1D("TEC6Full", "Full Tracker EndCap 6",220,0.,5.5);
  htmp[1] = ibooker.book1D("TEC6Fast", "Fast Tracker Endcap 6",220,0.,5.5);
  h100.push_back(htmp);

  // TEC7
  htmp[0] = ibooker.book1D("TEC7Full", "Full Tracker EndCap 7",220,0.,5.5);
  htmp[1] = ibooker.book1D("TEC7Fast", "Fast Tracker Endcap 7",220,0.,5.5);
  h100.push_back(htmp);

  // TEC8
  htmp[0] = ibooker.book1D("TEC8Full", "Full Tracker EndCap 8",220,0.,5.5);
  htmp[1] = ibooker.book1D("TEC8Fast", "Fast Tracker Endcap 8",220,0.,5.5);
  h100.push_back(htmp);

  // TEC9
  htmp[0] = ibooker.book1D("TEC9Full", "Full Tracker EndCap 9",220,0.,5.5);
  htmp[1] = ibooker.book1D("TEC9Fast", "Fast Tracker Endcap 9",220,0.,5.5);
  h100.push_back(htmp);

  // All TEC
  htmp[0] = ibooker.book1D("TECFull", "Full Tracker EndCap",220,0.,5.5);
  htmp[1] = ibooker.book1D("TECFast", "Fast Tracker EndCap",220,0.,5.5);
  h200.push_back(htmp);

  // All Outer 
  htmp[0] = ibooker.book1D("OuterFull", "Full Outer Tracker",220,0.,5.5);
  htmp[1] = ibooker.book1D("OuterFast", "Fast Outer Tracker",220,0.,5.5);
  h300.push_back(htmp);

  // Outer Cables
  htmp[0] = ibooker.book1D("TECCFull", "Full Tracker Outer Cables",220,0.,5.5);
  htmp[1] = ibooker.book1D("TECCFast", "Fast Tracker Outer Cables",220,0.,5.5);
  h100.push_back(htmp);

  // All TEC
  htmp[0] = ibooker.book1D("CablesFull", "Full Tracker Cables",220,0.,5.5);
  htmp[1] = ibooker.book1D("CablesFast", "Fast Tracker Cables",220,0.,5.5);
  h200.push_back(htmp);

  // All 
  htmp[0] = ibooker.book1D("TrackerFull", "Full Tracker",220,0.,5.5);
  htmp[1] = ibooker.book1D("TrackerFast", "Fast Tracker",220,0.,5.5);
  h300.push_back(htmp);

  // All Patrice : Finer granularity
  htmp[0] = ibooker.book1D("TrackerFull2", "Full Tracker 2",550,0.,5.5);
  htmp[1] = ibooker.book1D("TrackerFast2", "Fast Tracker 2",550,0.,5.5);
  h300.push_back(htmp);
}

void testMaterialEffects::dqmBeginRun(edm::Run const&, edm::EventSetup const& es)
{
  // init Particle data table (from Pythia)
  edm::ESHandle < HepPDT::ParticleDataTable > pdt;
  es.getData(pdt);
  mySimEvent[0]->initializePdt(&(*pdt));
  mySimEvent[1]->initializePdt(&(*pdt));

}

void
testMaterialEffects::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  ParticleTable::Sentry(mySimEvent[0]->theTable());

  if( ( nevt < 100 && nevt%10 == 0)   || 
      ( nevt < 1000 && nevt%100 == 0) || 
      nevt%1000 == 0 ) 
    std::cout<<"process entry "<< nevt << std::endl;
  nevt++; 

  //std::cout << "Fill full event " << std::endl;
  edm::Handle<std::vector<SimTrack> > fullSimTracks;
  iEvent.getByLabel("g4SimHits",fullSimTracks);
  edm::Handle<std::vector<SimVertex> > fullSimVertices;
  iEvent.getByLabel("g4SimHits",fullSimVertices);
  mySimEvent[0]->fill( *fullSimTracks, *fullSimVertices );
  
  //std::cout << "Fill fast event " << std::endl;
  edm::Handle<std::vector<SimTrack> > fastSimTracks;
  iEvent.getByLabel("famosSimHits",fastSimTracks);
  edm::Handle<std::vector<SimVertex> > fastSimVertices;
  iEvent.getByLabel("famosSimHits",fastSimVertices);
  mySimEvent[1]->fill( *fastSimTracks, *fastSimVertices );
  
  for ( unsigned ievt=0; ievt<2; ++ievt ) {

    //    std::cout << "Event number " << ievt << std::endl;
    //    mySimEvent[ievt]->print();

    int nvertices = mySimEvent[ievt]->nVertices();
    for(int i=0; i<nvertices; ++i) {
      FSimVertex& vertex = mySimEvent[ievt]->vertex(i);
      h0[ievt]->Fill(fabs(vertex.position().z()),vertex.position().pt());
    }
    
    // Loop over all tracks 
    int ntracks = mySimEvent[ievt]->nTracks();      
    
    for (int i=0;i<ntracks;++i) {
      
      FSimTrack& myTrack = mySimEvent[ievt]->track(i);
      std::vector<int> myGammas;

      // Select the original electrons
      if (abs(myTrack.type()) == 11 && myTrack.vertex().noParent())  {
	int firstDaughter = -1;
	int lastDaughter = -1;
	unsigned nbrems=0;
	unsigned nbremsmin=0;
	double feta=fabs(myTrack.momentum().eta());
	// Plot electron pseudo-rapidity
	h1[ievt]->Fill(feta);
	
	if ( myTrack.nDaughters() ) { 
	  firstDaughter = myTrack.daughters()[0];
	  lastDaughter = myTrack.daughters()[myTrack.nDaughters()-1];
	}
	
	XYZTLorentzVector theElectron=myTrack.momentum();
	//	std::cout << " The starting electron " << theElectron << " " 
	//		  << myTrack.vertex().position() << " " 
	//		  << myTrack.endVertex().position() << " "
	//		  << myTrack.nDaughters() << " " 
	//		  << firstDaughter << " " 
	//		  << lastDaughter << " " 
	//		  << std::endl;
	
	// Fill the photons.
	if(!(firstDaughter<0||lastDaughter<0)) {
	  
	  for(int igamma=firstDaughter;igamma<=lastDaughter;++igamma) {
	    FSimTrack myGamma = mySimEvent[ievt]->track(igamma);
	    if(myGamma.type()!=22) continue;
	    XYZTLorentzVector theFather=theElectron;
	    theElectron=theElectron-myGamma.momentum();
	    nbrems++;
	    if(myGamma.momentum().e() < 0.5 || 
//	       myGamma.momentum().e() > 10. || 
	       myGamma.momentum().e()/theElectron.e() < 0.005 ) continue;
	    nbremsmin++;
	    myGammas.push_back(igamma);

	    h2[ievt]->Fill(myGamma.momentum().e());
	    h3[ievt]->Fill(myGamma.momentum().e()/theFather.e());

	  }
	  h4[ievt]->Fill(nbrems);
	  h5[ievt]->Fill(nbremsmin);
	}
      } else {
	continue;
      }

      // Loop over all stored brems
      for(unsigned ig=0;ig<myGammas.size();++ig) {
	FSimTrack theGamma=mySimEvent[ievt]->track(myGammas[ig]);
	float radius = theGamma.vertex().position().pt();
	float zed    = fabs(theGamma.vertex().position().z());
	float eta    = fabs(theGamma.vertex().position().eta());

 	// Fill the individual layer histograms !
	bool filled = false;
	for ( unsigned hist=0; hist<h100.size() && !filled; ++hist ) {
	  if ( eta<5. && ( radius < trackerRadius[hist][ievt] && 
			   zed < trackerLength[hist][ievt] ) ) {
	    h100[hist][ievt]->Fill(eta);
	    filled = true;
	  }
	}
	if (!filled) h6[ievt]->Fill(zed,radius);

 	// Fill the block histograms !
	filled = false;
	for ( unsigned hist=0; hist<h200.size() && !filled; ++hist ) {
	  if ( eta<5. && ( radius < blockTrackerRadius[hist][ievt] && 
			   zed < blockTrackerLength[hist][ievt] ) ) {
	    h200[hist][ievt]->Fill(eta);
	    filled = true;
	  }
	}
	if (!filled) h7[ievt]->Fill(zed,radius);

        // Patrice
        if ( eta > 3.) {
          h13[ievt]->Fill(radius);
          h14[ievt]->Fill(radius);
          if ( eta <  3.61                 ) h15[ievt]->Fill(eta);
          if ( eta >= 3.61  && eta < 4.835 ) h16[ievt]->Fill(eta);
          if ( eta >= 4.835 && eta < 4.915 ) h17[ievt]->Fill(eta);
        }

 	// Fill the cumulative histograms !
	for ( unsigned hist=0; hist<h300.size(); ++hist ) {
	  if ( ( radius < subTrackerRadius[hist][ievt] && 
		 zed < subTrackerLength[hist][ievt] ) || 
	       ( hist == 2 && 
		 radius < subTrackerRadius[1][ievt] && 
		 zed < subTrackerLength[1][ievt] ) ) {
	    if ( ( hist <= 3 && eta < 5. ) || hist>=4 ) h300[hist][ievt]->Fill(eta);
	    if ( hist == 0 ) h8[ievt]->Fill(zed,radius);
	    if ( hist == 1 ) h9[ievt]->Fill(zed,radius);
	    if ( hist == 2 ) h10[ievt]->Fill(zed,radius);
	    if ( hist == 3 ) h11[ievt]->Fill(zed,radius);
	    if ( hist == 4 ) h12[ievt]->Fill(zed,radius);
	  }
	}

      }
    }
  }
}

//define this as a plug-in

DEFINE_FWK_MODULE(testMaterialEffects);
