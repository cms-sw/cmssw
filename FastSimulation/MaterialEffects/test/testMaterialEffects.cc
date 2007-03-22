// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

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
#include "FastSimulation/Utilities/interface/Histos.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <vector>
#include <string>
#include "TH2.h"
#include "TFile.h"
#include "TCanvas.h"

class testMaterialEffects : public edm::EDAnalyzer {
public :
  explicit testMaterialEffects(const edm::ParameterSet&);
  ~testMaterialEffects();

  virtual void analyze(const edm::Event&, const edm::EventSetup& );
  virtual void beginJob(const edm::EventSetup & c);
private:
  
  // See RecoParticleFlow/PFProducer/interface/PFProducer.h
  edm::ParameterSet particleFilter_;
  std::vector<FSimEvent*> mySimEvent;
  std::string simModuleLabel_;  
  DaqMonitorBEInterface * dbe;
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
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  h0[0] = dbe->book2D("radioFull", "Full Tracker radiography", 1000, 0.,320.,1000,0., 150. );
  h0[1] = dbe->book2D("radioFast", "Fast Tracker radiography", 1000, 0.,320.,1000,0., 150. );
  h1[0] = dbe->book1D("etaEFull", "Full Electron eta distribution",54,0.,2.7);
  h1[1] = dbe->book1D("etaEFast", "Fast Electron eta distribution",54,0.,2.7);
  h2[0] = dbe->book1D("EgammaFull", "Full Brem energy distribution",600,0.,300.);
  h2[1] = dbe->book1D("EgammaFast", "Fast Brem energy distribution",600,0.,300.);
  h3[0] = dbe->book1D("FEgammaFull", "Full Brem energy fraction distribution",1000,0.,1.);
  h3[1] = dbe->book1D("FEgammaFast", "Fast Brem energy fraction distribution",1000,0.,1.);
  h4[0] = dbe->book1D("NgammaFull", "Full Brem number",25,-0.5,24.5);
  h4[1] = dbe->book1D("NgammaFast", "Fast Brem number",25,-0.5,24.5);
  h5[0] = dbe->book1D("NgammaMinFull", "Full Brem number > Emin",25,-0.5,24.5);
  h5[1] = dbe->book1D("NgammaMinFast", "Fast Brem number > Emin",25,-0.5,24.5);
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

  // Beam Pipe
  htmp[0] = dbe->book1D("BeamPipeFull", "Full Beam Pipe",120,0.,3.);
  htmp[1] = dbe->book1D("BeamPipeFast", "Fast Beam Pipe",120,0.,3.);
  tmpRadius[0] = 3.8;
  tmpRadius[1] = 3.05;
  tmpLength[0] = 999.;
  tmpLength[1] = 27.;
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
  tmpRadius[0] = 6.0;
  tmpRadius[1] = 6.0;
  tmpLength[0] = 26.5;
  tmpLength[1] = 26.5;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // PIXB2
  htmp[0] = dbe->book1D("PXB2Full", "Full Pixel Barrel 2",120,0.,3.);
  htmp[1] = dbe->book1D("PXB2Fast", "Fast Pixel Barrel 2",120,0.,3.);
  tmpRadius[0] = 8.5;
  tmpRadius[1] = 8.5;
  tmpLength[0] = 26.5;
  tmpLength[1] = 26.5;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // PIXB3
  htmp[0] = dbe->book1D("PXB3Full", "Full Pixel Barrel 3",120,0.,3.);
  htmp[1] = dbe->book1D("PXB3Fast", "Fast Pixel Barrel 3",120,0.,3.);
  tmpRadius[0] = 11.5;
  tmpRadius[1] = 11.5;
  tmpLength[0] = 26.5;
  tmpLength[1] = 26.5;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // PIXB Cables
  htmp[0] = dbe->book1D("PXBCFull", "Full Pixel Barrel Cables",120,0.,3.);
  htmp[1] = dbe->book1D("PXBCFast", "Fast Pixel Barrel Cables",120,0.,3.);
  tmpRadius[0] = 18.0;
  tmpRadius[1] = 16.9;
  tmpLength[0] = 30.0;
  tmpLength[1] = 30.0;
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
  tmpRadius[0] = 15.5;
  tmpRadius[1] = 17.0;
  tmpLength[0] = 40.0;
  tmpLength[1] = 40.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // PIXD2
  htmp[0] = dbe->book1D("PXD2Full", "Full Pixel Disk 2",120,0.,3.);
  htmp[1] = dbe->book1D("PXD2Fast", "Fast Pixel Disk 2",120,0.,3.);
  tmpRadius[0] = 15.5;
  tmpRadius[1] = 17.0;
  tmpLength[0] = 55.0;
  tmpLength[1] = 50.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // PIXD Cables
  htmp[0] = dbe->book1D("PXDCFull", "Full Pixel Disk Cables",120,0.,3.);
  htmp[1] = dbe->book1D("PXDCFast", "Fast Pixel Disk Cables",120,0.,3.);
  tmpRadius[0] = 20.0;
  tmpRadius[1] = 18.0;
  tmpLength[0] = 999.0;
  tmpLength[1] = 999.0;
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
  tmpRadius[0] = 28.0;
  tmpRadius[1] = 28.0;
  tmpLength[0] = 70.0;
  tmpLength[1] = 73.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TIB2
  htmp[0] = dbe->book1D("TIB2Full", "Full Tracker Inner Barrel 2",120,0.,3.);
  htmp[1] = dbe->book1D("TIB2Fast", "Fast Tracker Inner Barrel 2",120,0.,3.);
  tmpRadius[0] = 37.0;
  tmpRadius[1] = 37.0;
  tmpLength[0] = 70.0;
  tmpLength[1] = 73.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TIB3
  htmp[0] = dbe->book1D("TIB3Full", "Full Tracker Inner Barrel 3",120,0.,3.);
  htmp[1] = dbe->book1D("TIB3Fast", "Fast Tracker Inner Barrel 3",120,0.,3.);
  tmpRadius[0] = 45.0;
  tmpRadius[1] = 45.0;
  tmpLength[0] = 70.0;
  tmpLength[1] = 73.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TIB4
  htmp[0] = dbe->book1D("TIB4Full", "Full Tracker Inner Barrel 4",120,0.,3.);
  htmp[1] = dbe->book1D("TIB4Fast", "Fast Tracker Inner Barrel 4",120,0.,3.);
  tmpRadius[0] = 52.0;
  tmpRadius[1] = 53.0;
  tmpLength[0] = 70.0;
  tmpLength[1] = 73.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TIB Cables
  htmp[0] = dbe->book1D("TIBCFull", "Full Tracker Inner Barrel Cables",120,0.,3.);
  htmp[1] = dbe->book1D("TIBCFast", "Fast Tracker Inner Barrel Cables",120,0.,3.);
  tmpRadius[0] = 53.0;
  tmpRadius[1] = 53.95;
  tmpLength[0] = 73.0;
  tmpLength[1] = 75.5;
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
  tmpRadius[0] = 52.0;
  tmpRadius[1] = 54.0;
  tmpLength[0] = 83.0;
  tmpLength[1] = 83.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TID2
  htmp[0] = dbe->book1D("TID2Full", "Full Tracker Inner Disk 2",120,0.,3.);
  htmp[1] = dbe->book1D("TID2Fast", "Fast Tracker Inner Disk 2",120,0.,3.);
  tmpRadius[0] = 52.0;
  tmpRadius[1] = 54.0;
  tmpLength[0] = 95.0;
  tmpLength[1] = 95.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TID3
  htmp[0] = dbe->book1D("TID3Full", "Full Tracker Inner Disk 3",120,0.,3.);
  htmp[1] = dbe->book1D("TID3Fast", "Fast Tracker Inner Disk 3",120,0.,3.);
  tmpRadius[0] = 52.0;
  tmpRadius[1] = 54.0;
  tmpLength[0] = 110.0;
  tmpLength[1] = 106.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TID Cables
  htmp[0] = dbe->book1D("TIDCFull", "Full Tracker Inner Disk Cables",120,0.,3.);
  htmp[1] = dbe->book1D("TIDCFast", "Fast Tracker Inner Disk Cables",120,0.,3.);
  tmpRadius[0] = 59.0;
  tmpRadius[1] = 55.0;
  tmpLength[0] = 122.0;
  tmpLength[1] = 109.0;
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
  tmpRadius[0] = 65.0;
  tmpRadius[1] = 65.0;
  tmpLength[0] = 109.0;
  tmpLength[1] = 109.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB2
  htmp[0] = dbe->book1D("TOB2Full", "Full Tracker Outer Barrel 2",120,0.,3.);
  htmp[1] = dbe->book1D("TOB2Fast", "Fast Tracker Outer Barrel 2",120,0.,3.);
  tmpRadius[0] = 75.0;
  tmpRadius[1] = 75.0;
  tmpLength[0] = 109.0;
  tmpLength[1] = 109.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB3
  htmp[0] = dbe->book1D("TOB3Full", "Full Tracker Outer Barrel 3",120,0.,3.);
  htmp[1] = dbe->book1D("TOB3Fast", "Fast Tracker Outer Barrel 3",120,0.,3.);
  tmpRadius[0] = 83.0;
  tmpRadius[1] = 83.0;
  tmpLength[0] = 109.0;
  tmpLength[1] = 109.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB4
  htmp[0] = dbe->book1D("TOB4Full", "Full Tracker Outer Barrel 4",120,0.,3.);
  htmp[1] = dbe->book1D("TOB4Fast", "Fast Tracker Outer Barrel 4",120,0.,3.);
  tmpRadius[0] = 92.0;
  tmpRadius[1] = 92.0;
  tmpLength[0] = 109.0;
  tmpLength[1] = 109.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB5
  htmp[0] = dbe->book1D("TOB5Full", "Full Tracker Outer Barrel 5",120,0.,3.);
  htmp[1] = dbe->book1D("TOB5Fast", "Fast Tracker Outer Barrel 5",120,0.,3.);
  tmpRadius[0] = 103.0;
  tmpRadius[1] = 103.0;
  tmpLength[0] = 109.0;
  tmpLength[1] = 109.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB6
  htmp[0] = dbe->book1D("TOB6Full", "Full Tracker Outer Barrel 6",120,0.,3.);
  htmp[1] = dbe->book1D("TOB6Fast", "Fast Tracker Outer Barrel 6",120,0.,3.);
  tmpRadius[0] = 113.0;
  tmpRadius[1] = 113.0;
  tmpLength[0] = 109.0;
  tmpLength[1] = 109.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TOB Cables
  htmp[0] = dbe->book1D("TOBCFull", "Full Tracker Outer Barrel Cables",120,0.,3.);
  htmp[1] = dbe->book1D("TOBCFast", "Fast Tracker Outer Barrel Cables",120,0.,3.);
  tmpRadius[0] = 110.0;
  tmpRadius[1] = 110.0;
  tmpLength[0] = 125.0;
  tmpLength[1] = 125.0;
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
  tmpRadius[0] = 110.0;
  tmpRadius[1] = 110.0;
  tmpLength[0] = 136.0;
  tmpLength[1] = 136.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC2
  htmp[0] = dbe->book1D("TEC2Full", "Full Tracker EndCap 2",120,0.,3.);
  htmp[1] = dbe->book1D("TEC2Fast", "Fast Tracker Endcap 2",120,0.,3.);
  tmpRadius[0] = 110.0;
  tmpRadius[1] = 110.0;
  tmpLength[0] = 150.0;
  tmpLength[1] = 150.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC3
  htmp[0] = dbe->book1D("TEC3Full", "Full Tracker EndCap 3",120,0.,3.);
  htmp[1] = dbe->book1D("TEC3Fast", "Fast Tracker Endcap 3",120,0.,3.);
  tmpRadius[0] = 110.0;
  tmpRadius[1] = 110.0;
  tmpLength[0] = 165.0;
  tmpLength[1] = 165.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC4
  htmp[0] = dbe->book1D("TEC4Full", "Full Tracker EndCap 4",120,0.,3.);
  htmp[1] = dbe->book1D("TEC4Fast", "Fast Tracker Endcap 4",120,0.,3.);
  tmpRadius[0] = 110.0;
  tmpRadius[1] = 110.0;
  tmpLength[0] = 180.0;
  tmpLength[1] = 180.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC5
  htmp[0] = dbe->book1D("TEC5Full", "Full Tracker EndCap 5",120,0.,3.);
  htmp[1] = dbe->book1D("TEC5Fast", "Fast Tracker Endcap 5",120,0.,3.);
  tmpRadius[0] = 110.0;
  tmpRadius[1] = 110.0;
  tmpLength[0] = 195.0;
  tmpLength[1] = 195.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC6
  htmp[0] = dbe->book1D("TEC6Full", "Full Tracker EndCap 6",120,0.,3.);
  htmp[1] = dbe->book1D("TEC6Fast", "Fast Tracker Endcap 6",120,0.,3.);
  tmpRadius[0] = 110.0;
  tmpRadius[1] = 110.0;
  tmpLength[0] = 211.0;
  tmpLength[1] = 211.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC7
  htmp[0] = dbe->book1D("TEC7Full", "Full Tracker EndCap 7",120,0.,3.);
  htmp[1] = dbe->book1D("TEC7Fast", "Fast Tracker Endcap 7",120,0.,3.);
  tmpRadius[0] = 110.0;
  tmpRadius[1] = 110.0;
  tmpLength[0] = 230.0;
  tmpLength[1] = 230.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC8
  htmp[0] = dbe->book1D("TEC8Full", "Full Tracker EndCap 8",120,0.,3.);
  htmp[1] = dbe->book1D("TEC8Fast", "Fast Tracker Endcap 8",120,0.,3.);
  tmpRadius[0] = 110.0;
  tmpRadius[1] = 110.0;
  tmpLength[0] = 252.0;
  tmpLength[1] = 252.0;
  h100.push_back(htmp);
  trackerRadius.push_back(tmpRadius);
  trackerLength.push_back(tmpLength);

  // TEC9
  htmp[0] = dbe->book1D("TEC9Full", "Full Tracker EndCap 9",120,0.,3.);
  htmp[1] = dbe->book1D("TEC9Fast", "Fast Tracker Endcap 9",120,0.,3.);
  tmpRadius[0] = 110.0;
  tmpRadius[1] = 110.0;
  tmpLength[0] = 272.0;
  tmpLength[1] = 272.0;
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
  tmpRadius[0] = 125.0;
  tmpRadius[1] = 121.0;
  tmpLength[0] = 301.0;
  tmpLength[1] = 301.0;
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

testMaterialEffects::~testMaterialEffects()
{
  dbe->save("test.root");
  //  delete mySimEvent;
}

void testMaterialEffects::beginJob(const edm::EventSetup & es)
{
  // init Particle data table (from Pythia)
  edm::ESHandle < HepPDT::ParticleDataTable > pdt;
  es.getData(pdt);
  if ( !ParticleTable::instance() ) ParticleTable::instance(&(*pdt));
  mySimEvent[0]->initializePdt(&(*pdt));
  mySimEvent[1]->initializePdt(&(*pdt));

}

void
testMaterialEffects::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  //  std::cout << "Fill full event " << std::endl;
  edm::Handle<std::vector<SimTrack> > fullSimTracks;
  iEvent.getByLabel("g4SimHits",fullSimTracks);
  edm::Handle<std::vector<SimVertex> > fullSimVertices;
  iEvent.getByLabel("g4SimHits",fullSimVertices);
  mySimEvent[0]->fill( *fullSimTracks, *fullSimVertices );
  
  //  std::cout << "Fill full event " << std::endl;
  edm::Handle<std::vector<SimTrack> > fastSimTracks;
  iEvent.getByLabel("famosSimHits",fastSimTracks);
  edm::Handle<std::vector<SimVertex> > fastSimVertices;
  iEvent.getByLabel("famosSimHits",fastSimVertices);
  mySimEvent[1]->fill( *fastSimTracks, *fastSimVertices );
  
  for ( unsigned ievt=0; ievt<2; ++ievt ) {

    //    std::cout << "Event number " << ievt << std::endl;
    //    mySimEvent[ievt]->print();

    const std::vector<FSimVertex>& fsimVertices = *(mySimEvent[ievt]->vertices() );
    for(unsigned i=0; i<fsimVertices.size(); i++) {
      h0[ievt]->Fill(fabs(fsimVertices[i].position().z()),
	    	          fsimVertices[i].position().perp());
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
	
	HepLorentzVector theElectron=myTrack.momentum();
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
	    HepLorentzVector theFather=theElectron;
	    theElectron=theElectron-myGamma.momentum();
	    nbrems++;
	    if(myGamma.momentum().e() < 0.5 || 
//	       myGamma.momentum().e() > 10. || 
	       myGamma.momentum().e()/theElectron.e() < 0.005 ) continue;
	    nbremsmin++;
	    myGammas.push_back(igamma);

	    h2[ievt]->Fill(myGamma.momentum().e());
	    h3[ievt]->Fill(myGamma.momentum().e()/theElectron.e());

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
	float radius = theGamma.vertex().position().perp();
	float zed    = fabs(theGamma.vertex().position().z());
	float eta    = fabs(theGamma.vertex().position().eta());

 	// Fill the individual layer histograms !
	bool filled = false;
	for ( unsigned hist=0; hist<h100.size() && !filled; ++hist ) {
	  if ( radius < trackerRadius[hist][ievt] && 
	       zed < trackerLength[hist][ievt] ) {
	    h100[hist][ievt]->Fill(eta);
	    filled = true;
	  }
	}
	if (!filled) h6[ievt]->Fill(zed,radius);

 	// Fill the block histograms !
	filled = false;
	for ( unsigned hist=0; hist<h200.size() && !filled; ++hist ) {
	  if ( radius < blockTrackerRadius[hist][ievt] && 
	       zed < blockTrackerLength[hist][ievt] ) {
	    h200[hist][ievt]->Fill(eta);
	    filled = true;
	  }
	}
	if (!filled) h7[ievt]->Fill(zed,radius);

 	// Fill the cumulative histograms !
	for ( unsigned hist=0; hist<h300.size(); ++hist ) {
	  if ( ( radius < subTrackerRadius[hist][ievt] && 
		 zed < subTrackerLength[hist][ievt] ) || 
	       ( hist == 2 && 
		 radius < subTrackerRadius[1][ievt] && 
		 zed < subTrackerLength[1][ievt] ) ) {
	    h300[hist][ievt]->Fill(eta);
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
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(testMaterialEffects);
