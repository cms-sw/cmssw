#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CalibTracker/SiStripLorentzAngle/interface/SiStripCalibLorentzAngle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CLHEP/Random/RandGauss.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

SiStripCalibLorentzAngle::SiStripCalibLorentzAngle(edm::ParameterSet const& conf) : ConditionDBWriter<SiStripLorentzAngle>(conf) , tTopo(nullptr), conf_(conf) {}

void SiStripCalibLorentzAngle::algoBeginJob(const edm::EventSetup& c){
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  c.get<IdealGeometryRecord>().get(tTopoHandle);
  tTopo = tTopoHandle.product();

  c.get<TrackerDigiGeometryRecord>().get(estracker);
  tracker=&(*estracker); 
  
  //get magnetic field and geometry from ES
  edm::ESHandle<MagneticField> magfield_;
  c.get<IdealMagneticFieldRecord>().get(magfield_);

  edm::ESHandle<SiStripLorentzAngle> SiStripLorentzAngle_;
  c.get<SiStripLorentzAngleRcd>().get(SiStripLorentzAngle_);
  detid_la= SiStripLorentzAngle_->getLorentzAngles();
  
  DQMStore* dbe_ = edm::Service<DQMStore>().operator->();
  
  std::string inputFile_ =conf_.getUntrackedParameter<std::string>("fileName", "LAProfiles.root");
  std::string LAreport_ =conf_.getUntrackedParameter<std::string>("LA_Report", "LA_Report.txt");
  std::string NoEntriesHisto_ =conf_.getUntrackedParameter<std::string>("NoEntriesHisto", "NoEntriesHisto.txt");
  std::string Dir_Name_ =conf_.getUntrackedParameter<std::string>("Dir_Name", "SiStrip");
  
  LayerDB = conf_.getUntrackedParameter<bool>("LayerDB", false);
  
  CalibByMC = conf_.getUntrackedParameter<bool>("CalibByMC", false);
  
  dbe_->open(inputFile_);
  
  // use SistripHistoId for producing histogram id (and title)
  SiStripHistoId hidmanager;
  
  edm::LogInfo("SiStripCalibLorentzAngle")<<"### DIR-NAME = "<<Dir_Name_;
  histolist= dbe_->getAllContents(Dir_Name_);
  std::vector<MonitorElement*>::iterator histo;
  
  hFile = new TFile (conf_.getUntrackedParameter<std::string>("out_fileName").c_str(), "RECREATE" );
    
  LorentzAngle_Plots = hFile->mkdir("LorentzAngle_Plots");
  Rootple = LorentzAngle_Plots->mkdir("Rootple");
  MuH = LorentzAngle_Plots->mkdir("MuH");
  TIB_MuH = MuH->mkdir("TIB_MuH");
  TOB_MuH = MuH->mkdir("TOB_MuH");
  MuH_vs_Phi = LorentzAngle_Plots->mkdir("MuH_vs_Phi");
  TIB_Phi = MuH_vs_Phi->mkdir("TIB_Phi");
  TOB_Phi = MuH_vs_Phi->mkdir("TOB_Phi");
  MuH_vs_Eta = LorentzAngle_Plots->mkdir("MuH_vs_Eta");
  TIB_Eta = MuH_vs_Eta->mkdir("TIB_Eta");
  TOB_Eta = MuH_vs_Eta->mkdir("TOB_Eta");
  FirstIT_GoodFit_Histos = LorentzAngle_Plots->mkdir("1IT_GoodFit_Histos");
  TIB_1IT_GoodFit = FirstIT_GoodFit_Histos->mkdir("TIB_1IT_GoodFit");
  TOB_1IT_GoodFit = FirstIT_GoodFit_Histos->mkdir("TOB_1IT_GoodFit");
  SecondIT_GoodFit_Histos = LorentzAngle_Plots->mkdir("2IT_GoodFit_Histos");
  TIB_2IT_GoodFit = SecondIT_GoodFit_Histos->mkdir("TIB_2IT_GoodFit");
  TOB_2IT_GoodFit = SecondIT_GoodFit_Histos->mkdir("TOB_2IT_GoodFit");
  SecondIT_BadFit_Histos = LorentzAngle_Plots->mkdir("2IT_BadFit_Histos");
  TIB_2IT_BadFit = SecondIT_BadFit_Histos->mkdir("TIB_2IT_BadFit");
  TOB_2IT_BadFit = SecondIT_BadFit_Histos->mkdir("TOB_2IT_BadFit");
    
  TH1Ds["LA_TIB"] = new TH1D("TanLAPerTesla TIB","TanLAPerTesla TIB",1000,-0.5,0.5);
  TH1Ds["LA_TIB"]->SetDirectory(MuH);
  TH1Ds["LA_TOB"] = new TH1D("TanLAPerTesla TOB","TanLAPerTesla TOB",1000,-0.5,0.5);
  TH1Ds["LA_TOB"]->SetDirectory(MuH);
  TH1Ds["LA_err_TIB"] = new TH1D("TanLAPerTesla Error TIB","TanLAPerTesla Error TIB",1000,0,1);
  TH1Ds["LA_err_TIB"]->SetDirectory(MuH);
  TH1Ds["LA_err_TOB"] = new TH1D("TanLAPerTesla Error TOB","TanLAPerTesla Error TOB",1000,0,1);
  TH1Ds["LA_err_TOB"]->SetDirectory(MuH);
  TH1Ds["LA_chi2norm_TIB"] = new TH1D("TanLAPerTesla Chi2norm TIB","TanLAPerTesla Chi2norm TIB",2000,0,10);
  TH1Ds["LA_chi2norm_TIB"]->SetDirectory(MuH);
  TH1Ds["LA_chi2norm_TOB"] = new TH1D("TanLAPerTesla Chi2norm TOB","TanLAPerTesla Chi2norm TOB",2000,0,10);
  TH1Ds["LA_chi2norm_TOB"]->SetDirectory(MuH);
  TH1Ds["MagneticField"] = new TH1D("MagneticField","MagneticField",500,0,5);
  TH1Ds["MagneticField"]->SetDirectory(MuH);
  
  TH2Ds["LA_TIB_graph"] = new TH2D("TanLAPerTesla TIB Layers","TanLAPerTesla TIB Layers",60,0,5,1000,-0.3,0.3);
  TH2Ds["LA_TIB_graph"]->SetDirectory(MuH);
  TH2Ds["LA_TIB_graph"]->SetNdivisions(6);
  TH2Ds["LA_TOB_graph"] = new TH2D("TanLAPerTesla TOB Layers","TanLAPerTesla TOB Layers",80,0,7,1000,-0.3,0.3);
  TH2Ds["LA_TOB_graph"]->SetDirectory(MuH);
  TH2Ds["LA_TOB_graph"]->SetNdivisions(8);

  TH1Ds["LA_TIB_1"] = new TH1D("TanLAPerTesla TIB1","TanLAPerTesla TIB1",2000,-0.5,0.5);
  TH1Ds["LA_TIB_1"]->SetDirectory(TIB_MuH);
  TH1Ds["LA_TIB_1_mono"] = new TH1D("TanLAPerTesla TIB1 MONO","TanLAPerTesla TIB1 MONO",2000,-0.5,0.5);
  TH1Ds["LA_TIB_1_mono"]->SetDirectory(TIB_MuH);
  TH1Ds["LA_TIB_1_stereo"] = new TH1D("TanLAPerTesla TIB1 STEREO","TanLAPerTesla TIB1 STEREO",2000,-0.5,0.5);
  TH1Ds["LA_TIB_1_stereo"]->SetDirectory(TIB_MuH);
  TH1Ds["LA_TIB_2"] = new TH1D("TanLAPerTesla TIB2","TanLAPerTesla TIB2",2000,-0.5,0.5);
  TH1Ds["LA_TIB_2"]->SetDirectory(TIB_MuH);
  TH1Ds["LA_TIB_2_mono"] = new TH1D("TanLAPerTesla TIB2 MONO","TanLAPerTesla TIB2 MONO",2000,-0.5,0.5);
  TH1Ds["LA_TIB_2_mono"]->SetDirectory(TIB_MuH);
  TH1Ds["LA_TIB_2_stereo"] = new TH1D("TanLAPerTesla TIB2 STEREO","TanLAPerTesla TIB2 STEREO",2000,-0.5,0.5);
  TH1Ds["LA_TIB_2_stereo"]->SetDirectory(TIB_MuH);
  TH1Ds["LA_TIB_3"] = new TH1D("TanLAPerTesla_TIB 3","TanLAPerTesla TIB3",2000,-0.5,0.5);
  TH1Ds["LA_TIB_3"]->SetDirectory(TIB_MuH);
  TH1Ds["LA_TIB_4"] = new TH1D("TanLAPerTesla_TIB 4","TanLAPerTesla TIB4",2000,-0.5,0.5);
  TH1Ds["LA_TIB_4"]->SetDirectory(TIB_MuH);
  
  TH1Ds["LA_TOB_1"] = new TH1D("TanLAPerTesla TOB1","TanLAPerTesla TOB1",2000,-0.5,0.5);
  TH1Ds["LA_TOB_1"]->SetDirectory(TOB_MuH);
  TH1Ds["LA_TOB_1_mono"] = new TH1D("TanLAPerTesla TOB1 MONO","TanLAPerTesla TOB1 MONO",2000,-0.5,0.5);
  TH1Ds["LA_TOB_1_mono"]->SetDirectory(TOB_MuH);
  TH1Ds["LA_TOB_1_stereo"] = new TH1D("TanLAPerTesla TOB1 STEREO","TanLAPerTesla TOB1 STEREO",2000,-0.5,0.5);
  TH1Ds["LA_TOB_1_stereo"]->SetDirectory(TOB_MuH);
  TH1Ds["LA_TOB_2"] = new TH1D("TanLAPerTesla TOB2","TanLAPerTesla TOB2",2000,-0.5,0.5);
  TH1Ds["LA_TOB_2"]->SetDirectory(TOB_MuH);
  TH1Ds["LA_TOB_2_mono"] = new TH1D("TanLAPerTesla TOB2 MONO","TanLAPerTesla TOB2 MONO",2000,-0.5,0.5);
  TH1Ds["LA_TOB_2_mono"]->SetDirectory(TOB_MuH);
  TH1Ds["LA_TOB_2_stereo"] = new TH1D("TanLAPerTesla TOB2 STEREO","TanLAPerTesla TOB2 STEREO",2000,-0.5,0.5);
  TH1Ds["LA_TOB_2_stereo"]->SetDirectory(TOB_MuH);
  TH1Ds["LA_TOB_3"] = new TH1D("TanLAPerTesla TOB3","TanLAPerTesla TOB3",2000,-0.5,0.5);
  TH1Ds["LA_TOB_3"]->SetDirectory(TOB_MuH);
  TH1Ds["LA_TOB_4"] = new TH1D("TanLAPerTesla TOB4","TanLAPerTesla TOB4",2000,-0.5,0.5);
  TH1Ds["LA_TOB_4"]->SetDirectory(TOB_MuH);
  TH1Ds["LA_TOB_5"] = new TH1D("TanLAPerTesla TOB5","TanLAPerTesla TOB5",2000,-0.5,0.5);
  TH1Ds["LA_TOB_5"]->SetDirectory(TOB_MuH);
  TH1Ds["LA_TOB_6"] = new TH1D("TanLAPerTesla TOB6","TanLAPerTesla TOB6",2000,-0.5,0.5); 
  TH1Ds["LA_TOB_6"]->SetDirectory(TOB_MuH);
    
  TH2Ds["LA_phi_TIB"] = new TH2D("TanLAPerTesla vs Phi TIB","TanLAPerTesla vs Phi TIB",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TIB"]->SetDirectory(MuH_vs_Phi);
  TH2Ds["LA_phi_TOB"] = new TH2D("TanLAPerTesla vs Phi TOB","TanLAPerTesla vs Phi TOB",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TOB"]->SetDirectory(MuH_vs_Phi);
  
  TH2Ds["LA_phi_TIB1"] = new TH2D("TanLAPerTesla vs Phi TIB1","TanLAPerTesla vs Phi TIB1",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TIB1"]->SetDirectory(TIB_Phi);
  TH2Ds["LA_phi_TIB1_mono"] = new TH2D("TanLAPerTesla vs Phi TIB1 MONO","TanLAPerTesla vs Phi TIB1 MONO",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TIB1_mono"]->SetDirectory(TIB_Phi);
  TH2Ds["LA_phi_TIB1_stereo"] = new TH2D("TanLAPerTesla vs Phi TIB1 STEREO","TanLAPerTesla vs Phi TIB1 STEREO",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TIB1_stereo"]->SetDirectory(TIB_Phi);
  TH2Ds["LA_phi_TIB2"] = new TH2D("TanLAPerTesla vs Phi TIB2","TanLAPerTesla vs Phi TIB2",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TIB2"]->SetDirectory(TIB_Phi);
  TH2Ds["LA_phi_TIB2_mono"] = new TH2D("TanLAPerTesla vs Phi TIB2 MONO","TanLAPerTesla vs Phi TIB2 MONO",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TIB2_mono"]->SetDirectory(TIB_Phi);
  TH2Ds["LA_phi_TIB2_stereo"] = new TH2D("TanLAPerTesla vs Phi TIB2 STEREO","TanLAPerTesla vs Phi TIB2 STEREO",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TIB2_stereo"]->SetDirectory(TIB_Phi);
  TH2Ds["LA_phi_TIB3"] = new TH2D("TanLAPerTesla vs Phi TIB3","TanLAPerTesla vs Phi TIB3",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TIB3"]->SetDirectory(TIB_Phi);
  TH2Ds["LA_phi_TIB4"] = new TH2D("TanLAPerTesla vs Phi TIB4","TanLAPerTesla vs Phi TIB4",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TIB4"]->SetDirectory(TIB_Phi);
  
  TH2Ds["LA_phi_TOB1"] = new TH2D("TanLAPerTesla vs Phi TOB1","TanLAPerTesla vs Phi TOB1",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TOB1"]->SetDirectory(TOB_Phi);
  TH2Ds["LA_phi_TOB1_mono"] = new TH2D("TanLAPerTesla vs Phi TOB1 MONO","TanLAPerTesla vs Phi TOB1 MONO",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TOB1_mono"]->SetDirectory(TOB_Phi);
  TH2Ds["LA_phi_TOB1_stereo"] = new TH2D("TanLAPerTesla vs Phi TOB1 STEREO","TanLAPerTesla vs Phi TOB1 STEREO",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TOB1_stereo"]->SetDirectory(TOB_Phi);
  TH2Ds["LA_phi_TOB2"] = new TH2D("TanLAPerTesla vs Phi TOB2","TanLAPerTesla vs Phi TOB2",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TOB2"]->SetDirectory(TOB_Phi);
  TH2Ds["LA_phi_TOB2_mono"] = new TH2D("TanLAPerTesla vs Phi TOB2 MONO","TanLAPerTesla vs Phi TOB2 MONO",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TOB2_mono"]->SetDirectory(TOB_Phi);
  TH2Ds["LA_phi_TOB2_stereo"] = new TH2D("TanLAPerTesla vs Phi TOB2 STEREO","TanLAPerTesla vs Phi TOB2 STEREO",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TOB2_stereo"]->SetDirectory(TOB_Phi);
  TH2Ds["LA_phi_TOB3"] = new TH2D("TanLAPerTesla vs Phi TOB3","TanLAPerTesla vs Phi TOB3",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TOB3"]->SetDirectory(TOB_Phi);
  TH2Ds["LA_phi_TOB4"] = new TH2D("TanLAPerTesla vs Phi TOB4","TanLAPerTesla vs Phi TOB4",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TOB4"]->SetDirectory(TOB_Phi);
  TH2Ds["LA_phi_TOB5"] = new TH2D("TanLAPerTesla vs Phi TOB5","TanLAPerTesla vs Phi TOB5",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TOB5"]->SetDirectory(TOB_Phi);
  TH2Ds["LA_phi_TOB6"] = new TH2D("TanLAPerTesla vs Phi TOB6","TanLAPerTesla vs Phi TOB6",800,-4,4,600,-0.3,0.3);
  TH2Ds["LA_phi_TOB6"]->SetDirectory(TOB_Phi);
  
  TH2Ds["LA_eta_TIB"] = new TH2D("TanLAPerTesla vs Eta TIB","TanLAPerTesla vs Eta TIB",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TIB"]->SetDirectory(MuH_vs_Eta);
  TH2Ds["LA_eta_TOB"] = new TH2D("TanLAPerTesla vs Eta TOB","TanLAPerTesla vs Eta TOB",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TOB"]->SetDirectory(MuH_vs_Eta);
  
  TH2Ds["LA_eta_TIB1"] = new TH2D("TanLAPerTesla vs Eta TIB1","TanLAPerTesla vs Eta TIB1",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TIB1"]->SetDirectory(TIB_Eta);
  TH2Ds["LA_eta_TIB1_mono"] = new TH2D("TanLAPerTesla vs Eta TIB1 MONO","TanLAPerTesla vs Eta TIB1 MONO",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TIB1_mono"]->SetDirectory(TIB_Eta);
  TH2Ds["LA_eta_TIB1_stereo"] = new TH2D("TanLAPerTesla vs Eta TIB1 STEREO","TanLAPerTesla vs Eta TIB1 STEREO",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TIB1_stereo"]->SetDirectory(TIB_Eta);
  TH2Ds["LA_eta_TIB2"] = new TH2D("TanLAPerTesla vs Eta TIB2","TanLAPerTesla vs Eta TIB2",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TIB2"]->SetDirectory(TIB_Eta);
  TH2Ds["LA_eta_TIB2_mono"] = new TH2D("TanLAPerTesla vs Eta TIB2 MONO","TanLAPerTesla vs Eta TIB2 MONO",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TIB2_mono"]->SetDirectory(TIB_Eta);
  TH2Ds["LA_eta_TIB2_stereo"] = new TH2D("TanLAPerTesla vs Eta TIB2 STEREO","TanLAPerTesla vs Eta TIB2 STEREO",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TIB2_stereo"]->SetDirectory(TIB_Eta);
  TH2Ds["LA_eta_TIB3"] = new TH2D("TanLAPerTesla vs Eta TIB3","TanLAPerTesla vs Eta TIB3",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TIB3"]->SetDirectory(TIB_Eta);
  TH2Ds["LA_eta_TIB4"] = new TH2D("TanLAPerTesla vs Eta TIB4","TanLAPerTesla vs Eta TIB4",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TIB4"]->SetDirectory(TIB_Eta);
  
  TH2Ds["LA_eta_TOB1"] = new TH2D("TanLAPerTesla vs Eta TOB1","TanLAPerTesla vs Eta TOB1",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TOB1"]->SetDirectory(TIB_Eta);
  TH2Ds["LA_eta_TOB1_mono"] = new TH2D("TanLAPerTesla vs Eta TOB1 MONO","TanLAPerTesla vs Eta TOB1 MONO",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TOB1_mono"]->SetDirectory(TIB_Eta);
  TH2Ds["LA_eta_TOB1_stereo"] = new TH2D("TanLAPerTesla vs Eta TOB1 STEREO","TanLAPerTesla vs Eta TOB1 STEREO",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TOB1_stereo"]->SetDirectory(TIB_Eta);
  TH2Ds["LA_eta_TOB2"] = new TH2D("TanLAPerTesla vs Eta TOB2","TanLAPerTesla vs Eta TOB2",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TOB2"]->SetDirectory(TIB_Eta);
  TH2Ds["LA_eta_TOB2_mono"] = new TH2D("TanLAPerTesla vs Eta TOB2 MONO","TanLAPerTesla vs Eta TOB2 MONO",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TOB2_mono"]->SetDirectory(TIB_Eta);
  TH2Ds["LA_eta_TOB2_stereo"] = new TH2D("TanLAPerTesla vs Eta TOB2 STEREO","TanLAPerTesla vs Eta TOB2 STEREO",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TOB2_stereo"]->SetDirectory(TIB_Eta);
  TH2Ds["LA_eta_TOB3"] = new TH2D("TanLAPerTesla vs Eta TOB3","TanLAPerTesla vs Eta TOB3",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TOB3"]->SetDirectory(TIB_Eta);
  TH2Ds["LA_eta_TOB4"] = new TH2D("TanLAPerTesla vs Eta TOB4","TanLAPerTesla vs Eta TOB4",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TOB4"]->SetDirectory(TIB_Eta);
  TH2Ds["LA_eta_TOB5"] = new TH2D("TanLAPerTesla vs Eta TOB5","TanLAPerTesla vs Eta TOB5",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TOB5"]->SetDirectory(TIB_Eta);
  TH2Ds["LA_eta_TOB6"] = new TH2D("TanLAPerTesla vs Eta TOB6","TanLAPerTesla vs Eta TOB6",800,-2.6,2.6,600,-0.3,0.3);
  TH2Ds["LA_eta_TOB6"]->SetDirectory(TIB_Eta);
  
  ModuleTree = new TTree("ModuleTree", "ModuleTree");
  ModuleTree->Branch("histoEntries", &histoEntries, "histoEntries/F");
  ModuleTree->Branch("globalX", &globalX, "globalX/F");
  ModuleTree->Branch("globalY", &globalY, "globalY/F");
  ModuleTree->Branch("globalZ", &globalZ, "globalZ/F");
  ModuleTree->Branch("gphi", &gphi, "gphi/F");
  ModuleTree->Branch("geta", &geta, "geta/F");
  ModuleTree->Branch("gR", &gR, "gR/F");
  ModuleTree->Branch("goodFit", &goodFit, "goodFit/I");
  ModuleTree->Branch("goodFit1IT", &goodFit1IT, "goodFit1IT/I");
  ModuleTree->Branch("badFit", &badFit, "badFit/I");
  ModuleTree->Branch("TIB", &TIB, "TIB/I");
  ModuleTree->Branch("TOB", &TOB, "TOB/I");
  ModuleTree->Branch("Layer", &Layer, "Layer/I");
  ModuleTree->Branch("MonoStereo", &MonoStereo, "MonoStereo/I");
  ModuleTree->Branch("theBfield", &theBfield, "theBfield/F");
  ModuleTree->Branch("muH", &muH, "muH/F");
  
  ModuleTree->SetDirectory(Rootple);
     
  int histocounter = 0;
  int NotEnoughEntries = 0;
  int ZeroEntries = 0;
  int GoodFit = 0;
  int FirstIT_goodfit = 0;
  int FirstIT_badfit = 0;
  int SecondIT_badfit = 0;
  int SecondIT_goodfit = 0;
  int no_mod_histo = 0;
  float chi2norm = 0;
  LocalPoint p =LocalPoint(0,0,0);
  
  double ModuleRangeMin=conf_.getParameter<double>("ModuleFitXMin");
  double ModuleRangeMax=conf_.getParameter<double>("ModuleFitXMax");
  double ModuleRangeMin2IT=conf_.getParameter<double>("ModuleFit2ITXMin");
  double ModuleRangeMax2IT=conf_.getParameter<double>("ModuleFit2ITXMax");
  double FitCuts_Entries=conf_.getParameter<double>("FitCuts_Entries");
  double FitCuts_p0=conf_.getParameter<double>("FitCuts_p0");
  double FitCuts_p1=conf_.getParameter<double>("FitCuts_p1");
  double FitCuts_p2=conf_.getParameter<double>("FitCuts_p2");
  double FitCuts_chi2=conf_.getParameter<double>("FitCuts_chi2");
  double FitCuts_ParErr_p0=conf_.getParameter<double>("FitCuts_ParErr_p0");
  double p0_guess=conf_.getParameter<double>("p0_guess");
  double p1_guess=conf_.getParameter<double>("p1_guess");
  double p2_guess=conf_.getParameter<double>("p2_guess");
  
  double TIB1calib = 1.;
  double TIB2calib = 1.;
  double TIB3calib = 1.;
  double TIB4calib = 1.;
  double TOB1calib = 1.;
  double TOB2calib = 1.;
  double TOB3calib = 1.;
  double TOB4calib = 1.;
  double TOB5calib = 1.;
  double TOB6calib = 1.;
  
  if(CalibByMC==true){
  //Calibration factors evaluated by using MC analysis
  TIB1calib=conf_.getParameter<double>("TIB1calib");
  TIB2calib=conf_.getParameter<double>("TIB2calib");
  TIB3calib=conf_.getParameter<double>("TIB3calib");
  TIB4calib=conf_.getParameter<double>("TIB4calib");
  TOB1calib=conf_.getParameter<double>("TOB1calib");
  TOB2calib=conf_.getParameter<double>("TOB2calib");
  TOB3calib=conf_.getParameter<double>("TOB3calib");
  TOB4calib=conf_.getParameter<double>("TOB4calib");
  TOB5calib=conf_.getParameter<double>("TOB5calib");
  TOB6calib=conf_.getParameter<double>("TOB6calib");
  }
  
  
  TF1 *fitfunc= new TF1("fitfunc","([4]/[3])*[1]*(TMath::Abs(x-[0]))+[2]",-1,1);
  TF1 *fitfunc2IT= new TF1("fitfunc2IT","([4]/[3])*[1]*(TMath::Abs(x-[0]))+[2]",-1,1);
 
  ofstream NoEntries;
  NoEntries.open(NoEntriesHisto_.c_str());
  ofstream Rep;
  Rep.open(LAreport_.c_str());
  
  gStyle->SetOptStat(1110);
  
  for(histo=histolist.begin();histo!=histolist.end();++histo){
  
  FitFunction = 0;
  FitFunction2IT = 0;
  bool Good2ITFit = false;
  bool ModuleHisto = true;
  
  histoEntries = -99;
  gphi=-99;
  geta=-99;
  gz = -99;
  gR=-1;
  globalX = -99;
  globalY = -99;
  globalZ = -99;
  goodFit = 0;
  goodFit1IT = 0;
  badFit = 0;
  muH = -1;
  TIB = 0;
  TOB = 0;
  MonoStereo = -1;
  
    uint32_t id=hidmanager.getComponentId((*histo)->getName());
    DetId detid(id);
    StripSubdetector subid(id);
    const GeomDetUnit * stripdet;
    MonoStereo = subid.stereo();
    
    if(!(stripdet=tracker->idToDetUnit(subid))){
    no_mod_histo++;
    ModuleHisto=false;
    edm::LogInfo("SiStripCalibLorentzAngle")<<"### NO MODULE HISTOGRAM";}
    
    if(stripdet!=0 && ModuleHisto==true){
    
      if(subid.subdetId() == int (StripSubdetector::TIB)){
      
      Layer = tTopo->tibLayer(detid);
      TIB = 1;}     
      if(subid.subdetId() == int (StripSubdetector::TOB)){
      
      Layer = tTopo->tobLayer(detid);
      TOB = 1;}
      
    //get module coordinates
    const GlobalPoint gposition = (stripdet->surface()).toGlobal(p);
    histoEntries = (*histo)->getEntries();
    globalX = gposition.x();
    globalY = gposition.y();
    globalZ = gposition.z();
    gphi = gposition.phi();
    geta = gposition.eta();
    gR = sqrt(pow(gposition.x(),2)+pow(gposition.y(),2));
    gz = gposition.z();
    
    //get magnetic field
    const StripGeomDetUnit* det = dynamic_cast<const StripGeomDetUnit*>(estracker->idToDetUnit(detid));
    if (det==0){
    edm::LogError("SiStripCalibLorentzAngle") << "[SiStripCalibLorentzAngle::getNewObject] the detID " << id << " doesn't seem to belong to Tracker" <<std::endl; 	
    continue;
    }
    LocalVector lbfield=(det->surface()).toLocal(magfield_->inTesla(det->surface().position()));
    theBfield = lbfield.mag();
    theBfield = (theBfield > 0) ? theBfield : 0.00001; 
    TH1Ds["MagneticField"]->Fill(theBfield);    
    }
    if(stripdet==0)continue;
      
    if(((*histo)->getEntries()<=FitCuts_Entries)&&ModuleHisto==true){
    if(((*histo)->getEntries()==0)&&ModuleHisto==true){    
    NoEntries<<"NO ENTRIES MODULE, ID = "<<id<<std::endl;    
    edm::LogInfo("SiStripCalibLorentzAngle")<<"### HISTOGRAM WITH 0 ENTRIES => TYPE:"<<subid.subdetId();
    ZeroEntries++;
    }else{    
    edm::LogInfo("SiStripCalibLorentzAngle")<<"### HISTOGRAM WITH NR. ENTRIES <= ENTRIES_CUT => TYPE:"<<subid.subdetId();
    NotEnoughEntries++;}    
    }
    
      std::string name;
      if(TIB==1){
      name+="TIB";
      }else{
      name+="TOB";}
      std::stringstream LayerStream;
      LayerStream<<Layer;
      name+=LayerStream.str();    
      std::stringstream idnum;
      idnum<<id;
      name+="_Id_";
      name+=idnum.str();
      
    gStyle->SetOptFit(111);  
    
    //Extract TProfile from Monitor Element to ProfileMap
    Profiles[name.c_str()] = new TProfile;   
    TProfile* theProfile=ExtractTObject<TProfile>().extract(*histo);   
    theProfile->Copy(*Profiles[name.c_str()]);    
    Profiles[name.c_str()]->SetName(name.c_str());
  
    if(((*histo)->getEntries()>FitCuts_Entries) && ModuleHisto==true){   
      histocounter++;            
      if(TIB==1){
      edm::LogInfo("SiStripCalibLorentzAngle")<<"TIB layer = "<<Layer;}    
      if(TOB==1){
      edm::LogInfo("SiStripCalibLorentzAngle")<<"TOB layer = "<<Layer;}     
      edm::LogInfo("SiStripCalibLorentzAngle")<<"id: "<<id;    
            
      float thickness=stripdet->specificSurface().bounds().thickness();
      const StripTopology& topol=(const StripTopology&)stripdet->topology();
      float pitch = topol.localPitch(p);
           	  
      fitfunc->SetParameter(0, p0_guess);
      fitfunc->SetParameter(1, p1_guess);
      fitfunc->SetParameter(2, p2_guess);
      fitfunc->FixParameter(3, pitch);
      fitfunc->FixParameter(4, thickness);
      
      Profiles[name.c_str()]->Fit("fitfunc","E","",ModuleRangeMin, ModuleRangeMax);
            
      FitFunction = Profiles[name.c_str()]->GetFunction("fitfunc");
      chi2norm = FitFunction->GetChisquare()/FitFunction->GetNDF();
      
      if(FitFunction->GetParameter(0)>FitCuts_p0 || FitFunction->GetParameter(1)<FitCuts_p1 || FitFunction->GetParameter(2)<FitCuts_p2 || chi2norm>FitCuts_chi2 || FitFunction->GetParError(0)<FitCuts_ParErr_p0){
      
      FirstIT_badfit++;   
            
      fitfunc2IT->SetParameter(0, p0_guess);
      fitfunc2IT->SetParameter(1, p1_guess);
      fitfunc2IT->SetParameter(2, p2_guess);
      fitfunc2IT->FixParameter(3, pitch);
      fitfunc2IT->FixParameter(4, thickness);
      
      //2nd Iteration
      Profiles[name.c_str()]->Fit("fitfunc2IT","E","",ModuleRangeMin2IT, ModuleRangeMax2IT);
      
      FitFunction = Profiles[name.c_str()]->GetFunction("fitfunc2IT");
      chi2norm = FitFunction->GetChisquare()/FitFunction->GetNDF();
                
      //2nd Iteration failed
      
      if(FitFunction->GetParameter(0)>FitCuts_p0 || FitFunction->GetParameter(1)<FitCuts_p1 || FitFunction->GetParameter(2)<FitCuts_p2 || chi2norm>FitCuts_chi2 || FitFunction->GetParError(0)<FitCuts_ParErr_p0){
      
      if(subid.subdetId() == int (StripSubdetector::TIB)){
      Profiles[name.c_str()]->SetDirectory(TIB_2IT_BadFit);
      }else{
      Profiles[name.c_str()]->SetDirectory(TOB_2IT_BadFit);} 
            
      SecondIT_badfit++;
      badFit=1;     
          
      }
      
      //2nd Iteration ok
     
      if(FitFunction->GetParameter(0)<FitCuts_p0 && FitFunction->GetParameter(1)>FitCuts_p1 && FitFunction->GetParameter(2)>FitCuts_p2 && chi2norm<FitCuts_chi2 && FitFunction->GetParError(0)>FitCuts_ParErr_p0){
      
      if(subid.subdetId() == int (StripSubdetector::TIB)){
      Profiles[name.c_str()]->SetDirectory(TIB_2IT_GoodFit);
      }else{
      Profiles[name.c_str()]->SetDirectory(TOB_2IT_GoodFit);} 
      
      SecondIT_goodfit++;      
      Good2ITFit = true;
      
      }
            
      }  
                       
      if(FitFunction->GetParameter(0)<FitCuts_p0 && FitFunction->GetParameter(1)>FitCuts_p1 && FitFunction->GetParameter(2)>FitCuts_p2 && chi2norm<FitCuts_chi2 && FitFunction->GetParError(0)>FitCuts_ParErr_p0){
      
	if(Good2ITFit==false){
	
	FirstIT_goodfit++;
	goodFit1IT = 1;
	
	if(subid.subdetId() == int (StripSubdetector::TIB)){
        Profiles[name.c_str()]->SetDirectory(TIB_1IT_GoodFit);
        }else{
        Profiles[name.c_str()]->SetDirectory(TOB_1IT_GoodFit);} 
	
	}
	
	GoodFit++;
	goodFit=1;
	
	LorentzAngle_Plots->cd();
            
	edm::LogInfo("SiStripCalibLorentzAngle")<<FitFunction->GetParameter(0);
	
	muH = -(FitFunction->GetParameter(0))/theBfield;
	
	if(TIB==1){
	if(Layer==1) muH = muH/TIB1calib;
	if(Layer==2) muH = muH/TIB2calib;
	if(Layer==3) muH = muH/TIB3calib;
	if(Layer==4) muH = muH/TIB4calib;
	}
	if(TOB==1){
	if(Layer==1) muH = muH/TOB1calib;
	if(Layer==2) muH = muH/TOB2calib;
	if(Layer==3) muH = muH/TOB3calib;
	if(Layer==4) muH = muH/TOB4calib;
	if(Layer==5) muH = muH/TOB5calib;
	if(Layer==6) muH = muH/TOB6calib;
	}
	
	detid_la[id]= muH;
	
	if(TIB==1){
	
	TH1Ds["LA_TIB"]->Fill(muH);
	TH1Ds["LA_err_TIB"]->Fill(FitFunction->GetParError(0)/theBfield);
	TH1Ds["LA_chi2norm_TIB"]->Fill(chi2norm);
	TH2Ds["LA_phi_TIB"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TIB"]->Fill(geta,muH);
	TH2Ds["LA_TIB_graph"]->Fill(Layer,muH);
	
	if(Layer==1){
	TH1Ds["LA_TIB_1"]->Fill(muH);
	TH2Ds["LA_phi_TIB1"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TIB1"]->Fill(geta,muH);
	if(MonoStereo==0){
	TH1Ds["LA_TIB_1_mono"]->Fill(muH);
	TH2Ds["LA_phi_TIB1_mono"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TIB1_mono"]->Fill(geta,muH);}
	if(MonoStereo==1){
	TH1Ds["LA_TIB_1_stereo"]->Fill(muH);
	TH2Ds["LA_phi_TIB1_stereo"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TIB1_stereo"]->Fill(geta,muH);}
	}
	
	if(Layer==2){
	TH1Ds["LA_TIB_2"]->Fill(muH);
	TH2Ds["LA_phi_TIB2"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TIB2"]->Fill(geta,muH);
	if(MonoStereo==0){
	TH1Ds["LA_TIB_2_mono"]->Fill(muH);
	TH2Ds["LA_phi_TIB2_mono"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TIB2_mono"]->Fill(geta,muH);}
	if(MonoStereo==1){
	TH1Ds["LA_TIB_2_stereo"]->Fill(muH);
	TH2Ds["LA_phi_TIB2_stereo"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TIB2_stereo"]->Fill(geta,muH);}
	}
	
	if(Layer==3){
	TH1Ds["LA_TIB_3"]->Fill(muH);
	TH2Ds["LA_phi_TIB3"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TIB3"]->Fill(geta,muH);
	}
	
	if(Layer==4){
	TH1Ds["LA_TIB_4"]->Fill(muH);
	TH2Ds["LA_phi_TIB4"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TIB4"]->Fill(geta,muH);
	}
	}	
	
	if(TOB==1){
	
        TH1Ds["LA_TOB"]->Fill(muH);
	TH1Ds["LA_err_TOB"]->Fill(FitFunction->GetParError(0)/theBfield);
	TH1Ds["LA_chi2norm_TOB"]->Fill(chi2norm);
	TH2Ds["LA_phi_TOB"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TOB"]->Fill(geta,muH);
	TH2Ds["LA_TOB_graph"]->Fill(Layer,muH);
	
	if(Layer==1){
	TH1Ds["LA_TOB_1"]->Fill(muH);
	TH2Ds["LA_phi_TOB1"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TOB1"]->Fill(geta,muH);
	if(MonoStereo==0){
	TH1Ds["LA_TOB_1_mono"]->Fill(muH);
	TH2Ds["LA_phi_TOB1_mono"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TOB1_mono"]->Fill(geta,muH);}
	if(MonoStereo==1){
	TH1Ds["LA_TOB_1_stereo"]->Fill(muH);
	TH2Ds["LA_phi_TOB1_stereo"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TOB1_stereo"]->Fill(geta,muH);}
	}
	
	if(Layer==2){
	TH1Ds["LA_TOB_2"]->Fill(muH);
	TH2Ds["LA_phi_TOB2"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TOB2"]->Fill(geta,muH);
	if(MonoStereo==0){
	TH1Ds["LA_TOB_2_mono"]->Fill(muH);
	TH2Ds["LA_phi_TOB2_mono"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TOB2_mono"]->Fill(geta,muH);}
	if(MonoStereo==1){
	TH1Ds["LA_TOB_2_stereo"]->Fill(muH);
	TH2Ds["LA_phi_TOB2_stereo"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TOB2_stereo"]->Fill(geta,muH);}
	}
	
	if(Layer==3){
	TH1Ds["LA_TOB_3"]->Fill(muH);
	TH2Ds["LA_phi_TOB3"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TOB3"]->Fill(geta,muH);
	}
	
	if(Layer==4){
	TH1Ds["LA_TOB_4"]->Fill(muH);
	TH2Ds["LA_phi_TOB4"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TOB4"]->Fill(geta,muH);
	}
	
	if(Layer==5){
	TH1Ds["LA_TOB_5"]->Fill(muH);
	TH2Ds["LA_phi_TOB5"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TOB5"]->Fill(geta,muH);
	}
	
	if(Layer==6){
	TH1Ds["LA_TOB_6"]->Fill(muH);
	TH2Ds["LA_phi_TOB6"]->Fill(gphi,muH);
	TH2Ds["LA_eta_TOB6"]->Fill(geta,muH);
	}
	}
	
       }
    }
    
    ModuleTree->Fill();
     
  }
  
  double GaussFitRange=conf_.getParameter<double>("GaussFitRange");
  
  TH1Ds["LA_TIB_1"]->Fit("gaus","","",-GaussFitRange,GaussFitRange);  
  mean_TIB1 = TH1Ds["LA_TIB_1"]->GetFunction("gaus")->GetParameter(1);
  float err_mean_TIB1 = TH1Ds["LA_TIB_1"]->GetFunction("gaus")->GetParError(1);
  float rms_TIB1 = TH1Ds["LA_TIB_1"]->GetFunction("gaus")->GetParameter(2);
  TH1Ds["LA_TIB_2"]->Fit("gaus","","",-GaussFitRange,GaussFitRange);  
  mean_TIB2 = TH1Ds["LA_TIB_2"]->GetFunction("gaus")->GetParameter(1);
  float err_mean_TIB2 = TH1Ds["LA_TIB_2"]->GetFunction("gaus")->GetParError(1);
  float rms_TIB2 = TH1Ds["LA_TIB_2"]->GetFunction("gaus")->GetParameter(2);
  TH1Ds["LA_TIB_3"]->Fit("gaus","","",-GaussFitRange,GaussFitRange);  
  mean_TIB3 = TH1Ds["LA_TIB_3"]->GetFunction("gaus")->GetParameter(1);
  float err_mean_TIB3 = TH1Ds["LA_TIB_3"]->GetFunction("gaus")->GetParError(1);
  float rms_TIB3 = TH1Ds["LA_TIB_3"]->GetFunction("gaus")->GetParameter(2);
  TH1Ds["LA_TIB_4"]->Fit("gaus","","",-GaussFitRange,GaussFitRange);  
  mean_TIB4 = TH1Ds["LA_TIB_4"]->GetFunction("gaus")->GetParameter(1);
  float err_mean_TIB4 = TH1Ds["LA_TIB_4"]->GetFunction("gaus")->GetParError(1);
  float rms_TIB4 = TH1Ds["LA_TIB_4"]->GetFunction("gaus")->GetParameter(2);

  TH1Ds["LA_TOB_1"]->Fit("gaus","","",-GaussFitRange,GaussFitRange);  
  mean_TOB1 = TH1Ds["LA_TOB_1"]->GetFunction("gaus")->GetParameter(1);
  float err_mean_TOB1 = TH1Ds["LA_TOB_1"]->GetFunction("gaus")->GetParError(1);
  float rms_TOB1 = TH1Ds["LA_TOB_1"]->GetFunction("gaus")->GetParameter(2);
  TH1Ds["LA_TOB_2"]->Fit("gaus","","",-GaussFitRange,GaussFitRange);  
  mean_TOB2 = TH1Ds["LA_TOB_2"]->GetFunction("gaus")->GetParameter(1);
  float err_mean_TOB2 = TH1Ds["LA_TOB_2"]->GetFunction("gaus")->GetParError(1);
  float rms_TOB2 = TH1Ds["LA_TOB_2"]->GetFunction("gaus")->GetParameter(2);
  TH1Ds["LA_TOB_3"]->Fit("gaus","","",-GaussFitRange,GaussFitRange);  
  mean_TOB3 = TH1Ds["LA_TOB_3"]->GetFunction("gaus")->GetParameter(1);
  float err_mean_TOB3 = TH1Ds["LA_TOB_3"]->GetFunction("gaus")->GetParError(1);
  float rms_TOB3 = TH1Ds["LA_TOB_3"]->GetFunction("gaus")->GetParameter(2);
  TH1Ds["LA_TOB_4"]->Fit("gaus","","",-GaussFitRange,GaussFitRange);  
  mean_TOB4 = TH1Ds["LA_TOB_4"]->GetFunction("gaus")->GetParameter(1);
  float err_mean_TOB4 = TH1Ds["LA_TOB_4"]->GetFunction("gaus")->GetParError(1);
  float rms_TOB4 = TH1Ds["LA_TOB_4"]->GetFunction("gaus")->GetParameter(2);
  TH1Ds["LA_TOB_5"]->Fit("gaus","","",-GaussFitRange,GaussFitRange);  
  mean_TOB5 = TH1Ds["LA_TOB_5"]->GetFunction("gaus")->GetParameter(1);
  float err_mean_TOB5 = TH1Ds["LA_TOB_5"]->GetFunction("gaus")->GetParError(1);
  float rms_TOB5 = TH1Ds["LA_TOB_5"]->GetFunction("gaus")->GetParameter(2);
  TH1Ds["LA_TOB_6"]->Fit("gaus","","",-GaussFitRange,GaussFitRange);  
  mean_TOB6 = TH1Ds["LA_TOB_6"]->GetFunction("gaus")->GetParameter(1);
  float err_mean_TOB6 = TH1Ds["LA_TOB_6"]->GetFunction("gaus")->GetParError(1);
  float rms_TOB6 = TH1Ds["LA_TOB_6"]->GetFunction("gaus")->GetParameter(2);

int nlayersTIB = 4;
float TIBx[4]={1,2,3,4};
float TIBex[4]={0,0,0,0};
float TIBy[4]={mean_TIB1, mean_TIB2, mean_TIB3, mean_TIB4};
float TIBey[4]={err_mean_TIB1, err_mean_TIB2, err_mean_TIB3, err_mean_TIB4};

int nlayersTOB = 6;
float TOBx[6]={1,2,3,4,5,6};
float TOBex[6]={0,0,0,0,0,0};
float TOBy[6]={mean_TOB1, mean_TOB2, mean_TOB3, mean_TOB4, mean_TOB5, mean_TOB6};
float TOBey[6]={err_mean_TOB1, err_mean_TOB2, err_mean_TOB3, err_mean_TOB4, err_mean_TOB5, err_mean_TOB6};

TIB_graph = new TGraphErrors(nlayersTIB,TIBx,TIBy,TIBex,TIBey);
TOB_graph = new TGraphErrors(nlayersTOB,TOBx,TOBy,TOBex,TOBey);

//TF1 *fit_TIB= new TF1("fit_TIB","[0]",0,4);
//TF1 *fit_TOB= new TF1("fit_TOB","[0]",0,6);

gStyle->SetOptFit(111);
gStyle->SetOptStat(111);

TIB_graph->SetTitle("TIB Layers #mu_{H}");
TIB_graph->GetXaxis()->SetTitle("Layers");
TIB_graph->GetXaxis()->SetNdivisions(4);
TIB_graph->GetYaxis()->SetTitle("#mu_{H}");
TIB_graph->SetMarkerStyle(20);
TIB_graph->GetYaxis()->SetTitleOffset(1.3);
TIB_graph->Fit("fit_TIB","E","",1,4);
meanMobility_TIB = TIB_graph->GetFunction("fit_TIB")->GetParameter(0);

TOB_graph->SetTitle("TOB Layers #mu_{H}");
TOB_graph->GetXaxis()->SetTitle("Layers");
TOB_graph->GetXaxis()->SetNdivisions(6);
TOB_graph->GetYaxis()->SetTitle("#mu_{H}");
TOB_graph->SetMarkerStyle(20);
TOB_graph->GetYaxis()->SetTitleOffset(1.3);
TOB_graph->Fit("fit_TOB","E","",1,6);
meanMobility_TOB = TOB_graph->GetFunction("fit_TOB")->GetParameter(0);

  TIB_graph->Write("TIB_graph");
  TOB_graph->Write("TOB_graph");

  Rep<<"- NR.OF TIB AND TOB MODULES = 7932"<<std::endl<<std::endl<<std::endl;
  Rep<<"- NO MODULE HISTOS FOUND = "<<no_mod_histo<<std::endl<<std::endl;
  Rep<<"- NR.OF HISTOS WITH ENTRIES > "<<FitCuts_Entries<<" = "<<histocounter<<std::endl<<std::endl;
  Rep<<"- NR.OF HISTOS WITH ENTRIES <= "<<FitCuts_Entries<<" (!=0) = "<<NotEnoughEntries<<std::endl<<std::endl;
  Rep<<"- NR.OF HISTOS WITH 0 ENTRIES = "<<ZeroEntries<<std::endl<<std::endl<<std::endl;
  Rep<<"- NR.OF GOOD FIT (FIRST IT + SECOND IT GOOD FIT)= "<<GoodFit<<std::endl<<std::endl;
  Rep<<"- NR.OF FIRST IT GOOD FIT = "<<FirstIT_goodfit<<std::endl<<std::endl;
  Rep<<"- NR.OF SECOND IT GOOD FIT = "<<SecondIT_goodfit<<std::endl<<std::endl;
  Rep<<"- NR.OF FIRST IT BAD FIT = "<<FirstIT_badfit<<std::endl<<std::endl;
  Rep<<"- NR.OF SECOND IT BAD FIT = "<<SecondIT_badfit<<std::endl<<std::endl<<std::endl;
  
  Rep<<"--------------- Mean MuH values per Layer -------------------"<<std::endl<<std::endl<<std::endl;
  Rep<<"TIB1 = "<<mean_TIB1<<" +- "<<err_mean_TIB1<<" RMS = "<<rms_TIB1<<std::endl;
  Rep<<"TIB2 = "<<mean_TIB2<<" +- "<<err_mean_TIB2<<" RMS = "<<rms_TIB2<<std::endl;
  Rep<<"TIB3 = "<<mean_TIB3<<" +- "<<err_mean_TIB3<<" RMS = "<<rms_TIB3<<std::endl;
  Rep<<"TIB4 = "<<mean_TIB4<<" +- "<<err_mean_TIB4<<" RMS = "<<rms_TIB4<<std::endl;
  Rep<<"TOB1 = "<<mean_TOB1<<" +- "<<err_mean_TOB1<<" RMS = "<<rms_TOB1<<std::endl;
  Rep<<"TOB2 = "<<mean_TOB2<<" +- "<<err_mean_TOB2<<" RMS = "<<rms_TOB2<<std::endl;
  Rep<<"TOB3 = "<<mean_TOB3<<" +- "<<err_mean_TOB3<<" RMS = "<<rms_TOB3<<std::endl;
  Rep<<"TOB4 = "<<mean_TOB4<<" +- "<<err_mean_TOB4<<" RMS = "<<rms_TOB4<<std::endl;
  Rep<<"TOB5 = "<<mean_TOB5<<" +- "<<err_mean_TOB5<<" RMS = "<<rms_TOB5<<std::endl;
  Rep<<"TOB6 = "<<mean_TOB6<<" +- "<<err_mean_TOB6<<" RMS = "<<rms_TOB6<<std::endl<<std::endl;
  Rep<<"Mean Hall Mobility TIB = "<<meanMobility_TIB<<" +- "<<TIB_graph->GetFunction("fit_TIB")->GetParError(0)<<std::endl;
  Rep<<"Mean Hall Mobility TOB = "<<meanMobility_TOB<<" +- "<<TOB_graph->GetFunction("fit_TOB")->GetParError(0)<<std::endl<<std::endl<<std::endl;
        
  Rep.close();
  NoEntries.close(); 
  
hFile->Write();
hFile->Close();
    
}

// Virtual destructor needed.

SiStripCalibLorentzAngle::~SiStripCalibLorentzAngle(){
 delete hFile;
}
  

// Analyzer: Functions that gets called by framework every event
   
SiStripLorentzAngle* SiStripCalibLorentzAngle::getNewObject(){

  SiStripLorentzAngle* LorentzAngle = new SiStripLorentzAngle();
  
  if(!LayerDB){
  for(std::map<uint32_t, float>::iterator it = detid_la.begin(); it != detid_la.end(); it++){
    
    float langle=it->second;
    if ( ! LorentzAngle->putLorentzAngle(it->first,langle) )
      edm::LogError("SiStripCalibLorentzAngle")<<"[SiStripCalibLorentzAngle::analyze] detid already exists"<<std::endl;
  }
  }
  
  else{
  
    const TrackerGeometry::DetIdContainer& Id = estracker->detIds();
    TrackerGeometry::DetIdContainer::const_iterator Iditer; 
       
    for(Iditer=Id.begin();Iditer!=Id.end();Iditer++){
  
    StripSubdetector subid(Iditer->rawId());
    
    hallMobility = 0.;
    
    if(subid.subdetId() == int (StripSubdetector::TIB)){
    
    uint32_t tibLayer = tTopo->tibLayer(*Iditer);
    if(tibLayer==1){
    hallMobility=mean_TIB1;}
    if(tibLayer==2){
    hallMobility=mean_TIB2;}
    if(tibLayer==3){
    hallMobility=mean_TIB3;}
    if(tibLayer==4){
    hallMobility=mean_TIB4;}
    if (!LorentzAngle->putLorentzAngle(Iditer->rawId(),hallMobility)) edm::LogError("SiStripLorentzAngleGenerator")<<" detid already exists"<<std::endl;
    }
    
    if(subid.subdetId() == int (StripSubdetector::TOB)){
    
    uint32_t tobLayer = tTopo->tobLayer(*Iditer);
    if(tobLayer==1){
    hallMobility=mean_TOB1;}
    if(tobLayer==2){
    hallMobility=mean_TOB2;}
    if(tobLayer==3){
    hallMobility=mean_TOB3;}
    if(tobLayer==4){
    hallMobility=mean_TOB4;}
    if(tobLayer==5){
    hallMobility=mean_TOB5;}
    if(tobLayer==6){
    hallMobility=mean_TOB6;}
    if (!LorentzAngle->putLorentzAngle(Iditer->rawId(),hallMobility)) edm::LogError("SiStripLorentzAngleGenerator")<<" detid already exists"<<std::endl;
    } 
     
    if( subid.subdetId() == int(StripSubdetector::TID) ) {
    hallMobility=meanMobility_TIB;
    if (!LorentzAngle->putLorentzAngle(Iditer->rawId(),hallMobility)) edm::LogError("SiStripLorentzAngleGenerator")<<" detid already exists"<<std::endl;
    } 
    
    if( subid.subdetId() == int(StripSubdetector::TEC) ) {
    
    if(tTopo->tecRing(subid)<5 ) {
    hallMobility=meanMobility_TIB;
    }else{
    hallMobility=meanMobility_TOB;
    }
    if (!LorentzAngle->putLorentzAngle(Iditer->rawId(),hallMobility)) edm::LogError("SiStripLorentzAngleGenerator")<<" detid already exists"<<std::endl;
    }
                 
  }
  }
  
  return LorentzAngle;
  
}





  
  
  


