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
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include <TProfile.h>
#include <TStyle.h>


SiStripCalibLorentzAngle::SiStripCalibLorentzAngle(edm::ParameterSet const& conf) : ConditionDBWriter<SiStripLorentzAngle>(conf) , conf_(conf){}


void SiStripCalibLorentzAngle::algoBeginJob(const edm::EventSetup& c){

  //  edm::ESHandle<TrackerGeometry> estracker;
  c.get<TrackerDigiGeometryRecord>().get(estracker);
  const TrackerGeometry *tracker=&(*estracker); 
  
  //c.get<IdealMagneticFieldRecord>().get(magfield_);
  
  //get magnetic field and geometry from ES
  edm::ESHandle<MagneticField> magfield_;
  c.get<IdealMagneticFieldRecord>().get(magfield_);
  //const MagneticField *  magfield=&(*magfield_);

  edm::ESHandle<SiStripLorentzAngle> SiStripLorentzAngle_;
  c.get<SiStripLorentzAngleRcd>().get(SiStripLorentzAngle_);
  detid_la= SiStripLorentzAngle_->getLorentzAngles();
  
  DQMStore* dbe_ = edm::Service<DQMStore>().operator->();
  std::string inputFile_ =conf_.getUntrackedParameter<std::string>("fileName", "LorentzAngle.root");
  std::string outputFile_ =conf_.getUntrackedParameter<std::string>("out_fileName", "LorentzAngle.root");
  std::string LAreport_ =conf_.getUntrackedParameter<std::string>("LA_Report", "LorentzAngle.root");
  std::string LAProbFit_ =conf_.getUntrackedParameter<std::string>("LA_ProbFit", "LorentzAngle.root");
  dbe_->open(inputFile_);
  
  // use SistripHistoId for producing histogram id (and title)
  SiStripHistoId hidmanager;
   
  std::vector<MonitorElement*> histolist= dbe_->getAllContents("/");
  std::vector<MonitorElement*>::iterator histo;
    
  dbe_->setCurrentFolder("LorentzAngle_Plots");
  dbe_->setCurrentFolder("LorentzAngle_Plots/Histos/TIB");
  dbe_->setCurrentFolder("LorentzAngle_Plots/Histos/TOB");
  dbe_->setCurrentFolder("LorentzAngle_Plots/Profiles/TIB");
  dbe_->setCurrentFolder("LorentzAngle_Plots/Profiles/TOB");
  dbe_->setCurrentFolder("LorentzAngle_Plots/2IT_BadFit_Histos/TIB");
  dbe_->setCurrentFolder("LorentzAngle_Plots/2IT_BadFit_Histos/TOB");
  dbe_->setCurrentFolder("LorentzAngle_Plots/2IT_GoodFit_Histos/TIB");
  dbe_->setCurrentFolder("LorentzAngle_Plots/2IT_GoodFit_Histos/TOB");
  dbe_->setCurrentFolder("LorentzAngle_Plots/1IT_GoodFit_Histos/TIB");
  dbe_->setCurrentFolder("LorentzAngle_Plots/1IT_GoodFit_Histos/TOB");
  
  dbe_->cd("LorentzAngle_Plots/Histos");
  
  MonitorElement * LA_plot=dbe_->book1D("TanLAPerTesla","TanLAPerTesla",1000,-0.5,0.5); 
  MonitorElement * LA_err_plot=dbe_->book1D("TanLAPerTesla Error","TanLAPerTesla Error",1000,0,1);
  MonitorElement * LA_chi2norm_plot=dbe_->book1D("TanLAPerTesla Chi2norm","TanLAPerTesla Chi2norm",2000,0,10);
  MonitorElement * MagneticField=dbe_->book1D("MagneticField","MagneticField",500,0,5);
  
  dbe_->cd("LorentzAngle_Plots/Histos/TIB");
  
  MonitorElement * LA_plot_TIB=dbe_->book1D("TanLAPerTesla_TIB","TanLAPerTesla_TIB",2000,-0.5,0.5);
  MonitorElement * LA_plot_TIB_1_mono=dbe_->book1D("TanLAPerTesla_TIB_1_MONO","TanLAPerTesla_TIB_1_MONO",2000,-0.5,0.5);
  MonitorElement * LA_plot_TIB_1_stereo=dbe_->book1D("TanLAPerTesla_TIB_1_STEREO","TanLAPerTesla_TIB_1_STEREO",2000,-0.5,0.5);
  MonitorElement * LA_plot_TIB_2_mono=dbe_->book1D("TanLAPerTesla_TIB_2_MONO","TanLAPerTesla_TIB_2_MONO",2000,-0.5,0.5);
  MonitorElement * LA_plot_TIB_2_stereo=dbe_->book1D("TanLAPerTesla_TIB_2_STEREO","TanLAPerTesla_TIB_2_STEREO",2000,-0.5,0.5);
  MonitorElement * LA_plot_TIB_3=dbe_->book1D("TanLAPerTesla_TIB_3","TanLAPerTesla_TIB_3",2000,-0.5,0.5);
  MonitorElement * LA_plot_TIB_4=dbe_->book1D("TanLAPerTesla_TIB_4","TanLAPerTesla_TIB_4",2000,-0.5,0.5);
  MonitorElement * LA_MONO_TIB=dbe_->book1D("TanLAPerTesla_MONO_TIB","TanLAPerTesla_MONO_TIB",2000,-0.5,0.5);
  MonitorElement * LA_STEREO_TIB=dbe_->book1D("TanLAPerTesla_STEREO_TIB","TanLAPerTesla_STEREO_TIB",2000,-0.5,0.5);
  
  dbe_->cd("LorentzAngle_Plots/Histos/TOB");
  
  MonitorElement * LA_plot_TOB=dbe_->book1D("TanLAPerTesla_TOB","TanLAPerTesla_TOB",2000,-0.5,0.5);
  MonitorElement * LA_plot_TOB_1_mono=dbe_->book1D("TanLAPerTesla_TOB_1_MONO","TanLAPerTesla_TOB_1_MONO",2000,-0.5,0.5);
  MonitorElement * LA_plot_TOB_1_stereo=dbe_->book1D("TanLAPerTesla_TOB_1_STEREO","TanLAPerTesla_TOB_1_STEREO",2000,-0.5,0.5);
  MonitorElement * LA_plot_TOB_2_mono=dbe_->book1D("TanLAPerTesla_TOB_2_MONO","TanLAPerTesla_TOB_2_MONO",2000,-0.5,0.5);
  MonitorElement * LA_plot_TOB_2_stereo=dbe_->book1D("TanLAPerTesla_TOB_2_STEREO","TanLAPerTesla_TOB_2_STEREO",2000,-0.5,0.5);
  MonitorElement * LA_plot_TOB_3=dbe_->book1D("TanLAPerTesla_TOB_3","TanLAPerTesla_TOB_3",2000,-0.5,0.5);
  MonitorElement * LA_plot_TOB_4=dbe_->book1D("TanLAPerTesla_TOB_4","TanLAPerTesla_TOB_4",2000,-0.5,0.5);
  MonitorElement * LA_plot_TOB_5=dbe_->book1D("TanLAPerTesla_TOB_5","TanLAPerTesla_TOB_5",2000,-0.5,0.5);
  MonitorElement * LA_plot_TOB_6=dbe_->book1D("TanLAPerTesla_TOB_6","TanLAPerTesla_TOB_6",2000,-0.5,0.5);  
  MonitorElement * LA_MONO_TOB=dbe_->book1D("TanLAPerTesla_MONO_TOB","TanLAPerTesla_MONO_TOB",2000,-0.5,0.5);
  MonitorElement * LA_STEREO_TOB=dbe_->book1D("TanLAPerTesla_STEREO_TOB","TanLAPerTesla_STEREO_TOB",2000,-0.5,0.5);
  
  dbe_->cd("LorentzAngle_Plots/Profiles");
    
  MonitorElement * LA_phi_plot=dbe_->bookProfile("TanLAPerTesla_vs_Phi","TanLAPerTesla_vs_Phi",200,-3.5,3.5,1000,-0.5,0.5,"");
  MonitorElement * LA_eta_plot=dbe_->bookProfile("TanLAPerTesla_vs_Eta","TanLAPerTesla_vs_Eta",200,-2.6,2.6,1000,-0.5,0.5,"");
  
  dbe_->cd("LorentzAngle_Plots/Profiles/TIB");
  
  MonitorElement * LA_eta_plot_TIB=dbe_->bookProfile("TanLAPerTesla_vs_Eta_TIB","TanLAPerTesla_vs_Eta_TIB",200,-3.5,3.5,1000,-0.5,0.5,"");
  MonitorElement * LA_Z_plot_MONO_TIB1=dbe_->bookProfile("TanLAPerTesla_vs_Z_MONO_TIB1","TanLAPerTesla_vs_Z_MONO_TIB1",200,-100,100,1000,-0.5,0.5,"");
  MonitorElement * LA_Z_plot_STEREO_TIB1=dbe_->bookProfile("TanLAPerTesla_vs_Z_STEREO_TIB1","TanLAPerTesla_vs_Z_STEREO_TIB1",200,-100,100,1000,-0.5,0.5,"");
  MonitorElement * LA_Z_plot_MONO_TIB2=dbe_->bookProfile("TanLAPerTesla_vs_Z_MONO_TIB2","TanLAPerTesla_vs_Z_MONO_TIB2",200,-100,100,1000,-0.5,0.5,"");
  MonitorElement * LA_Z_plot_STEREO_TIB2=dbe_->bookProfile("TanLAPerTesla_vs_Z_STEREO_TIB2","TanLAPerTesla_vs_Z_STEREO_TIB2",200,-100,100,1000,-0.5,0.5,"");
  MonitorElement * LA_Z_plot_TIB3=dbe_->bookProfile("TanLAPerTesla_vs_Z_TIB3","TanLAPerTesla_vs_Z_TIB3",200,-100,100,1000,-0.5,0.5,"");
  MonitorElement * LA_Z_plot_TIB4=dbe_->bookProfile("TanLAPerTesla_vs_Z_TIB4","TanLAPerTesla_vs_Z_TIB4",200,-100,100,1000,-0.5,0.5,"");
 
  dbe_->cd("LorentzAngle_Plots/Profiles/TOB");
  
  MonitorElement * LA_eta_plot_TOB=dbe_->bookProfile("TanLAPerTesla_vs_Eta_TOB","TanLAPerTesla_vs_Eta_TOB",200,-3.5,3.5,1000,-0.5,0.5,"");
  MonitorElement * LA_Z_plot_MONO_TOB1=dbe_->bookProfile("TanLAPerTesla_vs_Z_MONO_TOB1","TanLAPerTesla_vs_Z_MONO_TOB1",200,-100,100,1000,-0.5,0.5,"");
  MonitorElement * LA_Z_plot_STEREO_TOB1=dbe_->bookProfile("TanLAPerTesla_vs_Z_STEREO_TOB1","TanLAPerTesla_vs_Z_STEREO_TOB1",200,-100,100,1000,-0.5,0.5,"");
  MonitorElement * LA_Z_plot_MONO_TOB2=dbe_->bookProfile("TanLAPerTesla_vs_Z_MONO_TOB2","TanLAPerTesla_vs_Z_MONO_TOB2",200,-100,100,1000,-0.5,0.5,"");
  MonitorElement * LA_Z_plot_STEREO_TOB2=dbe_->bookProfile("TanLAPerTesla_vs_Z_STEREO_TOB2","TanLAPerTesla_vs_Z_STEREO_TOB2",200,-100,100,1000,-0.5,0.5,"");
  MonitorElement * LA_Z_plot_TOB3=dbe_->bookProfile("TanLAPerTesla_vs_Z_TOB3","TanLAPerTesla_vs_Z_TOB3",200,-100,100,1000,-0.5,0.5,"");
  MonitorElement * LA_Z_plot_TOB4=dbe_->bookProfile("TanLAPerTesla_vs_Z_TOB4","TanLAPerTesla_vs_Z_TOB4",200,-100,100,1000,-0.5,0.5,"");
  MonitorElement * LA_Z_plot_TOB5=dbe_->bookProfile("TanLAPerTesla_vs_Z_TOB5","TanLAPerTesla_vs_Z_TOB5",200,-100,100,1000,-0.5,0.5,"");
  MonitorElement * LA_Z_plot_TOB6=dbe_->bookProfile("TanLAPerTesla_vs_Z_TOB6","TanLAPerTesla_vs_Z_TOB6",200,-100,100,1000,-0.5,0.5,""); 
  
  
  histos[1] = LA_plot;
  histos[2] = LA_err_plot;
  histos[3] = LA_chi2norm_plot;
  histos[4] = LA_phi_plot;
  histos[5] = LA_eta_plot;
  
  histos[6] = LA_plot_TIB;
  histos[7] = LA_plot_TIB_1_mono;
  histos[8] = LA_plot_TIB_1_stereo;
  histos[9] = LA_plot_TIB_2_mono;
  histos[10] = LA_plot_TIB_2_stereo;
  histos[11] = LA_plot_TIB_3;
  histos[12] = LA_plot_TIB_4;
    
  histos[13] = LA_plot_TOB;
  histos[14] = LA_plot_TOB_1_mono;
  histos[15] = LA_plot_TOB_1_stereo;
  histos[16] = LA_plot_TOB_2_mono;
  histos[17] = LA_plot_TOB_2_stereo;
  histos[18] = LA_plot_TOB_3;
  histos[19] = LA_plot_TOB_4;
  histos[20] = LA_plot_TOB_5;
  histos[21] = LA_plot_TOB_6;
  
  histos[22] = LA_MONO_TIB;
  histos[23] = LA_STEREO_TIB;
  histos[24] = LA_eta_plot_TIB;
  TProfile* Eta_TIB=ExtractTObject<TProfile>().extract(histos[24]);
  Eta_TIB->GetXaxis()->SetTitle("#eta");
  Eta_TIB->GetYaxis()->SetTitle("TanLAPerTesla");
  Eta_TIB->GetYaxis()->CenterTitle();
  Eta_TIB->GetYaxis()->SetTitleOffset(1.3);

  histos[25] = LA_MONO_TOB;
  histos[26] = LA_STEREO_TOB;
  histos[27] = LA_eta_plot_TOB;
  TProfile* Eta_TOB=ExtractTObject<TProfile>().extract(histos[27]);
  Eta_TOB->GetXaxis()->SetTitle("#eta");
  Eta_TOB->GetYaxis()->SetTitle("TanLAPerTesla");
  Eta_TOB->GetYaxis()->CenterTitle();
  Eta_TOB->GetYaxis()->SetTitleOffset(1.3);
  
  histos[30] = LA_Z_plot_MONO_TIB1;
  histos[31] = LA_Z_plot_STEREO_TIB1;
  histos[32] = LA_Z_plot_MONO_TIB2;
  histos[33] = LA_Z_plot_STEREO_TIB2;
  histos[34] = LA_Z_plot_TIB3;
  histos[35] = LA_Z_plot_TIB4;
  
  histos[36] = LA_Z_plot_MONO_TOB1;
  histos[37] = LA_Z_plot_STEREO_TOB1;
  histos[38] = LA_Z_plot_MONO_TOB2;
  histos[39] = LA_Z_plot_STEREO_TOB2;
  histos[40] = LA_Z_plot_TOB3;
  histos[41] = LA_Z_plot_TOB4;
  histos[42] = LA_Z_plot_TOB5;
  histos[43] = LA_Z_plot_TOB6;
  
  histos[44] = MagneticField;
  
  hFile = new TFile (conf_.getUntrackedParameter<std::string>("treeName").c_str(), "RECREATE" );
  
  ModuleTree = new TTree("ModuleTree", "ModuleTree");
  ModuleTree->Branch("TreeHistoEntries", &TreeHistoEntries, "TreeHistoEntries/F");
  ModuleTree->Branch("TreeGlobalX", &TreeGlobalX, "TreeGlobalX/F");
  ModuleTree->Branch("TreeGlobalY", &TreeGlobalY, "TreeGlobalY/F");
  ModuleTree->Branch("TreeGlobalZ", &TreeGlobalZ, "TreeGlobalZ/F");
  ModuleTree->Branch("TreeGoodFit", &TreeGoodFit, "TreeGoodFit/I");
  ModuleTree->Branch("TreeBadFit", &TreeBadFit, "TreeBadFit/I");
  ModuleTree->Branch("muH", &muH, "muH/F");
  ModuleTree->Branch("TreeTIB", &TreeTIB, "TreeTIB/I");
  ModuleTree->Branch("TreeTOB", &TreeTOB, "TreeTOB/I");
  ModuleTree->Branch("Layer", &Layer, "Layer/I");
  ModuleTree->Branch("theBfield", &theBfield, "theBfield/F");
  ModuleTree->Branch("gphi", &gphi, "gphi/F");
  ModuleTree->Branch("geta", &geta, "geta/F");
  ModuleTree->Branch("gR", &gR, "gR/F");
  
  gphi=-99;
  geta=-99;
  gz = -99;
  Layer = 0;
     
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
  
  TF1 *fitfunc= new TF1("fitfunc","([4]/[3])*[1]*(TMath::Abs(x-[0]))+[2]",-1,1);
  TF1 *fitfunc2IT= new TF1("fitfunc2IT","([4]/[3])*[1]*(TMath::Abs(x-[0]))+[2]",-1,1);
 
  ofstream LA_pf;
  LA_pf.open(LAProbFit_.c_str());
  ofstream NoEntries;
  NoEntries.open("NoEntriesModules.txt");
  ofstream Rep;
  Rep.open(LAreport_.c_str());
  
  gStyle->SetOptStat(1110);
  
  for(histo=histolist.begin();histo!=histolist.end();++histo){
  
  /*double p0_guess=conf_.getParameter<double>("p0_guess");
  double p1_guess=conf_.getParameter<double>("p1_guess");
  double p2_guess=conf_.getParameter<double>("p2_guess");*/
  
  FitFunction = 0;
  FitFunction2IT = 0;
  bool Good2ITFit = false;
  bool ModuleHisto = true;
  //bool GoodPhi = false;
  TreeHistoEntries = -99;
  TreeGlobalX = -99;
  TreeGlobalY = -99;
  TreeGlobalZ = -99;
  TreeGoodFit = 0;
  TreeBadFit = 0;
  muH = -1;
  TreeTIB = 0;
  TreeTOB = 0;
  
    uint32_t id=hidmanager.getComponentId((*histo)->getName());
    DetId detid(id);
    StripSubdetector subid(id);
    const GeomDetUnit * stripdet;
    
    if(!(stripdet=tracker->idToDetUnit(subid))){
    no_mod_histo++;
    ModuleHisto=false;
    edm::LogInfo("SiStripCalibLorentzAngle")<<"### NO MODULE HISTOGRAM";}
    
    if(stripdet!=0 && ModuleHisto==true){
    
    //get module coordinates
    const GlobalPoint gposition = (stripdet->surface()).toGlobal(p);
    TreeHistoEntries = (*histo)->getEntries();
    TreeGlobalX = gposition.x();
    TreeGlobalY = gposition.y();
    TreeGlobalZ = gposition.z();
    
    //get magnetic field
    const StripGeomDetUnit* det = dynamic_cast<const StripGeomDetUnit*>(estracker->idToDetUnit(detid));
    if (det==0){
    edm::LogError("SiStripCalibLorentzAngle") << "[SiStripCalibLorentzAngle::getNewObject] the detID " << id << " doesn't seem to belong to Tracker" <<std::endl; 	
    continue;
    }
    LocalVector lbfield=(det->surface()).toLocal(magfield_->inTesla(det->surface().position()));
    theBfield = lbfield.mag();
    theBfield = (theBfield > 0) ? theBfield : 0.00001; 
    histos[44]->Fill(theBfield);
    
    }
    if(stripdet==0)continue;
      
    if(((*histo)->getEntries()<=FitCuts_Entries)&&ModuleHisto==true){
    uint32_t id0=hidmanager.getComponentId((*histo)->getName());
    DetId detid0(id0);
    StripSubdetector subid0(id0);
    
    if(((*histo)->getEntries()==0)&&ModuleHisto==true){
    
    NoEntries<<"NO ENTRIES MODULE, ID = "<<id<<std::endl;
    
    edm::LogInfo("SiStripCalibLorentzAngle")<<"### HISTOGRAM WITH 0 ENTRIES => TYPE:"<<subid0.subdetId();
    ZeroEntries++;
    }else{    
    edm::LogInfo("SiStripCalibLorentzAngle")<<"### HISTOGRAM WITH NR. ENTRIES <= ENTRIES_CUT => TYPE:"<<subid0.subdetId();
    NotEnoughEntries++;}
    
    }
  
    if(((*histo)->getEntries()>FitCuts_Entries)&&ModuleHisto==true){
    
      histocounter++;   
          
      if(subid.subdetId() == int (StripSubdetector::TIB)){
      TIBDetId TIBid=TIBDetId(subid);
      Layer = TIBid.layer();
      TreeTIB = 1;
      edm::LogInfo("SiStripCalibLorentzAngle")<<"TIB layer = "<<Layer;}
      
      if(subid.subdetId() == int (StripSubdetector::TOB)){
      TOBDetId TOBid=TOBDetId(subid);
      Layer = TOBid.layer();
      TreeTOB = 1;
      edm::LogInfo("SiStripCalibLorentzAngle")<<"TOB layer = "<<Layer;}
      
      edm::LogInfo("SiStripCalibLorentzAngle")<<"id: "<<id;    
            
      if(stripdet!=0){
      const GlobalPoint gposition = (stripdet->surface()).toGlobal(p);
      gphi = gposition.phi();
      geta = gposition.eta();
      gR = sqrt(pow(gposition.x(),2)+pow(gposition.y(),2));
      gz = gposition.z();}
      if(stripdet==0)continue;
      
      //if((gphi>-2.2 && gphi<-0.2) || (gphi>1 && gphi<2.4))GoodPhi=true;
            
      float thickness=stripdet->specificSurface().bounds().thickness();
      const StripTopology& topol=(StripTopology&)stripdet->topology();
      float pitch = topol.localPitch(p);
           
      TProfile* theProfile=ExtractTObject<TProfile>().extract(*histo);
            
      fitfunc->SetParameter(0, p0_guess);
      fitfunc->SetParameter(1, p1_guess);
      fitfunc->SetParameter(2, p2_guess);
      fitfunc->FixParameter(3, pitch);
      fitfunc->FixParameter(4, thickness);
      
      theProfile->Fit("fitfunc","E","",ModuleRangeMin, ModuleRangeMax);
      
      FitFunction = theProfile->GetFunction("fitfunc");
      chi2norm = FitFunction->GetChisquare()/FitFunction->GetNDF();
      
      if(FitFunction->GetParameter(0)>FitCuts_p0 || FitFunction->GetParameter(1)<FitCuts_p1 || FitFunction->GetParameter(2)<FitCuts_p2 || chi2norm>FitCuts_chi2 || FitFunction->GetParError(0)<FitCuts_ParErr_p0){
      
      FirstIT_badfit++;   
            
      fitfunc2IT->SetParameter(0, p0_guess);
      fitfunc2IT->SetParameter(1, p1_guess);
      fitfunc2IT->SetParameter(2, p2_guess);
      fitfunc2IT->FixParameter(3, pitch);
      fitfunc2IT->FixParameter(4, thickness);
      
      //2nd Iteration
      theProfile->Fit("fitfunc2IT","E","",ModuleRangeMin2IT, ModuleRangeMax2IT);
      
      FitFunction = theProfile->GetFunction("fitfunc2IT");
      chi2norm = FitFunction->GetChisquare()/FitFunction->GetNDF();
      
      //2nd Iteration failed
      if(FitFunction->GetParameter(0)>FitCuts_p0 || FitFunction->GetParameter(1)<FitCuts_p1 || FitFunction->GetParameter(2)<FitCuts_p2 || chi2norm>FitCuts_chi2 || FitFunction->GetParError(0)<FitCuts_ParErr_p0){
      
      if(subid.subdetId() == int (StripSubdetector::TIB)){
      dbe_->cd("LorentzAngle_Plots/2IT_BadFit_Histos/TIB");
      }else{
      dbe_->cd("LorentzAngle_Plots/2IT_BadFit_Histos/TOB");}  
            
      SecondIT_badfit++;
      TreeBadFit=1;     
      gStyle->SetOptFit(111);
      
      std::string name="Fit_Histo";    
      std::stringstream badfitnum;
      badfitnum<<SecondIT_badfit;
      name+=badfitnum.str();
      
      MonitorElement * fit_histo=dbe_->bookProfile(name.c_str(),theProfile);
          
      }
      
      //2nd Iteration ok
     
      if(FitFunction->GetParameter(0)<FitCuts_p0 && FitFunction->GetParameter(1)>FitCuts_p1 && FitFunction->GetParameter(2)>FitCuts_p2 && chi2norm<FitCuts_chi2 && FitFunction->GetParError(0)>FitCuts_ParErr_p0){
      
      if(subid.subdetId() == int (StripSubdetector::TIB)){
      dbe_->cd("LorentzAngle_Plots/2IT_GoodFit_Histos/TIB");
      }else{
      dbe_->cd("LorentzAngle_Plots/2IT_GoodFit_Histos/TOB");}
      
      SecondIT_goodfit++;      
      gStyle->SetOptFit(111);
      
      std::string name="Fit_Histo";    
      std::stringstream goodfitnum;
      goodfitnum<<SecondIT_goodfit;
      name+=goodfitnum.str();
           
      MonitorElement * fit_histo=dbe_->bookProfile(name.c_str(),theProfile);
	   
      Good2ITFit = true;
      }
            
      if(Good2ITFit){
      LA_pf<<"2IT Fit OK"<<std::endl;}
      else{
      LA_pf<<"###??? 2IT Fit Bad"<<std::endl;}
      LA_pf<<"Type = "<<subid.subdetId()<<" Layer = "<<Layer;
      if(subid.subdetId() == int (StripSubdetector::TIB)){
      TIBDetId TIBid=TIBDetId(subid);
      if(TIBid.string()[0]==1){LA_pf<<" Backward, ";}else{LA_pf<<" Forward, ";}
      if(TIBid.string()[1]==1){LA_pf<<" Int String, ";}else{LA_pf<<" Ext String, ";}
      LA_pf<<" Nr. String = "<<TIBid.string()[2];}
      if(subid.subdetId() == int (StripSubdetector::TOB)){
      TOBDetId TOBid=TOBDetId(subid);
      if(TOBid.rod()[0]==1){LA_pf<<" Backward, ";}else{LA_pf<<" Forward, ";}
      LA_pf<<" Nr. Rod = "<<TOBid.rod()[1];}    
      LA_pf<<" MonoStereo = "<<subid.stereo()<<" Id = "<<id<<std::endl;
      LA_pf<<"=> Fit_Par0 = "<<FitFunction->GetParameter(0)<<" Fit_Par1 = "<<FitFunction->GetParameter(1)<<" Fit_Par2 = "<<FitFunction->GetParameter(2);
      LA_pf<<" Chi2/NDF = "<<chi2norm<<std::endl<<std::endl; 
      
      }  
                       
      if(FitFunction->GetParameter(0)<FitCuts_p0 && FitFunction->GetParameter(1)>FitCuts_p1 && FitFunction->GetParameter(2)>FitCuts_p2 && chi2norm<FitCuts_chi2 && FitFunction->GetParError(0)>FitCuts_ParErr_p0){
      
	if(Good2ITFit==false){
	
	FirstIT_goodfit++;
	
	if(subid.subdetId() == int (StripSubdetector::TIB)){
        dbe_->cd("LorentzAngle_Plots/1IT_GoodFit_Histos/TIB");
        }else{
        dbe_->cd("LorentzAngle_Plots/1IT_GoodFit_Histos/TOB");}  
                
        gStyle->SetOptFit(111);
      
        std::string name="Fit_Histo";    
        std::stringstream FirstITgoodfitnum;
        FirstITgoodfitnum<<FirstIT_badfit;
        name+=FirstITgoodfitnum.str();
	
	MonitorElement * fit_histo=dbe_->bookProfile(name.c_str(),theProfile);
	
	}
	
	GoodFit++;
	TreeGoodFit=1;
	
	dbe_->cd("LorentzAngle_Plots");
            
	edm::LogInfo("SiStripCalibLorentzAngle")<<FitFunction->GetParameter(0);
	
	muH = -(FitFunction->GetParameter(0))/theBfield; 

	detid_la[id]= muH;
	
	histos[1]->Fill(muH);
	histos[2]->Fill(FitFunction->GetParError(0)/theBfield);
	histos[3]->Fill(chi2norm);
	histos[4]->Fill(gphi,muH);
	histos[5]->Fill(geta,muH);
	
	if(subid.subdetId() == int (StripSubdetector::TIB)){

        if(subid.stereo()==0){
        //MONO
        histos[22]->Fill(muH);
        }else{
        //STEREO
        histos[23]->Fill(muH);}
	
        histos[24]->Fill(geta,muH);
	histos[6]->Fill(muH);
	
	if((Layer==1)&&(subid.stereo()==0)){
	histos[7]->Fill(muH);
	histos[30]->Fill(gz,muH);}
	if((Layer==1)&&(subid.stereo()==1)){
	histos[8]->Fill(muH);
        histos[31]->Fill(gz,muH);}
	if((Layer==2)&&(subid.stereo()==0)){
	histos[9]->Fill(muH);
	histos[32]->Fill(gz,muH);}
	if((Layer==2)&&(subid.stereo()==1)){
	histos[10]->Fill(muH);
	histos[33]->Fill(gz,muH);}
	if(Layer==3){
	histos[11]->Fill(muH);
	histos[34]->Fill(gz,muH);}
	if(Layer==4){
	histos[12]->Fill(muH);
	histos[35]->Fill(gz,muH);}	
	}
	
	if(subid.subdetId() == int (StripSubdetector::TOB)){

        if(subid.stereo()==0){
        //MONO
        histos[25]->Fill(muH);
        }else{
        //STEREO
        histos[26]->Fill(muH);}
	
        histos[27]->Fill(geta,muH);
	histos[13]->Fill(muH);
	
	if((Layer==1)&&(subid.stereo()==0)){
	histos[14]->Fill(muH);
	histos[36]->Fill(gz,muH);}
	if((Layer==1)&&(subid.stereo()==1)){
	histos[15]->Fill(muH);
	histos[37]->Fill(gz,muH);}
	if((Layer==2)&&(subid.stereo()==0)){
	histos[16]->Fill(muH);
	histos[38]->Fill(gz,muH);}
	if((Layer==2)&&(subid.stereo()==1)){
	histos[17]->Fill(muH);
	histos[39]->Fill(gz,muH);}
	if(Layer==3){
	histos[18]->Fill(muH);
	histos[40]->Fill(gz,muH);}
	if(Layer==4){
	histos[19]->Fill(muH);
	histos[41]->Fill(gz,muH);}
	if(Layer==5){
	histos[20]->Fill(muH);
	histos[42]->Fill(gz,muH);}
	if(Layer==6){
	histos[21]->Fill(muH);
	histos[43]->Fill(gz,muH);}
	}
	
       }
    }
    
    ModuleTree->Fill();
     
  }
  
  
  Rep<<"- NR.OF TIB AND TOB MODULES = 7932"<<std::endl<<std::endl<<std::endl;
  Rep<<"- NO MODULE HISTOS FOUND = "<<no_mod_histo<<std::endl<<std::endl;
  Rep<<"- NR.OF HISTOS WITH ENTRIES > "<<FitCuts_Entries<<" = "<<histocounter<<std::endl<<std::endl;
  Rep<<"- NR.OF HISTOS WITH ENTRIES <= "<<FitCuts_Entries<<" (!=0) = "<<NotEnoughEntries<<std::endl<<std::endl;
  Rep<<"- NR.OF HISTOS WITH 0 ENTRIES = "<<ZeroEntries<<std::endl<<std::endl<<std::endl;
  Rep<<"- NR.OF GOOD FIT (FIRST IT + SECOND IT GOOD FIT)= "<<GoodFit<<std::endl<<std::endl;
  Rep<<"- NR.OF FIRST IT GOOD FIT = "<<FirstIT_goodfit<<std::endl<<std::endl;
  Rep<<"- NR.OF SECOND IT GOOD FIT = "<<SecondIT_goodfit<<std::endl<<std::endl;
  Rep<<"- NR.OF FIRST IT BAD FIT = "<<FirstIT_badfit<<std::endl<<std::endl;
  Rep<<"- NR.OF SECOND IT BAD FIT = "<<SecondIT_badfit<<std::endl<<std::endl;
    
  LA_pf.close();
  Rep.close();
  NoEntries.close();
  
  dbe_->save(outputFile_,"LorentzAngle_Plots/1IT_GoodFit_Histos/TIB");
  dbe_->save(outputFile_,"LorentzAngle_Plots/1IT_GoodFit_Histos/TOB");   
  dbe_->save(outputFile_,"LorentzAngle_Plots/2IT_BadFit_Histos/TIB");
  dbe_->save(outputFile_,"LorentzAngle_Plots/2IT_BadFit_Histos/TOB");
  dbe_->save(outputFile_,"LorentzAngle_Plots/2IT_GoodFit_Histos/TIB");
  dbe_->save(outputFile_,"LorentzAngle_Plots/2IT_GoodFit_Histos/TOB");
  dbe_->save(outputFile_,"LorentzAngle_Plots/Histos/TIB");
  dbe_->save(outputFile_,"LorentzAngle_Plots/Histos/TOB");
  dbe_->save(outputFile_,"LorentzAngle_Plots/Profiles/TIB");
  dbe_->save(outputFile_,"LorentzAngle_Plots/Profiles/TOB");
  dbe_->save(outputFile_,"LorentzAngle_Plots/Histos");
  dbe_->save(outputFile_,"LorentzAngle_Plots/Profiles");
  dbe_->save(outputFile_,"LorentzAngle_Plots"); 
  
  hFile->Write();
  hFile->Close();    
  
}

// Virtual destructor needed.

SiStripCalibLorentzAngle::~SiStripCalibLorentzAngle(){
}
  

// Analyzer: Functions that gets called by framework every event


SiStripLorentzAngle* SiStripCalibLorentzAngle::getNewObject(){

  SiStripLorentzAngle* LorentzAngle = new SiStripLorentzAngle();
  
  for(std::map<uint32_t, float>::iterator it = detid_la.begin(); it != detid_la.end(); it++){
    
    float langle=it->second;
    if ( ! LorentzAngle->putLorentzAngle(it->first,langle) )
      edm::LogError("SiStripCalibLorentzAngle")<<"[SiStripCalibLorentzAngle::analyze] detid already exists"<<std::endl;
  }
  
  return LorentzAngle;
}
