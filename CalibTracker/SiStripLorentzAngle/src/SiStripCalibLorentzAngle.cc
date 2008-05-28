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
#include <TF1.h>
#include <TProfile.h>


SiStripCalibLorentzAngle::SiStripCalibLorentzAngle(edm::ParameterSet const& conf) : ConditionDBWriter<SiStripLorentzAngle>(conf) , conf_(conf){}


void SiStripCalibLorentzAngle::algoBeginJob(const edm::EventSetup& c){

  //  edm::ESHandle<TrackerGeometry> estracker;
  c.get<TrackerDigiGeometryRecord>().get(estracker);
  const TrackerGeometry *tracker=&(*estracker); 
  
  c.get<IdealMagneticFieldRecord>().get(magfield_);

  edm::ESHandle<SiStripLorentzAngle> SiStripLorentzAngle_;
  c.get<SiStripLorentzAngleRcd>().get(SiStripLorentzAngle_);
  detid_la= SiStripLorentzAngle_->getLorentzAngles();

  DQMStore* dbe_ = edm::Service<DQMStore>().operator->();
  std::string inputFile_ =conf_.getUntrackedParameter<std::string>("fileName", "LorentzAngle.root");
  dbe_->open(inputFile_);
  // use SistripHistoId for producing histogram id (and title)
  SiStripHistoId hidmanager;
   
  TF1 *fitfunc=0;
  double ModuleRangeMin=conf_.getParameter<double>("ModuleFitXMin");
  double ModuleRangeMax=conf_.getParameter<double>("ModuleFitXMax");

  fitfunc= new TF1("fitfunc","([4]/[3])*[1]*(TMath::Abs(x-[0]))+[2]",-1,1);

  std::vector<MonitorElement*> histolist= dbe_->getAllContents("/");
  std::vector<MonitorElement*>::iterator histo;
  LocalPoint p =LocalPoint(0,0,0);
  
  dbe_->setCurrentFolder("LorentzAngle_Plots");
  
  MonitorElement * LA_plot=dbe_->book1D("TanLAPerTesla","TanLAPerTesla",2000,-0.5,0.5); 
  MonitorElement * LA_err_plot=dbe_->book1D("TanLAPerTesla Error","TanLAPerTesla Error",1000,0,0.2);
  MonitorElement * LA_chi2norm_plot=dbe_->book1D("TanLAPerTesla Chi2norm","TanLAPerTesla Chi2norm",2000,0,20);
  MonitorElement * LA_phi_plot=dbe_->bookProfile("TanLAPerTesla_vs_Phi","TanLAPerTesla_vs_Phi",200,-3.5,3.5,1000,-0.5,0.5,"");
  MonitorElement * LA_eta_plot=dbe_->bookProfile("TanLAPerTesla_vs_Eta","TanLAPerTesla_vs_Eta",200,-2.6,2.6,1000,-0.5,0.5,"");
  
  MonitorElement * LA_plot_TIB=dbe_->book1D("TanLAPerTesla_TIB","TanLAPerTesla_TIB",2000,-0.5,0.5);
  MonitorElement * LA_plot_TIB_1_mono=dbe_->book1D("TanLAPerTesla_TIB_1_MONO","TanLAPerTesla_TIB_1_MONO",2000,-0.5,0.5);
  MonitorElement * LA_plot_TIB_1_stereo=dbe_->book1D("TanLAPerTesla_TIB_1_STEREO","TanLAPerTesla_TIB_1_STEREO",2000,-0.5,0.5);
  MonitorElement * LA_plot_TIB_2_mono=dbe_->book1D("TanLAPerTesla_TIB_2_MONO","TanLAPerTesla_TIB_2_MONO",2000,-0.5,0.5);
  MonitorElement * LA_plot_TIB_2_stereo=dbe_->book1D("TanLAPerTesla_TIB_2_STEREO","TanLAPerTesla_TIB_2_STEREO",2000,-0.5,0.5);
  MonitorElement * LA_plot_TIB_3=dbe_->book1D("TanLAPerTesla_TIB_3","TanLAPerTesla_TIB_3",2000,-0.5,0.5);
  MonitorElement * LA_plot_TIB_4=dbe_->book1D("TanLAPerTesla_TIB_4","TanLAPerTesla_TIB_4",2000,-0.5,0.5);
  
  MonitorElement * LA_plot_TOB=dbe_->book1D("TanLAPerTesla_TOB","TanLAPerTesla_TOB",2000,-0.5,0.5);
  MonitorElement * LA_plot_TOB_1_mono=dbe_->book1D("TanLAPerTesla_TOB_1_MONO","TanLAPerTesla_TOB_1_MONO",2000,-0.5,0.5);
  MonitorElement * LA_plot_TOB_1_stereo=dbe_->book1D("TanLAPerTesla_TOB_1_STEREO","TanLAPerTesla_TOB_1_STEREO",2000,-0.5,0.5);
  MonitorElement * LA_plot_TOB_2_mono=dbe_->book1D("TanLAPerTesla_TOB_2_MONO","TanLAPerTesla_TOB_2_MONO",2000,-0.5,0.5);
  MonitorElement * LA_plot_TOB_2_stereo=dbe_->book1D("TanLAPerTesla_TOB_2_STEREO","TanLAPerTesla_TOB_2_STEREO",2000,-0.5,0.5);
  MonitorElement * LA_plot_TOB_3=dbe_->book1D("TanLAPerTesla_TOB_3","TanLAPerTesla_TOB_3",2000,-0.5,0.5);
  MonitorElement * LA_plot_TOB_4=dbe_->book1D("TanLAPerTesla_TOB_4","TanLAPerTesla_TOB_4",2000,-0.5,0.5);
  MonitorElement * LA_plot_TOB_5=dbe_->book1D("TanLAPerTesla_TOB_5","TanLAPerTesla_TOB_5",2000,-0.5,0.5);
  MonitorElement * LA_plot_TOB_6=dbe_->book1D("TanLAPerTesla_TOB_6","TanLAPerTesla_TOB_6",2000,-0.5,0.5);
  
  
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
  
  gphi=-99;
  geta=-99;
  
  int fitcounter = 0;
  int histocounter = 0;
  int badfit = 0;
  int no_mod_histo = 0;
  int Layer = 0;
  
  ofstream LA_neg;
  LA_neg.open("Negative_LA.txt");
  
  for(histo=histolist.begin();histo!=histolist.end();++histo){
  
  if((*histo)->getEntries()==0){
  uint32_t id0=hidmanager.getComponentId((*histo)->getName());
  DetId detid0(id0);
  StripSubdetector subid0(id0);
  edm::LogInfo("SiStripCalibLorentzAngle")<<"### HISTOGRAM WITH 0 ENTRIES => TYPE:"<<subid0.subdetId();}
  
    if((*histo)->getEntries()>100){
      histocounter++;
      
      uint32_t id=hidmanager.getComponentId((*histo)->getName());
      DetId detid(id);
      StripSubdetector subid(id);
      
      const GeomDetUnit * stripdet;
      if(!(stripdet=tracker->idToDetUnit(subid))){
      no_mod_histo++;
      edm::LogInfo("SiStripCalibLorentzAngle")<<"### NO MODULE HISTOGRAM";}
      
      if(subid.subdetId() == int (StripSubdetector::TIB)){
      TIBDetId TIBid=TIBDetId(subid);
      Layer = TIBid.layer();
      edm::LogInfo("SiStripCalibLorentzAngle")<<"TIB layer = "<<Layer;}
      
      if(subid.subdetId() == int (StripSubdetector::TOB)){
      TOBDetId TOBid=TIBDetId(subid);
      Layer = TOBid.layer();
      edm::LogInfo("SiStripCalibLorentzAngle")<<"TOB layer = "<<Layer;}
      
      edm::LogInfo("SiStripCalibLorentzAngle")<<"id: "<<id;    
            
      if(stripdet!=0){
      const GlobalPoint gposition = (stripdet->surface()).toGlobal(p);
      gphi = gposition.phi();
      geta = gposition.eta();}
      if(stripdet==0)continue;
            
      float thickness=stripdet->specificSurface().bounds().thickness();
      const StripTopology& topol=(StripTopology&)stripdet->topology();
      float pitch = topol.localPitch(p);
      
      TProfile* theProfile=ExtractTObject<TProfile>().extract(*histo);
      
      fitfunc->SetParameter(0, 0);
      fitfunc->SetParameter(1, 0);
      fitfunc->SetParameter(2, 1);
      fitfunc->FixParameter(3, pitch);
      fitfunc->FixParameter(4, thickness);
      int fitresult=theProfile->Fit(fitfunc,"N","",ModuleRangeMin, ModuleRangeMax);
      
      if(fitfunc->GetParameter(0)>0){
      LA_neg<<"Type = "<<subid.subdetId()<<" Layer = "<<Layer;
      if(subid.subdetId() == int (StripSubdetector::TIB)){
      TIBDetId TIBid=TIBDetId(subid);
      if(TIBid.string()[0]==1){LA_neg<<" Backward, ";}else{LA_neg<<" Forward, ";}
      if(TIBid.string()[1]==1){LA_neg<<" Int String, ";}else{LA_neg<<" Ext String, ";}
      LA_neg<<" Nr. String = "<<TIBid.string()[2];}
      if(subid.subdetId() == int (StripSubdetector::TOB)){
      TOBDetId TOBid=TOBDetId(subid);
      if(TOBid.rod()[0]==1){LA_neg<<" Backward, ";}else{LA_neg<<" Forward, ";}
      LA_neg<<" Nr. Rod = "<<TOBid.rod()[1];}    
      LA_neg<<" MonoStereo = "<<subid.stereo()<<" Id = "<<id<<" => Fit_Par0 = "<<fitfunc->GetParameter(0)<<" Fit_Par1 = "<<fitfunc->GetParameter(1)<<" Fit_Par2 = "<<fitfunc->GetParameter(2)<<" Chi2/NDF = "<<fitfunc->GetChisquare()/fitfunc->GetNDF()<<std::endl<<std::endl;
      }      
                
      
      if(fitfunc->GetParameter(0)<0&&fitfunc->GetParameter(1)>0&&fitfunc->GetParameter(2)>0){
	edm::LogInfo("SiStripCalibLorentzAngle")<<fitfunc->GetParameter(0);
	fitcounter++;
	
	// get magnetic field
	const StripGeomDetUnit* det = dynamic_cast<const StripGeomDetUnit*>(estracker->idToDetUnit(detid));
	if (det==0){
	  edm::LogError("SiStripCalibLorentzAngle") << "[SiStripCalibLorentzAngle::getNewObject] the detID " << id << " doesn't seem to belong to Tracker" << std::endl; 	
	  continue;
	}
	LocalVector lbfield=(det->surface()).toLocal(magfield_->inTesla(det->surface().position()));
	float theBfield = lbfield.mag();
	theBfield = (theBfield > 0) ? theBfield : 0.00001; 
	
	float muH = -(fitfunc->GetParameter(0))/theBfield; 

	detid_la[id]= muH;
	
	histos[1]->Fill(muH);
	histos[2]->Fill(fitfunc->GetParError(0)/theBfield);
	histos[3]->Fill(fitfunc->GetChisquare()/fitfunc->GetNDF());
	histos[4]->Fill(gphi,muH);
	histos[5]->Fill(geta,muH);
	
	if(subid.subdetId() == int (StripSubdetector::TIB)){
	histos[6]->Fill(muH);
	if((Layer==1)&&(subid.stereo()==0))histos[7]->Fill(muH);
	if((Layer==1)&&(subid.stereo()==1))histos[8]->Fill(muH);
	if((Layer==2)&&(subid.stereo()==0))histos[9]->Fill(muH);
	if((Layer==2)&&(subid.stereo()==1))histos[10]->Fill(muH);
	if(Layer==3)histos[11]->Fill(muH);
	if(Layer==4)histos[12]->Fill(muH);	
	}
	
	if(subid.subdetId() == int (StripSubdetector::TOB)){
	histos[13]->Fill(muH);
	if((Layer==1)&&(subid.stereo()==0))histos[14]->Fill(muH);
	if((Layer==1)&&(subid.stereo()==1))histos[15]->Fill(muH);
	if((Layer==2)&&(subid.stereo()==0))histos[16]->Fill(muH);
	if((Layer==2)&&(subid.stereo()==1))histos[17]->Fill(muH);
	if(Layer==3)histos[18]->Fill(muH);
	if(Layer==4)histos[19]->Fill(muH);
	if(Layer==5)histos[20]->Fill(muH);
	if(Layer==6)histos[21]->Fill(muH);
	}
	
       }else{
       badfit++;}
    } 
  }
  
  edm::LogInfo("SiStripCalibLorentzAngle")<<"### NR.OF TIB AND TOB MODULES = 7932";
  edm::LogInfo("SiStripCalibLorentzAngle")<<"### NR.OF HISTOS WITH ENTRIES > 100 = "<<histocounter;
  edm::LogInfo("SiStripCalibLorentzAngle")<<"### NO MODULE HISTOS = "<<no_mod_histo;
  edm::LogInfo("SiStripCalibLorentzAngle")<<"### NR.OF GOOD FIT = "<<fitcounter;
  edm::LogInfo("SiStripCalibLorentzAngle")<<"### NR.OF BAD FIT = "<<badfit;
    
  LA_neg.close();  
    
  dbe_->save("LA_plots.root","LorentzAngle_Plots");
}
// Virtual destructor needed.

SiStripCalibLorentzAngle::~SiStripCalibLorentzAngle() {  
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
