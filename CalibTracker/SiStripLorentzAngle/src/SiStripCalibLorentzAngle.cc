#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CalibTracker/SiStripLorentzAngle/interface/SiStripCalibLorentzAngle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CLHEP/Random/RandGauss.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include <TF1.h>
#include <TProfile.h>


SiStripCalibLorentzAngle::SiStripCalibLorentzAngle(edm::ParameterSet const& conf) : ConditionDBWriter<SiStripLorentzAngle>(conf) , conf_(conf){}


void SiStripCalibLorentzAngle::algoBeginJob(const edm::EventSetup& c){

  edm::ESHandle<TrackerGeometry> estracker;
  c.get<TrackerDigiGeometryRecord>().get(estracker);
  const TrackerGeometry *tracker=&(*estracker); 
  
  edm::ESHandle<SiStripLorentzAngle> SiStripLorentzAngle_;
  c.get<SiStripLorentzAngleRcd>().get(SiStripLorentzAngle_);
  std::map<unsigned int,float>  detid_la= SiStripLorentzAngle_->getLorentzAngles();

  DaqMonitorBEInterface* dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();
  std::string inputFile_ =conf_.getUntrackedParameter<std::string>("fileName", "LorentzAngle.root");
  dbe_->open(inputFile_);
  // use SistripHistoId for producing histogram id (and title)
  SiStripHistoId hidmanager;
  
  TF1 *fitfunc=0;
  double ModuleRangeMin=conf_.getParameter<double>("ModuleFitXMin");
  double ModuleRangeMax=conf_.getParameter<double>("ModuleFitXMax");
  double TIBRangeMin=conf_.getParameter<double>("TIBFitXMin");
  double TIBRangeMax=conf_.getParameter<double>("TIBFitXMax");
  double TOBRangeMin=conf_.getParameter<double>("TOBFitXMin");
  double TOBRangeMax=conf_.getParameter<double>("TOBFitXMax");
  fitfunc= new TF1("fitfunc","([4]/[3])*[1]*(TMath::Abs(x-[0]))+[2]",-1,1);

  std::vector<MonitorElement*> histolist= dbe_->getAllContents("/");
  std::vector<MonitorElement*>::iterator histo;
  LocalPoint p =LocalPoint(0,0,0);
  for(histo=histolist.begin();histo!=histolist.end();++histo){
    uint32_t id=hidmanager.getComponentId((*histo)->getName());
    edm::LogInfo("SiStripCalibLorentzAngle")<<"id: "<<id;
    
    DetId subid(id);
    const GeomDetUnit * stripdet=tracker->idToDetUnit(subid);
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
    if(fitfunc->GetParameter(1)>0&&fitfunc->GetParameter(2)>0){
      edm::LogInfo("SiStripCalibLorentzAngle")<<fitfunc->GetParameter(0);
      detid_la[id]=fitfunc->GetParameter(0);
    } 
  }
}
// Virtual destructor needed.

SiStripCalibLorentzAngle::~SiStripCalibLorentzAngle() {  
}  

// Analyzer: Functions that gets called by framework every event


SiStripLorentzAngle* SiStripCalibLorentzAngle::getNewObject(){

  SiStripLorentzAngle* LorentzAngle = new SiStripLorentzAngle();
  
  for(std::vector<std::pair<uint32_t, float> >::iterator it = detid_la.begin(); it != detid_la.end(); it++){
    
    float langle=it->second;
    if ( ! LorentzAngle->putLorentzAngle(it->first,langle) )
      edm::LogError("SiStripCalibLorentzAngle")<<"[SiStripCalibLorentzAngle::analyze] detid already exists"<<std::endl;
  }
  
  return LorentzAngle;
}
