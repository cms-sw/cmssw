#include "DQMOffline/CalibTracker/plugins/SiStripQualityHotStripIdentifierRoot.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include <iostream>
#include <fstream>
#include <sstream>

//Insert here the include to the algos
#include "CalibTracker/SiStripQuality/interface/SiStripHotStripAlgorithmFromClusterOccupancy.h"
#include "CalibTracker/SiStripQuality/interface/SiStripBadAPVAlgorithmFromClusterOccupancy.h"


SiStripQualityHotStripIdentifierRoot::SiStripQualityHotStripIdentifierRoot(const edm::ParameterSet& iConfig) : 
  ConditionDBWriter<SiStripBadStrip>::ConditionDBWriter<SiStripBadStrip>(iConfig),
  m_cacheID_(0), 
  dataLabel_(iConfig.getUntrackedParameter<std::string>("dataLabel","")),
  UseInputDB_(iConfig.getUntrackedParameter<bool>("UseInputDB",false)),
  conf_(iConfig),
  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
  filename(iConfig.getUntrackedParameter<std::string>("rootFilename","CondDB_TKCC_20X_v3_hlt_50822.root")),
  dirpath(iConfig.getUntrackedParameter<std::string>("rootDirPath",""))
{
  edm::LogInfo("SiStripQualityHotStripIdentifierRoot") << " ctor ";  
  reader = new SiStripDetInfoFileReader(fp_.fullPath());  
  
  dqmStore_ = edm::Service<DQMStore>().operator->(); 
  dqmStore_->setVerbose(iConfig.getUntrackedParameter<uint32_t>("verbosity",0)); 

  if(filename!=""){
    edm::LogInfo("SiStripQualityHotStripIdentifierRoot") << " before opening file " << filename.c_str();  
    dqmStore_->open(filename.c_str(), false, dirpath); 
    edm::LogInfo("SiStripQualityHotStripIdentifierRoot") << " after opening file ";  
  }
}


SiStripQualityHotStripIdentifierRoot::~SiStripQualityHotStripIdentifierRoot(){
  edm::LogInfo("SiStripQualityHotStripIdentifierRoot") << " dtor";
}

SiStripBadStrip* SiStripQualityHotStripIdentifierRoot::getNewObject(){

  edm::LogInfo("SiStripQualityHotStripIdentifierRoot") <<"SiStripQualityHotStripIdentifierRoot::getNewObject called"<<std::endl;

  SiStripBadStrip* obj=new SiStripBadStrip();
  
  edm::ParameterSet parameters=conf_.getParameter<edm::ParameterSet>("AlgoParameters");
  std::string AlgoName = parameters.getParameter<std::string>("AlgoName");

  if (AlgoName=="SiStripHotStripAlgorithmFromClusterOccupancy"){
    
    edm::LogInfo("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::getNewObject] call to SiStripHotStripAlgorithmFromClusterOccupancy"<<std::endl;

    theIdentifier= new SiStripHotStripAlgorithmFromClusterOccupancy(conf_);
    theIdentifier->setProbabilityThreshold(parameters.getUntrackedParameter<double>("ProbabilityThreshold",1.E-7));
    theIdentifier->setMinNumEntries(parameters.getUntrackedParameter<uint32_t>("MinNumEntries",100));
    theIdentifier->setMinNumEntriesPerStrip(parameters.getUntrackedParameter<uint32_t>("MinNumEntriesPerStrip",5));
    theIdentifier->setOccupancyThreshold(parameters.getUntrackedParameter<double>("OccupancyThreshold",1.E-5));
    theIdentifier->setNumberOfEvents(parameters.getUntrackedParameter<uint32_t>("NumberOfEvents",0));
    theIdentifier->setOutputFileName(conf_.getUntrackedParameter<std::string>("OccupancyRootFile","Occupancy.root"),conf_.getUntrackedParameter<bool>("WriteOccupancyRootFile",false));
    theIdentifier->setTrackerGeometry(_tracker);

    bookHistos();
  
    SiStripQuality* qobj = new SiStripQuality();
    theIdentifier->extractBadStrips(qobj,ClusterPositionHistoMap,SiStripQuality_);//here I insert SiStripQuality as input and get qobj as output
    
    //----------

    edm::LogInfo("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::getNewObject] copy SiStripObject in SiStripBadStrip"<<std::endl;

    std::stringstream ss;  
  
    SiStripBadStrip::RegistryIterator rIter=qobj->getRegistryVectorBegin();
    SiStripBadStrip::RegistryIterator rIterEnd=qobj->getRegistryVectorEnd();
    for(;rIter!=rIterEnd;++rIter){
      SiStripBadStrip::Range range(qobj->getDataVectorBegin()+rIter->ibegin,qobj->getDataVectorBegin()+rIter->iend);
      if ( ! obj->put(rIter->detid,range) )
	edm::LogError("SiStripQualityHotStripIdentifierRoot")<<"[SiStripQualityHotStripIdentifierRoot::getNewObject] detid already exists"<<std::endl;
    }
    edm::LogInfo("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::getNewObject] " << ss.str() << std::endl;
  }
  else if (AlgoName=="SiStripBadAPVAlgorithmFromClusterOccupancy")
    {

    edm::LogInfo("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::getNewObject] call to SiStripBadAPVAlgorithmFromClusterOccupancy"<<std::endl;

    theIdentifier2 = new SiStripBadAPVAlgorithmFromClusterOccupancy();
    theIdentifier2->setLowOccupancyThreshold(parameters.getUntrackedParameter<double>("LowOccupancyThreshold",5));
    theIdentifier2->setHighOccupancyThreshold(parameters.getUntrackedParameter<double>("HighOccupancyThreshold",10));
    theIdentifier2->setAbsoluteLowThreshold(parameters.getUntrackedParameter<double>("AbsoluteLowThreshold",0));
    theIdentifier2->setNumberIterations(parameters.getUntrackedParameter<uint32_t>("NumberIterations",2));
    theIdentifier2->setOutputFileName(conf_.getUntrackedParameter<std::string>("OccupancyRootFile","Occupancy.root"),conf_.getUntrackedParameter<bool>("WriteOccupancyRootFile",false));
    theIdentifier2->setTrackerGeometry(_tracker);

    bookHistos();
  
    SiStripQuality* qobj = new SiStripQuality();
    theIdentifier2->extractBadAPVs(qobj,ClusterPositionHistoMap);
    
    //----------

    edm::LogInfo("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::getNewObject] copy SiStripObject in SiStripBadStrip"<<std::endl;

    std::stringstream ss;  
  
    SiStripBadStrip::RegistryIterator rIter=qobj->getRegistryVectorBegin();
    SiStripBadStrip::RegistryIterator rIterEnd=qobj->getRegistryVectorEnd();
    for(;rIter!=rIterEnd;++rIter){
      SiStripBadStrip::Range range(qobj->getDataVectorBegin()+rIter->ibegin,qobj->getDataVectorBegin()+rIter->iend);
      if ( ! obj->put(rIter->detid,range) )
	edm::LogError("SiStripQualityHotStripIdentifierRoot")<<"[SiStripQualityHotStripIdentifierRoot::getNewObject] detid already exists"<<std::endl;
    }
    edm::LogInfo("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::getNewObject] " << ss.str() << std::endl;

    
    } else {
      edm::LogError("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::getNewObject] call for a unknow HotStrip identification algoritm"<<std::endl;
      
      std::vector<uint32_t> a;
      SiStripBadStrip::Range range(a.begin(),a.end());
      if ( ! obj->put(0xFFFFFFFF,range) )
	edm::LogError("SiStripQualityHotStripIdentifierRoot")<<"[SiStripQualityHotStripIdentifierRoot::getNewObject] detid already exists"<<std::endl;
    }

  return obj;
}

void SiStripQualityHotStripIdentifierRoot::algoBeginRun(const edm::Run& iRun,const edm::EventSetup& iSetup){
  iSetup.get<TrackerDigiGeometryRecord> ().get (theTrackerGeom);
  _tracker=&(* theTrackerGeom);

  if(UseInputDB_){
    unsigned long long cacheID = iSetup.get<SiStripQualityRcd>().cacheIdentifier();
    
    if (m_cacheID_ == cacheID) 
      return;
    
    m_cacheID_ = cacheID; 
    
    iSetup.get<SiStripQualityRcd>().get(dataLabel_,SiStripQuality_);
  }
}

void SiStripQualityHotStripIdentifierRoot::algoEndJob(){
  //Clear map
  ClusterPositionHistoMap.clear();
}


void SiStripQualityHotStripIdentifierRoot::bookHistos(){
  edm::LogInfo("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::bookHistos] " << dirpath << std::endl;
  std::vector<MonitorElement*> MEs = dqmStore_->getAllContents(dirpath);
							       //"DQMData/Run 50908/SiStrip/MechanicalView");
    //							       "/DQMData/Run 50908/SiStrip/Run summary/MechanicalView/TID/side_2/wheel_3/ring_2/mono_modules/module_402676874");
  edm::LogInfo("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::bookHistos] vector size " << MEs.size()<< std::endl;

  std::vector<MonitorElement*>::const_iterator iter=MEs.begin();
  std::vector<MonitorElement*>::const_iterator iterEnd=MEs.end();

  edm::ParameterSet parameters=conf_.getParameter<edm::ParameterSet>("AlgoParameters");
  bool gotNentries=true;
  if( parameters.getUntrackedParameter<uint32_t>("NumberOfEvents",0)==0 && parameters.getUntrackedParameter<double>("OccupancyThreshold",0)!=0)
    gotNentries=false;
  edm::LogInfo("SiStripQualityHotStripIdentifierRoot")<< "[SiStripQualityHotStripIdentifierRoot::bookHistos]  gotNentries flag " << gotNentries 
						      << " number of " << parameters.getUntrackedParameter<uint32_t>("NumberOfEvents",0)
						      << " occup " << parameters.getUntrackedParameter<double>("OccupancyThreshold",0)
						      << " OccupancyHisto" << parameters.getUntrackedParameter<std::string>("OccupancyHisto") << std::endl;
  
  for (; iter!=iterEnd;++iter) {
    std::string me_name = (*iter)->getName();
    
    if (!gotNentries && strstr(me_name.c_str(),"NumberOfClusterProfile__T")!=NULL && strstr(me_name.c_str(),"Total")==NULL ){
      theIdentifier->setNumberOfEvents( (int) ((TProfile*)(*iter)->getTProfile())->GetBinEntries(1) );
      gotNentries=true;
    }

    if (strstr(me_name.c_str(),(parameters.getUntrackedParameter<std::string>("OccupancyHisto")).c_str())==NULL)
      continue;

    unsigned int detid;
    char title[128];
    sprintf(title,"%s",me_name.c_str());
    char *ptr=strtok(title,"__");
    int c=0;
    while (ptr!=NULL){
      if (c==2){
	detid=atol(ptr);
	break;
      }
      ptr=strtok(NULL,"_");
      c++;
    }
    LogDebug("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::bookHistos] detid " << detid<< std::endl;

    ClusterPositionHistoMap[detid]=boost::shared_ptr<TH1F>(new TH1F(*(*iter)->getTH1F()));
    
  }
  
}

