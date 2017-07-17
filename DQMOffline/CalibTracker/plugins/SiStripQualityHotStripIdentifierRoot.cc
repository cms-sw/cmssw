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
#include "CalibTracker/SiStripQuality/interface/SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy.h"


SiStripQualityHotStripIdentifierRoot::SiStripQualityHotStripIdentifierRoot(const edm::ParameterSet& iConfig) : ConditionDBWriter<SiStripBadStrip>(iConfig),
  m_cacheID_(0), 
  dataLabel_(iConfig.getUntrackedParameter<std::string>("dataLabel","")),
  UseInputDB_(iConfig.getUntrackedParameter<bool>("UseInputDB",false)),
  conf_(iConfig),
  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
  _tracker(nullptr),
  tTopo(nullptr),
  filename(iConfig.getUntrackedParameter<std::string>("rootFilename","CondDB_TKCC_20X_v3_hlt_50822.root")),
  dirpath(iConfig.getUntrackedParameter<std::string>("rootDirPath","")),
  TotNumberOfEvents(0),
  MeanNumberOfCluster(0),
  calibrationthreshold(iConfig.getUntrackedParameter<uint32_t>("CalibrationThreshold",10000))
{
  reader = new SiStripDetInfoFileReader(fp_.fullPath());  
  
  dqmStore_ = edm::Service<DQMStore>().operator->(); 
  dqmStore_->setVerbose(iConfig.getUntrackedParameter<uint32_t>("verbosity",0)); 

  if(filename!=""){
    edm::LogInfo("SiStripQualityHotStripIdentifierRoot") << " before opening file " << filename.c_str();  
    dqmStore_->open(filename.c_str(), false); 
    edm::LogInfo("SiStripQualityHotStripIdentifierRoot") << " after opening file ";  
  }
}


SiStripQualityHotStripIdentifierRoot::~SiStripQualityHotStripIdentifierRoot(){
}

SiStripBadStrip* SiStripQualityHotStripIdentifierRoot::getNewObject(){

  edm::LogInfo("SiStripQualityHotStripIdentifierRoot") <<"SiStripQualityHotStripIdentifierRoot::getNewObject called"<<std::endl;

  SiStripBadStrip* obj=new SiStripBadStrip();

  edm::ParameterSet parameters=conf_.getParameter<edm::ParameterSet>("AlgoParameters");
  std::string AlgoName = parameters.getParameter<std::string>("AlgoName");

  bookHistos();

  if (TotNumberOfEvents>=calibrationthreshold)
    {
      edm::LogInfo("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::getNewObject] Total number of events is " << TotNumberOfEvents << "  which is bigger than  "<<calibrationthreshold<<". Calibration is preparing to be launched." <<std::endl;
      if (MeanNumberOfCluster > 0.01)
	{
	  edm::LogInfo("SiStripQualityHotStripIdentifierRoot") <<"Mean Number of Cluster is "<<MeanNumberOfCluster<<". Calibration is launched." <<std::endl;
	  
	  if (AlgoName=="SiStripHotStripAlgorithmFromClusterOccupancy")
	    {
	      
	      edm::LogInfo("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::getNewObject] call to SiStripHotStripAlgorithmFromClusterOccupancy"<<std::endl;
	      
	      theIdentifier= new SiStripHotStripAlgorithmFromClusterOccupancy(conf_, tTopo);
	      theIdentifier->setProbabilityThreshold(parameters.getUntrackedParameter<double>("ProbabilityThreshold",1.E-7));
	      theIdentifier->setMinNumEntries(parameters.getUntrackedParameter<uint32_t>("MinNumEntries",100));
	      theIdentifier->setMinNumEntriesPerStrip(parameters.getUntrackedParameter<uint32_t>("MinNumEntriesPerStrip",5));
	      theIdentifier->setOccupancyThreshold(parameters.getUntrackedParameter<double>("OccupancyThreshold",1.E-5));
	      theIdentifier->setNumberOfEvents(TotNumberOfEvents);
	      theIdentifier->setOutputFileName(conf_.getUntrackedParameter<std::string>("OccupancyRootFile","Occupancy.root"),conf_.getUntrackedParameter<bool>("WriteOccupancyRootFile",false));
	      theIdentifier->setTrackerGeometry(_tracker);
	      
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
	      
	      theIdentifier2 = new SiStripBadAPVAlgorithmFromClusterOccupancy(conf_, tTopo);
	      theIdentifier2->setLowOccupancyThreshold(parameters.getUntrackedParameter<double>("LowOccupancyThreshold",5));
	      theIdentifier2->setHighOccupancyThreshold(parameters.getUntrackedParameter<double>("HighOccupancyThreshold",10));
	      theIdentifier2->setAbsoluteLowThreshold(parameters.getUntrackedParameter<double>("AbsoluteLowThreshold",0));
	      theIdentifier2->setNumberIterations(parameters.getUntrackedParameter<uint32_t>("NumberIterations",2));
	      theIdentifier2->setAbsoluteOccupancyThreshold(parameters.getUntrackedParameter<double>("OccupancyThreshold",1.E-5));
	      theIdentifier2->setNumberOfEvents(TotNumberOfEvents);
	      theIdentifier2->setMinNumOfEvents();
	      theIdentifier2->setOutputFileName(conf_.getUntrackedParameter<std::string>("OccupancyRootFile","Occupancy.root"),conf_.getUntrackedParameter<bool>("WriteOccupancyRootFile",false));
	      theIdentifier2->setTrackerGeometry(_tracker);
	      
	      SiStripQuality* qobj = new SiStripQuality();
	      theIdentifier2->extractBadAPVs(qobj,ClusterPositionHistoMap,SiStripQuality_);
	      
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
	  else if (AlgoName=="SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy")
	    {
	      
	      edm::LogInfo("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::getNewObject] call to SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy"<<std::endl;
	      
	      theIdentifier3= new SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy(conf_, tTopo);
	      theIdentifier3->setProbabilityThreshold(parameters.getUntrackedParameter<double>("ProbabilityThreshold",1.E-7));
	      theIdentifier3->setMinNumEntries(parameters.getUntrackedParameter<uint32_t>("MinNumEntries",100));
	      theIdentifier3->setMinNumEntriesPerStrip(parameters.getUntrackedParameter<uint32_t>("MinNumEntriesPerStrip",5));
	      theIdentifier3->setNumberOfEvents(TotNumberOfEvents);
	      theIdentifier3->setOutputFileName(conf_.getUntrackedParameter<std::string>("OccupancyRootFile","Occupancy.root"),conf_.getUntrackedParameter<bool>("WriteOccupancyRootFile",false),conf_.getUntrackedParameter<std::string>("DQMHistoOutputFile","DQMHistos.root"),conf_.getUntrackedParameter<bool>("WriteDQMHistoOutputFile",false));
	      theIdentifier3->setTrackerGeometry(_tracker);
	      theIdentifier3->setLowOccupancyThreshold(parameters.getUntrackedParameter<double>("LowOccupancyThreshold",5));
	      theIdentifier3->setHighOccupancyThreshold(parameters.getUntrackedParameter<double>("HighOccupancyThreshold",10));
	      theIdentifier3->setAbsoluteLowThreshold(parameters.getUntrackedParameter<double>("AbsoluteLowThreshold",0));
	      theIdentifier3->setNumberIterations(parameters.getUntrackedParameter<uint32_t>("NumberIterations",2));
	      theIdentifier3->setAbsoluteOccupancyThreshold(parameters.getUntrackedParameter<double>("OccupancyThreshold",1.E-5));
	      theIdentifier3->setMinNumOfEvents();
	      
	      SiStripQuality* qobj = new SiStripQuality();
	      theIdentifier3->extractBadAPVSandStrips(qobj,ClusterPositionHistoMap,SiStripQuality_);//here I insert SiStripQuality as input and get qobj as output
	      
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
	  else
	    {
	      edm::LogError("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::getNewObject] call for a unknow HotStrip identification algoritm"<<std::endl;
	      
	      std::vector<uint32_t> a;
	      SiStripBadStrip::Range range(a.begin(),a.end());
	      if ( ! obj->put(0xFFFFFFFF,range) )
		edm::LogError("SiStripQualityHotStripIdentifierRoot")<<"[SiStripQualityHotStripIdentifierRoot::getNewObject] detid already exists"<<std::endl;
	    }
	}
      
      else
	{
	  edm::LogWarning("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::getNewObject] :: Empty Events :: " <<"Mean Number of Cluster is  " << MeanNumberOfCluster << " Calibration NOT launched" << std::endl;
	  setDoStore(false); // Don't put anything in the sqlite-file!                                                                                                
	}
    }
  else
    {
      edm::LogWarning("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::getNewObject] :: Not Enough Events :: " <<"Total number of events is  " << TotNumberOfEvents << " Calibration NOT launched" << std::endl;
      setDoStore(false); // Don't put anything in the sqlite-file!
    }
  
  return obj;
}

void SiStripQualityHotStripIdentifierRoot::algoBeginRun(const edm::Run& iRun,const edm::EventSetup& iSetup){
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  tTopo = tTopoHandle.product();
 
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
  
  // Check Number of Events
  for (; iter!=iterEnd;++iter) {
    std::string me_name = (*iter)->getName();
    
    if (!gotNentries && strstr(me_name.c_str(),"TotalNumberOfCluster__T")!=NULL && strstr(me_name.c_str(),"Profile")==NULL ){

      TotNumberOfEvents = ((TH1F*)(*iter)->getTH1F())->GetEntries();
      MeanNumberOfCluster = ((TH1F*)(*iter)->getTH1F())->GetMean();
      edm::LogInfo("SiStripQualityHotStripIdentifierRoot")<< "Total Number of Events: " << TotNumberOfEvents << std::endl;

      gotNentries=true;
      edm::LogInfo("SiStripQualityHotStripIdentifierRoot")<< "[SiStripQualityHotStripIdentifierRoot::bookHistos]  gotNentries flag " << gotNentries << std::endl;
      break;
    }
  }
  if (!gotNentries) {
    edm::LogWarning("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::bookHistos] :: Histogram with to check # of evemnts missing" <<std::endl;
  }
  for (; iter!=iterEnd;++iter) {
    std::string me_name = (*iter)->getName();
    
    if (strstr(me_name.c_str(),(parameters.getUntrackedParameter<std::string>("OccupancyHisto")).c_str())==NULL)
      continue;

    unsigned int detid=0;
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

