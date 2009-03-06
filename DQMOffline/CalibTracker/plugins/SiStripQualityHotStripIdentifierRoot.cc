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


SiStripQualityHotStripIdentifierRoot::SiStripQualityHotStripIdentifierRoot(const edm::ParameterSet& iConfig) : 
  ConditionDBWriter<SiStripBadStrip>::ConditionDBWriter<SiStripBadStrip>(iConfig),
  conf_(iConfig), 
  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
  Cluster_src_(iConfig.getParameter<edm::InputTag>( "Cluster_src" )),
  Track_src_(iConfig.getUntrackedParameter<edm::InputTag>( "Track_src" )),
  tracksCollection_in_EventTree(iConfig.getUntrackedParameter<bool>("RemoveTrackClusters",false)),
  filename(iConfig.getUntrackedParameter<std::string>("rootFilename","CondDB_TKCC_20X_v3_hlt_50822.root")),
  dirpath(iConfig.getUntrackedParameter<std::string>("rootDirPath",""))
{
  edm::LogInfo("SiStripQualityHotStripIdentifierRoot") << " ctor ";  
  reader = new SiStripDetInfoFileReader(fp_.fullPath());  

  edm::ParameterSet pset=iConfig.getUntrackedParameter< edm::ParameterSet > ("ClusterSelection",edm::ParameterSet());
  MinClusterWidth_=pset.getUntrackedParameter<uint32_t>("minWidth",1);
  MaxClusterWidth_=pset.getUntrackedParameter<uint32_t>("maxWidth",1000);

  edm::LogInfo("SiStripQualityHotStripIdentifierRoot") << " before opening file " << filename.c_str();  
  dqmStore_ = edm::Service<DQMStore>().operator->(); 
  dqmStore_->setVerbose(iConfig.getUntrackedParameter<uint32_t>("verbosity",0)); 

  dqmStore_->open(filename.c_str(), false, dirpath); 
  edm::LogInfo("SiStripQualityHotStripIdentifierRoot") << " after opening file ";  

  bookHistos();
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

    SiStripHotStripAlgorithmFromClusterOccupancy theIdentifier;
    theIdentifier.setProbabilityThreshold(parameters.getUntrackedParameter<double>("ProbabilityThreshold",1.E-7));
    theIdentifier.setMinNumEntries(parameters.getUntrackedParameter<uint32_t>("MinNumEntries",100));
    theIdentifier.setMinNumEntriesPerStrip(parameters.getUntrackedParameter<uint32_t>("MinNumEntriesPerStrip",5));

    SiStripQuality* qobj = new SiStripQuality();
    theIdentifier.extractBadStrips(qobj,ClusterPositionHistoMap);

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

  for (; iter!=iterEnd;++iter) {
    std::string me_name = (*iter)->getName();

    //edm::LogInfo("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::bookHistos] histo " << me_name << std::endl;

    if (strstr(me_name.c_str(),"ClusterPosition__det__")==NULL)
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

