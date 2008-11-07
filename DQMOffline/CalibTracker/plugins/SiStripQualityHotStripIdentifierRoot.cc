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

    theIdentifier= new SiStripHotStripAlgorithmFromClusterOccupancy();
    theIdentifier->setProbabilityThreshold(parameters.getUntrackedParameter<double>("ProbabilityThreshold",1.E-7));
    theIdentifier->setMinNumEntries(parameters.getUntrackedParameter<uint32_t>("MinNumEntries",100));
    theIdentifier->setMinNumEntriesPerStrip(parameters.getUntrackedParameter<uint32_t>("MinNumEntriesPerStrip",5));
    theIdentifier->setOccupancyThreshold(parameters.getUntrackedParameter<double>("OccupancyThreshold",1.E-5));
    theIdentifier->setNumberOfEvents(parameters.getUntrackedParameter<uint32_t>("NumberOfEvents",0));

    bookHistos();
  
    SiStripQuality* qobj = new SiStripQuality();
    theIdentifier->extractBadStrips(qobj,ClusterPositionHistoMap);

    std::vector<double> stripOccupancy=theIdentifier->getStripOccupancy();
    
    //*FIXME
    //Dump the occupancy of flagged bad strips in an histogram
    if(conf_.getUntrackedParameter<uint32_t>("OccupancyH_Nbin",0)!=0){
      edm::LogInfo("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::getNewObject] creating Occupancy histo"<<std::endl;
      //uint32_t xmin=*std::min_element(stripOccupancy.begin(),stripOccupancy.end());
      //uint32_t xmax=*std::max_element(stripOccupancy.begin(),stripOccupancy.end());

      TFile f(conf_.getUntrackedParameter<std::string>("OccupancyRootFile","Occupancy.root").c_str(),"RECREATE");
      
      int nbin=conf_.getUntrackedParameter<uint32_t>("OccupancyH_Nbin",0);
      double xmin=conf_.getUntrackedParameter<double>("OccupancyH_Xmin",0.);
      double xmax=conf_.getUntrackedParameter<double>("OccupancyH_Xmax",0.);
      TH1F hStripOccupancy("hStripOccupancy","hStripOccupancy",nbin,xmin,xmax);
      for(size_t idx=0;idx<stripOccupancy.size();++idx){
	LogDebug("SiStripQualityHotStripIdentifierRoot") <<" Occupancy " << stripOccupancy[idx] << " idx " << idx <<std::endl;
	hStripOccupancy.Fill(stripOccupancy[idx]);
      }

      hStripOccupancy.Write();
      f.Close();
    }
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
						      << " occup " << parameters.getUntrackedParameter<double>("OccupancyThreshold",0)<< std::endl;
  
  for (; iter!=iterEnd;++iter) {
    std::string me_name = (*iter)->getName();
    
    if (!gotNentries && strstr(me_name.c_str(),"NumberOfClusterProfile__T")!=NULL && strstr(me_name.c_str(),"Total")==NULL ){
      theIdentifier->setNumberOfEvents( (int) ((TProfile*)(*iter)->getTProfile())->GetBinEntries(1) );
      gotNentries=true;
    }

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

