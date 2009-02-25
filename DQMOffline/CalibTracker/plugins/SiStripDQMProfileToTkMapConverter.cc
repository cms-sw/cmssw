#include "DQMOffline/CalibTracker/plugins/SiStripDQMProfileToTkMapConverter.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"


SiStripDQMProfileToTkMapConverter::SiStripDQMProfileToTkMapConverter(const edm::ParameterSet& iConfig) : 
  conf_(iConfig), 
  filename(iConfig.getUntrackedParameter<std::string>("rootFilename","CondDB_TKCC_20X_v3_hlt_50822.root")),
  dirpath(iConfig.getUntrackedParameter<std::string>("rootDirPath","")),
  TkMapFileName_(iConfig.getUntrackedParameter<std::string>("TkMapFileName",""))
{
  edm::LogInfo("SiStripDQMProfileToTkMapConverter") << " ctor ";  
  reader = new SiStripDetInfoFileReader(fp_.fullPath());  
  
  edm::LogInfo("SiStripDQMProfileToTkMapConverter") << " before opening file " << filename.c_str();  
  dqmStore_ = edm::Service<DQMStore>().operator->(); 
  dqmStore_->setVerbose(iConfig.getUntrackedParameter<uint32_t>("verbosity",0)); 
  
  dqmStore_->open(filename.c_str(), false,dirpath); 
  edm::LogInfo("SiStripDQMProfileToTkMapConverter") << " after opening file ";  
}


SiStripDQMProfileToTkMapConverter::~SiStripDQMProfileToTkMapConverter(){
  edm::LogInfo("SiStripDQMProfileToTkMapConverter") << " dtor";
}

void SiStripDQMProfileToTkMapConverter::beginRun(const edm::Run& run, const edm::EventSetup& es){

  tkMap=new TrackerMap( "DigiOcc" );
  tkhisto=new TkHistoMap("DigiOcc","DigiOcc",-1.);
 
  //******************************
  //Get DetCabling to translate the DQM TProfile in TkMaps
  
  edm::ESHandle<SiStripDetCabling> tkmechstruct;
  es.get<SiStripDetCablingRcd>().get(tkmechstruct);
    
  // get list of active detectors from SiStripDetCabling
  std::vector<uint32_t> activeDets; 
  activeDets.clear(); // just in case
  tkmechstruct->addActiveDetectorsRawIds(activeDets);
  SiStripSubStructure substructure;
    
  std::vector<uint32_t> SelectedDetIds[34];
  size_t i=0;

  //i=0-3
  for(size_t j=1;j<5;j++)
    substructure.getTIBDetectors(activeDets,SelectedDetIds[i++],j,0,0,0);

  //i=4-9
  for(size_t j=1;j<7;j++)
    substructure.getTOBDetectors(activeDets,SelectedDetIds[i++],j,0,0);

  //i=10-12
  for(size_t j=1;j<4;j++)
    substructure.getTIDDetectors(activeDets,SelectedDetIds[i++],1,j,0,0);

  //i=13-15
  for(size_t j=1;j<4;j++)
    substructure.getTIDDetectors(activeDets,SelectedDetIds[i++],2,j,0,0);
	  
  //i=16-24
  for(size_t j=1;j<10;j++)
    substructure.getTECDetectors(activeDets,SelectedDetIds[i++],1,j,0,0,0,0);

  //i=25-33
  for(size_t j=1;j<10;j++)
    substructure.getTECDetectors(activeDets,SelectedDetIds[i++],2,j,0,0,0,0);
  
  //******************************


  //**********************
  // Get ME from the DQM root file and search for interesting histograms

  edm::LogInfo("SiStripDQMProfileToTkMapConverter") <<" [SiStripDQMProfileToTkMapConverter] dirpath " << dirpath << std::endl;
  
  std::vector<MonitorElement*> MEs = dqmStore_->getAllContents(dirpath);
  
  edm::LogInfo("SiStripQualityHotStripIdentifierRoot") <<" [SiStripQualityHotStripIdentifierRoot::bookHistos] vector size " << MEs.size()<< std::endl;

  std::vector<MonitorElement*>::const_iterator iter=MEs.begin();
  std::vector<MonitorElement*>::const_iterator iterEnd=MEs.end();

  std::stringstream ssFull, ssReport;

  const TProfile* histo;
  size_t layerIdx;
  for (; iter!=iterEnd;++iter) {
    std::string me_name = (*iter)->getName();
    
    if (strstr(me_name.c_str(),"NumberOfDigiProfile__")==NULL)
      continue;

    ssFull << me_name.c_str() << std::endl;
    layerIdx=0;
   
    std::stringstream ss;
    if(strstr(me_name.c_str(),"TIB")!=NULL){
      ss<<strstr(me_name.c_str(), "layer__")[7];
      ssReport << "\nTIB Layer " << ss.str() << "\n";
      layerIdx=atoi(ss.str().c_str());
    }else if(strstr(me_name.c_str(),"TOB")!=NULL){
      ss<<strstr(me_name.c_str(), "layer__")[7];
      layerIdx=atoi(ss.str().c_str())+4;
      ssReport << "\nTOB Layer " << ss.str() << "\n";
    }else if(strstr(me_name.c_str(),"TID")!=NULL){
      ss<<strstr(me_name.c_str(), "wheel__")[7];
      layerIdx=atoi(ss.str().c_str())+10;
      if(strstr(me_name.c_str(),"side__2")!=NULL){
	layerIdx+=3;
      }
      ssReport << "\nTID Disk " << ss.str() << "\n";
    }else if(strstr(me_name.c_str(),"TEC")!=NULL){
      ss<<strstr(me_name.c_str(), "wheel__")[7];
      layerIdx=atoi(ss.str().c_str())+16;
      if(strstr(me_name.c_str(),"side__2")!=NULL){
	layerIdx+=9;
      }
      ssReport << "\nTEC Wheel " << ss.str() << "\n";
    }      
    
    std::cout << ss.str() << " " << layerIdx << " selected dets size " << SelectedDetIds[layerIdx-1].size()<< std::endl;
    
    if(layerIdx!=0){
      histo= (TProfile *) (*(*iter)).getTProfile();
      for(int ibin=1;ibin<histo->GetNbinsX();++ibin){
	ssFull << layerIdx << " " << ibin << " " << SelectedDetIds[layerIdx-1][ibin-1] << " " <<  histo->GetBinContent(ibin) << std::endl;
	tkMap->fill(SelectedDetIds[layerIdx-1][ibin-1],1/(histo->GetBinContent(ibin)+0.0000001));
	tkhisto->fill(SelectedDetIds[layerIdx-1][ibin-1],1/(histo->GetBinContent(ibin)+0.0000001));
	
	if(histo->GetBinContent(ibin)==0)
	  ssReport << "\t" << layerIdx << " \t " << ibin << " \t " << SelectedDetIds[layerIdx-1][ibin-1] << std::endl;
      }
    }
  }
  LogDebug("SiStripDQMProfileToTkMapConverter") <<" [SiStripDQMProfileToTkMapConverter] fullPrint " << ssFull.str() << std::endl;
  edm::LogInfo("SiStripDQMProfileToTkMapConverter") <<" [SiStripDQMProfileToTkMapConverter] fullPrint " << ssReport.str() << std::endl;
}

void SiStripDQMProfileToTkMapConverter::endJob(){

  std::string filename=TkMapFileName_;
  if (filename!=""){
    edm::LogInfo("SiStripDQMProfileToTkMapConverter") <<" [SiStripDQMProfileToTkMapConverter] endJob Saving.........." << std::endl;
    tkMap->save(false,0,0,filename.c_str());
    filename.erase(filename.begin()+filename.find("."),filename.end());      
    tkhisto->save(filename+".root");
    tkhisto->saveAsCanvas(filename+"_Canvas.root",conf_.getUntrackedParameter<std::string>("TkMapDrawOption","E"));
  }

}



