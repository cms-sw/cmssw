#include "DQM/SiStripMonitorSummary/interface/SiStripQualityDQM.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "TCanvas.h"

// -----
SiStripQualityDQM::SiStripQualityDQM(const edm::EventSetup & eSetup,
                                         edm::ParameterSet const& hPSet,
                                         edm::ParameterSet const& fPSet):SiStripBaseCondObjDQM(eSetup, hPSet, fPSet){
  qualityLabel_ = fPSet.getParameter<std::string>("StripQualityLabel");

  // Build the Histo_TkMap:
  if(HistoMaps_On_ ) Tk_HM_ = new TkHistoMap("SiStrip/Histo_Map","Quality_TkMap",0.);

}
// -----



// -----
SiStripQualityDQM::~SiStripQualityDQM(){}
// -----


// -----
void SiStripQualityDQM::getActiveDetIds(const edm::EventSetup & eSetup){
  getConditionObject(eSetup);
  qualityHandle_->getDetIds(activeDetIds);

}
// -----

//================================================
// -----
void SiStripQualityDQM::fillModMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es){

  edm::ESHandle<TrackerTopology> tTopo;
  es.get<IdealGeometryRecord>().get(tTopo);
   
  ModMEs CondObj_ME;
  
  for(std::vector<uint32_t>::const_iterator detIter_ = selectedDetIds.begin();
                                            detIter_!= selectedDetIds.end();detIter_++){
    fillMEsForDet(CondObj_ME,*detIter_,tTopo);
      
  }
}    
// -----



//===================================================
// -----
void SiStripQualityDQM::fillMEsForDet(ModMEs selModME_, uint32_t selDetId_, edm::ESHandle<TrackerTopology>& tTopo){
    
  getModMEs(selModME_,selDetId_, tTopo);
  
  SiStripQuality::Range qualityRange = qualityHandle_->getRange(selDetId_);
  int nStrip =  reader->getNumberOfApvsAndStripLength(selDetId_).first*128;
  
  for( int istrip=0;istrip<nStrip;++istrip){
         selModME_.ProfileDistr->Fill(istrip+1,qualityHandle_->IsStripBad(qualityRange,istrip)?0.:1.);

  }// istrip
  
}    
// -----

//====================================================
// -----
void SiStripQualityDQM::fillSummaryMEs(const std::vector<uint32_t> & selectedDetIds, const edm::EventSetup& es){

  edm::ESHandle<TrackerTopology> tTopo;
  es.get<IdealGeometryRecord>().get(tTopo);

  for(std::vector<uint32_t>::const_iterator detIter_ = selectedDetIds.begin();
                                            detIter_!= selectedDetIds.end();detIter_++){
    fillMEsForLayer(/*SummaryMEsMap_,*/ *detIter_,tTopo);

  }

  for (std::map<uint32_t, ModMEs>::iterator iter=SummaryMEsMap_.begin(); iter!=SummaryMEsMap_.end(); iter++){

    ModMEs selME;
    selME = iter->second;

    if(hPSet_.getParameter<bool>("FillSummaryAtLayerLevel") && fPSet_.getParameter<bool>("OutputSummaryAtLayerLevelAsImage")){

      TCanvas c1("c1");
      selME.SummaryDistr->getTH1()->Draw();
      std::string name (selME.SummaryDistr->getTH1()->GetTitle());
      name+=".png";
      c1.Print(name.c_str());
    }
  }

}
// -----


//=================================================
// -----
void SiStripQualityDQM::fillMEsForLayer(/* std::map<uint32_t, ModMEs> selMEsMap_,*/ uint32_t selDetId_, edm::ESHandle<TrackerTopology>& tTopo){
  
  float numberOfBadStrips=0;
  
  SiStripHistoId hidmanager;

  if(hPSet_.getParameter<bool>("FillSummaryAtLayerLevel")){
      
    std::string hSummary_description;
    hSummary_description  = hPSet_.getParameter<std::string>("Summary_description");
      
    std::string hSummary_name; 
  
    // ----
    int subDetId_ = ((selDetId_>>25)&0x7);
  
    if( subDetId_<3 || subDetId_>6 ){ 
      edm::LogError("SiStripQualityDQM")
	<< "[SiStripQualityDQM::fillMEsForLayer] WRONG INPUT : no such subdetector type : "
	<< subDetId_ << " no folder set!" 
	<< std::endl;
      return;
    }
    // ----

    hSummary_name = hidmanager.createHistoLayer(hSummary_description, 
						"layer", 
						getLayerNameAndId(selDetId_,tTopo).first, 
						"") ;
        
    std::map<uint32_t, ModMEs>::iterator selMEsMapIter_ = SummaryMEsMap_.find(getLayerNameAndId(selDetId_,tTopo).second);
    
    ModMEs selME_;
    if ( selMEsMapIter_ != SummaryMEsMap_.end())
      selME_ =selMEsMapIter_->second;

    getSummaryMEs(selME_,selDetId_,tTopo);
  
  
    std::vector<uint32_t> sameLayerDetIds_;
    sameLayerDetIds_.clear();
    sameLayerDetIds_=GetSameLayerDetId(activeDetIds,selDetId_,tTopo);
    // -----
  //  unsigned int iBin=0;

    //%%%%%%%%%%%%%%%%%%%
    //%%% FIXME: patch to fix the bug causing double counting on each layer
    //%%%%%%%%%%%%%%%%%%%%%%%
    if(std::binary_search(alreadyFilledLayers.begin(),alreadyFilledLayers.end(),sameLayerDetIds_[0]))
      return;
    alreadyFilledLayers.push_back(sameLayerDetIds_[0]);
    sort(alreadyFilledLayers.begin(),alreadyFilledLayers.end());
    //%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%

    for(unsigned int i=0;i< sameLayerDetIds_.size(); i++){
    
      SiStripQuality::Range qualityRange = qualityHandle_->getRange(sameLayerDetIds_[i]);
      int nStrip =  reader->getNumberOfApvsAndStripLength(sameLayerDetIds_[i]).first*128;
    
      numberOfBadStrips=0;
    
      for( int istrip=0;istrip<nStrip;++istrip){
	if(qualityHandle_->IsStripBad(qualityRange,istrip)) { numberOfBadStrips++;}
      }
    
	float fr=100*float(numberOfBadStrips)/nStrip;
	selME_.SummaryDistr->Fill(i+1,fr);
	if(fr>20){
	  char c[9];
	  sprintf(c,"%d",sameLayerDetIds_[i]);
	  selME_.SummaryDistr->getTH1()->GetXaxis()->SetBinLabel(i+1,c);
	}

	// Fill the TkHistoMap with Quality output :
	if(HistoMaps_On_ ) Tk_HM_->setBinContent(sameLayerDetIds_[i], fr);

	// Fill the TkMap
	if(fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On")){
	  fillTkMap(sameLayerDetIds_[i], fr);
	}
    } 
  }//if Fill ...  
}  
// -----
 

//=============================
void SiStripQualityDQM::fillGrandSummaryMEs(const edm::EventSetup& eSetup){

  edm::ESHandle<TrackerTopology> tTopo;
  eSetup.get<IdealGeometryRecord>().get(tTopo);
     
  std::string hSummary_BadObjects_xTitle        = hPSet_.getParameter<std::string>("Summary_BadObjects_histo_xTitle");

  std::string hSummary_BadModules_name          = hPSet_.getParameter<std::string>("Summary_BadModules_histo_name");
  std::string hSummary_BadModules_yTitle        = hPSet_.getParameter<std::string>("Summary_BadModules_histo_yTitle");

  std::string hSummary_BadFibers_name          = hPSet_.getParameter<std::string>("Summary_BadFibers_histo_name");
  std::string hSummary_BadFibers_yTitle        = hPSet_.getParameter<std::string>("Summary_BadFibers_histo_yTitle");

  std::string hSummary_BadApvs_name          = hPSet_.getParameter<std::string>("Summary_BadApvs_histo_name");
  std::string hSummary_BadApvs_yTitle        = hPSet_.getParameter<std::string>("Summary_BadApvs_histo_yTitle");

  std::string hSummary_BadStrips_name          = hPSet_.getParameter<std::string>("Summary_BadStrips_histo_name");
  std::string hSummary_BadStrips_yTitle        = hPSet_.getParameter<std::string>("Summary_BadStrips_histo_yTitle");

  int NchX           = 34; 
  double LowX        = 0.5;
  double HighX       = 34.5;
                    


   
  MonitorElement *ME[4];         

  DQMStore* dqmStore_=edm::Service<DQMStore>().operator->();

  std::string FolderName=fPSet_.getParameter<std::string>("FolderName_For_QualityAndCabling_SummaryHistos");

  dqmStore_->setCurrentFolder(FolderName);

  ME[0] = dqmStore_->book1D(hSummary_BadModules_name,hSummary_BadModules_name,NchX,LowX,HighX);
  ME[0]->setAxisTitle(hSummary_BadObjects_xTitle,1);
  ME[0]->setAxisTitle(hSummary_BadModules_yTitle,2);

  ME[1] = dqmStore_->book1D(hSummary_BadFibers_name,hSummary_BadFibers_name,NchX,LowX,HighX);
  ME[1]->setAxisTitle(hSummary_BadObjects_xTitle,1);
  ME[1]->setAxisTitle(hSummary_BadFibers_yTitle,2);

  ME[2] = dqmStore_->book1D(hSummary_BadApvs_name,hSummary_BadApvs_name,NchX,LowX,HighX);
  ME[2]->setAxisTitle(hSummary_BadObjects_xTitle,1);
  ME[2]->setAxisTitle(hSummary_BadApvs_yTitle,2);

  ME[3] = dqmStore_->book1D(hSummary_BadStrips_name,hSummary_BadStrips_name,NchX,LowX,HighX);
  ME[3]->setAxisTitle(hSummary_BadObjects_xTitle,1);
  ME[3]->setAxisTitle(hSummary_BadStrips_yTitle,2);


  //==============================

  for(int i=0;i<4;++i){
    NTkBadComponent[i]=0;
    for(int j=0;j<19;++j){
      ssV[i][j].str("");
      for(int k=0;k<4;++k)
	NBadComponent[i][j][k]=0;
    }
  }

  std::stringstream ss;
  ss.str(""); 
  std::vector<uint32_t> detids=reader->getAllDetIds();
  std::vector<uint32_t>::const_iterator idet=detids.begin();
  for(;idet!=detids.end();++idet){
    ss << "detid " << (*idet) << " IsModuleUsable " << qualityHandle_->IsModuleUsable((*idet)) << "\n";
  }
  LogDebug("SiStripQualityStatistics") << ss.str() << std::endl;


  std::vector<SiStripQuality::BadComponent> BC = qualityHandle_->getBadComponentList();
  
  for (size_t i=0;i<BC.size();++i){
    
    //&&&&&&&&&&&&&
    //Full Tk
    //&&&&&&&&&&&&&

    if (BC[i].BadModule) 
      NTkBadComponent[0]++;
    if (BC[i].BadFibers) 
      NTkBadComponent[1]+= ( (BC[i].BadFibers>>2)&0x1 )+ ( (BC[i].BadFibers>>1)&0x1 ) + ( (BC[i].BadFibers)&0x1 );
    if (BC[i].BadApvs)
      NTkBadComponent[2]+= ( (BC[i].BadApvs>>5)&0x1 )+ ( (BC[i].BadApvs>>4)&0x1 ) + ( (BC[i].BadApvs>>3)&0x1 ) + 
	( (BC[i].BadApvs>>2)&0x1 )+ ( (BC[i].BadApvs>>1)&0x1 ) + ( (BC[i].BadApvs)&0x1 );

    //&&&&&&&&&&&&&&&&&
    //Single SubSyste
    //&&&&&&&&&&&&&&&&&

    int component;
    SiStripDetId a(BC[i].detid);
    if ( a.subdetId() == SiStripDetId::TIB ){
      //&&&&&&&&&&&&&&&&&
      //TIB
      //&&&&&&&&&&&&&&&&&
      
      component=tTopo->tibLayer(BC[i].detid);
      SetBadComponents(0, component, BC[i]);         

    } else if ( a.subdetId() == SiStripDetId::TID ) {
      //&&&&&&&&&&&&&&&&&
      //TID
      //&&&&&&&&&&&&&&&&&

      component=tTopo->tidSide(BC[i].detid)==2?tTopo->tidWheel(BC[i].detid):tTopo->tidWheel(BC[i].detid)+3;
      SetBadComponents(1, component, BC[i]);         

    } else if ( a.subdetId() == SiStripDetId::TOB ) {
      //&&&&&&&&&&&&&&&&&
      //TOB
      //&&&&&&&&&&&&&&&&&

      component=tTopo->tobLayer(BC[i].detid);
      SetBadComponents(2, component, BC[i]);         

    } else if ( a.subdetId() == SiStripDetId::TEC ) {
      //&&&&&&&&&&&&&&&&&
      //TEC
      //&&&&&&&&&&&&&&&&&

      component=tTopo->tecSide(BC[i].detid)==2?tTopo->tecWheel(BC[i].detid):tTopo->tecWheel(BC[i].detid)+9;
      SetBadComponents(3, component, BC[i]);         

    }    
  }

  //&&&&&&&&&&&&&&&&&&
  // Single Strip Info
  //&&&&&&&&&&&&&&&&&&

  SiStripQuality::RegistryIterator rbegin = qualityHandle_->getRegistryVectorBegin();
  SiStripQuality::RegistryIterator rend   = qualityHandle_->getRegistryVectorEnd();
  
  for (SiStripBadStrip::RegistryIterator rp=rbegin; rp != rend; ++rp) {
    uint32_t detid=rp->detid;

    int subdet=0;
    int component=0;
    SiStripDetId a(detid);
    if ( a.subdetId() == 3 ){
      subdet=0;
      component=tTopo->tibLayer(detid);
    } else if ( a.subdetId() == 4 ) {
      subdet=1;
      component=tTopo->tidSide(detid)==2?tTopo->tidWheel(detid):tTopo->tidWheel(detid)+3;
    } else if ( a.subdetId() == 5 ) {
      subdet=2;
      component=tTopo->tobLayer(detid);
    } else if ( a.subdetId() == 6 ) {
      subdet=3;
      component=tTopo->tecSide(detid)==2?tTopo->tecWheel(detid):tTopo->tecWheel(detid)+9;
    } 

    SiStripQuality::Range sqrange = SiStripQuality::Range( qualityHandle_->getDataVectorBegin()+rp->ibegin , qualityHandle_->getDataVectorBegin()+rp->iend );
        
    for(int it=0;it<sqrange.second-sqrange.first;it++){
      unsigned int range=qualityHandle_->decode( *(sqrange.first+it) ).range;
      NTkBadComponent[3]+=range;
      NBadComponent[subdet][0][3]+=range;
      NBadComponent[subdet][component][3]+=range;
    }
  }
  
 
  //&&&&&&&&&&&&&&&&&&
  // printout
  //&&&&&&&&&&&&&&&&&&

  ss.str("");
  ss << "\n-----------------\nGlobal Info\n-----------------";
  ss << "\nBadComponent \t   Modules \tFibers \tApvs\tStrips\n----------------------------------------------------------------";
  ss << "\nTracker:\t\t"<<NTkBadComponent[0]<<"\t"<<NTkBadComponent[1]<<"\t"<<NTkBadComponent[2]<<"\t"<<NTkBadComponent[3];
  ss<< "\n";
  ss << "\nTIB:\t\t\t"<<NBadComponent[0][0][0]<<"\t"<<NBadComponent[0][0][1]<<"\t"<<NBadComponent[0][0][2]<<"\t"<<NBadComponent[0][0][3];
  ss << "\nTID:\t\t\t"<<NBadComponent[1][0][0]<<"\t"<<NBadComponent[1][0][1]<<"\t"<<NBadComponent[1][0][2]<<"\t"<<NBadComponent[1][0][3];
  ss << "\nTOB:\t\t\t"<<NBadComponent[2][0][0]<<"\t"<<NBadComponent[2][0][1]<<"\t"<<NBadComponent[2][0][2]<<"\t"<<NBadComponent[2][0][3];
  ss << "\nTEC:\t\t\t"<<NBadComponent[3][0][0]<<"\t"<<NBadComponent[3][0][1]<<"\t"<<NBadComponent[3][0][2]<<"\t"<<NBadComponent[3][0][3];
  ss << "\n";

  for (int i=1;i<5;++i){
    ss << "\nTIB Layer " << i   << " :\t\t"<<NBadComponent[0][i][0]<<"\t"<<NBadComponent[0][i][1]<<"\t"<<NBadComponent[0][i][2]<<"\t"<<NBadComponent[0][i][3];
    std::stringstream binlabel;
      binlabel<<"TIB L "<<i;

    for(int j=0;j<4;j++){
      ME[j]->Fill(i,NBadComponent[0][i][j]);
      ME[j]->getTH1()->GetXaxis()->SetBinLabel(i,binlabel.str().c_str());
    }
  }
  ss << "\n";
  for (int i=1;i<4;++i){
    ss << "\nTID+ Disk " << i   << " :\t\t"<<NBadComponent[1][i][0]<<"\t"<<NBadComponent[1][i][1]<<"\t"<<NBadComponent[1][i][2]<<"\t"<<NBadComponent[1][i][3];
    std::stringstream binlabel;
      binlabel<<"TID+ D "<<i;
    
    for(int j=0;j<4;j++){
      ME[j]->Fill(i+4,NBadComponent[1][i][j]);
      ME[j]->getTH1()->GetXaxis()->SetBinLabel(i+4,binlabel.str().c_str());
    }
  }
  for (int i=4;i<7;++i){
    ss << "\nTID- Disk " << i-3 << " :\t\t"<<NBadComponent[1][i][0]<<"\t"<<NBadComponent[1][i][1]<<"\t"<<NBadComponent[1][i][2]<<"\t"<<NBadComponent[1][i][3];
    std::stringstream binlabel;
      binlabel<<"TID- D "<<i-3;

    for(int j=0;j<4;j++){
      ME[j]->Fill(i+4,NBadComponent[1][i][j]);
      ME[j]->getTH1()->GetXaxis()->SetBinLabel(i+4,binlabel.str().c_str());
    }
  }
  ss << "\n";
  for (int i=1;i<7;++i){
    ss << "\nTOB Layer " << i   << " :\t\t"<<NBadComponent[2][i][0]<<"\t"<<NBadComponent[2][i][1]<<"\t"<<NBadComponent[2][i][2]<<"\t"<<NBadComponent[2][i][3];
    std::stringstream binlabel;
      binlabel<<"TOB L "<<i;

    for(int j=0;j<4;j++){
      ME[j]->Fill(i+10,NBadComponent[2][i][j]);
      ME[j]->getTH1()->GetXaxis()->SetBinLabel(i+10,binlabel.str().c_str());
    }  
  }
  ss << "\n";
  for (int i=1;i<10;++i){
    ss << "\nTEC+ Disk " << i   << " :\t\t"<<NBadComponent[3][i][0]<<"\t"<<NBadComponent[3][i][1]<<"\t"<<NBadComponent[3][i][2]<<"\t"<<NBadComponent[3][i][3];
    std::stringstream binlabel;
      binlabel<<"TEC+ D "<<i;

    for(int j=0;j<4;j++){
      ME[j]->Fill(i+16,NBadComponent[3][i][j]);
      ME[j]->getTH1()->GetXaxis()->SetBinLabel(i+16,binlabel.str().c_str());
    }
  }
  for (int i=10;i<19;++i){
    ss << "\nTEC- Disk " << i-9 << " :\t\t"<<NBadComponent[3][i][0]<<"\t"<<NBadComponent[3][i][1]<<"\t"<<NBadComponent[3][i][2]<<"\t"<<NBadComponent[3][i][3];
    std::stringstream binlabel;
      binlabel<<"TEC- D "<<i-9;

    for(int j=0;j<4;j++){
      ME[j]->Fill(i+16,NBadComponent[3][i][j]);
      ME[j]->getTH1()->GetXaxis()->SetBinLabel(i+16,binlabel.str().c_str());
    }
  }
  ss<< "\n";

  ss << "\n----------------------------------------------------------------\n\t\t   Detid  \tModules Fibers Apvs\n----------------------------------------------------------------";
  for (int i=1;i<5;++i)
    ss << "\nTIB Layer " << i << " :" << ssV[0][i].str();
  ss << "\n";
  for (int i=1;i<4;++i)
    ss << "\nTID+ Disk " << i << " :" << ssV[1][i].str();
  for (int i=4;i<7;++i)
    ss << "\nTID- Disk " << i-3 << " :" << ssV[1][i].str();
  ss << "\n";
  for (int i=1;i<7;++i)
    ss << "\nTOB Layer " << i << " :" << ssV[2][i].str();
  ss << "\n";
  for (int i=1;i<10;++i)
    ss << "\nTEC+ Disk " << i << " :" << ssV[3][i].str();
  for (int i=10;i<19;++i)
    ss << "\nTEC- Disk " << i-9 << " :" << ssV[3][i].str();


  edm::LogInfo("SiStripQualityStatistics") << ss.str() << std::endl;

  for (int i=0; i<4; i++){
    TCanvas c1("c1");
    ME[i]->getTH1()->Draw();
    std::string name (ME[i]->getTH1()->GetTitle());
    name+=".png";
    c1.Print(name.c_str());
  }


} 

//=====================
void SiStripQualityDQM::SetBadComponents(int i, int component,SiStripQuality::BadComponent& BC){

  int napv=reader->getNumberOfApvsAndStripLength(BC.detid).first;

  ssV[i][component] << "\n\t\t " 
		    << BC.detid 
		    << " \t " << BC.BadModule << " \t " 
		    << ( (BC.BadFibers)&0x1 ) << " ";
  if (napv==4)
    ssV[i][component] << "x " <<( (BC.BadFibers>>1)&0x1 );
  
  if (napv==6)
    ssV[i][component] << ( (BC.BadFibers>>1)&0x1 ) << " "
		      << ( (BC.BadFibers>>2)&0x1 );
  ssV[i][component] << " \t " 
		    << ( (BC.BadApvs)&0x1 ) << " " 
		    << ( (BC.BadApvs>>1)&0x1 ) << " ";
  if (napv==4) 
    ssV[i][component] << "x x " << ( (BC.BadApvs>>2)&0x1 ) << " " 
		      << ( (BC.BadApvs>>3)&0x1 );
  if (napv==6) 
    ssV[i][component] << ( (BC.BadApvs>>2)&0x1 ) << " " 
		      << ( (BC.BadApvs>>3)&0x1 ) << " " 
		      << ( (BC.BadApvs>>4)&0x1 ) << " " 
		      << ( (BC.BadApvs>>5)&0x1 ) << " "; 

  if (BC.BadApvs){
    NBadComponent[i][0][2]+= ( (BC.BadApvs>>5)&0x1 )+ ( (BC.BadApvs>>4)&0x1 ) + ( (BC.BadApvs>>3)&0x1 ) + 
      ( (BC.BadApvs>>2)&0x1 )+ ( (BC.BadApvs>>1)&0x1 ) + ( (BC.BadApvs)&0x1 );
    NBadComponent[i][component][2]+= ( (BC.BadApvs>>5)&0x1 )+ ( (BC.BadApvs>>4)&0x1 ) + ( (BC.BadApvs>>3)&0x1 ) + 
      ( (BC.BadApvs>>2)&0x1 )+ ( (BC.BadApvs>>1)&0x1 ) + ( (BC.BadApvs)&0x1 );
    //    tkMap->fillc(BC.detid,0xff0000);
  }
  if (BC.BadFibers){ 
    NBadComponent[i][0][1]+= ( (BC.BadFibers>>2)&0x1 )+ ( (BC.BadFibers>>1)&0x1 ) + ( (BC.BadFibers)&0x1 );
    NBadComponent[i][component][1]+= ( (BC.BadFibers>>2)&0x1 )+ ( (BC.BadFibers>>1)&0x1 ) + ( (BC.BadFibers)&0x1 );
    //    tkMap->fillc(BC.detid,0x0000ff);
  }   
  if (BC.BadModule){
    NBadComponent[i][0][0]++;
    NBadComponent[i][component][0]++;
    //    tkMap->fillc(BC.detid,0x0);
  }
}

