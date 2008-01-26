#include "AnalysisExamples/SiStripDetectorPerformance/plugins/ClusterThr.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "sstream"

#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TNtupleD.h"
#include "TKey.h"
#include "langaus.C"

namespace cms{
  ClusterThr::ClusterThr(edm::ParameterSet const& conf): 
    conf_(conf),
    fileName_(conf.getParameter<std::string>("fileName")), 
    ClusterInfo_src_( conf.getParameter<edm::InputTag>( "ClusterInfo_src" ) ),
    NoiseMode_( conf.getParameter<uint32_t>("NoiseMode") ),
    ModulesToBeExcluded_(conf.getParameter< std::vector<uint32_t> >("ModulesToBeExcluded") ),
    subDets(conf.getParameter< std::vector<std::string> >("SubDets") ),
    layers(conf.getParameter< std::vector<uint32_t> >("Layers") ),
    ThC_(conf.getParameter< edm::ParameterSet > ("ThC") ),
    ThS_(conf.getParameter< edm::ParameterSet > ("ThS") ),
    ThN_(conf.getParameter< edm::ParameterSet > ("ThN") ),
    startThC_( ThC_.getParameter<double>("startThC") ),
    stopThC_(ThC_.getParameter<double>("stopThC") ),
    stepThC_(ThC_.getParameter<double>("stepThC") ),
    startThS_( ThS_.getParameter<double>("startThS")),
    stopThS_( ThS_.getParameter<double>("stopThS")),
    stepThS_( ThS_.getParameter<double>("stepThS")),
    startThN_( ThN_.getParameter<double>("startThN")),
    stopThN_( ThN_.getParameter<double>("stopThN")),
    stepThN_( ThN_.getParameter<double>("stepThN"))
  {
  }
  
  ClusterThr::~ClusterThr(){
  }
  
  void ClusterThr::beginRun(const edm::Run& run, const edm::EventSetup& es ) {

    es.get<SiStripDetCablingRcd>().get( SiStripDetCabling_ );

    es.get<SiStripQualityRcd>().get(SiStripQuality_);


    book();
  }

  void ClusterThr::book() {

    TFileDirectory ClusterNoise = fFile->mkdir( "ClusterNoise" );
    TFileDirectory ClusterSignal = fFile->mkdir("ClusterSignal");
    TFileDirectory ClusterStoN = fFile->mkdir("ClusterStoN");
    TFileDirectory ClusterWidth = fFile->mkdir("ClusterWidth");
    TFileDirectory ClusterPos = fFile->mkdir("ClusterPos");
    TFileDirectory ClusterNum = fFile->mkdir("ClusterNum");

    //Create histograms
    Hlist = new THashList();

    for (float Thc=startThC_;Thc<stopThC_;Thc+=stepThC_)
      for (float Ths=startThS_;Ths<stopThS_ && Ths<=Thc; Ths+=stepThS_)	
	for (float Thn=startThN_;Thn<stopThN_ && Thn<=Ths; Thn+=stepThN_)
	  for (int k=0;k<2;k++){
	    
	    char capp[128];
	    sprintf(capp,"_%s_Th_%2.1f_%2.1f_%2.1f",k==0?"S":"B",Thc,Ths,Thn);
	    TString appString(capp);
	    
	    //Cluster Width
	    name="cWidth"+appString;
	    bookHlist("TH1ClusterWidth",ClusterWidth, name, "Nstrip" );
	    
	    //Cluster Noise
	    name="cNoise"+appString;
	    bookHlist("TH1ClusterNoise",ClusterNoise, name, "ADC count" );
	    
	    //Cluster Signal
	    name="cSignal"+appString;
	    bookHlist("TH1ClusterSignal",ClusterSignal, name, "ADC count" );
	    
	    //Cluster StoN
	    name="cStoN"+appString;
	    bookHlist("TH1ClusterStoN",ClusterStoN, name );

	    //Cluster Pos
	    name="cPos"+appString;
	    bookHlist("TH1ClusterPos",ClusterPos, name, "Nstrip" );

	    //Cluster Number
	    name="cNum"+appString;
	    bookHlist("TH1ClusterNum",ClusterNum, name );
	  }


  }
  
  //------------------------------------------------------------------------------------------
  
  void ClusterThr::endJob() {  
    TNtupleD *tntuple = fFile->make<TNtupleD>("results","results","Tc:Ts:Tn:NTs:Ns:MeanWs:RmsWs:SckewWs:MPVs:FWHMs:NTb:Nb:MeanWb:RmsWb:SckewWb");

    std::vector<double> values(tntuple->GetNvar(),0);

    for (float Tc=startThC_;Tc<stopThC_;Tc+=stepThC_)
      for (float Ts=startThS_;Ts<stopThS_ && Ts<=Tc; Ts+=stepThS_)	
	for (float Tn=startThN_;Tn<stopThN_ && Tn<=Ts; Tn+=stepThN_){

	  char cappS[128],cappB[128];
	  sprintf(cappS,"_S_Th_%2.1f_%2.1f_%2.1f",Tc,Ts,Tn);
	  sprintf(cappB,"_B_Th_%2.1f_%2.1f_%2.1f",Tc,Ts,Tn);
	  TString appS(cappS);
	  TString appB(cappB);
	  
	  values[0]=Tc;
	  values[1]=Ts;
	  values[2]=Tn;
	  for (int k=0;k<2;k++){
	    if(k==0){
	      values[iNs]=((TH1F*) Hlist->FindObject("cNum"+appS))->GetMean();
	      values[iNTs]=((TH1F*) Hlist->FindObject("cWidth" +appS))->GetEntries();
	      values[iMeanWs]=((TH1F*) Hlist->FindObject("cWidth" +appS))->GetMean();
	      values[iRmsWs]=((TH1F*) Hlist->FindObject("cWidth" +appS))->GetRMS();
	      values[iSckewWs]=((TH1F*) Hlist->FindObject("cWidth" +appS))->GetSkewness();
	      TH1F *h = ((TH1F*) Hlist->FindObject("cStoN"+appS));
	      double peak=0;
	      double fwhm=0;
	      langausN(h,peak,fwhm,0,0,true,"RB");
	      LogTrace("ClusterThr") << "Fitting" << std::endl;
	      h->Write();
	      values[iMPVs]=peak;
	      values[iFWHMs]=fwhm;
	    }else{
	      values[iNb]=((TH1F*) Hlist->FindObject("cNum"+appB))->GetMean();
	      values[iNTb]=((TH1F*) Hlist->FindObject("cWidth" +appB))->GetEntries();
	      values[iMeanWb]=((TH1F*) Hlist->FindObject("cWidth" +appB))->GetMean();
	      values[iRmsWb]=((TH1F*) Hlist->FindObject("cWidth" +appB))->GetRMS();
	      values[iSckewWb]=((TH1F*) Hlist->FindObject("cWidth" +appB))->GetSkewness();
	    }
	  }
	  tntuple->Fill((double*) &values[0]);
	}  
    
    tntuple->Write();
    
    LogTrace("ClusterThr") << "[ClusterThr::endJob()] ........ Closed"<< std::endl;
    
  }

//------------------------------------------------------------------------

    void ClusterThr::analyze(const edm::Event& e, const edm::EventSetup& es){

      runNb   = e.id().run();
      eventNb = e.id().event();
      LogTrace("ClusterAnalysis") << "[ClusterThr::analyze] Processing run " << runNb << " event " << eventNb << std::endl;
            
      e.getByLabel( ClusterInfo_src_, dsvSiStripClusterInfo);
 
      edm::DetSetVector<SiStripClusterInfo>::const_iterator DSViter=dsvSiStripClusterInfo->begin();
      for (; DSViter!=dsvSiStripClusterInfo->end();DSViter++){
	uint32_t detid=DSViter->id;
	
	if (std::find(ModulesToBeExcluded_.begin(),ModulesToBeExcluded_.end(),detid)!=ModulesToBeExcluded_.end()){
	  LogTrace("ClusterThr") << "skipping module " << detid << std::endl;
	  continue;
	}
	
	SiStripDetId a(detid);
	if (a.det()!=DetId::Tracker)
	  continue;
	
	std::string subdet;
	unsigned short int layer;
	
	if ( a.subdetId() == 3 ){
	  TIBDetId b(detid);
	  subdet="TIB";
	  layer=b.layer();
	} else if ( a.subdetId() == 4 ) {
	  TIDDetId b(detid);
	  subdet="TID";
	  layer=b.wheel();
	} else if ( a.subdetId() == 5 ) {
	  TOBDetId b(detid);
	  subdet="TOB";
	  layer=b.layer();
	} else if ( a.subdetId() == 6 ) {
	  TECDetId b(detid);
	  subdet="TEC";
	  layer=b.wheel();
	} 
      
    
	if (subDets.begin()!=subDets.end())
	  if (std::find(subDets.begin(),subDets.end(),subdet)==subDets.end()){
	    LogTrace("ClusterThr") << "Skipping SubDet " << subdet << std::endl;
	    continue;
	  }
	if (layers.begin()!=layers.end())
	  if (std::find(layers.begin(),layers.end(),layer)==layers.end()){
	    LogTrace("ClusterThr") << "Skipping Layer " << layer << std::endl;
	    continue;
	  }

	std::vector<SiStripClusterInfo> vSiStripClusterInfo = DSViter->data;
	
	for (size_t icluster=0; icluster!=vSiStripClusterInfo.size(); icluster++){
	  
	  SiStripClusterInfo siStripClusterInfo=vSiStripClusterInfo[icluster];
	  //Perform quality discrimination      
	  const  edm::ParameterSet ps_b = conf_.getParameter<edm::ParameterSet>("BadModuleStudies");
	  if  ( ps_b.getParameter<bool>("Bad") ) {//it will perform Bad modules discrimination
	    short n_Apv;
	    switch((int)siStripClusterInfo.firstStrip()/128){
	    case 0:
	      n_Apv=0;
	      break;
	    case 1:
	      n_Apv=1;
	      break;
	    case 2:
	      n_Apv=2;
	      break;
	    case 3:
	      n_Apv=3;
	      break;
	    case 4:
	      n_Apv=4;
	      break;
	    case 5:
	      n_Apv=5;
	      break;
	    }
	    
	    if ( ps_b.getParameter<bool>("justGood") ){//it will analyze just good modules 
	      //	  LogTrace("SiStripQuality") << "Just good module selected " << std::endl;
	      if(SiStripQuality_->IsModuleBad(detid)){
		LogTrace("SiStripQuality") << "\n Excluding cluster on bad module " << detid << std::endl;
		continue;
	      }else if(SiStripQuality_->IsApvBad(detid, n_Apv)){
		//	    LogTrace("SiStripQuality") << "\n Excluding bad module and APV " << detid << n_Apv << std::endl;
		continue;
	      }
	    }else{
	      //	  LogTrace("SiStripQuality") << "Just bad module selected " << std::endl;
	      if(!SiStripQuality_->IsModuleBad(detid) || !SiStripQuality_->IsApvBad(detid, n_Apv)){
		//	    LogTrace("SiStripQuality") << "\n Skipping good module " << detid << std::endl;
		continue;
	      }
	    }
	  }
	  
	  bool passedSeed=true;
	  for (float Ths=startThS_;Ths<stopThS_ && passedSeed;Ths+=stepThS_){	
	    for (float Thn=startThN_;Thn<stopThN_ && Thn<=Ths && passedSeed; Thn+=stepThN_){    
	      bool passedClus=true;
	      for (float Thc=startThC_;Thc<stopThC_ && passedClus && passedSeed;Thc+=stepThC_){
		if (Thc<Ths)
		  continue;
		clusterizer(siStripClusterInfo,Thc,Ths,Thn,passedSeed,passedClus);
	      }
	    }
	  }
	}
      }  

      for (float Thc=startThC_;Thc<stopThC_;Thc+=stepThC_)
	for (float Ths=startThS_;Ths<stopThS_ && Ths<=Thc; Ths+=stepThS_)	
	  for (float Thn=startThN_;Thn<stopThN_ && Thn<=Ths; Thn+=stepThN_)
	    for (int k=0;k<2;k++){
	      
	      char capp[128];
	      sprintf(capp,"_%s_Th_%2.1f_%2.1f_%2.1f",k==0?"S":"B",Thc,Ths,Thn);
	      std::map<std::string,int>::iterator iter=cNum.find(capp);
	      if (iter!=cNum.end()){
		if(iter->second == 0) LogTrace("ClusterThr") << "Filling with a zero for " << capp << std::endl;
		((TH1F*) Hlist->FindObject("cNum"+TString(capp)))->Fill(iter->second);
		iter->second=0;
	      }
	    }
    }//close ievent loop

//------------------------------------------------------------------------
    bool ClusterThr::clusterizer(SiStripClusterInfo& siStripClusterInfo,float Thc,float Ths,float Thn,bool& passedSeed,bool& passedClus){

      LogTrace("ClusterThr") << "Clusterizer begin..." << std::endl;
      //takes the parameters for max Background threshold and the min Signal threshold to define the overlap region
      const edm::ParameterSet StoNThr_ = conf_.getParameter<edm::ParameterSet>("StoNThr");
      double StoNBmax_ = StoNThr_.getParameter<double>("StoNBmax");
      double StoNSmin_ = StoNThr_.getParameter<double>("StoNSmin");
      
      //takes the noises of the strips involved in the cluster   
      const std::vector<float>&  stripNoises_ = siStripClusterInfo.stripNoises();
      
      //Clusterizer Selection
      
      // if the max charge of the cluster is less than the SeedThr*NoiseOfTheMaxChargeStrip, it goes out of the loop
      if(siStripClusterInfo.maxCharge()<Ths*stripNoises_[siStripClusterInfo.maxPos()-siStripClusterInfo.firstStrip()]){
	passedSeed=false;
	LogTrace("ClusterThr") << "Exit from loop due to Seed at " << Thc << " " << Ths << " " << Thn << std::endl;
	return false;
      }

      LogTrace("ClusterThr") << "Seed threshold ok at " << Thc << " " << Ths << " " << Thn << std::endl;

      //initialization of the variables       
      float Signal=0;
      float Noise=0;
      float Nstrip=0;
      float Pos=0;

      //takes the amplitudes of the strips of the cluster
      const std::vector<uint16_t>&  stripAmplitudes_ = siStripClusterInfo.stripAmplitudes();
      for (size_t istrip=0;istrip<stripAmplitudes_.size();istrip++){

	//if the amplitude of the strip is less than the NeighbourThr*NoiseOfTheStrip it goes out
	if (stripAmplitudes_[istrip]<Thn*stripNoises_[istrip])
	  continue;
	Signal+=stripAmplitudes_[istrip];//increase Signal with the sum of the amplitudes of all the strips of the cluster
	Noise+=stripNoises_[istrip]*stripNoises_[istrip];//increase Noise with the sum of the strip noise squared
	Nstrip++;//increase the strip counter
	Pos+=istrip*stripAmplitudes_[istrip];//increase Pos with ??????
      }
      
      float NoiseNorm=0;//initialize Normalized Noise
      //different ways of calculating noise
      if (NoiseMode_==0){
	Noise=sqrt(Noise/Nstrip);
	NoiseNorm=Noise;
      }else if(NoiseMode_==1){
	Noise=sqrt(Noise);
	NoiseNorm=Noise/sqrt(Nstrip);
      }else{
	Noise=sqrt(Noise);
	NoiseNorm=stripNoises_[siStripClusterInfo.maxPos()-siStripClusterInfo.firstStrip()];
      }
      
      float StoN=Signal/NoiseNorm;
      Pos/=Signal;
      Pos+=siStripClusterInfo.firstStrip();
      //Cluster
      if (Signal < Thc * Noise){
	passedClus=false;
	LogTrace("ClusterThr") << "Exit from loop due to Clus at " << Thc << " " << Ths << " " << Thn << std::endl;
	return false;
      }

      LogTrace("ClusterThr") << "Cluster Threshold ok at " << Thc << " " << Ths << " " << Thn << std::endl;
	
      if (StoN>StoNBmax_ && StoN<StoNSmin_)
 	return true; //doesn't fill histos if StoN in overlap region btw Signal and Background
      
      if (StoN>StoNBmax_) {
	LogTrace("ClusterThr") << "Signal cluster" << std::endl;
      }else{
	LogTrace("ClusterThr") << "Background cluster" << std::endl;
      }

      LogTrace("ClusterThr") << "Signal " << Signal << " Noise " << Noise << " Nstrip " << Nstrip << " StoN " << StoN << std::endl;
      
      char capp[128];
      sprintf(capp,"_%s_Th_%2.1f_%2.1f_%2.1f",StoN>StoNBmax_?"S":"B",Thc,Ths,Thn);
      TString app(capp);
      cNum[capp]++;//increase cNum for this cluster and for this three thresholds
            
      LogTrace("ClusterThr") << "cNum just increased for " << capp << " " << cNum[capp] << std::endl;

      try{      
	((TH1F*) Hlist->FindObject("cSignal"+app))->Fill(Signal);
	((TH1F*) Hlist->FindObject("cNoise" +app))->Fill(NoiseNorm);
	((TH1F*) Hlist->FindObject("cStoN"  +app))->Fill(StoN);
	((TH1F*) Hlist->FindObject("cWidth" +app))->Fill(Nstrip);
	((TH1F*) Hlist->FindObject("cPos"   +app))->Fill(Pos);
      }catch(cms::Exception& e){
	edm::LogError("ClusterThr") << "[ClusterThr::fillHistos]  cms::Exception:  DetName " << e.what() ;
	}

      return true;
    }

  //---------------------------------------------------------------------------------------------

  void ClusterThr::bookHlist(char* ParameterSetLabel, TFileDirectory subDir, TString & HistoName, char* xTitle){
    Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
    TH1F* p = subDir.make<TH1F>(HistoName,HistoName,
				Parameters.getParameter<int32_t>("Nbinx"),
				Parameters.getParameter<double>("xmin"),
				Parameters.getParameter<double>("xmax")
				);
    if ( xTitle != "" )
      p->SetXTitle(xTitle);
    Hlist->Add(p);

  }
}
