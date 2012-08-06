#include "CalibTracker/SiPixelTools/interface/SiPixelDQMRocLevelAnalyzer.h"

SiPixelDQMRocLevelAnalyzer::SiPixelDQMRocLevelAnalyzer(const edm::ParameterSet& iConfig):conf_(iConfig)
{


}


SiPixelDQMRocLevelAnalyzer::~SiPixelDQMRocLevelAnalyzer()
{


}

// ------------ method called to for each event  ------------
void
SiPixelDQMRocLevelAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;


#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}


// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelDQMRocLevelAnalyzer::beginJob()
{
  std::string filename = conf_.getUntrackedParameter<std::string>("fileName");
  bRS = conf_.getUntrackedParameter<bool>("barrelRocStud");
  fRS = conf_.getUntrackedParameter<bool>("endcapRocStud");
  bPixelAlive = conf_.getUntrackedParameter<bool>("pixelAliveStudy");
  double pixelAliveThresh = conf_.getUntrackedParameter<double>("pixelAliveThreshold");
  bool bThreshold = conf_.getUntrackedParameter<bool>("thresholdStudy");
  bool bNoise = conf_.getUntrackedParameter<bool>("noiseStudy");
  bool bGain = conf_.getUntrackedParameter<bool>("gainStudy");
  bool bPedestal = conf_.getUntrackedParameter<bool>("pedestalStudy");

  if(!bRS && !fRS) return;

  //Open file and get MEs
  dbe = edm::Service<DQMStore>().operator->();
  dbe->open(filename);
  mes = dbe->getAllContents("");
  std::cout << "found " << mes.size() << " monitoring elements!" << std::endl;


  //BARREL
  if(bRS){
    //PIXELALIVE
    if(bPixelAlive){
      bhPixelAlive = fs_->make<TH1F>("bpixAlive","PixelAliveSummary_mean_Barrel",11520, 0., 11520.);
      bhPixelAlive_dist = fs_->make<TH1F>("bpixAliveDist","Mean PixelAliveSummary_mean_Barrel Disbribution", 110, 0., 1.1);
    }
    //THRESHOLD
    if(bThreshold){
      bhThresholdMean = fs_->make<TH1F>("bthreshMean","ScurveThresholdSummary_mean_Barrel",11520,0.,11520.);
      bhThresholdMean_dist = fs_->make<TH1F>("bthreshMeanDist","Mean ScurveThresholdSummary_mean_Barrel Distribution",600,0.,150.);
      bhThresholdRMS = fs_->make<TH1F>("bthreshRMS","ScurveThresholdSummary_RMS_Barrel",11520,0.,11520.);
      bhThresholdRMS_dist = fs_->make<TH1F>("bthreshRMSDist","Mean ScurveThresholdSummary_RMS_Barrel Distribution",800,0.,80.);
    }
    //NOISE
    if(bNoise){
      bhNoiseMean = fs_->make<TH1F>("bnoiseMean","ScurveSigmasSummary_mean_Barrel",11520,0.,11520.);
      bhNoiseMean_dist = fs_->make<TH1F>("bnoiseMeanDist","Mean ScurveSigmasSummary_mean_Barrel Distribution",256,-2.,6.);
      bhNoiseRMS = fs_->make<TH1F>("bnoiseRMS","ScurveSigmasSummary_RMS_Barrel",11520,0.,11520.);
      bhNoiseRMS_dist = fs_->make<TH1F>("bnoiseRMSDist","Mean ScurveSigmasSummary_RMS_Barrel Distribution",768,0.,8.);
    }
    //GAIN
    if(bGain){
      bhGainMean = fs_->make<TH1F>("bgainMean","ScurveGainSummary_mean_Barrel",11520,0.,11520.);
      bhGainMean_dist = fs_->make<TH1F>("bgainMeanDist","Mean ScurveGainSummary_mean_Barrel Distribution",80,0.,8.);
      bhGainRMS = fs_->make<TH1F>("bgainRMS","ScurveGainSummary_RMS_Barrel",11520,0.,11520.);
      bhGainRMS_dist = fs_->make<TH1F>("bgainRMSDist","Mean ScurveGainSummary_RMS_Barrel Distribution",100,0.,10.);
    }
    //PEDESTAL
    if(bPedestal){
      bhPedestalMean = fs_->make<TH1F>("bpedestalMean","ScurvePedestalSummary_mean_Barrel",11520,0.,11520.);
      bhPedestalMean_dist = fs_->make<TH1F>("bpedestalMeanDist","Mean ScurvePedestalSummary_mean_Barrel Distribution",600,0.,300.);
      bhPedestalRMS = fs_->make<TH1F>("bpedestalRMS","ScurvePedestalSummary_RMS_Barrel",11520,0.,11520.);
      bhPedestalRMS_dist = fs_->make<TH1F>("bpedestalRMSDist","Mean ScurvePedestalSummary_RMS_Barrel Distribution",1000,0.,100.);
    }
  }

  //ENDCAP
  if(fRS){
    //PIXELALIVE
    if(bPixelAlive){
      ehPixelAlive = fs_->make<TH1F>("fpixAlive","PixelAliveSummary_mean_Endcap",4320, 0., 4320.);
      ehPixelAlive_dist = fs_->make<TH1F>("fpixAliveDist","Mean PixelAliveSummary_mean_Endcap Disbribution", 110, 0., 1.1);
    }
    //THRESHOLD
    if(bThreshold){
      ehThresholdMean = fs_->make<TH1F>("fthreshMean","ScurveThresholdSummary_mean_Endcap",4320,0.,4320.);
      ehThresholdMean_dist = fs_->make<TH1F>("fthreshMeanDist","Mean ScurveThresholdSummary_mean_Endcap Distribution",600,0.,150.);
      ehThresholdRMS = fs_->make<TH1F>("fthreshRMS","ScurveThresholdSummary_RMS_Endcap",4320,0.,4320.);
      ehThresholdRMS_dist = fs_->make<TH1F>("fthreshRMSDist","Mean ScurveThresholdSummary_RMS_Endcap Distribution",800,0.,80.);
    }
    //NOISE
    if(bNoise){
      ehNoiseMean = fs_->make<TH1F>("fnoiseMean","ScurveSigmasSummary_mean_Endcap",4320,0.,4320.);
      ehNoiseMean_dist = fs_->make<TH1F>("fnoiseMeanDist","Mean ScurveSigmasSummary_mean_Endcap Distribution",256,-2.,6.);
      ehNoiseRMS = fs_->make<TH1F>("fnoiseRMS","ScurveSigmasSummary_RMS_Endcap",4320,0.,4320.);
      ehNoiseRMS_dist = fs_->make<TH1F>("fnoiseRMSDist","Mean ScurveSigmasSummary_RMS_Endcap Distribution",384,0.,4.);
    }
    //GAIN
    if(bGain){
      ehGainMean = fs_->make<TH1F>("fgainMean","ScurveGainSummary_mean_Endcap",4320,0.,4320.);
      ehGainMean_dist = fs_->make<TH1F>("fgainMeanDist","Mean ScurveGainSummary_mean_Endcap Distribution",600,0.,150.);
      ehGainRMS = fs_->make<TH1F>("fgainRMS","ScurveGainSummary_RMS_Endcap",4320,0.,4320.);
      ehGainRMS_dist = fs_->make<TH1F>("fgainRMSDist","Mean ScurveGainSummary_RMS_Endcap Distribution",800,0.,80.);
    }
    //PEDESTAL
    if(bPedestal){
      ehPedestalMean = fs_->make<TH1F>("fpedestalMean","ScurvePedestalSummary_mean_Endcap",4320,0.,4320.);
      ehPedestalMean_dist = fs_->make<TH1F>("fpedestalMeanDist","Mean ScurvePedestalSummary_mean_Endcap Distribution",600,0.,150.);
      ehPedestalRMS = fs_->make<TH1F>("fpedestalRMS","ScurvePedestalSummary_RMS_Endcap",4320,0.,4320.);
      ehPedestalRMS_dist = fs_->make<TH1F>("fpedestalRMSDist","Mean ScurvePedestalSummary_RMS_Endcap Distribution",800,0.,80.);
    }
  }
  //PIXELALIVE
  if(bPixelAlive){
    RocSummary("pixelAlive_siPixelCalibDigis_");
    if(bRS){
      FillRocLevelHistos(bhPixelAlive, bhPixelAlive_dist, vbpixCN, vbpixM);

      //print a list of pixels with pixelAlive quantity below pixelAliveThresh
      for(unsigned int i=0; i<vbpixCN.size(); i++){
	
	if(vbpixM[i]<pixelAliveThresh){
	  double temp = vbpixCN[i];
	  int shell = (int)((temp-1)/2880); //0 mi, 1 mo, 2 pi, 3 po
	  temp -= shell *2880;
	  int lay = 1;
	  if(temp>576){
	    temp -= 576; lay++; 
	  } 
	  if(temp>960){
	    temp -= 960; lay++;
	  }
	  int lad = 1; 
	  if(temp > 32){
	    temp -=32; lad++;
	  }
	  while(temp>64){
	    temp -=64; lad++;
	  }
	  int mod =1; int modsize = 16;
	  if(lad ==1 || (lay == 1 && lad == 10) || (lay == 2 && lad == 16) || (lay == 3 && lad == 22) ) modsize = 8; 
	  while(temp>modsize){
	    temp -= modsize; mod ++;
	  }
	  std::cout << vbpixCN[i] << " " << vbpixM[i] << ":\n";
	  std::cout << "Shell "; 
	  switch(shell){
	  case 0: std::cout << "mI"; break;
	  case 1: std::cout << "mO"; break;
	  case 2: std::cout << "pI"; break;
	  case 3: std::cout << "pO"; break;
	  } 
	  std::cout << " Lay" << lay << " Lad" << lad << " Mod" << mod << " Chip" << temp << "\n\n"; 
	  
	}
	
      }
    }
    if(fRS){
      FillRocLevelHistos(ehPixelAlive, ehPixelAlive_dist, vfpixCN, vfpixM);
      //print a list of pixels with pixelAlive quantity below pixelAliveThresh
      for(unsigned int i=0; i<vfpixCN.size(); i++){
	if(vfpixM[i]<pixelAliveThresh){
	  double temp = vfpixCN[i];
	  int hcyl = (int)((temp-1)/1080); //0 mi, 1 mo, 2 pi, 3 po
	  temp -= hcyl *1080; 
	  int disk = 1; 
	  if(temp > 540){
	    temp -= 540; disk++;
	  }
	  int blade = 1;
	  while(temp>45){
	    temp-=45; blade++;
	  }
	  int panel = 1, mod = 1;
	  if(temp<22){
	    //panel 1
	    if(temp>16){
	      temp-=16; mod = 4;
	    }
	    else if(temp>8){
	      temp-=8; mod = 3;
	    }
	    else if(temp>2){
	      temp-=2; mod = 2;
	    }
	  } 
	  else{
	    //panel 2
	    temp-=21; panel++;
	    if(temp>14){
	      temp-=14; mod = 3;
	    }
	    else if(temp>6){
	      temp-=6; mod = 2;
	    }
	  }
	  
	  std::cout << vfpixCN[i] << " " << vfpixM[i] << ":\n";
	  std::cout << "HalfCylinder "; 
	  switch(hcyl){
	  case 0: std::cout << "mI"; break;
	  case 1: std::cout << "mO"; break;
	  case 2: std::cout << "pI"; break;
	  case 3: std::cout << "pO"; break;
	  } 
	  std::cout << " Disk" << disk << " Blade" << blade << " Panel" << panel << " Mod" << mod << " Chip" << temp << "\n\n"; 
	  
	}
      }
    }

  }
  //THRESHOLD
  if(bThreshold) {
    vbpixCN.clear(); 
    vbpixM.clear();
    vbpixSD.clear();
    vfpixCN.clear(); 
    vfpixM.clear();
    vfpixSD.clear();

    RocSummary("ScurveThresholds_siPixelCalibDigis_");
    if(bRS){
      FillRocLevelHistos(bhThresholdMean, bhThresholdMean_dist, vbpixCN, vbpixM);
      FillRocLevelHistos(bhThresholdRMS, bhThresholdRMS_dist, vbpixCN, vbpixSD);
  }
    if(fRS){
      FillRocLevelHistos(ehThresholdMean, ehThresholdMean_dist, vfpixCN, vfpixM);
      FillRocLevelHistos(ehThresholdRMS, ehThresholdRMS_dist, vfpixCN, vfpixSD);
    }
  }
  //NOISE
  if(bNoise){
    vbpixCN.clear(); 
    vbpixM.clear();
    vbpixSD.clear();
    vfpixCN.clear(); 
    vfpixM.clear();
    vfpixSD.clear();

    RocSummary("ScurveSigmas_siPixelCalibDigis_");
    if(bRS){
      FillRocLevelHistos(bhNoiseMean, bhNoiseMean_dist, vbpixCN, vbpixM);
      FillRocLevelHistos(bhNoiseRMS, bhNoiseRMS_dist, vbpixCN, vbpixSD);
    }
    if(fRS){
      FillRocLevelHistos(ehNoiseMean, ehNoiseMean_dist, vfpixCN, vfpixM);
      FillRocLevelHistos(ehNoiseRMS, ehNoiseRMS_dist, vfpixCN, vfpixSD);
    }
  }
  //GAIN
  if(bGain){
    vbpixCN.clear(); 
    vbpixM.clear();
    vbpixSD.clear();
    vfpixCN.clear(); 
    vfpixM.clear();
    vfpixSD.clear();

    RocSummary("Gain2d_siPixelCalibDigis_");
    if(bRS){
      FillRocLevelHistos(bhGainMean, bhGainMean_dist, vbpixCN, vbpixM);
      FillRocLevelHistos(bhGainRMS, bhGainRMS_dist, vbpixCN, vbpixSD);
    }
    if(fRS){
      FillRocLevelHistos(ehGainMean, ehGainMean_dist, vfpixCN, vfpixM);
      FillRocLevelHistos(ehGainRMS, ehGainRMS_dist, vfpixCN, vfpixSD);
    }
  }
  //PEDESTAL
  if(bPedestal){
    vbpixCN.clear(); 
    vbpixM.clear();
    vbpixSD.clear();
    vfpixCN.clear(); 
    vfpixM.clear();
    vfpixSD.clear();

    RocSummary("Pedestal2d_siPixelCalibDigis_");
    if(bRS){
      FillRocLevelHistos(bhPedestalMean, bhPedestalMean_dist, vbpixCN, vbpixM);
      FillRocLevelHistos(bhPedestalRMS, bhPedestalRMS_dist, vbpixCN, vbpixSD);
    }
    if(fRS){
      FillRocLevelHistos(ehPedestalMean, ehPedestalMean_dist, vfpixCN, vfpixM);
      FillRocLevelHistos(ehPedestalRMS, ehPedestalRMS_dist, vfpixCN, vfpixSD);
    }
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelDQMRocLevelAnalyzer::endJob() {
}

void 
SiPixelDQMRocLevelAnalyzer::RocSummary(std::string tagname){
    
  int maxcrow, maxccol;
  std::string name, path = "first"; 
  std::string oldPath = ""; 
  int moduleNumber = 0;
  int bchipNumber = 0; 
  int fchipNumber = 0;
  bool bbarrel = false, bforward = false, bhalfMod = false, bwasHM = false;
  bool bPFFM = false;
  bool bMNCS = false; 
  int panelNumber = -1;

  for(std::vector<MonitorElement*>::const_iterator ime = mes.begin(); ime!=mes.end(); ++ime){
    bwasHM = bhalfMod;
    //set default values
    bMNCS = false;
    bbarrel = false; bforward = false; bhalfMod = false;
    //set name, (old) path
    name = (*ime)->getName();
    oldPath = path; 
    path = (*ime)->getPathname();
 
    //determine module number if any
    if(path.find("Module_")<path.size()){
      if(path!=oldPath) {
	moduleNumber++;
	if(moduleNumber!=1) bMNCS = true;
	
      } 
    }
    else {
      if(moduleNumber>0) bMNCS = true;
      moduleNumber =0;
    }

    
    //find out location (barrel (hm), endcap)
    if(path.find("Barrel/") < path.size()) {
      bbarrel = true;
      if(path.find("H/") < path.size()) bhalfMod = true;
      }
    if(path.find("Endcap/") < path.size()) {
      bforward = true;
      panelNumber = -1;
      if(path.find("Panel_1") < path.size()) panelNumber = 1;
      if(path.find("Panel_2") < path.size()) panelNumber = 2; 

    }
    
    
    //find tagname in histoname
    if(name.find(tagname)<name.size()) bPFFM = true;
    else{
      //adjust chip number if necessary, skip ME
      if(bMNCS){
	if(!bPFFM){
	  if(bbarrel && bRS){
	    if(bPixelAlive){
	      int a = 16; 
	      if(bwasHM) a = 8; 
	      for(int i=0; i<a; i++){
		bchipNumber++;
		vbpixCN.push_back(bchipNumber);
		vbpixM.push_back(0);
	      }
	    }
	    else{
	      if(bwasHM) bchipNumber += 8;
	      else bchipNumber += 16; 
	    }
	  }
	  if(bforward && fRS){   

	    int maxcol = 2; 
	    int mod = moduleNumber; 
	    if(mod>1) mod--;
	    else{
	      if(panelNumber == 1) mod = 4; 
	      else mod = 3;
	    }
	    if(panelNumber==1 && (mod==1 || mod==4)) maxcol = 1;
	    if(bPixelAlive){
	      for(int i=0; i<maxcol*(mod + panelNumber); i++){
		fchipNumber++;
		vfpixCN.push_back(fchipNumber);
		vfpixM.push_back(0);
	      }
	    }
	    else{
	      fchipNumber += maxcol * (mod + panelNumber);
	    }
	  }
	}
	else bPFFM = false;
      }     
      continue;
    }

    //BARREL ROC LEVEL PLOTS

    if(bbarrel && bRS){
      maxccol=8, maxcrow=2;
      if(bhalfMod) {
	maxcrow=1;
      }

      RocSumOneModule(maxcrow, maxccol, (*ime), vbpixCN, vbpixM, vbpixSD, bchipNumber);

    }


    //ENDCAP ROC LEVEL PLOTS
    if(bforward && fRS){
      maxccol = moduleNumber + panelNumber; 
      maxcrow = 2; 
      if(panelNumber==1 && (moduleNumber==1 || moduleNumber==4)) maxcrow = 1;

      RocSumOneModule(maxcrow, maxccol, (*ime), vfpixCN, vfpixM, vfpixSD, fchipNumber);
    }
  }

  std::cout << "Number of Chips: b" << bchipNumber << " f" << fchipNumber << " " << tagname << std::endl;

}

void 
SiPixelDQMRocLevelAnalyzer::RocSumOneModule(int maxr, int maxc, MonitorElement* const &me, std::vector<double> &vecCN, std::vector<double> &vecMean, std::vector<double> &vecSD, int &chipNumber){

  float temp, sum, nentries; 
  for(int cr=0; cr<maxr; cr++){
    for(int cc=0; cc<maxc; cc++){
      //compute mean of 1 ROC
      chipNumber++;
      sum = 0; 
      nentries = 0;
      //sum for 1 ROC
      for(int c=1; c<53; c++){
	for(int r=1; r<81; r++){	 
	  
	  temp = me->getBinContent(52*cc+c, 80*cr+r);
	  
	  if(temp!=0.){
	    sum += temp;
	    nentries++;
	  }
	}
      }
      if(nentries==0 && bPixelAlive){
	vecCN.push_back(chipNumber);
	vecMean.push_back(0);
      }
      if(nentries!=0){
	double mean = sum/nentries;
	double avsd = 0.; 
	int ne = 0; 
	vecCN.push_back(chipNumber);
	vecMean.push_back(mean);
	
	//computing std dev.
	for(int c=1; c<53; c++){
	  for(int r=1; r<81; r++){	   
	    temp = me->getBinContent(52*cc+c, 80*cr+r);
	    if(temp!=0){ 
	      avsd += (temp-mean)*(temp-mean);
	      ne++;
	      
	    }
	  }
	}
	avsd = avsd/ne; 
	avsd = sqrt(avsd);
	vecSD.push_back(avsd);
      }
      
    }
  }
}

void 
SiPixelDQMRocLevelAnalyzer::FillRocLevelHistos(TH1F *hrocdep, TH1F *hdist, std::vector<double> &vecx, std::vector<double> &vecy){
  if(vecx.size() == vecy.size()){
    for(unsigned int i=0; i<vecx.size(); i++){
      hrocdep->Fill(vecx[i],vecy[i]);
      hdist->Fill(vecy[i]);
    }
  }
}

// //define this as a plug-in
// DEFINE_FWK_MODULE(SiPixelDQMRocLevelAnalyzer);
