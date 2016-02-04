#include "DQM/HcalMonitorTasks/interface/HcalDetDiagNoiseMonitor.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "TFile.h"
#include "TTree.h"
#include <TVector2.h>
#include <TVector3.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHFRingEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHFBitCounts.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTProducer.h" 
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "HLTrigger/HLTanalyzers/interface/JetUtil.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
// #include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
//#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "RecoMET/METAlgorithms/interface/HcalNoiseRBXArray.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

// this is to retrieve HCAL LogicalMap
#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"

#include <math.h>

using namespace reco;

////////////////////////////////////////////////////////////////////////////////////////////
static const float adc2fC[128]={-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5, 10.5,11.5,12.5,
                   13.5,15.,17.,19.,21.,23.,25.,27.,29.5,32.5,35.5,38.5,42.,46.,50.,54.5,59.5,
		   64.5,59.5,64.5,69.5,74.5,79.5,84.5,89.5,94.5,99.5,104.5,109.5,114.5,119.5,
		   124.5,129.5,137.,147.,157.,167.,177.,187.,197.,209.5,224.5,239.5,254.5,272.,
		   292.,312.,334.5,359.5,384.5,359.5,384.5,409.5,434.5,459.5,484.5,509.5,534.5,
		   559.5,584.5,609.5,634.5,659.5,684.5,709.5,747.,797.,847.,897.,947.,997.,
		   1047.,1109.5,1184.5,1259.5,1334.5,1422.,1522.,1622.,1734.5,1859.5,1984.5,
		   1859.5,1984.5,2109.5,2234.5,2359.5,2484.5,2609.5,2734.5,2859.5,2984.5,
		   3109.5,3234.5,3359.5,3484.5,3609.5,3797.,4047.,4297.,4547.,4797.,5047.,
		   5297.,5609.5,5984.5,6359.5,6734.5,7172.,7672.,8172.,8734.5,9359.5,9984.5};
////////////////////////////////////////////////////////////////////////////////////////////
static std::string subdets[11]={"HBM","HBP","HEM","HEP","HO2M","HO1M","HO0","HO1P","HO2P","HFM","HFP"};
static std::string HB_RBX[36]={
"HBM01","HBM02","HBM03","HBM04","HBM05","HBM06","HBM07","HBM08","HBM09","HBM10","HBM11","HBM12","HBM13","HBM14","HBM15","HBM16","HBM17","HBM18",
"HBP01","HBP02","HBP03","HBP04","HBP05","HBP06","HBP07","HBP08","HBP09","HBP10","HBP11","HBP12","HBP13","HBP14","HBP15","HBP16","HBP17","HBP18"};
static std::string HE_RBX[36]={
"HEM01","HEM02","HEM03","HEM04","HEM05","HEM06","HEM07","HEM08","HEM09","HEM10","HEM11","HEM12","HEM13","HEM14","HEM15","HEM16","HEM17","HEM18",
"HEP01","HEP02","HEP03","HEP04","HEP05","HEP06","HEP07","HEP08","HEP09","HEP10","HEP11","HEP12","HEP13","HEP14","HEP15","HEP16","HEP17","HEP18"};
static std::string HO_RBX[36]={
"HO2M02","HO2M04","HO2M06","HO2M08","HO2M10","HO2M12","HO1M02","HO1M04","HO1M06","HO1M08","HO1M10","HO1M12",
"HO001","HO002","HO003","HO004","HO005","HO006","HO007","HO008","HO009","HO010","HO011","HO012",
"HO1P02","HO1P04","HO1P06","HO1P08","HO1P10","HO1P12","HO2P02","HO2P04","HO2P06","HO2P08","HO2P10","HO2P12",
};


class HcalDetDiagNoiseRMData{
public:
  HcalDetDiagNoiseRMData(){
    n_th_hi=n_th_lo=0;
    energy=0;
  };
  int    n_th_hi;
  int    n_th_lo;
  double energy; 
};

class HcalDetDiagNoiseRMSummary{
public:
  HcalDetDiagNoiseRMSummary(){ 
     std::string subdets[11]={"HBM","HBP","HEM","HEP","HO2M","HO1M","HO0","HO1P","HO2P","HFM","HFP"};
     reset(); 
     for(int sd=0;sd<11;sd++) for(int sect=1;sect<=18;sect++) for(int rm=1;rm<=4;rm++){
        std::stringstream tempss;
        tempss << std::setw(2) << std::setfill('0') << sect;
        std::string rbx= subdets[sd]+tempss.str();
        HcalFrontEndId id(rbx,rm,1,1,1,1,1);
        if(id.rawId()==0) continue;
        SubDetIndex[id.rmIndex()]=sd; 
     }
     for(int i=0;i<HcalFrontEndId::maxRmIndex;i++) Ref[i]=0;
  }
  void reset(int subdet=-1){
     if(subdet==-1){
       for(int i=0;i<HcalFrontEndId::maxRmIndex;i++) AboveThHi[i]=0; 
       for(int i=0;i<11;i++) events[i]=0;
     }else{
        std::string subdets[11]={"HBM","HBP","HEM","HEP","HO2M","HO1M","HO0","HO1P","HO2P","HFM","HFP"};
        for(int sect=1;sect<=18;sect++) for(int rm=1;rm<=4;rm++){
	   std::stringstream tempss;
           tempss << std::setw(2) << std::setfill('0') << sect;
           std::string rbx= subdets[subdet]+tempss.str();
           HcalFrontEndId id(rbx,rm,1,1,1,1,1);
           if(id.rawId()==0) continue;
           AboveThHi[id.rmIndex()]=0; 
	   events[subdet]=0;
	}
     }
  }
  void SetReference(int index,double val){
     if(index<0 || index>=HcalFrontEndId::maxRmIndex) return;
     Ref[index]=val;
  } 
  double GetReference(int index){
     if(index<0 || index>=HcalFrontEndId::maxRmIndex) return 0;
     return Ref[index];
  } 
  bool GetRMStatusValue(const std::string& rbx,int rm,double *val){
     int index=GetRMindex(rbx,rm);
     if(index<0 || index>=HcalFrontEndId::maxRmIndex) return false;
     if(events[SubDetIndex[index]]>10){ *val=(double)AboveThHi[index]/(double)events[SubDetIndex[index]]; return true; }
     *val=0; return true; 
  }
  void AddNoiseStat(int rm_index){
     AboveThHi[rm_index]++;
     events[SubDetIndex[rm_index]]++;
  }
  int GetSubDetIndex(const std::string& rbx){
      return SubDetIndex[GetRMindex(rbx,2)];
  }
  
  int GetRMindex(const std::string& rbx,int rm){
      if(rbx.substr(0,3)=="HO0"){
         int sect=atoi(rbx.substr(3,2).c_str());
         if(sect>12) return -1;
	 if(rm==1 && (sect==2  || sect==3 || sect==6 || sect==7 || sect==10 || sect==11)) return -1;
         if(rm==4 && (sect==12 || sect==1 || sect==4 || sect==5 || sect==8  || sect==9 )) return -1;
      }
      if(rbx.substr(0,3)=="HO1" || rbx.substr(0,3)=="HO2"){ 
         int sect=atoi(rbx.substr(4,2).c_str());
	 if(sect>12) return -1;
         if(sect==1 || sect==3 || sect==5 || sect==7 || sect==9 || sect==11) return -1;
      }
      HcalFrontEndId id(rbx,rm,1,1,1,1,1);
      if(id.rawId()==0) return -1;
      return id.rmIndex(); 
  }
  int GetStat(int subdet){ return events[subdet]; }
private:  
  int    AboveThHi  [HcalFrontEndId::maxRmIndex];
  int    SubDetIndex[HcalFrontEndId::maxRmIndex];
  double Ref[HcalFrontEndId::maxRmIndex];
  int    events[11];
};




HcalDetDiagNoiseMonitor::HcalDetDiagNoiseMonitor(const edm::ParameterSet& ps) 
{
  ievt_=0;
  run_number=-1;
  NoisyEvents=0;
  
  Online_                = ps.getUntrackedParameter<bool>("online",false);
  mergeRuns_             = ps.getUntrackedParameter<bool>("mergeRuns",false);
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("TaskFolder","DetDiagNoiseMonitor_Hcal"); 
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;
  AllowedCalibTypes_     = ps.getUntrackedParameter<std::vector<int> > ("AllowedCalibTypes");
  skipOutOfOrderLS_      = ps.getUntrackedParameter<bool>("skipOutOfOrderLS","false");
  NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",4000);
  makeDiagnostics_       = ps.getUntrackedParameter<bool>("makeDiagnostics",false);


  UseDB            = ps.getUntrackedParameter<bool>  ("UseDB"  , false);
  ReferenceData    = ps.getUntrackedParameter<std::string>("NoiseReferenceData" ,"");
  OutputFilePath   = ps.getUntrackedParameter<std::string>("OutputFilePath", "");
  HPDthresholdHi   = ps.getUntrackedParameter<double>("NoiseThresholdHPDhi",30.0);
  HPDthresholdLo   = ps.getUntrackedParameter<double>("NoiseThresholdHPDlo",12.0);
  SiPMthreshold    = ps.getUntrackedParameter<double>("NoiseThresholdSiPM",150.0);
  SpikeThreshold   = ps.getUntrackedParameter<double>("NoiseThresholdSpike",0.06);
  UpdateEvents     = ps.getUntrackedParameter<int>   ("NoiseUpdateEvents",200);
  
  rawDataLabel_ = ps.getUntrackedParameter<edm::InputTag>("RawDataLabel",edm::InputTag("source",""));
  digiLabel_     = ps.getUntrackedParameter<edm::InputTag>("digiLabel",edm::InputTag("hcalDigis"));
  
  hlTriggerResults_				= ps.getUntrackedParameter<edm::InputTag>("HLTriggerResults",edm::InputTag("TriggerResults","","HLT"));
  MetSource_					= ps.getUntrackedParameter<edm::InputTag>("MetSource",edm::InputTag("met"));
  JetSource_          				= ps.getUntrackedParameter<edm::InputTag>("JetSource",edm::InputTag("iterativeCone5CaloJets"));
  TrackSource_          			= ps.getUntrackedParameter<edm::InputTag>("TrackSource",edm::InputTag("generalTracks"));
  VertexSource_          			= ps.getUntrackedParameter<edm::InputTag>("VertexSource",edm::InputTag("offlinePrimaryVertices"));
  UseVertexCuts_         			= ps.getUntrackedParameter<bool>("UseVertexCuts",true);
  rbxCollName_    				= ps.getUntrackedParameter<std::string>("rbxCollName","hcalnoise");
  PhysDeclaredRequirement_ 			= ps.getUntrackedParameter<std::string>("PhysDeclaredRequirement","HLT_PhysicsDeclared");
  MonitoringTriggerRequirement_			= ps.getUntrackedParameter<std::string>("MonitoringTriggerRequirement","HLT_MET100");
  UseMonitoringTrigger_				= ps.getUntrackedParameter<bool>("UseMonitoringTrigger",false);
  JetMinEt_ 					= ps.getUntrackedParameter<double>("JetMinEt",10.0);
  JetMaxEta_ 					= ps.getUntrackedParameter<double>("JetMaxEta",2.0);
  ConstituentsToJetMatchingDeltaR_ 		= ps.getUntrackedParameter<double>("ConstituentsToJetMatchingDeltaR",0.5);
  TrackMaxIp_ 					= ps.getUntrackedParameter<double>("TrackMaxIp",0.1);
  TrackMinThreshold_ 				= ps.getUntrackedParameter<double>("TrackMinThreshold",1.0);
  MinJetChargeFraction_ 			= ps.getUntrackedParameter<double>("MinJetChargeFraction",0.05);
  MaxJetHadronicEnergyFraction_ 		= ps.getUntrackedParameter<double>("MaxJetHadronicEnergyFraction",0.98);
  caloTowerCollName_				= ps.getParameter<edm::InputTag>("caloTowerCollName");

// ####################################

  lumi.clear();
  RBXSummary = 0;
  RBXCurrentSummary = 0;
// ####################################

}

HcalDetDiagNoiseMonitor::~HcalDetDiagNoiseMonitor(){}

void HcalDetDiagNoiseMonitor::cleanup(){
  if(dbe_){
    dbe_->setCurrentFolder(subdir_);
    dbe_->removeContents();
    dbe_ = 0;
  }
} 
void HcalDetDiagNoiseMonitor::reset(){}


void HcalDetDiagNoiseMonitor::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  if (debug_>1) std::cout <<"HcalDetDiagNoiseMonitor::beginRun"<<std::endl;
  HcalBaseDQMonitor::beginRun(run,c);

  if (tevt_==0) this->setup(); // set up histograms if they have not been created before
  if (mergeRuns_==false)
    this->reset();

  return;

} // void HcalNDetDiagNoiseMonitor::beginRun(...)

void HcalDetDiagNoiseMonitor::setup()
{

  // Call base class setup
  HcalBaseDQMonitor::setup();
  if (!dbe_) return;

  RBXSummary = new HcalDetDiagNoiseRMSummary();
  RBXCurrentSummary = new HcalDetDiagNoiseRMSummary();

  //char *name;
  std::string name;
  if(dbe_!=NULL){    
     dbe_->setCurrentFolder(subdir_);   
     meEVT_ = dbe_->bookInt("HcalNoiseMonitor Event Number");
     dbe_->setCurrentFolder(subdir_+"Summary Plots");
     
     name="RBX Pixel multiplicity";   PixelMult        = dbe_->book1D(name,name,73,0,73);
     name="HPD energy";               HPDEnergy        = dbe_->book1D(name,name,200,0,2500);
     name="RBX energy";               RBXEnergy        = dbe_->book1D(name,name,200,0,3500);
     name="HB RM Noise Fraction Map"; HB_RBXmapRatio   = dbe_->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HB RM Spike Map";          HB_RBXmapSpikeCnt= dbe_->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HB RM Spike Amplitude Map";HB_RBXmapSpikeAmp= dbe_->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HE RM Noise Fraction Map"; HE_RBXmapRatio   = dbe_->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HE RM Spike Map";          HE_RBXmapSpikeCnt= dbe_->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HE RM Spike Amplitude Map";HE_RBXmapSpikeAmp= dbe_->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HO RM Noise Fraction Map"; HO_RBXmapRatio   = dbe_->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HO RM Spike Map";          HO_RBXmapSpikeCnt= dbe_->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HO RM Spike Amplitude Map";HO_RBXmapSpikeAmp= dbe_->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
 
     dbe_->setCurrentFolder(subdir_+"Current Plots");
     name="HB RM Noise Fraction Map (current status)"; HB_RBXmapRatioCur = dbe_->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HE RM Noise Fraction Map (current status)"; HE_RBXmapRatioCur = dbe_->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HO RM Noise Fraction Map (current status)"; HO_RBXmapRatioCur = dbe_->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     
     std::string title="RM";
     HB_RBXmapRatio->setAxisTitle(title);
     HB_RBXmapRatioCur->setAxisTitle(title);
     HB_RBXmapSpikeAmp->setAxisTitle(title);
     HB_RBXmapSpikeCnt->setAxisTitle(title);
     HE_RBXmapRatio->setAxisTitle(title);
     HE_RBXmapRatioCur->setAxisTitle(title);
     HE_RBXmapSpikeAmp->setAxisTitle(title);
     HE_RBXmapSpikeCnt->setAxisTitle(title);
     HO_RBXmapRatio->setAxisTitle(title);
     HO_RBXmapRatioCur->setAxisTitle(title);
     HO_RBXmapSpikeAmp->setAxisTitle(title);
     HO_RBXmapSpikeCnt->setAxisTitle(title);
         
     for(int i=0;i<36;i++){
        HB_RBXmapRatio->setBinLabel(i+1,HB_RBX[i],2);
        HB_RBXmapRatioCur->setBinLabel(i+1,HB_RBX[i],2);
        HB_RBXmapSpikeAmp->setBinLabel(i+1,HB_RBX[i],2); 
        HB_RBXmapSpikeCnt->setBinLabel(i+1,HB_RBX[i],2);
        HE_RBXmapRatio->setBinLabel(i+1,HE_RBX[i],2);
        HE_RBXmapRatioCur->setBinLabel(i+1,HE_RBX[i],2);
        HE_RBXmapSpikeAmp->setBinLabel(i+1,HE_RBX[i],2);
        HE_RBXmapSpikeCnt->setBinLabel(i+1,HE_RBX[i],2);
        HO_RBXmapRatio->setBinLabel(i+1,HO_RBX[i],2);
        HO_RBXmapRatioCur->setBinLabel(i+1,HO_RBX[i],2);
        HO_RBXmapSpikeAmp->setBinLabel(i+1,HO_RBX[i],2);
        HO_RBXmapSpikeCnt->setBinLabel(i+1,HO_RBX[i],2);
     }

// ###################################################################################################################

     if(!Online_) {

       dbe_->setCurrentFolder(subdir_+"MetExpressStreamNoiseMonitoring");

  Met	= dbe_->book1D("Met","Met",200,0,2000);
  Mephi= dbe_->book1D(" Mephi"," Mephi",70,-3.5,3.5);
  Mex= dbe_->book1D("Mex","Mex",200,-1000,1000);
  SumEt= dbe_->book1D("SumEt","SumEt",200,0,2000);
  HaEtHB= dbe_->book1D("HaEtHB","HaEtHB",200,0,2000);
  HaEtHE= dbe_->book1D("HaEtHE","HaEtHE",200,0,2000);
  HaEtHF= dbe_->book1D("HaEtHF","HaEtHF",200,0,2000);
  EmEtHF= dbe_->book1D("EmEtHF","EmEtHF",200,0,2000);
  NLumiSections   = dbe_->book1D("NLumiSections","NLumiSections",1,0,1);
  Met_PhysicsCategory= dbe_->book1D("Met_PhysicsCategory","Met_PhysicsCategory",200,0,2000);
  Mephi_PhysicsCategory= dbe_->book1D("Mephi_PhysicsCategory","Mephi_PhysicsCategory",70,-3.5,3.5);
  Mex_PhysicsCategory= dbe_->book1D("Mex_PhysicsCategory","Mex_PhysicsCategory",200,-1000,1000);
  SumEt_PhysicsCategory= dbe_->book1D("SumEt_PhysicsCategory","SumEt_PhysicsCategory",200,0,2000);
  HaEtHB_PhysicsCategory= dbe_->book1D("HaEtHB_PhysicsCategory","HaEtHB_PhysicsCategory",200,0,2000);
  HaEtHE_PhysicsCategory= dbe_->book1D("HaEtHE_PhysicsCategory","HaEtHE_PhysicsCategory",200,0,2000);
  HaEtHF_PhysicsCategory= dbe_->book1D("HaEtHF_PhysicsCategory","HaEtHF_PhysicsCategory",200,0,2000);
  EmEtHF_PhysicsCategory= dbe_->book1D("EmEtHF_PhysicsCategory","EmEtHF_PhysicsCategory",200,0,2000);
  HCALFraction= dbe_->book1D("HCALFraction","HCALFraction",55,0,1.1);
  chargeFraction= dbe_->book1D("chargeFraction","chargeFraction",30,0,1.5);
  HCALFractionVSchargeFraction= dbe_->book2D("HCALFractionVSchargeFraction","HCALFractionVSchargeFraction",55,0,1.1,30,0,1.5);
  JetEt= dbe_->book1D("JetEt","JetEt",200,0,2000);
  JetEta= dbe_->book1D("JetEta","JetEta",200,-10,10);
  JetPhi= dbe_->book1D("JetPhi","JetPhi",70,-3.5,3.5);
  HCALFraction_PhysicsCategory= dbe_->book1D("HCALFraction_PhysicsCategory","HCALFraction_PhysicsCategory",55,0,1.1);
  chargeFraction_PhysicsCategory= dbe_->book1D("chargeFraction_PhysicsCategory","chargeFraction_PhysicsCategory",30,0,1.5);
  HCALFractionVSchargeFraction_PhysicsCategory= dbe_->book2D("HCALFractionVSchargeFraction_PhysicsCategory","HCALFractionVSchargeFraction_PhysicsCategory",55,0,1.1,30,0,1.5);
  JetEt_PhysicsCategory= dbe_->book1D("JetEt_PhysicsCategory","JetEt_PhysicsCategory",200,0,2000);
  JetEta_PhysicsCategory= dbe_->book1D("JetEta_PhysicsCategory","JetEta_PhysicsCategory",200,-10,10);
  JetPhi_PhysicsCategory= dbe_->book1D("JetPhi_PhysicsCategory","JetPhi_PhysicsCategory",70,-3.5,3.5);
  JetEt_TaggedAnomalous= dbe_->book1D("JetEt_TaggedAnomalous","JetEt_TaggedAnomalous",200,0,2000);
  JetEta_TaggedAnomalous= dbe_->book1D("JetEta_TaggedAnomalous","JetEta_TaggedAnomalous",200,-10,10);
  JetPhi_TaggedAnomalous= dbe_->book1D("JetPhi_TaggedAnomalous","JetPhi_TaggedAnomalous",70,-3.5,3.5);
  JetEt_TaggedAnomalous_PhysicsCategory= dbe_->book1D("JetEt_TaggedAnomalous_PhysicsCategory","JetEt_TaggedAnomalous_PhysicsCategory",200,0,2000);
  JetEta_TaggedAnomalous_PhysicsCategory= dbe_->book1D("JetEta_TaggedAnomalous_PhysicsCategory","JetEta_TaggedAnomalous_PhysicsCategory",200,-10,10);
  JetPhi_TaggedAnomalous_PhysicsCategory= dbe_->book1D("JetPhi_TaggedAnomalous_PhysicsCategory","JetPhi_TaggedAnomalous_PhysicsCategory",70,-3.5,3.5);
  HFtowerRatio= dbe_->book1D("HFtowerRatio","HFtowerRatio",30,-1.9,1.1);
  HFtowerPt= dbe_->book1D("HFtowerPt","HFtowerPt",200,0,2000);
  HFtowerEta= dbe_->book1D("HFtowerEta","HFtowerEta",200,-10,10);
  HFtowerPhi= dbe_->book1D("HFtowerPhi","HFtowerPhi",70,-3.5,3.5);
  HFtowerRatio_PhysicsCategory= dbe_->book1D("HFtowerRatio_PhysicsCategory","HFtowerRatio_PhysicsCategory",30,-1.9,1.1);;
  HFtowerPt_PhysicsCategory= dbe_->book1D("HFtowerPt_PhysicsCategory","HFtowerPt_PhysicsCategory",200,0,2000);
  HFtowerEta_PhysicsCategory= dbe_->book1D("HFtowerEta_PhysicsCategory","HFtowerEta_PhysicsCategory",200,-10,10);
  HFtowerPhi_PhysicsCategory= dbe_->book1D("HFtowerPhi_PhysicsCategory","HFtowerPhi_PhysicsCategory",70,-3.5,3.5);
  HFtowerPt_TaggedAnomalous= dbe_->book1D("HFtowerPt_TaggedAnomalous","HFtowerPt_TaggedAnomalous",200,0,2000);
  HFtowerEta_TaggedAnomalous= dbe_->book1D("HFtowerEta_TaggedAnomalous","HFtowerEta_TaggedAnomalous",200,-10,10);
  HFtowerPhi_TaggedAnomalous= dbe_->book1D("HFtowerPhi_TaggedAnomalous","HFtowerPhi_TaggedAnomalous",70,-3.5,3.5);
  HFtowerPt_TaggedAnomalous_PhysicsCategory= dbe_->book1D("HFtowerPt_TaggedAnomalous_PhysicsCategory","HFtowerPt_TaggedAnomalous_PhysicsCategory",200,0,2000);
  HFtowerEta_TaggedAnomalous_PhysicsCategory= dbe_->book1D("HFtowerEta_TaggedAnomalous_PhysicsCategory","HFtowerEta_TaggedAnomalous_PhysicsCategory",200,-10,10);
  HFtowerPhi_TaggedAnomalous_PhysicsCategory= dbe_->book1D("HFtowerPhi_TaggedAnomalous_PhysicsCategory","HFtowerPhi_TaggedAnomalous_PhysicsCategory",70,-3.5,3.5);
  RBXMaxZeros= dbe_->book1D("RBXMaxZeros","RBXMaxZeros",30,0,30);
  RBXHitsHighest= dbe_->book1D("RBXHitsHighest","RBXHitsHighest",80,0,80);
  RBXE2tsOverE10ts= dbe_->book1D("RBXE2tsOverE10ts","RBXE2tsOverE10ts",50,0,2);
  HPDHitsHighest= dbe_->book1D("HPDHitsHighest","HPDHitsHighest",20,0,20);
  HPDE2tsOverE10ts= dbe_->book1D("HPDE2tsOverE10ts","HPDE2tsOverE10ts",50,0,2);
  RBXMaxZeros_PhysicsCategory= dbe_->book1D("RBXMaxZeros_PhysicsCategory","RBXMaxZeros_PhysicsCategory",30,0,30);
  RBXHitsHighest_PhysicsCategory= dbe_->book1D("RBXHitsHighest_PhysicsCategory","RBXHitsHighest_PhysicsCategory",80,0,80);
  RBXE2tsOverE10ts_PhysicsCategory= dbe_->book1D("RBXE2tsOverE10ts_PhysicsCategory","RBXE2tsOverE10ts_PhysicsCategory",50,0,2);
  HPDHitsHighest_PhysicsCategory= dbe_->book1D("HPDHitsHighest_PhysicsCategory","HPDHitsHighest_PhysicsCategory",20,0,20);
  HPDE2tsOverE10ts_PhysicsCategory= dbe_->book1D("HPDE2tsOverE10ts_PhysicsCategory","HPDE2tsOverE10ts_PhysicsCategory",50,0,2);
  Met_TaggedHBHEAnomalous= dbe_->book1D("Met_TaggedHBHEAnomalous","Met_TaggedHBHEAnomalous",200,0,2000);
  Mephi_TaggedHBHEAnomalous= dbe_->book1D("Mephi_TaggedHBHEAnomalous","Mephi_TaggedHBHEAnomalous",70,-3.5,3.5);
  Mex_TaggedHBHEAnomalous= dbe_->book1D("Mex_TaggedHBHEAnomalous","Mex_TaggedHBHEAnomalous",200,-1000,1000);
  SumEt_TaggedHBHEAnomalous= dbe_->book1D("SumEt_TaggedHBHEAnomalous","SumEt_TaggedHBHEAnomalous",200,0,2000);
  HaEtHB_TaggedHBHEAnomalous= dbe_->book1D("HaEtHB_TaggedHBHEAnomalous","HaEtHB_TaggedHBHEAnomalous",200,0,2000);
  HaEtHE_TaggedHBHEAnomalous= dbe_->book1D("HaEtHE_TaggedHBHEAnomalous","HaEtHE_TaggedHBHEAnomalous",200,0,2000);
  HaEtHF_TaggedHBHEAnomalous= dbe_->book1D("HaEtHF_TaggedHBHEAnomalous","HaEtHF_TaggedHBHEAnomalous",200,0,2000);
  EmEtHF_TaggedHBHEAnomalous= dbe_->book1D("EmEtHF_TaggedHBHEAnomalous","EmEtHF_TaggedHBHEAnomalous",200,0,2000);
  RBXMaxZeros_TaggedHBHEAnomalous= dbe_->book1D("RBXMaxZeros_TaggedHBHEAnomalous","RBXMaxZeros_TaggedHBHEAnomalous",30,0,30);
  RBXHitsHighest_TaggedHBHEAnomalous= dbe_->book1D("RBXHitsHighest_TaggedHBHEAnomalous","TaggedHBHEAnomalous",80,0,80);
  RBXE2tsOverE10ts_TaggedHBHEAnomalous= dbe_->book1D("RBXE2tsOverE10ts_TaggedHBHEAnomalous","RBXE2tsOverE10ts_TaggedHBHEAnomalous",50,0,2);
  HPDHitsHighest_TaggedHBHEAnomalous= dbe_->book1D("HPDHitsHighest_TaggedHBHEAnomalous","HPDHitsHighest_TaggedHBHEAnomalous",20,0,20);
  HPDE2tsOverE10ts_TaggedHBHEAnomalous= dbe_->book1D("HPDE2tsOverE10ts_TaggedHBHEAnomalous","HPDE2tsOverE10ts_TaggedHBHEAnomalous",50,0,2);
  Met_TaggedHBHEAnomalous_PhysicsCategory= dbe_->book1D("Met_TaggedHBHEAnomalous_PhysicsCategory","Met_TaggedHBHEAnomalous_PhysicsCategory",200,0,2000);
  Mephi_TaggedHBHEAnomalous_PhysicsCategory= dbe_->book1D("Mephi_TaggedHBHEAnomalous_PhysicsCategory","Mephi_TaggedHBHEAnomalous_PhysicsCategory",70,-3.5,3.5);
  Mex_TaggedHBHEAnomalous_PhysicsCategory= dbe_->book1D("Mex_TaggedHBHEAnomalous_PhysicsCategory","Mex_TaggedHBHEAnomalous_PhysicsCategory",200,-1000,1000);
  SumEt_TaggedHBHEAnomalous_PhysicsCategory= dbe_->book1D("SumEt_TaggedHBHEAnomalous_PhysicsCategory","SumEt_TaggedHBHEAnomalous_PhysicsCategory",200,0,2000);
  HaEtHB_TaggedHBHEAnomalous_PhysicsCategory= dbe_->book1D("HaEtHB_TaggedHBHEAnomalous_PhysicsCategory","HaEtHB_TaggedHBHEAnomalous_PhysicsCategory",200,0,2000);
  HaEtHE_TaggedHBHEAnomalous_PhysicsCategory= dbe_->book1D("HaEtHE_TaggedHBHEAnomalous_PhysicsCategory","HaEtHE_TaggedHBHEAnomalous_PhysicsCategory",200,0,2000);
  HaEtHF_TaggedHBHEAnomalous_PhysicsCategory= dbe_->book1D("HaEtHF_TaggedHBHEAnomalous_PhysicsCategory","HaEtHF_TaggedHBHEAnomalous_PhysicsCategory",200,0,2000);
  EmEtHF_TaggedHBHEAnomalous_PhysicsCategory= dbe_->book1D("EmEtHF_TaggedHBHEAnomalous_PhysicsCategory","EmEtHF_TaggedHBHEAnomalous_PhysicsCategory",200,0,2000);
  RBXMaxZeros_TaggedHBHEAnomalous_PhysicsCategory= dbe_->book1D("RBXMaxZeros_TaggedHBHEAnomalous_PhysicsCategory","RBXMaxZeros_TaggedHBHEAnomalous_PhysicsCategory",30,0,30);
  RBXHitsHighest_TaggedHBHEAnomalous_PhysicsCategory= dbe_->book1D("RBXHitsHighest_TaggedHBHEAnomalous_PhysicsCategory","RBXHitsHighest_TaggedHBHEAnomalous_PhysicsCategory",80,0,80);
  RBXE2tsOverE10ts_TaggedHBHEAnomalous_PhysicsCategory= dbe_->book1D("RBXE2tsOverE10ts_TaggedHBHEAnomalous_PhysicsCategory","RBXE2tsOverE10ts_TaggedHBHEAnomalous_PhysicsCategory",50,0,2);
  HPDHitsHighest_TaggedHBHEAnomalous_PhysicsCategory= dbe_->book1D("HPDHitsHighest_TaggedHBHEAnomalous_PhysicsCategory","HPDHitsHighest_TaggedHBHEAnomalous_PhysicsCategory",20,0,20);
  HPDE2tsOverE10ts_TaggedHBHEAnomalous_PhysicsCategory= dbe_->book1D("HPDE2tsOverE10ts_TaggedHBHEAnomalous_PhysicsCategory","HPDE2tsOverE10ts_TaggedHBHEAnomalous_PhysicsCategory",50,0,2);
  Met_TaggedHFAnomalous= dbe_->book1D("Met_TaggedHFAnomalous","Met_TaggedHFAnomalous",200,0,2000);
  Mephi_TaggedHFAnomalous= dbe_->book1D("Mephi_TaggedHFAnomalous","Mephi_TaggedHFAnomalous",70,-3.5,3.5);
  Mex_TaggedHFAnomalous= dbe_->book1D("Mex_TaggedHFAnomalous","Mex_TaggedHFAnomalous",200,-1000,1000);
  SumEt_TaggedHFAnomalous= dbe_->book1D("SumEt_TaggedHFAnomalous","SumEt_TaggedHFAnomalous",200,0,2000);
  HaEtHB_TaggedHFAnomalous= dbe_->book1D("HaEtHB_TaggedHFAnomalous","HaEtHB_TaggedHFAnomalous",200,0,2000);
  HaEtHE_TaggedHFAnomalous= dbe_->book1D("HaEtHE_TaggedHFAnomalous","HaEtHE_TaggedHFAnomalous",200,0,2000);
  HaEtHF_TaggedHFAnomalous= dbe_->book1D("HaEtHF_TaggedHFAnomalous","HaEtHF_TaggedHFAnomalous",200,0,2000);
  EmEtHF_TaggedHFAnomalous= dbe_->book1D("EmEtHF_TaggedHFAnomalous","EmEtHF_TaggedHFAnomalous",200,0,2000);
  Met_TaggedHFAnomalous_PhysicsCategory= dbe_->book1D("Met_TaggedHFAnomalous_PhysicsCategory","Met_TaggedHFAnomalous_PhysicsCategory",200,0,2000);
  Mephi_TaggedHFAnomalous_PhysicsCategory= dbe_->book1D("Mephi_TaggedHFAnomalous_PhysicsCategory","Mephi_TaggedHFAnomalous_PhysicsCategory",70,-3.5,3.5);
  Mex_TaggedHFAnomalous_PhysicsCategory= dbe_->book1D("Mex_TaggedHFAnomalous_PhysicsCategory","Mex_TaggedHFAnomalous_PhysicsCategory",200,-1000,1000);
  SumEt_TaggedHFAnomalous_PhysicsCategory= dbe_->book1D("SumEt_TaggedHFAnomalous_PhysicsCategory","SumEt_TaggedHFAnomalous_PhysicsCategory",200,0,2000);
  HaEtHB_TaggedHFAnomalous_PhysicsCategory= dbe_->book1D("HaEtHB_TaggedHFAnomalous_PhysicsCategory","HaEtHB_TaggedHFAnomalous_PhysicsCategory",200,0,2000);
  HaEtHE_TaggedHFAnomalous_PhysicsCategory= dbe_->book1D("HaEtHE_TaggedHFAnomalous_PhysicsCategory","HaEtHE_TaggedHFAnomalous_PhysicsCategory",200,0,2000);
  HaEtHF_TaggedHFAnomalous_PhysicsCategory= dbe_->book1D("HaEtHF_TaggedHFAnomalous_PhysicsCategory","HaEtHF_TaggedHFAnomalous_PhysicsCategory",200,0,2000);
  EmEtHF_TaggedHFAnomalous_PhysicsCategory= dbe_->book1D("EmEtHF_TaggedHFAnomalous_PhysicsCategory","EmEtHF_TaggedHFAnomalous_PhysicsCategory",200,0,2000);
  Met_TaggedHBHEHFAnomalous= dbe_->book1D("Met_TaggedHBHEHFAnomalous","Met_TaggedHBHEHFAnomalous",200,0,2000);
  Mephi_TaggedHBHEHFAnomalous= dbe_->book1D("Mephi_TaggedHBHEHFAnomalous","Mephi_TaggedHBHEHFAnomalous",70,-3.5,3.5);
  Mex_TaggedHBHEHFAnomalous= dbe_->book1D("Mex_TaggedHBHEHFAnomalous","Mex_TaggedHBHEHFAnomalous",200,-1000,1000);
  SumEt_TaggedHBHEHFAnomalous= dbe_->book1D("SumEt_TaggedHBHEHFAnomalous","SumEt_TaggedHBHEHFAnomalous",200,0,2000);
  HaEtHB_TaggedHBHEHFAnomalous= dbe_->book1D("HaEtHB_TaggedHBHEHFAnomalous","HaEtHB_TaggedHBHEHFAnomalous",200,0,2000);
  HaEtHE_TaggedHBHEHFAnomalous= dbe_->book1D("HaEtHE_TaggedHBHEHFAnomalous","HaEtHE_TaggedHBHEHFAnomalous",200,0,2000);
  HaEtHF_TaggedHBHEHFAnomalous= dbe_->book1D("HaEtHF_TaggedHBHEHFAnomalous","HaEtHF_TaggedHBHEHFAnomalous",200,0,2000);
  EmEtHF_TaggedHBHEHFAnomalous= dbe_->book1D("EmEtHF_TaggedHBHEHFAnomalous","EmEtHF_TaggedHBHEHFAnomalous",200,0,2000);
  Met_TaggedHBHEHFAnomalous_PhysicsCategory= dbe_->book1D("Met_TaggedHBHEHFAnomalous_PhysicsCategory","Met_TaggedHBHEHFAnomalous_PhysicsCategory",200,0,2000);
  Mephi_TaggedHBHEHFAnomalous_PhysicsCategory= dbe_->book1D("Mephi_TaggedHBHEHFAnomalous_PhysicsCategory","Mephi_TaggedHBHEHFAnomalous_PhysicsCategory",70,-3.5,3.5);
  Mex_TaggedHBHEHFAnomalous_PhysicsCategory= dbe_->book1D("Mex_TaggedHBHEHFAnomalous_PhysicsCategory","Mex_TaggedHBHEHFAnomalous_PhysicsCategory",200,-1000,1000);
  SumEt_TaggedHBHEHFAnomalous_PhysicsCategory= dbe_->book1D("SumEt_TaggedHBHEHFAnomalous_PhysicsCategory","SumEt_TaggedHBHEHFAnomalous_PhysicsCategory",200,0,2000);
  HaEtHB_TaggedHBHEHFAnomalous_PhysicsCategory= dbe_->book1D("HaEtHB_TaggedHBHEHFAnomalous_PhysicsCategory","HaEtHB_TaggedHBHEHFAnomalous_PhysicsCategory",200,0,2000);
  HaEtHE_TaggedHBHEHFAnomalous_PhysicsCategory= dbe_->book1D("HaEtHE_TaggedHBHEHFAnomalous_PhysicsCategory","HaEtHE_TaggedHBHEHFAnomalous_PhysicsCategory",200,0,2000);
  HaEtHF_TaggedHBHEHFAnomalous_PhysicsCategory= dbe_->book1D("HaEtHF_TaggedHBHEHFAnomalous_PhysicsCategory","HaEtHF_TaggedHBHEHFAnomalous_PhysicsCategory",200,0,2000);
  EmEtHF_TaggedHBHEHFAnomalous_PhysicsCategory= dbe_->book1D("EmEtHF_TaggedHBHEHFAnomalous_PhysicsCategory","EmEtHF_TaggedHBHEHFAnomalous_PhysicsCategory",200,0,2000);


     }

// ###################################################################################################################

  } 
  ReferenceRun="UNKNOWN";
  IsReference=false;
  //LoadReference();
  gen =new HcalLogicalMapGenerator();
  lmap =new HcalLogicalMap(gen->createMap());

  return;
} 

void HcalDetDiagNoiseMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){


  if (!IsAllowedCalibType()) return;
  if (LumiInOrder(iEvent.luminosityBlock())==false) return;
  HcalBaseDQMonitor::analyze(iEvent, iSetup);

  bool isNoiseEvent=false;   
  if(!dbe_) return;

   run_number=iEvent.id().run();

   // We do not want to look at Abort Gap events
   edm::Handle<FEDRawDataCollection> rawdata;
   iEvent.getByLabel(rawDataLabel_,rawdata);
   //checking FEDs for calibration information
   for(int i=FEDNumbering::MINHCALFEDID;i<=FEDNumbering::MAXHCALFEDID; i++) {
       const FEDRawData& fedData = rawdata->FEDData(i) ;
       if ( fedData.size() < 24 ) continue ;
       if(((const HcalDCCHeader*)(fedData.data()))->getCalibType()!=hc_Null) return;
   }
  
   HcalDetDiagNoiseRMData RMs[HcalFrontEndId::maxRmIndex];
   
   edm::Handle<HBHEDigiCollection> hbhe; 
   iEvent.getByLabel(digiLabel_,hbhe);
   for(HBHEDigiCollection::const_iterator digi=hbhe->begin();digi!=hbhe->end();digi++){
     double max=-100,sum,energy=0;
     for(int i=0;i<digi->size()-1;i++){
       sum=adc2fC[digi->sample(i).adc()&0xff]+adc2fC[digi->sample(i+1).adc()&0xff]; 
       if(max<sum) max=sum;
     }
     if(max>HPDthresholdLo){
       for(int i=0;i<digi->size();i++) energy+=adc2fC[digi->sample(i).adc()&0xff]-2.5;
       HcalFrontEndId lmap_entry=lmap->getHcalFrontEndId(digi->id());
       int index=lmap_entry.rmIndex(); if(index>=HcalFrontEndId::maxRmIndex) continue;
       RMs[index].n_th_lo++;
       if(max>HPDthresholdHi){ RMs[index].n_th_hi++; isNoiseEvent=true;}
       RMs[index].energy+=energy;
     }
   }
   edm::Handle<HODigiCollection> ho; 
   iEvent.getByLabel(digiLabel_,ho);
   for(HODigiCollection::const_iterator digi=ho->begin();digi!=ho->end();digi++){
     double max=-100,energy=0; int Eta=digi->id().ieta(); int Phi=digi->id().iphi();
     for(int i=0;i<digi->size()-1;i++){
       if(max<adc2fC[digi->sample(i).adc()&0xff]) max=adc2fC[digi->sample(i).adc()&0xff];
     }
     if((Eta>=11 && Eta<=15 && Phi>=59 && Phi<=70) || (Eta>=5 && Eta<=10 && Phi>=47 && Phi<=58)){
       if(max>SiPMthreshold){
	 for(int i=0;i<digi->size();i++) energy+=adc2fC[digi->sample(i).adc()&0xff]-11.0;
	 HcalFrontEndId lmap_entry=lmap->getHcalFrontEndId(digi->id());
	 int index=lmap_entry.rmIndex(); if(index>=HcalFrontEndId::maxRmIndex) continue;
	 RMs[index].n_th_hi++; isNoiseEvent=true;
	 RMs[index].energy+=energy;
	        }	          
     }else{
       if(max>HPDthresholdLo){
	 for(int i=0;i<digi->size();i++) energy+=adc2fC[digi->sample(i).adc()&0xff]-2.5;
	 HcalFrontEndId lmap_entry=lmap->getHcalFrontEndId(digi->id());
	 int index=lmap_entry.rmIndex(); if(index>=HcalFrontEndId::maxRmIndex) continue;
	 RMs[index].n_th_lo++;
	 if(max>HPDthresholdHi){ RMs[index].n_th_hi++; isNoiseEvent=true;}
	 RMs[index].energy+=energy;
       }
     }		          
   }   

   if(isNoiseEvent){
     NoisyEvents++;
     
     // RMs loop
     for(int i=0;i<HcalFrontEndId::maxRmIndex;i++){
       if(RMs[i].n_th_hi>0){
	 RBXCurrentSummary->AddNoiseStat(i);
	 RBXSummary->AddNoiseStat(i);
	 HPDEnergy->Fill(RMs[i].energy);
       }
     }
   }  
   // RBX loop
   for(int sd=0;sd<9;sd++) for(int sect=1;sect<=18;sect++){
     std::stringstream tempss;
     tempss << std::setw(2) << std::setfill('0') << sect;
     std::string rbx= subdets[sd]+tempss.str();
     
     double rbx_energy=0;int pix_mult=0; bool isValidRBX=false;
     for(int rm=1;rm<=4;rm++){
       int index=RBXSummary->GetRMindex(rbx,rm);
       if(index>0 && index<HcalFrontEndId::maxRmIndex){
	 rbx_energy+=RMs[index].energy;
	 pix_mult+=RMs[index].n_th_lo; 
	 isValidRBX=true;
       }
     }
     if(isValidRBX){
       PixelMult->Fill(pix_mult);
       RBXEnergy->Fill(rbx_energy);
     }
   }
   
   UpdateHistos();

   // ###################################################################################################################

   if(!Online_) {

     // hlt trigger results
     edm::Handle<edm::TriggerResults> hltTriggerResultHandle;

     if (!iEvent.getByLabel(hlTriggerResults_, hltTriggerResultHandle))
       {
	 if (debug_>0) edm::LogWarning("HcalMonitorTasks")<<" HcalDetDiagNoiseMonitor:  Trigger results handle "<<hlTriggerResults_<<" not found!";
	 return;
       }

     bool useEventForMonitoring = false;
     bool passedPhysDeclared = false;
     // Require a valid handle
     if(!hltTriggerResultHandle.isValid()) { std::cout << "invalid handle for HLT TriggerResults" << std::endl; }
     else {
       // # of triggers
       int ntrigs = hltTriggerResultHandle->size();

       const edm::TriggerNames & triggerNames = iEvent.triggerNames(*hltTriggerResultHandle);

       //       triggerNames_.init(* hltTriggerResultHandle);
       for (int itrig = 0; itrig != ntrigs; ++itrig){
         // obtain the trigger name
//         string trigName = triggerNames_.triggerName(itrig);
         std::string trigName = triggerNames.triggerName(itrig);

         // did the trigger fire?
         bool accept = hltTriggerResultHandle->accept(itrig);
         if(UseMonitoringTrigger_) {
           if((trigName == MonitoringTriggerRequirement_) && (accept)) {useEventForMonitoring = true;}
         } else {
           useEventForMonitoring = true;
         }
         if( ((trigName == PhysDeclaredRequirement_) && (accept)) ) {passedPhysDeclared = true;}
       }
     }
     if(!(useEventForMonitoring)) {return;}

     bool passedVertexCuts = true;
     if(UseVertexCuts_) {
       // vertex collection
       edm::Handle<VertexCollection> _primaryEventVertexCollection;
       iEvent.getByLabel(VertexSource_, _primaryEventVertexCollection);
       if(_primaryEventVertexCollection.isValid()) {
         const reco::Vertex& thePrimaryEventVertex = (*(_primaryEventVertexCollection)->begin());
         if( (!(thePrimaryEventVertex.isFake())) && (thePrimaryEventVertex.ndof() > 4) && (fabs(thePrimaryEventVertex.z()) < 20) && (fabs(thePrimaryEventVertex.position().rho()) <= 2) ) {
           passedVertexCuts = true;
         } else {
           passedVertexCuts = false;
         }
       } else {
         passedVertexCuts = false;
       }
     }

     // met collection
     edm::Handle<CaloMETCollection> metHandle;

     if (!iEvent.getByLabel(MetSource_, metHandle))
       {
	 if (debug_>0) edm::LogWarning("HcalMonitorTasks")<<" HcalDetDiagNoiseMonitor:  CaloMET collection with handle "<<MetSource_<<" not found!";
	 return;
       }

     const CaloMETCollection *metCol = metHandle.product();
     const CaloMET met = metCol->front();

     // Fill a histogram with the met for all events
     Met->Fill(met.pt());
     Mephi->Fill(met.phi());
     Mex->Fill(met.px());
     SumEt->Fill(met.sumEt());
     HaEtHB->Fill(met.hadEtInHB());
     HaEtHE->Fill(met.hadEtInHE());
     HaEtHF->Fill(met.hadEtInHF());
     EmEtHF->Fill(met.emEtInHF());
     if((passedPhysDeclared) && (passedVertexCuts)) {
       Met_PhysicsCategory->Fill(met.pt());
       Mephi_PhysicsCategory->Fill(met.phi());
       Mex_PhysicsCategory->Fill(met.px());
       SumEt_PhysicsCategory->Fill(met.sumEt());
       HaEtHB_PhysicsCategory->Fill(met.hadEtInHB());
       HaEtHE_PhysicsCategory->Fill(met.hadEtInHE());
       HaEtHF_PhysicsCategory->Fill(met.hadEtInHF());
       EmEtHF_PhysicsCategory->Fill(met.emEtInHF());
     }

     bool found = false;
     for(unsigned int i=0; i!=lumi.size(); ++i) { if(lumi.at(i) == iEvent.luminosityBlock()) {found = true; break;} }
     if(!found) {lumi.push_back(iEvent.luminosityBlock()); NLumiSections->Fill(0.5);}

     // jet collection
     edm::Handle<CaloJetCollection> calojetHandle;
     if (!iEvent.getByLabel(JetSource_, calojetHandle))
       {
        if (debug_>0) edm::LogWarning("HcalMonitorTasks")<<" HcalDetDiagNoiseMonitor:  CaloJet collection with handle "<<JetSource_<<" not found!";
         return;
       }
     // track collection
     edm::Handle<TrackCollection> trackHandle;
     if (!iEvent.getByLabel(TrackSource_, trackHandle))
       {
         if (debug_>0) edm::LogWarning("HcalMonitorTasks")<<" HcalDetDiagNoiseMonitor:  Track collection with handle "<<TrackSource_<<" not found!";
         return;
       }
     // HcalNoise RBX collection
     edm::Handle<HcalNoiseRBXCollection> rbxnoisehandle;
     if (!iEvent.getByLabel(rbxCollName_, rbxnoisehandle))
       {
         if (debug_>0) edm::LogWarning("HcalMonitorTasks")<<" HcalDetDiagNoiseMonitor:  HcalNoiseRBX collection with handle "<<rbxCollName_<<" not found!";
         return;
       }

     bool isAnomalous_BasedOnHCALFraction = false;
     bool isAnomalous_BasedOnCF = false;
     HcalNoisyJetContainer.clear();
     for(CaloJetCollection::const_iterator calojetIter = calojetHandle->begin();calojetIter != calojetHandle->end();++calojetIter) {
       if( (calojetIter->et() > JetMinEt_) && (fabs(calojetIter->eta()) < JetMaxEta_) ) {
         math::XYZTLorentzVector result (0,0,0,0);
         for(TrackCollection::const_iterator trackIter = trackHandle->begin(); trackIter != trackHandle->end(); ++trackIter) {
           double dR = deltaR2((*trackIter).eta(),(*trackIter).phi(),(*calojetIter).eta(),(*calojetIter).phi());
           if(sqrt(dR) <= ConstituentsToJetMatchingDeltaR_) {
             if( (fabs(trackIter->d0()) <= TrackMaxIp_) && (trackIter->pt() >= TrackMinThreshold_) ) {
               result += math::XYZTLorentzVector (trackIter->px(), trackIter->py(), trackIter->pz(), trackIter->p());
             }
           }
         }
         HCALFraction->Fill(calojetIter->energyFractionHadronic());
         chargeFraction->Fill(result.pt() / calojetIter->pt());
         HCALFractionVSchargeFraction->Fill(calojetIter->energyFractionHadronic(), result.pt() / calojetIter->pt());
         JetEt->Fill(calojetIter->et());
         JetEta->Fill(calojetIter->eta());
         JetPhi->Fill(calojetIter->phi());
         if((passedPhysDeclared) && (passedVertexCuts)) {
           HCALFraction_PhysicsCategory->Fill(calojetIter->energyFractionHadronic());
           chargeFraction_PhysicsCategory->Fill(result.pt() / calojetIter->pt());
           HCALFractionVSchargeFraction_PhysicsCategory->Fill(calojetIter->energyFractionHadronic(), result.pt() / calojetIter->pt());
           JetEt_PhysicsCategory->Fill(calojetIter->et());
           JetEta_PhysicsCategory->Fill(calojetIter->eta());
           JetPhi_PhysicsCategory->Fill(calojetIter->phi());
         }
         if((result.pt() / calojetIter->pt()) <= MinJetChargeFraction_) {isAnomalous_BasedOnCF = true;}
         if(calojetIter->energyFractionHadronic() >= MaxJetHadronicEnergyFraction_) {isAnomalous_BasedOnHCALFraction = true;}
         if( ((result.pt() / calojetIter->pt()) <= MinJetChargeFraction_) && (calojetIter->energyFractionHadronic() >= MaxJetHadronicEnergyFraction_) ) {
           JetEt_TaggedAnomalous->Fill(calojetIter->et());
           JetEta_TaggedAnomalous->Fill(calojetIter->eta());
           JetPhi_TaggedAnomalous->Fill(calojetIter->phi());
           if((passedPhysDeclared) && (passedVertexCuts)) {
             JetEt_TaggedAnomalous_PhysicsCategory->Fill(calojetIter->et());
             JetEta_TaggedAnomalous_PhysicsCategory->Fill(calojetIter->eta());
             JetPhi_TaggedAnomalous_PhysicsCategory->Fill(calojetIter->phi());
           }
           HcalNoisyJetContainer.push_back(*calojetIter);
         }
       }
     }

     // CaloTower collection
     edm::Handle<CaloTowerCollection> towerhandle;
     if (!iEvent.getByLabel(caloTowerCollName_, towerhandle))
       {
         if (debug_>0) edm::LogWarning("HcalMonitorTasks")<<" HcalDetDiagNoiseMonitor:  CaloTower collection with handle "<<caloTowerCollName_<<" not found!";
         return;
       }

     CaloTowerCollection::const_iterator ihighesttower;
     HcalNoiseRBXArray thearray;
     double HighestEnergyTower = 0;
     bool foundTowerMatch = false;
     for(std::vector<CaloJet>::iterator itjet = HcalNoisyJetContainer.begin(); itjet != HcalNoisyJetContainer.end(); ++itjet) {
       for(CaloTowerCollection::const_iterator itower = towerhandle->begin(); itower!=towerhandle->end(); ++itower) {
         double dR = deltaR2((*itower).eta(),(*itower).phi(),(*itjet).eta(),(*itjet).phi());
         if((sqrt(dR) <= ConstituentsToJetMatchingDeltaR_) && ((*itower).energy() > HighestEnergyTower)) {
           HighestEnergyTower = (*itower).energy();
           ihighesttower = itower;
           foundTowerMatch = true;
         }
       }
     }
     std::vector<std::vector<HcalNoiseHPD>::iterator> hpditervec;
     hpditervec.clear();
     std::vector<int> nid;
     nid.clear();
     std::vector<int> nidd;
     nidd.clear();
     if(foundTowerMatch) {
       const CaloTower& twr=(*ihighesttower);
       thearray.findHPD(twr, hpditervec);
       for(std::vector<std::vector<HcalNoiseHPD>::iterator>::iterator itofit=hpditervec.begin();itofit!=hpditervec.end(); ++itofit) {nid.push_back((*itofit)->idnumber());}
       if(nid.size() > 0) {
         double HighestEnergyMatch = 0;
         for(HcalNoiseRBXCollection::const_iterator rit=rbxnoisehandle->begin(); rit!=rbxnoisehandle->end(); ++rit) 
	   {
	     HcalNoiseRBX rbx = (*rit);
	     std::vector<HcalNoiseHPD> theHPDs = rbx.HPDs();
	     for(std::vector<HcalNoiseHPD>::const_iterator hit=theHPDs.begin(); hit!=theHPDs.end(); ++hit) 
	       //for(std::vector<HcalNoiseHPD>::const_iterator hit=rbx.HPDsBegin(); hit!=rbx.HPDsEnd(); ++hit) 
	       {
		 HcalNoiseHPD hpd=(*hit);
		 for(int iii=0; iii < (int)(nid.size()); iii++) 
		   {
		     if((nid.at(iii) == (int)(hpd.idnumber())) && (hpd.recHitEnergy(1.0) > HighestEnergyMatch))
		       {
			 HighestEnergyMatch = hpd.recHitEnergy(1.0);
			 nidd.clear();
			 nidd.push_back(hpd.idnumber());
		       }
		   }
	       }
	   }
       }
     }

     bool isHFAnomalous = false;
     for(CaloTowerCollection::const_iterator itower = towerhandle->begin(); itower!=towerhandle->end(); ++itower) {
       if( fabs((*itower).ieta()) > 29 ) {
         TVector3 * towerL = new TVector3;
         TVector3 * towerS = new TVector3;
         towerL->SetPtEtaPhi(itower->emEt() + 0.5 * itower->hadEt(), (*itower).eta(), (*itower).phi());
         towerS->SetPtEtaPhi(0.5 * itower->hadEt(), (*itower).eta(), (*itower).phi());
         //tower masked
         int isLongMasked=0;
         int isShortMasked=0;
         if( (*itower).ieta() == 37 && (*itower).iphi() == 67) {isLongMasked = 1;}
         if( (*itower).ieta() == 29 && (*itower).iphi() == 67) {isLongMasked = 1;}
         if( (*itower).ieta() == 35 && (*itower).iphi() == 67) {isLongMasked = 1;}
         if( (*itower).ieta() == 29 && (*itower).iphi() == 67) {isShortMasked = 1;}
         if( (*itower).ieta() == 30 && (*itower).iphi() == 67) {isShortMasked = 1;}
         if( (*itower).ieta() == 32 && (*itower).iphi() == 67) {isShortMasked = 1;}
         if( (*itower).ieta() == 36 && (*itower).iphi() == 67) {isShortMasked = 1;}
         if( (*itower).ieta() == 38 && (*itower).iphi() == 67) {isShortMasked = 1;}
         float towerPt = itower->emEt() + itower->hadEt();
         float towerEta = (*itower).eta();
         float towerPhi = (*itower).phi();
         float ET_cut_tcMET      = 5;
         float Rplus_cut_tcMET   = 0.99;
         float Rminus_cut_tcMET  = 0.8;
         Float_t ratio_tcMET     = -1.5;
         if( (itower->emEt() + itower->hadEt()) > ET_cut_tcMET && isShortMasked==0 && isLongMasked==0 ) {
           ratio_tcMET = (fabs(towerL->Mag()) - fabs(towerS->Mag())) / (fabs(towerL->Mag()) + fabs(towerS->Mag()));
           HFtowerRatio->Fill(ratio_tcMET);
           HFtowerPt->Fill(towerPt);
           HFtowerEta->Fill(towerEta);
           HFtowerPhi->Fill(towerPhi);
           if((passedPhysDeclared) && (passedVertexCuts)) {
             HFtowerRatio_PhysicsCategory->Fill(ratio_tcMET);
             HFtowerPt_PhysicsCategory->Fill(towerPt);
             HFtowerEta_PhysicsCategory->Fill(towerEta);
             HFtowerPhi_PhysicsCategory->Fill(towerPhi);
           }
           if( ratio_tcMET < -Rminus_cut_tcMET || ratio_tcMET > Rplus_cut_tcMET ) {
             isHFAnomalous = true;
             HFtowerPt_TaggedAnomalous->Fill(towerPt);
             HFtowerEta_TaggedAnomalous->Fill(towerEta);
             HFtowerPhi_TaggedAnomalous->Fill(towerPhi);
             if((passedPhysDeclared) && (passedVertexCuts)) {
               HFtowerPt_TaggedAnomalous_PhysicsCategory->Fill(towerPt);
               HFtowerEta_TaggedAnomalous_PhysicsCategory->Fill(towerEta);
               HFtowerPhi_TaggedAnomalous_PhysicsCategory->Fill(towerPhi);
             }
           }
         }
         delete towerL;
         delete towerS;
       }
     }

     bool isHbHeAnomalous = false;
     if((isAnomalous_BasedOnCF) && (isAnomalous_BasedOnHCALFraction)) {isHbHeAnomalous = true;}

     for(HcalNoiseRBXCollection::const_iterator rit=rbxnoisehandle->begin(); rit!=rbxnoisehandle->end(); ++rit) {
       HcalNoiseRBX rbx = (*rit);
       numRBXhits = rbx.numRecHits(1.0);
       rbxenergy = rbx.recHitEnergy(1.0);
       hpdEnergyHighest = 0.;
       nHitsHighest = 0.;
       totale2ts=rbx.allChargeHighest2TS();
       totale10ts=rbx.allChargeTotal();
       std::vector<HcalNoiseHPD> theHPDs = rbx.HPDs();
       for(std::vector<HcalNoiseHPD>::const_iterator hit=theHPDs.begin(); hit!=theHPDs.end(); ++hit) {
         HcalNoiseHPD hpd=(*hit);
         if ( hpd.recHitEnergy(1.0) > hpdEnergyHighest ) {
           hpdEnergyHighest = hpd.recHitEnergy(1.0);
           nHitsHighest     = hpd.numRecHits(1.0);
           e2ts=hpd.bigChargeHighest2TS();
           e10ts=hpd.bigChargeTotal();
         }
       }
       RBXMaxZeros->Fill(rbx.maxZeros());
       RBXHitsHighest->Fill(numRBXhits);
       RBXE2tsOverE10ts->Fill(totale10ts ? totale2ts/totale10ts : -999);
       HPDHitsHighest->Fill(nHitsHighest);
       HPDE2tsOverE10ts->Fill(e10ts ? e2ts/e10ts : -999);
       if((passedPhysDeclared) && (passedVertexCuts)) {
         RBXMaxZeros_PhysicsCategory->Fill(rbx.maxZeros());
         RBXHitsHighest_PhysicsCategory->Fill(numRBXhits);
         RBXE2tsOverE10ts_PhysicsCategory->Fill(totale10ts ? totale2ts/totale10ts : -999);
         HPDHitsHighest_PhysicsCategory->Fill(nHitsHighest);
         HPDE2tsOverE10ts_PhysicsCategory->Fill(e10ts ? e2ts/e10ts : -999);
       }
     }
     if( (isHbHeAnomalous) && (!(isHFAnomalous)) ) {
       Met_TaggedHBHEAnomalous->Fill(met.pt());
       Mephi_TaggedHBHEAnomalous->Fill(met.phi());
       Mex_TaggedHBHEAnomalous->Fill(met.px());
       SumEt_TaggedHBHEAnomalous->Fill(met.sumEt());
       HaEtHB_TaggedHBHEAnomalous->Fill(met.hadEtInHB());
       HaEtHE_TaggedHBHEAnomalous->Fill(met.hadEtInHE());
       HaEtHF_TaggedHBHEAnomalous->Fill(met.hadEtInHF());
       EmEtHF_TaggedHBHEAnomalous->Fill(met.emEtInHF());
       if(nidd.size() > 0) {
         for(HcalNoiseRBXCollection::const_iterator rit=rbxnoisehandle->begin(); rit!=rbxnoisehandle->end(); ++rit) {
           HcalNoiseRBX rbx = (*rit);
           numRBXhits = rbx.numRecHits(1.0);
           totale2ts=rbx.allChargeHighest2TS();
           totale10ts=rbx.allChargeTotal();
           bool isNoisyRBX = false;
           std::vector<HcalNoiseHPD> theHPDs = rbx.HPDs();
           for(std::vector<HcalNoiseHPD>::const_iterator hit=theHPDs.begin(); hit!=theHPDs.end(); ++hit) {
             HcalNoiseHPD hpd=(*hit);
             if((int)(hpd.idnumber()) == nidd.at(0)) {
               isNoisyRBX = true;
               nHitsHighest     = hpd.numRecHits(1.0);
               e2ts=hpd.bigChargeHighest2TS();
               e10ts=hpd.bigChargeTotal();
             }
           }
           if(isNoisyRBX) {
             RBXMaxZeros_TaggedHBHEAnomalous->Fill(rbx.maxZeros());
             RBXHitsHighest_TaggedHBHEAnomalous->Fill(numRBXhits);
             RBXE2tsOverE10ts_TaggedHBHEAnomalous->Fill(totale10ts ? totale2ts/totale10ts : -999);
             HPDHitsHighest_TaggedHBHEAnomalous->Fill(nHitsHighest);
             HPDE2tsOverE10ts_TaggedHBHEAnomalous->Fill(e10ts ? e2ts/e10ts : -999);
           }
         }
       }
       if((passedPhysDeclared) && (passedVertexCuts)) {
         Met_TaggedHBHEAnomalous_PhysicsCategory->Fill(met.pt());
         Mephi_TaggedHBHEAnomalous_PhysicsCategory->Fill(met.phi());
         Mex_TaggedHBHEAnomalous_PhysicsCategory->Fill(met.px());
         SumEt_TaggedHBHEAnomalous_PhysicsCategory->Fill(met.sumEt());
         HaEtHB_TaggedHBHEAnomalous_PhysicsCategory->Fill(met.hadEtInHB());
         HaEtHE_TaggedHBHEAnomalous_PhysicsCategory->Fill(met.hadEtInHE());
         HaEtHF_TaggedHBHEAnomalous_PhysicsCategory->Fill(met.hadEtInHF());
         EmEtHF_TaggedHBHEAnomalous_PhysicsCategory->Fill(met.emEtInHF());
         if(nidd.size() > 0) {
           for(HcalNoiseRBXCollection::const_iterator rit=rbxnoisehandle->begin(); rit!=rbxnoisehandle->end(); ++rit) {
             HcalNoiseRBX rbx = (*rit);
             numRBXhits = rbx.numRecHits(1.0);
             totale2ts=rbx.allChargeHighest2TS();
             totale10ts=rbx.allChargeTotal();
             bool isNoisyRBX = false;
             std::vector<HcalNoiseHPD> theHPDs = rbx.HPDs();
             for(std::vector<HcalNoiseHPD>::const_iterator hit=theHPDs.begin(); hit!=theHPDs.end(); ++hit) {
               HcalNoiseHPD hpd=(*hit);
               if((int)(hpd.idnumber()) == nidd.at(0)) {
                 isNoisyRBX = true;
                 nHitsHighest     = hpd.numRecHits(1.0);
                 e2ts=hpd.bigChargeHighest2TS();
                 e10ts=hpd.bigChargeTotal();
               }
             }
             if(isNoisyRBX) {
               RBXMaxZeros_TaggedHBHEAnomalous_PhysicsCategory->Fill(rbx.maxZeros());
               RBXHitsHighest_TaggedHBHEAnomalous_PhysicsCategory->Fill(numRBXhits);
               RBXE2tsOverE10ts_TaggedHBHEAnomalous_PhysicsCategory->Fill(totale10ts ? totale2ts/totale10ts : -999);
               HPDHitsHighest_TaggedHBHEAnomalous_PhysicsCategory->Fill(nHitsHighest);
               HPDE2tsOverE10ts_TaggedHBHEAnomalous_PhysicsCategory->Fill(e10ts ? e2ts/e10ts : -999);
             }
           }
         }
       }
     }
     if( (!(isHbHeAnomalous)) && ((isHFAnomalous)) ) {
       Met_TaggedHFAnomalous->Fill(met.pt());
       Mephi_TaggedHFAnomalous->Fill(met.phi());
       Mex_TaggedHFAnomalous->Fill(met.px());
       SumEt_TaggedHFAnomalous->Fill(met.sumEt());
       HaEtHB_TaggedHFAnomalous->Fill(met.hadEtInHB());
       HaEtHE_TaggedHFAnomalous->Fill(met.hadEtInHE());
       HaEtHF_TaggedHFAnomalous->Fill(met.hadEtInHF());
       EmEtHF_TaggedHFAnomalous->Fill(met.emEtInHF());
       if((passedPhysDeclared) && (passedVertexCuts)) {
         Met_TaggedHFAnomalous_PhysicsCategory->Fill(met.pt());
         Mephi_TaggedHFAnomalous_PhysicsCategory->Fill(met.phi());
         Mex_TaggedHFAnomalous_PhysicsCategory->Fill(met.px());
         SumEt_TaggedHFAnomalous_PhysicsCategory->Fill(met.sumEt());
         HaEtHB_TaggedHFAnomalous_PhysicsCategory->Fill(met.hadEtInHB());
         HaEtHE_TaggedHFAnomalous_PhysicsCategory->Fill(met.hadEtInHE());
         HaEtHF_TaggedHFAnomalous_PhysicsCategory->Fill(met.hadEtInHF());
         EmEtHF_TaggedHFAnomalous_PhysicsCategory->Fill(met.emEtInHF());
       }
     }
     if( ((isHbHeAnomalous)) && ((isHFAnomalous)) ) {
       Met_TaggedHBHEHFAnomalous->Fill(met.pt());
       Mephi_TaggedHBHEHFAnomalous->Fill(met.phi());
       Mex_TaggedHBHEHFAnomalous->Fill(met.px());
       SumEt_TaggedHBHEHFAnomalous->Fill(met.sumEt());
       HaEtHB_TaggedHBHEHFAnomalous->Fill(met.hadEtInHB());
       HaEtHE_TaggedHBHEHFAnomalous->Fill(met.hadEtInHE());
       HaEtHF_TaggedHBHEHFAnomalous->Fill(met.hadEtInHF());
       EmEtHF_TaggedHBHEHFAnomalous->Fill(met.emEtInHF());
       if(nidd.size() > 0) {
         for(HcalNoiseRBXCollection::const_iterator rit=rbxnoisehandle->begin(); rit!=rbxnoisehandle->end(); ++rit) {
           HcalNoiseRBX rbx = (*rit);
           numRBXhits = rbx.numRecHits(1.0);
           totale2ts=rbx.allChargeHighest2TS();
           totale10ts=rbx.allChargeTotal();
           bool isNoisyRBX = false;
           std::vector<HcalNoiseHPD> theHPDs = rbx.HPDs();
           for(std::vector<HcalNoiseHPD>::const_iterator hit=theHPDs.begin(); hit!=theHPDs.end(); ++hit) {
             HcalNoiseHPD hpd=(*hit);
             if((int)(hpd.idnumber()) == nidd.at(0)) {
               isNoisyRBX = true;
               nHitsHighest     = hpd.numRecHits(1.0);
               e2ts=hpd.bigChargeHighest2TS();
               e10ts=hpd.bigChargeTotal();
             }
           }
           if(isNoisyRBX) {
             RBXMaxZeros_TaggedHBHEAnomalous->Fill(rbx.maxZeros());
             RBXHitsHighest_TaggedHBHEAnomalous->Fill(numRBXhits);
             RBXE2tsOverE10ts_TaggedHBHEAnomalous->Fill(totale10ts ? totale2ts/totale10ts : -999);
             HPDHitsHighest_TaggedHBHEAnomalous->Fill(nHitsHighest);
             HPDE2tsOverE10ts_TaggedHBHEAnomalous->Fill(e10ts ? e2ts/e10ts : -999);
           }
         }
       }
       if((passedPhysDeclared) && (passedVertexCuts)) {
         Met_TaggedHBHEHFAnomalous_PhysicsCategory->Fill(met.pt());
         Mephi_TaggedHBHEHFAnomalous_PhysicsCategory->Fill(met.phi());
         Mex_TaggedHBHEHFAnomalous_PhysicsCategory->Fill(met.px());
         SumEt_TaggedHBHEHFAnomalous_PhysicsCategory->Fill(met.sumEt());
         HaEtHB_TaggedHBHEHFAnomalous_PhysicsCategory->Fill(met.hadEtInHB());
         HaEtHE_TaggedHBHEHFAnomalous_PhysicsCategory->Fill(met.hadEtInHE());
         HaEtHF_TaggedHBHEHFAnomalous_PhysicsCategory->Fill(met.hadEtInHF());
         EmEtHF_TaggedHBHEHFAnomalous_PhysicsCategory->Fill(met.emEtInHF());
         if(nidd.size() > 0) {
           for(HcalNoiseRBXCollection::const_iterator rit=rbxnoisehandle->begin(); rit!=rbxnoisehandle->end(); ++rit) {
             HcalNoiseRBX rbx = (*rit);
             numRBXhits = rbx.numRecHits(1.0);
             totale2ts=rbx.allChargeHighest2TS();
             totale10ts=rbx.allChargeTotal();
             bool isNoisyRBX = false;
             std::vector<HcalNoiseHPD> theHPDs = rbx.HPDs();
             for(std::vector<HcalNoiseHPD>::const_iterator hit=theHPDs.begin(); hit!=theHPDs.end(); ++hit) {
               HcalNoiseHPD hpd=(*hit);
               if((int)(hpd.idnumber()) == nidd.at(0)) {
                 isNoisyRBX = true;
                 nHitsHighest     = hpd.numRecHits(1.0);
                 e2ts=hpd.bigChargeHighest2TS();
                 e10ts=hpd.bigChargeTotal();
               }
             }
             if(isNoisyRBX) {
               RBXMaxZeros_TaggedHBHEAnomalous_PhysicsCategory->Fill(rbx.maxZeros());
               RBXHitsHighest_TaggedHBHEAnomalous_PhysicsCategory->Fill(numRBXhits);
               RBXE2tsOverE10ts_TaggedHBHEAnomalous_PhysicsCategory->Fill(totale10ts ? totale2ts/totale10ts : -999);
               HPDHitsHighest_TaggedHBHEAnomalous_PhysicsCategory->Fill(nHitsHighest);
               HPDE2tsOverE10ts_TaggedHBHEAnomalous_PhysicsCategory->Fill(e10ts ? e2ts/e10ts : -999);
             }
           }
         }
       }
     }

   } //if (!Online_)

// ###################################################################################################################
       
   if((ievt_%100)==0 && debug_>0)
     std::cout <<ievt_<<"\t"<<NoisyEvents<<std::endl;

   return;
}

void HcalDetDiagNoiseMonitor::UpdateHistos()
{
  int first_rbx=0,last_rbx=0;  
  for(int sd=0;sd<9;sd++)
    {
      if(RBXCurrentSummary->GetStat(sd)>=UpdateEvents)
	{
	  if(sd==0){ first_rbx=0;  last_rbx=18;} //HBM
	  if(sd==1){ first_rbx=18; last_rbx=36;} //HBP
	  if(sd==0 || sd==1){  // update HB plots
	    for(int rbx=first_rbx;rbx<last_rbx;rbx++)for(int rm=1;rm<=4;rm++){
	      double val1=0,val2=0;
	      if(RBXSummary->GetRMStatusValue(HB_RBX[rbx],rm,&val1)){
		HB_RBXmapRatio->setBinContent(rm,rbx+1,val1);
		if(RBXCurrentSummary->GetRMStatusValue(HB_RBX[rbx],rm,&val2)){
		  HB_RBXmapRatioCur->setBinContent(rm,rbx+1,val2);
		  if((val2-val1)>SpikeThreshold){
		    double n=HB_RBXmapSpikeCnt->getBinContent(rm,rbx+1);
		    double a=HB_RBXmapSpikeAmp->getBinContent(rm,rbx+1);
		    HB_RBXmapSpikeCnt->Fill(rm,rbx+1,1);
		    HB_RBXmapSpikeAmp->setBinContent(rm,rbx+1,((val2-val1)+a*n)/(n+1));
		  }
		}
	      }
	    }	
	  }
	  if(sd==2){ first_rbx=0;  last_rbx=18;} //HEM
	  if(sd==3){ first_rbx=18; last_rbx=36;} //HEP
	  if(sd==2 || sd==3){  // update HB plots
	    for(int rbx=first_rbx;rbx<last_rbx;rbx++)for(int rm=1;rm<=4;rm++){
              double val1=0,val2=0;
              if(RBXSummary->GetRMStatusValue(HE_RBX[rbx],rm,&val1)){
	        HE_RBXmapRatio->setBinContent(rm,rbx+1,val1);
                if(RBXCurrentSummary->GetRMStatusValue(HE_RBX[rbx],rm,&val2)){
		  HE_RBXmapRatioCur->setBinContent(rm,rbx+1,val2);
		  if((val2-val1)>SpikeThreshold){
		    double n=HE_RBXmapSpikeCnt->getBinContent(rm,rbx+1);
		    double a=HE_RBXmapSpikeAmp->getBinContent(rm,rbx+1);
		    HE_RBXmapSpikeCnt->Fill(rm,rbx+1,1);
		    HE_RBXmapSpikeAmp->setBinContent(rm,rbx+1,((val2-val1)+a*n)/(n+1));
		  }
	        }
	      }
	    }	
	  }
	  if(sd==4){ first_rbx=0;  last_rbx=6;}   //HO2M
	  if(sd==5){ first_rbx=6;  last_rbx=12;}  //HO1M
	  if(sd==6){ first_rbx=12;  last_rbx=24;} //HO0
	  if(sd==7){ first_rbx=24;  last_rbx=30;} //HO1P
	  if(sd==8){ first_rbx=30;  last_rbx=36;} //HO2P
	  if(sd>3){ // update HO plots
	    for(int rbx=first_rbx;rbx<last_rbx;rbx++)for(int rm=1;rm<=4;rm++){
              double val1=0,val2=0;
              if(RBXSummary->GetRMStatusValue(HO_RBX[rbx],rm,&val1)){
	        HO_RBXmapRatio->setBinContent(rm,rbx+1,val1);
                if(RBXCurrentSummary->GetRMStatusValue(HO_RBX[rbx],rm,&val2)){
		  HO_RBXmapRatioCur->setBinContent(rm,rbx+1,val2);
		  if((val2-val1)>SpikeThreshold){
		    double n=HO_RBXmapSpikeCnt->getBinContent(rm,rbx+1);
		    double a=HO_RBXmapSpikeAmp->getBinContent(rm,rbx+1);
		    HO_RBXmapSpikeCnt->Fill(rm,rbx+1,1);
		    HO_RBXmapSpikeAmp->setBinContent(rm,rbx+1,((val2-val1)+a*n)/(n+1));
		  }
	        }
	      }
	    }		
	  }
	  
	  RBXCurrentSummary->reset(sd); 
	}  //if(RBXCurrentSummary->GetStat(sd)>=UpdateEvents)
    } //sd=0;sd<9
} // UpdateHistos

void HcalDetDiagNoiseMonitor::SaveReference(){
char   RBX[20];
int    RM_INDEX,RM;
double VAL;
    if(UseDB==false){
       char str[100]; 
       sprintf(str,"%sHcalDetDiagNoiseData_run%06i.root",OutputFilePath.c_str(),run_number);
       TFile *theFile = new TFile(str, "RECREATE");
       if(!theFile->IsOpen()) return;
       theFile->cd();
       sprintf(str,"%d",run_number); TObjString run(str);    run.Write("run number");
       sprintf(str,"%d",ievt_);      TObjString events(str); events.Write("Total events processed");
       
       TTree *tree   =new TTree("HCAL Noise data","HCAL Noise data");
       if(tree==0)   return;
       tree->Branch("RBX",            &RBX,      "RBX/C");
       tree->Branch("rm",             &RM,       "rm/I");
       tree->Branch("rm_index",       &RM_INDEX, "rm_index/I");
       tree->Branch("relative_noise", &VAL,      "relative_noise/D");
       for(int sd=0;sd<9;sd++) for(int sect=1;sect<=18;sect++) for(int rm=1;rm<=4;rm++){
           std::stringstream tempss;
           tempss << std::setw(2) << std::setfill('0') << sect;
           std::string rbx= subdets[sd]+tempss.str();
           double val;
           if(RBXCurrentSummary->GetRMStatusValue(rbx,rm,&val)){
	       sprintf(RBX,"%s",(char *)rbx.c_str());
	       RM=rm;
	       RM_INDEX=RBXCurrentSummary->GetRMindex(rbx,rm);
	       val=VAL;
               tree->Fill();
           }
       }     
       theFile->Write();
       theFile->Close();
   }
}

void HcalDetDiagNoiseMonitor::LoadReference(){
TFile *f;
int    RM_INDEX;
double VAL;
   if(UseDB==false){
     f = new TFile(ReferenceData.c_str(),"READ");
      if(!f->IsOpen()){ return ;}
      TObjString *STR=(TObjString *)f->Get("run number");
      
      if(STR){ std::string Ref(STR->String()); ReferenceRun=Ref;}
      
      TTree*  t=(TTree*)f->Get("HCAL Noise data");
      if(!t) return;
      t->SetBranchAddress("rm_index",       &RM_INDEX);
      t->SetBranchAddress("relative_noise", &VAL);
      for(int ievt=0;ievt<t->GetEntries();ievt++){
         t->GetEntry(ievt);
	 RBXCurrentSummary->SetReference(RM_INDEX,VAL);
	 RBXSummary->SetReference(RM_INDEX,VAL);
      }
      f->Close();
      IsReference=true;
   }
} 

void HcalDetDiagNoiseMonitor::done(){   /*SaveReference();*/ } 

DEFINE_FWK_MODULE (HcalDetDiagNoiseMonitor);
