#include "DQM/HcalMonitorTasks/interface/HcalDetDiagNoiseMonitor.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
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
#include "HLTrigger/HLTanalyzers/interface/JetUtil.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
// #include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
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
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"


#include <math.h>

using namespace reco;

////////////////////////////////////////////////////////////////////////////////////////////
constexpr float adc2fC[128]={-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5, 10.5,11.5,12.5,
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
constexpr char const *subdets[11]={"HBM","HBP","HEM","HEP","HO1M","HO0","HO1P","HO2M","HO2P","HFM","HFP"};
constexpr char const *HB_RBX[36]={
"HBM01","HBM02","HBM03","HBM04","HBM05","HBM06","HBM07","HBM08","HBM09","HBM10","HBM11","HBM12","HBM13","HBM14","HBM15","HBM16","HBM17","HBM18",
"HBP01","HBP02","HBP03","HBP04","HBP05","HBP06","HBP07","HBP08","HBP09","HBP10","HBP11","HBP12","HBP13","HBP14","HBP15","HBP16","HBP17","HBP18"};
constexpr char const *HE_RBX[36]={
"HEM01","HEM02","HEM03","HEM04","HEM05","HEM06","HEM07","HEM08","HEM09","HEM10","HEM11","HEM12","HEM13","HEM14","HEM15","HEM16","HEM17","HEM18",
"HEP01","HEP02","HEP03","HEP04","HEP05","HEP06","HEP07","HEP08","HEP09","HEP10","HEP11","HEP12","HEP13","HEP14","HEP15","HEP16","HEP17","HEP18"};
constexpr char const *HO_RBX[36]={
"HO2M02","HO2M04","HO2M06","HO2M08","HO2M10","HO2M12","HO1M02","HO1M04","HO1M06","HO1M08","HO1M10","HO1M12",
"HO001","HO002","HO003","HO004","HO005","HO006","HO007","HO008","HO009","HO010","HO011","HO012",
"HO1P02","HO1P04","HO1P06","HO1P08","HO1P10","HO1P12","HO2P02","HO2P04","HO2P06","HO2P08","HO2P10","HO2P12",
};


class HcalDetDiagNoiseRMData{
public:
  HcalDetDiagNoiseRMData(){
     reset();
     reset_LS();
  };
  void reset(){
    n_th_hi=n_th_300=n_pix_1=n_pix_8=pix=n_pix=0;
  }  
  void reset_LS(){
    n_th_hi_LS=n_th_300_LS=0;
  }
  int    n_th_hi;
  int    n_th_300;
  int    n_pix_1;
  int    n_pix_8;
  int    pix;
  int    n_pix; 
  int    n_th_hi_LS;
  int    n_th_300_LS;
};
class HcalDetDiagNoiseRMEvent{
public:
  HcalDetDiagNoiseRMEvent(){
    reset();
  }
  void reset(){
   n_pix_hi=n_pix_lo=energy=n_zero=0;
  }
  int n_pix_hi;
  int n_pix_lo;
  int n_zero;
  float energy;
};

class HcalDetDiagNoiseRMSummary{
public:
  HcalDetDiagNoiseRMSummary(){ 
     reset();
  }
  void reset(){
     for(int i=0;i<HcalFrontEndId::maxRmIndex;i++) rm[i].reset(); 
  }
  void reset_LS(){
     for(int i=0;i<HcalFrontEndId::maxRmIndex;i++) rm[i].reset_LS(); 
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
  HcalDetDiagNoiseRMData rm[HcalFrontEndId::maxRmIndex];
};

HcalDetDiagNoiseMonitor::HcalDetDiagNoiseMonitor(const edm::ParameterSet& ps):
  HcalBaseDQMonitor(ps)
 {

  tok_tb_ = consumes<HcalTBTriggerData>(ps.getParameter<edm::InputTag>("hcalTBTriggerDataTag"));

  ievt_=0;
  run_number=-1;
  NoisyEvents=0;
  LocalRun=false; 
  dataset_seq_number=1;
  FirstOrbit=FirstOrbitLS=0xFFFFFFFF;
  LastOrbit=LastOrbitLS=0;

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
  skipOutOfOrderLS_      = ps.getUntrackedParameter<bool>("skipOutOfOrderLS",false);
  NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",4000);
  makeDiagnostics_       = ps.getUntrackedParameter<bool>("makeDiagnostics",false);

  UseDB            = ps.getUntrackedParameter<bool>  ("UseDB"  , false);
  OutputFilePath   = ps.getUntrackedParameter<std::string>("OutputFilePath", "");
  HPDthresholdHi   = ps.getUntrackedParameter<double>("NoiseThresholdHPDhi",49.0);
  HPDthresholdLo   = ps.getUntrackedParameter<double>("NoiseThresholdHPDlo",10.0);
  SpikeThreshold   = ps.getUntrackedParameter<double>("NoiseSpikeThreshold",0.5);
  Overwrite        = ps.getUntrackedParameter<bool>  ("Overwrite",true);

  tok_raw_ =  consumes<FEDRawDataCollection>(ps.getUntrackedParameter<edm::InputTag>("RawDataLabel",edm::InputTag("source","")));
  digiLabel_     = ps.getUntrackedParameter<edm::InputTag>("digiLabel",edm::InputTag("hcalDigis"));
  tok_l1_  = consumes<L1GlobalTriggerReadoutRecord>(ps.getUntrackedParameter<edm::InputTag>("gtLabel"));

  
  tok_hbhe_ = consumes<HBHEDigiCollection>(digiLabel_);
  tok_ho_ = consumes<HODigiCollection>(digiLabel_);
  
  RMSummary = 0;
  needLogicalMap_=true;
  setupDone_ = false;
}

void HcalDetDiagNoiseMonitor::reset(){}


void HcalDetDiagNoiseMonitor::bookHistograms(DQMStore::IBooker &ib, const edm::Run& run, const edm::EventSetup& c)
{
  if (debug_>1) std::cout <<"HcalDetDiagNoiseMonitor::bookHistograms"<<std::endl;
  HcalBaseDQMonitor::bookHistograms(ib,run,c);

  if (tevt_==0) this->setup(ib); // set up histograms if they have not been created before
  if (mergeRuns_==false)
    this->reset();

  return;

} 

void HcalDetDiagNoiseMonitor::setup(DQMStore::IBooker &ib){
  if (setupDone_)
    return;
  setupDone_ = true;
  // Call base class setup
  HcalBaseDQMonitor::setup(ib);
  RMSummary = new HcalDetDiagNoiseRMSummary();

  std::string name;
     ib.setCurrentFolder(subdir_);   
     meEVT_ = ib.bookInt("HcalNoiseMonitor Event Number");
     ib.setCurrentFolder(subdir_+"Common Plots");
     
     name="RBX Pixel multiplicity";     PixelMult        = ib.book1D(name,name,73,0,73);
     name="HPD energy";                 HPDEnergy        = ib.book1D(name,name,200,0,2500);
     name="RBX energy";                 RBXEnergy        = ib.book1D(name,name,200,0,3500);
     name="Number of zero TS per RBX";  NZeroes          = ib.book1D(name,name,100,0,100);
     name="Trigger BX Tbit11";          TriggerBx11      = ib.book1D(name,name,4000,0,4000);
     name="Trigger BX Tbit12";          TriggerBx12      = ib.book1D(name,name,4000,0,4000);

     ib.setCurrentFolder(subdir_+"HBHE Plots");
     name="HBP HPD Noise Rate Pixel above 50fC"; HBP_Rate50    = ib.book1D(name,name,73,0,73);
     name="HBM HPD Noise Rate Pixel above 50fC"; HBM_Rate50    = ib.book1D(name,name,73,0,73);
     name="HEP HPD Noise Rate Pixel above 50fC"; HEP_Rate50    = ib.book1D(name,name,73,0,73);
     name="HEM HPD Noise Rate Pixel above 50fC"; HEM_Rate50    = ib.book1D(name,name,73,0,73);
     name="HBP HPD Noise Rate HPD above 300fC";  HBP_Rate300   = ib.book1D(name,name,73,0,73);
     name="HBM HPD Noise Rate HPD above 300fC";  HBM_Rate300   = ib.book1D(name,name,73,0,73);
     name="HEP HPD Noise Rate HPD above 300fC";  HEP_Rate300   = ib.book1D(name,name,73,0,73);
     name="HEM HPD Noise Rate HPD above 300fC";  HEM_Rate300   = ib.book1D(name,name,73,0,73);

     ib.setCurrentFolder(subdir_+"HO Plots");
     name="HO0  HPD Noise Rate Pixel above 50fC"; HO0_Rate50   = ib.book1D(name,name,49,0,49);
     name="HO1P HPD Noise Rate Pixel above 50fC"; HO1P_Rate50   = ib.book1D(name,name,48,0,48);
     name="HO1M HPD Noise Rate Pixel above 50fC"; HO1M_Rate50   = ib.book1D(name,name,48,0,48);
     name="HO0 HPD Noise Rate HPD above 300fC";   HO0_Rate300  = ib.book1D(name,name,48,0,48);
     name="HO1P HPD Noise Rate HPD abGetRMindexove 300fC";  HO1P_Rate300 = ib.book1D(name,name,48,0,48);
     name="HO1M HPD Noise Rate HPD above 300fC";  HO1M_Rate300 = ib.book1D(name,name,48,0,48);
      

     ib.setCurrentFolder(subdir_+"Noise Spike Plots");

     name="HB RM Spike Map";          HB_RBXmapSpikeCnt= ib.book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HE RM Spike Map";          HE_RBXmapSpikeCnt= ib.book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HO RM Spike Map";          HO_RBXmapSpikeCnt= ib.book2D(name,name,4,0.5,4.5,36,0.5,36.5);

     std::string title="RM";
     HB_RBXmapSpikeCnt->setAxisTitle(title);
     HE_RBXmapSpikeCnt->setAxisTitle(title);
     HO_RBXmapSpikeCnt->setAxisTitle(title);
 
     for(int i=0;i<36;i++){
        HB_RBXmapSpikeCnt->setBinLabel(i+1,HB_RBX[i],2);
        HE_RBXmapSpikeCnt->setBinLabel(i+1,HE_RBX[i],2);
        HO_RBXmapSpikeCnt->setBinLabel(i+1,HO_RBX[i],2);
     }


  return;
} 

void HcalDetDiagNoiseMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  getLogicalMap(iSetup);
  HcalBaseDQMonitor::analyze(iEvent, iSetup);
  if (!IsAllowedCalibType()) return;
  if (LumiInOrder(iEvent.luminosityBlock())==false) return;
//  HcalBaseDQMonitor::analyze(iEvent, iSetup);
  bool isNoiseEvent=false;  
  int orbit=-1111;
  int bx=-1111;

  // for local runs 
  edm::Handle<HcalTBTriggerData> trigger_data;
  iEvent.getByToken(tok_tb_, trigger_data);
  if(trigger_data.isValid()){
      if(trigger_data->triggerWord()>1000) isNoiseEvent=true;
      LocalRun=true;
  }

  // We do not want to look at Abort Gap events
  edm::Handle<FEDRawDataCollection> rawdata;
  iEvent.getByToken(tok_raw_,rawdata);
  //checking FEDs for calibration information
  for(int i=FEDNumbering::MINHCALFEDID;
		  i<=FEDNumbering::MAXHCALuTCAFEDID; i++) 
  {
	  if (i>FEDNumbering::MAXHCALFEDID && i<FEDNumbering::MINHCALuTCAFEDID)
		continue;

      const FEDRawData& fedData = rawdata->FEDData(i) ;
      if ( fedData.size() < 24 ) continue ;
      orbit= ((const HcalDCCHeader*)(fedData.data()))->getOrbitNumber();
      bx=((const HcalDCCHeader*)(fedData.data()))->getBunchId();
      if(((const HcalDCCHeader*)(fedData.data()))->getCalibType()!=hc_Null) return;
  }

  // Check GCT trigger bits
  edm::Handle< L1GlobalTriggerReadoutRecord > gtRecord;
  iEvent.getByToken(tok_l1_, gtRecord);
  if(gtRecord.isValid()){
    const TechnicalTriggerWord tWord = gtRecord->technicalTriggerWord();
    if(tWord.at(11) || tWord.at(12)) isNoiseEvent=true;
    if(tWord.at(11)){ TriggerBx11->Fill(bx);}
    if(tWord.at(12)){ TriggerBx12->Fill(bx);}
  }
 
  if(!isNoiseEvent) return;
  if(ievt_==0){ FirstOrbit=orbit; FirstOrbitLS=orbit; newLS=true;}
  if(LastOrbit <orbit) LastOrbit=orbit; 
  if(FirstOrbit>orbit) FirstOrbit=orbit;
  if(LastOrbitLS <orbit) LastOrbitLS=orbit; 
  if(FirstOrbitLS>orbit) FirstOrbitLS=orbit;
  if(newLS){ 
     FirstOrbitLS=orbit; 
     newLS=false;
  }

  if(!LocalRun){
     double TIME=(double)(LastOrbit-FirstOrbit)/11223.0;
     if(TIME>1800.0){
        UpdateHistos();
        SaveRates();
        RMSummary->reset();
        FirstOrbit=orbit; 
     }
  }

  meEVT_->Fill(++ievt_);

  run_number=iEvent.id().run();

  HcalDetDiagNoiseRMEvent RMs[HcalFrontEndId::maxRmIndex];
   
   edm::Handle<HBHEDigiCollection> hbhe; 
   iEvent.getByToken(tok_hbhe_,hbhe);
   for(HBHEDigiCollection::const_iterator digi=hbhe->begin();digi!=hbhe->end();digi++){
     double max=-100/*,sum*/,energy=0; int n_zero=0;
     for(int i=0;i<digi->size();i++){
       //       sum=adc2fC[digi->sample(i).adc()&0xff]; 
       if(max<adc2fC[digi->sample(i).adc()&0xff]) max=adc2fC[digi->sample(i).adc()&0xff];
       if(adc2fC[digi->sample(i).adc()&0xff]==0) n_zero++;
     }
     HcalFrontEndId lmap_entry=logicalMap_->getHcalFrontEndId(digi->id());
     int index=lmap_entry.rmIndex(); if(index>=HcalFrontEndId::maxRmIndex) continue;
     RMs[index].n_zero++;
     if(max>HPDthresholdLo){
       for(int i=0;i<digi->size();i++) energy+=adc2fC[digi->sample(i).adc()&0xff]-2.5;
       RMs[index].n_pix_lo++;
       if(max>HPDthresholdHi){ RMs[index].n_pix_hi++; isNoiseEvent=true;}
       RMs[index].energy+=energy;
     }
   }

   edm::Handle<HODigiCollection> ho; 
   iEvent.getByToken(tok_ho_,ho);
   for(HODigiCollection::const_iterator digi=ho->begin();digi!=ho->end();digi++){
     double max=-100,energy=0; int Eta=digi->id().ieta(); int Phi=digi->id().iphi(); int n_zero=0;
     for(int i=0;i<digi->size()-1;i++){
       if(max<adc2fC[digi->sample(i).adc()&0xff]) max=adc2fC[digi->sample(i).adc()&0xff];
       if(adc2fC[digi->sample(i).adc()&0xff]==0) n_zero++;
     }
     if((Eta>=11 && Eta<=15 && Phi>=59 && Phi<=70) || (Eta>=5 && Eta<=10 && Phi>=47 && Phi<=58)){
       continue; // ignory SiPMs
     }else{
       HcalFrontEndId lmap_entry=logicalMap_->getHcalFrontEndId(digi->id());
       int index=lmap_entry.rmIndex(); if(index>=HcalFrontEndId::maxRmIndex) continue;
       RMs[index].n_zero++;
       if(max>HPDthresholdLo){
	 for(int i=0;i<digi->size();i++) energy+=adc2fC[digi->sample(i).adc()&0xff]-2.5;
	 RMs[index].n_pix_lo++;
	 if(max>HPDthresholdHi){ RMs[index].n_pix_hi++; isNoiseEvent=true;}
	 RMs[index].energy+=energy;
       }
     }		          
   }   

   NoisyEvents++;
      
   // RMs loop
   for(int i=0;i<HcalFrontEndId::maxRmIndex;i++){
      if(RMs[i].n_pix_hi>0){
 	 HPDEnergy->Fill(RMs[i].energy);
 	 RMSummary->rm[i].n_th_hi++;
         RMSummary->rm[i].n_th_hi_LS++;
         if(RMs[i].energy>300) RMSummary->rm[i].n_th_300++;
         if(RMs[i].energy>300) RMSummary->rm[i].n_th_300_LS++;
         if(RMs[i].n_pix_lo>1) RMSummary->rm[i].n_pix_1++;
         if(RMs[i].n_pix_lo>8) RMSummary->rm[i].n_pix_8++;
         RMSummary->rm[i].pix+=RMs[i].n_pix_lo;
         RMSummary->rm[i].n_pix++;
      }
   }

   // RBX loop
   for(int sd=0;sd<7;sd++) for(int sect=1;sect<=18;sect++){
     std::stringstream tempss;
     tempss << std::setw(2) << std::setfill('0') << sect;
     std::string rbx= subdets[sd]+tempss.str();
     
     double rbx_energy=0;int pix_mult=0; int n_zero=0; bool isValidRBX=false;
     for(int rm=1;rm<=4;rm++){
       int index=RMSummary->GetRMindex(rbx,rm);
       if(index>0 && index<HcalFrontEndId::maxRmIndex){
	 rbx_energy+=RMs[index].energy;
	 pix_mult+=RMs[index].n_pix_lo; 
         n_zero+=RMs[index].n_zero; 
	 isValidRBX=true;
       }
     }
     if(isValidRBX){
       PixelMult->Fill(pix_mult);
       RBXEnergy->Fill(rbx_energy);
       NZeroes->Fill(n_zero);
     }
   }

   if((ievt_%100)==0 && debug_>0)
     std::cout <<ievt_<<"\t"<<NoisyEvents<<std::endl;
   return;
}

void HcalDetDiagNoiseMonitor::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c){
int first_rbx=0,last_rbx=0;  
  //double TIME=(double)(LastOrbitLS-FirstOrbitLS)/11223.0;
  double TIME=23.0;
  newLS=true;
  if(TIME==0) return;
 
  for(int sd=0;sd<9;sd++){
	  if(sd==0){ first_rbx=0;  last_rbx=18;} //HBM
	  if(sd==1){ first_rbx=18; last_rbx=36;} //HBP
	  if(sd==0 || sd==1){  // update HB plots
	    for(int rbx=first_rbx;rbx<last_rbx;rbx++)for(int rm=1;rm<=4;rm++){
               int index=RMSummary->GetRMindex(HB_RBX[rbx],rm);
               if(index<0 || index>=HcalFrontEndId::maxRmIndex) continue;
               double val=RMSummary->rm[index].n_th_hi_LS/TIME;
               if(val>SpikeThreshold){
                  HB_RBXmapSpikeCnt->Fill(rm,rbx+1,1);
                  //printf("%s %i %f (%f)\n",HO_RBX[rbx].c_str(),rm,RMSummary->rm[index].n_th_hi_LS/TIME,TIME);
               }
               RMSummary->rm[index].reset_LS();
	    }
          }
	  if(sd==2){ first_rbx=0;  last_rbx=18;} //HEM
	  if(sd==3){ first_rbx=18; last_rbx=36;} //HEP
	  if(sd==2 || sd==3){  // update HB plots
	    for(int rbx=first_rbx;rbx<last_rbx;rbx++)for(int rm=1;rm<=4;rm++){
              int index=RMSummary->GetRMindex(HE_RBX[rbx],rm);
              if(index<0 || index>=HcalFrontEndId::maxRmIndex) continue;
              double val=RMSummary->rm[index].n_th_hi_LS/TIME;
              if(val>SpikeThreshold){
                  HE_RBXmapSpikeCnt->Fill(rm,rbx+1,1);
                  //printf("%s %i %f (%f)\n",HO_RBX[rbx].c_str(),rm,RMSummary->rm[index].n_th_hi_LS/TIME,TIME);
              }
              RMSummary->rm[index].reset_LS();
	    }
          }
	  if(sd==4){ first_rbx=6;  last_rbx=12;}  //HO1M
	  if(sd==5){ first_rbx=12;  last_rbx=24;} //HO0
	  if(sd==6){ first_rbx=24;  last_rbx=30;} //HO1P
	  if(sd>3){
            for(int rbx=first_rbx;rbx<last_rbx;rbx++)for(int rm=1;rm<=4;rm++){
              int index=RMSummary->GetRMindex(HO_RBX[rbx],rm);
              if(index<0 || index>=HcalFrontEndId::maxRmIndex) continue;
              double val=RMSummary->rm[index].n_th_hi_LS/TIME;
              if(val>SpikeThreshold){
                  HO_RBXmapSpikeCnt->Fill(rm,rbx+1,1);
                  //printf("%s %i %f (%f)\n",HO_RBX[rbx].c_str(),rm,RMSummary->rm[index].n_th_hi_LS/TIME,TIME);
              }
              RMSummary->rm[index].reset_LS();
	   }
         }
    } //sd=0;sd<9
}

void HcalDetDiagNoiseMonitor::UpdateHistos(){ 
   int first_rbx=0; 
   double TIME=(double)(LastOrbit-FirstOrbit)/11223.0;
   if(TIME==0) return;
   for(int sd=0;sd<9;sd++){
      if(sd==0){ first_rbx=0; } //HBM
      if(sd==1){ first_rbx=18;} //HBP
      if(sd==0 || sd==1){  // update HB plots
	  for(int rbx=0;rbx<18;rbx++)for(int rm=1;rm<=4;rm++){
             int index=RMSummary->GetRMindex(HB_RBX[rbx+first_rbx],rm);
             if(index<0 || index>=HcalFrontEndId::maxRmIndex) continue;
             if(sd==0){
                 HBM_Rate50->setBinContent(rbx*4+rm,RMSummary->rm[index].n_th_hi/TIME);
                 HBM_Rate300->setBinContent(rbx*4+rm,RMSummary->rm[index].n_th_300/TIME);
             }
             if(sd==1){
                 HBP_Rate50->setBinContent(rbx*4+rm,RMSummary->rm[index].n_th_hi/TIME);
                 HBP_Rate300->setBinContent(rbx*4+rm,RMSummary->rm[index].n_th_300/TIME);
             }
	  }	
      }
      if(sd==2){ first_rbx=0;} //HEM
      if(sd==3){ first_rbx=18;} //HEP
      if(sd==2 || sd==3){  // update HB plots
	  for(int rbx=0;rbx<18;rbx++)for(int rm=1;rm<=4;rm++){
             int index=RMSummary->GetRMindex(HE_RBX[rbx+first_rbx],rm);
             if(index<0 || index>=HcalFrontEndId::maxRmIndex) continue;
             if(sd==2){
                 HEM_Rate50->setBinContent(rbx*4+rm,RMSummary->rm[index].n_th_hi/TIME);
                 HEM_Rate300->setBinContent(rbx*4+rm,RMSummary->rm[index].n_th_300/TIME);
             }
             if(sd==3){
                 HEP_Rate50->setBinContent(rbx*4+rm,RMSummary->rm[index].n_th_hi/TIME);
                 HEP_Rate300->setBinContent(rbx*4+rm,RMSummary->rm[index].n_th_300/TIME);
             }
	  }	
      }
      int n=0;
      if(sd==4){ first_rbx=6; n=6;}  //HO1M
      if(sd==5){ first_rbx=12;n=12;} //HO0
      if(sd==6){ first_rbx=24;n=6;} //HO1P
      if(sd>3){ // update HO plots
	  for(int rbx=0;rbx<n;rbx++)for(int rm=1;rm<=4;rm++){
             int index=RMSummary->GetRMindex(HO_RBX[rbx+first_rbx],rm);
             if(index<0 || index>=HcalFrontEndId::maxRmIndex) continue;
             if(sd==4){
                 HO1M_Rate50->setBinContent(rbx*4*2+rm,RMSummary->rm[index].n_th_hi/TIME);
                 HO1M_Rate300->setBinContent(rbx*4*2+rm,RMSummary->rm[index].n_th_300/TIME);
             }
             if(sd==5){
                 HO0_Rate50->setBinContent(rbx*4+rm,RMSummary->rm[index].n_th_hi/TIME);
                 HO0_Rate300->setBinContent(rbx*4+rm,RMSummary->rm[index].n_th_300/TIME);
             } 
             if(sd==5){
                 HO1P_Rate50->setBinContent(rbx*4*2+rm,RMSummary->rm[index].n_th_hi/TIME);
                 HO1P_Rate300->setBinContent(rbx*4+rm,RMSummary->rm[index].n_th_300/TIME);
             } 
         }
      }
   } //sd=0;sd<9
} 

void HcalDetDiagNoiseMonitor::SaveRates(){
char   RBX[20];
int    RM;
float VAL1,VAL2,VAL3,VAL4,VAL5;
char str[500]; 
    double TIME=(double)(LastOrbit-FirstOrbit)/11223.0;
    if(TIME==0) return;
    if(OutputFilePath.size()>0){
       if(!Overwrite){
          sprintf(str,"%sHcalDetDiagNoiseData_run%06i_%i.root",OutputFilePath.c_str(),run_number,dataset_seq_number);
       }else{
          sprintf(str,"%sHcalDetDiagNoiseData.root",OutputFilePath.c_str());
       }
       TFile *theFile = new TFile(str, "RECREATE");
       if(!theFile->IsOpen()) return;
       theFile->cd();
       sprintf(str,"%d",run_number); TObjString run(str);    run.Write("run number");
       sprintf(str,"%d",ievt_);      TObjString events(str); events.Write("Total events processed");
       sprintf(str,"%d",dataset_seq_number);      TObjString dsnum(str);  dsnum.Write("Dataset number");
       Long_t t; t=time(0); strftime(str,30,"%F %T",localtime(&t)); TObjString tm(str);  tm.Write("Dataset creation time");

       TTree *tree   =new TTree("HCAL Noise data","HCAL Noise data");
       if(tree==0)   return;
       tree->Branch("RBX",            &RBX,     "RBX/C");
       tree->Branch("rm",             &RM,      "rm/I");
       tree->Branch("RATE_50",        &VAL1,    "RATE_50");
       tree->Branch("RATE_300",       &VAL2,    "RATE_300");
       tree->Branch("RATE_PIX1",      &VAL3,    "RATE_PIX1");
       tree->Branch("RATE_PIX8",      &VAL4,    "RATE_PIX8");
       tree->Branch("RATE_PIXMEAN",   &VAL5,    "RATE_PIXMEAN");
       for(int rbx=0;rbx<36;rbx++) for(int rm=1;rm<=4;rm++){
           int index=RMSummary->GetRMindex(HB_RBX[rbx],rm);
           if(index<0 || index>=HcalFrontEndId::maxRmIndex) continue;
	       sprintf(RBX,"%s",HB_RBX[rbx]);
	       RM=rm;
	       VAL1=RMSummary->rm[index].n_th_hi/TIME;
	       VAL2=RMSummary->rm[index].n_th_300/TIME;
	       VAL3=RMSummary->rm[index].n_pix_1/TIME;
	       VAL4=RMSummary->rm[index].n_pix_8/TIME;
	       if(RMSummary->rm[index].n_pix>0)VAL5=RMSummary->rm[index].pix/RMSummary->rm[index].n_pix; else VAL5=0;
               tree->Fill();
       }
       for(int rbx=0;rbx<36;rbx++) for(int rm=1;rm<=4;rm++){
           int index=RMSummary->GetRMindex(HE_RBX[rbx],rm);
           if(index<0 || index>=HcalFrontEndId::maxRmIndex) continue;
	       sprintf(RBX,"%s",HE_RBX[rbx]);
	       RM=rm;
	       VAL1=RMSummary->rm[index].n_th_hi/TIME;
	       VAL2=RMSummary->rm[index].n_th_300/TIME;
	       VAL3=RMSummary->rm[index].n_pix_1/TIME;
	       VAL4=RMSummary->rm[index].n_pix_8/TIME;
	       if(RMSummary->rm[index].n_pix>0)VAL5=RMSummary->rm[index].pix/RMSummary->rm[index].n_pix; else VAL5=0;
               tree->Fill();
       }
       for(int rbx=0;rbx<36;rbx++) for(int rm=1;rm<=4;rm++){
           int index=RMSummary->GetRMindex(HO_RBX[rbx],rm);
           if(index<0 || index>=HcalFrontEndId::maxRmIndex) continue;
	       sprintf(RBX,"%s",HO_RBX[rbx]);
	       RM=rm;
	       VAL1=RMSummary->rm[index].n_th_hi/TIME;
	       VAL2=RMSummary->rm[index].n_th_300/TIME;
	       VAL3=RMSummary->rm[index].n_pix_1/TIME;
	       VAL4=RMSummary->rm[index].n_pix_8/TIME;
	       if(RMSummary->rm[index].n_pix>0)VAL5=RMSummary->rm[index].pix/RMSummary->rm[index].n_pix; else VAL5=0;
               tree->Fill();
       }
       theFile->Write();
       theFile->Close();
       theFile->Delete();
       dataset_seq_number++;

   }
}


void HcalDetDiagNoiseMonitor::done(){}
 
HcalDetDiagNoiseMonitor::~HcalDetDiagNoiseMonitor()
{
  if(LocalRun) UpdateHistos(); SaveRates(); 

  if ( RMSummary ) delete RMSummary;
}

DEFINE_FWK_MODULE (HcalDetDiagNoiseMonitor);
