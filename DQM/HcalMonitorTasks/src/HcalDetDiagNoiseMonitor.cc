#include "DQM/HcalMonitorTasks/interface/HcalDetDiagNoiseMonitor.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"

#include "TFile.h"
#include "TTree.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

HcalDetDiagNoiseMonitor::HcalDetDiagNoiseMonitor() {
  ievt_=0;
  run_number=-1;
  NoisyEvents=0;

// ####################################

  lumi.clear();

// ####################################

}

HcalDetDiagNoiseMonitor::~HcalDetDiagNoiseMonitor(){}

void HcalDetDiagNoiseMonitor::clearME(){
  if(m_dbe){
    m_dbe->setCurrentFolder(baseFolder_);
    m_dbe->removeContents();
    m_dbe = 0;
  }
} 
void HcalDetDiagNoiseMonitor::reset(){}

void HcalDetDiagNoiseMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){
  m_dbe=NULL;
  ievt_=0;
  if(dbe!=NULL) m_dbe=dbe;
  clearME();
 
  UseDB            = ps.getUntrackedParameter<bool>  ("UseDB"  , false);
  ReferenceData    = ps.getUntrackedParameter<string>("NoiseReferenceData" ,"");
  OutputFilePath   = ps.getUntrackedParameter<string>("OutputFilePath", "");
  HPDthresholdHi   = ps.getUntrackedParameter<double>("NoiseThresholdHPDhi",30.0);
  HPDthresholdLo   = ps.getUntrackedParameter<double>("NoiseThresholdHPDlo",12.0);
  SiPMthreshold    = ps.getUntrackedParameter<double>("NoiseThresholdSiPM",150.0);
  SpikeThreshold   = ps.getUntrackedParameter<double>("NoiseThresholdSpike",0.06);
  UpdateEvents     = ps.getUntrackedParameter<int>   ("NoiseUpdateEvents",200);
  
  FEDRawDataCollection_ = ps.getUntrackedParameter<edm::InputTag>("FEDRawDataCollection",edm::InputTag("source",""));
  inputLabelDigi_       = ps.getParameter<edm::InputTag>         ("digiLabel");

// ###################################################################################################################

  hlTriggerResults_				= ps.getUntrackedParameter<edm::InputTag>("HLTriggerResults",edm::InputTag("TriggerResults","","HLT"));
  MetSource_					= ps.getUntrackedParameter<edm::InputTag>("MetSource",edm::InputTag("met"));
  JetSource_          				= ps.getUntrackedParameter<edm::InputTag>("JetSource",edm::InputTag("iterativeCone5CaloJets"));
  TrackSource_          			= ps.getUntrackedParameter<edm::InputTag>("TrackSource",edm::InputTag("generalTracks"));
  rbxCollName_    				= ps.getUntrackedParameter<std::string>("rbxCollName","hcalnoise");
  TriggerRequirement_ 				= ps.getUntrackedParameter<string>("TriggerRequirement","HLT_MET100");
  UseMetCutInsteadOfTrigger_			= ps.getUntrackedParameter<bool>("UseMetCutInsteadOfTrigger",true);
  MetCut_					= ps.getUntrackedParameter<double>("MetCut",0.0);
  JetMinEt_ 					= ps.getUntrackedParameter<double>("JetMinEt",20.0);
  JetMaxEta_ 					= ps.getUntrackedParameter<double>("JetMaxEta",2.0);
  ConstituentsToJetMatchingDeltaR_ 		= ps.getUntrackedParameter<double>("ConstituentsToJetMatchingDeltaR",0.5);
  TrackMaxIp_ 					= ps.getUntrackedParameter<double>("TrackMaxIp",0.1);
  TrackMinThreshold_ 				= ps.getUntrackedParameter<double>("TrackMinThreshold",1.0);
  MinJetChargeFraction_ 			= ps.getUntrackedParameter<double>("MinJetChargeFraction",0.05);
  MaxJetHadronicEnergyFraction_ 		= ps.getUntrackedParameter<double>("MaxJetHadronicEnergyFraction",0.98);
  caloTowerCollName_				= ps.getParameter<edm::InputTag>("caloTowerCollName");

// ###################################################################################################################

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  
  HcalBaseMonitor::setup(ps,dbe);

  baseFolder_ = rootFolder_+"HcalNoiseMonitor";
  //char *name;
  std::string name;
  if(m_dbe!=NULL){    
     m_dbe->setCurrentFolder(baseFolder_);   
     meEVT_ = m_dbe->bookInt("HcalNoiseMonitor Event Number");
     m_dbe->setCurrentFolder(baseFolder_+"/Summary Plots");
     
     name="RBX Pixel multiplicity";   PixelMult        = m_dbe->book1D(name,name,73,0,73);
     name="HPD energy";               HPDEnergy        = m_dbe->book1D(name,name,200,0,2500);
     name="RBX energy";               RBXEnergy        = m_dbe->book1D(name,name,200,0,3500);
     name="HB RM Noise Fraction Map"; HB_RBXmapRatio   = m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HB RM Spike Map";          HB_RBXmapSpikeCnt= m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HB RM Spike Amplitude Map";HB_RBXmapSpikeAmp= m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HE RM Noise Fraction Map"; HE_RBXmapRatio   = m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HE RM Spike Map";          HE_RBXmapSpikeCnt= m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HE RM Spike Amplitude Map";HE_RBXmapSpikeAmp= m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HO RM Noise Fraction Map"; HO_RBXmapRatio   = m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HO RM Spike Map";          HO_RBXmapSpikeCnt= m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HO RM Spike Amplitude Map";HO_RBXmapSpikeAmp= m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
 
     m_dbe->setCurrentFolder(baseFolder_+"/Current Plots");
     name="HB RM Noise Fraction Map (current status)"; HB_RBXmapRatioCur = m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HE RM Noise Fraction Map (current status)"; HE_RBXmapRatioCur = m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HO RM Noise Fraction Map (current status)"; HO_RBXmapRatioCur = m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     
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

       m_dbe->setCurrentFolder(baseFolder_+"/MetExpressStreamNoiseMonitoring");
       title="MET (GeV) All Events";
       name="MET_All_Events";				Met_AllEvents        = m_dbe->book1D(name,name,200,0,2000);
       Met_AllEvents->setAxisTitle(title);
       title="MET #phi All Events";
       name="METphi_All_Events";				Mephi_AllEvents        = m_dbe->book1D(name,name,70,-3.5,3.5);
       Mephi_AllEvents->setAxisTitle(title);
       title="MEx (GeV) All Events";
       name="MEx_All_Events";				Mex_AllEvents        = m_dbe->book1D(name,name,200,-1000,1000);
       Mex_AllEvents->setAxisTitle(title);
       title="MEy (GeV) All Events";
       name="MEy_All_Events";				Mey_AllEvents        = m_dbe->book1D(name,name,200,-1000,1000);
       Mey_AllEvents->setAxisTitle(title);
       title="SumET (GeV) All Events";
       name="SumEt_All_Events";				SumEt_AllEvents        = m_dbe->book1D(name,name,200,0,2000);
       SumEt_AllEvents->setAxisTitle(title);
       title="Number of LumiSections";
       name="NLumiSections";				NLumiSections        = m_dbe->book1D(name,name,1,0,1);
       NLumiSections->setAxisTitle(title);

       m_dbe->setCurrentFolder(baseFolder_+"/MetExpressStreamNoiseMonitoring"+"/SelectedForNoiseMonitoring");
       title="MET (GeV) passing selections";
       name="MET_pass_selections";			Met_passingTrigger   = m_dbe->book1D(name,name,200,0,2000);
       Met_passingTrigger->setAxisTitle(title);
       title="MET #phi passing selections";
       name="METphi_pass_selections";			Mephi_passingTrigger   = m_dbe->book1D(name,name,70,-3.5,3.5);
       Mephi_passingTrigger->setAxisTitle(title);
       title="MEx (GeV) passing selections";
       name="MEx_pass_selections";			Mex_passingTrigger   = m_dbe->book1D(name,name,200,-1000,1000);
       Mex_passingTrigger->setAxisTitle(title);
       title="MEy (GeV) passing selections";
       name="MEy_pass_selections";			Mey_passingTrigger   = m_dbe->book1D(name,name,200,-1000,1000);
       Mey_passingTrigger->setAxisTitle(title);
       title="SumET (GeV) passing selections";
       name="SumEt_pass_selections";			SumEt_passingTrigger   = m_dbe->book1D(name,name,200,0,2000);
       SumEt_passingTrigger->setAxisTitle(title);
       title="MET (GeV) corrected for noise";
       name="MET_pass_selections_correctfornoise";	CorrectedMet_passingTrigger   = m_dbe->book1D(name,name,200,0,2000);
       CorrectedMet_passingTrigger->setAxisTitle(title);
       title="MET #phi corrected for noise";
       name="METphi_pass_selections_correctfornoise";	CorrectedMephi_passingTrigger   = m_dbe->book1D(name,name,70,-3.5,3.5);
       CorrectedMephi_passingTrigger->setAxisTitle(title);
       title="MEx (GeV) corrected for noise";
       name="MEx_pass_selections_correctfornoise";	CorrectedMex_passingTrigger   = m_dbe->book1D(name,name,200,-1000,1000);
       CorrectedMex_passingTrigger->setAxisTitle(title);
       title="MEy (GeV) corrected for noise";
       name="MEy_pass_selections_correctfornoise";	CorrectedMey_passingTrigger   = m_dbe->book1D(name,name,200,-1000,1000);
       CorrectedMey_passingTrigger->setAxisTitle(title);
       title="SumET (GeV) corrected for noise";
       name="SumEt_pass_selections_correctfornoise";	CorrectedSumEt_passingTrigger   = m_dbe->book1D(name,name,200,0,2000);
       CorrectedSumEt_passingTrigger->setAxisTitle(title);
       title="Jet Hadronic Energy Fraction - passing selections";
       name="Jets_passing_selections_HadF";		HCALFraction_passingTrigger   = m_dbe->book1D(name,name,55,0,1.1);
       HCALFraction_passingTrigger->setAxisTitle(title);
       title="Jet Charge Fraction - passing selections";
       name="Jets_passing_selections_CHF";		chargeFraction_passingTrigger   = m_dbe->book1D(name,name,30,0,1.5);
       chargeFraction_passingTrigger->setAxisTitle(title);
       title="Jet E_{T} (GeV) - passing selections";
       name="Jets_Et_passing_selections";		JetEt_passingTrigger   = m_dbe->book1D(name,name,200,0,2000);
       JetEt_passingTrigger->setAxisTitle(title);
       title="Jet #eta - passing selections";
       name="Jets_Eta_passing_selections";		JetEta_passingTrigger   = m_dbe->book1D(name,name,100,-5,5);
       JetEta_passingTrigger->setAxisTitle(title);
       title="Jet #phi - passing selections";
       name="Jets_Phi_passing_selections";		JetPhi_passingTrigger   = m_dbe->book1D(name,name,70,-3.5,3.5);
       JetPhi_passingTrigger->setAxisTitle(title);
       title="Jet HadF vs CHF - passing selections";
       name="Jets_passing_selections_HadF_vs_CHF"; 	HCALFractionVSchargeFraction_passingTrigger   = m_dbe->book2D(name,name,55,0,1.1,30,0,1.5);
       HCALFractionVSchargeFraction_passingTrigger->setAxisTitle("HadF",1);
       HCALFractionVSchargeFraction_passingTrigger->setAxisTitle("CHF",2);

       m_dbe->setCurrentFolder(baseFolder_+"/MetExpressStreamNoiseMonitoring"+"/SelectedForNoiseMonitoring"+"/HcalNoiseCategory");
       title="'Noise' Jets E_{T} (GeV)";
       name="Noise_Jets_Et_passing_selections";		JetEt_passingTrigger_TaggedAnomalous   = m_dbe->book1D(name,name,200,0,2000);
       JetEt_passingTrigger_TaggedAnomalous->setAxisTitle(title);
       title="'Noise' Jets #eta";
       name="Noise_Jets_Eta_passing_selections";		JetEta_passingTrigger_TaggedAnomalous   = m_dbe->book1D(name,name,100,-5,5);
       JetEta_passingTrigger_TaggedAnomalous->setAxisTitle(title);
       title="'Noise' Jets #phi";
       name="Noise_Jets_Phi_passing_selections";		JetPhi_passingTrigger_TaggedAnomalous   = m_dbe->book1D(name,name,70,-3.5,3.5);
       JetPhi_passingTrigger_TaggedAnomalous->setAxisTitle(title);
       title="MET (GeV) passing selections & Categorized as 'Hcal Noise'";
       name="Hcal_Noise_MET_pass_selections";		Met_passingTrigger_HcalNoiseCategory   = m_dbe->book1D(name,name,200,0,2000);
       Met_passingTrigger_HcalNoiseCategory->setAxisTitle(title);
       title="MET #phi passing selections & Categorized as 'Hcal Noise'";
       name="Hcal_Noise_METphi_pass_selections";		Mephi_passingTrigger_HcalNoiseCategory   = m_dbe->book1D(name,name,70,-3.5,3.5);
       Mephi_passingTrigger_HcalNoiseCategory->setAxisTitle(title);
       title="MEx (GeV) passing selections & Categorized as 'Hcal Noise'";
       name="Hcal_Noise_MEx_pass_selections";		Mex_passingTrigger_HcalNoiseCategory   = m_dbe->book1D(name,name,200,-1000,1000);
       Mex_passingTrigger_HcalNoiseCategory->setAxisTitle(title);
       title="MEy (GeV) passing selections & Categorized as 'Hcal Noise'";
       name="Hcal_Noise_MEy_pass_selections";		Mey_passingTrigger_HcalNoiseCategory   = m_dbe->book1D(name,name,200,-1000,1000);
       Mey_passingTrigger_HcalNoiseCategory->setAxisTitle(title);
       title="SumET (GeV) passing selections & Categorized as 'Hcal Noise'";
       name="Hcal_Noise_SumEt_pass_selections";		SumEt_passingTrigger_HcalNoiseCategory   = m_dbe->book1D(name,name,200,0,2000);
       SumEt_passingTrigger_HcalNoiseCategory->setAxisTitle(title);
       title="RBX Max Zeros - passing selections & Categorized as 'Hcal Noise'";
       name="Hcal_Noise_RBX_Max_Zeros_pass_selections";	RBXMaxZeros_passingTrigger_HcalNoiseCategory   = m_dbe->book1D(name,name,30,0,30);
       RBXMaxZeros_passingTrigger_HcalNoiseCategory->setAxisTitle(title);
       title="RBX E(2ts)/E(10ts) - passing selections & Categorized as 'Hcal Noise'";
       name="Hcal_Noise_RBX_E2tsOverE10ts_pass_selections";	RBXE2tsOverE10ts_passingTrigger_HcalNoiseCategory   = m_dbe->book1D(name,name,50,0,2);
       RBXE2tsOverE10ts_passingTrigger_HcalNoiseCategory->setAxisTitle(title);
       title="RBX N RecHits - passing selections & Categorized as 'Hcal Noise'";
       name="Hcal_Noise_RBX_Nhits_pass_selections";	RBXHitsHighest_passingTrigger_HcalNoiseCategory   = m_dbe->book1D(name,name,80,0,80);
       RBXHitsHighest_passingTrigger_HcalNoiseCategory->setAxisTitle(title);
       title="HPD E(2ts)/E(10ts) - passing selections & Categorized as 'Hcal Noise'";
       name="Hcal_Noise_HPD_E2tsOverE10ts_pass_selections";	HPDE2tsOverE10ts_passingTrigger_HcalNoiseCategory   = m_dbe->book1D(name,name,50,0,2);
       HPDE2tsOverE10ts_passingTrigger_HcalNoiseCategory->setAxisTitle(title);
       title="HPD N RecHits - passing selections & Categorized as 'Hcal Noise'";
       name="Hcal_Noise_HPD_Nhits_pass_selections";	HPDHitsHighest_passingTrigger_HcalNoiseCategory   = m_dbe->book1D(name,name,30,0,30);
       HPDHitsHighest_passingTrigger_HcalNoiseCategory->setAxisTitle(title);

       m_dbe->setCurrentFolder(baseFolder_+"/MetExpressStreamNoiseMonitoring"+"/SelectedForNoiseMonitoring"+"/PhysicsCategory");
       title="MET (GeV) passing selections & Categorized as 'Physics'";
       name="Physics_MET_pass_selections";		Met_passingTrigger_PhysicsCategory   = m_dbe->book1D(name,name,200,0,2000);
       Met_passingTrigger_PhysicsCategory->setAxisTitle(title);
       title="MET #phi passing selections & Categorized as 'Physics'";
       name="Physics_METphi_pass_selections";           Mephi_passingTrigger_PhysicsCategory   = m_dbe->book1D(name,name,70,-3.5,3.5);
       Mephi_passingTrigger_PhysicsCategory->setAxisTitle(title);
       title="MEx (GeV) passing selections & Categorized as 'Physics'";
       name="Physics_MEx_pass_selections";           Mex_passingTrigger_PhysicsCategory   = m_dbe->book1D(name,name,200,-1000,1000);
       Mex_passingTrigger_PhysicsCategory->setAxisTitle(title);
       title="MEy (GeV) passing selections & Categorized as 'Physics'";
       name="Physics_MEy_pass_selections";           Mey_passingTrigger_PhysicsCategory   = m_dbe->book1D(name,name,200,-1000,1000);
       Mey_passingTrigger_PhysicsCategory->setAxisTitle(title);
       title="SumET (GeV) passing selections & Categorized as 'Physics'";
       name="Physics_SumEt_pass_selections";         SumEt_passingTrigger_PhysicsCategory   = m_dbe->book1D(name,name,200,0,2000);
       SumEt_passingTrigger_PhysicsCategory->setAxisTitle(title);
       title="RBX Max Zeros - passing selections & Categorized as 'Physics'";
       name="Physics_RBX_Max_Zeros_pass_selections";   RBXMaxZeros_passingTrigger_PhysicsCategory   = m_dbe->book1D(name,name,30,0,30);
       RBXMaxZeros_passingTrigger_PhysicsCategory->setAxisTitle(title);
       title="RBX E(2ts)/E(10ts) - passing selections & Categorized as 'Physics'";
       name="Physics_RBX_E2tsOverE10ts_pass_selections";       RBXE2tsOverE10ts_passingTrigger_PhysicsCategory   = m_dbe->book1D(name,name,50,0,2);
       RBXE2tsOverE10ts_passingTrigger_PhysicsCategory->setAxisTitle(title);
       title="RBX N RecHits - passing selections & Categorized as 'Physics'";
       name="Physics_RBX_Nhits_pass_selections";       RBXHitsHighest_passingTrigger_PhysicsCategory   = m_dbe->book1D(name,name,80,0,80);
       RBXHitsHighest_passingTrigger_PhysicsCategory->setAxisTitle(title);
       title="HPD E(2ts)/E(10ts) - passing selections & Categorized as 'Physics'";
       name="Physics_HPD_E2tsOverE10ts_pass_selections";       HPDE2tsOverE10ts_passingTrigger_PhysicsCategory   = m_dbe->book1D(name,name,50,0,2);
       HPDE2tsOverE10ts_passingTrigger_PhysicsCategory->setAxisTitle(title);
       title="HPD N RecHits - passing selections & Categorized as 'Physics'";
       name="Physics_HPD_Nhits_pass_selections";       HPDHitsHighest_passingTrigger_PhysicsCategory   = m_dbe->book1D(name,name,30,0,30);
       HPDHitsHighest_passingTrigger_PhysicsCategory->setAxisTitle(title);

     }

// ###################################################################################################################

  } 
  ReferenceRun="UNKNOWN";
  IsReference=false;
  //LoadReference();
  lmap =new HcalLogicalMap(gen.createMap());

  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDetDiagNoiseMonitor Setup -> "<<cpu_timer.cpuTime()<<std::endl;
    }

  return;
} 

void HcalDetDiagNoiseMonitor::processEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup, const HcalDbService& cond){
  bool isNoiseEvent=false;   
  if(!m_dbe) return;
   
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }


   ievt_++;
   meEVT_->Fill(ievt_);
   run_number=iEvent.id().run();

   // We do not want to look at Abort Gap events
   edm::Handle<FEDRawDataCollection> rawdata;
   iEvent.getByLabel(FEDRawDataCollection_,rawdata);
   //checking FEDs for calibration information
   for(int i=FEDNumbering::MINHCALFEDID;i<=FEDNumbering::MAXHCALFEDID; i++) {
       const FEDRawData& fedData = rawdata->FEDData(i) ;
       if ( fedData.size() < 24 ) continue ;
       if(((const HcalDCCHeader*)(fedData.data()))->getCalibType()!=hc_Null) return;
   }
  
   HcalDetDiagNoiseRMData RMs[HcalFrontEndId::maxRmIndex];
   
   try{
         edm::Handle<HBHEDigiCollection> hbhe; 
         iEvent.getByLabel(inputLabelDigi_,hbhe);
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
   }catch(...){}      
   try{
         edm::Handle<HODigiCollection> ho; 
         iEvent.getByLabel(inputLabelDigi_,ho);
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
   }catch(...){}  
//    try{ //curently we don't want to look at PMTs
//          edm::Handle<HFDigiCollection> hf;
//          iEvent.getByType(hf);
//          for(HFDigiCollection::const_iterator digi=hf->begin();digi!=hf->end();digi++){
//             
// 	     for(int i=0;i<digi->size();i++); 
// 	     
//          }   
//    }catch(...){}    
   if(isNoiseEvent){
      NoisyEvents++;
      
      // RMs loop
      for(int i=0;i<HcalFrontEndId::maxRmIndex;i++){
        if(RMs[i].n_th_hi>0){
	   RBXCurrentSummary.AddNoiseStat(i);
	   RBXSummary.AddNoiseStat(i);
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
         int index=RBXSummary.GetRMindex(rbx,rm);
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

     // met collection
     edm::Handle<CaloMETCollection> metHandle;
     iEvent.getByLabel(MetSource_, metHandle);
     const CaloMETCollection *metCol = metHandle.product();
     const CaloMET met = metCol->front();

     // Fill a histogram with the met for all events
     Met_AllEvents->Fill(met.pt());
     Mephi_AllEvents->Fill(met.phi());
     Mex_AllEvents->Fill(met.px());
     Mey_AllEvents->Fill(met.py());
     SumEt_AllEvents->Fill(met.sumEt());

     bool found = false;
     for(unsigned int i=0; i!=lumi.size(); ++i) { if(lumi.at(i) == iEvent.luminosityBlock()) {found = true; break;} }
     if(!found) {lumi.push_back(iEvent.luminosityBlock()); NLumiSections->Fill(0.5);}

     bool passedTrigger = false;
     if(!UseMetCutInsteadOfTrigger_) {
       // trigger results
       edm::Handle<edm::TriggerResults> hltTriggerResultHandle;
       iEvent.getByLabel(hlTriggerResults_, hltTriggerResultHandle);
       // Require a valid handle
       if(!hltTriggerResultHandle.isValid()) { std::cout << "invalid handle for HLT TriggerResults" << std::endl; }
       else {
         // # of triggers
         int ntrigs = hltTriggerResultHandle->size();
/*
         if (ntrigs==0) {std::cout << "%HLTInfo -- No trigger name given in TriggerResults of the input " << std::endl;}
         else {std::cout << "%HLTInfo --  Number of HLT Triggers: " << ntrigs << std::endl;}
*/
         triggerNames_.init(* hltTriggerResultHandle);
         for (int itrig = 0; itrig != ntrigs; ++itrig){
           // obtain the trigger name
           string trigName = triggerNames_.triggerName(itrig);
           // did the trigger fire?
           bool accept = hltTriggerResultHandle->accept(itrig);
/*
           std::cout << "%HLTInfo --  HLTTrigger(" << itrig << "): " << trigName << " = " << accept << std::endl;
*/
           if((trigName == TriggerRequirement_) && (accept)) {passedTrigger = true;}
         }
       }
     }

     // fill histograms for events that passed the user defined criteria (HLT_MET100 or Met>X for noise studies)
     if( ((passedTrigger) && (!UseMetCutInsteadOfTrigger_)) || ((UseMetCutInsteadOfTrigger_) && (met.pt() >= MetCut_)) ) {

       // jet collection
       Handle<CaloJetCollection> calojetHandle;
       if (!iEvent.getByLabel(JetSource_, calojetHandle))
	 {
	  if (fVerbosity) LogWarning("HcalMonitorTasks")<<" HcalDetDiagNoiseMonitor:  CaloJet collection with handle "<<JetSource_<<" not found!";
	   return;
	 }
       // track collection
       Handle<TrackCollection> trackHandle;
       if (!iEvent.getByLabel(TrackSource_, trackHandle))
	 {
	   if (fVerbosity) LogWarning("HcalMonitorTasks")<<" HcalDetDiagNoiseMonitor:  Track collection with handle "<<TrackSource_<<" not found!";
	   return;
	 }
       // HcalNoise RBX collection
       Handle<HcalNoiseRBXCollection> rbxnoisehandle;
       if (!iEvent.getByLabel(rbxCollName_, rbxnoisehandle))
	 {
	   if (fVerbosity) LogWarning("HcalMonitorTasks")<<" HcalDetDiagNoiseMonitor:  HcalNoiseRBX collection with handle "<<rbxCollName_<<" not found!";
	   return;
	 }
       // CaloTower collection
       edm::Handle<CaloTowerCollection> towerhandle;
       if (!iEvent.getByLabel(caloTowerCollName_, towerhandle))
	 {
	   if (fVerbosity) LogWarning("HcalMonitorTasks")<<" HcalDetDiagNoiseMonitor:  CaloTower collection with handle "<<caloTowerCollName_<<" not found!";
	   return;
	 }

       Met_passingTrigger->Fill(met.pt());
       Mephi_passingTrigger->Fill(met.phi());
       Mex_passingTrigger->Fill(met.px());
       Mey_passingTrigger->Fill(met.py());
       SumEt_passingTrigger->Fill(met.sumEt());
       bool isAnomalous_BasedOnHCALFraction = false;
       bool isAnomalous_BasedOnCF = false;
       double deltapx = 0;
       double deltapy = 0;
       double deltaet = 0;
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
           HCALFraction_passingTrigger->Fill(calojetIter->energyFractionHadronic());
           chargeFraction_passingTrigger->Fill(result.pt() / calojetIter->pt());
           HCALFractionVSchargeFraction_passingTrigger->Fill(calojetIter->energyFractionHadronic(), result.pt() / calojetIter->pt());
           JetEt_passingTrigger->Fill(calojetIter->et());
           JetEta_passingTrigger->Fill(calojetIter->eta());
           JetPhi_passingTrigger->Fill(calojetIter->phi());
           if((result.pt() / calojetIter->pt()) <= MinJetChargeFraction_) {isAnomalous_BasedOnCF = true;}
           if(calojetIter->energyFractionHadronic() >= MaxJetHadronicEnergyFraction_) {isAnomalous_BasedOnHCALFraction = true;}
           if( ((result.pt() / calojetIter->pt()) <= MinJetChargeFraction_) && (calojetIter->energyFractionHadronic() >= MaxJetHadronicEnergyFraction_) ) {
             JetEt_passingTrigger_TaggedAnomalous->Fill(calojetIter->et());
             JetEta_passingTrigger_TaggedAnomalous->Fill(calojetIter->eta());
             JetPhi_passingTrigger_TaggedAnomalous->Fill(calojetIter->phi());
             deltapx = deltapx + calojetIter->px();
             deltapy = deltapy + calojetIter->py();
             deltaet = deltaet + calojetIter->et();
             if(calojetIter->energyFractionHadronic() >= MaxJetHadronicEnergyFraction_) { HcalNoisyJetContainer.push_back(*calojetIter); }
           }
         }
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
           for(HcalNoiseRBXCollection::const_iterator rit=rbxnoisehandle->begin(); rit!=rbxnoisehandle->end(); ++rit) {
             HcalNoiseRBX rbx = (*rit);
             std::vector<HcalNoiseHPD> theHPDs = rbx.HPDs();
             for(std::vector<HcalNoiseHPD>::const_iterator hit=theHPDs.begin(); hit!=theHPDs.end(); ++hit) {
               HcalNoiseHPD hpd=(*hit);
               for(int iii=0; iii < (int)(nid.size()); iii++) {
                 if((nid.at(iii) == (int)(hpd.idnumber())) && (hpd.recHitEnergy(1.0) > HighestEnergyMatch)) {
                   HighestEnergyMatch = hpd.recHitEnergy(1.0);
                   nidd.clear();
                   nidd.push_back(hpd.idnumber());
                 }
               }
             }
           }
         }
       }
       if( (isAnomalous_BasedOnCF) ) {
         if(!isAnomalous_BasedOnHCALFraction) {
           Met_passingTrigger_PhysicsCategory->Fill(met.pt());
           Mephi_passingTrigger_PhysicsCategory->Fill(met.phi());
           Mex_passingTrigger_PhysicsCategory->Fill(met.px());
           Mey_passingTrigger_PhysicsCategory->Fill(met.py());
           SumEt_passingTrigger_PhysicsCategory->Fill(met.sumEt());
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
             RBXMaxZeros_passingTrigger_PhysicsCategory->Fill(rbx.maxZeros());
             RBXHitsHighest_passingTrigger_PhysicsCategory->Fill(numRBXhits);
             RBXE2tsOverE10ts_passingTrigger_PhysicsCategory->Fill(totale10ts ? totale2ts/totale10ts : -999);
             HPDHitsHighest_passingTrigger_PhysicsCategory->Fill(nHitsHighest);
             HPDE2tsOverE10ts_passingTrigger_PhysicsCategory->Fill(e10ts ? e2ts/e10ts : -999);
           }
         }
         if(isAnomalous_BasedOnHCALFraction) {
           Met_passingTrigger_HcalNoiseCategory->Fill(met.pt());
           Mephi_passingTrigger_HcalNoiseCategory->Fill(met.phi());
           Mex_passingTrigger_HcalNoiseCategory->Fill(met.px());
           Mey_passingTrigger_HcalNoiseCategory->Fill(met.py());
           SumEt_passingTrigger_HcalNoiseCategory->Fill(met.sumEt());
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
                 RBXMaxZeros_passingTrigger_HcalNoiseCategory->Fill(rbx.maxZeros());
                 RBXHitsHighest_passingTrigger_HcalNoiseCategory->Fill(numRBXhits);
                 RBXE2tsOverE10ts_passingTrigger_HcalNoiseCategory->Fill(totale10ts ? totale2ts/totale10ts : -999);
                 HPDHitsHighest_passingTrigger_HcalNoiseCategory->Fill(nHitsHighest);
                 HPDE2tsOverE10ts_passingTrigger_HcalNoiseCategory->Fill(e10ts ? e2ts/e10ts : -999);
               }
             }
           }
         }
       } else {
         Met_passingTrigger_PhysicsCategory->Fill(met.pt());
         Mephi_passingTrigger_PhysicsCategory->Fill(met.phi());
         Mex_passingTrigger_PhysicsCategory->Fill(met.px());
         Mey_passingTrigger_PhysicsCategory->Fill(met.py());
         SumEt_passingTrigger_PhysicsCategory->Fill(met.sumEt());
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
           RBXMaxZeros_passingTrigger_PhysicsCategory->Fill(rbx.maxZeros());
           RBXHitsHighest_passingTrigger_PhysicsCategory->Fill(numRBXhits);
           RBXE2tsOverE10ts_passingTrigger_PhysicsCategory->Fill(totale10ts ? totale2ts/totale10ts : -999);
           HPDHitsHighest_passingTrigger_PhysicsCategory->Fill(nHitsHighest);
           HPDE2tsOverE10ts_passingTrigger_PhysicsCategory->Fill(e10ts ? e2ts/e10ts : -999);
         }
       }
       double correctedMEx = met.px() + deltapx;
       double correctedMEy = met.py() + deltapy;
       double correctedMEphi = (correctedMEx==0 && correctedMEy==0) ? 0 : atan2(correctedMEy,correctedMEx);
       double correctedMET = sqrt((correctedMEx * correctedMEx) + (correctedMEy * correctedMEy));
       double correctedSumET = met.sumEt() - deltaet;
       CorrectedMet_passingTrigger->Fill(correctedMET);
       CorrectedMephi_passingTrigger->Fill(correctedMEphi);
       CorrectedMex_passingTrigger->Fill(correctedMEx);
       CorrectedMey_passingTrigger->Fill(correctedMEy);
       CorrectedSumEt_passingTrigger->Fill(correctedSumET);
     }

   } //if (!Online_)

// ###################################################################################################################
       
   if((ievt_%100)==0 && fVerbosity)
     std::cout <<ievt_<<"\t"<<NoisyEvents<<std::endl;

   if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalDetDiagNoiseMonitor PROCESSEVENT -> "<<cpu_timer.cpuTime()<<std::endl;
    }

   return;
}

void HcalDetDiagNoiseMonitor::UpdateHistos(){
int first_rbx=0,last_rbx=0;  
  for(int sd=0;sd<9;sd++){
     if(RBXCurrentSummary.GetStat(sd)>=UpdateEvents){
        if(sd==0){ first_rbx=0;  last_rbx=18;} //HBM
        if(sd==1){ first_rbx=18; last_rbx=36;} //HBP
        if(sd==0 || sd==1){  // update HB plots
           for(int rbx=first_rbx;rbx<last_rbx;rbx++)for(int rm=1;rm<=4;rm++){
              double val1=0,val2=0;
              if(RBXSummary.GetRMStatusValue(HB_RBX[rbx],rm,&val1)){
	        HB_RBXmapRatio->setBinContent(rm,rbx+1,val1);
                if(RBXCurrentSummary.GetRMStatusValue(HB_RBX[rbx],rm,&val2)){
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
              if(RBXSummary.GetRMStatusValue(HE_RBX[rbx],rm,&val1)){
	        HE_RBXmapRatio->setBinContent(rm,rbx+1,val1);
                if(RBXCurrentSummary.GetRMStatusValue(HE_RBX[rbx],rm,&val2)){
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
              if(RBXSummary.GetRMStatusValue(HO_RBX[rbx],rm,&val1)){
	        HO_RBXmapRatio->setBinContent(rm,rbx+1,val1);
                if(RBXCurrentSummary.GetRMStatusValue(HO_RBX[rbx],rm,&val2)){
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
	
        RBXCurrentSummary.reset(sd); 
	// disabled output statement
        //printf("update %i\n",sd); 
     }
  } 
} 

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
           if(RBXCurrentSummary.GetRMStatusValue(rbx,rm,&val)){
	       sprintf(RBX,"%s",(char *)rbx.c_str());
	       RM=rm;
	       RM_INDEX=RBXCurrentSummary.GetRMindex(rbx,rm);
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
      try{ 
         f = new TFile(ReferenceData.c_str(),"READ");
      }catch(...){ return ;}
      if(!f->IsOpen()){ return ;}
      TObjString *STR=(TObjString *)f->Get("run number");
      
      if(STR){ string Ref(STR->String()); ReferenceRun=Ref;}
      
      TTree*  t=(TTree*)f->Get("HCAL Noise data");
      if(!t) return;
      t->SetBranchAddress("rm_index",       &RM_INDEX);
      t->SetBranchAddress("relative_noise", &VAL);
      for(int ievt=0;ievt<t->GetEntries();ievt++){
         t->GetEntry(ievt);
	 RBXCurrentSummary.SetReference(RM_INDEX,VAL);
	 RBXSummary.SetReference(RM_INDEX,VAL);
      }
      f->Close();
      IsReference=true;
   }
} 

void HcalDetDiagNoiseMonitor::done(){   /*SaveReference();*/ } 
