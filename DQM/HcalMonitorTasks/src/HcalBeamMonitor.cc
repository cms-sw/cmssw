#include "DQM/HcalMonitorTasks/interface/HcalBeamMonitor.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "Geometry/HcalTowerAlgo/src/HcalHardcodeGeometryData.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"

#include <iomanip>
#include <cmath>

// define sizes of ieta arrays for each subdetector

#define PI        3.1415926535897932
#define HBETASIZE 34  // one more bin than needed, I think
#define HEETASIZE 60  // ""
#define HOETASIZE 32  // ""
#define HFETASIZE 84  // ""

/*  Task calculates various moments of Hcal recHits 

    v1.0
    16 August 2008
    by Jeff Temple

*/

// constructor
HcalBeamMonitor::HcalBeamMonitor(const edm::ParameterSet& ps):
  HcalBaseDQMonitor(ps),
  ETA_OFFSET_HB(16),
  ETA_OFFSET_HE(29),
  ETA_BOUND_HE(17),
  ETA_OFFSET_HO(15),
  ETA_OFFSET_HF(41),
  ETA_BOUND_HF(29)
{
  Online_                = ps.getUntrackedParameter<bool>("online",false);
  mergeRuns_             = ps.getUntrackedParameter<bool>("mergeRuns",false);
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("TaskFolder","BeamMonitor_Hcal");
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;
  AllowedCalibTypes_     = ps.getUntrackedParameter<std::vector<int> > ("AllowedCalibTypes");
  skipOutOfOrderLS_      = ps.getUntrackedParameter<bool>("skipOutOfOrderLS",true);
  NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",4000);
  makeDiagnostics_       = ps.getUntrackedParameter<bool>("makeDiagnostics",false);

  // Beam Monitor-specific stuff

  // Collection type info
  digiLabel_             =ps.getUntrackedParameter<edm::InputTag>("digiLabel");
  tok_hfdigi_ = consumes<HFDigiCollection>(digiLabel_);
  hbheRechitLabel_       = ps.getUntrackedParameter<edm::InputTag>("hbheRechitLabel");

  tok_hbhe_ = consumes<HBHERecHitCollection>(hbheRechitLabel_);

  hoRechitLabel_         = ps.getUntrackedParameter<edm::InputTag>("hoRechitLabel");
  tok_ho_ = consumes<HORecHitCollection>(hoRechitLabel_);

  hfRechitLabel_         = ps.getUntrackedParameter<edm::InputTag>("hfRechitLabel");
  tok_hf_ = consumes<HFRecHitCollection>(hfRechitLabel_);

  // minimum events required in lumi block for tests to be processed
  minEvents_       = ps.getUntrackedParameter<int>("minEvents",500); 
  lumiqualitydir_ = ps.getUntrackedParameter<std::string>("lumiqualitydir","");
  if (lumiqualitydir_.size()>0 && lumiqualitydir_.substr(lumiqualitydir_.size()-1,lumiqualitydir_.size())!="/")
    lumiqualitydir_.append("/");
  occThresh_ = ps.getUntrackedParameter<double>("occupancyThresh",0.0625);  // energy required to be counted by dead/hot checks
  hotrate_        = ps.getUntrackedParameter<double>("hotrate",0.25);
  minBadCells_    = ps.getUntrackedParameter<int>("minBadCells",10);
  Overwrite_      = ps.getUntrackedParameter<bool>("Overwrite",false);
  setupDone_      = false;
}



HcalBeamMonitor::~HcalBeamMonitor() {}

void HcalBeamMonitor::reset() 
{
  CenterOfEnergyRadius->Reset();
  CenterOfEnergy->Reset();
  COEradiusVSeta->Reset();

  HBCenterOfEnergyRadius->Reset();
  HBCenterOfEnergy->Reset();
  HECenterOfEnergyRadius->Reset();
  HECenterOfEnergy->Reset();
  HOCenterOfEnergyRadius->Reset();
  HOCenterOfEnergy->Reset();
  HFCenterOfEnergyRadius->Reset();
  HFCenterOfEnergy->Reset();

  Etsum_eta_L->Reset();
  Etsum_eta_S->Reset();
  Etsum_phi_L->Reset();
  Etsum_phi_S->Reset();
  Etsum_ratio_p->Reset();
  Etsum_ratio_m->Reset();
  Etsum_map_L->Reset();
  Etsum_map_S->Reset();
  Etsum_ratio_map->Reset();
  Etsum_rphi_L->Reset();
  Etsum_rphi_S->Reset();
  Energy_Occ->Reset();

  Occ_rphi_L->Reset();
  Occ_rphi_S->Reset();
  Occ_eta_L->Reset();
  Occ_eta_S->Reset();
  Occ_phi_L->Reset();
  Occ_phi_S->Reset();
  Occ_map_L->Reset();
  Occ_map_S->Reset();
  
  HFlumi_ETsum_perwedge->Reset();
  HFlumi_Occupancy_above_thr_r1->Reset();
  HFlumi_Occupancy_between_thrs_r1->Reset();
  HFlumi_Occupancy_below_thr_r1->Reset();
  HFlumi_Occupancy_above_thr_r2->Reset();
  HFlumi_Occupancy_between_thrs_r2->Reset();
  HFlumi_Occupancy_below_thr_r2->Reset();

  HFlumi_Occupancy_per_channel_vs_lumiblock_RING1->Reset();
  HFlumi_Occupancy_per_channel_vs_lumiblock_RING2->Reset();
  HFlumi_Occupancy_per_channel_vs_BX_RING1->Reset();
  HFlumi_Occupancy_per_channel_vs_BX_RING2->Reset();
  HFlumi_ETsum_vs_BX->Reset();
  HFlumi_Et_per_channel_vs_lumiblock->Reset();

  HFlumi_occ_LS->Reset();
  HFlumi_total_hotcells->Reset();
  HFlumi_total_deadcells->Reset();
  HFlumi_diag_hotcells->Reset();
  HFlumi_diag_deadcells->Reset();


  HFlumi_Ring1Status_vs_LS->Reset();
  HFlumi_Ring2Status_vs_LS->Reset();
}


void HcalBeamMonitor::setup(DQMStore::IBooker &ib)
{
  if (setupDone_)
    return;
  setupDone_ = true;
   if (debug_>0) std::cout <<"<HcalBeamMonitor::setup> Setup in progress..."<<std::endl;
  HcalBaseDQMonitor::setup(ib);

  //jason's
  ib.setCurrentFolder(subdir_);
  CenterOfEnergyRadius = ib.book1D("CenterOfEnergyRadius",
				       "Center Of Energy radius",
				       200,0,1);
      
  CenterOfEnergyRadius->setAxisTitle("(normalized) radius",1);
      
  CenterOfEnergy = ib.book2D("CenterOfEnergy",
				 "Center of Energy;normalized x coordinate;normalize y coordinate",
				 40,-1,1,
				 40,-1,1);

  COEradiusVSeta = ib.bookProfile("COEradiusVSeta",
				      "Center of Energy radius vs i#eta",
				      172,-43,43,
				      20,0,1);
  COEradiusVSeta->setAxisTitle("i#eta",1);
  COEradiusVSeta->setAxisTitle("(normalized) radius",2);
      
  std::stringstream histname;
  std::stringstream histtitle;
  ib.setCurrentFolder(subdir_+"HB");
  HBCenterOfEnergyRadius = ib.book1D("HBCenterOfEnergyRadius",
					 "HB Center Of Energy radius",
					 200,0,1);
  HBCenterOfEnergy = ib.book2D("HBCenterOfEnergy",
				   "HB Center of Energy",
				   40,-1,1,
				   40,-1,1);
  if (makeDiagnostics_)
    {
      for (int i=-16;i<=16;++i)
	{
	  if (i==0) continue;
	  histname.str("");
	  histtitle.str("");
	  histname<<"HB_CenterOfEnergyRadius_ieta"<<i;
	  histtitle<<"HB Center Of Energy ieta = "<<i;
	  HB_CenterOfEnergyRadius[i+ETA_OFFSET_HB]=ib.book1D(histname.str().c_str(),
								 histtitle.str().c_str(),
								 200,0,1);
	} // end of HB loop
    }
  ib.setCurrentFolder(subdir_+"HE");
  HECenterOfEnergyRadius = ib.book1D("HECenterOfEnergyRadius",
					 "HE Center Of Energy radius",
					 200,0,1);
  HECenterOfEnergy = ib.book2D("HECenterOfEnergy",
				   "HE Center of Energy",
				   40,-1,1,
				   40,-1,1);

  if (makeDiagnostics_)
    {
      for (int i=-29;i<=29;++i)
	{
	  if (abs(i)<ETA_BOUND_HE) continue;
	  histname.str("");
	  histtitle.str("");
	  histname<<"HE_CenterOfEnergyRadius_ieta"<<i;
	  histtitle<<"HE Center Of Energy ieta = "<<i;
	  HE_CenterOfEnergyRadius[i+ETA_OFFSET_HE]=ib.book1D(histname.str().c_str(),
								 histtitle.str().c_str(),
								 200,0,1);
	} // end of HE loop
    }
  ib.setCurrentFolder(subdir_+"HO");
  HOCenterOfEnergyRadius = ib.book1D("HOCenterOfEnergyRadius",
					 "HO Center Of Energy radius",
					 200,0,1);
  HOCenterOfEnergy = ib.book2D("HOCenterOfEnergy",
				   "HO Center of Energy",
				   40,-1,1,
				   40,-1,1);
  if (makeDiagnostics_)
    {
      for (int i=-15;i<=15;++i)
	{
	  if (i==0) continue;
	  histname.str("");
	  histtitle.str("");
	  histname<<"HO_CenterOfEnergyRadius_ieta"<<i;
	  histtitle<<"HO Center Of Energy radius ieta = "<<i;
	  HO_CenterOfEnergyRadius[i+ETA_OFFSET_HO]=ib.book1D(histname.str().c_str(),
								 histtitle.str().c_str(),
								 200,0,1);
	} // end of HO loop
    }
  ib.setCurrentFolder(subdir_+"HF");
  HFCenterOfEnergyRadius = ib.book1D("HFCenterOfEnergyRadius",
					 "HF Center Of Energy radius",
					 200,0,1);
  HFCenterOfEnergy = ib.book2D("HFCenterOfEnergy",
				   "HF Center of Energy",
				   40,-1,1,
				   40,-1,1);
  if (makeDiagnostics_)
    {
      for (int i=-41;i<=41;++i)
	{
	  if (abs(i)<ETA_BOUND_HF) continue;
	  histname.str("");
	  histtitle.str("");
	  histname<<"HF_CenterOfEnergyRadius_ieta"<<i;
	  histtitle<<"HF Center Of Energy radius ieta = "<<i;
	  HF_CenterOfEnergyRadius[i+ETA_OFFSET_HF]=ib.book1D(histname.str().c_str(),
								 histtitle.str().c_str(),
								 200,0,1);
	} // end of HF loop
    }
      
  ib.setCurrentFolder(subdir_+"Lumi");
  // Wenhan's 
  // reducing bins from ",200,0,2000" to ",40,0,800"
      
  float radiusbins[13]={169,201,240,286,340,406,483,576,686,818,975,1162,1300};
  float phibins[71]={-3.5,-3.4,-3.3,-3.2,-3.1,
		     -3.0,-2.9,-2.8,-2.7,-2.6,-2.5,-2.4,-2.3,-2.2,-2.1,
		     -2.0,-1.9,-1.8,-1.7,-1.6,-1.5,-1.4,-1.3,-1.2,-1.1,
		     -1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,
		     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
		     1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
		     2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
		     3.0, 3.1, 3.2, 3.3, 3.4, 3.5};
  Etsum_eta_L=ib.bookProfile("Et Sum vs Eta Long Fiber","Et Sum per Area vs Eta Long Fiber",27,0,27,100,0,100);
  Etsum_eta_S=ib.bookProfile("Et Sum vs Eta Short Fiber","Et Sum per Area vs Eta Short Fiber",27,0,27,100,0,100);
  Etsum_phi_L=ib.bookProfile("Et Sum vs Phi Long Fiber","Et Sum per Area vs Phi Long Fiber",36,0.5,72.5,100,0,100);
  Etsum_phi_S=ib.bookProfile("Et Sum vs Phi Short Fiber","Et Sum per Area crossing vs Phi Short Fiber",36,0.5,72.5,100,0,100);

  Etsum_ratio_p=ib.book1D("Occ vs PMT events HF+","Energy difference of Long and Short Fiber HF+ in PMT events",105,0.,1.05);
  Energy_Occ=ib.book1D("Occ vs Energy","Occupancy vs Energy",200,0,2000);
  Etsum_ratio_m=ib.book1D("Occ vs PMT events HF-","Energy difference of Long and Short Fiber HF- in PMT events",105,0.,1.05);
  Etsum_map_L=ib.book2D("EtSum 2D phi and eta Long Fiber","Et Sum 2D phi and eta Long Fiber",27,0,27,36,0.5,72.5);
  Etsum_map_S=ib.book2D("EtSum 2D phi and eta Short Fiber","Et Sum 2D phi and eta Short Fiber",27,0,27,36,0.5,72.5);

  Etsum_rphi_S=ib.book2D("EtSum 2D phi and radius Short Fiber","Et Sum 2D phi and radius Short Fiber",12, radiusbins, 70, phibins);
  Etsum_rphi_L=ib.book2D("EtSum 2D phi and radius Long Fiber","Et Sum 2D phi and radius Long Fiber",12, radiusbins, 70, phibins);

  Etsum_ratio_map=ib.book2D("Abnormal PMT events","Abnormal PMT events",
				8,0,8,36, 0.5,72.5);
  SetEtaLabels(Etsum_ratio_map);

  HFlumi_occ_LS = ib.book2D("HFlumi_occ_LS","HFlumi occupancy for current LS",
				8,0,8,36, 0.5,72.5);
  SetEtaLabels(HFlumi_occ_LS);
      
  HFlumi_total_deadcells = ib.book2D("HFlumi_total_deadcells","Number of dead lumi channels for LS with at least 10 bad channels",
					 8,0,8,36,0.5,72.5);
  SetEtaLabels(HFlumi_total_deadcells);
  HFlumi_total_hotcells = ib.book2D("HFlumi_total_hotcells","Number of hot lumi channels for LS with at least 10 bad channels",
					8,0,8,36,0.5,72.5);
  SetEtaLabels(HFlumi_total_hotcells);

  HFlumi_diag_deadcells = ib.book2D("HFlumi_diag_deadcells","Channels that had no hit for at least one LS",
				       8,0,8,36,0.5,72.5);
  SetEtaLabels(HFlumi_diag_deadcells);
  HFlumi_diag_hotcells = ib.book2D("HFlumi_diag_hotcells","Channels that appeared hot for at least one LS",
				      8,0,8,36,0.5,72.5);
  SetEtaLabels(HFlumi_diag_hotcells);



  Occ_rphi_S=ib.book2D("Occ 2D phi and radius Short Fiber","Occupancy 2D phi and radius Short Fiber",12, radiusbins, 70, phibins);
  Occ_rphi_L=ib.book2D("Occ 2D phi and radius Long Fiber","Occupancy 2D phi and radius Long Fiber",12, radiusbins, 70, phibins);
  Occ_eta_S=ib.bookProfile("Occ vs iEta Short Fiber","Occ per Bunch crossing vs iEta Short Fiber",27,0,27,40,0,800);
  Occ_eta_L=ib.bookProfile("Occ vs iEta Long Fiber","Occ per Bunch crossing vs iEta Long Fiber",27,0,27,40,0,800);
      
  Occ_phi_L=ib.bookProfile("Occ vs iPhi Long Fiber","Occ per Bunch crossing vs iPhi Long Fiber",36,0.5,72.5,40,0,800);
      
  Occ_phi_S=ib.bookProfile("Occ vs iPhi Short Fiber","Occ per Bunch crossing vs iPhi Short Fiber",36,0.5,72.5,40,0,800);
      
  Occ_map_L=ib.book2D("Occ_map Long Fiber","Occ Map long Fiber (above threshold)",27,0,27,36,0.5,72.5);
  Occ_map_S=ib.book2D("Occ_map Short Fiber","Occ Map Short Fiber (above threshold)",27,0,27,36,0.5,72.5);

  std::stringstream binlabel;
  for (int zz=0;zz<27;++zz)
    {
      if (zz<13)
	binlabel<<zz-41;
      else if (zz==13)
	binlabel<<"NULL";
      else
	binlabel<<zz+15;
      Occ_eta_S->setBinLabel(zz+1,binlabel.str().c_str());
      Occ_eta_L->setBinLabel(zz+1,binlabel.str().c_str());
      Occ_map_S->setBinLabel(zz+1,binlabel.str().c_str());
      Occ_map_L->setBinLabel(zz+1,binlabel.str().c_str());
      Etsum_eta_S->setBinLabel(zz+1,binlabel.str().c_str());
      Etsum_eta_L->setBinLabel(zz+1,binlabel.str().c_str());
      Etsum_map_S->setBinLabel(zz+1,binlabel.str().c_str());
      Etsum_map_L->setBinLabel(zz+1,binlabel.str().c_str());
      binlabel.str("");
    }

  //HFlumi plots
  HFlumi_ETsum_perwedge =  ib.book1D("HF lumi ET-sum per wedge","HF lumi ET-sum per wedge;wedge",36,1,37);
  HFlumi_ETsum_perwedge->getTH1F()->SetMinimum(0); 
      
  HFlumi_Occupancy_above_thr_r1 =  ib.book1D("HF lumi Occupancy above threshold ring1","HF lumi Occupancy above threshold ring1;wedge",36,1,37);
  HFlumi_Occupancy_between_thrs_r1 = ib.book1D("HF lumi Occupancy between thresholds ring1","HF lumi Occupancy between thresholds ring1;wedge",36,1,37);
  HFlumi_Occupancy_below_thr_r1 = ib.book1D("HF lumi Occupancy below threshold ring1","HF lumi Occupancy below threshold ring1;wedge",36,1,37);
  HFlumi_Occupancy_above_thr_r2 = ib.book1D("HF lumi Occupancy above threshold ring2","HF lumi Occupancy above threshold ring2;wedge",36,1,37);
  HFlumi_Occupancy_between_thrs_r2 = ib.book1D("HF lumi Occupancy between thresholds ring2","HF lumi Occupancy between thresholds ring2;wedge",36,1,37);
  HFlumi_Occupancy_below_thr_r2 = ib.book1D("HF lumi Occupancy below threshold ring2","HF lumi Occupancy below threshold ring2;wedge",36,1,37);

  HFlumi_Occupancy_above_thr_r1->getTH1F()->SetMinimum(0);
  HFlumi_Occupancy_between_thrs_r1->getTH1F()->SetMinimum(0);
  HFlumi_Occupancy_below_thr_r1->getTH1F()->SetMinimum(0);
  HFlumi_Occupancy_above_thr_r2->getTH1F()->SetMinimum(0);
  HFlumi_Occupancy_between_thrs_r2->getTH1F()->SetMinimum(0);
  HFlumi_Occupancy_below_thr_r2->getTH1F()->SetMinimum(0);
 
  HFlumi_Occupancy_per_channel_vs_lumiblock_RING1 = ib.bookProfile("HFlumiRing1OccupancyPerChannelVsLB",
								      "HFlumi Occupancy per channel vs lumi-block (RING 1);LS; -ln(empty fraction)",
								      NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,10000);
  HFlumi_Occupancy_per_channel_vs_lumiblock_RING2 = ib.bookProfile("HFlumiRing2OccupancyPerChannelVsLB","HFlumi Occupancy per channel vs lumi-block (RING 2);LS; -ln(empty fraction)",NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,10000);

  HFlumi_Occupancy_per_channel_vs_BX_RING1 = ib.bookProfile("HFlumi Occupancy per channel vs BX (RING 1)","HFlumi Occupancy per channel vs BX (RING 1);BX; -ln(empty fraction)",4000,0,4000,100,0,10000);
  HFlumi_Occupancy_per_channel_vs_BX_RING2 = ib.bookProfile("HFlumi Occupancy per channel vs BX (RING 2)","HFlumi Occupancy per channel vs BX (RING 2);BX; -ln(empty fraction)",4000,0,4000,100,0,10000);
  HFlumi_ETsum_vs_BX = ib.bookProfile("HFlumi_ETsum_vs_BX","HFlumi ETsum vs BX; BX; ETsum",4000,0,4000,100,0,10000);

  HFlumi_Et_per_channel_vs_lumiblock = ib.bookProfile("HFlumi Et per channel vs lumi-block","HFlumi Et per channel vs lumi-block;LS;ET",NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,10000);

  HFlumi_Occupancy_per_channel_vs_lumiblock_RING1->getTProfile()->SetMarkerStyle(20);
  HFlumi_Occupancy_per_channel_vs_lumiblock_RING2->getTProfile()->SetMarkerStyle(20);
  HFlumi_Et_per_channel_vs_lumiblock->getTProfile()->SetMarkerStyle(20);

  HFlumi_Ring1Status_vs_LS = ib.bookProfile("HFlumi_Ring1Status_vs_LS","Fraction of good Ring 1 channels vs LS;LS; Fraction of Good Channels",NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,10000);
  HFlumi_Ring2Status_vs_LS = ib.bookProfile("HFlumi_Ring2Status_vs_LS","Fraction of good Ring 2 channels vs LS;LS; Fraction of Good Channels",NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,10000);
  HFlumi_Ring1Status_vs_LS->getTProfile()->SetMarkerStyle(20);
  HFlumi_Ring2Status_vs_LS->getTProfile()->SetMarkerStyle(20);

  return;
}

void HcalBeamMonitor::bookHistograms(DQMStore::IBooker &ib, const edm::Run& run, const edm::EventSetup& c)
{
  HcalBaseDQMonitor::bookHistograms(ib,run,c);  

  if (debug_>1) std::cout <<"HcalBeamMonitor::bookHistograms"<<std::endl;
  HcalBaseDQMonitor::bookHistograms(ib,run,c);

  lastProcessedLS_=0;
  runNumber_=run.id().run();
  if (lumiqualitydir_.size()>0 && Online_==true)
    {
      if (Overwrite_==false)
	outfile_ <<lumiqualitydir_<<"HcalHFLumistatus_"<<runNumber_<<".txt";
      else
	outfile_ <<lumiqualitydir_<<"HcalHFLumistatus.txt";
      std::ofstream outStream(outfile_.str().c_str()); // recreate the file, rather than appending to it
      outStream<<"## Run "<<runNumber_<<std::endl;
      outStream<<"## LumiBlock\tRing1Status\t\tRing2Status\t\tGlobalStatus\tNentries"<<std::endl;
      outStream.close();
    }

  // Get expected good channels in run according to channel quality database
  // Get channel quality status info for each run

  // Default number of expected good channels in the run
  ring1totalchannels_=144;
  ring2totalchannels_=144;
  BadCells_.clear(); // remove any old maps
  // Get Channel quality info for the run
  // Exclude bad channels from overall calculation
  edm::ESHandle<HcalChannelQuality> p;
  c.get<HcalChannelQualityRcd>().get("withTopo",p);
  const HcalChannelQuality* chanquality = p.product();
  std::vector<DetId> mydetids = chanquality->getAllChannels();
  
  for (unsigned int i=0;i<mydetids.size();++i)
    {
      if (mydetids[i].det()!=DetId::Hcal) continue;
      HcalDetId id=mydetids[i];
      
      if (id.subdet()!=HcalForward) continue;
      if ((id.depth()==1 && (abs(id.ieta())==33 || abs(id.ieta())==34)) ||
	  (id.depth()==2 && (abs(id.ieta())==35 || abs(id.ieta())==36)))
	{
	  const HcalChannelStatus* origstatus=chanquality->getValues(id);
	  HcalChannelStatus mystatus(origstatus->rawId(),origstatus->getValue());
	  if (mystatus.isBitSet(HcalChannelStatus::HcalCellHot)) 
	    BadCells_[id]=HcalChannelStatus::HcalCellHot;
	  
	  else if (mystatus.isBitSet(HcalChannelStatus::HcalCellDead))
	    BadCells_[id]=HcalChannelStatus::HcalCellDead;
	  
	  if (mystatus.isBitSet(HcalChannelStatus::HcalCellHot) || 
	      mystatus.isBitSet(HcalChannelStatus::HcalCellDead))
	    {
	      if (id.depth()==1) --ring1totalchannels_;
	      else if (id.depth()==2) --ring2totalchannels_;
	    }
	} // if ((id.depth()==1) ...
    } // for (unsigned int i=0;...)
    
  if (tevt_==0) this->setup(ib); // create all histograms; not necessary if merging runs together
  if (mergeRuns_==false) this->reset(); // call reset at start of all runs

  return;

} // void HcalBeamMonitor::bookHistograms(const edm::Run& run, const edm::EventSetup& c)


void HcalBeamMonitor::analyze(const edm::Event& e, const edm::EventSetup& c)
{
  HcalBaseDQMonitor::analyze(e,c);
  if (!IsAllowedCalibType()) return;
  if (LumiInOrder(e.luminosityBlock())==false) return;

  // try to get rechits and digis
  edm::Handle<HFDigiCollection> hf_digi;

  edm::Handle<HBHERecHitCollection> hbhe_rechit;
  edm::Handle<HORecHitCollection> ho_rechit;
  edm::Handle<HFRecHitCollection> hf_rechit;
  
  if (!(e.getByToken(tok_hfdigi_,hf_digi)))
    {
      edm::LogWarning("HcalBeamMonitor")<< digiLabel_<<" hf_digi not available";
      return;
    }

  if (!(e.getByToken(tok_hbhe_,hbhe_rechit)))
    {
      edm::LogWarning("HcalBeamMonitor")<< hbheRechitLabel_<<" hbhe_rechit not available";
      return;
    }

  if (!(e.getByToken(tok_hf_,hf_rechit)))
    {
      edm::LogWarning("HcalBeamMonitor")<< hfRechitLabel_<<" hf_rechit not available";
      return;
    }
  if (!(e.getByToken(tok_ho_,ho_rechit)))
    {
      edm::LogWarning("HcalBeamMonitor")<< hoRechitLabel_<<" ho_rechit not available";
      return;
    }

  //good event; increment counters and process
//  HcalBaseDQMonitor::analyze(e,c);
  processEvent(*hbhe_rechit, *ho_rechit, *hf_rechit, *hf_digi, e.bunchCrossing());

} //void HcalBeamMonitor::analyze(const edm::Event& e, const edm::EventSetup& c)


void HcalBeamMonitor::processEvent(const HBHERecHitCollection& hbheHits,
				   const HORecHitCollection& hoHits,
				   const HFRecHitCollection& hfHits,
                                   const HFDigiCollection& hf,
				   int   bunchCrossing
				   )
  
{ 
  //processEvent loop
  HBHERecHitCollection::const_iterator HBHEiter;
  HORecHitCollection::const_iterator HOiter;
  HFRecHitCollection::const_iterator HFiter;

  double totalX=0;
  double totalY=0;
  double totalE=0;

  double HBtotalX=0;
  double HBtotalY=0;
  double HBtotalE=0;
  double HEtotalX=0;
  double HEtotalY=0;
  double HEtotalE=0;
  double HOtotalX=0;
  double HOtotalY=0;
  double HOtotalE=0;
  double HFtotalX=0;
  double HFtotalY=0;
  double HFtotalE=0;
     
  float hitsp[13][36][2];
  float hitsm[13][36][2];
  float hitsp_Et[13][36][2];
  float hitsm_Et[13][36][2];
  
  for(int m=0;m<13;m++){
    for(int n=0;n<36;n++){
      hitsp[m][n][0]=0;
      hitsp[m][n][1]=0; 
      hitsm[m][n][0]=0;
      hitsm[m][n][1]=0;

      hitsp_Et[m][n][0]=0;
      hitsp_Et[m][n][1]=0; 
      hitsm_Et[m][n][0]=0;
      hitsm_Et[m][n][1]=0;
    }
  }

  if(hbheHits.size()>0)
    {
      double HB_weightedX[HBETASIZE]={0.};
      double HB_weightedY[HBETASIZE]={0.};
      double HB_energy[HBETASIZE]={0.};
      
      double HE_weightedX[HEETASIZE]={0.};
      double HE_weightedY[HEETASIZE]={0.};
      double HE_energy[HEETASIZE]={0.};
      
      int ieta, iphi;
      
      for (HBHEiter=hbheHits.begin(); 
	   HBHEiter!=hbheHits.end(); 
	   ++HBHEiter) 
	{ 
	  
	  // loop over all hits
	  if (HBHEiter->energy()<0) continue; // don't consider negative-energy cells
	  HcalDetId id(HBHEiter->detid().rawId());
	  ieta=id.ieta();
	  iphi=id.iphi();
	  
	  int index=-1;
	  if ((HcalSubdetector)(id.subdet())==HcalBarrel)
	    {
	      HBtotalX+=HBHEiter->energy()*cos(PI*iphi/36.);
	      HBtotalY+=HBHEiter->energy()*sin(PI*iphi/36.);
	      HBtotalE+=HBHEiter->energy();
	      
	      index=ieta+ETA_OFFSET_HB;
	      if (index<0 || index>= HBETASIZE) continue;
	      HB_weightedX[index]+=HBHEiter->energy()*cos(PI*iphi/36.);
	      HB_weightedY[index]+=HBHEiter->energy()*sin(PI*iphi/36.);
	      HB_energy[index]+=HBHEiter->energy();
	    } // if id.subdet()==HcalBarrel
	  
	  else
	    {
	      HEtotalX+=HBHEiter->energy()*cos(PI*iphi/36.);
	      HEtotalY+=HBHEiter->energy()*sin(PI*iphi/36.);
	      HEtotalE+=HBHEiter->energy();
	      
	      index=ieta+ETA_OFFSET_HE;
	      if (index<0 || index>= HEETASIZE) continue;
	      HE_weightedX[index]+=HBHEiter->energy()*cos(PI*iphi/36.);
	      HE_weightedY[index]+=HBHEiter->energy()*sin(PI*iphi/36.);
	      HE_energy[index]+=HBHEiter->energy();
	    }
	} // for (HBHEiter=hbheHits.begin()...
	  // Fill each histogram
      
      int hbeta=ETA_OFFSET_HB;
      for (int i=-1*hbeta;i<=hbeta;++i)
	{
	  if (i==0) continue;
	  int index = i+ETA_OFFSET_HB;
	  if (index<0 || index>= HBETASIZE) continue;
	  if (HB_energy[index]==0) continue;
	  double moment=pow(HB_weightedX[index],2)+pow(HB_weightedY[index],2);
	  moment=pow(moment,0.5);
	  moment/=HB_energy[index];
	  if (moment!=0)
	    {
	      if (makeDiagnostics_) HB_CenterOfEnergyRadius[index]->Fill(moment);
	      COEradiusVSeta->Fill(i,moment);
	    }
	} // for (int i=-1*hbeta;i<=hbeta;++i)

      int heeta=ETA_OFFSET_HE;
      for (int i=-1*heeta;i<=heeta;++i)
	{
	  if (i==0) continue;
	  if (i>-1*ETA_BOUND_HE && i <ETA_BOUND_HE) continue;
	  int index = i + ETA_OFFSET_HE;
	  if (index<0 || index>= HEETASIZE) continue;
	  if (HE_energy[index]==0) continue;
	  double moment=pow(HE_weightedX[index],2)+pow(HE_weightedY[index],2);
	  moment=pow(moment,0.5);
	  moment/=HE_energy[index];
	  if (moment!=0)
	    {
	      if (makeDiagnostics_) HE_CenterOfEnergyRadius[index]->Fill(moment);
	      COEradiusVSeta->Fill(i,moment);
	    }
	} // for (int i=-1*heeta;i<=heeta;++i)

    } // if (hbheHits.size()>0)

  
  // HO loop
  if(hoHits.size()>0)
    {
      double HO_weightedX[HOETASIZE]={0.};
      double HO_weightedY[HOETASIZE]={0.};
      double HO_energy[HOETASIZE]={0.};
      double offset;

      int ieta, iphi;
      for (HOiter=hoHits.begin(); 
	   HOiter!=hoHits.end(); 
	   ++HOiter) 
	{ 
	  // loop over all cells
	  if (HOiter->energy()<0) continue;  // don't include negative-energy cells?
	  HcalDetId id(HOiter->detid().rawId());
	  ieta=id.ieta();
	  iphi=id.iphi();

	  HOtotalX+=HOiter->energy()*cos(PI*iphi/36.);
	  HOtotalY+=HOiter->energy()*sin(PI*iphi/36.);
	  HOtotalE+=HOiter->energy();

	  int index=ieta+ETA_OFFSET_HO;
	  if (index<0 || index>= HOETASIZE) continue;
	  HO_weightedX[index]+=HOiter->energy()*cos(PI*iphi/36.);
	  HO_weightedY[index]+=HOiter->energy()*sin(PI*iphi/36.);
	  HO_energy[index]+=HOiter->energy();
	} // for (HOiter=hoHits.begin();...)
	  
      for (int i=-1*ETA_OFFSET_HO;i<=ETA_OFFSET_HO;++i)
	{
	  if (i==0) continue;
	  int index = i + ETA_OFFSET_HO;
	  if (index < 0 || index>= HOETASIZE) continue;
	  if (HO_energy[index]==0) continue;
	  double moment=pow(HO_weightedX[index],2)+pow(HO_weightedY[index],2);
	  moment=pow(moment,0.5);
	  moment/=HO_energy[index];
	  // Shift HO values by 0.5 units in eta relative to HB
	  offset = (i>0 ? 0.5: -0.5);
	  if (moment!=0)
	    {
	      if (makeDiagnostics_) HO_CenterOfEnergyRadius[index]->Fill(moment);
	      COEradiusVSeta->Fill(i+offset,moment);
	    }
	} // for (int i=-1*hoeta;i<=hoeta;++i)
    } // if (hoHits.size()>0)
    
  ///////////////////////////////////
  // HF loop

  Etsum_ratio_map->Fill(-1,-1,1); // fill underflow bin with number of events
  {
    if(hfHits.size()>0)
      {
	double HF_weightedX[HFETASIZE]={0.};
	double HF_weightedY[HFETASIZE]={0.};
	double HF_energy[HFETASIZE]={0.};
	double offset;
	
	// Assume ZS until shown otherwise
	double emptytowersRing1 = ring1totalchannels_;
	double emptytowersRing2 = ring2totalchannels_;
	double ZStowersRing1 = ring1totalchannels_;
	double ZStowersRing2 = ring2totalchannels_;
	
	int ieta, iphi;
	float et,eta,phi,r;

	HFlumi_occ_LS->Fill(-1,-1,1 ); // event counter in occupancy histogram underflow bin
	// set maximum to HFlumi_occ_LS->getBinContent(0,0)?  
	// that won't work -- offline will add multiple histograms, and maximum will get screwed up?
	// No, we can add it here, but we also need a call to setMaximum in the client as well.
	HFlumi_occ_LS->getTH2F()->SetMaximum(HFlumi_occ_LS->getBinContent(0,0));

	double etx=0, ety=0;

	for (HFiter=hfHits.begin(); 
	     HFiter!=hfHits.end(); 
	     ++HFiter) 
	  {  // loop on hfHits
	    // If hit present, don't count it as ZS any more
	    ieta = HFiter->id().ieta();
	    iphi = HFiter->id().iphi();

	    int binieta=ieta;
	    if (ieta<0) binieta+=41;
	    else if (ieta>0) binieta-=15;

	    // Count that hit was found in one of the rings used for luminosity calculation.
	    // If so, decrease the number of empty channels per ring by 1
	    if (abs(ieta)>=33 && abs(ieta)<=36) // luminosity ring check
	      {
		// don't subtract away cells that have already been removed as bad
		if (BadCells_.find(HFiter->id())==BadCells_.end()) // bad cell not found
		  {
		    if ((abs(ieta)<35) && HFiter->id().depth()==1) --ZStowersRing1;
		    else if ((abs(ieta)>34) && HFiter->id().depth()==2) -- ZStowersRing2;
		  }
	      }

	    if (HFiter->energy()<0) continue;  // don't include negative-energy cells?

	    eta=theHFEtaBounds[abs(ieta)-29];
	    et=HFiter->energy()/cosh(eta)/area[abs(ieta)-29];
	    if (abs(ieta)>=33 && abs(ieta)<=36) // Luminosity ring check
	      {
		// don't count cells that are below threshold, or that have been marked bad in Chan Stat DB
		if (et>=occThresh_ && BadCells_.find(HFiter->id())==BadCells_.end() ) // minimum ET threshold
		  {
		    if ((abs(ieta)<35) && HFiter->id().depth()==1) --emptytowersRing1;
		    else if ((abs(ieta)>34) && HFiter->id().depth()==2) -- emptytowersRing2;
		  }
	      }
	    r=radius[abs(ieta)-29];
	    if(HFiter->id().iphi()<37)
	      phi=HFiter->id().iphi()*0.087266;
	    else phi=(HFiter->id().iphi()-72)*0.087266;
           
	    
	    if (HFiter->id().depth()==1)
	      {
		Etsum_eta_L->Fill(binieta,et);
		Etsum_phi_L->Fill(iphi,et);
		Etsum_map_L->Fill(binieta,iphi,et);
		Etsum_rphi_L->Fill(r,phi,et);
	      
		if(ieta>0) {
		  hitsp[ieta-29][(HFiter->id().iphi()-1)/2][0]=HFiter->energy();
		  hitsp_Et[ieta-29][(HFiter->id().iphi()-1)/2][0]=et;
		}
		else if(ieta<0) {
		  hitsm[-ieta-29][(HFiter->id().iphi()-1)/2][0]=HFiter->energy(); 
		  hitsm_Et[-ieta-29][(HFiter->id().iphi()-1)/2][0]=et; 
		}
	      } // if (HFiter->id().depth()==1)
         
	    //Fill 3 histos for Short Fibers :
	    if (HFiter->id().depth()==2)
	      {
		Etsum_eta_S->Fill(binieta,et);
		Etsum_phi_S->Fill(iphi,et);
		Etsum_rphi_S->Fill(r,phi,et); 
		Etsum_map_S->Fill(binieta,iphi,et);
		if(ieta>0)  
		  {
		    hitsp[ieta-29][(HFiter->id().iphi()-1)/2][1]=HFiter->energy();
		    hitsp_Et[ieta-29][(HFiter->id().iphi()-1)/2][1]=et;
		  }
		else if(ieta<0)  { 
		  hitsm[-ieta-29][(HFiter->id().iphi()-1)/2][1]=HFiter->energy();
		  hitsm_Et[-ieta-29][(HFiter->id().iphi()-1)/2][1]=et;
		}
          
	      } // depth()==2
	    Energy_Occ->Fill(HFiter->energy()); 
            
	    //HF: no non-threshold occupancy map is filled?

	    if ((abs(ieta) == 33 || abs(ieta) == 34) && HFiter->id().depth() == 1)
	      { 
		etx+=et*cos(PI*iphi/36.);
		ety+=et*sin(PI*iphi/36.);

		HFlumi_Et_per_channel_vs_lumiblock->Fill(currentLS,et);
		if (et>occThresh_)
		  {
		    int etabin=0;
		    if (ieta<0)
		      etabin=36+ieta; // bins 0-3 correspond to ieta = -36, -35, -34, -33
		    else
		      etabin=ieta-29; // bins 4-7 correspond to ieta = 33, 34, 35, 36
		    HFlumi_occ_LS->Fill(etabin,HFiter->id().iphi());
		  }
	      }

	    else if ((abs(ieta) == 35 || abs(ieta) == 36) && HFiter->id().depth() == 2)
	      { 
		etx+=et*cos(PI*iphi/36.);
		ety+=et*sin(PI*iphi/36.);

		HFlumi_Et_per_channel_vs_lumiblock->Fill(currentLS,et);
		if (et>occThresh_)
		  {
		    int etabin=0;
		    if (ieta<0)
		      etabin=36+ieta; // bins 0-3 correspond to ieta = -36, -35, -34, -33
		    else
		      etabin=ieta-29; // bins 4-7 correspond to ieta = 33, 34, 35, 36
		    HFlumi_occ_LS->Fill(etabin,HFiter->id().iphi());
		  }
	      }

	    // Fill occupancy plots.
	    
	    int value=0;
	    if(et>occThresh_) value=1;

	    if (HFiter->id().depth()==1)
	      {
		Occ_eta_L->Fill(binieta,value);
		Occ_phi_L->Fill(iphi,value);
		Occ_map_L->Fill(binieta,iphi,value);
		Occ_rphi_L->Fill(r,phi,value);
	      }
	      
	    else if (HFiter->id().depth()==2)
	      {
		Occ_eta_S->Fill(binieta,value);
		Occ_phi_S->Fill(iphi,value);
		Occ_map_S->Fill(binieta,iphi,value);
		Occ_rphi_S->Fill(r,phi,value);
	      }  
	    HcalDetId id(HFiter->detid().rawId());

	    HFtotalX+=HFiter->energy()*cos(PI*iphi/36.);
	    HFtotalY+=HFiter->energy()*sin(PI*iphi/36.);
	    HFtotalE+=HFiter->energy();

	    int index=ieta+ETA_OFFSET_HF;
	    if (index<0 || index>= HFETASIZE) continue;
	    HF_weightedX[index]+=HFiter->energy()*cos(PI*iphi/36.);
	    HF_weightedY[index]+=HFiter->energy()*sin(PI*iphi/36.);
	    HF_energy[index]+=HFiter->energy();
	    
	  } // for (HFiter=hfHits.begin();...)
	
	// looped on all HF hits; calculate empty fraction
	//  empty towers  = # of cells with ET < 0.0625 GeV, or cells missing because of ZS
	//  Calculated as :  144 - (# of cells with ET >= 0.0625 GeV)
	//  At some point, allow for calculations when channels are masked (and less than 144 channels expected)

	// Check Ring 1
	double logvalue=-1;
	if (ring1totalchannels_>0)
	  {
	    if (emptytowersRing1>0)
	      logvalue=-1.*log(emptytowersRing1/ring1totalchannels_);
	    HFlumi_Occupancy_per_channel_vs_lumiblock_RING1->Fill(currentLS,logvalue);
	    HFlumi_Occupancy_per_channel_vs_BX_RING1->Fill(bunchCrossing,logvalue);
	  }
	// Check Ring 2
	logvalue=-1;
	if (ring2totalchannels_>0)
	  {
	    if (emptytowersRing2>0)
	      logvalue=-1.*log(emptytowersRing2/ring2totalchannels_);
	    HFlumi_Occupancy_per_channel_vs_lumiblock_RING2->Fill(currentLS,logvalue);
	    HFlumi_Occupancy_per_channel_vs_BX_RING2->Fill(bunchCrossing,logvalue);
	  }

	HFlumi_ETsum_vs_BX->Fill(bunchCrossing,pow(etx*etx+ety*ety,0.5));
	int hfeta=ETA_OFFSET_HF;
	for (int i=-1*hfeta;i<=hfeta;++i)
	  {
	    if (i==0) continue;
	    if (i>-1*ETA_BOUND_HF && i <ETA_BOUND_HF) continue;
	    int index = i + ETA_OFFSET_HF;
	    if (index<0 || index>= HFETASIZE) continue;
	    if (HF_energy[index]==0) continue;
	    double moment=pow(HF_weightedX[index],2)+pow(HF_weightedY[index],2);
	    moment=pow(moment,0.5);
	    moment/=HF_energy[index];
	    offset = (i>0 ? 0.5: -0.5);
	    if (moment!=0)
	      {
		if (makeDiagnostics_) HF_CenterOfEnergyRadius[index]->Fill(moment);
		COEradiusVSeta->Fill(i+offset,moment);
	      }
	  } // for (int i=-1*hfeta;i<=hfeta;++i)
	float ratiom,ratiop;
	  
	for(int i=0;i<13;i++){
	  for(int j=0;j<36;j++){
	      
	    if(hitsp[i][j][0]==hitsp[i][j][1]) continue;
	      
	    if (hitsp[i][j][0] < 1.2 && hitsp[i][j][1] < 1.8) continue;
	    //use only lumi rings
	    if (((i+29) < 33) || ((i+29) > 36)) continue;
	    ratiop=fabs((fabs(hitsp[i][j][0])-fabs(hitsp[i][j][1]))/(fabs(hitsp[i][j][0])+fabs(hitsp[i][j][1])));
	    //cout<<ratiop<<std::endl;
	    if ((hitsp_Et[i][j][0] > 5. && hitsp[i][j][1] < 1.8) || (hitsp_Et[i][j][1] > 5. &&  hitsp[i][j][0] < 1.2)){
	      Etsum_ratio_p->Fill(ratiop);
	      if(abs(ratiop>0.95)) Etsum_ratio_map->Fill(i,2*j+1); // i=4,5,6,7 for HFlumi rings 
	    }
	  }
	}
	  
	for(int p=0;p<13;p++){
	  for(int q=0;q<36;q++){
	      
	    if(hitsm[p][q][0]==hitsm[p][q][1]) continue;

	    if (hitsm[p][q][0] < 1.2 && hitsm[p][q][1] < 1.8) continue;
	    //use only lumi rings
	    if (((p+29) < 33) || ((p+29) > 36)) continue;
	    ratiom=fabs((fabs(hitsm[p][q][0])-fabs(hitsm[p][q][1]))/(fabs(hitsm[p][q][0])+fabs(hitsm[p][q][1])));         
	    if ((hitsm_Et[p][q][0] > 5. && hitsm[p][q][1] < 1.8) || (hitsm_Et[p][q][1] > 5. && hitsm[p][q][0] < 1.2)){
	      Etsum_ratio_m->Fill(ratiom);
	      if(abs(ratiom>0.95)) Etsum_ratio_map->Fill(7-p,2*q+1); // p=4,5,6,7 for HFlumi rings
	      //p=7:  ieta=-36; p=4:  ieta=-33
	    }
	  }
	} 
      } // if (hfHits.size()>0)
  
    totalX=HBtotalX+HEtotalX+HOtotalX+HFtotalX;
    totalY=HBtotalY+HEtotalY+HOtotalY+HFtotalY;
    totalE=HBtotalE+HEtotalE+HOtotalE+HFtotalE;

    double moment;
    if (HBtotalE>0)
      {
	moment=pow(HBtotalX*HBtotalX+HBtotalY*HBtotalY,0.5)/HBtotalE;
	HBCenterOfEnergyRadius->Fill(moment);
	HBCenterOfEnergy->Fill(HBtotalX/HBtotalE, HBtotalY/HBtotalE);
      }
    if (HEtotalE>0)
      {
	moment=pow(HEtotalX*HEtotalX+HEtotalY*HEtotalY,0.5)/HEtotalE;
	HECenterOfEnergyRadius->Fill(moment);
	HECenterOfEnergy->Fill(HEtotalX/HEtotalE, HEtotalY/HEtotalE);
      }
    if (HOtotalE>0)
      {
	moment=pow(HOtotalX*HOtotalX+HOtotalY*HOtotalY,0.5)/HOtotalE;
	HOCenterOfEnergyRadius->Fill(moment);
	HOCenterOfEnergy->Fill(HOtotalX/HOtotalE, HOtotalY/HOtotalE);
      }
    if (HFtotalE>0)
      {
	moment=pow(HFtotalX*HFtotalX+HFtotalY*HFtotalY,0.5)/HFtotalE;
	HFCenterOfEnergyRadius->Fill(moment);
	HFCenterOfEnergy->Fill(HFtotalX/HFtotalE, HFtotalY/HFtotalE);
      }
    if (totalE>0)
      {
	moment = pow(totalX*totalX+totalY*totalY,0.5)/totalE;
	CenterOfEnergyRadius->Fill(moment);
	CenterOfEnergy->Fill(totalX/totalE, totalY/totalE);
      }


    
    for (HFDigiCollection::const_iterator j=hf.begin(); j!=hf.end(); j++){
      const HFDataFrame digi = (const HFDataFrame)(*j);
      //  calibs_= cond.getHcalCalibrations(digi.id());  // Old method was made private. 
      //       float en=0;
      //       float ts =0; float bs=0;
      //       int maxi=0; float maxa=0;
      //       for(int i=sigS0_; i<=sigS1_; i++){
      // 	if(digi.sample(i).adc()>maxa){maxa=digi.sample(i).adc(); maxi=i;}
      //       }
      //       for(int i=sigS0_; i<=sigS1_; i++){	  
      // 	float tmp1 =0;   
      //         int j1=digi.sample(i).adc();
      //         tmp1 = (LedMonAdc2fc[j1]+0.5);   	  
      // 	en += tmp1-calibs_.pedestal(digi.sample(i).capid());
      // 	if(i>=(maxi-1) && i<=maxi+1){
      // 	  ts += i*(tmp1-calibs_.pedestal(digi.sample(i).capid()));
      // 	  bs += tmp1-calibs_.pedestal(digi.sample(i).capid());
      // 	}
      //       }

      //---HFlumiplots
      int theTStobeused = 6;
      // will have masking later:
      int mask=1; 
      if(mask!=1) continue;
      //if we want to sum the 10 TS instead of just taking one:
      for (int i=0; i<digi.size(); i++) {
	if (i==theTStobeused) {
	  float tmpET =0;
	  int jadc=digi.sample(i).adc();
	  //NOW LUT used in HLX are only identy LUTs, so Et filled
	  //with unlinearised adc, ie tmpET = jadc
	  //	  tmpET = (adc2fc[jadc]+0.5);
	  tmpET = jadc;

	  //-find which wedge we are in
	  //  ETsum and Occupancy will be summed for both L and S
	  if(digi.id().ieta()>28){
	    if((digi.id().iphi()==1)||(digi.id().iphi()==71)){
	      HFlumi_ETsum_perwedge->Fill(1,tmpET);
              if((digi.id().ieta()==33)||(digi.id().ieta()==34)) {
		if(jadc>100) HFlumi_Occupancy_above_thr_r1->Fill(1,1);
		if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r1->Fill(1,1);
		if(jadc<10) HFlumi_Occupancy_below_thr_r1->Fill(1,1);
	      }
	      else if((digi.id().ieta()==35)||(digi.id().ieta()==36)) {
		if(jadc>100) HFlumi_Occupancy_above_thr_r2->Fill(1,1);
		if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r2->Fill(1,1);
		if(jadc<10) HFlumi_Occupancy_below_thr_r2->Fill(1,1);
	      }
	    }
	    else {
	      for (int iwedge=2; iwedge<19; iwedge++) {
		int itmp=4*(iwedge-1);
		if( (digi.id().iphi()==(itmp+1)) || (digi.id().iphi()==(itmp-1))) {
                  HFlumi_ETsum_perwedge->Fill(iwedge,tmpET);
		  if((digi.id().ieta()==33)||(digi.id().ieta()==34)) {
		    if(jadc>100) HFlumi_Occupancy_above_thr_r1->Fill(iwedge,1);
		    if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r1->Fill(iwedge,1);
		    if(jadc<10) HFlumi_Occupancy_below_thr_r1->Fill(iwedge,1);
		  }
		  else if((digi.id().ieta()==35)||(digi.id().ieta()==36)) {
		    if(jadc>100) HFlumi_Occupancy_above_thr_r2->Fill(iwedge,1);
		    if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r2->Fill(iwedge,1);
		    if(jadc<10) HFlumi_Occupancy_below_thr_r2->Fill(iwedge,1);
		  }
                  iwedge=99;
		}
	      }
	    }
	  }  //--endif ieta in HF+
	  else if(digi.id().ieta()<-28){
	    if((digi.id().iphi()==1)||(digi.id().iphi()==71)){
	      HFlumi_ETsum_perwedge->Fill(19,tmpET);
              if((digi.id().ieta()==-33)||(digi.id().ieta()==-34)) {
		if(jadc>100) HFlumi_Occupancy_above_thr_r1->Fill(19,1);
		if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r1->Fill(19,1);
		if(jadc<10) HFlumi_Occupancy_below_thr_r1->Fill(19,1);
	      }
	      else if((digi.id().ieta()==-35)||(digi.id().ieta()==-36)) {
		if(jadc>100) HFlumi_Occupancy_above_thr_r2->Fill(19,1);
		if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r2->Fill(19,1);
		if(jadc<10) HFlumi_Occupancy_below_thr_r2->Fill(19,1);
	      }
	    }
	    else {
	      for (int iw=2; iw<19; iw++) {
		int itemp=4*(iw-1);
		if( (digi.id().iphi()==(itemp+1)) || (digi.id().iphi()==(itemp-1))) {
                  HFlumi_ETsum_perwedge->Fill(iw+18,tmpET);
		  if((digi.id().ieta()==-33)||(digi.id().ieta()==-34)) {
		    if(jadc>100) HFlumi_Occupancy_above_thr_r1->Fill(iw+18,1);
		    if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r1->Fill(iw+18,1);
		    if(jadc<10) HFlumi_Occupancy_below_thr_r1->Fill(iw+18,1);
		  }
		  else if((digi.id().ieta()==-35)||(digi.id().ieta()==-36)) {
		    if(jadc>100) HFlumi_Occupancy_above_thr_r2->Fill(iw+18,1);
		    if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r2->Fill(iw+18,1);
		    if(jadc<10) HFlumi_Occupancy_below_thr_r2->Fill(iw+18,1);
		  }
                  iw=99;
		}
	      }
	    }
	  }//---endif ieta inHF-
	}//---endif TS=nr6
      } 
    }//------end loop over TS for lumi
    return;
  }
}
 // void HcalBeamMonitor::processEvent(const HBHERecHit Collection&hbheHits; ...)


void HcalBeamMonitor::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
					   const edm::EventSetup& c)
  
{
  // reset histograms that get updated each luminosity section

  if (LumiInOrder(lumiSeg.luminosityBlock())==false) return;
  HcalBaseDQMonitor::beginLuminosityBlock(lumiSeg,c);

  if (lumiSeg.luminosityBlock()==lastProcessedLS_) return; // we're seeing more events from current lumi section (after some break) -- should not reset histogram
  ProblemsCurrentLB->Reset();
  HFlumi_occ_LS->Reset();
  std::stringstream title;
  title <<"HFlumi occupancy for LS # " <<currentLS;
  HFlumi_occ_LS->getTH2F()->SetTitle(title.str().c_str());
  return;
} // void HcalBeamMonitor::beginLuminosityBlock()

void HcalBeamMonitor::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
					 const edm::EventSetup& c)
{
  if (debug_>1) std::cout <<"<HcalBeamMonitor::endLuminosityBlock>"<<std::endl;
  if (LumiInOrder(lumiSeg.luminosityBlock())==false)
    {
      if (debug_>1)  
	std::cout <<"<HcalBeamMonitor::endLuminosityBlock>  Failed LumiInOrder test!"<<std::endl;
      return;
    }
  lastProcessedLS_=lumiSeg.luminosityBlock();
  float Nentries=HFlumi_occ_LS->getBinContent(-1,-1);
  if (debug_>3) 
    std::cout <<"Number of entries in this LB = "<<Nentries<<std::endl;

  if (Nentries<minEvents_) 
    {
      // not enough entries to determine status; fill everything with -1 and return
      HFlumi_Ring1Status_vs_LS->Fill(currentLS,-1);
      HFlumi_Ring2Status_vs_LS->Fill(currentLS,-1);
      if (Online_==false)
	return;
      // write to output file if required (Online running)
      if (lumiqualitydir_.size()==0)
	return;
      // dump out lumi quality file
      std::ofstream outStream(outfile_.str().c_str(),std::ios::app);
      outStream<<currentLS<<"\t\t-1\t\t\t-1\t\t\t-1\t\t"<<Nentries<<std::endl;
      outStream.close();
      return;
    }
  if (Nentries==0) return;


  HFlumi_total_deadcells->Fill(-1,-1,1); // counts good lumi sections in underflow bin
  HFlumi_total_hotcells->Fill(-1,-1,1);
  HFlumi_diag_deadcells->Fill(-1,-1,1); // counts good lumi sections in underflow bin
  HFlumi_diag_hotcells->Fill(-1,-1,1);

  // ADD IETA MAP
  int ietamap[8]={-36,-35,-34,-33,33,34,35,36};
  int ieta=-1, iphi = -1, depth=-1;
  int badring1=0;
  int badring2=0;
  int ndeadcells=0;
  int nhotcells=0;
  
  // Loop over cells once to count hot & dead chanels
  for (int x=1;x<=HFlumi_occ_LS->getTH2F()->GetNbinsX();++x)
    {
      for (int y=1;y<=HFlumi_occ_LS->getTH2F()->GetNbinsY();++y)
	{

	  // Skip over channels that are flagged as bad
	  if (x<=8)
	    ieta=ietamap[x-1];
	  else
	    ieta=-1;
	  iphi=2*y-1;
	  if (abs(ieta)==33 || abs(ieta)==34)  depth=1;
	  else if (abs(ieta)==35 || abs(ieta)==36) depth =2;
	  else depth = -1;
	  if (depth !=-1 && ieta!=1)
	    {
	      HcalDetId thisID(HcalForward, ieta, iphi, depth);
	      if (BadCells_.find(thisID)!=BadCells_.end())
		continue;
	    }
	  double Ncellhits=HFlumi_occ_LS->getBinContent(x,y);
	  if (Ncellhits==0)
	    {
	      ++ndeadcells;
	      HFlumi_diag_deadcells->Fill(x-1,2*y-1,1);
	    }
	  // hot if present in more than 25% of events in the LS
	  if (Ncellhits>hotrate_*Nentries) 
	    {
	      ++nhotcells;
	      HFlumi_diag_hotcells->Fill(x-1,2*y-1,1);
	    }
	  if (Ncellhits==0 || Ncellhits>hotrate_*Nentries) // cell was either hot or dead
	    {
	      if (depth==1)  badring1++;
	      else if (depth==2)  badring2++;
	    }
	} // loop over y
    } // loop over x

  // Fill problem histogram underflow bind with number of events
  ProblemsCurrentLB->Fill(-1,-1,levt_);
  if (ndeadcells+nhotcells>=minBadCells_)
    {
      // Fill with number of error channels * events (assume bad for all events in LS)
      ProblemsCurrentLB->Fill(6,0,(ndeadcells+nhotcells)*levt_);
      for (int x=1;x<=HFlumi_occ_LS->getTH2F()->GetNbinsX();++x)
	{
	  for (int y=1;y<=HFlumi_occ_LS->getTH2F()->GetNbinsY();++y)
	    {
	      if (x<=8)
		ieta=ietamap[x-1];
	      else
		ieta=-1;
	      iphi=2*y-1;
	      if (abs(ieta)==33 || abs(ieta)==34)  depth=1;
	      else if (abs(ieta)==35 || abs(ieta)==36) depth =2;
	      else depth = -1;
	      if (depth !=-1 && ieta!=1)
		{
		  // skip over channels that are flagged as bad
		  HcalDetId thisID(HcalForward, ieta, iphi, depth);
		  if (BadCells_.find(thisID)!=BadCells_.end())
		    continue;
		}
	      double Ncellhits=HFlumi_occ_LS->getBinContent(x,y);
	      if (Ncellhits==0)
		{
		  // One new luminosity section found with no entries for the cell in question
		  HFlumi_total_deadcells->Fill(x-1,2*y-1,1);
		} // dead cell check
	      
	      // hot if present in more than 25% of events in the LS
	      if (Ncellhits>hotrate_*Nentries)
		{
		  HFlumi_total_hotcells->Fill(x-1,2*y-1,1);
		} // hot cell check
	    } // loop over y
	} // loop over x
    } // if (ndeadcells+nhotcells>=minBadCells_)

  // Fill fraction of bad channels found in this LS
  double ring1status=0;
  double ring2status=0;
  if (ring1totalchannels_==0)
    ring1status=0;
  else
    ring1status=1-1.*badring1/ring1totalchannels_;
  HFlumi_Ring1Status_vs_LS->Fill(currentLS,ring1status);
  if (ring2totalchannels_==0)
    ring2status=0;
  else
    ring2status=1-1.*badring2/ring2totalchannels_;
  HFlumi_Ring2Status_vs_LS->Fill(currentLS,ring2status);  
  
  // Good status:  ring1 and ring2 status both > 90%
  int totalstatus=0;
  if (ring1status>0.9 && ring2status>0.9)
    totalstatus=1;
  else 
    {
      if (ring1status<=0.9)
	totalstatus-=2;
      if (ring2status<=0.9)
	totalstatus-=4;
    }

  if (lumiqualitydir_.size()==0)
    return;
  // dump out lumi quality file
  std::ofstream outStream(outfile_.str().c_str(),std::ios::app);
  outStream.precision(6);
  outStream<<currentLS<<"\t\t"<<ring1status<<"\t\t"<<ring2status<<"\t\t"<<totalstatus<<"\t\t"<<Nentries<<std::endl;
  outStream.close();
  return;
}

const float HcalBeamMonitor::area[]={0.111,0.175,0.175,0.175,0.175,0.175,0.174,0.178,0.172,0.175,0.178,0.346,0.604};
const float HcalBeamMonitor::radius[]={1300,1162,975,818,686,576,483,406,340,286,240,201,169};

void HcalBeamMonitor::SetEtaLabels(MonitorElement * h)
{
  h->getTH2F()->GetXaxis()->SetBinLabel(1,"-36S");
  h->getTH2F()->GetXaxis()->SetBinLabel(2,"-35S");
  h->getTH2F()->GetXaxis()->SetBinLabel(3,"-34L");
  h->getTH2F()->GetXaxis()->SetBinLabel(4,"-33L");
  h->getTH2F()->GetXaxis()->SetBinLabel(5,"33L");
  h->getTH2F()->GetXaxis()->SetBinLabel(6,"34L");
  h->getTH2F()->GetXaxis()->SetBinLabel(7,"35S");
  h->getTH2F()->GetXaxis()->SetBinLabel(8,"36S");
  return;
}

DEFINE_FWK_MODULE(HcalBeamMonitor);

