#include "DQMOffline/Hcal/interface/HcalRecHitsAnalyzer.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

HcalRecHitsAnalyzer::HcalRecHitsAnalyzer(edm::ParameterSet const& conf) {

  // DQM ROOT output
  outputFile_ = conf.getUntrackedParameter<std::string>("outputFile", "myfile.root");
  
  if ( outputFile_.size() != 0 ) {
    edm::LogInfo("OutputInfo") << " Hcal RecHit Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("OutputInfo") << " Hcal RecHit Task histograms will NOT be saved";
  }
  
  nevtot = 0;
 
  hcalselector_ = conf.getUntrackedParameter<std::string>("hcalselector", "all");
  ecalselector_ = conf.getUntrackedParameter<std::string>("ecalselector", "yes");
  eventype_     = conf.getUntrackedParameter<std::string>("eventype", "single");
  sign_         = conf.getUntrackedParameter<std::string>("sign", "*");
  //useAllHistos_ = conf.getUntrackedParameter<bool>("useAllHistos", false);

  //Collections
  tok_hbhe_ = consumes<HBHERecHitCollection>(conf.getUntrackedParameter<edm::InputTag>("HBHERecHitCollectionLabel"));
  tok_hf_  = consumes<HFRecHitCollection>(conf.getUntrackedParameter<edm::InputTag>("HFRecHitCollectionLabel"));
  tok_ho_ = consumes<HORecHitCollection>(conf.getUntrackedParameter<edm::InputTag>("HORecHitCollectionLabel"));
  tok_EB_ = consumes<EBRecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEB"));
  tok_EE_ = consumes<EERecHitCollection>(edm::InputTag("ecalRecHit","EcalRecHitsEE"));

  subdet_ = 5;
  if (hcalselector_ == "noise") subdet_ = 0;
  if (hcalselector_ == "HB"   ) subdet_ = 1;
  if (hcalselector_ == "HE"   ) subdet_ = 2;
  if (hcalselector_ == "HO"   ) subdet_ = 3;
  if (hcalselector_ == "HF"   ) subdet_ = 4;
  if (hcalselector_ == "all"  ) subdet_ = 5;
  if (hcalselector_ == "ZS"   ) subdet_ = 6;

  etype_ = 1;
  if (eventype_ == "multi") etype_ = 2;

  iz = 1;
  if(sign_ == "-") iz = -1;
  if(sign_ == "*") iz = 0;

  imc = 0;

  }

  void HcalRecHitsAnalyzer::dqmBeginRun(const edm::Run& run, const edm::EventSetup& es){
  
    edm::ESHandle<HcalDDDRecConstants> pHRNDC;
    es.get<HcalRecNumberingRecord>().get( pHRNDC );
    hcons = &(*pHRNDC);
    maxDepthHB_ = hcons->getMaxDepth(0);
    maxDepthHE_ = hcons->getMaxDepth(1);
    maxDepthHF_ = hcons->getMaxDepth(2);
    maxDepthHO_ = hcons->getMaxDepth(3);

    edm::ESHandle<CaloGeometry> geometry;

    es.get<CaloGeometryRecord > ().get(geometry);

    const CaloGeometry* geo = geometry.product();
    const HcalGeometry* gHB = (HcalGeometry*)(geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel));
    const HcalGeometry* gHE = (HcalGeometry*)(geo->getSubdetectorGeometry(DetId::Hcal,HcalEndcap));
    const HcalGeometry* gHO = (HcalGeometry*)(geo->getSubdetectorGeometry(DetId::Hcal,HcalOuter));
    const HcalGeometry* gHF = (HcalGeometry*)(geo->getSubdetectorGeometry(DetId::Hcal,HcalForward));

    nChannels_[1] = gHB->getHxSize(1); 
    nChannels_[2] = gHE->getHxSize(2); 
    nChannels_[3] = gHO->getHxSize(3); 
    nChannels_[4] = gHF->getHxSize(4); 

    nChannels_[0] = nChannels_[1] + nChannels_[2] + nChannels_[3] + nChannels_[4];

    //std::cout << "Channels HB:" << nChannels_[1] << " HE:" << nChannels_[2] << " HO:" << nChannels_[3] << " HF:" << nChannels_[4] << std::endl;

    //We hardcode the HF depths because in the dual readout configuration, rechits are not defined for depths 3&4
    maxDepthHF_ = (maxDepthHF_ > 2 ? 2 : maxDepthHF_); //We retain the dynamic possibility that HF might have 0 or 1 depths

    maxDepthAll_ = ( maxDepthHB_ + maxDepthHO_ > maxDepthHE_ ? maxDepthHB_ + maxDepthHO_ : maxDepthHE_ );
    maxDepthAll_ = ( maxDepthAll_ > maxDepthHF_ ? maxDepthAll_ : maxDepthHF_ );

    //Get Phi segmentation from geometry, use the max phi number so that all iphi values are included.

    int NphiMax = hcons->getNPhi(0);

    NphiMax = (hcons->getNPhi(1) > NphiMax ? hcons->getNPhi(1) : NphiMax);
    NphiMax = (hcons->getNPhi(2) > NphiMax ? hcons->getNPhi(2) : NphiMax);
    NphiMax = (hcons->getNPhi(3) > NphiMax ? hcons->getNPhi(3) : NphiMax);

    //Center the iphi bins on the integers
    iphi_min_ = 0.5;
    iphi_max_ = NphiMax + 0.5;
    iphi_bins_ = (int) (iphi_max_ - iphi_min_);

    //Retain classic behavior, all plots have same ieta range.

    int iEtaMax = (hcons->getEtaRange(0).second > hcons->getEtaRange(1).second ? hcons->getEtaRange(0).second : hcons->getEtaRange(1).second);
    iEtaMax = (iEtaMax > hcons->getEtaRange(2).second ? iEtaMax : hcons->getEtaRange(2).second);
    iEtaMax = (iEtaMax > hcons->getEtaRange(3).second ? iEtaMax : hcons->getEtaRange(3).second);

    //Give an empty bin around the subdet ieta range to make it clear that all ieta rings have been included
    ieta_min_ = -iEtaMax - 1.5;
    ieta_max_ = iEtaMax + 1.5;
    ieta_bins_ = (int) (ieta_max_ - ieta_min_);

    }

 void HcalRecHitsAnalyzer::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & /* iRun*/, edm::EventSetup const & /* iSetup */)

{ 

    Char_t histo[200];

    ibooker.setCurrentFolder("HcalRecHitsD/HcalRecHitTask");

    // General counters (drawn)

    //Produce both a total per subdetector, and number of rechits per subdetector depth
    // The bins are 1 unit wide, and the range is determined by the number of channels per subdetector

    for(int depth = 0; depth <= maxDepthHB_; depth++){
      if(depth == 0){ sprintf  (histo, "N_HB" );}
      else{           sprintf  (histo, "N_HB_depth%d",depth );}
      int NBins = (int) (nChannels_[1] * 1.1);
      Nhb.push_back( ibooker.book1D(histo, histo, NBins, 0., (float)NBins) );
    } 
    for(int depth = 0; depth <= maxDepthHE_; depth++){
      if(depth == 0){ sprintf  (histo, "N_HE" );}
      else{           sprintf  (histo, "N_HE_depth%d",depth );}
      int NBins	= (int) (nChannels_[2] * 1.1);
      Nhe.push_back( ibooker.book1D(histo, histo, NBins,0., (float)NBins) );
    } 
    for(int depth = 0; depth <= maxDepthHO_; depth++){
      if(depth == 0){ sprintf  (histo, "N_HO" );}
      else{           sprintf  (histo, "N_HO_depth%d",depth );}
      int NBins	= (int) (nChannels_[3] * 1.1);
      Nho.push_back( ibooker.book1D(histo, histo, NBins,0., (float)NBins) );
    } 
    for(int depth = 0; depth <= maxDepthHF_; depth++){
      if(depth == 0){ sprintf  (histo, "N_HF" );}
      else{           sprintf  (histo, "N_HF_depth%d",depth );}
      int NBins	= (int) (nChannels_[4] * 1.1);
      Nhf.push_back( ibooker.book1D(histo, histo, NBins,0., (float)NBins) );
    } 

    // ZS
    if(subdet_ == 6) {

    }

    // ALL others, except ZS
    else {  
      for(int depth = 1; depth <= maxDepthAll_; depth++){
        sprintf  (histo, "emap_depth%d",depth );
        emap.push_back( ibooker.book2D(histo, histo, 84, -42., 42., 72, 0., 72.) );
      } 

      //The mean energy histos are drawn, but not the RMS or emean seq
      
      for (int depth = 1; depth <= maxDepthHB_; depth++) {
	sprintf  (histo, "emean_vs_ieta_HB%d",depth );
	emean_vs_ieta_HB.push_back( ibooker.bookProfile(histo, histo, ieta_bins_, ieta_min_, ieta_max_, -10., 2000., " ") );

	sprintf  (histo, "emean_vs_ieta_M0_HB%d",depth );
	emean_vs_ieta_HBM0.push_back( ibooker.bookProfile(histo, histo, ieta_bins_, ieta_min_, ieta_max_, -10., 2000., " ") );

	sprintf  (histo, "emean_vs_ieta_M3_HB%d",depth );
	emean_vs_ieta_HBM3.push_back( ibooker.bookProfile(histo, histo, ieta_bins_, ieta_min_, ieta_max_, -10., 2000., " ") );
      }
      for (int depth = 1; depth <= maxDepthHE_; depth++) {
	sprintf  (histo, "emean_vs_ieta_HE%d",depth );
	emean_vs_ieta_HE.push_back( ibooker.bookProfile(histo, histo, ieta_bins_, ieta_min_, ieta_max_, -10., 2000., " ") );

	sprintf  (histo, "emean_vs_ieta_M0_HE%d",depth );
	emean_vs_ieta_HEM0.push_back( ibooker.bookProfile(histo, histo, ieta_bins_, ieta_min_, ieta_max_, -10., 2000., " ") );

	sprintf  (histo, "emean_vs_ieta_M3_HE%d",depth );
	emean_vs_ieta_HEM3.push_back( ibooker.bookProfile(histo, histo, ieta_bins_, ieta_min_, ieta_max_, -10., 2000., " ") );
      }
      for (int depth = 1; depth <= maxDepthHF_; depth++) {
	sprintf  (histo, "emean_vs_ieta_HF%d",depth );
	emean_vs_ieta_HF.push_back( ibooker.bookProfile(histo, histo, ieta_bins_, ieta_min_, ieta_max_, -10., 2000., " ") );
      }
      sprintf  (histo, "emean_vs_ieta_HO" );
      emean_vs_ieta_HO = ibooker.bookProfile(histo, histo, ieta_bins_, ieta_min_, ieta_max_, -10., 2000., " " );

      //The only occupancy histos drawn are occupancy vs. ieta
      //but the maps are needed because this is where the latter are filled from

      for (int depth = 1; depth <= maxDepthHB_; depth++) {
         sprintf  (histo, "occupancy_map_HB%d",depth );
         occupancy_map_HB.push_back( ibooker.book2D(histo, histo, ieta_bins_, ieta_min_, ieta_max_, iphi_bins_, iphi_min_, iphi_max_) );
      }

      for (int depth = 1; depth <= maxDepthHE_; depth++) {
         sprintf  (histo, "occupancy_map_HE%d",depth );
         occupancy_map_HE.push_back( ibooker.book2D(histo, histo, ieta_bins_, ieta_min_, ieta_max_, iphi_bins_, iphi_min_, iphi_max_) );
      }

      sprintf  (histo, "occupancy_map_HO" );
      occupancy_map_HO = ibooker.book2D(histo, histo, ieta_bins_, ieta_min_, ieta_max_, iphi_bins_, iphi_min_, iphi_max_);      

      for (int depth = 1; depth <= maxDepthHF_; depth++) {
         sprintf  (histo, "occupancy_map_HF%d",depth );
         occupancy_map_HF.push_back( ibooker.book2D(histo, histo, ieta_bins_, ieta_min_, ieta_max_, iphi_bins_, iphi_min_, iphi_max_) );
      }

      //These are drawn

      for (int depth = 1; depth <= maxDepthHB_; depth++) {
         sprintf  (histo, "occupancy_vs_ieta_HB%d",depth );
         occupancy_vs_ieta_HB.push_back( ibooker.book1D(histo, histo, ieta_bins_, ieta_min_, ieta_max_) );
      }

      for (int depth = 1; depth <= maxDepthHE_; depth++) {
         sprintf  (histo, "occupancy_vs_ieta_HE%d",depth );
         occupancy_vs_ieta_HE.push_back( ibooker.book1D(histo, histo, ieta_bins_, ieta_min_, ieta_max_) );
      }

      sprintf  (histo, "occupancy_vs_ieta_HO" );
      occupancy_vs_ieta_HO = ibooker.book1D(histo, histo, ieta_bins_, ieta_min_, ieta_max_);

      for (int depth = 1; depth <= maxDepthHF_; depth++) {
         sprintf  (histo, "occupancy_vs_ieta_HF%d",depth );
         occupancy_vs_ieta_HF.push_back( ibooker.book1D(histo, histo, ieta_bins_, ieta_min_, ieta_max_) );
      }


      //All status word histos except HF67 are drawn
      sprintf (histo, "HcalRecHitTask_RecHit_StatusWord_HB" ) ;
      RecHit_StatusWord_HB = ibooker.book1D(histo, histo, 32 , -0.5, 31.5); 
      
      sprintf (histo, "HcalRecHitTask_RecHit_StatusWord_HE" ) ;
      RecHit_StatusWord_HE = ibooker.book1D(histo, histo, 32 , -0.5, 31.5); 

      sprintf (histo, "HcalRecHitTask_RecHit_StatusWord_HF" ) ;
      RecHit_StatusWord_HF = ibooker.book1D(histo, histo, 32 , -0.5, 31.5); 

      sprintf (histo, "HcalRecHitTask_RecHit_StatusWord_HO" ) ;
      RecHit_StatusWord_HO = ibooker.book1D(histo, histo, 32 , -0.5, 31.5); 

      //Aux status word histos
      sprintf (histo, "HcalRecHitTask_RecHit_Aux_StatusWord_HB" ) ;
      RecHit_Aux_StatusWord_HB = ibooker.book1D(histo, histo, 32 , -0.5, 31.5); 
      
      sprintf (histo, "HcalRecHitTask_RecHit_Aux_StatusWord_HE" ) ;
      RecHit_Aux_StatusWord_HE = ibooker.book1D(histo, histo, 32 , -0.5, 31.5); 

      sprintf (histo, "HcalRecHitTask_RecHit_Aux_StatusWord_HF" ) ;
      RecHit_Aux_StatusWord_HF = ibooker.book1D(histo, histo, 32 , -0.5, 31.5); 

      sprintf (histo, "HcalRecHitTask_RecHit_Aux_StatusWord_HO" ) ;
      RecHit_Aux_StatusWord_HO = ibooker.book1D(histo, histo, 32 , -0.5, 31.5); 

    }  // end-of (subdet_ =! 6)

      //Status word correlations
      sprintf (histo, "HcalRecHitTask_RecHit_StatusWordCorr_HB");
      RecHit_StatusWordCorr_HB = ibooker.book2D(histo, histo, 2, -0.5, 1.5, 2, -0.5, 1.5);

      sprintf (histo, "HcalRecHitTask_RecHit_StatusWordCorr_HE");
      RecHit_StatusWordCorr_HE = ibooker.book2D(histo, histo, 2, -0.5, 1.5, 2, -0.5, 1.5);


    //======================= Now various cases one by one ===================

    //Histograms drawn for single pion scan
    if(subdet_ != 0 && imc != 0) { // just not for noise  
      sprintf (histo, "HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths");
      meEnConeEtaProfile = ibooker.bookProfile(histo, histo, ieta_bins_, ieta_min_, ieta_max_, -100., 2000., " ");  
      
      sprintf (histo, "HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_E");
      meEnConeEtaProfile_E = ibooker.bookProfile(histo, histo, ieta_bins_, ieta_min_, ieta_max_, -100., 2000., " ");  
      
      sprintf (histo, "HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_EH");
      meEnConeEtaProfile_EH = ibooker.bookProfile(histo, histo, ieta_bins_, ieta_min_, ieta_max_, -100., 2000., " ");  
    }

    // ************** HB **********************************
    if (subdet_ == 1 || subdet_ == 5 ){

      //Only severity level, energy of rechits and overall HB timing histos are drawn  

      sprintf(histo, "HcalRecHitTask_severityLevel_HB");
      sevLvl_HB = ibooker.book1D(histo, histo, 25, -0.5, 24.5); 

      sprintf (histo, "HcalRecHitTask_energy_of_rechits_HB" ) ;
      meRecHitsEnergyHB = ibooker.book1D(histo, histo, 2010 , -10. , 2000.); 
      
      sprintf (histo, "HcalRecHitTask_energy_of_rechits_M0_HB" ) ;
      meRecHitsEnergyHBM0 = ibooker.book1D(histo, histo, 2010 , -10. , 2000.); 
      
      sprintf (histo, "HcalRecHitTask_energy_of_rechits_M3_HB" ) ;
      meRecHitsEnergyHBM3 = ibooker.book1D(histo, histo, 2010 , -10. , 2000.); 
      
      sprintf (histo, "HcalRecHitTask_energy_of_rechits_M2vM0_HB" ) ;
      meRecHitsEnergyM2vM0HB = ibooker.book2D(histo, histo, 42 , -10. , 200., 42, -10., 200.); 
      
      sprintf (histo, "HcalRecHitTask_energy_of_rechits_M3vM0_HB" ) ;
      meRecHitsEnergyM3vM0HB = ibooker.book2D(histo, histo, 42 , -10. , 200., 42, -10., 200.); 
      
      sprintf (histo, "HcalRecHitTask_energy_of_rechits_M3vM2_HB" ) ;
      meRecHitsEnergyM3vM2HB = ibooker.book2D(histo, histo, 42 , -10. , 200., 42, -10., 200.); 
      
      sprintf (histo, "HcalRecHitTask_M2Log10Chi2_of_rechits_HB" ) ;
      meRecHitsM2Chi2HB = ibooker.book1D(histo, histo, 120 , -2. , 10.); 
      
      sprintf (histo, "HcalRecHitTask_timing_HB" ) ;
      meTimeHB = ibooker.book1D(histo, histo, 70, -48., 92.); 

      //High, medium and low histograms to reduce RAM usage
      sprintf (histo, "HcalRecHitTask_timing_vs_energy_Low_HB" ) ;
      meTE_Low_HB = ibooker.book2D(histo, histo, 50, -5., 45.,  70, -48., 92.);

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_HB" ) ;
      meTE_HB = ibooker.book2D(histo, histo, 150, -5., 295.,  70, -48., 92.);

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_High_HB" ) ;
      meTE_High_HB = ibooker.book2D(histo, histo, 150, -5., 2995.,  70, -48., 92.);
      
      sprintf (histo, "HcalRecHitTask_timing_vs_energy_profile_Low_HB" ) ;
      meTEprofileHB_Low = ibooker.bookProfile(histo, histo, 50, -5., 45., -48., 92., " "); 

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_profile_HB" ) ;
      meTEprofileHB = ibooker.bookProfile(histo, histo, 150, -5., 295., -48., 92., " "); 

      sprintf (histo, "HcalRecHitTask_Log10Chi2_vs_energy_profile_HB" ) ;
      meLog10Chi2profileHB = ibooker.bookProfile(histo, histo, 150, -5., 295., -2., 10., " "); 

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_profile_High_HB" ) ;
      meTEprofileHB_High = ibooker.bookProfile(histo, histo, 150, -5., 2995., -48., 92., " "); 

    }
    
    // ********************** HE ************************************
    if ( subdet_ == 2 || subdet_ == 5 ){


      //Only severity level, energy of rechits and overall HB timing histos are drawn  
      sprintf(histo, "HcalRecHitTask_severityLevel_HE");
      sevLvl_HE = ibooker.book1D(histo, histo, 25, -0.5, 24.5); 
      
      sprintf (histo, "HcalRecHitTask_energy_of_rechits_HE" ) ;
      meRecHitsEnergyHE = ibooker.book1D(histo, histo, 2010, -10., 2000.);
      
      sprintf (histo, "HcalRecHitTask_energy_of_rechits_M0_HE" ) ;
      meRecHitsEnergyHEM0 = ibooker.book1D(histo, histo, 2010, -10., 2000.);
      
      sprintf (histo, "HcalRecHitTask_energy_of_rechits_M3_HE" ) ;
      meRecHitsEnergyHEM3 = ibooker.book1D(histo, histo, 2010, -10., 2000.);
      
      sprintf (histo, "HcalRecHitTask_energy_of_rechits_M2vM0_HE" ) ;
      meRecHitsEnergyM2vM0HE = ibooker.book2D(histo, histo, 42 , -10. , 200., 42, -10., 200.); 
      
      sprintf (histo, "HcalRecHitTask_energy_of_rechits_M3vM0_HE" ) ;
      meRecHitsEnergyM3vM0HE = ibooker.book2D(histo, histo, 42 , -10. , 200., 42, -10., 200.); 
      
      sprintf (histo, "HcalRecHitTask_energy_of_rechits_M2vM0_HE" ) ;
      meRecHitsEnergyM3vM2HE = ibooker.book2D(histo, histo, 42 , -10. , 200., 42, -10., 200.); 
      
      sprintf (histo, "HcalRecHitTask_M2Log10Chi2_of_rechits_HE" ) ;
      meRecHitsM2Chi2HE = ibooker.book1D(histo, histo, 120 , -2. , 10.); 
      
      sprintf (histo, "HcalRecHitTask_timing_HE" ) ;
      meTimeHE = ibooker.book1D(histo, histo, 70, -48., 92.); 
      
      sprintf (histo, "HcalRecHitTask_timing_vs_energy_Low_HE" ) ;
      meTE_Low_HE = ibooker.book2D(histo, histo, 80, -5., 75.,  70, -48., 92.);

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_HE" ) ;
      meTE_HE = ibooker.book2D(histo, histo, 200, -5., 395.,  70, -48., 92.);
      
      sprintf (histo, "HcalRecHitTask_timing_vs_energy_profile_Low_HE" ) ;
      meTEprofileHE_Low = ibooker.bookProfile(histo, histo, 80, -5., 75., -48., 92., " "); 

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_profile_HE" ) ;
      meTEprofileHE = ibooker.bookProfile(histo, histo, 200, -5., 395., -48., 92., " "); 
      
      sprintf (histo, "HcalRecHitTask_Log10Chi2_vs_energy_profile_HE" ) ;
      meLog10Chi2profileHE = ibooker.bookProfile(histo, histo, 200, -5., 395., -2., 10., " "); 
      

    }

    // ************** HO ****************************************
    if ( subdet_ == 3 || subdet_ == 5  ){
      
      //Only severity level, energy of rechits and overall HB timing histos are drawn  

      sprintf(histo, "HcalRecHitTask_severityLevel_HO");
      sevLvl_HO = ibooker.book1D(histo, histo, 25, -0.5, 24.5); 

      sprintf (histo, "HcalRecHitTask_energy_of_rechits_HO" ) ;
      meRecHitsEnergyHO = ibooker.book1D(histo, histo, 2010 , -10. , 2000.);
      
      sprintf (histo, "HcalRecHitTask_timing_HO" ) ;
      meTimeHO = ibooker.book1D(histo, histo, 70, -48., 92.); 
      
      sprintf (histo, "HcalRecHitTask_timing_vs_energy_HO" ) ;
      meTE_HO= ibooker.book2D(histo, histo, 60, -5., 55., 70, -48., 92.);

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_High_HO" ) ;
      meTE_High_HO= ibooker.book2D(histo, histo, 100, -5., 995., 70, -48., 92.);
      
      sprintf (histo, "HcalRecHitTask_timing_vs_energy_profile_HO" ) ;
      meTEprofileHO = ibooker.bookProfile(histo, histo, 60, -5., 55., -48., 92., " "); 

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_profile_High_HO" ) ;
      meTEprofileHO_High = ibooker.bookProfile(histo, histo, 100, -5., 995., -48., 92., " "); 
      
    }   
  
    // ********************** HF ************************************
    if ( subdet_ == 4 || subdet_ == 5 ){

      //Only severity level, energy of rechits and overall HB timing histos are drawn  
      
      sprintf(histo, "HcalRecHitTask_severityLevel_HF");
      sevLvl_HF = ibooker.book1D(histo, histo, 25, -0.5, 24.5); 

      sprintf (histo, "HcalRecHitTask_energy_of_rechits_HF" ) ;
      meRecHitsEnergyHF = ibooker.book1D(histo, histo, 2010 , -10. , 2000.); 

      sprintf (histo, "HcalRecHitTask_timing_HF" ) ;
      meTimeHF = ibooker.book1D(histo, histo, 70, -48., 92.); 
      
      sprintf (histo, "HcalRecHitTask_timing_vs_energy_Low_HF" ) ;
      meTE_Low_HF = ibooker.book2D(histo, histo, 100, -5., 195., 70, -48., 92.);

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_HF" ) ;
      meTE_HF = ibooker.book2D(histo, histo, 200, -5., 995., 70, -48., 92.);
      
      sprintf (histo, "HcalRecHitTask_timing_vs_energy_profile_Low_HF" ) ;
      meTEprofileHF_Low = ibooker.bookProfile(histo, histo, 100, -5., 195., -48., 92., " "); 

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_profile_HF" ) ;
      meTEprofileHF = ibooker.bookProfile(histo, histo, 200, -5., 995., -48., 92., " "); 

    }

}


void HcalRecHitsAnalyzer::analyze(edm::Event const& ev, edm::EventSetup const& c) {

  using namespace edm;

  // cuts for each subdet_ector mimiking  "Scheme B"
  //  double cutHB = 0.9, cutHE = 1.4, cutHO = 1.1, cutHFL = 1.2, cutHFS = 1.8; 

  // energy in HCAL
  double eHcal        = 0.;
  // Total numbet of RecHits in HCAL, in the cone, above 1 GeV theshold
  int nrechits       = 0;
  int nrechitsThresh = 0;

  // energy in ECAL
  double eEcal       = 0.;
  double eEcalB      = 0.;
  double eEcalE      = 0.;
  double eEcalCone   = 0.;

  // HCAL energy around MC eta-phi at all depths;
  double partR = 0.3;

  // Single particle samples: actual eta-phi position of cluster around
  // hottest cell
  double etaHot  = 99999.; 
  double phiHot  = 99999.; 

  //   previously was:  c.get<IdealGeometryRecord>().get (geometry);
  c.get<CaloGeometryRecord>().get (geometry);

  // HCAL Topology **************************************************
  edm::ESHandle<HcalTopology> topo;
  c.get<HcalRecNumberingRecord>().get(topo);
  theHcalTopology = topo.product();

  // HCAL channel status map ****************************************
  edm::ESHandle<HcalChannelQuality> hcalChStatus;
  c.get<HcalChannelQualityRcd>().get( "withTopo", hcalChStatus );
  theHcalChStatus = hcalChStatus.product();

  // Assignment of severity levels **********************************
  edm::ESHandle<HcalSeverityLevelComputer> hcalSevLvlComputerHndl;
  c.get<HcalSeverityLevelComputerRcd>().get(hcalSevLvlComputerHndl);
  theHcalSevLvlComputer = hcalSevLvlComputerHndl.product(); 

  // Fill working vectors of HCAL RecHits quantities (all of these are drawn)
  fillRecHitsTmp(subdet_, ev); 

  // HB   
  if( subdet_ ==5 || subdet_ == 1 ){ 
     for(unsigned int iv=0; iv<hcalHBSevLvlVec.size(); iv++){
        sevLvl_HB->Fill(hcalHBSevLvlVec[iv]);
     }    
  }
  // HE   
  if( subdet_ ==5 || subdet_ == 2 ){
     for(unsigned int iv=0; iv<hcalHESevLvlVec.size(); iv++){
        sevLvl_HE->Fill(hcalHESevLvlVec[iv]);
     }
  }
  // HO 
  if( subdet_ ==5 || subdet_ == 3 ){
     for(unsigned int iv=0; iv<hcalHOSevLvlVec.size(); iv++){
        sevLvl_HO->Fill(hcalHOSevLvlVec[iv]);
     }
  }
  // HF 
  if( subdet_ ==5 || subdet_ == 4 ){
     for(unsigned int iv=0; iv<hcalHFSevLvlVec.size(); iv++){
        sevLvl_HF->Fill(hcalHFSevLvlVec[iv]);
     }
  } 

  //===========================================================================
  // IN ALL other CASES : ieta-iphi maps 
  //===========================================================================

  // ECAL 
  if(ecalselector_ == "yes" && (subdet_ == 1 || subdet_ == 2 || subdet_ == 5)) {
    Handle<EBRecHitCollection> rhitEB;


      ev.getByToken(tok_EB_, rhitEB);

    EcalRecHitCollection::const_iterator RecHit = rhitEB.product()->begin();  
    EcalRecHitCollection::const_iterator RecHitEnd = rhitEB.product()->end();  
    
    for (; RecHit != RecHitEnd ; ++RecHit) {
       
      double en  = RecHit->energy();
      eEcal  += en;
      eEcalB += en;


    }

    
    Handle<EERecHitCollection> rhitEE;
 
      ev.getByToken(tok_EE_, rhitEE);

    RecHit = rhitEE.product()->begin();  
    RecHitEnd = rhitEE.product()->end();  
    
    for (; RecHit != RecHitEnd ; ++RecHit) {
      
      double en   = RecHit->energy();
      eEcal  += en;
      eEcalE += en;


    }
  }     // end of ECAL selection 

  // Counting, including ZS items
  // Filling HCAL maps  ----------------------------------------------------
  //   double maxE = -99999.;
  
  // element 0: any depth. element 1,2,..: depth 1,2
  std::vector<int> nhb_v(maxDepthHB_+1,0);
  std::vector<int> nhe_v(maxDepthHE_+1,0);
  std::vector<int> nho_v(maxDepthHO_+1,0);
  std::vector<int> nhf_v(maxDepthHF_+1,0);

  for (unsigned int i = 0; i < cen.size(); i++) {
    
    int sub       = csub[i];
    int depth     = cdepth[i];
    int ieta      = cieta[i]; 
    int iphi      = ciphi[i]; 
    double en     = cen[i]; 
    double enM0   = cenM0[i];
    double enM3   = cenM3[i];
    //     double eta    = ceta[i]; 
    //     double phi    = cphi[i]; 
    uint32_t stwd = cstwd[i];
    uint32_t auxstwd = cauxstwd[i];
    //    double z   = cz[i];

    //Make sure that an invalid depth won't cause an error. We should probably report the problem as well.
    if( depth < 1 ) continue;
    if( sub == 1 && depth > maxDepthHB_ ) continue;
    if( sub == 2 && depth > maxDepthHE_ ) continue;
    if( sub == 3 && depth > maxDepthHO_ ) continue;
    if( sub == 4 && depth > maxDepthHF_ ) continue;

    if( sub ==1 ){ nhb_v[depth]++; nhb_v[0]++;} // element 0: any depth, element 1,2,..: depth 1,2,...
    if( sub ==2 ){ nhe_v[depth]++; nhe_v[0]++;} //
    if( sub ==3 ){ nho_v[depth]++; nho_v[0]++;} //
    if( sub ==4 ){ nhf_v[depth]++; nhf_v[0]++;} //

    if( subdet_ == 6) {                                    // ZS specific
    }

    if( subdet_ != 6) {  
      int ieta2 = ieta;
      int depth2 = depth;
      if(sub == 4){
	if (ieta2 < 0) ieta2--;
        else ieta2++;
      }
      if(sub == 3) depth2 = maxDepthAll_ - maxDepthHO_ + depth; //This will use the last depths for HO	
      emap[depth2-1]->Fill(double(ieta2),double(iphi),en);

      // to distinguish HE and HF
      if( depth == 1 || depth == 2 ) {
        int ieta1 =  ieta;
	if(sub == 4) { 
	  if (ieta1 < 0) ieta1--;
          else  ieta1++;   
	}
      }

      if ( sub == 1){
	 emean_vs_ieta_HB[depth-1]->Fill(double(ieta), en);
	 emean_vs_ieta_HBM0[depth-1]->Fill(double(ieta), enM0);
	 emean_vs_ieta_HBM3[depth-1]->Fill(double(ieta), enM3);
	 occupancy_map_HB[depth-1]->Fill(double(ieta),double(iphi));
      }
      if ( sub == 2){
	 emean_vs_ieta_HE[depth-1]->Fill(double(ieta), en);
	 emean_vs_ieta_HEM0[depth-1]->Fill(double(ieta), enM0);
	 emean_vs_ieta_HEM3[depth-1]->Fill(double(ieta), enM3);
	 occupancy_map_HE[depth-1]->Fill(double(ieta),double(iphi));
      }
      if ( sub == 3){
	 emean_vs_ieta_HO->Fill(double(ieta), en);
	 occupancy_map_HO->Fill(double(ieta),double(iphi));
      }
      if ( sub == 4){
	 emean_vs_ieta_HF[depth-1]->Fill(double(ieta), en);
	 occupancy_map_HF[depth-1]->Fill(double(ieta),double(iphi));
      }
    }


    
    //32-bit status word  
    uint32_t statadd;
    unsigned int isw67 = 0;

    //Statusword correlation
    unsigned int sw27 = 27;
    unsigned int sw13 = 13;

    uint32_t statadd27 = 0x1<<sw27;
    uint32_t statadd13 = 0x1<<sw13;

    float status27 = 0;
    float status13 = 0;

    if(stwd & statadd27) status27 = 1;
    if(stwd & statadd13) status13 = 1;

    if        (sub == 1){
      RecHit_StatusWordCorr_HB->Fill(status13, status27);
    } else if (sub == 2){
      RecHit_StatusWordCorr_HE->Fill(status13, status27);
    }


    for (unsigned int isw = 0; isw < 32; isw++){
      statadd = 0x1<<(isw);
      if (stwd & statadd){
	if      (sub == 1) RecHit_StatusWord_HB->Fill(isw);
	else if (sub == 2) RecHit_StatusWord_HE->Fill(isw);
	else if (sub == 3) RecHit_StatusWord_HO->Fill(isw);
	else if (sub == 4){
	  RecHit_StatusWord_HF->Fill(isw);
	  if (isw == 6) isw67 += 1;
	  if (isw == 7) isw67 += 2;
	}
      }
    }

    for (unsigned int isw =0; isw < 32; isw++){
      statadd = 0x1<<(isw);
      if( auxstwd & statadd ){
        if      (sub == 1) RecHit_Aux_StatusWord_HB->Fill(isw);
        else if (sub == 2) RecHit_Aux_StatusWord_HE->Fill(isw);
        else if (sub == 3) RecHit_Aux_StatusWord_HO->Fill(isw);
        else if (sub == 4) RecHit_Aux_StatusWord_HF->Fill(isw);
      }

    }

  } 

    for(int depth = 0; depth <= maxDepthHB_; depth++) Nhb[depth]->Fill(double(nhb_v[depth]));
    for(int depth = 0; depth <= maxDepthHE_; depth++) Nhe[depth]->Fill(double(nhe_v[depth]));
    for(int depth = 0; depth <= maxDepthHO_; depth++) Nho[depth]->Fill(double(nho_v[depth]));
    for(int depth = 0; depth <= maxDepthHF_; depth++) Nhf[depth]->Fill(double(nhf_v[depth]));

  //===========================================================================
  // SUBSYSTEMS,  
  //===========================================================================
  
  if ((subdet_ != 6) && (subdet_ != 0)) {

    double clusEta = 999.;
    double clusPhi = 999.; 
    double clusEn  = 0.;
    
    double HcalCone    = 0.;

    int ietaMax   =  9999;
    //     double enMax1 = -9999.;
    //     double enMax2 = -9999.;
    //     double enMax3 = -9999.;
    //     double enMax4 = -9999.;
    //     double enMax  = -9999.;
    //     double etaMax =  9999.;

    //   CYCLE over cells ====================================================

    for (unsigned int i = 0; i < cen.size(); i++) {
      int sub    = csub[i];
      double eta = ceta[i]; 
      double phi = cphi[i]; 
      double en  = cen[i]; 
      double enM0  = cenM0[i]; 
      double enM3  = cenM3[i]; 
      double chi2  = cchi2[i];
      double t   = ctime[i];
//       int   ieta = cieta[i];

      double rhot = dR(etaHot, phiHot, eta, phi); 
      if(rhot < partR && en > 1.) { 
	clusEta = (clusEta * clusEn + eta * en)/(clusEn + en);
    	clusPhi = phi12(clusPhi, clusEn, phi, en); 
        clusEn += en;
      }

      nrechits++;	    
      eHcal += en;

      if(en > 1. ) nrechitsThresh++;
      
      //The energy and overall timing histos are drawn while
      //the ones split by depth are not
      if(sub == 1 && (subdet_ == 1 || subdet_ == 5)) {  
	meTimeHB->Fill(t);
	meRecHitsEnergyHB->Fill(en);
	meRecHitsEnergyHBM0->Fill(enM0);
	meRecHitsEnergyHBM3->Fill(enM3);

        meRecHitsEnergyM2vM0HB->Fill(enM0,en);
        meRecHitsEnergyM3vM0HB->Fill(enM0,enM3);
        meRecHitsEnergyM3vM2HB->Fill(en,enM3);

        meRecHitsM2Chi2HB->Fill(log10(chi2));
        meLog10Chi2profileHB->Fill(en,log10(chi2));
	
	meTE_Low_HB->Fill( en, t);
	meTE_HB->Fill( en, t);
	meTE_High_HB->Fill( en, t);
	meTEprofileHB_Low->Fill(en, t);
	meTEprofileHB->Fill(en, t);
	meTEprofileHB_High->Fill(en, t);
      }     
      if(sub == 2 && (subdet_ == 2 || subdet_ == 5)) {  
	meTimeHE->Fill(t);
	meRecHitsEnergyHE->Fill(en);
	meRecHitsEnergyHEM0->Fill(enM0);
	meRecHitsEnergyHEM3->Fill(enM3);

        meRecHitsEnergyM2vM0HE->Fill(enM0,en);
        meRecHitsEnergyM3vM0HE->Fill(enM0,enM3);
        meRecHitsEnergyM3vM2HE->Fill(en,enM3);

        meRecHitsM2Chi2HE->Fill(log10(chi2));
        meLog10Chi2profileHE->Fill(en,log10(chi2));	

	meTE_Low_HE->Fill( en, t);
	meTE_HE->Fill( en, t);
	meTEprofileHE_Low->Fill(en, t);
	meTEprofileHE->Fill(en, t);
      }
      if(sub == 4 && (subdet_ == 4 || subdet_ == 5)) {  
	meTimeHF->Fill(t);
	meRecHitsEnergyHF->Fill(en);	  

	meTE_Low_HF->Fill(en, t);
	meTE_HF->Fill(en, t);
	meTEprofileHF_Low->Fill(en, t);
	meTEprofileHF->Fill(en, t);

      }
      if(sub == 3 && (subdet_ == 3 || subdet_ == 5)) {  
	meTimeHO->Fill(t);
	meRecHitsEnergyHO->Fill(en);

	meTE_HO->Fill( en, t);
	meTE_High_HO->Fill( en, t);
	meTEprofileHO->Fill(en, t);
	meTEprofileHO_High->Fill(en, t);
      }
    }

    if(imc != 0) {
      //Cone by depth are not drawn, the others are used for pion scan
      meEnConeEtaProfile       ->Fill(double(ietaMax),  HcalCone);   // 
      meEnConeEtaProfile_E     ->Fill(double(ietaMax), eEcalCone);   
      meEnConeEtaProfile_EH    ->Fill(double(ietaMax),  HcalCone+eEcalCone); 
    }

    // Single particle samples ONLY !  ======================================
    // Fill up some histos for "integrated" subsustems. 
    // These are not drawn
  }

  nevtot++;
}


///////////////////////////////////////////////////////////////////////////////
void HcalRecHitsAnalyzer::fillRecHitsTmp(int subdet_, edm::Event const& ev){
  
  using namespace edm;
  
  
  // initialize data vectors
  csub.clear();
  cen.clear();
  cenM0.clear();
  cenM3.clear();
  cchi2.clear();
  ceta.clear();
  cphi.clear();
  ctime.clear();
  cieta.clear();
  ciphi.clear();
  cdepth.clear();
  cz.clear();
  cstwd.clear();
  cauxstwd.clear();
  hcalHBSevLvlVec.clear();
  hcalHESevLvlVec.clear();
  hcalHFSevLvlVec.clear();
  hcalHOSevLvlVec.clear(); 

  if( subdet_ == 1 || subdet_ == 2  || subdet_ == 5 || subdet_ == 6 || subdet_ == 0) {
    
    //HBHE
    edm::Handle<HBHERecHitCollection> hbhecoll;
    ev.getByToken(tok_hbhe_, hbhecoll);
    
    for (HBHERecHitCollection::const_iterator j=hbhecoll->begin(); j != hbhecoll->end(); j++) {
      HcalDetId cell(j->id());
      const HcalGeometry* cellGeometry = 
	(HcalGeometry*)(geometry->getSubdetectorGeometry(cell));
      double eta  = cellGeometry->getPosition(cell).eta () ;
      double phi  = cellGeometry->getPosition(cell).phi () ;
      double zc   = cellGeometry->getPosition(cell).z ();
      int sub     = cell.subdet();
      int depth   = cell.depth();
      int inteta  = cell.ieta();
      int intphi  = cell.iphi();
      double en   = j->energy();
      double enM0 = j->eraw();
      double enM3 = j->eaux();
      double chi2 = j->chi2();
      double t    = j->time();
      int stwd    = j->flags();
      int auxstwd = j->aux();
      
      int severityLevel = hcalSevLvl( (CaloRecHit*) &*j );
      if( cell.subdet()==HcalBarrel ){
         hcalHBSevLvlVec.push_back(severityLevel);
      }else if (cell.subdet()==HcalEndcap ){
         hcalHESevLvlVec.push_back(severityLevel);
      } 
      
      if((iz > 0 && eta > 0.) || (iz < 0 && eta <0.) || iz == 0) { 
	
	csub.push_back(sub);
	cen.push_back(en);
	cenM0.push_back(enM0);
	cenM3.push_back(enM3);
        cchi2.push_back(chi2);
	ceta.push_back(eta);
	cphi.push_back(phi);
	ctime.push_back(t);
	cieta.push_back(inteta);
	ciphi.push_back(intphi);
	cdepth.push_back(depth);
	cz.push_back(zc);
	cstwd.push_back(stwd);
        cauxstwd.push_back(auxstwd);
      }
    }
    
  }

  if( subdet_ == 4 || subdet_ == 5 || subdet_ == 6 || subdet_ == 0) {

    //HF
    edm::Handle<HFRecHitCollection> hfcoll;
    ev.getByToken(tok_hf_, hfcoll);

    for (HFRecHitCollection::const_iterator j = hfcoll->begin(); j != hfcoll->end(); j++) {
      HcalDetId cell(j->id());
      const CaloCellGeometry* cellGeometry =
	geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
      double eta   = cellGeometry->getPosition().eta () ;
      double phi   = cellGeometry->getPosition().phi () ;
      double zc     = cellGeometry->getPosition().z ();
      int sub      = cell.subdet();
      int depth    = cell.depth();
      int inteta   = cell.ieta();
      int intphi   = cell.iphi();
      double en    = j->energy();
      double enM0  = 0.;
      double enM3  = 0.;
      double chi2  = 0.;
      double t     = j->time();
      int stwd     = j->flags();
      int auxstwd  = j->aux();

      int severityLevel = hcalSevLvl( (CaloRecHit*) &*j );
      if( cell.subdet()==HcalForward ){
         hcalHFSevLvlVec.push_back(severityLevel);
      } 

      if((iz > 0 && eta > 0.) || (iz < 0 && eta <0.) || iz == 0) { 
	
	csub.push_back(sub);
	cen.push_back(en);
	cenM0.push_back(enM0);
	cenM3.push_back(enM3);
        cchi2.push_back(chi2);
	ceta.push_back(eta);
	cphi.push_back(phi);
	ctime.push_back(t);
	cieta.push_back(inteta);
	ciphi.push_back(intphi);
	cdepth.push_back(depth);
	cz.push_back(zc);
	cstwd.push_back(stwd);
        cauxstwd.push_back(auxstwd);
      }
    }
  }

  //HO
  if( subdet_ == 3 || subdet_ == 5 || subdet_ == 6 || subdet_ == 0) {
  
    edm::Handle<HORecHitCollection> hocoll;
    ev.getByToken(tok_ho_, hocoll);
    
    for (HORecHitCollection::const_iterator j = hocoll->begin(); j != hocoll->end(); j++) {
      HcalDetId cell(j->id());
      const CaloCellGeometry* cellGeometry =
	geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
      double eta   = cellGeometry->getPosition().eta () ;
      double phi   = cellGeometry->getPosition().phi () ;
      double zc    = cellGeometry->getPosition().z ();
      int sub      = cell.subdet();
      int depth    = cell.depth();
      int inteta   = cell.ieta();
      int intphi   = cell.iphi();
      double t     = j->time();
      double en    = j->energy();
      double enM0  = 0.;
      double enM3  = 0.;
      double chi2 = 0.;
      int stwd     = j->flags();
      int auxstwd  = j->aux();

      int severityLevel = hcalSevLvl( (CaloRecHit*) &*j );
      if( cell.subdet()==HcalOuter ){
         hcalHOSevLvlVec.push_back(severityLevel);
      } 
      
      if((iz > 0 && eta > 0.) || (iz < 0 && eta <0.) || iz == 0) { 
	csub.push_back(sub);
	cen.push_back(en);
	cenM0.push_back(enM0);
	cenM3.push_back(enM3);
        cchi2.push_back(chi2);
	ceta.push_back(eta);
	cphi.push_back(phi);
	ctime.push_back(t);
	cieta.push_back(inteta);
	ciphi.push_back(intphi);
	cdepth.push_back(depth);
	cz.push_back(zc);
	cstwd.push_back(stwd);
        cauxstwd.push_back(auxstwd);
      }
    }
  }
}

double HcalRecHitsAnalyzer::dR(double eta1, double phi1, double eta2, double phi2) { 
  double PI = 3.1415926535898;
  double deltaphi= phi1 - phi2;
  if( phi2 > phi1 ) { deltaphi= phi2 - phi1;}
  if(deltaphi > PI) { deltaphi = 2.*PI - deltaphi;}
  double deltaeta = eta2 - eta1;
  double tmp = sqrt(deltaeta* deltaeta + deltaphi*deltaphi);
  return tmp;
}

double HcalRecHitsAnalyzer::phi12(double phi1, double en1, double phi2, double en2) {
  // weighted mean value of phi1 and phi2
  
  double tmp;
  double PI = 3.1415926535898;
  double a1 = phi1; double a2  = phi2;

  if( a1 > 0.5*PI  && a2 < 0.) a2 += 2*PI; 
  if( a2 > 0.5*PI  && a1 < 0.) a1 += 2*PI; 
  tmp = (a1 * en1 + a2 * en2)/(en1 + en2);
  if(tmp > PI) tmp -= 2.*PI; 
 
  return tmp;

}

double HcalRecHitsAnalyzer::dPhiWsign(double phi1, double phi2) {
  // clockwise      phi2 w.r.t phi1 means "+" phi distance
  // anti-clockwise phi2 w.r.t phi1 means "-" phi distance 

  double PI = 3.1415926535898;
  double a1 = phi1; double a2  = phi2;
  double tmp =  a2 - a1;
  if( a1*a2 < 0.) {
    if(a1 > 0.5 * PI)  tmp += 2.*PI;
    if(a2 > 0.5 * PI)  tmp -= 2.*PI;
  }
  return tmp;

}

int HcalRecHitsAnalyzer::hcalSevLvl(const CaloRecHit* hit){

   HcalDetId id = hit->detid();
   if (theHcalTopology->withSpecialRBXHBHE() && id.subdet() == HcalEndcap) {
     id = theHcalTopology->idFront(id);
   }

   const uint32_t recHitFlag = hit->flags();
   const uint32_t dbStatusFlag = theHcalChStatus->getValues(id)->getValue();

   int severityLevel = theHcalSevLvlComputer->getSeverityLevel(id, recHitFlag, dbStatusFlag);

   return severityLevel;

} 

DEFINE_FWK_MODULE(HcalRecHitsAnalyzer);

