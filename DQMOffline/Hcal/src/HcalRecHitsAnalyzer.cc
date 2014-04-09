#include "DQMOffline/Hcal/interface/HcalRecHitsAnalyzer.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
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
  useAllHistos_ = conf.getUntrackedParameter<bool>("useAllHistos", false);

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
  
  void HcalRecHitsAnalyzer::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & /* iRun*/, edm::EventSetup const & /* iSetup */)

{ 

    Char_t histo[200];

    ibooker.setCurrentFolder("HcalRecHitsD/HcalRecHitTask");

    // General counters (drawn)
    sprintf  (histo, "N_HB" );
    Nhb = ibooker.book1D(histo, histo, 2600,0.,2600.);
    sprintf  (histo, "N_HE" );
    Nhe = ibooker.book1D(histo, histo, 2600,0.,2600.);
    sprintf  (histo, "N_HO" );
    Nho = ibooker.book1D(histo, histo, 2200,0.,2200.);
    sprintf  (histo, "N_HF" );
    Nhf = ibooker.book1D(histo, histo, 1800,0., 1800.);

    // ZS
    if(subdet_ == 6) {

      for (unsigned int i1 = 0;  i1 < 82; i1++) {
	for (unsigned int i2 = 0;  i2 < 72; i2++) {
	  for (unsigned int i3 = 0;  i3 < 4;  i3++) {
	    for (unsigned int i4 = 0;  i4 < 4;  i4++) {
	      emap_min [i1][i2][i3][i4] = 99999.;     
	    }
	  }
	}
      }

      //None of the ZS histos are drawn
      if (useAllHistos_){
	sprintf  (histo, "ZSmin_map_depth1" );
	map_depth1 = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
	sprintf  (histo, "ZSmin_map_depth2" );
	map_depth2 = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
	sprintf  (histo, "ZSmin_map_depth3" );
	map_depth3 = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
	sprintf  (histo, "ZSmin_map_depth4" );
	map_depth4 = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
	
      
	sprintf  (histo, "ZS_Nreco_HB1" );
	ZS_nHB1 = ibooker.book1D(histo, histo, 2500, 0., 2500.);
	sprintf  (histo, "ZS_Nreco_HB2" );
	ZS_nHB2 = ibooker.book1D(histo, histo,  500, 0.,  500.);
	sprintf  (histo, "ZS_Nreco_HE1" );
	ZS_nHE1 = ibooker.book1D(histo, histo, 2000, 0., 2000.);
	sprintf  (histo, "ZS_Nreco_HE2" );
	ZS_nHE2 = ibooker.book1D(histo, histo, 2000, 0., 2000.);
	sprintf  (histo, "ZS_Nreco_HE3" );
	ZS_nHE3 = ibooker.book1D(histo, histo,  500, 0.,  500.);
	sprintf  (histo, "ZS_Nreco_HO" );
	ZS_nHO  = ibooker.book1D(histo, histo, 2500, 0., 2500.);
	sprintf  (histo, "ZS_Nreco_HF1" );
	ZS_nHF1 = ibooker.book1D(histo, histo, 1000, 0., 1000.);
	sprintf  (histo, "ZS_Nreco_HF2" );
	ZS_nHF2 = ibooker.book1D(histo, histo, 1000, 0., 1000.);
      
	sprintf  (histo, "ZSmin_simple1D_HB1" );
	ZS_HB1 = ibooker.book1D(histo, histo,120, -2., 10.);
	sprintf  (histo, "ZSmin_simple1D_HB2" );
	ZS_HB2 = ibooker.book1D(histo, histo,120, -2., 10.);
	sprintf  (histo, "ZSmin_simple1D_HE1" );
	ZS_HE1 = ibooker.book1D(histo, histo,120, -2., 10.);
	sprintf  (histo, "ZSmin_simple1D_HE2" );
	ZS_HE2 = ibooker.book1D(histo, histo,120, -2., 10.);
	sprintf  (histo, "ZSmin_simple1D_HE3" );
	ZS_HE3 = ibooker.book1D(histo, histo,120, -2., 10.);
	sprintf  (histo, "ZSmin_simple1D_HO" );
	ZS_HO = ibooker.book1D(histo, histo,120, -2., 10.);
	sprintf  (histo, "ZSmin_simple1D_HF1" );
	ZS_HF1 = ibooker.book1D(histo, histo,200, -10., 10.);
	sprintf  (histo, "ZSmin_simple1D_HF2" );
	ZS_HF2 = ibooker.book1D(histo, histo,200, -10., 10.);
	
	sprintf  (histo, "ZSmin_sequential1D_HB1" );
	ZS_seqHB1 = ibooker.book1D(histo, histo,2400, -1200., 1200.);
	sprintf  (histo, "ZSmin_sequential1D_HB2" );
	ZS_seqHB2 = ibooker.book1D(histo, histo,2400, -1200., 1200.);
	sprintf  (histo, "ZSmin_sequential1D_HE1" );
	ZS_seqHE1 = ibooker.book1D(histo, histo,4400, -2200., 2200.);
	sprintf  (histo, "ZSmin_sequential1D_HE2" );
	ZS_seqHE2 = ibooker.book1D(histo, histo,4400, -2200., 2200.);
	sprintf  (histo, "ZSmin_sequential1D_HE3" );
	ZS_seqHE3 = ibooker.book1D(histo, histo,4400, -2200., 2200.);
	sprintf  (histo, "ZSmin_sequential1D_HO" );
	ZS_seqHO  = ibooker.book1D(histo, histo,2400, -1200., 1200.);
	sprintf  (histo, "ZSmin_sequential1D_HF1" );
	ZS_seqHF1 = ibooker.book1D(histo, histo,6000, -3000., 3000.);
	sprintf  (histo, "ZSmin_sequential1D_HF2" );
	ZS_seqHF2 = ibooker.book1D(histo, histo,6000, -3000., 3000.);
      }
    }

    // ALL others, except ZS
    else {
  
      sprintf  (histo, "emap_depth1" );
      emap_depth1 = ibooker.book2D(histo, histo, 84, -42., 42., 72, 0., 72.);
      sprintf  (histo, "emap_depth2" );
      emap_depth2 = ibooker.book2D(histo, histo, 84, -42., 42., 72, 0., 72.);
      sprintf  (histo, "emap_depth3" );
      emap_depth3 = ibooker.book2D(histo, histo, 84, -42., 42., 72, 0., 72.);
      sprintf  (histo, "emap_depth4" );
      emap_depth4 = ibooker.book2D(histo, histo, 84, -42., 42., 72, 0., 72.);
      
      if (useAllHistos_){
	
	if (ecalselector_ == "yes") {
	  sprintf  (histo, "map_ecal" );
	  map_ecal = ibooker.book2D(histo, histo, 70, -3.045, 3.045, 72, -3.1415926536, 3.1415926536);
	}
      }
      
      //The mean energy histos are drawn, but not the RMS or emean seq
      sprintf  (histo, "emean_vs_ieta_HB1" );
      emean_vs_ieta_HB1 = ibooker.bookProfile(histo, histo, 82, -41., 41., 2010, -10., 2000.);
      sprintf  (histo, "emean_vs_ieta_HB2" );
      emean_vs_ieta_HB2 = ibooker.bookProfile(histo, histo, 82, -41., 41., 2010, -10., 2000.);
      sprintf  (histo, "emean_vs_ieta_HE1" );
      emean_vs_ieta_HE1 = ibooker.bookProfile(histo, histo, 82, -41., 41., 2010, -10. ,2000.);
      sprintf  (histo, "emean_vs_ieta_HE2" );
      emean_vs_ieta_HE2 = ibooker.bookProfile(histo, histo, 82, -41., 41., 2010, -10., 2000.);
      sprintf  (histo, "emean_vs_ieta_HE3" );
      emean_vs_ieta_HE3 = ibooker.bookProfile(histo, histo, 82, -41., 41., 2010, -10., 2000. );
      sprintf  (histo, "emean_vs_ieta_HO" );
      emean_vs_ieta_HO = ibooker.bookProfile(histo, histo, 82, -41., 41., 2010, -10., 2000. );
      sprintf  (histo, "emean_vs_ieta_HF1" );
      emean_vs_ieta_HF1 = ibooker.bookProfile(histo, histo, 82, -41., 41., 2010, -10., 2000. );
      sprintf  (histo, "emean_vs_ieta_HF2" );
      emean_vs_ieta_HF2 = ibooker.bookProfile(histo, histo, 82, -41., 41., 2010, -10., 2000. );

      if (useAllHistos_){
	sprintf  (histo, "RMS_vs_ieta_HB1" );
	RMS_vs_ieta_HB1 = ibooker.book1D(histo, histo, 82, -41., 41.);
	sprintf  (histo, "RMS_vs_ieta_HB2" );
	RMS_vs_ieta_HB2 = ibooker.book1D(histo, histo, 82, -41., 41.);
	sprintf  (histo, "RMS_vs_ieta_HE1" );
	RMS_vs_ieta_HE1 = ibooker.book1D(histo, histo, 82, -41., 41.);
	sprintf  (histo, "RMS_vs_ieta_HE2" );
	RMS_vs_ieta_HE2 = ibooker.book1D(histo, histo, 82, -41., 41.);
	sprintf  (histo, "RMS_vs_ieta_HE3" );
	RMS_vs_ieta_HE3 = ibooker.book1D(histo, histo, 82, -41., 41.);
	sprintf  (histo, "RMS_vs_ieta_HO" );
	RMS_vs_ieta_HO = ibooker.book1D(histo, histo, 82, -41., 41.);
	sprintf  (histo, "RMS_vs_ieta_HF1" );
	RMS_vs_ieta_HF1 = ibooker.book1D(histo, histo, 82, -41., 41.);
	sprintf  (histo, "RMS_vs_ieta_HF2" );
	RMS_vs_ieta_HF2 = ibooker.book1D(histo, histo, 82, -41., 41.);
	
	// Sequential emean and RMS
	sprintf  (histo, "emean_seq_HB1" );
	emean_seqHB1 = ibooker.bookProfile(histo, histo, 2400, -1200., 1200.,  2010, -10., 2000. );
	sprintf  (histo, "emean_seq_HB2" );
	emean_seqHB2 = ibooker.bookProfile(histo, histo, 2400, -1200., 1200.,  2010, -10., 2000. );
	sprintf  (histo, "emean_seq_HE1" );
	emean_seqHE1 = ibooker.bookProfile(histo, histo, 4400, -2200., 2200.,  2010, -10., 2000. );
	sprintf  (histo, "emean_seq_HE2" );
	emean_seqHE2 = ibooker.bookProfile(histo, histo, 4400, -2200., 2200.,  2010, -10., 2000. );
	sprintf  (histo, "emean_seq_HE3" );
	emean_seqHE3 = ibooker.bookProfile(histo, histo, 4400, -2200., 2200.,  2010, -10., 2000. );
	sprintf  (histo, "emean_seq_HO" );
	emean_seqHO = ibooker.bookProfile(histo, histo,  2400, -1200., 1200.,  2010, -10., 2000. );
	sprintf  (histo, "emean_seq_HF1" );
	emean_seqHF1 = ibooker.bookProfile(histo, histo, 6000, -3000., 3000.,  2010, -10., 2000. );
	sprintf  (histo, "emean_seq_HF2" );
	emean_seqHF2 = ibooker.bookProfile(histo, histo, 6000, -3000., 3000.,  2010, -10., 2000. );
	
	sprintf  (histo, "RMS_seq_HB1" );
	RMS_seq_HB1 = ibooker.book1D(histo, histo, 2400, -1200., 1200.);
	sprintf  (histo, "RMS_seq_HB2" );
	RMS_seq_HB2 = ibooker.book1D(histo, histo, 2400, -1200., 1200.);
	sprintf  (histo, "RMS_seq_HE1" );
	RMS_seq_HE1 = ibooker.book1D(histo, histo, 4400, -2200., 2200.);
	sprintf  (histo, "RMS_seq_HE2" );
	RMS_seq_HE2 = ibooker.book1D(histo, histo, 4400, -2200., 2200.);
	sprintf  (histo, "RMS_seq_HE3" );
	RMS_seq_HE3 = ibooker.book1D(histo, histo, 4400, -2200., 2200.);
	sprintf  (histo, "RMS_seq_HO" );
	RMS_seq_HO = ibooker.book1D(histo, histo, 2400, -1200., 1200.);
	sprintf  (histo, "RMS_seq_HF1" );
	RMS_seq_HF1 = ibooker.book1D(histo, histo, 6000, -3000., 3000.);
	sprintf  (histo, "RMS_seq_HF2" );
	RMS_seq_HF2 = ibooker.book1D(histo, histo, 6000, -3000., 3000.);
      }
      // Occupancy
      //The only occupancy histos drawn are occupancy vs. ieta
      //but the maps are needed because this is where the latter are filled from
      sprintf  (histo, "occupancy_map_HB1" );
      occupancy_map_HB1 = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
      sprintf  (histo, "occupancy_map_HB2" );
      occupancy_map_HB2 = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
      sprintf  (histo, "occupancy_map_HE1" );
      occupancy_map_HE1 = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
      sprintf  (histo, "occupancy_map_HE2" );
      occupancy_map_HE2 = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);      
      sprintf  (histo, "occupancy_map_HE3" );
      occupancy_map_HE3 = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
      sprintf  (histo, "occupancy_map_HO" );
      occupancy_map_HO = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);      
      sprintf  (histo, "occupancy_map_HF1" );
      occupancy_map_HF1 = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
      sprintf  (histo, "occupancy_map_HF2" );
      occupancy_map_HF2 = ibooker.book2D(histo, histo, 82, -41., 41., 72, 0., 72.);
      
      //These are drawn
      sprintf  (histo, "occupancy_vs_ieta_HB1" );
      occupancy_vs_ieta_HB1 = ibooker.book1D(histo, histo, 82, -41., 41.);
      sprintf  (histo, "occupancy_vs_ieta_HB2" );
      occupancy_vs_ieta_HB2 = ibooker.book1D(histo, histo, 82, -41., 41.);
      sprintf  (histo, "occupancy_vs_ieta_HE1" );
      occupancy_vs_ieta_HE1 = ibooker.book1D(histo, histo, 82, -41., 41.);
      sprintf  (histo, "occupancy_vs_ieta_HE2" );
      occupancy_vs_ieta_HE2 = ibooker.book1D(histo, histo, 82, -41., 41.);
      sprintf  (histo, "occupancy_vs_ieta_HE3" );
      occupancy_vs_ieta_HE3 = ibooker.book1D(histo, histo, 82, -41., 41.);
      sprintf  (histo, "occupancy_vs_ieta_HO" );
      occupancy_vs_ieta_HO = ibooker.book1D(histo, histo, 82, -41., 41.);
      sprintf  (histo, "occupancy_vs_ieta_HF1" );
      occupancy_vs_ieta_HF1 = ibooker.book1D(histo, histo, 82, -41., 41.);
      sprintf  (histo, "occupancy_vs_ieta_HF2" );
      occupancy_vs_ieta_HF2 = ibooker.book1D(histo, histo, 82, -41., 41.);
      
      //These are not
      if (useAllHistos_){
	sprintf  (histo, "occ_sequential1D_HB1" );
	occupancy_seqHB1 = ibooker.book1D(histo, histo,2400, -1200., 1200.);
	sprintf  (histo, "occ_sequential1D_HB2" );
	occupancy_seqHB2 = ibooker.book1D(histo, histo,2400, -1200., 1200.);
	sprintf  (histo, "occ_sequential1D_HE1" );
	occupancy_seqHE1 = ibooker.book1D(histo, histo,4400, -2200., 2200.);
	sprintf  (histo, "occ_sequential1D_HE2" );
	occupancy_seqHE2 = ibooker.book1D(histo, histo,4400, -2200., 2200.);
	sprintf  (histo, "occ_sequential1D_HE3" );
	occupancy_seqHE3 = ibooker.book1D(histo, histo,4400, -2200., 2200.);
	sprintf  (histo, "occ_sequential1D_HO" );
	occupancy_seqHO  = ibooker.book1D(histo, histo,2400, -1200., 1200.);
	sprintf  (histo, "occ_sequential1D_HF1" );
	occupancy_seqHF1 = ibooker.book1D(histo, histo,6000, -3000., 3000.);
	sprintf  (histo, "occ_sequential1D_HF2" );
	occupancy_seqHF2 = ibooker.book1D(histo, histo,6000, -3000., 3000.);
      }

      //All status word histos except HF67 are drawn
      sprintf (histo, "HcalRecHitTask_RecHit_StatusWord_HB" ) ;
      RecHit_StatusWord_HB = ibooker.book1D(histo, histo, 32 , -0.5, 31.5); 
      
      sprintf (histo, "HcalRecHitTask_RecHit_StatusWord_HE" ) ;
      RecHit_StatusWord_HE = ibooker.book1D(histo, histo, 32 , -0.5, 31.5); 

      sprintf (histo, "HcalRecHitTask_RecHit_StatusWord_HF" ) ;
      RecHit_StatusWord_HF = ibooker.book1D(histo, histo, 32 , -0.5, 31.5); 

      if (useAllHistos_){
	sprintf (histo, "HcalRecHitTask_RecHit_StatusWord_HF67" ) ;
	RecHit_StatusWord_HF67 = ibooker.book1D(histo, histo, 3 , 0.5, 3.5); 
      }
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

    //======================= Now various cases one by one ===================

    //Histograms drawn for single pion scan
    if(subdet_ != 0 && imc != 0) { // just not for noise  
      sprintf (histo, "HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths");
      meEnConeEtaProfile = ibooker.bookProfile(histo, histo, 82, -41., 41.,        2100, -100., 2000.);  
      
      sprintf (histo, "HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_E");
      meEnConeEtaProfile_E = ibooker.bookProfile(histo, histo, 82, -41., 41.,      2100, -100., 2000.);  
      
      sprintf (histo, "HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_EH");
      meEnConeEtaProfile_EH = ibooker.bookProfile(histo, histo, 82, -41., 41.,     2100, -100., 2000.);  
    }
    //The other cone profile, delta ieta/phi and noise histos are not drawn
    if (useAllHistos_){
      if(subdet_ != 0 && imc != 0) { // just not for noise  
	
	//    meEnConeEtaProfiel_depth1->Fill(eta_RecHit, HcalCone_d1);
	
	sprintf (histo, "HcalRecHitTask_En_rechits_cone_profile_vs_ieta_depth1");
	meEnConeEtaProfile_depth1 = ibooker.bookProfile(histo, histo, 82, -41., 41., 2100, -100., 2000.);   
	
	sprintf (histo, "HcalRecHitTask_En_rechits_cone_profile_vs_ieta_depth2");
	meEnConeEtaProfile_depth2 = ibooker.bookProfile(histo, histo, 82, -41., 41., 2100, -100., 2000.);  
	
	sprintf (histo, "HcalRecHitTask_En_rechits_cone_profile_vs_ieta_depth3");
	meEnConeEtaProfile_depth3 = ibooker.bookProfile(histo, histo, 82, -41., 41., 2100, -100., 2000.);  
	
	sprintf (histo, "HcalRecHitTask_En_rechits_cone_profile_vs_ieta_depth4");
	meEnConeEtaProfile_depth4 = ibooker.bookProfile(histo, histo, 82, -41., 41., 2100, -100., 2000.);  
	
      }
      
      if(etype_ == 1 && subdet_ != 0) { // single part., not for noise
	
	sprintf  (histo, "Delta_phi_cluster-MC");
	meDeltaPhi =  ibooker.book2D(histo, histo, 520, -5.2, 5.2, 60, -0.6, 0.6);
	
	sprintf  (histo, "Delta_eta_cluster-MC");
	meDeltaEta =  ibooker.book2D(histo, histo, 520, -5.2, 5.2, 60, -0.6, 0.6);
      
      }
      // NOISE-specific
      
      if (hcalselector_ == "noise" ){
	
	sprintf  (histo, "e_hb" ) ;
	e_hb = ibooker.book1D(histo, histo,1000, -5., 5.);
	sprintf  (histo, "e_he" ) ;
	e_he = ibooker.book1D(histo, histo,1000, -5., 5.);
	sprintf  (histo, "e_ho" ) ;
	e_ho = ibooker.book1D(histo, histo,1000, -5., 5.);
	sprintf  (histo, "e_hfl" ) ;
	e_hfl = ibooker.book1D(histo, histo,2000, -10., 10.);
	sprintf  (histo, "e_hfs" ) ;
	e_hfs = ibooker.book1D(histo, histo,2000, -10., 10.);
      }
    }
    // ************** HB **********************************
    if (subdet_ == 1 || subdet_ == 5 ){

      //Only severity level, energy of rechits and overall HB timing histos are drawn  
      if (useAllHistos_){
	if(etype_ == 1 && subdet_ == 1 ) { 
	  if(imc != 0) {
	    sprintf (histo, "HcalRecHitTask_number_of_rechits_in_cone_HB" ) ;
	    meNumRecHitsConeHB    = ibooker.book1D(histo, histo, 100, 0., 100.);
	    
	    sprintf (histo, "HcalRecHitTask_sum_of_rechits_energy_in_cone_HB" ) ;
	    meSumRecHitsEnergyConeHB = ibooker.book1D(histo,histo, 60 ,-20., 280.);
	  }
	  
	  sprintf (histo, "HcalRecHitTask_number_of_rechits_above_1GeV_HB");
	  meNumRecHitsThreshHB = ibooker.book1D(histo, histo,  30, 0., 30.); 
	  
	  sprintf (histo, "HcalRecHitTask_sum_of_rechits_energy_HB" ) ;
	  meSumRecHitsEnergyHB = ibooker.book1D(histo,histo, 60 , -20., 280.);
	
	  if (ecalselector_ == "yes") {  
	    if(imc != 0) {
	      sprintf (histo, "HcalRecHitTask_number_of_ecalrechits_in_cone_HB");
	      meNumEcalRecHitsConeHB = ibooker.book1D(histo, histo, 300, 0., 300.);	    
	      sprintf (histo, "HcalRecHitTask_energy_ecal_plus_hcal_in_cone_HB");
	      meEcalHcalEnergyConeHB =  ibooker.book1D(histo,histo, 60 , -20., 280.);
	    }
	    
	    sprintf (histo, "HcalRecHitTask_energy_hcal_vs_ecal_HB");
	    meEnergyHcalVsEcalHB = ibooker.book2D(histo, histo, 300, 0., 150., 300, 0., 150.);  	
	    sprintf (histo, "HcalRecHitTask_energy_ecal_plus_hcal_HB" ) ;
	    meEcalHcalEnergyHB = ibooker.book1D(histo,histo, 60 , -20., 280.);
	  }
	}
      }
      
      sprintf(histo, "HcalRecHitTask_severityLevel_HB");
      sevLvl_HB = ibooker.book1D(histo, histo, 25, -0.5, 24.5); 

      sprintf (histo, "HcalRecHitTask_energy_of_rechits_HB" ) ;
      meRecHitsEnergyHB = ibooker.book1D(histo, histo, 2010 , -10. , 2000.); 
      
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
      meTEprofileHB_Low = ibooker.bookProfile(histo, histo, 50, -5., 45., 70, -48., 92.); 

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_profile_HB" ) ;
      meTEprofileHB = ibooker.bookProfile(histo, histo, 150, -5., 295., 70, -48., 92.); 

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_profile_High_HB" ) ;
      meTEprofileHB_High = ibooker.bookProfile(histo, histo, 150, -5., 2995., 70, -48., 92.); 

    }
    
    // ********************** HE ************************************
    if ( subdet_ == 2 || subdet_ == 5 ){

      //None of these are drawn
      if (useAllHistos_){
	if(etype_ == 1 && subdet_ == 2 ) { 
	  
	  if(imc != 0) {
	    sprintf (histo, "HcalRecHitTask_number_of_rechits_in_cone_HE" ) ;
	    meNumRecHitsConeHE    = ibooker.book1D(histo, histo, 100, 0., 100.);
	    
	    sprintf (histo, "HcalRecHitTask_sum_of_rechits_energy_in_cone_HE" ) ;
	    meSumRecHitsEnergyConeHE = ibooker.book1D(histo,histo, 60 ,-20., 280.);
	  }	
	  
	  sprintf (histo, "HcalRecHitTask_number_of_rechits_above_1GeV_HE");
	  meNumRecHitsThreshHE = ibooker.book1D(histo, histo,  30, 0., 30.);  
	  
	  sprintf (histo, "HcalRecHitTask_sum_of_rechits_energy_HE" ) ;
	  meSumRecHitsEnergyHE = ibooker.book1D(histo,histo, 60 , -20., 280.);
	  
	  if (ecalselector_ == "yes") {  	
	    sprintf (histo, "HcalRecHitTask_energy_ecal_plus_hcal_HE" ) ;
	    meEcalHcalEnergyHE = ibooker.book1D(histo,histo, 80, -20., 380.);
	    
	    sprintf (histo, "HcalRecHitTask_energy_hcal_vs_ecal_HE");
	    meEnergyHcalVsEcalHE = ibooker.book2D(histo, histo, 300, 0., 150., 300, 0., 150.);
	    if(imc != 0) {
	      sprintf (histo, "HcalRecHitTask_number_of_ecalrechits_in_cone_HE");
	      meNumEcalRecHitsConeHE = ibooker.book1D(histo, histo, 300, 0., 300.);   
	      sprintf (histo, "HcalRecHitTask_energy_ecal_plus_hcal_in_cone_HE");
	      meEcalHcalEnergyConeHE =  ibooker.book1D(histo,histo, 60,-20., 280.);
	    }
	  }	      
	}
      }
      
      //Only severity level, energy of rechits and overall HB timing histos are drawn  
      sprintf(histo, "HcalRecHitTask_severityLevel_HE");
      sevLvl_HE = ibooker.book1D(histo, histo, 25, -0.5, 24.5); 
      
      sprintf (histo, "HcalRecHitTask_energy_of_rechits_HE" ) ;
      meRecHitsEnergyHE = ibooker.book1D(histo, histo, 510, -10., 500.); 
      
      sprintf (histo, "HcalRecHitTask_timing_HE" ) ;
      meTimeHE = ibooker.book1D(histo, histo, 70, -48., 92.); 
      
      sprintf (histo, "HcalRecHitTask_timing_vs_energy_Low_HE" ) ;
      meTE_Low_HE = ibooker.book2D(histo, histo, 80, -5., 75.,  70, -48., 92.);

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_HE" ) ;
      meTE_HE = ibooker.book2D(histo, histo, 200, -5., 395.,  70, -48., 92.);
      
      sprintf (histo, "HcalRecHitTask_timing_vs_energy_profile_Low_HE" ) ;
      meTEprofileHE_Low = ibooker.bookProfile(histo, histo, 80, -5., 75., 70, -48., 92.); 

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_profile_HE" ) ;
      meTEprofileHE = ibooker.bookProfile(histo, histo, 200, -5., 395., 70, -48., 92.); 
      
    }

    // ************** HO ****************************************
    if ( subdet_ == 3 || subdet_ == 5  ){
      
      //Only severity level, energy of rechits and overall HB timing histos are drawn  
      if (useAllHistos_){
	if(etype_ == 1 && subdet_ == 3) { 
	  if (imc != 0) {
	    sprintf (histo, "HcalRecHitTask_number_of_rechits_in_cone_HO" ) ;
	    meNumRecHitsConeHO    = ibooker.book1D(histo, histo, 100, 0 , 100.);
	    
	    sprintf (histo, "HcalRecHitTask_sum_of_rechits_energy_in_cone_HO" ) ;
	    meSumRecHitsEnergyConeHO = ibooker.book1D(histo,histo, 80 ,-20., 380.);
	  }
	  
	  sprintf (histo, "HcalRecHitTask_number_of_rechits_above_1GeV_HO");
	  meNumRecHitsThreshHO = ibooker.book1D(histo, histo,   100, 0., 100.);   
	  
	  sprintf (histo, "HcalRecHitTask_sum_of_rechits_energy_HO" ) ;
	  meSumRecHitsEnergyHO = ibooker.book1D(histo,histo, 80 , -20., 380.);
	}
      }      
      
      sprintf(histo, "HcalRecHitTask_severityLevel_HO");
      sevLvl_HO = ibooker.book1D(histo, histo, 25, -0.5, 24.5); 

      sprintf (histo, "HcalRecHitTask_energy_of_rechits_HO" ) ;
      meRecHitsEnergyHO = ibooker.book1D(histo, histo, 510 , -10. , 500.); 
      
      sprintf (histo, "HcalRecHitTask_timing_HO" ) ;
      meTimeHO = ibooker.book1D(histo, histo, 70, -48., 92.); 
      
      sprintf (histo, "HcalRecHitTask_timing_vs_energy_HO" ) ;
      meTE_HO= ibooker.book2D(histo, histo, 60, -5., 55., 70, -48., 92.);

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_High_HO" ) ;
      meTE_High_HO= ibooker.book2D(histo, histo, 100, -5., 995., 70, -48., 92.);
      
      sprintf (histo, "HcalRecHitTask_timing_vs_energy_profile_HO" ) ;
      meTEprofileHO = ibooker.bookProfile(histo, histo, 60, -5., 55.,  70, -48., 92.); 

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_profile_High_HO" ) ;
      meTEprofileHO_High = ibooker.bookProfile(histo, histo, 100, -5., 995.,  70, -48., 92.); 
      
    }   
  
    // ********************** HF ************************************
    if ( subdet_ == 4 || subdet_ == 5 ){

      //Only severity level, energy of rechits and overall HB timing histos are drawn  
      if (useAllHistos_){
	if(etype_ == 1 &&  subdet_ == 4) { 
	  
	  if(imc != 0) {
	    sprintf (histo, "HcalRecHitTask_number_of_rechits_in_cone_HF" ) ;
	    meNumRecHitsConeHF    = ibooker.book1D(histo, histo, 30, 0 , 30.);
	    
	    sprintf (histo, "HcalRecHitTask_sum_of_rechits_energy_in_cone_HF" ) ;
	    meSumRecHitsEnergyConeHF = ibooker.book1D(histo,histo,100, -20., 180.);
	    
	    sprintf (histo, "HcalRecHitTask_sum_of_rechits_energy_in_cone_HFL" );
	    meSumRecHitsEnergyConeHFL = ibooker.book1D(histo,histo,100,-20., 180.);
	    
	    sprintf (histo, "HcalRecHitTask_sum_of_rechits_energy_in_cone_HFS");
	    meSumRecHitsEnergyConeHFS = ibooker.book1D(histo,histo,100,-20., 180.);
	  }
	  sprintf (histo, "HcalRecHitTask_sum_of_rechits_energy_HF" ) ;
	  meSumRecHitsEnergyHF = ibooker.book1D(histo,histo, 80 , -20., 380.);  
	}
      }
      
      sprintf(histo, "HcalRecHitTask_severityLevel_HF");
      sevLvl_HF = ibooker.book1D(histo, histo, 25, -0.5, 24.5); 

      sprintf (histo, "HcalRecHitTask_energy_of_rechits_HF" ) ;
      meRecHitsEnergyHF = ibooker.book1D(histo, histo, 1010 , -10. , 1000.); 

      sprintf (histo, "HcalRecHitTask_timing_HF" ) ;
      meTimeHF = ibooker.book1D(histo, histo, 70, -48., 92.); 
      
      sprintf (histo, "HcalRecHitTask_timing_vs_energy_Low_HF" ) ;
      meTE_Low_HF = ibooker.book2D(histo, histo, 100, -5., 195., 70, -48., 92.);

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_HF" ) ;
      meTE_HF = ibooker.book2D(histo, histo, 200, -5., 995., 70, -48., 92.);
      
      sprintf (histo, "HcalRecHitTask_timing_vs_energy_profile_Low_HF" ) ;
      meTEprofileHF_Low = ibooker.bookProfile(histo, histo, 100, -5., 195., 70, -48., 92.); 

      sprintf (histo, "HcalRecHitTask_timing_vs_energy_profile_HF" ) ;
      meTEprofileHF = ibooker.bookProfile(histo, histo, 200, -5., 995., 70, -48., 92.); 

    }
//  }  //end-of if(_dbe) 

}


HcalRecHitsAnalyzer::~HcalRecHitsAnalyzer() { }

void HcalRecHitsAnalyzer::endJob() { }


void HcalRecHitsAnalyzer::beginJob(){ }

void HcalRecHitsAnalyzer::analyze(edm::Event const& ev, edm::EventSetup const& c) {

  using namespace edm;

  // cuts for each subdet_ector mimiking  "Scheme B"
  //  double cutHB = 0.9, cutHE = 1.4, cutHO = 1.1, cutHFL = 1.2, cutHFS = 1.8; 

  // energy in HCAL
  double eHcal        = 0.;
  double eHcalCone    = 0.;  
  double eHcalConeHB  = 0.;  
  double eHcalConeHE  = 0.;  
  double eHcalConeHO  = 0.;  
  double eHcalConeHF  = 0.;  
  double eHcalConeHFL = 0.;  
  double eHcalConeHFS = 0.;  
  // Total numbet of RecHits in HCAL, in the cone, above 1 GeV theshold
  int nrechits       = 0;
  int nrechitsCone   = 0;
  int nrechitsThresh = 0;

  // energy in ECAL
  double eEcal       = 0.;
  double eEcalB      = 0.;
  double eEcalE      = 0.;
  double eEcalCone   = 0.;
  int numrechitsEcal = 0;

  // HCAL energy around MC eta-phi at all depths;
  double partR = 0.3;
//   double ehcal_coneMC_1 = 0.;
//   double ehcal_coneMC_2 = 0.;
//   double ehcal_coneMC_3 = 0.;
//   double ehcal_coneMC_4 = 0.;

  // Cone size for serach of the hottest HCAL cell around MC
//   double searchR = 1.0; 
  double eps     = 0.001;

  // Single particle samples: actual eta-phi position of cluster around
  // hottest cell
  double etaHot  = 99999.; 
  double phiHot  = 99999.; 

  //   std::cout << "*** 2" << std::endl; 
  //   previously was:  c.get<IdealGeometryRecord>().get (geometry);
  c.get<CaloGeometryRecord>().get (geometry);

  // HCAL channel status map ****************************************
  edm::ESHandle<HcalChannelQuality> hcalChStatus;
  c.get<HcalChannelQualityRcd>().get( hcalChStatus );
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

  //  std::cout << "*** 3" << std::endl; 


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
      EBDetId EBid = EBDetId(RecHit->id());
       
      const CaloCellGeometry* cellGeometry =
	geometry->getSubdetectorGeometry (EBid)->getGeometry (EBid) ;
      double eta = cellGeometry->getPosition ().eta () ;
      double phi = cellGeometry->getPosition ().phi () ;
      double en  = RecHit->energy();
      eEcal  += en;
      eEcalB += en;

      if (useAllHistos_) map_ecal->Fill(eta, phi, en);

    }

    
    Handle<EERecHitCollection> rhitEE;
 
      ev.getByToken(tok_EE_, rhitEE);

    RecHit = rhitEE.product()->begin();  
    RecHitEnd = rhitEE.product()->end();  
    
    for (; RecHit != RecHitEnd ; ++RecHit) {
      EEDetId EEid = EEDetId(RecHit->id());
      
      const CaloCellGeometry* cellGeometry =
	geometry->getSubdetectorGeometry (EEid)->getGeometry (EEid) ;
      double eta = cellGeometry->getPosition ().eta () ;
      double phi = cellGeometry->getPosition ().phi () ;	
      double en   = RecHit->energy();
      eEcal  += en;
      eEcalE += en;

      if (useAllHistos_) map_ecal->Fill(eta, phi, en);

    }
  }     // end of ECAL selection 


  //     std::cout << "*** 4" << std::endl; 


  // Counting, including ZS items
  // Filling HCAL maps  ----------------------------------------------------
//   double maxE = -99999.;
  
  int nhb1 = 0;
  int nhb2 = 0;
  int nhe1 = 0;
  int nhe2 = 0;
  int nhe3 = 0;
  int nho  = 0;
  int nhf1 = 0;
  int nhf2 = 0;  
  
  for (unsigned int i = 0; i < cen.size(); i++) {
    
    int sub       = csub[i];
    int depth     = cdepth[i];
    int ieta      = cieta[i]; 
    int iphi      = ciphi[i]; 
    double en     = cen[i]; 
    //     double eta    = ceta[i]; 
    //     double phi    = cphi[i]; 
    uint32_t stwd = cstwd[i];
    uint32_t auxstwd = cauxstwd[i];
    //    double z   = cz[i];

    int index = ieta * 72 + iphi; //  for sequential histos
    
    /*   
	 std::cout << "*** point 4-1" << " ieta, iphi, depth, sub = "
	 << ieta << ", " << iphi << ", " << depth << ", " << sub  
	 << std::endl;
    */
    
    
    if( sub == 1 && depth == 1)  nhb1++;
    if( sub == 1 && depth == 2)  nhb2++;
    if( sub == 2 && depth == 1)  nhe1++;
    if( sub == 2 && depth == 2)  nhe2++;
    if( sub == 2 && depth == 3)  nhe3++;
    if( sub == 3 && depth == 4)  nho++;
    if( sub == 4 && depth == 1)  nhf1++;
    if( sub == 4 && depth == 2)  nhf2++;
    
    if( subdet_ == 6) {                                    // ZS specific
      if( en < emap_min[ieta+41][iphi][depth-1][sub-1] )
	emap_min[ieta+41][iphi][depth-1][sub-1] = en;
    }
    
//     double emin = 1.;
//     if(fabs(eta) > 3.) emin = 5.; 
    
    /*   
    if(ieta == 27 ) { 
      std::cout << "*** ieta=28, iphi = " << iphi << "  det = " 
		<< sub << "  depth = " << depth << std::endl;
    }
    */

    if( subdet_ != 6) {  

      //      std::cout << "*** 4-1" << std::endl; 
      //The emean_vs_ieta histos are drawn as well as the e_maps


      // to distinguish HE and HF
      if( depth == 1 || depth == 2 ) {
        int ieta1 =  ieta;
	if(sub == 4) { 
	  if (ieta1 < 0) ieta1--;
          else  ieta1++;   
	}
	if (depth == 1) emap_depth1->Fill(double(ieta1), double(iphi), en);
	if (depth == 2) emap_depth2->Fill(double(ieta1), double(iphi), en);
      }

      if( depth == 3) emap_depth3->Fill(double(ieta), double(iphi), en);
      if( depth == 4) emap_depth4->Fill(double(ieta), double(iphi), en);
      
      if (depth == 1 && sub == 1 ) {
	emean_vs_ieta_HB1->Fill(double(ieta), en);
	occupancy_map_HB1->Fill(double(ieta), double(iphi));          
	if(useAllHistos_){
	  emean_seqHB1->Fill(double(index), en);
	}
      }
      if (depth == 2  && sub == 1) {
	emean_vs_ieta_HB2->Fill(double(ieta), en);
	occupancy_map_HB2->Fill(double(ieta), double(iphi));          
	if(useAllHistos_){
	  emean_seqHB2->Fill(double(index), en);
	}
      }
      if (depth == 1 && sub == 2) {
	emean_vs_ieta_HE1->Fill(double(ieta), en);
	occupancy_map_HE1->Fill(double(ieta), double(iphi));   
	if(useAllHistos_){
	  emean_seqHE1->Fill(double(index), en);
	}
      }
      if (depth == 2 && sub == 2) {
	emean_vs_ieta_HE2->Fill(double(ieta), en);
	occupancy_map_HE2->Fill(double(ieta), double(iphi));          
	if(useAllHistos_){
	  emean_seqHE2->Fill(double(index), en);
	}
      }
      if (depth == 3 && sub == 2) {
	emean_vs_ieta_HE3->Fill(double(ieta), en);
	occupancy_map_HE3->Fill(double(ieta), double(iphi));          
	if(useAllHistos_){
	  emean_seqHE3->Fill(double(index), en);
	}
      }
      if (depth == 4 ) {
	emean_vs_ieta_HO->Fill(double(ieta), en);
	occupancy_map_HO->Fill(double(ieta), double(iphi));          
	if(useAllHistos_){
	  emean_seqHO->Fill(double(index), en);
	}
      }
      if (depth == 1 && sub == 4) {
	emean_vs_ieta_HF1->Fill(double(ieta), en);
	occupancy_map_HF1->Fill(double(ieta), double(iphi));          
	if(useAllHistos_){
	  emean_seqHF1->Fill(double(index), en);
	}
      }
      if (depth == 2 && sub == 4) {
	emean_vs_ieta_HF2->Fill(double(ieta), en);
	occupancy_map_HF2->Fill(double(ieta), double(iphi));          
	if(useAllHistos_){
	  emean_seqHF2->Fill(double(index), en);
	}
      }
    }
    

//     if( r < partR ) {
//       if (depth == 1) ehcal_coneMC_1 += en; 
//       if (depth == 2) ehcal_coneMC_2 += en; 
//       if (depth == 3) ehcal_coneMC_3 += en; 
//       if (depth == 4) ehcal_coneMC_4 += en; 
//     }
    
    //32-bit status word  
    uint32_t statadd;
    unsigned int isw67 = 0;
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
    if (isw67 != 0 && useAllHistos_) RecHit_StatusWord_HF67->Fill(isw67); //This one is not drawn

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
 
  //  std::cout << "*** 4-2" << std::endl; 
  
  if( subdet_ == 6 && useAllHistos_) {               // ZS plots; not drawn
    ZS_nHB1->Fill(double(nhb1));  
    ZS_nHB2->Fill(double(nhb2));  
    ZS_nHE1->Fill(double(nhe1));  
    ZS_nHE2->Fill(double(nhe2));  
    ZS_nHE3->Fill(double(nhe3));  
    ZS_nHO ->Fill(double(nho));  
    ZS_nHF1->Fill(double(nhf1));  
    ZS_nHF2->Fill(double(nhf2));  
  }
  else{ 
    Nhb->Fill(double(nhb1 + nhb2));
    Nhe->Fill(double(nhe1 + nhe2 + nhe3));
    Nho->Fill(double(nho));
    Nhf->Fill(double(nhf1 + nhf2));

  }

  //  std::cout << "*** 5" << std::endl; 
    

  //  NOISE ================================================================= 
  //Not drawn
  if (hcalselector_ == "noise" && useAllHistos_) {
    for (unsigned int i = 0; i < cen.size(); i++) {
      
      int sub   = csub[i];
      int depth = cdepth[i];
      double en = cen[i]; 
      
      if (sub == 1) e_hb->Fill(en);
      if (sub == 2) e_he->Fill(en);  
      if (sub == 3) e_ho->Fill(en);  
      if (sub == 4) {
	if(depth == 1)  
	  e_hfl->Fill(en);  
	else 
	  e_hfs->Fill(en);  
      }
    }
  }

  //===========================================================================
  // SUBSYSTEMS,  
  //===========================================================================
  
  else if ((subdet_ != 6) && (subdet_ != 0)) {

    //       std::cout << "*** 6" << std::endl; 
    
    
    double clusEta = 999.;
    double clusPhi = 999.; 
    double clusEn  = 0.;
    
    double HcalCone_d1 = 0.;
    double HcalCone_d2 = 0.;
    double HcalCone_d3 = 0.;
    double HcalCone_d4 = 0.;
    double HcalCone    = 0.;

    int ietaMax1  =  9999;
    int ietaMax2  =  9999;
    int ietaMax3  =  9999;
    int ietaMax4  =  9999;
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
      int depth  = cdepth[i];
      double eta = ceta[i]; 
      double phi = cphi[i]; 
      double en  = cen[i]; 
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
	
	meTE_Low_HB->Fill( en, t);
	meTE_HB->Fill( en, t);
	meTE_High_HB->Fill( en, t);
	meTEprofileHB_Low->Fill(en, t);
	meTEprofileHB->Fill(en, t);
	meTEprofileHB_High->Fill(en, t);

	if (useAllHistos_){
	  if      (depth == 1) meTE_HB1->Fill( en, t);
	  else if (depth == 2) meTE_HB2->Fill( en, t);
	}
      }     
      if(sub == 2 && (subdet_ == 2 || subdet_ == 5)) {  
	meTimeHE->Fill(t);
	meRecHitsEnergyHE->Fill(en);

	meTE_Low_HE->Fill( en, t);
	meTE_HE->Fill( en, t);
	meTEprofileHE_Low->Fill(en, t);
	meTEprofileHE->Fill(en, t);

	if (useAllHistos_){
	  if      (depth == 1) meTE_HE1->Fill( en, t);
	  else if (depth == 2) meTE_HE2->Fill( en, t);
	}
      }
      if(sub == 4 && (subdet_ == 4 || subdet_ == 5)) {  
	meTimeHF->Fill(t);
	meRecHitsEnergyHF->Fill(en);	  

	meTE_Low_HF->Fill(en, t);
	meTE_HF->Fill(en, t);
	meTEprofileHF_Low->Fill(en, t);
	meTEprofileHF->Fill(en, t);

	if (useAllHistos_){
	  if   (depth == 1) meTE_HFL->Fill( en, t);
	  else              meTE_HFS->Fill( en, t);
	}
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
      if (useAllHistos_){
	meEnConeEtaProfile_depth1->Fill(double(ietaMax1), HcalCone_d1);
	meEnConeEtaProfile_depth2->Fill(double(ietaMax2), HcalCone_d2);
	meEnConeEtaProfile_depth3->Fill(double(ietaMax3), HcalCone_d3);
	meEnConeEtaProfile_depth4->Fill(double(ietaMax4), HcalCone_d4);
      }
      meEnConeEtaProfile       ->Fill(double(ietaMax),  HcalCone);   // 
      meEnConeEtaProfile_E     ->Fill(double(ietaMax), eEcalCone);   
      meEnConeEtaProfile_EH    ->Fill(double(ietaMax),  HcalCone+eEcalCone); 
    }

    //     std::cout << "*** 7" << std::endl; 

    
    // Single particle samples ONLY !  ======================================
    // Fill up some histos for "integrated" subsustems. 
    // These are not drawn
    if(etype_ == 1 && useAllHistos_) {

      /*
      std::cout << "*** point 7-1" << "  eta_MC, phi_MC   clusEta, clusPhi = "
                << eta_MC  << ", " << phi_MC << "   "
		<< clusEta << ", " << clusPhi 
		<< std::endl;
      */    

//       double phidev = dPhiWsign(clusPhi, phi_MC);
//       meDeltaPhi->Fill(eta_MC, phidev);
//       double etadev = clusEta - eta_MC;
//       meDeltaEta->Fill(eta_MC, etadev);

      if(subdet_ == 1) {
	meSumRecHitsEnergyHB->Fill(eHcal);
	if(imc != 0) meSumRecHitsEnergyConeHB->Fill(eHcalConeHB);    
	if(imc != 0) meNumRecHitsConeHB->Fill(double(nrechitsCone));
	meNumRecHitsThreshHB->Fill(double(nrechitsThresh));
      }

      if(subdet_ == 2) {
        meSumRecHitsEnergyHE->Fill(eHcal);
	if(imc != 0) meSumRecHitsEnergyConeHE->Fill(eHcalConeHE);    
	if(imc != 0) meNumRecHitsConeHE->Fill(double(nrechitsCone));
	meNumRecHitsThreshHE->Fill(double(nrechitsThresh));
      }

      if(subdet_ == 3) {
	meSumRecHitsEnergyHO->Fill(eHcal);
	if(imc != 0) meSumRecHitsEnergyConeHO->Fill(eHcalConeHO);    
	if(imc != 0) meNumRecHitsConeHO->Fill(double(nrechitsCone));
	meNumRecHitsThreshHO->Fill(double(nrechitsThresh));
      }

      if(subdet_ == 4) {
        if(eHcalConeHF > eps ) {
	  meSumRecHitsEnergyHF ->Fill(eHcal);
	  if(imc != 0) { 
	    meSumRecHitsEnergyConeHF ->Fill(eHcalConeHF);    
	    meNumRecHitsConeHF->Fill(double(nrechitsCone));
	    meSumRecHitsEnergyConeHFL ->Fill(eHcalConeHFL);    
	    meSumRecHitsEnergyConeHFS ->Fill(eHcalConeHFS);    
	  }
	}
      }

      //         std::cout << "*** 8" << std::endl; 


      // Also combine with ECAL if needed 
      if(subdet_ == 1  && ecalselector_ == "yes") {
	
	/*
	  std::cout << "*** point 8-1" 
	  << "  eEcalB " << eEcalB << "  eHcal " << eHcal
	  << "  eEcalCone " <<  eEcalCone << "  eHcalCone " 
		  << eHcalCone
		  << "  numrechitsEcal " <<  numrechitsEcal
		  << std::endl;
		  
	*/
	
       	meEcalHcalEnergyHB->Fill(eEcalB+eHcal);
      	meEcalHcalEnergyConeHB->Fill(eEcalCone+eHcalCone);
      	meNumEcalRecHitsConeHB->Fill(double(numrechitsEcal));
	
      }
      
      if(subdet_ == 2  && ecalselector_ == "yes"){
	
	/*
	  std::cout << "*** point 8-2a" 
	  << "  eEcalE " << eEcalE << "  eHcal " << eHcal
	  << "  eEcalCone " <<  eEcalCone << "  eHcalCone " 
	  << eHcalCone
	  << "  numrechitsEcal " <<  numrechitsEcal
	  << std::endl;
	*/
	
	meEcalHcalEnergyHE->Fill(eEcalE+eHcal);
	if(imc != 0) meEcalHcalEnergyConeHE->Fill(eEcalCone+eHcalCone);
	if(imc != 0) meNumEcalRecHitsConeHE->Fill(double(numrechitsEcal));
      } 

      // Banana plots finally
      if(imc != 0) {
	if(subdet_ == 1 && ecalselector_ == "yes")
	  meEnergyHcalVsEcalHB -> Fill(eEcalCone,eHcalCone);
	if(subdet_ == 2 && ecalselector_ == "yes") 
	  meEnergyHcalVsEcalHE -> Fill(eEcalCone,eHcalCone);
      }
    }
  }
  //  std::cout << "*** 9" << std::endl; 

  nevtot++;
}


///////////////////////////////////////////////////////////////////////////////
void HcalRecHitsAnalyzer::fillRecHitsTmp(int subdet_, edm::Event const& ev){
  
  using namespace edm;
  
  
  // initialize data vectors
  csub.clear();
  cen.clear();
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
      const CaloCellGeometry* cellGeometry =
	geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
      double eta  = cellGeometry->getPosition().eta () ;
      double phi  = cellGeometry->getPosition().phi () ;
      double zc   = cellGeometry->getPosition().z ();
      int sub     = cell.subdet();
      int depth   = cell.depth();
      int inteta  = cell.ieta();
      if(inteta > 0) inteta -= 1;
      int intphi  = cell.iphi()-1;
      double en   = j->energy();
      double t    = j->time();
      int stwd    = j->flags();
      int auxstwd = j->aux();
      
      int serivityLevel = hcalSevLvl( (CaloRecHit*) &*j );
      if( cell.subdet()==HcalBarrel ){
         hcalHBSevLvlVec.push_back(serivityLevel);
      }else if (cell.subdet()==HcalEndcap ){
         hcalHESevLvlVec.push_back(serivityLevel);
      } 
      
      if((iz > 0 && eta > 0.) || (iz < 0 && eta <0.) || iz == 0) { 
	
	csub.push_back(sub);
	cen.push_back(en);
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
      if(inteta > 0) inteta -= 1;
      int intphi   = cell.iphi()-1;
      double en    = j->energy();
      double t     = j->time();
      int stwd     = j->flags();
      int auxstwd  = j->aux();

      int serivityLevel = hcalSevLvl( (CaloRecHit*) &*j );
      if( cell.subdet()==HcalForward ){
         hcalHFSevLvlVec.push_back(serivityLevel);
      } 

      if((iz > 0 && eta > 0.) || (iz < 0 && eta <0.) || iz == 0) { 
	
	csub.push_back(sub);
	cen.push_back(en);
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
      if(inteta > 0) inteta -= 1;
      int intphi   = cell.iphi()-1;
      double t     = j->time();
      double en    = j->energy();
      int stwd     = j->flags();
      int auxstwd  = j->aux();

      int serivityLevel = hcalSevLvl( (CaloRecHit*) &*j );
      if( cell.subdet()==HcalOuter ){
         hcalHOSevLvlVec.push_back(serivityLevel);
      } 
      
      if((iz > 0 && eta > 0.) || (iz < 0 && eta <0.) || iz == 0) { 
	csub.push_back(sub);
	cen.push_back(en);
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

   const DetId id = hit->detid();

   const uint32_t recHitFlag = hit->flags();
   const uint32_t dbStatusFlag = theHcalChStatus->getValues(id)->getValue();

   int severityLevel = theHcalSevLvlComputer->getSeverityLevel(id, recHitFlag, dbStatusFlag);

   return severityLevel;

} 

DEFINE_FWK_MODULE(HcalRecHitsAnalyzer);

