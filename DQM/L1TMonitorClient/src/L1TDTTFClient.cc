/*
 * \file L1TDTTFClient.cc
 *
 * \author G. Codispoti
 *
 */


#include "DQM/L1TMonitorClient/interface/L1TDTTFClient.h"

/// base services
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/MakerMacros.h"


L1TDTTFClient::L1TDTTFClient(const edm::ParameterSet& ps)
  : l1tdttffolder_ ( ps.getUntrackedParameter<std::string> ("l1tSourceFolder", "L1T/L1TDTTF") ),
    dttfSource_( ps.getParameter< edm::InputTag >("dttfSource") ),
    online_( ps.getUntrackedParameter<bool>("online", true) ),
    resetafterlumi_( ps.getUntrackedParameter<int>("resetAfterLumi", 3) ),
    counterLS_(0), occupancy_r_(0)
{
  edm::LogInfo( "L1TDTTFClient");
}


//--------------------------------------------------------
L1TDTTFClient::~L1TDTTFClient(){
  edm::LogInfo("L1TDTTFClient")<<"[L1TDTTFClient]: ending... ";
}


//--------------------------------------------------------
void L1TDTTFClient::book(DQMStore::IBooker &ibooker)
{
  
  edm::LogInfo("L1TDTTFClient")<<"[L1TDTTFClient]: Begin Job";

  wheelpath_[0] = l1tdttffolder_ + "/02-WHEEL_N2";
  wheelpath_[1] = l1tdttffolder_ + "/03-WHEEL_N1";
  wheelpath_[2] = l1tdttffolder_ + "/04-WHEEL_N0";
  wheelpath_[3] = l1tdttffolder_ + "/05-WHEEL_P0";
  wheelpath_[4] = l1tdttffolder_ + "/06-WHEEL_P1";
  wheelpath_[5] = l1tdttffolder_ + "/07-WHEEL_P2";

  wheel_[0] = "N2";
  wheel_[1] = "N1";
  wheel_[2] = "N0";
  wheel_[3] = "P0";
  wheel_[4] = "P1";
  wheel_[5] = "P2";

  inclusivepath_ = l1tdttffolder_ + "/01-INCLUSIVE";
  gmtpath_ = l1tdttffolder_ + "/08-GMT_MATCH";
  testpath_ = l1tdttffolder_ + "/09-TEST";

  /// occupancy summary
  char hname[100];//histo name
  char mename[100];//ME name

  /// SUMMARY
  ibooker.setCurrentFolder(inclusivepath_); 
  /// DTTF Tracks per Wheel ditribution
  sprintf(hname, "dttf_02_nTracks");
  sprintf(mename, "DTTF Tracks by Wheel");
  dttf_nTracks_integ = ibooker.book1D(hname, mename, 6, 0, 6);
  setWheelLabel( dttf_nTracks_integ );

  /// DTTF Tracks distribution by Sector and Wheel
  sprintf(hname, "dttf_03_tracks_occupancy_summary");
  sprintf(mename, "DTTF Tracks Occupancy");
  dttf_occupancySummary = ibooker.book2D( hname, mename, 6, 0, 6, 12, 1, 13 );
  setWheelLabel( dttf_occupancySummary );
  dttf_occupancySummary->setAxisTitle("Sector", 2);

  /// RESET 04

  /// DTTF Tracks BX Distribution by Wheel
  sprintf(hname, "dttf_05_bx_occupancy");
  sprintf(mename, "DTTF Tracks BX Distribution by Wheel");
  dttf_bx_summary = ibooker.book2D(hname, mename, 6, 0, 6, 3, -1, 2 );
  setWheelLabel( dttf_bx_summary );
  dttf_bx_summary->setAxisTitle("BX", 2 );

  /// Fraction of DTTF Tracks BX w.r.t. Tracks with BX=0
  sprintf(hname, "dttf_06_bx");
  sprintf(mename, "DTTF Tracks BX w.r.t. Tracks with BX=0");
  dttf_bx_integ = ibooker.book1D( hname, mename, 3, -1.5, 1.5 );
  dttf_bx_integ->setAxisTitle("BX", 1);

  /// DTTF Tracks Quality distribution
  sprintf(hname, "dttf_07_quality");
  sprintf(mename, "DTTF Tracks Quality distribution");
  dttf_quality_integ = ibooker.book1D(hname, mename, 7, 1, 8);
  setQualLabel( dttf_quality_integ, 1);

  /// DTTF Tracks Quality distribution by Wheel
  sprintf(hname, "dttf_08_quality_occupancy");
  sprintf(mename, "DTTF Tracks Quality distribution by Wheel");
  dttf_quality_summary = ibooker.book2D( hname, mename, 6, 0, 6, 7, 1, 8 );
  dttf_quality_summary->setAxisTitle("Wheel", 1);
  setQualLabel( dttf_quality_summary, 2);
  setWheelLabel( dttf_quality_summary );

  /// Fraction of DTTF Tracks with Quality>4 by Sector and Wheel
  sprintf(hname, "dttf_09_highQuality_Tracks");
  sprintf(mename, "Fraction of DTTF Tracks with Quality>3");
  dttf_highQual_Summary = ibooker.book2D( hname, mename, 6, 0, 6, 12, 1, 13 );
  setWheelLabel( dttf_highQual_Summary );
  dttf_highQual_Summary->setAxisTitle("Sector", 2);


  /// #eta-#phi Distribution of DTTF Tracks with coarse #eta assignment
  sprintf(hname, "dttf_10_phi_vs_etaCoarse");
  sprintf(mename, "#eta-#phi DTTF Tracks occupancy (coarse #eta only, packed values)");
  dttf_phi_eta_coarse_integ = ibooker.book2D( hname, mename, 64, 0, 64,
					    144, -6, 138. );
  dttf_phi_eta_coarse_integ->setAxisTitle("#eta", 1);
  dttf_phi_eta_coarse_integ->setAxisTitle("#phi", 2);

  /// #eta-#phi Distribution of DTTF Tracks with fine #eta assignment
  sprintf(hname, "dttf_11_phi_vs_etaFine");
  sprintf(mename, "#eta-#phi DTTF Tracks occupancy (fine #eta only, packed values)");
  dttf_phi_eta_fine_integ = ibooker.book2D( hname, mename, 64, 0, 64,
					  144, -6, 138. );
  dttf_phi_eta_fine_integ->setAxisTitle("#eta", 1);
  dttf_phi_eta_fine_integ->setAxisTitle("#phi", 2);

  /// #eta-#phi Distribution of DTTF Tracks
  sprintf(hname, "dttf_12_phi_vs_eta");
  sprintf(mename, "#eta-#phi DTTF Tracks occupancy");
  dttf_phi_eta_integ = ibooker.book2D( hname, mename, 64, -1.2, 1.2,
				     144, -15, 345. );
  dttf_phi_eta_integ->setAxisTitle("#eta", 1);
  dttf_phi_eta_integ->setAxisTitle("#phi", 2);

  /// Fraction of DTTF Tracks with Fine #eta Assignment
  sprintf(hname, "dttf_13_eta_fine_fraction");
  sprintf(mename, "Fraction of DTTF Tracks with Fine #eta Assignment");
  dttf_eta_fine_fraction = ibooker.book1D( hname, mename, 6, 0, 6 );
  setWheelLabel(dttf_eta_fine_fraction);
  dttf_eta_fine_fraction->setAxisTitle("", 2);

  //////// TH1F

  /// DTTF Tracks #eta distribution (Packed values)
  sprintf(hname, "dttf_14_eta");
  sprintf(mename, "DTTF Tracks #eta distribution (Packed values)");
  dttf_eta_integ = ibooker.book1D(hname, mename, 64, -0.5, 63.5);
  dttf_eta_integ->setAxisTitle("#eta", 1);

  /// DTTF Tracks Phi distribution (Packed values)
  sprintf(hname, "dttf_15_phi");
  sprintf(mename, "DTTF Tracks Phi distribution (Packed values)");
  dttf_phi_integ = ibooker.book1D(hname, mename, 144, -6, 138. );
  dttf_phi_integ->setAxisTitle("#phi", 1);

  /// DTTF Tracks p_{T} distribution (Packed values)
  sprintf(hname, "dttf_16_pt");
  sprintf(mename, "DTTF Tracks p_{T} distribution (Packed values)");
  dttf_pt_integ  = ibooker.book1D(hname, mename, 32, -0.5, 31.5);
  dttf_pt_integ->setAxisTitle("p_{T}", 1);
  /// DTTF Tracks Charge distribution
  sprintf(hname, "dttf_17_charge");
  sprintf(mename, "DTTF Tracks Charge distribution");
  dttf_q_integ = ibooker.book1D(hname, mename, 2, -0.5, 1.5);
  dttf_q_integ->setAxisTitle("Charge", 1);



  /// DTTF 2nd Tracks Only Distribution by Sector and Wheel w.r.t. the total Number of tracks
  sprintf(hname, "dttf_18_2ndTrack_occupancy_summary");
  sprintf(mename, "DTTF 2nd Tracks occupancy w.r.t. Tracks with BX=0");
  dttf_2ndTrack_Summary = ibooker.book2D( hname, mename, 6, 0, 6, 12, 1, 13 );
  setWheelLabel( dttf_2ndTrack_Summary );



  ////////////////////////////////////////////////////////
  /// GMT matching
  ////////////////////////////////////////////////////////
  ibooker.setCurrentFolder( gmtpath_ );
  sprintf(hname, "dttf_gmt_fract_matching" );
  sprintf(mename, "Fraction of DTTF tracks matching with GMT tracks" );
  dttf_gmt_matching = ibooker.book1D( hname, mename, 3, 1, 4);
  dttf_gmt_matching->setBinLabel(1, "GMT Only", 1);
  dttf_gmt_matching->setBinLabel(2, "Matching", 1);
  dttf_gmt_matching->setBinLabel(3, "DTTF Only", 1);


  ////////////////////////////////////////////////////////
  /// TEST directory for stats comparisons
  ////////////////////////////////////////////////////////

  ibooker.setCurrentFolder( testpath_ );
  sprintf(hname, "dttf_01_tracks_occupancy_test_summary");
  sprintf(mename, "DTTF Tracks Occupancy");
  dttf_occupancySummary_test = ibooker.book2D( hname, mename, 6, 0, 6, 12, 1, 13 );
  setWheelLabel( dttf_occupancySummary_test );
  dttf_occupancySummary_test->setAxisTitle("Sector", 2);


  ////////////////////////////////////////////////////////
  /////// Second Track
  ////////////////////////////////////////////////////////
  ibooker.setCurrentFolder( inclusivepath_ + "/2ND_TRACK_ONLY");

  /// DTTF 2nd Tracks per Wheel distribution
  sprintf(hname, "dttf_01_nTracks_2ndTrack");
  sprintf(mename, "DTTF 2nd Tracks per Wheel distribution");
  dttf_nTracks_integ_2ndTrack = ibooker.book1D(hname, mename, 6, 0, 6);
  setWheelLabel( dttf_nTracks_integ_2ndTrack );

  /// DTTF 2nd Tracks distribution by Sector and Wheel
  sprintf(hname, "dttf_02_occupancy_summary_2ndTrack");
  sprintf(mename, "DTTF 2nd Tracks distribution by Sector and Wheel");
  dttf_occupancySummary_2ndTrack = ibooker.book2D( hname, mename, 6, 0, 6,
						 12, 1, 13 );
  setWheelLabel( dttf_occupancySummary_2ndTrack );

  /// DTTF 2nd Tracks BX Distribution by Wheel
  sprintf(hname, "dttf_03_bx_occupancy_2ndTrack");
  sprintf(mename, "DTTF 2nd Tracks BX Distribution by Wheel");
  dttf_bx_summary_2ndTrack = ibooker.book2D( hname, mename, 6, 0, 6, 3, -1, 2 );
  setWheelLabel( dttf_bx_summary_2ndTrack );
  dttf_bx_summary_2ndTrack->setAxisTitle("BX", 2 );

  /// Fraction of DTTF Tracks BX w.r.t. Tracks with BX=0
  sprintf(hname, "dttf_04_bx_2ndTrack");
  sprintf(mename, "Fraction of DTTF Tracks BX w.r.t. Tracks with BX=0");
  dttf_bx_integ_2ndTrack = ibooker.book1D( hname, mename, 3, -1.5, 1.5 );
  dttf_bx_integ_2ndTrack->setAxisTitle("BX", 1);

  /// Quality
  sprintf(hname, "dttf_05_quality_2ndTrack");
  sprintf(mename, "DTTF 2nd Tracks Quality distribution");
  dttf_quality_integ_2ndTrack  = ibooker.book1D(hname, mename, 7, 0.5, 7.5);
  dttf_quality_integ_2ndTrack->setAxisTitle("Quality", 1);
  setQualLabel( dttf_quality_integ_2ndTrack, 1);

  /// DTTF 2nd Tracks Tracks Quality distribution by Wheel
  sprintf(hname, "dttf_06_quality_occupancy_2ndTrack");
  sprintf(mename, "DTTF 2nd Tracks Quality distribution by Wheel");
  dttf_quality_summary_2ndTrack = ibooker.book2D( hname, mename,
						6, 0, 6, 7, 1, 8 );
  dttf_quality_summary_2ndTrack->setAxisTitle("Wheel", 1);
  setQualLabel( dttf_quality_summary_2ndTrack, 2);
  setWheelLabel( dttf_quality_summary_2ndTrack );

  /// Fraction of DTTF 2nd Tracks with Quality>4 by Sector and Wheel
  sprintf(hname, "dttf_07_highQuality_Tracks_2ndTrack");
  sprintf(mename, "Fraction of DTTF 2nd Tracks with Quality>3");
  dttf_highQual_Summary_2ndTrack = ibooker.book2D( hname, mename, 6, 0, 6,
						 12, 1, 13 );
  dttf_highQual_Summary_2ndTrack->setAxisTitle("Sector", 2);
  setWheelLabel( dttf_highQual_Summary_2ndTrack );

  /// #eta-#phi Distribution of DTTF 2nd Tracks
  sprintf(hname, "dttf_08_phi_vs_eta_2ndTrack");
  sprintf(mename, "#eta-#phi DTTF Tracks occupancy for 2nd Tracks");
  dttf_phi_eta_integ_2ndTrack = ibooker.book2D( hname, mename, 64, 0, 64,
					      144, -6, 138. );
  dttf_phi_eta_integ_2ndTrack->setAxisTitle("#eta", 1);
  dttf_phi_eta_integ_2ndTrack->setAxisTitle("#phi", 2);

  /// #eta Distribution of DTTF 2nd Tracks
  sprintf(hname, "dttf_09_eta_2ndTrack");
  sprintf(mename, "DTTF 2nd Tracks #eta distribution (Packed values)");
  dttf_eta_integ_2ndTrack = ibooker.book1D(hname, mename, 64, -0.5, 63.5);
  dttf_eta_integ_2ndTrack->setAxisTitle("#eta", 1);

  /// phi Distribution of DTTF 2nd Tracks
  sprintf(hname, "dttf_10_phi_2ndTrack");
  sprintf(mename, "DTTF 2nd Tracks Phi distribution (Packed values)");
  dttf_phi_integ_2ndTrack = ibooker.book1D(hname, mename, 144, -6, 138 );
  dttf_phi_integ_2ndTrack->setAxisTitle("#phi", 1);

  /// pt Distribution of DTTF 2nd Tracks
  sprintf(hname, "dttf_11_pt_2ndTrack");
  sprintf(mename, "DTTF 2nd Tracks p_{T} distribution (Packed values)");
  dttf_pt_integ_2ndTrack  = ibooker.book1D(hname, mename, 32, -0.5, 31.5);
  dttf_pt_integ_2ndTrack->setAxisTitle("p_{T}", 1);

  /// charge Distribution of DTTF 2nd Tracks
  sprintf(hname, "dttf_12_charge_2ndTrack");
  sprintf(mename, "DTTF 2nd Tracks Charge distribution");
  dttf_q_integ_2ndTrack = ibooker.book1D(hname, mename, 2, -0.5, 1.5);
  dttf_q_integ_2ndTrack->setAxisTitle("Charge", 1);


  for ( unsigned int wh = 0; wh < 6 ; ++wh ) {
    ibooker.setCurrentFolder( wheelpath_[wh] );

    /// number of tracks per wheel
    sprintf( hname, "dttf_02_nTracks_wh%s", wheel_[wh].c_str() );
    sprintf( mename, "Wheel %s - Number of Tracks", wheel_[wh].c_str() );
    dttf_nTracks_wheel[wh] = ibooker.book1D( hname, mename, 12, 1, 13);
    dttf_nTracks_wheel[wh]->setAxisTitle("sector", 1);
 
    /// Tracks BX distribution by Sector for each wheel
    sprintf(hname, "dttf_03_bx_occupancy_wh%s",  wheel_[wh].c_str() );
    sprintf(mename, "Wheel %s - DTTF Tracks BX distribution by Sector",
    	    wheel_[wh].c_str());
    dttf_bx_wheel_summary[wh] = ibooker.book2D( hname, mename, 12, 1, 13, 3, -1, 2);
    dttf_bx_wheel_summary[wh]->setAxisTitle("BX", 2 );
    dttf_bx_wheel_summary[wh]->setAxisTitle("Sector", 1 );

    /// bx for each wheel
    sprintf(hname, "dttf_04_bx_wh%s", wheel_[wh].c_str());
    sprintf(mename, "Wheel %s - DTTF Tracks BX w.r.t. Tracks with BX=0", wheel_[wh].c_str());
    dttf_bx_wheel_integ[wh] = ibooker.book1D(hname, mename, 3, -1.5, 1.5);
    dttf_bx_wheel_integ[wh]->setAxisTitle("BX", 1);

    /// quality per wheel
    sprintf(hname, "dttf_05_quality_wh%s", wheel_[wh].c_str() );
    sprintf(mename, "Wheel %s - Tracks Quality Distribution", wheel_[wh].c_str() );
    dttf_quality_wheel[wh] = ibooker.book1D(hname, mename, 7, 1, 8);
    dttf_quality_wheel[wh]->setAxisTitle("Quality", 1);
    setQualLabel(dttf_quality_wheel[wh], 1);

    /// quality per wheel 2D
    sprintf(hname, "dttf_06_quality_summary_wh%s", wheel_[wh].c_str() );
    sprintf(mename, "Wheel %s - Tracks Quality Distribution by Sector",
	    wheel_[wh].c_str() );
    dttf_quality_summary_wheel[wh] = ibooker.book2D(hname, mename, 12, 1, 13, 7, 1, 8);
    dttf_quality_summary_wheel[wh]->setAxisTitle("Sector", 1);
    // dttf_quality_summary_wheel[wh]->setAxisTitle("Quality", 2);
    setQualLabel(dttf_quality_summary_wheel[wh], 2);

    /// eta assigment for each wheel
    sprintf(hname, "dttf_09_etaFine_fraction_wh%s",  wheel_[wh].c_str() );
    sprintf(mename, "Wheel %s - Fraction of DTTF Tracks with fine #eta assignment",
   	    wheel_[wh].c_str());
    dttf_fine_fraction_wh[wh] = ibooker.book1D( hname, mename, 12, 1, 13);
    dttf_fine_fraction_wh[wh]->setAxisTitle("Sector", 1 );


    /// DTTF Tracks #eta distribution (Packed values)
    sprintf(hname, "dttf_10_eta_wh%s", wheel_[wh].c_str() );
    sprintf(mename, "Wheel %s - DTTF Tracks #eta distribution (Packed values)",
	    wheel_[wh].c_str() );
    dttf_eta_wheel[wh] = ibooker.book1D(hname, mename, 64, -0.5, 63.5 );
    dttf_eta_wheel[wh]->setAxisTitle("#eta", 1 );

    /// DTTF Tracks Phi distribution (Packed values)
    sprintf(hname, "dttf_11_phi_wh%s", wheel_[wh].c_str() );
    sprintf(mename, "Wheel %s - DTTF Tracks Phi distribution (Packed values)",
	    wheel_[wh].c_str() );
    dttf_phi_wheel[wh] = ibooker.book1D(hname, mename, 144, -6, 138. );
    dttf_phi_wheel[wh]->setAxisTitle("#phi", 1);

    /// DTTF Tracks p_{T} distribution (Packed values)
    sprintf(hname, "dttf_12_pt_wh%s", wheel_[wh].c_str() );
    sprintf(mename, "Wheel %s - DTTF Tracks p_{T} distribution (Packed values)",
	    wheel_[wh].c_str() );
    dttf_pt_wheel[wh] = ibooker.book1D(hname, mename, 32, -0.5, 31.5);
    dttf_pt_wheel[wh]->setAxisTitle("p_{T}", 1);

    /// DTTF Tracks Charge distribution
    sprintf(hname, "dttf_13_charge_wh%s", wheel_[wh].c_str() );
    sprintf(mename, "Wheel %s - DTTF Tracks Charge distribution",
	    wheel_[wh].c_str() );
    dttf_q_wheel[wh] = ibooker.book1D(hname, mename, 2, -0.5, 1.5);
    dttf_q_wheel[wh]->setAxisTitle("Charge", 1);

  }


  //// 2ND track by wheel
  for ( unsigned int wh = 0; wh < 6 ; ++wh ) {
    ibooker.setCurrentFolder( wheelpath_[wh]  + "/2ND_TRACK_ONLY" );

    /// number of 2nd tracks per wheel
    sprintf( hname, "dttf_01_nTracks_wh%s_2ndTrack", wheel_[wh].c_str() );
    sprintf( mename, "Wheel %s - DTTF Number of 2nd Tracks with BX=0",
	     wheel_[wh].c_str() );
    dttf_nTracks_wheel_2ndTrack[wh] = ibooker.book1D( hname, mename,
						     12, 1, 13);
    dttf_nTracks_wheel_2ndTrack[wh]->setAxisTitle("sector", 1);
 
    /// bx for each wheel
    sprintf(hname, "dttf_02_bx_occupancy_wh%s_2ndTrack",  wheel_[wh].c_str() );
    sprintf(mename, "Wheel %s - DTTF 2nd Tracks BX distribution by Sector",
    	    wheel_[wh].c_str());
    dttf_bx_wheel_summary_2ndTrack[wh] = ibooker.book2D( hname, mename, 12, 1, 13, 3, -1, 2);
    dttf_bx_wheel_summary_2ndTrack[wh]->setAxisTitle("BX", 2 );
    dttf_bx_wheel_summary_2ndTrack[wh]->setAxisTitle("Sector", 1 );

    /// bx for each wheel
    sprintf(hname, "dttf_03_bx_wh%s_2ndTrack", wheel_[wh].c_str());
    sprintf(mename, "Wheel %s - 2nd Tracks BX Distribution", wheel_[wh].c_str());
    dttf_bx_wheel_integ_2ndTrack[wh] = ibooker.book1D(hname, mename, 3, -1.5, 1.5);
    dttf_bx_wheel_integ_2ndTrack[wh]->setAxisTitle("BX", 1);


  }



}

//--------------------------------------------------------

void L1TDTTFClient::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter){

  //Processing by run ID
  book(ibooker);
  makeSummary(igetter);

}

//--------------------------------------------------------
void L1TDTTFClient::dqmEndLuminosityBlock(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter, const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c)
{
  /// Processing by Luminosity Block
  if ( online_ && !( counterLS_ % resetafterlumi_ ) ) {
    book(ibooker);
    char hname[60];
    sprintf( hname, "%s/dttf_04_tracks_occupancy_by_lumi",
	     inclusivepath_.c_str() );

    occupancy_r_ = getTH2F(igetter, hname);
    if ( ! occupancy_r_ ) {
      edm::LogError("L1TDTTFClient::beginLuminosityBlock:ME")
          << "Failed to get TH2D " << std::string(hname);
    } else {
      edm::LogInfo("L1TDTTFClient::beginLuminosityBlock:RESET") << "Reseting plots by lumi!";
      occupancy_r_->Reset();
    }
  }
  
  if (  online_  ) {
    makeSummary(igetter);
    if ( occupancy_r_ ) normalize( occupancy_r_ );
  }

}

//--------------------------------------------------------
void L1TDTTFClient::makeSummary(DQMStore::IGetter &igetter)
{
  ////////////////////////////////////////////////////////
  /// Build Summariy plots
  ////////////////////////////////////////////////////////
  buildSummaries(igetter);

  ////////////////////////////////////////////////////////
  /// RESCALE PLOTS
  ////////////////////////////////////////////////////////
  double scale = 0;
  double entries = dttf_occupancySummary->getTH2F()->Integral();

  if ( entries ) {

    /// BX has simply all entries
    normalize( dttf_bx_summary->getTH2F() );

    /// Scale plots with all entries
    scale = 1 / entries;
    normalize( dttf_occupancySummary->getTH2F(), scale );
    normalize( dttf_nTracks_integ->getTH1F(), scale );

    /// Scale plots with only physical entries (no N0 duplicates)
    double physEntries = dttf_eta_integ->getTH1F()->Integral();
    if ( physEntries > 0 ) {
      double physScale = 1 / physEntries;

      normalize( dttf_phi_eta_integ->getTH2F(), physScale );

      normalize( dttf_phi_eta_fine_integ->getTH2F(), physScale );
      normalize( dttf_phi_eta_coarse_integ->getTH2F(), physScale );
      normalize( dttf_quality_summary->getTH2F(), physScale );

      normalize( dttf_eta_integ->getTH1F(), physScale );
      normalize( dttf_q_integ->getTH1F(), physScale );
      normalize( dttf_pt_integ->getTH1F(), physScale );
      normalize( dttf_phi_integ->getTH1F(), physScale );
      normalize( dttf_quality_integ->getTH1F(), physScale );

    }

  }

  ////////////////////////////////////////////////////////
  /// RESCALE PLOTS FOR 2nd tracks
  ////////////////////////////////////////////////////////

  double entries2ndTrack = dttf_occupancySummary_2ndTrack->getTH2F()->Integral();
  if ( entries2ndTrack > 0 ) {

    /// BX has simply all entries
    normalize( dttf_bx_summary_2ndTrack->getTH2F() ); //

    /// buildHigh Quality Summary Plot
    TH2F * ratio = dttf_occupancySummary_2ndTrack->getTH2F();
    buildHighQualityPlot( igetter, ratio, dttf_highQual_Summary_2ndTrack,
			  "%s/2ND_TRACK_ONLY/dttf_05_quality_summary_wh%s_2ndTrack" );

    normalize( dttf_2ndTrack_Summary->getTH2F(), scale ); //

    /// Scale plots with all entries
    double scale2nd = 1 / entries2ndTrack;
    normalize( dttf_occupancySummary_2ndTrack->getTH2F(), scale2nd ); //
    normalize( dttf_nTracks_integ_2ndTrack->getTH1F(), scale2nd ); //


    /// Scale plots with only physical entries (no N0 duplicates)
    normalize( dttf_phi_eta_integ_2ndTrack->getTH2F() );
    normalize( dttf_q_integ_2ndTrack->getTH1F() );
    normalize( dttf_pt_integ_2ndTrack->getTH1F() );
    normalize( dttf_phi_integ_2ndTrack->getTH1F() );
    normalize( dttf_eta_integ_2ndTrack->getTH1F() );
    normalize( dttf_quality_integ_2ndTrack->getTH1F() );
    normalize( dttf_quality_summary_2ndTrack->getTH2F() );

  }

  /// GMT
  setGMTsummary(igetter);

}




//--------------------------------------------------------
void L1TDTTFClient::buildSummaries(DQMStore::IGetter &igetter)
{

  char hname[100];
  int wheelBx[4]; /// needed for bx_integ
  int wheelBx2nd[4]; /// needed for bx_summary 2nd
  int wheelSumBx[4]; /// needed for bx_integ
  int wheelSumBx2nd[4]; /// needed for bx_summary 2nd
  int qualities[8]; /// needed for by wheel qualities

  memset( wheelSumBx, 0, 4 * sizeof(int) );
  memset( wheelSumBx2nd, 0, 4 * sizeof(int) );

  /// reset histograms
  dttf_eta_integ->Reset();
  dttf_q_integ->Reset();
  dttf_pt_integ->Reset();
  dttf_phi_integ->Reset();
  dttf_quality_integ->Reset();
  dttf_phi_eta_integ->Reset();
  dttf_phi_eta_fine_integ->Reset();
  dttf_phi_eta_coarse_integ->Reset();
  dttf_q_integ_2ndTrack->Reset();
  dttf_pt_integ_2ndTrack->Reset();
  dttf_phi_integ_2ndTrack->Reset();
  dttf_eta_integ_2ndTrack->Reset();

  /// ????
  dttf_quality_integ_2ndTrack->Reset();
  dttf_quality_summary_2ndTrack->Reset();
 
  for ( unsigned int wh = 0; wh < 6 ; ++wh ) {

    dttf_eta_wheel[wh]->Reset();
    dttf_q_wheel[wh]->Reset();
    dttf_pt_wheel[wh]->Reset();
    dttf_phi_wheel[wh]->Reset();
    dttf_quality_wheel[wh]->Reset();
    dttf_quality_summary_wheel[wh]->Reset();
 
    double wheelEtaAll = 0; /// needed for fine fraction
    double wheelEtaFine = 0; /// needed for fine fraction
    memset( wheelBx, 0, 4 * sizeof(int) );
    memset( wheelBx2nd, 0, 4 * sizeof(int) );

    /// for quality
    memset( qualities, 0, 8 * sizeof(int) );

    ////////////////////////////////////////////////////////
    /// PHI vs Eta
    ////////////////////////////////////////////////////////
    buildPhiEtaPlotOFC( igetter, dttf_phi_eta_fine_integ, dttf_phi_eta_coarse_integ,
			dttf_phi_eta_integ,
			"%s/dttf_07_phi_vs_etaFine_wh%s",
			"%s/dttf_08_phi_vs_etaCoarse_wh%s", wh );

    buildPhiEtaPlotO( igetter, dttf_phi_eta_integ_2ndTrack,
		      "%s/2ND_TRACK_ONLY/dttf_06_phi_vs_eta_wh%s_2ndTrack",
		      wh );


    ////////////////////////////////////////////////////////
    /// Loop over sectors 
    ////////////////////////////////////////////////////////
    for ( unsigned int sector = 1; sector < 13; ++sector ) {

      ////////////////////////////////////////////////////////
      //// BX by sector
      ////////////////////////////////////////////////////////
      sprintf( hname, "%s/BX_BySector/dttf_bx_wh%s_se%d",
	       wheelpath_[wh].c_str(), wheel_[wh].c_str(), sector );
      
      TH1F * bxsector = getTH1F(igetter,hname);
      if ( ! bxsector ) {
	edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get TH1D "
						       << std::string(hname);
      } else {
      
	for ( unsigned int bx = 1; bx < 4 ; ++bx ) {

	  int bxval = bxsector->GetBinContent( bx );

	  if ( bx == 2 ) {
	    // if ( wh == 2 )
	    //   dttf_occupancySummary->setBinContent( wh+1, sector, bxval*14 );
	    // else
	    //   dttf_occupancySummary->setBinContent( wh+1, sector, bxval );
            dttf_occupancySummary->setBinContent( wh+1, sector, bxval );
            dttf_occupancySummary_test->setBinContent( wh+1, sector, bxval );
	    dttf_nTracks_wheel[wh]->setBinContent(sector, bxval );
	  }
	  wheelBx[bx] += bxval;
	  dttf_bx_wheel_summary[wh]->setBinContent( sector, bx, bxval );
	}
      }


      ////////////////////////////////////////////////////////
      //// BX 2nd by sector
      ////////////////////////////////////////////////////////
      sprintf( hname, "%s/BX_BySector/2ND_TRACK_ONLY/dttf_bx_2ndTrack_wh%s_se%d",
	       wheelpath_[wh].c_str(), wheel_[wh].c_str(), sector );
      
      TH1F * bxsector2nd = getTH1F(igetter,hname);
      if ( ! bxsector2nd ) {
	edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get TH1D "
						       << std::string(hname);
      } else {
      
	for ( unsigned int bx = 1; bx < 4 ; ++bx ) {
	  int bxval = bxsector2nd->GetBinContent( bx );

	  if ( bx == 2 ) {
	    dttf_2ndTrack_Summary->setBinContent( wh+1, sector, bxval );
	    dttf_occupancySummary_2ndTrack->setBinContent(wh+1, sector, bxval);
	    dttf_nTracks_wheel_2ndTrack[wh]->setBinContent(sector, bxval );
	  }
	  wheelBx2nd[bx] += bxval;
	  dttf_bx_wheel_summary_2ndTrack[wh]->setBinContent(sector, bx, bxval);
	}
      }

      ////////////////////////////////////////////////////////
      /// Charge by sector
      ////////////////////////////////////////////////////////
      sprintf( hname, "%s/Charge/dttf_charge_wh%s_se%d", 
	       wheelpath_[wh].c_str(), wheel_[wh].c_str(), sector );
      TH1F * tmp = getTH1F(igetter,hname);
      if ( ! tmp ) {
	edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get TH1D "
						       << std::string(hname);
      } else {
	dttf_q_integ->getTH1F()->Add( tmp );
	dttf_q_wheel[wh]->getTH1F()->Add( tmp );
      }

      ////////////////////////////////////////////////////////
      /// PT by sector
      ////////////////////////////////////////////////////////
      sprintf( hname, "%s/PT/dttf_pt_wh%s_se%d", 
	       wheelpath_[wh].c_str(), wheel_[wh].c_str(), sector );
      tmp = getTH1F(igetter, hname);
      if ( ! tmp ) {
	edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get TH1D "
						       << std::string(hname);
      } else {
	dttf_pt_integ->getTH1F()->Add( tmp );
	dttf_pt_wheel[wh]->getTH1F()->Add( tmp );
      }


      ////////////////////////////////////////////////////////
      /// Phi by sector
      ////////////////////////////////////////////////////////
      sprintf( hname, "%s/Phi/dttf_phi_wh%s_se%d", 
	       wheelpath_[wh].c_str(), wheel_[wh].c_str(), sector );
      tmp = getTH1F(igetter,hname);
      if ( ! tmp ) {
	edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get TH1D "
						       << std::string(hname);
      } else {
	dttf_phi_integ->getTH1F()->Add( tmp );
	dttf_phi_wheel[wh]->getTH1F()->Add( tmp );
      }


      ////////////////////////////////////////////////////////
      /// Quality by sector
      ////////////////////////////////////////////////////////
      double highQual = 0; /// needed for high quality plot
      double denHighQual = 0; /// needed for high quality plot (denominator)
      sprintf( hname, "%s/Quality/dttf_qual_wh%s_se%d", 
	       wheelpath_[wh].c_str(), wheel_[wh].c_str(), sector );
      tmp = getTH1F(igetter,hname);
      if ( ! tmp ) {
	edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get TH1D "
						       << std::string(hname);
      } else {

	dttf_quality_wheel[wh]->getTH1F()->Add( tmp );


	for ( unsigned int qual = 1; qual < 4 ; ++qual ) {
	  double bincontent = tmp->GetBinContent( qual );
	  qualities[qual] += bincontent;
	  denHighQual += bincontent;
	  dttf_quality_summary_wheel[wh]->setBinContent(sector, qual, bincontent);
	}

	for ( unsigned int qual = 4; qual < 8 ; ++qual ) {
	  double bincontent = tmp->GetBinContent( qual );
	  qualities[qual] += bincontent;
	  dttf_quality_summary_wheel[wh]->setBinContent(sector, qual, bincontent);
	  denHighQual += bincontent;
	  highQual += bincontent;
	}

      }
      if ( denHighQual > 0 ) highQual /= denHighQual;
      dttf_highQual_Summary->setBinContent( wh+1, sector, highQual );


      ////////////////////////////////////////////////////////
      /// eta fine by sector
      ////////////////////////////////////////////////////////
      sprintf( hname, "%s/Eta/dttf_eta_wh%s_se%d", 
	       wheelpath_[wh].c_str(), wheel_[wh].c_str(), sector );
      tmp = getTH1F(igetter,hname);
      if ( ! tmp ) {
	edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get TH1D "
						       << std::string(hname);
      } else {
	dttf_eta_integ->getTH1F()->Add( tmp );
	dttf_eta_wheel[wh]->getTH1F()->Add( tmp );
      }


      ////////////////////////////////////////////////////////
      /// eta fine fraction by sector
      ////////////////////////////////////////////////////////
      sprintf( hname, "%s/EtaFineFraction/dttf_etaFine_fraction_wh%s_se%d", 
	       wheelpath_[wh].c_str(), wheel_[wh].c_str(), sector );
      tmp = getTH1F(igetter,hname);
      if ( ! tmp ) {
	edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get TH1D "
						       << std::string(hname);
      } else {
	double fine = tmp->GetBinContent( 1 );
	double coarse = tmp->GetBinContent( 2 );
	double tot = fine + coarse;
	wheelEtaAll += tot;
	wheelEtaFine += fine;
	if ( tot > 0 ) {
	  dttf_fine_fraction_wh[wh]->setBinContent( sector, fine/tot );
	}
      }

    }

    ////////////////////////////////////////////////////////
    /// still eta: fraction by wheel
    ////////////////////////////////////////////////////////
    if ( wheelEtaAll > 0 ) {
      dttf_eta_fine_fraction->setBinContent( wh+1, wheelEtaFine/wheelEtaAll );
    }

    ////////////////////////////////////////////////////////
    /// integ summary
    ////////////////////////////////////////////////////////
    dttf_nTracks_integ->setBinContent( wh+1, wheelBx[2] );
    dttf_nTracks_integ_2ndTrack->setBinContent( wh+1, wheelBx2nd[2] );

    ////////////////////////////////////////////////////////
    /// still bx: wheel summary & inclusive
    ////////////////////////////////////////////////////////
    for ( unsigned int bx = 1; bx < 4; ++bx ) {

      dttf_bx_wheel_integ[wh]->setBinContent( bx, wheelBx[bx] );
      dttf_bx_summary->setBinContent( wh+1, bx, wheelBx[bx] );
      wheelSumBx[bx] += wheelBx[bx];

      dttf_bx_wheel_integ_2ndTrack[wh]->setBinContent( bx, wheelBx2nd[bx] );
      dttf_bx_summary_2ndTrack->setBinContent( wh+1, bx, wheelBx2nd[bx] );
      wheelSumBx2nd[bx] += wheelBx2nd[bx];

    }


    ////////////////////////////////////////////////////////
    /// by wheel quality: integ summary
    ////////////////////////////////////////////////////////
    for ( unsigned int qual = 1; qual < 8 ; ++qual ) {
      dttf_quality_summary->setBinContent( wh+1, qual, qualities[qual] );
      dttf_quality_integ->getTH1F()->AddBinContent( qual, qualities[qual] );
    }


    ////////////////////////////////////////////////////////
    /// by wheel rescaling bx by wheel and number of tracks distribution
    ////////////////////////////////////////////////////////
    normalize( dttf_bx_wheel_summary[wh]->getTH2F() );
    normalize( dttf_bx_wheel_summary_2ndTrack[wh]->getTH2F() );
    normalize( dttf_nTracks_wheel[wh]->getTH1F() );
    normalize( dttf_nTracks_wheel_2ndTrack[wh]->getTH1F() );
    normalize( dttf_quality_summary_wheel[wh]->getTH2F() );
    normalize( dttf_quality_wheel[wh]->getTH1F() );
    normalize( dttf_eta_wheel[wh]->getTH1F() );
    normalize( dttf_q_wheel[wh]->getTH1F() );
    normalize( dttf_pt_wheel[wh]->getTH1F() );
    normalize( dttf_phi_wheel[wh]->getTH1F() );

    ////////////////////////////////////////////////////////
    /// by wheel rescaling bx distributions
    ////////////////////////////////////////////////////////
    double scale = wheelBx[2];
    if ( scale > 0 ) {
      scale = 1/scale;
      normalize( dttf_bx_wheel_integ[wh]->getTH1F(), scale );
    }

    scale = wheelBx2nd[2];
    if ( scale > 0 ) {
      scale = 1/scale;
      normalize( dttf_bx_wheel_integ_2ndTrack[wh]->getTH1F(), scale );
    }




    ////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////
    /// MORE ON 2ND TRACKS
    ////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////
    /// eta by sector
    ////////////////////////////////////////////////////////
    sprintf( hname, "%s/2ND_TRACK_ONLY/dttf_07_eta_wh%s_2ndTrack", 
	     wheelpath_[wh].c_str(), wheel_[wh].c_str() );
    
    TH1F * tmp = getTH1F(igetter,hname);
    if ( ! tmp ) {
      edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get TH1D "
						     << std::string(hname);
    } else {
      dttf_eta_integ_2ndTrack->getTH1F()->Add( tmp );
    }

    ////////////////////////////////////////////////////////
    /// Phi by wheel 2nd Tracks
    ////////////////////////////////////////////////////////
    sprintf( hname, "%s/2ND_TRACK_ONLY/dttf_08_phi_wh%s_2ndTrack", 
	     wheelpath_[wh].c_str(), wheel_[wh].c_str() );
    tmp = getTH1F(igetter,hname);
    if ( ! tmp ) {
      edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get TH1D "
						     << std::string(hname);
    } else {
      dttf_phi_integ_2ndTrack->getTH1F()->Add( tmp );
    }


    ////////////////////////////////////////////////////////
    /// PT by wheel 2nd Tracks
    ////////////////////////////////////////////////////////
    sprintf( hname, "%s/2ND_TRACK_ONLY/dttf_09_pt_wh%s_2ndTrack", 
	     wheelpath_[wh].c_str(), wheel_[wh].c_str() );
    tmp = getTH1F(igetter,hname);
    if ( ! tmp ) {
      edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get TH1D "
						     << std::string(hname);
    } else {
      dttf_pt_integ_2ndTrack->getTH1F()->Add( tmp );
    }

    ////////////////////////////////////////////////////////
    /// Charge by wheel 2nd Tracks
    ////////////////////////////////////////////////////////
    sprintf( hname, "%s/2ND_TRACK_ONLY/dttf_10_charge_wh%s_2ndTrack", 
	     wheelpath_[wh].c_str(), wheel_[wh].c_str() );
    tmp = getTH1F(igetter,hname);
    if ( ! tmp ) {
      edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get TH1D "
						     << std::string(hname);
    } else {
      dttf_q_integ_2ndTrack->getTH1F()->Add( tmp );
    }

    ////////////////////////////////////////////////////////
    /// Quality by wheel 2nd Tracks
    ////////////////////////////////////////////////////////
    sprintf( hname, "%s/2ND_TRACK_ONLY/dttf_04_quality_wh%s_2ndTrack", 
	     wheelpath_[wh].c_str(), wheel_[wh].c_str() );
    tmp = getTH1F(igetter,hname);
    if ( ! tmp ) {
      edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get TH1D "
						     << std::string(hname);
    } else {
      for ( unsigned int qual = 1; qual < 8; ++qual ) {
	double binqual = tmp->GetBinContent( qual );
	dttf_quality_integ_2ndTrack->getTH1F()->AddBinContent( qual, binqual );
	dttf_quality_summary_2ndTrack->setBinContent( wh+1, qual, binqual );
      }
    }


  }

  ////////////////////////////////////////////////////////
  /// still bx: scaling integrals
  ////////////////////////////////////////////////////////
  for ( unsigned int bx = 1; bx < 4; ++bx ) {
    dttf_bx_integ->setBinContent( bx, wheelSumBx[bx] );
    dttf_bx_integ_2ndTrack->setBinContent( bx, wheelSumBx2nd[bx] );
  }

  ////////////////////////////////////////////////////////
  /// rescaling bx distributions
  ////////////////////////////////////////////////////////
  double scale = wheelSumBx[2];
  if ( scale > 0 ) {
    scale = 1./scale;
    normalize( dttf_bx_integ->getTH1F(), scale );
  }

  scale = wheelSumBx2nd[2];
  if ( scale > 0 ) {
    scale = 1./scale;
    normalize( dttf_bx_integ_2ndTrack->getTH1F(), scale );
  }


}



//--------------------------------------------------------
void L1TDTTFClient::setGMTsummary(DQMStore::IGetter &igetter)
{
  char hname[60];
  sprintf( hname, "%s/dttf_tracks_with_gmt_match", gmtpath_.c_str() );
  TH2F * gmt_match = getTH2F(igetter,hname);
  if ( ! gmt_match ) {
    edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get TH1D "
						   << std::string(hname);
    return;
  }



  sprintf( hname, "%s/dttf_tracks_without_gmt_match", gmtpath_.c_str() );
  TH2F * gmt_missed = getTH2F(igetter,hname);
  if ( ! gmt_missed ) {
    edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get TH1D "
						   << std::string(hname);
    return;
  }


  sprintf( hname, "%s/dttf_missing_tracks_in_gmt", gmtpath_.c_str() );
  TH2F * gmt_ghost = getTH2F(igetter,hname);
  if ( ! gmt_ghost ) {
    edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get TH1D "
						   << std::string(hname);
    return;
  }

  int match = gmt_match->Integral();
  int missed = gmt_missed->Integral();
  int ghost = gmt_ghost->Integral();
  float tot = match + missed + ghost;
  if ( tot > 0 ) {
    double val = ghost/tot;
    dttf_gmt_matching->setBinContent( 1, val );
    val = match/tot;
    dttf_gmt_matching->setBinContent( 2, val );
    val = missed/tot;
    dttf_gmt_matching->setBinContent( 3, val );
  }
}



//--------------------------------------------------------
TH1F * L1TDTTFClient::getTH1F(DQMStore::IGetter &igetter, const char * hname)
{

  MonitorElement * me = igetter.get(hname);
  if ( ! me ) {
    edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get ME "
						   << std::string(hname);
    return NULL;
  }

  return me->getTH1F();

}


//--------------------------------------------------------
TH2F * L1TDTTFClient::getTH2F(DQMStore::IGetter &igetter, const char * hname)
{

  MonitorElement * me = igetter.get(hname);
  if ( ! me ) {
    edm::LogError("L1TDTTFClient::makeSummary:ME") << "Failed to get ME "
						   << std::string(hname);
    return NULL;
  }

  return me->getTH2F();

}


//--------------------------------------------------------

//--------------------------------------------------------
void L1TDTTFClient::buildHighQualityPlot( DQMStore::IGetter &igetter, TH2F * occupancySummary,
					  MonitorElement * highQual_Summary,
					  const std::string & path )
{

  char hname[150];

  ////////////////////////////////////////////////////////
  /// high quality  TOBE IMPROVED
  ////////////////////////////////////////////////////////
  for ( unsigned int wh = 0; wh < 6 ; ++wh ) {
    sprintf( hname, path.c_str(), wheelpath_[wh].c_str(), wheel_[wh].c_str() );

    TH2F * quality = getTH2F(igetter, hname);
    if ( ! quality ) {
      edm::LogError("L1TDTTFClient::buildHighQualityPlot")
	<< "Failed to get TH2F " << std::string(hname);
    } else {

      for ( unsigned int sec = 1; sec < 13 ; ++sec ) {
	double denHighQual = occupancySummary->GetBinContent( wh+1, sec );
	double val = 0;
	if ( denHighQual > 0 ) {
	  for ( unsigned int qual = 4; qual < 8 ; ++qual ) {
	    val += quality->GetBinContent( qual, sec );
	  }
	  val /= denHighQual;
	}
	highQual_Summary->setBinContent( wh+1, sec, val );
      }
    }
  }
}

//--------------------------------------------------------
void L1TDTTFClient::buildPhiEtaPlotOFC( DQMStore::IGetter &igetter, MonitorElement * phi_eta_fine_integ,
					MonitorElement * phi_eta_coarse_integ,
					MonitorElement * phi_eta_integ,
					const std::string & path_fine,
					const std::string & path_coarse,
					int wh )
{

  char hname[150];
  sprintf( hname, path_fine.c_str(),
	   wheelpath_[wh].c_str(), wheel_[wh].c_str() );

  TH2F * phi_vs_eta_fine = getTH2F(igetter, hname);
  if ( ! phi_vs_eta_fine ) {
    edm::LogError("L1TDTTFClient::buildPhiEtaPloOtFC")
      << "Failed to get TH1D " << std::string(hname);
  }

  sprintf( hname, path_coarse.c_str(),
	   wheelpath_[wh].c_str(), wheel_[wh].c_str() );
  TH2F * phi_vs_eta_coarse = getTH2F(igetter,hname);
  if ( ! phi_vs_eta_coarse ) {
    edm::LogError("L1TDTTFClient::buildPhiEtaPlotOFC")
      << "Failed to get TH1D " << std::string(hname);
  }

  if ( ! phi_vs_eta_fine || ! phi_vs_eta_coarse ) {
    return;
  }

  for ( unsigned int phi = 1; phi < 145 ; ++phi ) {
    float start = 0;
    int nbins = 0;
    switch ( wh ) {
    case 0 : start = 0;  nbins = 18; break; // N2
    case 1 : start = 8;  nbins = 20; break; // N1
    case 2 : start = 22; nbins = 10; break; // N0
    case 3 : start = 22; nbins = 20; break; // P0
    case 4 : start = 36; nbins = 20; break; // P1
    case 5 : start = 46; nbins = 18; break; // P2
    default : start = 0; nbins = 0; break; // BOH
    }

    for ( int eta = 1; eta <= nbins ; ++eta ) {
      double setbin = eta + start;

      double valfine = phi_vs_eta_fine->GetBinContent( eta, phi )
	+ phi_eta_fine_integ->getBinContent( setbin, phi );

      double valcoarse = phi_vs_eta_coarse->GetBinContent( eta, phi )
	+ phi_eta_coarse_integ->getBinContent( setbin, phi );

      phi_eta_fine_integ->setBinContent( setbin, phi, valfine );
      phi_eta_coarse_integ->setBinContent( setbin, phi, valcoarse );
      phi_eta_integ->setBinContent( setbin, phi, valfine+valcoarse );

    }

    // double underflow_f = phi_vs_eta_fine->GetBinContent( 0, phi )
    //   + phi_eta_fine_integ->getBinContent( 1, phi );
    // phi_eta_fine_integ->setBinContent( 1, phi, underflow_f );
    // 
    // double underflow_c = phi_vs_eta_coarse->GetBinContent( 0, phi )
    //   + phi_eta_coarse_integ->getBinContent( 1, phi );
    // phi_eta_coarse_integ->setBinContent( 1, phi, underflow_c );
    // 
    // double overflow_f = phi_vs_eta_fine->GetBinContent( nbins+1, phi )
    //   + phi_eta_fine_integ->getBinContent( 64 );
    // phi_eta_fine_integ->setBinContent( 64, phi, overflow_f );
    // 
    // double overflow_c = phi_vs_eta_coarse->GetBinContent( nbins+1, phi )
    //   + phi_eta_coarse_integ->getBinContent( 64, phi );
    // phi_eta_coarse_integ->setBinContent( 64, phi, overflow_c );
    // 
    // double underflow = underflow_f + underflow_c;
    // phi_eta_integ->setBinContent( 1, phi, underflow );
    // 
    // double overflow = overflow_f + overflow_c;
    // phi_eta_integ->setBinContent( 64, phi, overflow );

  }

}


//--------------------------------------------------------
void L1TDTTFClient::buildPhiEtaPlotO( DQMStore::IGetter &igetter,MonitorElement * phi_eta_integ,
				      const std::string & path,
				      int wh )
{
  char hname[100];
  sprintf( hname, path.c_str(),
	   wheelpath_[wh].c_str(), wheel_[wh].c_str() );

  TH2F * phi_vs_eta = getTH2F(igetter, hname);
  if ( ! phi_vs_eta ) {
    edm::LogError("L1TDTTFClient::buildPhiEtaPlotO:ME") << "Failed to get TH1D "
						   << std::string(hname);
  } else {

    for ( unsigned int phi = 1; phi < 145 ; ++phi ) {
      float start = 0;
      int nbins = 0;
      switch ( wh ) {
      case 0 : start = 0;  nbins = 18; break; // N2
      case 1 : start = 8;  nbins = 20; break; // N1
      case 2 : start = 22; nbins = 10; break; // N0
      case 3 : start = 22; nbins = 20; break; // P0
      case 4 : start = 36; nbins = 20; break; // P1
      case 5 : start = 46; nbins = 18; break; // P2
      default : start = 0; nbins = 0; break; // BOH
      }

      for ( int eta = 1; eta <= nbins ; ++eta ) {
	double setbin = eta + start;
	double val = phi_vs_eta->GetBinContent( eta, phi )
	  + phi_eta_integ->getBinContent( setbin, phi );
	phi_eta_integ->setBinContent( setbin, phi, val );
      }

      double underflow = phi_vs_eta->GetBinContent( 0, phi )
	+ phi_eta_integ->getBinContent( 1, phi );
      phi_eta_integ->setBinContent( 1, phi, underflow );

      double overflow = phi_vs_eta->GetBinContent( nbins+1, phi )
	+ phi_eta_integ->getBinContent( 64 );
      phi_eta_integ->setBinContent( 64, phi, overflow );

    }
  }
}


// //--------------------------------------------------------
// void L1TDTTFClient::buildPhiEtaPlot( MonitorElement * phi_eta_integ,
// 				     const std::string & path ,
// 				     int wh)
// {
//   char hname[60];
//   sprintf( hname, "%s/dttf_phi_eta_wh%s",
// 	   wheelpath_[wh].c_str(), wheel_[wh].c_str() );
// 
//   TH2F * phi_vs_eta = getTH2F(hname);
//   if ( ! phi_vs_eta ) {
//     edm::LogError("L1TDTTFClient::buildPhiEtaPlot:ME") << "Failed to get TH1D "
// 						   << std::string(hname);
//   } else {
// 
// 
//     for ( unsigned int phi = 1; phi < 145 ; ++phi ) {
//       for ( unsigned int eta = 1; eta < 65 ; ++eta ) {
// 	double val = phi_vs_eta->GetBinContent( eta, phi )
// 	  + dttf_phi_eta_integ->getBinContent( eta, phi );
// 	dttf_phi_eta_integ->setBinContent( eta, phi, val );
//       }
// 
//     }
//   }
// }
// 
// 
// //--------------------------------------------------------
// void L1TDTTFClient::buildPhiEtaPlotFC( MonitorElement * phi_eta_fine_integ,
// 				       MonitorElement * phi_eta_coarse_integ,
// 				       MonitorElement * phi_eta_integ,
// 				       const std::string & path_fine,
// 				       const std::string & path_coarse,
// 				       int wh )
// {
// 
//   char hname[60];
// 
//   sprintf( hname, path_fine.c_str(),
// 	   wheelpath_[wh].c_str(), wheel_[wh].c_str() );
//   TH2F * phi_vs_eta_fine = getTH2F(hname);
//   if ( ! phi_vs_eta_fine ) {
//     edm::LogError("L1TDTTFClient::buildPhiEtaPlotFC")
//       << "Failed to get TH1D " << std::string(hname);
//   }
// 
// 
//   sprintf( hname, path_coarse.c_str(),
// 	   wheelpath_[wh].c_str(), wheel_[wh].c_str() );
//   TH2F * phi_vs_eta_coarse = getTH2F(hname);
//   if ( ! phi_vs_eta_coarse ) {
//     edm::LogError("L1TDTTFClient::buildPhiEtaPlotFC")
//       << "Failed to get TH1D " << std::string(hname);
//   }
// 
//   if ( ! phi_vs_eta_fine || ! phi_vs_eta_coarse ) {
//     return;
//   }
// 
//   for ( unsigned int phi = 1; phi < 145 ; ++phi ) {
//     for ( unsigned int eta = 1; eta < 65 ; ++eta ) {
// 
//       double valfine = phi_vs_eta_fine->GetBinContent( eta, phi )
// 	+ dttf_phi_eta_fine_integ->getBinContent( eta, phi );
//       dttf_phi_eta_fine_integ->setBinContent( eta, phi, valfine );
//       double valcoarse = phi_vs_eta_coarse->GetBinContent( eta, phi )
// 	+ dttf_phi_eta_coarse_integ->getBinContent( eta, phi );
//       dttf_phi_eta_coarse_integ->setBinContent( eta, phi, valcoarse );
// 
//       dttf_phi_eta_integ->setBinContent( eta, phi, valfine + valcoarse );
// 
//     }
// 
//   }
// }




//--------------------------------------------------------
void L1TDTTFClient::setWheelLabel(MonitorElement *me)
{
  me->setAxisTitle("Wheel", 1);
  me->setBinLabel(1, "N2", 1);
  me->setBinLabel(2, "N1", 1);
  me->setBinLabel(3, "N0", 1);
  me->setBinLabel(4, "P0", 1);
  me->setBinLabel(5, "P1", 1);
  me->setBinLabel(6, "P2", 1);

}


//--------------------------------------------------------
void L1TDTTFClient::setQualLabel(MonitorElement *me, int axis)
{

  if( axis == 1 )
    me->setAxisTitle("Quality", axis);
  me->setBinLabel(1, "T34", axis);
  me->setBinLabel(2, "T23/24", axis);
  me->setBinLabel(3, "T12/13/14", axis);
  me->setBinLabel(4, "T234", axis);
  me->setBinLabel(5, "T134", axis);
  me->setBinLabel(6, "T123/124", axis);
  me->setBinLabel(7, "T1234", axis);
}









