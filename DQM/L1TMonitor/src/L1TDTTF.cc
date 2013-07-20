/*
 * \file L1TDTTF.cc
 *
 * $Date: 2013/05/22 17:25:02 $
 * $Revision: 1.29 $
 * \author J. Berryhill
 *
 * $Log: L1TDTTF.cc,v $
 * Revision 1.29  2013/05/22 17:25:02  deguio
 * removing the use of catch(...)
 *
 * Revision 1.28  2010/11/02 13:58:20  gcodispo
 * Added protection against missing products
 *
 * Revision 1.27  2010/11/01 11:27:53  gcodispo
 * Cleaned up 2nd track sections
 *
 * Revision 1.26  2010/10/27 13:59:25  gcodispo
 * Changed name to 2nd track quality (same convention as for all tracks)
 *
 * Revision 1.25  2010/10/27 13:37:08  gcodispo
 * Changed name to 2nd track quality (same convention as for all tracks)
 *
 * Revision 1.24  2010/10/27 08:08:52  gcodispo
 * Graphic improvements (names, titles, labels...)
 *
 * Revision 1.23  2010/10/19 12:14:56  gcodispo
 * New DTTF DQM version
 * - added L1TDTTFClient in order to use proper normalization
 * - cleaned up most of the code, removed useless plots
 * - reduced overall number of bins from 118325 to 104763 plus saved 1920 bins from wrongly called L1TDTTPGClient
 * - added match with GMT inputremoved useless plots
 * - added eta fine fraction plots
 * - added quality distribution plots
 *
 * Revision 1.22  2009/11/19 15:09:18  puigh
 * modify beginJob
 *
 * Revision 1.21  2009/10/12 10:16:42  nuno
 * bug fix; letting the package compile again
 *
 * Revision 1.20  2009/08/03 21:11:22  lorenzo
 * added dttf phi and theta
 *
 * Revision 1.19  2008/07/29 14:18:27  wteo
 * updated and added more MEs
 *
 * Revision 1.15  2008/06/10 18:01:55  lorenzo
 * reduced n histos
 *
 * Revision 1.14  2008/05/09 16:42:27  ameyer
 * *** empty log message ***
 *
 * Revision 1.13  2008/04/30 08:44:21  lorenzo
 * new dttf source, not based on gmt record
 *
 * Revision 1.20  2008/03/20 19:38:25  berryhil
 *
 *
 * organized message logger
 *
 * Revision 1.19  2008/03/14 20:35:46  berryhil
 *
 *
 * stripped out obsolete parameter settings
 *
 * rpc tpg restored with correct dn access and dbe handling
 *
 * Revision 1.18  2008/03/12 17:24:24  berryhil
 *
 *
 * eliminated log files, truncated HCALTPGXana histo output
 *
 * Revision 1.17  2008/03/10 09:29:52  lorenzo
 * added MEs
 *
 * Revision 1.16  2008/03/01 00:40:00  lat
 * DQM core migration.
 *
 * $Log: L1TDTTF.cc,v $
 * Revision 1.29  2013/05/22 17:25:02  deguio
 * removing the use of catch(...)
 *
 * Revision 1.28  2010/11/02 13:58:20  gcodispo
 * Added protection against missing products
 *
 * Revision 1.27  2010/11/01 11:27:53  gcodispo
 * Cleaned up 2nd track sections
 *
 * Revision 1.26  2010/10/27 13:59:25  gcodispo
 * Changed name to 2nd track quality (same convention as for all tracks)
 *
 * Revision 1.25  2010/10/27 13:37:08  gcodispo
 * Changed name to 2nd track quality (same convention as for all tracks)
 *
 * Revision 1.24  2010/10/27 08:08:52  gcodispo
 * Graphic improvements (names, titles, labels...)
 *
 * Revision 1.23  2010/10/19 12:14:56  gcodispo
 * New DTTF DQM version
 * - added L1TDTTFClient in order to use proper normalization
 * - cleaned up most of the code, removed useless plots
 * - reduced overall number of bins from 118325 to 104763 plus saved 1920 bins from wrongly called L1TDTTPGClient
 * - added match with GMT inputremoved useless plots
 * - added eta fine fraction plots
 * - added quality distribution plots
 *
 * Revision 1.22  2009/11/19 15:09:18  puigh
 * modify beginJob
 *
 * Revision 1.21  2009/10/12 10:16:42  nuno
 * bug fix; letting the package compile again
 *
 * Revision 1.20  2009/08/03 21:11:22  lorenzo
 * added dttf phi and theta
 *
 * Revision 1.19  2008/07/29 14:18:27  wteo
 * updated and added more MEs
 *
 * Revision 1.15  2008/06/10 18:01:55  lorenzo
 * reduced n histos
 *
 * Revision 1.14  2008/05/09 16:42:27  ameyer
 * *** empty log message ***
 *
 * Revision 1.13  2008/04/30 08:44:21  lorenzo
 * new dttf source, not based on gmt record
 *
 * Revision 1.20  2008/03/20 19:38:25  berryhil
 *
 *
 * organized message logger
 *
 * Revision 1.19  2008/03/14 20:35:46  berryhil
 *
 *
 * stripped out obsolete parameter settings
 *
 * rpc tpg restored with correct dn access and dbe handling
 *
 * Revision 1.18  2008/03/12 17:24:24  berryhil
 *
 *
 * eliminated log files, truncated HCALTPGXana histo output
 *
 * Revision 1.17  2008/03/10 09:29:52  lorenzo
 * added MEs
 *
 * Revision 1.15  2008/01/22 18:56:01  muzaffar
 * include cleanup. Only for cc/cpp files
 *
 * Revision 1.14  2007/12/21 17:41:20  berryhil
 *
 *
 * try/catch removal
 *
 * Revision 1.13  2007/11/19 15:08:22  lorenzo
 * changed top folder name
 *
 * Revision 1.12  2007/08/15 18:56:25  berryhil
 *
 *
 * split histograms by bx; add Maiken's bx classifier plots
 *
 * Revision 1.11  2007/07/26 09:37:09  berryhil
 *
 *
 * set verbose false for all modules
 * set verbose fix for DTTPG tracks
 *
 * Revision 1.10  2007/07/25 09:03:58  berryhil
 *
 *
 * conform to DTTFFEDReader input tag.... for now
 *
 * Revision 1.9  2007/07/12 16:06:18  wittich
 * add simple phi output track histograms.
 * note that the label of this class is different than others
 * from the DTFFReader creates.
 *
 */

#include "DQM/L1TMonitor/interface/L1TDTTF.h"

/// base services
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

/// DT input
// #include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
// #include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
// #include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
// #include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"

/// output tracks
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h"

/// GMT
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"


/// GlobalMuon try
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"



//--------------------------------------------------------
L1TDTTF::L1TDTTF(const edm::ParameterSet& ps)
  : dttpgSource_( ps.getParameter< edm::InputTag >("dttpgSource") ),
    gmtSource_( ps.getParameter< edm::InputTag >("gmtSource") ),
    muonCollectionLabel_( ps.getParameter<edm::InputTag>("MuonCollection") ),
    l1tsubsystemfolder_( ps.getUntrackedParameter<std::string>("l1tSystemFolder",
							       "L1T/L1TDTTF")),
    online_( ps.getUntrackedParameter<bool>("online", true) ),
    verbose_( ps.getUntrackedParameter<bool>("verbose", false) )

{

  std::string trstring =
    dttpgSource_.label() + ":DATA:" + dttpgSource_.process();
  trackInputTag_ = edm::InputTag(trstring);

  /// Verbose?
  if ( verbose_ ) edm::LogInfo("L1TDTTF: constructor") << "Verbose enabled";

  /// Use DQMStore?
  dbe_ = NULL;
  if ( ps.getUntrackedParameter<bool>("DQMStore", false) ) {
    dbe_ = edm::Service<DQMStore>().operator->();
    dbe_->setVerbose(0);
    dbe_->setCurrentFolder(l1tsubsystemfolder_);
  }

  /// Use ROOT Output?
  if ( ps.getUntrackedParameter<bool>("disableROOToutput", false) ) {

    outputFile_ = "";

  } else {

    outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "");
    if ( ! outputFile_.empty() ) {
      edm::LogInfo("L1TDTTF: constructor")
	<< "L1T Monitoring histograms will be saved to " << outputFile_;
    }

  }

}



//--------------------------------------------------------
L1TDTTF::~L1TDTTF()
{
  /// Nothing to destroy
}



//--------------------------------------------------------
void L1TDTTF::beginJob(void)
{
 /// testing purposes
  nev_ = 0;
  nev_dttf_ = 0;
  nev_dttf_track2_ = 0;

  // get hold of back-end interface

  if ( dbe_ ) {

    std::string dttf_trk_folder = l1tsubsystemfolder_;

    char hname[100]; /// histo name
    char htitle[100]; /// histo title

    ///////////// OPTIMIZE
    float start = 0;
    float stop = 0;
    int nbins = 0;
    ///////////// OPTIMIZE

    /// DTTF Output (6 wheels)
    dbe_->setCurrentFolder(dttf_trk_folder);

    std::string wheelpath[6] = { "/02-WHEEL_N2",
				 "/03-WHEEL_N1",
				 "/04-WHEEL_N0",
				 "/05-WHEEL_P0",
				 "/06-WHEEL_P1",
				 "/07-WHEEL_P2" };


    char c_whn[6][3] = { "N2", "N1", "N0", "P0", "P1", "P2" };
    // char bxn [3][3] = { "N1", "0", "P1" };
    // char bxn[3][25] = {"/BX_NONZERO_ONLY/BX_N1", "", "/BX_NONZERO_ONLY/BX_P1"};

    for ( int iwh = 0; iwh < 6; ++iwh ) {

      bookEta( iwh, nbins, start, stop ); ///******************

      ////////////////////////////
      /// Per wheel summaries
      ////////////////////////////
      std::string dttf_trk_folder_wheel = dttf_trk_folder + wheelpath[iwh];
      dbe_->setCurrentFolder(dttf_trk_folder_wheel);

      /// number of  tracks per event per wheel
      sprintf(hname, "dttf_01_nTracksPerEvent_wh%s", c_whn[iwh]);
      sprintf(htitle, "Wheel %s - Number Tracks Per Event", c_whn[iwh]);
      dttf_nTracksPerEvent_wheel[iwh] = dbe_->book1D(hname, htitle,
						     10, 0.5, 10.5);
      dttf_nTracksPerEvent_wheel[iwh]->setAxisTitle("# tracks/event", 1);

      /// phi vs etafine - for each wheel
      sprintf(hname, "dttf_07_phi_vs_etaFine_wh%s", c_whn[iwh]);
      sprintf(htitle, "Wheel %s -   #eta-#phi DTTF Tracks occupancy (fine #eta only, unpacked values)", c_whn[iwh]);
      dttf_phi_eta_fine_wheel[iwh] = dbe_->book2D(hname, htitle,
						  nbins, start-0.5, stop-0.5,
						  144, -6, 138);
      // 144, -0.5, 143.5);
      
      dttf_phi_eta_fine_wheel[iwh]->setAxisTitle("#eta", 1);
      dttf_phi_eta_fine_wheel[iwh]->setAxisTitle("#phi", 2);

      /// phi vs etacoarse - for each wheel
      sprintf(hname, "dttf_08_phi_vs_etaCoarse_wh%s", c_whn[iwh]);
      sprintf(htitle, "Wheel %s -   #eta-#phi DTTF Tracks occupancy (coarse #eta only, unpacked values)", c_whn[iwh]);
      dttf_phi_eta_coarse_wheel[iwh] = dbe_->book2D(hname, htitle,
						    nbins, start-0.5, stop-0.5,
						    144, -6, 138);
      // 144, -0.5, 143.5);
      dttf_phi_eta_coarse_wheel[iwh]->setAxisTitle("#eta", 1);
      dttf_phi_eta_coarse_wheel[iwh]->setAxisTitle("#phi", 2);

      /////////////////////////////////////////////
      /// Per wheel summaries : 2ND_TRACK_ONLY
      std::string dttf_trk_folder_wheel_2ndtrack =
	dttf_trk_folder_wheel + "/2ND_TRACK_ONLY";
      dbe_->setCurrentFolder(dttf_trk_folder_wheel_2ndtrack);


      /// DTTF Tracks Quality distribution
      sprintf(hname, "dttf_04_quality_wh%s_2ndTrack", c_whn[iwh]);
      sprintf(htitle, "Wheel %s - 2nd Tracks Quality distribution", c_whn[iwh]);
      dttf_quality_wheel_2ndTrack[iwh] = dbe_->book1D(hname, htitle, 7, 1, 8);
      setQualLabel( dttf_quality_wheel_2ndTrack[iwh], 1);

      /// quality per wheel  2ND TRACK
      sprintf(hname, "dttf_05_quality_summary_wh%s_2ndTrack", c_whn[iwh]);
      sprintf(htitle, "Wheel %s - 2nd Tracks - Quality", c_whn[iwh]);
      dttf_quality_summary_wheel_2ndTrack[iwh] = dbe_->book2D(hname, htitle,
						      12, 1, 13, 7, 1, 8 );
      dttf_quality_summary_wheel_2ndTrack[iwh]->setAxisTitle("Sector", 1);
      setQualLabel( dttf_quality_summary_wheel_2ndTrack[iwh], 2);
      // dttf_quality_summary_wheel_2ndTrack[iwh]->setAxisTitle("Quality", 2);

      /// phi vs eta - for each wheel 2ND TRACK
      sprintf(hname, "dttf_06_phi_vs_eta_wh%s_2ndTrack", c_whn[iwh]);
      sprintf(htitle, "Wheel %s -   #eta-#phi Distribution of DTTF 2nd Tracks",
	      c_whn[iwh]);

      dttf_phi_eta_wheel_2ndTrack[iwh] = dbe_->book2D(hname, htitle,
	  					      nbins, start-0.5,stop-0.5,
                                                      144, -6, 138);
      // 144, -0.5, 143.5);
      dttf_phi_eta_wheel_2ndTrack[iwh]->setAxisTitle("#eta", 1);
      dttf_phi_eta_wheel_2ndTrack[iwh]->setAxisTitle("#phi", 2);



      /// DTTF Tracks #eta distribution (Packed values)
      sprintf(hname, "dttf_07_eta_wh%s_2ndTrack", c_whn[iwh]);
      sprintf(htitle, "Wheel %s - DTTF 2nd Tracks #eta distribution (Packed values)",
	      c_whn[iwh]);
      dttf_eta_wheel_2ndTrack[iwh] = dbe_->book1D(hname, htitle, 64, -0.5, 63.5);
      dttf_eta_wheel_2ndTrack[iwh]->setAxisTitle("#eta", 1);

      /// DTTF Tracks Phi distribution (Packed values)
      sprintf(hname, "dttf_08_phi_wh%s_2ndTrack", c_whn[iwh]);
      sprintf(htitle, "Wheel %s - DTTF 2nd Tracks Phi distribution (Packed values)",
	      c_whn[iwh]);
      dttf_phi_wheel_2ndTrack[iwh] = dbe_->book1D(hname, htitle, 144, -6, 138. );
      dttf_phi_wheel_2ndTrack[iwh]->setAxisTitle("#phi", 1);

      /// DTTF Tracks p_{T} distribution (Packed values)
      sprintf(hname, "dttf_09_pt_wh%s_2ndTrack", c_whn[iwh]);
      sprintf(htitle, "Wheel %s - DTTF 2nd Tracks p_{T} distribution (Packed values)",
	      c_whn[iwh]);
      dttf_pt_wheel_2ndTrack[iwh]  = dbe_->book1D(hname, htitle, 32, -0.5, 31.5);
      dttf_pt_wheel_2ndTrack[iwh]->setAxisTitle("p_{T}", 1);

      /// DTTF Tracks Charge distribution
      sprintf(hname, "dttf_10_charge_wh%s_2ndTrack", c_whn[iwh]);
      sprintf(htitle, "Wheel %s - DTTF 2nd Tracks Charge distribution", c_whn[iwh]);
      dttf_q_wheel_2ndTrack[iwh] = dbe_->book1D(hname, htitle, 2, -0.5, 1.5);
      dttf_q_wheel_2ndTrack[iwh]->setAxisTitle("Charge", 1);




       ///////////////////////////////////////////////////////
       /// Go in detailed subfolders
       ///////////////////////////////////////////////////////

      /// number of tracks per event folder
      std::string dttf_trk_folder_nTracksPerEvent = dttf_trk_folder_wheel + "/TracksPerEvent";
      dbe_->setCurrentFolder(dttf_trk_folder_nTracksPerEvent);

      for(int ise = 0; ise < 12; ++ise) {
	sprintf(hname, "dttf_nTracksPerEvent_wh%s_se%d", c_whn[iwh], ise+1);
	sprintf(htitle, "Wheel %s Sector %d - Number of Tracks Per Event",
		c_whn[iwh], ise+1);
	dttf_nTracksPerEv[iwh][ise] = dbe_->book1D(hname, htitle, 2, 0.5, 2.5);
	dttf_nTracksPerEv[iwh][ise]->setAxisTitle("# tracks/event", 1);
      }


      /// BX_SECTORS for each wheel
      std::string dttf_trk_folder_wh_bxsec_all =
	dttf_trk_folder_wheel + "/BX_BySector";
      dbe_->setCurrentFolder(dttf_trk_folder_wh_bxsec_all);

      for(int ise = 0; ise < 12; ++ise ) {
	sprintf(hname, "dttf_bx_wh%s_se%d", c_whn[iwh], ise+1);
	sprintf(htitle, "Wheel %s Sector %d - BX Distribution",
		c_whn[iwh], ise+1);
	dttf_bx[iwh][ise] = dbe_->book1D(hname, htitle, 3, -1.5, 1.5);
	dttf_bx[iwh][ise]->setAxisTitle("BX", 1);
      }

      std::string dttf_trk_folder_wh_bxsec_trk2 =
	dttf_trk_folder_wheel + "/BX_BySector/2ND_TRACK_ONLY";
      dbe_->setCurrentFolder(dttf_trk_folder_wh_bxsec_trk2);

      for(int ise = 0; ise < 12; ++ise ) {
	sprintf(hname, "dttf_bx_2ndTrack_wh%s_se%d", c_whn[iwh], ise+1);
	sprintf(htitle, "Wheel %s Sector %d - BX 2nd Tracks only",
		c_whn[iwh], ise+1);
	dttf_bx_2ndTrack[iwh][ise] = dbe_->book1D(hname, htitle, 3, -1.5, 1.5);
	dttf_bx_2ndTrack[iwh][ise]->setAxisTitle("BX", 1);
      }

      /// CHARGE folder
      std::string dttf_trk_folder_charge = dttf_trk_folder_wheel + "/Charge";
      dbe_->setCurrentFolder(dttf_trk_folder_charge);

      for(int ise = 0; ise < 12; ++ise) {
	sprintf(hname, "dttf_charge_wh%s_se%d", c_whn[iwh], ise+1);
	sprintf(htitle, "Wheel %s Sector %d - Packed Charge", c_whn[iwh], ise+1);
	dttf_q[iwh][ise] = dbe_->book1D(hname, htitle, 2, -0.5, 1.5);
	dttf_q[iwh][ise]->setAxisTitle("Charge", 1);
      }

      /// PT folder
      std::string dttf_trk_folder_pt = dttf_trk_folder_wheel + "/PT";
      dbe_->setCurrentFolder(dttf_trk_folder_pt);

      for(int ise = 0; ise < 12; ++ise ) {
	sprintf(hname, "dttf_pt_wh%s_se%d", c_whn[iwh], ise+1);
	sprintf(htitle, "Wheel %s Sector %d - Packed p_{T}",
		c_whn[iwh], ise + 1 );
	dttf_pt[iwh][ise]= dbe_->book1D(hname, htitle, 32, -0.5, 31.5);
	dttf_pt[iwh][ise]->setAxisTitle("p_{T}", 1);
      }

      /// PHI folder
      std::string dttf_trk_folder_phi = dttf_trk_folder_wheel + "/Phi";
      dbe_->setCurrentFolder(dttf_trk_folder_phi);

      for(int ise = 0; ise < 12; ++ise ) {
	sprintf(hname, "dttf_phi_wh%s_se%d", c_whn[iwh], ise+1);
	sprintf(htitle, "Wheel %s Sector %d - Packed Phi", c_whn[iwh], ise+1);
	dttf_phi[iwh][ise] = dbe_->book1D(hname, htitle, 144, -6, 138);
	dttf_phi[iwh][ise]->setAxisTitle("#phi", 1);
	//dttf_phi[iwh][ise] = dbe_->book1D(title,title, 32,-16.5, 15.5);
      }

      /// QUALITY folder
      std::string dttf_trk_folder_quality = dttf_trk_folder_wheel + "/Quality";
      dbe_->setCurrentFolder(dttf_trk_folder_quality);

      for(int ise = 0; ise < 12; ++ise){
	sprintf(hname, "dttf_qual_wh%s_se%d", c_whn[iwh], ise+1);
	sprintf(htitle, "Wheel %s Sector %d - Packed Quality",
		c_whn[iwh], ise+1);
	dttf_qual[iwh][ise] = dbe_->book1D(hname, htitle, 7, 1, 8);
	dttf_qual[iwh][ise]->setAxisTitle("Quality", 1);
	setQualLabel( dttf_qual[iwh][ise], 1 );
      }

      /// ETA folder
      std::string dttf_trk_folder_eta = dttf_trk_folder_wheel + "/Eta";
      dbe_->setCurrentFolder(dttf_trk_folder_eta);

      for (int ise = 0; ise < 12; ++ise ) {

	sprintf(hname, "dttf_eta_wh%s_se%d", c_whn[iwh], ise+1);
	sprintf(htitle, "Wheel %s Sector %d - Packed #eta",
		c_whn[iwh], ise+1);
	dttf_eta[iwh][ise] = dbe_->book1D(hname, htitle, 64, -0.5, 63.5);
	dttf_eta[iwh][ise]->setAxisTitle("#eta", 1);

      }

      /// ETA folder
      dttf_trk_folder_eta = dttf_trk_folder_wheel + "/EtaFineFraction";
      dbe_->setCurrentFolder(dttf_trk_folder_eta);

      for (int ise = 0; ise < 12; ++ise ) {

	sprintf(hname, "dttf_etaFine_fraction_wh%s_se%d", c_whn[iwh], ise+1);
	sprintf(htitle, "Wheel %s Sector %d - Eta Fine Fraction",
		c_whn[iwh], ise+1);
	dttf_eta_fine_fraction[iwh][ise] = dbe_->book1D(hname, htitle, 2, 0, 2);
	dttf_eta_fine_fraction[iwh][ise]->setAxisTitle("#eta", 1);
	dttf_eta_fine_fraction[iwh][ise]->setBinLabel(1, "fine", 1);
	dttf_eta_fine_fraction[iwh][ise]->setBinLabel(2, "coarse", 1);

      }

    }

    ///////////////////////////////////////////////////////
    /// integrated values: always packed
    ///////////////////////////////////////////////////////
    std::string dttf_trk_folder_inclusive = dttf_trk_folder + "/01-INCLUSIVE";
    dbe_->setCurrentFolder(dttf_trk_folder_inclusive);


    sprintf(hname, "dttf_01_nTracksPerEvent_integ");
    sprintf(htitle, "Number of DTTF Tracks Per Event");
    dttf_nTracksPerEvent_integ = dbe_->book1D(hname, htitle, 20, 0.5, 20.5);
    dttf_nTracksPerEvent_integ->setAxisTitle("# tracks/event", 1);

    ///////// ?????????
    // sprintf(hname, "dttf_10_qual_eta_distr");
    // sprintf(htitle, "DTTF Tracks Quality vs Eta Distribution");
    // dttf_qual_eta_integ = dbe_->book2D(hname, htitle, 64, 0, 64, 7, 1, 8);
    // setQualLabel( dttf_qual_eta_integ, 2);

    /// Only for online: occupancy summary - reset
    if ( online_ ) {
      sprintf(hname, "dttf_04_tracks_occupancy_by_lumi");
      sprintf(htitle, "DTTF Tracks in the last LumiSections");
      dttf_spare = dbe_->book2D(hname, htitle, 6, 0, 6, 12, 1, 13);
      setWheelLabel( dttf_spare );
      dttf_spare->setAxisTitle("Sector", 2);
      dttf_spare->getTH2F()->GetXaxis()->SetNdivisions(12);
    } else {

      sprintf(hname, "dttf_04_global_muons_request");
      sprintf(htitle, "Tracks compatible with a Global Muon in the Barrel");
      dttf_spare = dbe_->book1D(hname, htitle, 4, -0.5, 3.5 );
      dttf_spare->setBinLabel(1, "No tracks", 1);
      dttf_spare->setBinLabel(2, "No tracks but GM", 1);
      dttf_spare->setBinLabel(3, "Tracks wo GM", 1);
      dttf_spare->setBinLabel(4, "Tracks w GM", 1);

    }

    std::string dttf_trk_folder_integrated_gmt =
      dttf_trk_folder + "/08-GMT_MATCH";
    dbe_->setCurrentFolder(dttf_trk_folder_integrated_gmt);

    sprintf(hname, "dttf_tracks_with_gmt_match");
    sprintf(htitle, "DTTF Tracks With a Match in GMT");
    dttf_gmt_match = dbe_->book2D(hname, htitle, 6, 0., 6., 12, 1., 13.);
    setWheelLabel( dttf_gmt_match );

    sprintf(hname, "dttf_tracks_without_gmt_match");
    sprintf(htitle, "DTTF Tracks Without a Match in GMT");
    dttf_gmt_missed = dbe_->book2D(hname, htitle, 6, 0., 6., 12, 1., 13.);
    setWheelLabel( dttf_gmt_missed );

    sprintf(hname, "dttf_missing_tracks_in_gmt");
    sprintf(htitle, "GMT Tracks Without a Corresponding Track in DTTF");
    dttf_gmt_ghost = dbe_->book2D(hname, htitle, 5, -2, 3, 12, 1, 13.);

    dttf_gmt_ghost->setBinLabel(1, "N2", 1);
    dttf_gmt_ghost->setBinLabel(2, "N1", 1);
    dttf_gmt_ghost->setBinLabel(3, "N0/P0", 1);
    dttf_gmt_ghost->setBinLabel(4, "P1", 1);
    dttf_gmt_ghost->setBinLabel(5, "P2", 1);


    // sprintf(hname, "dttf_eta_phi_missing_tracks_in_gmt");
    // sprintf(htitle, "GMT Tracks Without a Corresponding Track in DTTF");
    // dttf_gmt_ghost_phys = dbe_->book2D(hname, htitle, 64, 0., 64., 144, 0., 144. );


  }

}



//--------------------------------------------------------
void L1TDTTF::endJob(void)
{
  if (verbose_) {
    edm::LogInfo("EndJob") << "L1TDTTF: end job....";
    edm::LogInfo("EndJob") << "analyzed " << nev_ << " events";
    edm::LogInfo("EndJob") << "containing at least one dttf track : "
			   << nev_dttf_;
    edm::LogInfo("EndJob") << "containing two dttf tracks : "
			   << nev_dttf_track2_;
  }
    
  if ( outputFile_.size() != 0  && dbe_ ) dbe_->save(outputFile_);

}



//--------------------------------------------------------
void L1TDTTF::analyze(const edm::Event& event,
		      const edm::EventSetup& eventSetup)
{


  if ( verbose_ )
    edm::LogInfo("L1TDTTF::Analyze::start") << "#################### START";

  /// counters
  ++nev_;
  memset( numTracks, 0, 72 * sizeof(int) );

  /// tracks handle
  edm::Handle<L1MuDTTrackContainer > myL1MuDTTrackContainer;
  try {
    event.getByLabel(trackInputTag_, myL1MuDTTrackContainer);
  } catch (cms::Exception& iException) {
    edm::LogError("L1TDTTF::analyze::DataNotFound")
      << "can't getByLabel L1MuDTTrackContainer with label " 
      << dttpgSource_.label() << ":DATA:" << dttpgSource_.process();
    return;
  }

  if ( !myL1MuDTTrackContainer.isValid() ) {
    edm::LogError("L1TDTTF::analyze::DataNotFound")
      << "can't find L1MuDTTrackContainer with label " 
      << dttpgSource_.label() << ":DATA:" << dttpgSource_.process();
    return;
  }

  L1MuDTTrackContainer::TrackContainer * trackContainer =
    myL1MuDTTrackContainer->getContainer();

  /// dttf counters
  if ( trackContainer->size() > 0 ) {
    ++nev_dttf_;
    if( trackContainer->size() > 1 ) ++nev_dttf_track2_;
  }

  ///////////////////////////
  /// selection for offline
  //////////////////////////
  bool accept = true;
  if ( ! online_ ) {

    try {

      edm::Handle<reco::MuonCollection> muons;
      event.getByLabel(muonCollectionLabel_, muons);

      accept = false;
      if ( muons.isValid() ) {
	for (reco::MuonCollection::const_iterator recoMu = muons->begin();
	     recoMu!=muons->end(); ++recoMu ) {
	  if ( fabs( recoMu->eta() ) < 1.4 ) {
	    if ( verbose_ ) {
	      edm::LogInfo("L1TDTTFClient::Analyze:GM") << "Found a global muon!";
	    }
	    accept = true;
	    break;
	  }

	}

	/// global muon selection plot
	if ( ! accept ) {
	  dttf_spare->Fill( trackContainer->size() ? 1 : 0 );

	  if ( verbose_ ) {
	    edm::LogInfo("L1TDTTFClient::Analyze:GM")
	      << "No global muons in this event!";
	  }

	} else {
	  dttf_spare->Fill( trackContainer->size() ? 2 : 3 );
	}

      } else {
	/// in case of problems accept all
	accept = true;
	edm::LogWarning("L1TDTTFClient::Analyze:GM")
	  <<  "Invalid MuonCollection with label "
	  << muonCollectionLabel_.label();
      }


    } catch (cms::Exception& iException) {
      /// in case of problems accept all
      accept = true;
      edm::LogError("DataNotFound") << "Unable to getByLabel MuonCollection with label "
				    << muonCollectionLabel_.label() ;
    }

  }


  ////////GMT
  std::vector<L1MuRegionalCand> gmtBx0DttfCandidates;

  try {

    edm::Handle<L1MuGMTReadoutCollection> pCollection;
    event.getByLabel(gmtSource_, pCollection);

    if ( !pCollection.isValid() ) {
      edm::LogError("DataNotFound") << "can't find L1MuGMTReadoutCollection with label "
				    << gmtSource_.label() ;
    }

    // get GMT readout collection
    L1MuGMTReadoutCollection const* gmtrc = pCollection.product();
    std::vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();

    std::vector<L1MuGMTReadoutRecord>::const_iterator RRItr;

    for ( RRItr = gmt_records.begin(); RRItr != gmt_records.end(); ++RRItr ) {
    
      std::vector<L1MuRegionalCand> dttfCands = RRItr->getDTBXCands();
      std::vector<L1MuRegionalCand>::iterator dttfCand;

      for( dttfCand = dttfCands.begin(); dttfCand != dttfCands.end();
	   ++dttfCand ) {

	if(dttfCand->empty()) continue;
	/// take only bx=0
	if ( RRItr->getBxInEvent() ) continue;

	//       dttf_gmt_ghost_phys->Fill( dttfCand->eta_packed(),
	// 				 dttfCand->phi_packed() );
	gmtBx0DttfCandidates.push_back( *dttfCand );

      }
    }

  } catch (cms::Exception& iException) {
    edm::LogError("DataNotFound") << "Unable to getByLabel L1MuGMTReadoutCollection with label "
				  << gmtSource_.label() ;
  }


  // fill MEs if all selections are passed
  if ( accept ) fillMEs( trackContainer, gmtBx0DttfCandidates );

  /// in Gmt but not in DTTF
  std::vector<L1MuRegionalCand>::iterator dttfCand;
  for( dttfCand = gmtBx0DttfCandidates.begin();
       dttfCand != gmtBx0DttfCandidates.end(); ++dttfCand ) {
    if( dttfCand->empty() ) continue;

    /// in phys values
    /// double phi= dttfCand->phiValue();
    /// int sector = 1 + (phi + 15)/30; /// in phys values
    int phi= dttfCand->phi_packed();
    int sector = 1 + (phi + 6)/12;
    if (sector > 12 ) sector -= 12;
    double eta = dttfCand->etaValue();

    int wheel = -3;
    if ( eta < -0.74 ) {
      wheel = -2;
    } else if ( eta < -0.3 ) {
      wheel = -1;

    } else if ( eta < 0.3 ) {
      wheel = 0;

    } else if ( eta < 0.74 ) {
      wheel = 1;
    } else {
      wheel = 2;
    }

    dttf_gmt_ghost->Fill( wheel, sector );
    // dttf_gmt_ghost_phys->Fill( dttfCand->eta_packed(),
    //                            dttfCand->phi_packed() );
  }


  /// Per event summaries
  int numTracksInt = 0;

  for ( int w = 0; w < 6; ++w ) {

    int numTracks_wh = 0;
    for ( int s = 0; s < 12; ++s ) {

      dttf_nTracksPerEv[w][s]->Fill( numTracks[w][s] );

      numTracks_wh += numTracks[w][s];

    }

    numTracksInt += numTracks_wh;
    dttf_nTracksPerEvent_wheel[w]->Fill( numTracks_wh );

  }

  dttf_nTracksPerEvent_integ->Fill( numTracksInt );



}




//--------------------------------------------------------
void L1TDTTF::fillMEs( std::vector<L1MuDTTrackCand> * trackContainer,
		       std::vector<L1MuRegionalCand> & gmtDttfCands )
{

  L1MuDTTrackContainer::TrackContainer::const_iterator track
    = trackContainer->begin();
  L1MuDTTrackContainer::TrackContainer::const_iterator trackEnd
    = trackContainer->end();

  for ( ; track != trackEnd; ++track ) {

    if ( verbose_ ) {
      edm::LogInfo("L1TDTTF::Analyze") << "bx = " << track->bx();
      edm::LogInfo("L1TDTTF::Analyze") << "quality (packed) = "
				       << track->quality_packed();
      edm::LogInfo("L1TDTTF::Analyze") << "pt      (packed) = "
				       << track->pt_packed()
				       << "  , pt  (GeV) = " << track->ptValue();
      edm::LogInfo("L1TDTTF::Analyze") << "phi     (packed) = "
				       << track->phi_packed()
				       << " , phi (rad) = " << track->phiValue();
      edm::LogInfo("L1TDTTF::Analyze") << "charge  (packed) = "
				       << track->charge_packed();
    }


    /// Forget  N0 with zero eta value for physical values
    if ( ( track->whNum() == -1 ) && ! track->eta_packed() ) {
      edm::LogInfo("L1TDTTF::Analyze") << "Skipping N0 with zero eta value";

      continue;
    }


    int bxindex = track->bx() + 1;
    int se = track->scNum(); /// from 0 to 11
    int sector = se + 1; /// from 1 to 12
    int whindex = track->whNum(); /// wh has possible values {-3,-2,-1,1,2,3}

    whindex = ( whindex < 0 ) ? whindex + 3 : whindex + 2; /// make wh2 go from 0 to 5

    if ( whindex < 0 || whindex > 5 ) {
      edm::LogError("L1TDTTF::Analyze::WHEEL_ERROR") << track->whNum()
						     << "(" << whindex << ")";
      continue;
    }

    if ( se < 0 || se > 11 ) {
      edm::LogError("L1TDTTF::Analyze::SECTOR_ERROR") << se;
      continue;
    }

    /// useful conversions

    /// calculate phi in physical coordinates: keep it int, set labels later
    // int phi_local = track->phi_packed();//range: 0 < phi_local < 31
    // if ( phi_local > 15 ) phi_local -= 32; //range: -16 < phi_local < 15

    // int phi_global = phi_local + se * 12; //range: -16 < phi_global < 147
    // if(phi_global < 0) phi_global += 144; //range: 0 < phi_global < 147
    // if(phi_global > 143) phi_global -= 144; //range: 0 < phi_global < 143
    // // float phi_phys = phi_global * 2.5 + 1.25;

    /// new attempt
    int phi_global = track->phi_packed();
    phi_global =  (phi_global > 15 ? phi_global - 32 : phi_global ) + se * 12;
    if ( phi_global < -6 ) phi_global += 144; //range: 0 < phi_global < 147
    if ( phi_global > 137 ) phi_global -= 144; //range: 0 < phi_global < 143

    // int eta_global = track->eta_packed();
    // int eta_global = track->eta_packed() - 32;
    // dttf_eta[bxindex][whindex][se]->Fill(eta_global);
    // float eta_phys = eta_global / 2.4 ;

    ///////////////////////////////////
    //// Starting BX distributions
    ///////////////////////////////////

    /// Fill per sector bx   WHEEL_%s/dttf_bx_wh%s
    dttf_bx[whindex][se]->Fill(track->bx());

    /// Fill per sector 2nd bx
    if( track->TrkTag() == 1 ) {

      /// WHEEL_%s/BX_SECTORS/TRACK_2_ONLY/dttf_bx_2ndTrack_wh%s_se%d
      dttf_bx_2ndTrack[whindex][se]->Fill(track->bx());

    }

    /////////////////////////////////////////
    //// use only bx=0 for for othe plots!
    /////////////////////////////////////////


    if ( bxindex == 1 ) {

      /// COUNTERS global
      ++numTracks[whindex][se];

      /// Fill per sector phi:    WHEEL_%s/BX_%d/dttf_phi_wh%s_se%d
      dttf_phi[whindex][se]->Fill(phi_global);

      /// Fill per sector quality WHEEL_%s/BX_%d/dttf_qual_wh%s_se%d
      dttf_qual[whindex][se]->Fill(track->quality_packed());

      /// Fill per sector pt      WHEEL_%s/BX_%d/dttf_pt_wh%s_se%d
      dttf_pt[whindex][se]->Fill(track->pt_packed());

      /// Fill per sector charge  WHEEL_%s/BX_%d/dttf_q_wh%s_se%d
      dttf_q[whindex][se]->Fill(track->charge_packed());


      /// Fill per sector eta     WHEEL_%s/BX_%d/dttf_eta_wh%s_se%d
      dttf_eta[whindex][se]->Fill( track->eta_packed() );

      if( track->isFineHalo() ) {
	
	dttf_eta_fine_fraction[whindex][se]->Fill( 0 );

	/// WHEEL_%s/dttf_phi_eta_wh%s
	dttf_phi_eta_fine_wheel[whindex]->Fill( track->eta_packed(), phi_global );

      } else {

	dttf_eta_fine_fraction[whindex][se]->Fill( 1 );

	/// WHEEL_%s/dttf_phi_eta_wh%s
	dttf_phi_eta_coarse_wheel[whindex]->Fill( track->eta_packed(), phi_global );
      }

      /// Only for online: INCLUSIVE/dttf_occupancy_summary_r
      if ( online_ ) {
	dttf_spare->Fill( whindex, sector );
      }

      ///////// ?????????
      // dttf_qual_eta_integ->Fill(track->eta_packed(), track->quality_packed());

      /// second track summary
      if ( track->TrkTag() == 1 ) {

	/// WHEEL_%s/dttf_phi_integ
	dttf_phi_wheel_2ndTrack[whindex]->Fill(phi_global);

	/// WHEEL_%s/dttf_pt_integ
	dttf_pt_wheel_2ndTrack[whindex]->Fill(track->pt_packed());

	/// WHEEL_%s/dttf_eta_integ
	dttf_eta_wheel_2ndTrack[whindex]->Fill(track->eta_packed());

	/// WHEEL_%s/dttf_qual_integ
	dttf_quality_wheel_2ndTrack[whindex]->Fill(track->quality_packed());

	/// WHEEL_%s/dttf_q_integ
	dttf_q_wheel_2ndTrack[whindex]->Fill(track->charge_packed());

	/// WHEEL_%s/dttf_quality_wh%s
	dttf_quality_summary_wheel_2ndTrack[whindex]->Fill( sector, track->quality_packed() );

	/// WHEEL_%s/dttf_phi_eta_wh%s
	dttf_phi_eta_wheel_2ndTrack[whindex]->Fill( track->eta_packed(), phi_global );

      }

      ///////// GMT
      bool match = false;
      std::vector<L1MuRegionalCand>::iterator dttfCand;
        /// gmt phi_packed() goes from 0 to 143
      unsigned int gmt_phi = ( phi_global < 0 ? phi_global + 144 : phi_global );

      for ( dttfCand = gmtDttfCands.begin(); dttfCand != gmtDttfCands.end();
	    ++dttfCand ) {

	/// calculate phi in physical coordinates: keep it int, set labels later
	if ( dttfCand->empty() ) continue;
	if ( ( dttfCand->phi_packed() == gmt_phi ) &&
	     dttfCand->quality_packed() == track->quality_packed() ) {
	  match = true;
	  dttfCand->reset();
	  break;
	}


      }

      if ( match ) {
	dttf_gmt_match->Fill( whindex, sector );
      } else {
	dttf_gmt_missed->Fill( whindex, sector );
      }

    }

  }

}


//--------------------------------------------------------
void L1TDTTF::setQualLabel(MonitorElement *me, int axis)
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

//--------------------------------------------------------
void L1TDTTF::setWheelLabel(MonitorElement *me)
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
void L1TDTTF::bookEta( int wh, int & nbins, float & start, float & stop )
{

  switch ( wh ) {
  case 0 : start = 0;  stop = 18; nbins = 18; break; // N2
  case 1 : start = 8;  stop = 28; nbins = 20; break; // N1
  case 2 : start = 22; stop = 32; nbins = 10; break; // N0
  case 3 : start = 22; stop = 42; nbins = 20; break; // P0
  case 4 : start = 36; stop = 56; nbins = 20; break; // P1
  case 5 : start = 46; stop = 64; nbins = 18; break; // P2
  default : start = 0; stop = 0;  nbins = 0;  break; // BOH
  }

}






//       ///////////////////////////////////////////////////////
//       /// dttf measures per wheel: per BX assignment
//       ///////////////////////////////////////////////////////

//       ///      for ( int ibx = 0; ibx < 3; ++ibx ) {
//       /// LEAVING ONLY BX0!!!
//       for ( int ibx = 1; ibx < 2; ++ibx ) {
// 	int tbx = ibx - 1;

// 	std::string dttf_trk_folder_bx = dttf_trk_folder_wheel + bxn[ibx];
// 	dbe_->setCurrentFolder(dttf_trk_folder_bx);

// 	/// QUALITY folder
// 	std::string dttf_trk_folder_quality = dttf_trk_folder_bx + "/Quality";
// 	dbe_->setCurrentFolder(dttf_trk_folder_quality);

// 	for(int ise = 0; ise < 12; ++ise){
// 	  sprintf(hname, "dttf_qual_bx%d_wh%s_se%d", tbx, c_whn[iwh], ise+1);
// 	  sprintf(htitle, "Packed Quality bx%d wh%s se%d", tbx, c_whn[iwh], ise+1);
// 	  dttf_qual[ibx][iwh][ise] = dbe_->book1D(hname, htitle, 8, -0.5, 7.5);
// 	  dttf_qual[ibx][iwh][ise]->setAxisTitle("Quality", 1);
// 	}

// 	/// PHI folder
// 	std::string dttf_trk_folder_phi = dttf_trk_folder_bx + "/Phi";
// 	dbe_->setCurrentFolder(dttf_trk_folder_phi);

// 	for(int ise = 0; ise < 12; ++ise ) {
// 	  sprintf(hname, "dttf_phi_bx%d_wh%s_se%d", tbx, c_whn[iwh], ise+1);
// 	  sprintf(htitle, "Packed Phi bx%d wh%s se%d", tbx, c_whn[iwh], ise+1);
// 	  dttf_phi[ibx][iwh][ise] = dbe_->book1D(hname, htitle,
// 						144, -0.5, 143.5);
// 	  dttf_phi[ibx][iwh][ise]->setAxisTitle("#phi", 1);
// 	  //dttf_phi[ibx][iwh][ise] = dbe_->book1D(title,title, 32,-16.5, 15.5);
// 	}

// 	/// ETA folder
// 	std::string dttf_trk_folder_eta = dttf_trk_folder_bx + "/Eta";
// 	dbe_->setCurrentFolder(dttf_trk_folder_eta);

// 	for (int ise = 0; ise < 12; ++ise ) {

// 	  // sprintf(hname, "dttf_eta_bx%d_wh%s_se%d", tbx, c_whn[iwh], ise+1);
// 	  // sprintf(htitle, "Packed Eta bx%d wh%s se%d", tbx, c_whn[iwh], ise+1);
// 	  // //dttf_eta[ibx][iwh][ise] = dbe_->book1D(hname,title,64,-32.5,32.5);//fix range and bin size!
// 	  // dttf_eta[ibx][iwh][ise] = dbe_->book1D(hname, htitle, 64, -0.5, 63.5);
// 	  // dttf_eta[ibx][iwh][ise]->setAxisTitle("#eta", 1);



// 	  sprintf(hname, "dttf_eta_fine_bx%d_wh%s_se%d", tbx, c_whn[iwh], ise+1);
// 	  sprintf(htitle, "Packed Eta Fine bx%d wh%s se%d", tbx, c_whn[iwh], ise+1);
// 	  dttf_eta_fine[ibx][iwh][ise] = dbe_->book1D(hname, htitle, 64, -0.5, 63.5);
// 	  dttf_eta_fine[ibx][iwh][ise]->setAxisTitle("#eta", 1);



// 	  sprintf(hname, "dttf_eta_coarse_bx%d_wh%s_se%d", tbx, c_whn[iwh], ise+1);
// 	  sprintf(htitle, "Packed Eta Coarse bx%d wh%s se%d", tbx, c_whn[iwh], ise+1);
// 	  dttf_eta_coarse[ibx][iwh][ise] = dbe_->book1D(hname, htitle, 64, -0.5, 63.5);
// 	  dttf_eta_coarse[ibx][iwh][ise]->setAxisTitle("#eta", 1);

// 	}

// 	/// PT folder
// 	std::string dttf_trk_folder_pt = dttf_trk_folder_bx + "/PT";
// 	dbe_->setCurrentFolder(dttf_trk_folder_pt);

// 	for(int ise = 0; ise < 12; ++ise ) {
// 	  sprintf(hname, "dttf_pt_bx%d_wh%s_se%d", tbx, c_whn[iwh], ise+1);
// 	  sprintf(htitle, "Packed PT bx%d wh%s se%d", tbx, c_whn[iwh], ise+1);
// 	  dttf_pt[ibx][iwh][ise]= dbe_->book1D(hname, htitle, 32, -0.5, 31.5);
// 	  dttf_pt[ibx][iwh][ise]->setAxisTitle("p_{T}", 1);
// 	}

// 	/// CHARGE folder
// 	std::string dttf_trk_folder_charge = dttf_trk_folder_bx + "/Charge";
// 	dbe_->setCurrentFolder(dttf_trk_folder_charge);

// 	for(int ise = 0; ise < 12; ++ise) {
// 	  sprintf(hname, "dttf_q_bx%d_wh%s_se%d", tbx, c_whn[iwh], ise+1);
// 	  sprintf(htitle, "Packed Charge  bx%d wh%s se%d", tbx, c_whn[iwh], ise+1);
// 	  dttf_q[ibx][iwh][ise] = dbe_->book1D(hname, htitle, 2, -0.5, 1.5);
// 	  dttf_q[ibx][iwh][ise]->setAxisTitle("Charge", 1);
// 	}

// 	/// number of tracks per event folder
// 	std::string dttf_trk_folder_nTracksPerEvent = dttf_trk_folder_bx+"/TracksPerEvent";
// 	dbe_->setCurrentFolder(dttf_trk_folder_nTracksPerEvent);

// 	for(int ise = 0; ise < 12; ++ise) {
// 	  sprintf(hname, "dttf_nTracksPerEvent_bx%d_wh%s_se%d", tbx, c_whn[iwh], ise+1);
// 	  sprintf(htitle, "Num Tracks Per Event bx%d wh%s se%d", tbx, c_whn[iwh], ise+1);
// 	  dttf_nTracksPerEv[ibx][iwh][ise] = dbe_->book1D(hname, htitle, 2, 0.5, 2.5);
// 	  dttf_nTracksPerEv[ibx][iwh][ise]->setAxisTitle("# tracks/event", 1);
// 	}

//       }








//--------------------------------------------------------
