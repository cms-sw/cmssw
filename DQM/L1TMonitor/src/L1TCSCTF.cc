/*
 * \file L1TCSCTF.cc
 *
 * \author J. Berryhill
 *
 */

#include "DQM/L1TMonitor/interface/L1TCSCTF.h"
#include "DQMServices/Core/interface/DQMStore.h"

// includes to fetch all reguired data products from the edm::Event
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"


using namespace std;
using namespace edm;

L1TCSCTF::L1TCSCTF(const ParameterSet& ps)
// if some piece of data is absent - configure corresponding source with 'null:'
//  : csctfSource_( ps.getParameter< InputTag >("csctfSource") )
  : gmtProducer( ps.getParameter< InputTag >("gmtProducer") ),
    lctProducer( ps.getParameter< InputTag >("lctProducer") ),
    trackProducer( ps.getParameter< InputTag >("trackProducer") ),
    statusProducer( ps.getParameter< InputTag >("statusProducer") ),
    mbProducer( ps.getParameter< InputTag >("mbProducer") )
{

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if(verbose_) edm::LogInfo("DataNotFound") << "L1TCSCTF: constructor...." << endl;

  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
  if ( outputFile_.size() != 0 )
    {
      edm::LogInfo("DataNotFound") << "L1T Monitoring histograms will be saved to " << outputFile_.c_str() << endl;
    }

  bool disable = ps.getUntrackedParameter<bool>("disableROOToutput", false);
  if(disable){
    outputFile_="";
  }

  gangedME11a_ = ps.getUntrackedParameter<bool>("gangedME11a", false);

  // instantiate standard on-fly SR LUTs from CSC TF emulator package
  bzero(srLUTs_,sizeof(srLUTs_));
  int endcap=1, sector=1; // assume SR LUTs are all same for every sector in either of endcaps
  bool TMB07=true; // specific TMB firmware
  // Create a dummy pset for SR LUTs
  edm::ParameterSet srLUTset;
  srLUTset.addUntrackedParameter<bool>("ReadLUTs", false);
  srLUTset.addUntrackedParameter<bool>("Binary",   false);
  srLUTset.addUntrackedParameter<std::string>("LUTPath", "./");
  for(int station=1,fpga=0; station<=4 && fpga<5; station++)
    {
      if(station==1)
	for(int subSector=0; subSector<2 && fpga<5; subSector++)
	  srLUTs_[fpga++] = new CSCSectorReceiverLUT(endcap, sector, subSector+1, station, srLUTset, TMB07);
      else
	srLUTs_[fpga++] = new CSCSectorReceiverLUT(endcap, sector, 0, station, srLUTset, TMB07);
    }

  //set Token(-s)
  edm::InputTag statusTag_(statusProducer.label(),statusProducer.instance());
  edm::InputTag corrlctsTag_(lctProducer.label(),lctProducer.instance());
  edm::InputTag tracksTag_(trackProducer.label(),trackProducer.instance());
  edm::InputTag dtStubsTag_(mbProducer.label(),  mbProducer.instance());
  edm::InputTag mbtracksTag_(trackProducer.label(),trackProducer.instance());

  gmtProducerToken_ = consumes<L1MuGMTReadoutCollection>(ps.getParameter< InputTag >("gmtProducer"));
  statusToken_ = consumes<L1CSCStatusDigiCollection>(statusTag_);
  corrlctsToken_ = consumes<CSCCorrelatedLCTDigiCollection>(corrlctsTag_);
  tracksToken_ = consumes<L1CSCTrackCollection>(tracksTag_);
  dtStubsToken_ = consumes<CSCTriggerContainer<csctf::TrackStub> >(dtStubsTag_);
  mbtracksToken_ = consumes<L1CSCTrackCollection>(mbtracksTag_);
}

L1TCSCTF::~L1TCSCTF()
{

  for(int i=0; i<5; i++)
    delete srLUTs_[i]; //free the array of pointers
}

void L1TCSCTF::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c){
}

void L1TCSCTF::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const&, edm::EventSetup const&)
{
  m_scalesCacheID  = -999;
  m_ptScaleCacheID = -999;

  nev_ = 0;
  ibooker.setCurrentFolder("L1T/L1TCSCTF");

      //  Error counting histogram:
      //  1) checks TF data integrity (error rate - first bin),
      //  2) monitors sychronization on input links (4 errors types: SE/SM/BX/AF; ORed for all time bins, links, and SPs),
      //  3) reports FMM status (if in any SP FMM status != "Ready" - fill the last bin)
  csctferrors = ibooker.book1D("CSCTF_errors","CSCTF Errors",6,0,6);
  csctferrors->setAxisTitle("Error type",1);
  csctferrors->setAxisTitle("Number of Errors",2);
  csctferrors->setBinLabel(1,"Corruptions",1);
  csctferrors->setBinLabel(2,"Synch. Err.",1);
  csctferrors->setBinLabel(3,"Synch. Mod.",1);
  csctferrors->setBinLabel(4,"BX mismatch",1);
  csctferrors->setBinLabel(5,"Time misalign.",1);
  csctferrors->setBinLabel(6,"FMM != Ready",1);

      //  Occupancy histogram Eta x Y, where Y:
      //  1) Phi_packed of input LCTs from 1st, 2nd, 3rd, and 4th stations
      //  2) Phi_packed of output tracks
      //  (all 12 SPs - 360 degree coveradge)
  csctfoccupancies = ibooker.book2D("CSCTF_occupancies", "CSCTF Occupancies", 64,-32,31,32,0,6.2);
  csctfoccupancies->setAxisTitle("#eta",1);
  csctfoccupancies->setAxisTitle("#phi",2);
  csctfoccupancies->setBinLabel( 1,"-2.5", 1);
  csctfoccupancies->setBinLabel( 8,"-2.1", 1);
  csctfoccupancies->setBinLabel(18,"-1.6", 1);
  csctfoccupancies->setBinLabel(26,"-1.2", 1);
  csctfoccupancies->setBinLabel(32,"-0.9", 1);
  csctfoccupancies->setBinLabel(33, "0.9", 1);
  csctfoccupancies->setBinLabel(39, "1.2", 1);
  csctfoccupancies->setBinLabel(47, "1.6", 1);
  csctfoccupancies->setBinLabel(57, "2.1", 1);
  csctfoccupancies->setBinLabel(64, "2.5", 1);

      // ... and for halo muons only
  csctfoccupancies_H = ibooker.book2D("CSCTF_occupancies_H", "CSCTF Halo Occupancies", 64,-32,31,32,0,6.2);
  csctfoccupancies_H->setAxisTitle("#eta",1);
  csctfoccupancies_H->setAxisTitle("#phi",2);
  csctfoccupancies_H->setBinLabel( 1,"-2.5", 1);
  csctfoccupancies_H->setBinLabel( 8,"-2.1", 1);
  csctfoccupancies_H->setBinLabel(18,"-1.6", 1);
  csctfoccupancies_H->setBinLabel(26,"-1.2", 1);
  csctfoccupancies_H->setBinLabel(32,"-0.9", 1);
  csctfoccupancies_H->setBinLabel(33, "0.9", 1);
  csctfoccupancies_H->setBinLabel(39, "1.2", 1);
  csctfoccupancies_H->setBinLabel(47, "1.6", 1);
  csctfoccupancies_H->setBinLabel(57, "2.1", 1);
  csctfoccupancies_H->setBinLabel(64, "2.5", 1);

      //haloDelEta12  = ibooker.book1D("CSCTF_Halo_Eta12", "#Delta #eta_{12} for Halo Muons", 40, -0.20,0.30);
      //haloDelEta112 = ibooker.book1D("CSCTF_Halo_Eta112","#Delta #eta_{112} for Halo Muons", 40, -0.20,0.30);
      //haloDelEta13  = ibooker.book1D("CSCTF_Halo_Eta13", "#Delta #eta_{13} for Halo Muons", 40, -0.20,0.30);
      //haloDelEta113 = ibooker.book1D("CSCTF_Halo_Eta113","#Delta #eta_{113} for Halo Muons", 40, -0.20,0.30);

      // Quality VS Mode
  trackModeVsQ = ibooker.book2D("CSCTF_Track_ModeVsQual","CSC Track Mode Vs Quality", 19, -0.5, 18.5, 4, 0, 4);
  trackModeVsQ->setAxisTitle("Track Type", 1);
  trackModeVsQ->setBinLabel(1,"No Track",1);
  trackModeVsQ->setBinLabel(2,"Bad Phi/Single",1);
  trackModeVsQ->setBinLabel(3,"ME1-2-3",1);
  trackModeVsQ->setBinLabel(4,"ME1-2-4",1);
  trackModeVsQ->setBinLabel(5,"ME1-3-4",1);
  trackModeVsQ->setBinLabel(6,"ME2-3-4",1);
  trackModeVsQ->setBinLabel(7,"ME1-2",1);
  trackModeVsQ->setBinLabel(8,"ME1-3",1);
  trackModeVsQ->setBinLabel(9,"ME2-3",1);
  trackModeVsQ->setBinLabel(10,"ME2-4",1);
  trackModeVsQ->setBinLabel(11,"ME3-4",1);
  trackModeVsQ->setBinLabel(12,"MB1-ME3",1);
  trackModeVsQ->setBinLabel(13,"MB1-ME2",1);
  trackModeVsQ->setBinLabel(14,"ME1-4",1);
  trackModeVsQ->setBinLabel(15,"MB1-ME1",1);
  trackModeVsQ->setBinLabel(16,"Halo Trigger",1);
  trackModeVsQ->setBinLabel(17,"MB1-ME1-2",1);
  trackModeVsQ->setBinLabel(18,"MB1-ME1-3",1);
  trackModeVsQ->setBinLabel(19,"MB1-ME2-3",1);

  trackModeVsQ->setAxisTitle("Quality",2);
  trackModeVsQ->setBinLabel(1,"0",2);
  trackModeVsQ->setBinLabel(2,"1",2);
  trackModeVsQ->setBinLabel(3,"2",2);
  trackModeVsQ->setBinLabel(4,"3",2);

      // Mode
  csctfTrackM = ibooker.book1D("CSCTF_Track_Mode","CSC Track Mode", 19, -0.5, 18.5);
  csctfTrackM->setAxisTitle("Track Type", 1);
  csctfTrackM->setBinLabel(1,"No Track",1);
  csctfTrackM->setBinLabel(2,"Bad Phi/Single",1);
  csctfTrackM->setBinLabel(3,"ME1-2-3",1);
  csctfTrackM->setBinLabel(4,"ME1-2-4",1);
  csctfTrackM->setBinLabel(5,"ME1-3-4",1);
  csctfTrackM->setBinLabel(6,"ME2-3-4",1);
  csctfTrackM->setBinLabel(7,"ME1-2",1);
  csctfTrackM->setBinLabel(8,"ME1-3",1);
  csctfTrackM->setBinLabel(9,"ME2-3",1);
  csctfTrackM->setBinLabel(10,"ME2-4",1);
  csctfTrackM->setBinLabel(11,"ME3-4",1);
  csctfTrackM->setBinLabel(12,"MB1-ME3",1);
  csctfTrackM->setBinLabel(13,"MB1-ME2",1);
  csctfTrackM->setBinLabel(14,"ME1-4",1);
  csctfTrackM->setBinLabel(15,"MB1-ME1",1);
  csctfTrackM->setBinLabel(16,"Halo Trigger",1);
  csctfTrackM->setBinLabel(17,"MB1-ME1-2",1);
  csctfTrackM->setBinLabel(18,"MB1-ME1-3",1);
  csctfTrackM->setBinLabel(19,"MB1-ME2-3",1);

      // Chamber Occupancy
  csctfChamberOccupancies = ibooker.book2D("CSCTF_Chamber_Occupancies","CSCTF Chamber Occupancies", 54, -0.05, 5.35, 10, -5.5, 4.5);
  csctfChamberOccupancies->setAxisTitle("Sector, (chambers 1-9 not labeled)",1);
  csctfChamberOccupancies->setBinLabel(1,"ME-4",2);
  csctfChamberOccupancies->setBinLabel(2,"ME-3",2);
  csctfChamberOccupancies->setBinLabel(3,"ME-2",2);
  csctfChamberOccupancies->setBinLabel(4,"ME-1b",2);
  csctfChamberOccupancies->setBinLabel(5,"ME-1a",2);
  csctfChamberOccupancies->setBinLabel(6,"ME+1a",2);
  csctfChamberOccupancies->setBinLabel(7,"ME+1b",2);
  csctfChamberOccupancies->setBinLabel(8,"ME+2",2);
  csctfChamberOccupancies->setBinLabel(9,"ME+3",2);
  csctfChamberOccupancies->setBinLabel(10,"ME+4",2);
  csctfChamberOccupancies->setBinLabel(1, "1",1);
  csctfChamberOccupancies->setBinLabel(10,"2",1);
  csctfChamberOccupancies->setBinLabel(19,"3",1);
  csctfChamberOccupancies->setBinLabel(28,"4",1);
  csctfChamberOccupancies->setBinLabel(37,"5",1);
  csctfChamberOccupancies->setBinLabel(46,"6",1);

      // Track Phi
  csctfTrackPhi = ibooker.book1D("CSCTF_Track_Phi", "CSCTF Track #phi",144,0,2*M_PI);
  csctfTrackPhi->setAxisTitle("Track #phi", 1);

      // Track Eta
  csctfTrackEta = ibooker.book1D("CSCTF_Track_Eta", "CSCTF Track #eta",64,-32,32);
  csctfTrackEta->setAxisTitle("Track #eta", 1);
  csctfTrackEta->setBinLabel( 1,"-2.5", 1);
  csctfTrackEta->setBinLabel( 8,"-2.1", 1);
  csctfTrackEta->setBinLabel(18,"-1.6", 1);
  csctfTrackEta->setBinLabel(26,"-1.2", 1);
  csctfTrackEta->setBinLabel(32,"-0.9", 1);
  csctfTrackEta->setBinLabel(33, "0.9", 1);
  csctfTrackEta->setBinLabel(39, "1.2", 1);
  csctfTrackEta->setBinLabel(47, "1.6", 1);
  csctfTrackEta->setBinLabel(57, "2.1", 1);
  csctfTrackEta->setBinLabel(64, "2.5", 1);

      // Track Eta Low Quality
  csctfTrackEtaLowQ = ibooker.book1D("CSCTF_Track_Eta_LowQ", "CSCTF Track #eta LQ",64,-32,32);
  csctfTrackEtaLowQ->setAxisTitle("Track #eta", 1);
  csctfTrackEtaLowQ->setBinLabel( 1,"-2.5", 1);
  csctfTrackEtaLowQ->setBinLabel( 8,"-2.1", 1);
  csctfTrackEtaLowQ->setBinLabel(18,"-1.6", 1);
  csctfTrackEtaLowQ->setBinLabel(26,"-1.2", 1);
  csctfTrackEtaLowQ->setBinLabel(32,"-0.9", 1);
  csctfTrackEtaLowQ->setBinLabel(33, "0.9", 1);
  csctfTrackEtaLowQ->setBinLabel(39, "1.2", 1);
  csctfTrackEtaLowQ->setBinLabel(47, "1.6", 1);
  csctfTrackEtaLowQ->setBinLabel(57, "2.1", 1);
  csctfTrackEtaLowQ->setBinLabel(64, "2.5", 1);


      // Track Eta High Quality
  csctfTrackEtaHighQ = ibooker.book1D("CSCTF_Track_Eta_HighQ", "CSCTF Track #eta HQ",64,-32,32);
  csctfTrackEtaHighQ->setAxisTitle("Track #eta", 1);
  csctfTrackEtaHighQ->setBinLabel( 1,"-2.5", 1);
  csctfTrackEtaHighQ->setBinLabel( 8,"-2.1", 1);
  csctfTrackEtaHighQ->setBinLabel(18,"-1.6", 1);
  csctfTrackEtaHighQ->setBinLabel(26,"-1.2", 1);
  csctfTrackEtaHighQ->setBinLabel(32,"-0.9", 1);
  csctfTrackEtaHighQ->setBinLabel(33, "0.9", 1);
  csctfTrackEtaHighQ->setBinLabel(39, "1.2", 1);
  csctfTrackEtaHighQ->setBinLabel(47, "1.6", 1);
  csctfTrackEtaHighQ->setBinLabel(57, "2.1", 1);
  csctfTrackEtaHighQ->setBinLabel(64, "2.5", 1);


      // Halo Phi
  csctfTrackPhi_H = ibooker.book1D("CSCTF_Track_Phi_H", "CSCTF Halo #phi",144,0,2*M_PI);
  csctfTrackPhi_H->setAxisTitle("Track #phi", 1);

      // Halo Eta
  csctfTrackEta_H = ibooker.book1D("CSCTF_Track_Eta_H", "CSCTF Halo #eta",64,-32,32);
  csctfTrackEta_H->setAxisTitle("Track #eta", 1);
  csctfTrackEta_H->setBinLabel( 1,"-2.5", 1);
  csctfTrackEta_H->setBinLabel( 8,"-2.1", 1);
  csctfTrackEta_H->setBinLabel(18,"-1.6", 1);
  csctfTrackEta_H->setBinLabel(26,"-1.2", 1);
  csctfTrackEta_H->setBinLabel(32,"-0.9", 1);
  csctfTrackEta_H->setBinLabel(33, "0.9", 1);
  csctfTrackEta_H->setBinLabel(39, "1.2", 1);
  csctfTrackEta_H->setBinLabel(47, "1.6", 1);
  csctfTrackEta_H->setBinLabel(57, "2.1", 1);
  csctfTrackEta_H->setBinLabel(64, "2.5", 1);

      // Track Timing
  csctfbx = ibooker.book2D("CSCTF_bx","CSCTF BX", 12,1,13, 7,-3,3) ;
  csctfbx->setAxisTitle("Sector (Endcap)", 1);
  csctfbx->setBinLabel( 1," 1 (+)",1);
  csctfbx->setBinLabel( 2," 2 (+)",1);
  csctfbx->setBinLabel( 3," 3 (+)",1);
  csctfbx->setBinLabel( 4," 4 (+)",1);
  csctfbx->setBinLabel( 5," 5 (+)",1);
  csctfbx->setBinLabel( 6," 6 (+)",1);
  csctfbx->setBinLabel( 7," 7 (-)",1);
  csctfbx->setBinLabel( 8," 8 (-)",1);
  csctfbx->setBinLabel( 9," 9 (-)",1);
  csctfbx->setBinLabel(10,"10 (-)",1);
  csctfbx->setBinLabel(11,"11 (-)",1);
  csctfbx->setBinLabel(12,"12 (-)",1);

  csctfbx->setAxisTitle("CSCTF BX", 2);
  csctfbx->setBinLabel( 1, "-3", 2);
  csctfbx->setBinLabel( 2, "-2", 2);
  csctfbx->setBinLabel( 3, "-1", 2);
  csctfbx->setBinLabel( 4, "-0", 2);
  csctfbx->setBinLabel( 5, " 1", 2);
  csctfbx->setBinLabel( 6, " 2", 2);
  csctfbx->setBinLabel( 7, " 3", 2);

      // Halo Timing
  csctfbx_H = ibooker.book2D("CSCTF_bx_H","CSCTF HALO BX", 12,1,13, 7,-3,3) ;
  csctfbx_H->setAxisTitle("Sector (Endcap)", 1);
  csctfbx_H->setBinLabel( 1," 1 (+)",1);
  csctfbx_H->setBinLabel( 2," 2 (+)",1);
  csctfbx_H->setBinLabel( 3," 3 (+)",1);
  csctfbx_H->setBinLabel( 4," 4 (+)",1);
  csctfbx_H->setBinLabel( 5," 5 (+)",1);
  csctfbx_H->setBinLabel( 6," 6 (+)",1);
  csctfbx_H->setBinLabel( 7," 7 (-)",1);
  csctfbx_H->setBinLabel( 8," 8 (-)",1);
  csctfbx_H->setBinLabel( 9," 9 (-)",1);
  csctfbx_H->setBinLabel(10,"10 (-)",1);
  csctfbx_H->setBinLabel(11,"11 (-)",1);
  csctfbx_H->setBinLabel(12,"12 (-)",1);

  csctfbx_H->setAxisTitle("CSCTF BX", 2);
  csctfbx_H->setBinLabel( 1, "-3", 2);
  csctfbx_H->setBinLabel( 2, "-2", 2);
  csctfbx_H->setBinLabel( 3, "-1", 2);
  csctfbx_H->setBinLabel( 4, "-0", 2);
  csctfbx_H->setBinLabel( 5, " 1", 2);
  csctfbx_H->setBinLabel( 6, " 2", 2);
  csctfbx_H->setBinLabel( 7, " 3", 2);

      // Number of Tracks Stubs
  cscTrackStubNumbers = ibooker.book1D("CSCTF_TrackStubs", "Number of Stubs in CSCTF Tracks", 5, 0, 5);
  cscTrackStubNumbers->setBinLabel( 1, "0", 1);
  cscTrackStubNumbers->setBinLabel( 2, "1", 1);
  cscTrackStubNumbers->setBinLabel( 3, "2", 1);
  cscTrackStubNumbers->setBinLabel( 4, "3", 1);
  cscTrackStubNumbers->setBinLabel( 5, "4", 1);

      // Number of Tracks
  csctfntrack = ibooker.book1D("CSCTF_ntrack","Number of CSCTracks found per event", 5, 0, 5 ) ;
  csctfntrack->setBinLabel( 1, "0", 1);
  csctfntrack->setBinLabel( 2, "1", 1);
  csctfntrack->setBinLabel( 3, "2", 1);
  csctfntrack->setBinLabel( 4, "3", 1);
  csctfntrack->setBinLabel( 5, "4", 1);
      //}

  char hname [200];
  char htitle[200];

  for(int i=0; i<12; i++) {

    sprintf(hname ,"DTstubsTimeTrackMenTimeArrival_%d",i+1);
    sprintf(htitle,"T_{track} - T_{DT stub} sector %d",i+1);

    DTstubsTimeTrackMenTimeArrival[i] = ibooker.book2D(hname,htitle, 7,-3,3, 2,1,3);
    DTstubsTimeTrackMenTimeArrival[i]->getTH2F()->SetMinimum(0);

    // axis makeup
    DTstubsTimeTrackMenTimeArrival[i]->setAxisTitle("bx_{CSC track} - bx_{DT stub}",1);
    DTstubsTimeTrackMenTimeArrival[i]->setAxisTitle("subsector",2);

    DTstubsTimeTrackMenTimeArrival[i]->setBinLabel(1,"-3",1);
    DTstubsTimeTrackMenTimeArrival[i]->setBinLabel(2,"-2",1);
    DTstubsTimeTrackMenTimeArrival[i]->setBinLabel(3,"-1",1);
    DTstubsTimeTrackMenTimeArrival[i]->setBinLabel(4, "0",1);
    DTstubsTimeTrackMenTimeArrival[i]->setBinLabel(5,"+1",1);
    DTstubsTimeTrackMenTimeArrival[i]->setBinLabel(6,"+2",1);
    DTstubsTimeTrackMenTimeArrival[i]->setBinLabel(7,"+3",1);

    DTstubsTimeTrackMenTimeArrival[i]->setBinLabel(1,"sub1",2);
    DTstubsTimeTrackMenTimeArrival[i]->setBinLabel(2,"sub2",2);

  }


  // NEW: CSC EVENT LCT PLOTS
  csctflcts = ibooker.book2D("CSCTF_LCT", "CSCTF LCTs", 12,1,13, 18,0,18);
  csctflcts->setAxisTitle("CSCTF LCT BX",1);
  csctflcts->setBinLabel(1,"1",1);
  csctflcts->setBinLabel(2,"2",1);
  csctflcts->setBinLabel(3,"3",1);
  csctflcts->setBinLabel(4,"4",1);
  csctflcts->setBinLabel(5,"5",1);
  csctflcts->setBinLabel(6,"6",1);
  csctflcts->setBinLabel(7,"7",1);
  csctflcts->setBinLabel(8,"8",1);
  csctflcts->setBinLabel(9,"9",1);
  csctflcts->setBinLabel(10,"10",1);
  csctflcts->setBinLabel(11,"11",1);
  csctflcts->setBinLabel(12,"12",1);

  int ihist = 0;
  for (int iEndcap = 0; iEndcap < 2; iEndcap++) {
    for (int iStation = 1; iStation < 5; iStation++) {
      for (int iRing = 1; iRing < 4; iRing++) {
        if (iStation != 1 && iRing > 2) continue;
        TString signEndcap="+";
        if(iEndcap==0) signEndcap="-";

        char lcttitle[200];
        snprintf(lcttitle,200,"ME%s%d/%d", signEndcap.Data(), iStation, iRing);
        if(ihist<=8){
                csctflcts -> setBinLabel(9-ihist,lcttitle,2);
        }
        else    csctflcts -> setBinLabel(ihist+1,lcttitle,2);

        ihist++;
      }
    }
  }


  // plots for ME1/1 chambers
  me11_lctStrip = ibooker.book1D("CSC_ME11_LCT_Strip", "CSC_ME11_LCT_Strip", 223, 0, 223);
  me11_lctStrip->setAxisTitle("Cathode HalfStrip, ME1/1", 1);

  me11_lctWire  = ibooker.book1D("CSC_ME11_LCT_Wire", "CSC_ME11_LCT_Wire", 112, 0, 112);
  me11_lctWire->setAxisTitle("Anode Wiregroup, ME1/1", 1);

  me11_lctLocalPhi = ibooker.book1D("CSC_ME11_LCT_LocalPhi", "CSC_ME11_LCT_LocalPhi", 200,0,1024);
  me11_lctLocalPhi ->setAxisTitle("LCT Local #it{#phi}, ME1/1", 1);

  me11_lctPackedPhi = ibooker.book1D("CSC_ME11_LCT_PackedPhi", "CSC_ME11_LCT_PackedPhi", 200,0,4096);
  me11_lctPackedPhi ->setAxisTitle("LCT Packed #it{#phi}, ME1/1",1);

  me11_lctGblPhi = ibooker.book1D("CSC_ME11_LCT_GblPhi", "CSC_ME11_LCT_GblPhi", 200, 0, 2*M_PI);
  me11_lctGblPhi ->setAxisTitle("LCT Global #it{#phi}, ME1/1", 1);

  me11_lctGblEta = ibooker.book1D("CSC_ME11_LCT_GblEta", "CSC_ME11_LCT_GblEta", 50, 0.9, 2.5);
  me11_lctGblEta ->setAxisTitle("LCT Global #eta, ME1/1", 1);


  // plots for ME4/2 chambers
  me42_lctGblPhi = ibooker.book1D("CSC_ME42_LCT_GblPhi", "CSC_ME42_LCT_GblPhi", 200, 0, 2*M_PI);
  me42_lctGblPhi ->setAxisTitle("LCT Global #it{#phi}, ME4/2", 1);

  me42_lctGblEta = ibooker.book1D("CSC_ME42_LCT_GblEta", "CSC_ME42_LCT_GblEta", 50, 0.9, 2.5);
  me42_lctGblEta ->setAxisTitle("LCT Global #eta, ME4/2", 1);

  //
  csc_strip_MEplus11= ibooker.book2D("csc_strip_MEplus11", "csc_strip_MEplus11", 36,1,37, 240,0,240);
  csc_strip_MEplus11->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEplus11->setAxisTitle("ME+1/1", 1);
  csc_strip_MEplus12= ibooker.book2D("csc_strip_MEplus12", "csc_strip_MEplus12", 36,1,37, 240,0,240);
  csc_strip_MEplus12->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEplus12->setAxisTitle("ME+1/2", 1);
  csc_strip_MEplus13= ibooker.book2D("csc_strip_MEplus13", "csc_strip_MEplus13", 36,1,37, 240,0,240);
  csc_strip_MEplus13->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEplus13->setAxisTitle("ME+1/3", 1);
  csc_strip_MEplus21= ibooker.book2D("csc_strip_MEplus21", "csc_strip_MEplus21", 18,1,19, 240,0,240);
  csc_strip_MEplus21->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEplus21->setAxisTitle("ME+2/1", 1);
  csc_strip_MEplus22= ibooker.book2D("csc_strip_MEplus22", "csc_strip_MEplus22", 36,1,37, 240,0,240);
  csc_strip_MEplus22->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEplus22->setAxisTitle("ME+2/2", 1);
  csc_strip_MEplus31= ibooker.book2D("csc_strip_MEplus31", "csc_strip_MEplus31", 18,1,19, 240,0,240);
  csc_strip_MEplus31->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEplus31->setAxisTitle("ME+3/1", 1);
  csc_strip_MEplus32= ibooker.book2D("csc_strip_MEplus32", "csc_strip_MEplus32", 36,1,37, 240,0,240);
  csc_strip_MEplus32->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEplus32->setAxisTitle("ME+3/2", 1);
  csc_strip_MEplus41= ibooker.book2D("csc_strip_MEplus41", "csc_strip_MEplus41", 18,1,19, 240,0,240);
  csc_strip_MEplus41->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEplus41->setAxisTitle("ME+4/1", 1);
  csc_strip_MEplus42= ibooker.book2D("csc_strip_MEplus42", "csc_strip_MEplus42", 36,1,37, 240,0,240);
  csc_strip_MEplus42->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEplus42->setAxisTitle("ME+4/2", 1);

  csc_strip_MEminus11= ibooker.book2D("csc_strip_MEminus11", "csc_strip_MEminus11", 36,1,37, 240,0,240);
  csc_strip_MEminus11->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEminus11->setAxisTitle("ME-1/1", 1);
  csc_strip_MEminus12= ibooker.book2D("csc_strip_MEminus12", "csc_strip_MEminus12", 36,1,37, 240,0,240);
  csc_strip_MEminus12->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEminus12->setAxisTitle("ME-1/2", 1);
  csc_strip_MEminus13= ibooker.book2D("csc_strip_MEminus13", "csc_strip_MEminus13", 36,1,37, 240,0,240);
  csc_strip_MEminus13->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEminus13->setAxisTitle("ME-1/3", 1);
  csc_strip_MEminus21= ibooker.book2D("csc_strip_MEminus21", "csc_strip_MEminus21", 18,1,19, 240,0,240);
  csc_strip_MEminus21->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEminus21->setAxisTitle("ME-2/1", 1);
  csc_strip_MEminus22= ibooker.book2D("csc_strip_MEminus22", "csc_strip_MEminus22", 36,1,37, 240,0,240);
  csc_strip_MEminus22->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEminus22->setAxisTitle("ME-2/2", 1);
  csc_strip_MEminus31= ibooker.book2D("csc_strip_MEminus31", "csc_strip_MEminus31", 18,1,19, 240,0,240);
  csc_strip_MEminus31->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEminus31->setAxisTitle("ME-3/1", 1);
  csc_strip_MEminus32= ibooker.book2D("csc_strip_MEminus32", "csc_strip_MEminus32", 36,1,37, 240,0,240);
  csc_strip_MEminus32->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEminus32->setAxisTitle("ME-3/2", 1);
  csc_strip_MEminus41= ibooker.book2D("csc_strip_MEminus41", "csc_strip_MEminus41", 18,1,19, 240,0,240);
  csc_strip_MEminus41->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEminus41->setAxisTitle("ME-4/1", 1);
  csc_strip_MEminus42= ibooker.book2D("csc_strip_MEminus42", "csc_strip_MEminus42", 36,1,37, 240,0,240);
  csc_strip_MEminus42->setAxisTitle("Cathode HalfStrip", 2);
  csc_strip_MEminus42->setAxisTitle("ME-4/2", 1);

  csc_wire_MEplus11= ibooker.book2D("csc_wire_MEplus11", "csc_wire_MEplus11", 36,1,37, 120,0,120);
  csc_wire_MEplus11->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEplus11->setAxisTitle("ME+1/1", 1);
  csc_wire_MEplus12= ibooker.book2D("csc_wire_MEplus12", "csc_wire_MEplus12", 36,1,37, 120,0,120);
  csc_wire_MEplus12->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEplus12->setAxisTitle("ME+1/2", 1);
  csc_wire_MEplus13= ibooker.book2D("csc_wire_MEplus13", "csc_wire_MEplus13", 36,1,37, 120,0,120);
  csc_wire_MEplus13->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEplus13->setAxisTitle("ME+1/3", 1);
  csc_wire_MEplus21= ibooker.book2D("csc_wire_MEplus21", "csc_wire_MEplus21", 18,1,19, 120,0,120);
  csc_wire_MEplus21->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEplus21->setAxisTitle("ME+2/1", 1);
  csc_wire_MEplus22= ibooker.book2D("csc_wire_MEplus22", "csc_wire_MEplus22", 36,1,37, 120,0,120);
  csc_wire_MEplus22->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEplus22->setAxisTitle("ME+2/2", 1);
  csc_wire_MEplus31= ibooker.book2D("csc_wire_MEplus31", "csc_wire_MEplus31", 18,1,19, 120,0,120);
  csc_wire_MEplus31->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEplus31->setAxisTitle("ME+3/1", 1);
  csc_wire_MEplus32= ibooker.book2D("csc_wire_MEplus32", "csc_wire_MEplus32", 36,1,37, 120,0,120);
  csc_wire_MEplus32->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEplus32->setAxisTitle("ME+3/2", 1);
  csc_wire_MEplus41= ibooker.book2D("csc_wire_MEplus41", "csc_wire_MEplus41", 18,1,19, 120,0,120);
  csc_wire_MEplus41->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEplus41->setAxisTitle("ME+4/1", 1);
  csc_wire_MEplus42= ibooker.book2D("csc_wire_MEplus42", "csc_wire_MEplus42", 36,1,37, 120,0,120);
  csc_wire_MEplus42->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEplus42->setAxisTitle("ME+4/2", 1);

  csc_wire_MEminus11= ibooker.book2D("csc_wire_MEminus11", "csc_wire_MEminus11", 36,1,37, 120,0,120);
  csc_wire_MEminus11->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEminus11->setAxisTitle("ME-1/1", 1);
  csc_wire_MEminus12= ibooker.book2D("csc_wire_MEminus12", "csc_wire_MEminus12", 36,1,37, 120,0,120);
  csc_wire_MEminus12->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEminus12->setAxisTitle("ME-1/2", 1);
  csc_wire_MEminus13= ibooker.book2D("csc_wire_MEminus13", "csc_wire_MEminus13", 36,1,37, 120,0,120);
  csc_wire_MEminus13->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEminus13->setAxisTitle("ME-1/3", 1);
  csc_wire_MEminus21= ibooker.book2D("csc_wire_MEminus21", "csc_wire_MEminus21", 18,1,19, 120,0,120);
  csc_wire_MEminus21->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEminus21->setAxisTitle("ME-2/1", 1);
  csc_wire_MEminus22= ibooker.book2D("csc_wire_MEminus22", "csc_wire_MEminus22", 36,1,37, 120,0,120);
  csc_wire_MEminus22->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEminus22->setAxisTitle("ME-2/2", 1);
  csc_wire_MEminus31= ibooker.book2D("csc_wire_MEminus31", "csc_wire_MEminus31", 18,1,19, 120,0,120);
  csc_wire_MEminus31->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEminus31->setAxisTitle("ME-3/1", 1);
  csc_wire_MEminus32= ibooker.book2D("csc_wire_MEminus32", "csc_wire_MEminus32", 36,1,37, 120,0,120);
  csc_wire_MEminus32->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEminus32->setAxisTitle("ME-3/2", 1);
  csc_wire_MEminus41= ibooker.book2D("csc_wire_MEminus41", "csc_wire_MEminus41", 18,1,19, 120,0,120);
  csc_wire_MEminus41->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEminus41->setAxisTitle("ME-4/1", 1);
  csc_wire_MEminus42= ibooker.book2D("csc_wire_MEminus42", "csc_wire_MEminus42", 36,1,37, 120,0,120);
  csc_wire_MEminus42->setAxisTitle("Anode Wiregroup", 2);
  csc_wire_MEminus42->setAxisTitle("ME-4/2", 1);



  for(int cscid = 1; cscid < 37; cscid++){
        char bxtitle[100];
        sprintf(bxtitle,"%d", cscid);

        csc_strip_MEplus11 ->setBinLabel(cscid,bxtitle,1);
        csc_strip_MEplus12 ->setBinLabel(cscid,bxtitle,1);
        csc_strip_MEplus13 ->setBinLabel(cscid,bxtitle,1);
        csc_strip_MEplus22 ->setBinLabel(cscid,bxtitle,1);
        csc_strip_MEplus32 ->setBinLabel(cscid,bxtitle,1);
        csc_strip_MEplus42 ->setBinLabel(cscid,bxtitle,1);

        csc_strip_MEminus11 ->setBinLabel(cscid,bxtitle,1);
        csc_strip_MEminus12 ->setBinLabel(cscid,bxtitle,1);
        csc_strip_MEminus13 ->setBinLabel(cscid,bxtitle,1);
        csc_strip_MEminus22 ->setBinLabel(cscid,bxtitle,1);
        csc_strip_MEminus32 ->setBinLabel(cscid,bxtitle,1);
        csc_strip_MEminus42 ->setBinLabel(cscid,bxtitle,1);

        csc_wire_MEplus11 ->setBinLabel(cscid,bxtitle,1);
        csc_wire_MEplus12 ->setBinLabel(cscid,bxtitle,1);
        csc_wire_MEplus13 ->setBinLabel(cscid,bxtitle,1);
        csc_wire_MEplus22 ->setBinLabel(cscid,bxtitle,1);
        csc_wire_MEplus32 ->setBinLabel(cscid,bxtitle,1);
        csc_wire_MEplus42 ->setBinLabel(cscid,bxtitle,1);

        csc_wire_MEminus11 ->setBinLabel(cscid,bxtitle,1);
        csc_wire_MEminus12 ->setBinLabel(cscid,bxtitle,1);
        csc_wire_MEminus13 ->setBinLabel(cscid,bxtitle,1);
        csc_wire_MEminus22 ->setBinLabel(cscid,bxtitle,1);
        csc_wire_MEminus32 ->setBinLabel(cscid,bxtitle,1);
        csc_wire_MEminus42 ->setBinLabel(cscid,bxtitle,1);
  }


  for(int cscid = 1; cscid < 19; cscid++){
        char bxtitle[100];
        sprintf(bxtitle,"%d", cscid);

        csc_strip_MEplus21 ->setBinLabel(cscid,bxtitle,1);
        csc_strip_MEplus31 ->setBinLabel(cscid,bxtitle,1);
        csc_strip_MEplus41 ->setBinLabel(cscid,bxtitle,1);

        csc_strip_MEminus21 ->setBinLabel(cscid,bxtitle,1);
        csc_strip_MEminus31 ->setBinLabel(cscid,bxtitle,1);
        csc_strip_MEminus41 ->setBinLabel(cscid,bxtitle,1);

        csc_wire_MEplus21 ->setBinLabel(cscid,bxtitle,1);
        csc_wire_MEplus31 ->setBinLabel(cscid,bxtitle,1);
        csc_wire_MEplus41 ->setBinLabel(cscid,bxtitle,1);

        csc_wire_MEminus21 ->setBinLabel(cscid,bxtitle,1);
        csc_wire_MEminus31 ->setBinLabel(cscid,bxtitle,1);
        csc_wire_MEminus41 ->setBinLabel(cscid,bxtitle,1);

  }

}

void L1TCSCTF::analyze(const Event& e, const EventSetup& c)
{

  if( c.get< L1MuTriggerScalesRcd > ().cacheIdentifier() != m_scalesCacheID ||
      c.get< L1MuTriggerPtScaleRcd >().cacheIdentifier() != m_ptScaleCacheID ){

    ESHandle< L1MuTriggerScales > scales;
    c.get< L1MuTriggerScalesRcd >().get(scales);
    ts = scales.product();
    ESHandle< L1MuTriggerPtScale > ptscales;
    c.get< L1MuTriggerPtScaleRcd >().get(ptscales);
    tpts = ptscales.product();
    m_scalesCacheID  = c.get< L1MuTriggerScalesRcd  >().cacheIdentifier();
    m_ptScaleCacheID = c.get< L1MuTriggerPtScaleRcd >().cacheIdentifier();

    edm::LogInfo("L1TCSCTF")  << "Changing triggerscales and triggerptscales...";
  }

  int NumCSCTfTracksRep = 0;
  nev_++;
  if(verbose_) edm::LogInfo("DataNotFound") << "L1TCSCTF: analyze...." << endl;

  edm::Handle<L1MuGMTReadoutCollection> pCollection;
  if( gmtProducer.label() != "null" )
    { // GMT block
      e.getByToken(gmtProducerToken_, pCollection);
      if (!pCollection.isValid())
        {
          edm::LogInfo("DataNotFound") << "can't find L1MuGMTReadoutCollection with label ";  // << csctfSource_.label() ;
          return;
        }

      L1MuGMTReadoutCollection const* gmtrc = pCollection.product();
      vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
      vector<L1MuGMTReadoutRecord>::const_iterator RRItr;

      // Look if the readout window contains one (and only one CSC cands)
      // to make it simpler I reject events with more than a CSC cand in the
      // same readout window

      // count non-empty candidates in this bx
      int bxWindow = 0;
      int nCands   = 0;

      for( RRItr = gmt_records.begin(); RRItr != gmt_records.end(); RRItr++ ) {
        bxWindow++;

        // get the csc candidates
        vector<L1MuRegionalCand> INPCands = RRItr->getCSCCands();
        vector<L1MuRegionalCand>::const_iterator INPItr;

        BxInEvent_ = 0;
        isCSCcand_ = false;
        int  nCandsBx  = 0;

        for( INPItr = INPCands.begin(); INPItr != INPCands.end(); ++INPItr ) {
          if(!INPItr->empty())
            {
              nCandsBx++;
              nCands++;
              BxInEvent_ = RRItr->getBxInEvent();
              if (verbose_) edm::LogInfo("DataNotFound") << "cand " << nCandsBx << " -> assigned CSCTF bx: " << INPItr->bx() << endl;
            }
        }
        if (verbose_)
          if(nCandsBx) edm::LogInfo("DataNotFound") << nCandsBx << " cands in bx: " << BxInEvent_ << endl;
      }

      if (nCands != 1) return;
      else isCSCcand_ = true;
      if (verbose_) edm::LogInfo("DataNotFound") << "bxWindow: " << bxWindow << endl;

      int ncsctftrack = 0;
      if (verbose_)
        {
          edm::LogInfo("DataNotFound") << "\tCSCTFCand ntrack " << ncsctftrack << endl;
        }
    } // end of GMT block

  L1ABXN = -999;
  if( statusProducer.label() != "null" )
    {
      edm::Handle<L1CSCStatusDigiCollection> status;
      e.getByToken(statusToken_, status);
      bool integrity=status->first, se=false, sm=false, bx=false, af=false, fmm=false;
      int nStat = 0;

      for(std::vector<L1CSCSPStatusDigi>::const_iterator stat=status->second.begin(); stat!=status->second.end(); stat++)
        {
          se |= stat->SEs()&0xFFF;
          sm |= stat->SMs()&0xFFF;
          bx |= stat->BXs()&0xFFF;
          af |= stat->AFs()&0xFFF;
          fmm|= stat->FMM()!=8;

          if(stat->VPs() != 0)
            {
              L1ABXN += stat->BXN();
              nStat++;
            }
        }
      // compute the average
      if(nStat!=0) L1ABXN /= nStat;
      if(integrity) csctferrors->Fill(0.5);
      if(se)        csctferrors->Fill(1.5);
      if(sm)        csctferrors->Fill(2.5);
      if(bx)        csctferrors->Fill(3.5);
      if(af)        csctferrors->Fill(4.5);
      if(fmm)       csctferrors->Fill(5.5);
    }

  if( lctProducer.label() != "null" )
    {
      edm::ESHandle<CSCGeometry> pDD;
      c.get<MuonGeometryRecord>().get( pDD );
      CSCTriggerGeometry::setGeometry(pDD);

      edm::Handle<CSCCorrelatedLCTDigiCollection> corrlcts;
      e.getByToken(corrlctsToken_, corrlcts);

      for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator csc=corrlcts.product()->begin(); csc!=corrlcts.product()->end(); csc++)
        {
          CSCCorrelatedLCTDigiCollection::Range range1 = corrlcts.product()->get((*csc).first);
          for(CSCCorrelatedLCTDigiCollection::const_iterator lct=range1.first; lct!=range1.second; lct++)
            {
              int endcap  = (*csc).first.endcap()-1;
              int station = (*csc).first.station()-1;
              int sector  = (*csc).first.triggerSector()-1;
              int subSector = CSCTriggerNumbering::triggerSubSectorFromLabels((*csc).first);
              int ring      = (*csc).first.ring();
              int cscId   = (*csc).first.triggerCscId()-1;
              int fpga    = ( subSector ? subSector-1 : station+1 );
              int strip     = lct -> getStrip();
              int keyWire   = lct -> getKeyWG();
              int bx        = lct -> getBX();


              int endcapAssignment = 1;
              int shift = 1;
              float sectorArg = sector;
              //float sectorArg = j;

              if( endcap == 1 ){
                endcapAssignment = -1;
                shift = 2;
                //sectorArg = sector - 6;
              }

              int signedStation = (station + shift)* endcapAssignment;
              if( (station == 0) && (endcap == 0)) signedStation = subSector - 1;
              if( (station == 0) && (endcap == 1)) signedStation = (-1)*subSector;

              float chamberArg1 = cscId * 0.1 + sectorArg;
              //float chamberArg1 = i*0.1 + sectorArg;
              //std::cout << "First" << i << " " << sectorArg << " " << chamberArg1 << std::endl;

              float chamberArg11 = chamberArg1;
              if(sectorArg == 1) chamberArg1 = chamberArg11 - 0.1;
              if(sectorArg == 2) chamberArg1 = chamberArg11 - 0.2;
              if(sectorArg == 3) chamberArg1 = chamberArg11 - 0.3;
              if(sectorArg == 4) chamberArg1 = chamberArg11 - 0.4;
              if(sectorArg == 5) chamberArg1 = chamberArg11 - 0.5;

              //std::cout << "cscId, station, sector, endcap, sectorArg, chamber Arg: " << cscId << ", " << station << ", " <<sector << ", " << endcap << ", " << chamberArg1 << ", " << signedStation << std::endl;

              csctfChamberOccupancies->Fill(chamberArg1, signedStation);
              //int bunchX = ( (lct->getBX()) - 6 );

              //int timingSectorArg = 3*(sector) + (lct->getMPCLink());
              //if( endcap == 1) timingSectorArg = 3*(sector + 6) + (lct->getMPCLink());
              //std::cout << "Sector, MPCLink, TSA, endcap: " << sector << ", " << lct->getMPCLink() << ", " << timingSectorArg << ", " << endcap << std::endl;

              //csctfbx->Fill(timingSectorArg, bunchX );
              //std::cout << "LCT'S, encap: " << endcap << ", station: " << station << ", sector: " << sector << ", subSector: " << subSector << ", cscId: " << cscId << std:: endl;
              //End JAG

              // Check if Det Id is within pysical range:
              if( endcap<0||endcap>1 || sector<0||sector>6 || station<0||station>3 || cscId<0||cscId>8 || fpga<0||fpga>4)
                {
                  edm::LogError("L1CSCTF: CSC TP are out of range: ")<<"  endcap: "<<(endcap+1)<<"  station: "<<(station+1) <<"  sector: "<<(sector+1)<<"  subSector: "<<subSector<<"  fpga: "<<fpga<<"  cscId: "<<(cscId+1);
                  continue;
                }
              lclphidat lclPhi;
              try {
                lclPhi = srLUTs_[fpga]->localPhi(lct->getStrip(), lct->getPattern(), lct->getQuality(), lct->getBend(), gangedME11a_);
              } catch(cms::Exception &) {
                bzero(&lclPhi,sizeof(lclPhi));
              }

              gblphidat gblPhi;
              try {
                gblPhi = srLUTs_[fpga]->globalPhiME(lclPhi.phi_local, lct->getKeyWG(), cscId+1, gangedME11a_);
              } catch(cms::Exception &) {
                bzero(&gblPhi,sizeof(gblPhi));
              }

              gbletadat gblEta;
              try {
                gblEta = srLUTs_[fpga]->globalEtaME(lclPhi.phi_bend_local, lclPhi.phi_local, lct->getKeyWG(), cscId+1, gangedME11a_);
              } catch(cms::Exception &) {
                bzero(&gblEta,sizeof(gblEta));
              }


              //TrackStub
              csctf::TrackStub theStub((*lct), (*csc).first);
              theStub.setPhiPacked(gblPhi.global_phi);
              theStub.setEtaPacked(gblEta.global_eta);

              float etaG = theStub.etaValue();
              float phiG = fmod( theStub.phiValue()+15.0*M_PI/180+(sector)*60.0*M_PI/180, 2.*M_PI );


	      //BX plots
	      // endcap==1: minus side; endcap==0: plus side
	      // station=0,1,2,3; ring=1,2,3;
              if(endcap==1) {
                    if(station==0) {
                        if(ring==1)             csctflcts -> Fill(bx, 8.5);
                        else if(ring==2)        csctflcts -> Fill(bx, 7.5);
                        else                    csctflcts -> Fill(bx, 6.5);
                    } else if(station==1) {
                        if(ring==1)             csctflcts -> Fill(bx, 5.5);
                        else                    csctflcts -> Fill(bx, 4.5);
                    } else if(station==2) {
                        if(ring==1)             csctflcts -> Fill(bx, 3.5);
                        else                    csctflcts -> Fill(bx, 2.5);
                    } else if(station==3) {
                        if(ring==1)             csctflcts -> Fill(bx, 1.5);
                        else                    csctflcts -> Fill(bx, 0.5);
                    }

              } else {
                    if(station==0) {
                        if(ring==1)             csctflcts -> Fill(bx, 9.5);
                        else if(ring==2)        csctflcts -> Fill(bx, 10.5);
                        else                    csctflcts -> Fill(bx, 11.5);
                    } else if(station==1) {
                        if(ring==1)             csctflcts -> Fill(bx, 12.5);
                        else                    csctflcts -> Fill(bx, 13.5);
                    } else if(station==2) {
                        if(ring==1)             csctflcts -> Fill(bx, 14.5);
                        else                    csctflcts -> Fill(bx, 15.5);
                    } else if(station==3) {
                        if(ring==1)             csctflcts -> Fill(bx, 16.5);
                        else                    csctflcts -> Fill(bx, 17.5);
                    }
              }





               // only for ME1/1
              if(station == 0 && ring == 1){
                me11_lctStrip    -> Fill(strip);
                me11_lctWire     -> Fill(keyWire);
                me11_lctLocalPhi -> Fill(lclPhi.phi_local);
                me11_lctPackedPhi-> Fill(theStub.phiPacked());
                me11_lctGblPhi   -> Fill(phiG);
                me11_lctGblEta   -> Fill(etaG);
              }

              // only for ME4/2
              if(station == 3 && ring == 2){
                me42_lctGblPhi   -> Fill(phiG);
                me42_lctGblEta   -> Fill(etaG);
              }


                //ME1/1
              if (station == 0 && ring == 1){
                int realID = cscId+6*sector+3*subSector;
                if(realID>36) realID -= 36;
                if(endcap == 0) { csc_strip_MEplus11  -> Fill(realID,strip); csc_wire_MEplus11  -> Fill(realID,keyWire); }
                if(endcap == 1) { csc_strip_MEminus11 -> Fill(realID,strip); csc_wire_MEminus11 -> Fill(realID,keyWire); }
              }
                //ME1/2
              if (station == 0 && ring == 2){
                int realID = (cscId-3)+6*sector+3*subSector;
                if(realID>36) realID -= 36;
                if(endcap == 0) { csc_strip_MEplus12  -> Fill(realID,strip); csc_wire_MEplus12  -> Fill(realID,keyWire); }
                if(endcap == 1) { csc_strip_MEminus12 -> Fill(realID,strip); csc_wire_MEminus12 -> Fill(realID,keyWire); }
              }
                //ME1/3
              if (station == 0 && ring == 3){
                int realID = (cscId-6)+6*sector+3*subSector;
                if(realID>36) realID -= 36;
                if(endcap == 0) { csc_strip_MEplus13  -> Fill(realID,strip); csc_wire_MEplus13  -> Fill(realID,keyWire); }
                if(endcap == 1) { csc_strip_MEminus13 -> Fill(realID,strip); csc_wire_MEminus13 -> Fill(realID,keyWire); }
              }
                //ME2/1
              if (station == 1 && ring == 1){
                int realID = cscId+3*sector+2;
                if(realID>18) realID -= 18;
                if(endcap == 0) { csc_strip_MEplus21  -> Fill(realID,strip); csc_wire_MEplus21  -> Fill(realID,keyWire); }
                if(endcap == 1) { csc_strip_MEminus21 -> Fill(realID,strip); csc_wire_MEminus21 -> Fill(realID,keyWire); }
              }
                //ME2/2
              if (station == 1 && ring == 2){
                int realID = (cscId-3)+6*sector+3;
                if(realID>36) realID -= 36;
                if(endcap == 0) { csc_strip_MEplus22  -> Fill(realID,strip); csc_wire_MEplus22  -> Fill(realID,keyWire); }
                if(endcap == 1) { csc_strip_MEminus22 -> Fill(realID,strip); csc_wire_MEminus22 -> Fill(realID,keyWire); }
              }

                //ME3/1
              if (station == 2 && ring == 1){
                int realID = cscId+3*sector+2;
                if(realID>18) realID -= 18;
                if(endcap == 0) { csc_strip_MEplus31  -> Fill(realID,strip); csc_wire_MEplus31  -> Fill(realID,keyWire); }
                if(endcap == 1) { csc_strip_MEminus31 -> Fill(realID,strip); csc_wire_MEminus31 -> Fill(realID,keyWire); }
              }

                //ME3/2
              if (station == 2 && ring == 2){
                int realID = (cscId-3)+6*sector+3;
                if(realID>36) realID -= 36;
                if(endcap == 0) { csc_strip_MEplus32  -> Fill(realID,strip); csc_wire_MEplus32  -> Fill(realID,keyWire); }
                if(endcap == 1) { csc_strip_MEminus32 -> Fill(realID,strip); csc_wire_MEminus32 -> Fill(realID,keyWire); }
              }
                //ME4/1
              if (station == 3 && ring == 1){
                int realID = cscId+3*sector+2;
                if(realID>18) realID -= 18;
                if(endcap == 0) { csc_strip_MEplus41  -> Fill(realID,strip); csc_wire_MEplus41  -> Fill(realID,keyWire);}
                if(endcap == 1) { csc_strip_MEminus41 -> Fill(realID,strip); csc_wire_MEminus41 -> Fill(realID,keyWire);}
              }
                //ME4/2
              if (station == 3 && ring == 2){
                int realID = (cscId-3)+6*sector+3;
                if(realID>36) realID -= 36;
                if(endcap == 0) { csc_strip_MEplus42  -> Fill(realID,strip); csc_wire_MEplus42  -> Fill(realID,keyWire); }
                if(endcap == 1) { csc_strip_MEminus42 -> Fill(realID,strip); csc_wire_MEminus42 -> Fill(realID,keyWire); }
              }






              // SR LUT gives packed eta and phi values -> normilize them to 1 by scale them to 'max' and shift by 'min'
              //float etaP = gblEta.global_eta/127*1.5 + 0.9;
              //float phiP =  (gblPhi.global_phi);// + ( sector )*4096 + station*4096*12) * 1./(4*4096*12);
              //std::cout << "LCT Eta & Phi Coordinates: " << etaP << ", " << phiP << "." << std::endl;
              //csctfoccupancies->Fill( gblEta.global_eta/127. * 1.5 + 0.9, (gblPhi.global_phi + ( sector + (endcap?0:6) )*4096 + station*4096*12) * 1./(4*4096*12) );
            }//lct != range1.scond
        }//csc!=corrlcts.product()->end()
    }// lctProducer.label() != "null"



  if( trackProducer.label() != "null" )
    {
      edm::Handle<L1CSCTrackCollection> tracks;
      e.getByToken(tracksToken_, tracks);
      for(L1CSCTrackCollection::const_iterator trk=tracks->begin(); trk<tracks->end(); trk++)
        {

          NumCSCTfTracksRep++;
	  long LUTAdd = trk->first.ptLUTAddress();
	  int trigMode = ( (LUTAdd)&0xf0000 ) >> 16;
	  int trEta = (trk->first.eta_packed() );


          // trk->first.endcap() = 2 for - endcap
          //                     = 1 for + endcap
          //int trEndcap = (trk->first.endcap()==2 ? trk->first.endcap()-3 : trk->first.endcap());
	  if( trk->first.endcap() != 1)
	    {
	      int holder = trEta;
	      trEta = -1*holder;
	      trEta -= 1;
	    }

          int trSector = 6*(trk->first.endcap()-1)+trk->first.sector();
          int trBX     = trk->first.BX();

          //Here is what is done with output phi value:
          //output_phi = (phi / 32) * 3 /16
          //where:
          //phi is 12-bit phi, 4096 bins covering 62 degrees
          //output_phi is 5-bit value

          //Easy to see that output_phi can have values from 0 to 23, or 24 total combinations.
          //This gives per-bin phi value of 62/24 = 2.583333 degrees.

          // Sector 1 nominally starts at 15 degrees but there 1 degree overlap between sectors so 14 degrees effectively
          //double trPhi = trk->first.localPhi() * 62. / 24.;
          double trPhi     = ts->getPhiScale()->getLowEdge(trk->first.localPhi());
          double trPhi02PI = fmod(trPhi +
                                  ((trSector-1)*M_PI/3) +
                                  (M_PI*14/180.), 2*M_PI);

	  if (trigMode == 15) {
            csctfTrackPhi_H    -> Fill( trPhi02PI );
            csctfTrackEta_H    -> Fill( trEta );
            csctfoccupancies_H -> Fill( trEta, trPhi02PI );
            csctfbx_H          -> Fill( trSector, trBX );
          }
          else{
            csctfTrackPhi    -> Fill( trPhi02PI );
            csctfTrackEta    -> Fill( trEta );
            csctfoccupancies -> Fill( trEta, trPhi02PI );
            csctfbx          -> Fill( trSector, trBX );

            // Low Quality / High Quality Eta Distributions
            //|eta| < 2.1
            if (abs(trEta) < 24) {
              if (trigMode ==  2 ||
                  trigMode ==  3 ||
                  trigMode ==  4 ||
                  trigMode ==  5 ||
                  trigMode ==  6 ||
                  trigMode ==  7 ||
                  trigMode == 11 ||
                  trigMode == 12 ||
                  trigMode == 13 ||
                  trigMode == 14  )  csctfTrackEtaHighQ -> Fill (trEta);

              if (trigMode ==  8 ||
                  trigMode ==  9 ||
                  trigMode == 10  )  csctfTrackEtaLowQ  -> Fill (trEta);
            }
            else {//|eta| > 2.1
              if (trigMode ==  2 ||
                  trigMode ==  3 ||
                  trigMode ==  4 ||
                  trigMode ==  5  )  csctfTrackEtaHighQ -> Fill (trEta);
              else
                                     csctfTrackEtaLowQ  -> Fill (trEta);
            }
          }

          csctfTrackM->Fill( trk->first.modeExtended() );

          // we monitor the track quality only on the first link
          // so let's make sure to fill the plot if there is something that
          // is read from the hardware
          int trRank   = trk->first.rank();
          if (trRank) {
            int trQuality = ((trRank>>5)&0x3);
            trackModeVsQ->Fill( trk->first.modeExtended(), trQuality );
          }

          /*
             OLD METHOD FOR FILLING HALO PLOTS, IMPROVED METHOD USING ASSOCIATED TRACK STUBS
             BELOW ~LINE 605
             if( trigMode == 15 )
             {

             double haloVals[4][4];
             for( int i = 0; i < 4; i++)
             {
             haloVals[i][0] = 0;
             }

             edm::Handle<CSCCorrelatedLCTDigiCollection> corrlcts;
             for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator csc=corrlcts.product()->begin(); csc!=corrlcts.product()->end(); csc++)
             {
             CSCCorrelatedLCTDigiCollection::Range range1 = corrlcts.product()->get((*csc).first);
             for(CSCCorrelatedLCTDigiCollection::const_iterator lct=range1.first; lct!=range1.second; lct++)
             {
             int endcap  = (*csc).first.endcap()-1;
             int station = (*csc).first.station()-1;
             int sector  = (*csc).first.triggerSector()-1;
             int cscId   = (*csc).first.triggerCscId()-1;
             int subSector = CSCTriggerNumbering::triggerSubSectorFromLabels((*csc).first);
             int fpga    = ( subSector ? subSector-1 : station+1 );

             if(station != 4)
             {
             int modEnd = 1;
             if( endcap == 0 ) modEnd = -1;
             int indexHalo = modEnd + station;
             if(haloVals[indexHalo][0] == 1.0) haloVals[indexHalo][3] = 1.0;
             if(haloVals[indexHalo][0] == 0) haloVals[indexHalo][0] = 1.0;
             haloVals[indexHalo][1] = sector*1.0;

             lclphidat lclPhi;
             lclPhi = srLUTs_[fpga]->localPhi(lct->getStrip(), lct->getPattern(), lct->getQuality(), lct->getBend());
             gblphidat gblPhi;
             gblPhi = srLUTs_[fpga]->globalPhiME(lclPhi.phi_local, lct->getKeyWG(), cscId+1);
             gbletadat gblEta;
             gblEta = srLUTs_[fpga]->globalEtaME(lclPhi.phi_bend_local, lclPhi.phi_local, lct->getKeyWG(), cscId+1);

             haloVals[indexHalo][2] = gblEta.global_eta/127. * 1.5 + 0.9;
             } //station1 or 2
             } //lct first to second
             } //corrlcts

             if( (haloVals[0][0] == 1.) && (haloVals[1][0] == 1.) && (haloVals[0][3] != 1.) && (haloVals[1][3] != 1.)  )
             {
             if( haloVals[0][1] == haloVals[1][1] ){
             double delEta23 = haloVals[1][2] - haloVals[0][2];
             haloDelEta23->Fill( delEta23 );
             }
             }

             if( (haloVals[2][0] == 1.) && (haloVals[3][0] == 1.) && (haloVals[2][3] != 1.) && (haloVals[3][3] != 1.)  )
             {
             if( haloVals[2][1] == haloVals[3][1] ){
             double delEta23 = haloVals[3][2] - haloVals[2][2];
             haloDelEta23->Fill( delEta23 );
             }
             }
             } //halo trigger
          */

          int cscTrackStub = 0;
          //float haloEta[3];
          //for(int i=0; i<3; i++) haloEta[i]=-1.0;
          //bool haloME11 = false;
          CSCCorrelatedLCTDigiCollection lctsOfTracks=trk->second;
          for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator trackStub=lctsOfTracks.begin(); trackStub!=lctsOfTracks.end(); trackStub++)
            {
              CSCCorrelatedLCTDigiCollection::Range range2 = lctsOfTracks.get((*trackStub).first);
              for(CSCCorrelatedLCTDigiCollection::const_iterator lct=range2.first; lct!=range2.second; lct++)
                {
//                   int station = (*trackStub).first.station()-1;
//                   if(station != 4)
//                     {
//                       // int endcap  = (*trackStub).first.endcap()-1;
//                       // int sector  = (*trackStub).first.triggerSector()-1;
//                       int cscId   = (*trackStub).first.triggerCscId()-1;
//                       int subSector = CSCTriggerNumbering::triggerSubSectorFromLabels((*trackStub).first);
//                       int fpga    = ( subSector ? subSector-1 : station+1 );

//                       lclphidat lclPhi;
//                       lclPhi = srLUTs_[fpga]->localPhi(lct->getStrip(), lct->getPattern(), lct->getQuality(), lct->getBend());
//                       gblphidat gblPhi;
//                       gblPhi = srLUTs_[fpga]->globalPhiME(lclPhi.phi_local, lct->getKeyWG(), cscId+1);
//                       gbletadat gblEta;
//                       gblEta = srLUTs_[fpga]->globalEtaME(lclPhi.phi_bend_local, lclPhi.phi_local, lct->getKeyWG(), cscId+1);
//                       haloEta[station-1] = gblEta.global_eta/127. * 1.5 + 0.9;
//                       if(station==1 && cscId<2) haloME11 = true;
//                    }
                  cscTrackStub++;
                }
            }
          cscTrackStubNumbers->Fill(cscTrackStub);

//           if(trigMode == 15)
//             {
//               float dEta13 = haloEta[2]-haloEta[0];
//               float dEta12 = haloEta[1]-haloEta[0];
//               if(haloME11)
//                 {
//                   if(haloEta[1]!=-1.0) haloDelEta112->Fill(dEta12);
//                   if(haloEta[2]!=-1.0) haloDelEta113->Fill(dEta13);
//                 } else {
//                 if(haloEta[1]!=-1.0) haloDelEta12->Fill(dEta12);
//                 if(haloEta[2]!=-1.0) haloDelEta13->Fill(dEta13);
//               }
//             }
          //



        }
    }
  csctfntrack->Fill(NumCSCTfTracksRep);


  if( mbProducer.label() != "null" )
    {
      // handle to needed collections
      edm::Handle<CSCTriggerContainer<csctf::TrackStub> > dtStubs;
      e.getByToken(dtStubsToken_, dtStubs);
      edm::Handle<L1CSCTrackCollection> tracks;
      e.getByToken(mbtracksToken_, tracks);

      // loop on the DT stubs
      std::vector<csctf::TrackStub> vstubs = dtStubs->get();
      for(std::vector<csctf::TrackStub>::const_iterator stub=vstubs.begin();
          stub!=vstubs.end(); stub++)
        {
          if (verbose_)
            {
              edm::LogInfo("DataNotFound") << "\n mbEndcap: "               << stub->endcap();
              edm::LogInfo("DataNotFound") << "\n stub->getStrip()[FLAG]: " << stub->getStrip();
              edm::LogInfo("DataNotFound") << "\n stub->getKeyWG()[CAL]: "  << stub->getKeyWG();
              edm::LogInfo("DataNotFound") << "\n stub->BX(): "             << stub->BX();
              edm::LogInfo("DataNotFound") << "\n stub->sector(): "         << stub->sector();
              edm::LogInfo("DataNotFound") << "\n stub->subsector(): "      << stub->subsector();
              edm::LogInfo("DataNotFound") << "\n stub->station(): "        << stub->station();
              edm::LogInfo("DataNotFound") << "\n stub->phiPacked(): "      << stub->phiPacked();
              edm::LogInfo("DataNotFound") << "\n stub->getBend(): "        << stub->getBend();
              edm::LogInfo("DataNotFound") << "\n stub->getQuality(): "     << stub->getQuality();
              edm::LogInfo("DataNotFound") << "\n stub->cscid(): "          << stub->cscid() << endl;
            }
          // define the sector ID
          int mbId = (stub->endcap()==2) ? 6 : 0;
          mbId += stub->sector();
          // *** do not fill if CalMB variable is set ***
          // horrible! They used the same class to write up the LCT and MB info,
          // but given the MB does not have strip and WG they replaced this two
          // with the flag and cal bits... :S
          if (stub->getKeyWG() == 0) //!CAL as Janos adviced
            {
              // if FLAG =1, muon belong to previous BX
              int bxDT     = stub->BX()-stub->getStrip(); // correct by the FLAG
              int subDT    = stub->subsector();

              // Fill the event only if CSC had or would have triggered
              if (isCSCcand_)
                {
                  //look for tracks in the event and compare the matching DT stubs
                  int trkBX = 0;
                  for(L1CSCTrackCollection::const_iterator trk=tracks->begin(); trk<tracks->end(); trk++)
                    {
                      trkBX = trk->first.BX();
                      int trkId = (trk->first.endcap()==2) ? 6 : 0;
                      trkId += trk->first.sector();
                      if (verbose_){
                        edm::LogInfo("DataNotFound") << "\n trk BX: "  << trkBX
                                                     << " Sector: "    << trkId
                                                     << " SubSector: " << trk->first.subsector()
                                                     << " Endcap: "    << trk->first.endcap();

                        edm::LogInfo("DataNotFound") << "\n DT  BX: "    << stub->BX()
                                                     << " Sector: "      << mbId
                                                     << " SubSector: "   << stub->subsector()
                                                     << " Endcap: "      << stub->endcap() << endl;
                      }

                      if (mbId == trkId)
                        {
                          if (verbose_) {
                            edm::LogInfo("DataNotFound") << " --> MATCH" << endl;
                            edm::LogInfo("DataNotFound") << "Fill :" << trkBX+6-bxDT << " -- " << subDT << " -- cands" << endl;
                          }
                          // DT bx ranges from 3 to 9
                          // trk bx ranges from -3 to 3
                          DTstubsTimeTrackMenTimeArrival[mbId-1]->Fill(bxDT-trkBX-6,subDT);//subsec
                        }
                    }// loop on the tracks
                }//if (isCSCcand_){
            }//if (stub->getKeyWG() == 0) {
        }
    }
}
