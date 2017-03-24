/****************************************************************************
*
* This is a part of TotemDQM and TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*   Rafał Leszko (rafal.leszko@gmail.com)
*
****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/CTPPSDigi/interface/TotemVFATStatus.h"
#include "DataFormats/CTPPSDigi/interface/TotemFEDInfo.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/TotemRPLocalTrack.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSDiamondDigi.h"

#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"

#include <string>

#include "TRandom3.h"

#define HPTDC_bin_width 25e-9/1024

//----------------------------------------------------------------------------------------------------

 
class TotemRPDiamondDQMSource: public DQMEDAnalyzer
{
  public:
    TotemRPDiamondDQMSource(const edm::ParameterSet& ps);
    virtual ~TotemRPDiamondDQMSource();
  
  protected:
    void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
    void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
    void analyze(edm::Event const& e, edm::EventSetup const& eSetup);
    void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup);
    void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup);
    void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  private:
    unsigned int verbosity;
    bool excludeMultipleHits;

    edm::EDGetTokenT< edm::DetSetVector<TotemVFATStatus> > tokenStatus;
    edm::EDGetTokenT< edm::DetSetVector<TotemRPLocalTrack> > tokenLocalTrack;
    edm::EDGetTokenT< edm::DetSetVector<CTPPSDiamondDigi> > tokenDigi;
    edm::EDGetTokenT< edm::DetSetVector<CTPPSDiamondRecHit> > tokenDiamondHit;
    edm::EDGetTokenT< edm::DetSetVector<CTPPSDiamondLocalTrack> > tokenDiamondTrack;
    edm::EDGetTokenT<std::vector<TotemFEDInfo>> tokenFEDInfo;

    /// plots related to one RP
    struct PotPlots
    {
      MonitorElement *frame_problem=NULL, *frame_missing=NULL;

      MonitorElement *activity_per_bx=NULL, *activity_per_bx_short=NULL;
      MonitorElement *activity_per_bx_plus1=NULL, *activity_per_bx_short_plus1=NULL;
      MonitorElement *activity_per_bx_minus1=NULL, *activity_per_bx_short_minus1=NULL;
      MonitorElement *activity_per_fedbx=NULL, *activity_per_fedbx_short=NULL;

      MonitorElement *hitDistribution2d=NULL;
      MonitorElement *hitDistribution2dOOT=NULL;
      MonitorElement *activePlanes=NULL;
      
      MonitorElement *trackDistribution=NULL;
      MonitorElement *trackDistribution_mius1=NULL;
      MonitorElement *trackDistributionOOT=NULL;
      
      MonitorElement *stripTomographyAllFar = NULL, *stripTomographyAllNear = NULL;
      MonitorElement *stripTomographyAllFar_plus1 = NULL, *stripTomographyAllNear_plus1 = NULL;
      MonitorElement *stripTomographyAllFar_minus1 = NULL, *stripTomographyAllNear_minus1 = NULL;
      
      MonitorElement *leadingEdgeCumulativePot, *timeOverThresholdCumulativePot, *leadingTrailingCorrelationPot;
      MonitorElement *leading_without_trailing_cumulative_pot;
      
      MonitorElement *ec_check;
      
      MonitorElement *error_flags_cumulative = NULL;

      PotPlots() {}
      PotPlots(DQMStore::IBooker &ibooker, unsigned int id);
    };

    std::map<unsigned int, PotPlots> potPlots;
    int diff_tmp_582, diff_tmp_583;
    
    /// plots related to one RP plane
    struct PlanePlots
    {
      MonitorElement *digi_profile_cumulative = NULL;
      MonitorElement *hit_profile = NULL;	//FIXME Old
      MonitorElement *hitProfile = NULL;
      MonitorElement *hit_multiplicity = NULL;
      MonitorElement *threshold_voltage = NULL;
      
      MonitorElement *stripTomography_far = NULL;
      MonitorElement *stripTomography_near = NULL;


      PlanePlots() {}
      PlanePlots(DQMStore::IBooker &ibooker, unsigned int id);
    };

    std::map<unsigned int, PlanePlots> planePlots;
    
    /// plots related to one RP plane
    struct ChannelPlots
    {
      MonitorElement *error_flags = NULL;
      MonitorElement *leading_edge_cumulative = NULL;
      MonitorElement *time_over_threshold_cumulative = NULL;
      MonitorElement *leading_trailing_correlation = NULL;
      MonitorElement *leading_without_trailing = NULL;
      MonitorElement *hit_rate = NULL;
      MonitorElement *channel_ec_check;

      ChannelPlots() {}
      ChannelPlots(DQMStore::IBooker &ibooker, unsigned int id);
    };

    std::map<unsigned int, ChannelPlots> channelPlots;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

TotemRPDiamondDQMSource::PotPlots::PotPlots(DQMStore::IBooker &ibooker, unsigned int id)
{
  std::string path_, title_;
  CTPPSDiamondDetId(id).rpName(path_, CTPPSDiamondDetId::nPath);
  path_.replace(0, 5, "TimingDiamond");
  ibooker.setCurrentFolder(string("CTPPS/") + path_);
//   if (verbosity) cout<<"Booking RP: "<<path_<<"\t\t"<<id<<endl;

  CTPPSDiamondDetId(id).rpName(title_, CTPPSDiamondDetId::nFull);



  activity_per_bx = ibooker.book1D("activity per BX", title_+";Event.BX", 4002, -1.5, 4000. + 0.5);
  activity_per_bx_short = ibooker.book1D("activity per BX (short)", title_+";Event.BX", 102, -1.5, 100. + 0.5);
  
  activity_per_bx_plus1 = ibooker.book1D("activity per BX OOT +1", title_+";Event.BX", 4002, -1.5, 4000. + 0.5);
  activity_per_bx_short_plus1 = ibooker.book1D("activity per BX OOT +1 (short)", title_+";Event.BX", 102, -1.5, 100. + 0.5);
  
  activity_per_bx_minus1 = ibooker.book1D("activity per BX OOT -1", title_+";Event.BX", 4002, -1.5, 4000. + 0.5);
  activity_per_bx_short_minus1 = ibooker.book1D("activity per BX OOT -1 (short)", title_+";Event.BX", 102, -1.5, 100. + 0.5);

  activity_per_fedbx = ibooker.book1D("activity per FED BX", title_+";Event.BX", 4002, -1.5, 4000. + 0.5);
  activity_per_fedbx_short = ibooker.book1D("activity per FED BX (short)", title_+";Event.BX", 102, -1.5, 100. + 0.5);

  
  hitDistribution2d = ibooker.book2D("hits in planes (2D)", title_+";plane number;x (mm)", 9, -0.5, 4, 1900, -1, 18);
  hitDistribution2dOOT= ibooker.book2D("hits with OOT (x0_25) in planes (2D)", title_+";plane number;x (mm)", 17, -0.25, 4, 600, 0, 18);
  activePlanes = ibooker.book1D("active planes", title_+";number of active planes", 6, -0.5, 5.5);
  
  trackDistribution = ibooker.book1D("tracks", title_+";x (mm)", 950, -1, 18);
  trackDistribution_mius1 = ibooker.book1D("tracks OOT -1", title_+";x (mm)", 950, -1, 18);
  trackDistributionOOT = ibooker.book2D("tracks with OOT", title_+";plane number;x (mm)", 9, -0.5, 4, 600, 0, 18);
  
  stripTomographyAllFar = ibooker.book2D("Tomography all far", "Tomography with strips far (all planes);x (mm);y (mm)", 400, 0, 200, 200, -50, 50 );
  stripTomographyAllNear = ibooker.book2D("Tomography all near", "Tomography with strips near (all planes);x (mm);y (mm)", 400, 0, 200, 200, -50, 50 );
  
  stripTomographyAllFar_plus1 = ibooker.book2D("Tomography all far OOT +1", "Tomography with strips far (all planes);x (mm);y (mm)", 400, 0, 200, 200, -50, 50 );
  stripTomographyAllNear_plus1 = ibooker.book2D("Tomography all near OOT +1", "Tomography with strips near (all planes);x (mm);y (mm)", 400, 0, 200, 200, -50, 50 );

  stripTomographyAllFar_minus1 = ibooker.book2D("Tomography all far OOT -1", "Tomography with strips far (all planes);x (mm);y (mm)", 400, 0, 200, 200, -50, 50 );
  stripTomographyAllNear_minus1 = ibooker.book2D("Tomography all near OOT -1", "Tomography with strips near (all planes);x (mm);y (mm)", 400, 0, 200, 200, -50, 50 );  
  
  leadingEdgeCumulativePot = ibooker.book1D("Leading Edge", title_+";leading edge", 1001,0,100);
  timeOverThresholdCumulativePot = ibooker.book1D("Time over Threshold", title_+";time over threshold", 2e3,-1000,1000);
  leadingTrailingCorrelationPot = ibooker.book2D("Leading Trailing Correlation", title_+";leading;trailing", 3e2,0,60, 3e2,0,60);
  
  leading_without_trailing_cumulative_pot = ibooker.book1D("Leading Edges Without Trailing", title_+";leading edges without trailing", 4,0.5,4.5);
  leading_without_trailing_cumulative_pot->getTH1F()->GetXaxis()->SetBinLabel(1, "Nothing");
  leading_without_trailing_cumulative_pot->getTH1F()->GetXaxis()->SetBinLabel(2, "Leading only");
  leading_without_trailing_cumulative_pot->getTH1F()->GetXaxis()->SetBinLabel(3, "Trailing only");
  leading_without_trailing_cumulative_pot->getTH1F()->GetXaxis()->SetBinLabel(4, "Both");
  
  ec_check = ibooker.book2D("optorxEC(8bit) - vfatEC", title_+";HPTDC ID;optorxEC-vfatEC", 9,-0.5,4,512,-256,256);
  
  error_flags_cumulative = ibooker.book1D("hptdc_Errors", title_, 16, -0.5, 16.5);
  for (short error_index=1; error_index<16; ++error_index) 
    error_flags_cumulative->getTH1F()->GetXaxis()->SetBinLabel(error_index, HPTDCErrorFlags::getHPTDCErrorName(error_index-1).c_str());
  error_flags_cumulative->getTH1F()->GetXaxis()->SetBinLabel(16, "MH");
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

TotemRPDiamondDQMSource::PlanePlots::PlanePlots(DQMStore::IBooker &ibooker, unsigned int id)
{
  std::string path_, title_;
  CTPPSDiamondDetId(id).planeName(path_, CTPPSDiamondDetId::nPath);
  path_.replace(0, 5, "TimingDiamond");
  ibooker.setCurrentFolder(string("CTPPS/") + path_);
//   if (verbosity) cout<<"Booking Plane: "<<path_<<"\t\t"<<id<<endl;

  CTPPSDiamondDetId(id).planeName(title_, CTPPSDiamondDetId::nFull);

  digi_profile_cumulative = ibooker.book1D("digi profile", title_, 12, -0.5, 11.5);
  hit_profile = ibooker.book1D("hit (old) profile", title_+";hits/detector/event", 180, 0, 18);
  hitProfile = ibooker.book1D("hit profile", title_+";hits/detector/event", 180, 0, 18);
  hit_multiplicity = ibooker.book1D("hit multiplicity", title_+";x (mm)", 13, -0.5, 12.5);
  
  threshold_voltage = ibooker.book1D("Threshold Voltage", title_+";threshold voltage", 12, -0.5, 11.5);
  
  stripTomography_far = ibooker.book2D("Tomography_far", "Tomography with strips far;x (mm);y (mm)", 500, 0, 50, 500, -30, 70 );
  stripTomography_near = ibooker.book2D("Tomography_near", "Tomography with strips near;x (mm);y (mm)", 500, 0, 50, 500, -30, 70 );
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

TotemRPDiamondDQMSource::ChannelPlots::ChannelPlots(DQMStore::IBooker &ibooker, unsigned int id)
{
  std::string path_, title_;
  CTPPSDiamondDetId(id).channelName(path_, CTPPSDiamondDetId::nPath);	//TODO
  path_.replace(0, 5, "TimingDiamond");
  ibooker.setCurrentFolder(string("CTPPS/") + path_);
//   if (verbosity) cout<<"Booking Channel: "<<path_<<"\t\t"<<id<<endl;

//   string title_ = DiamondDetId::planeName(id, DiamondDetId::nFull);	//TODO
  CTPPSDiamondDetId(id).channelName(title_, CTPPSDiamondDetId::nFull);
  
  leading_without_trailing = ibooker.book1D("Leading Edge Without Trailing", title_+";leading edge without trailing", 4,0.5,4.5);
  leading_without_trailing->getTH1F()->GetXaxis()->SetBinLabel(1, "Nothing");
  leading_without_trailing->getTH1F()->GetXaxis()->SetBinLabel(2, "Leading only");
  leading_without_trailing->getTH1F()->GetXaxis()->SetBinLabel(3, "Trailer only");
  leading_without_trailing->getTH1F()->GetXaxis()->SetBinLabel(4, "Full");

  error_flags = ibooker.book1D("hptdc_Errors", title_+";channel number", 16, -0.5, 16.5);
  for (short error_index=1; error_index<16; ++error_index) 
    error_flags->getTH1F()->GetXaxis()->SetBinLabel(error_index, HPTDCErrorFlags::getHPTDCErrorName(error_index-1).c_str());
  error_flags->getTH1F()->GetXaxis()->SetBinLabel(16, "MH");
  
  leading_edge_cumulative = ibooker.book1D("Leading Edge", title_+";leading edge", 1001,0,100e-9);
  time_over_threshold_cumulative = ibooker.book1D("Time over Threshold", title_+";time over threshold", 2e3,-1e-6,1e-6);
  leading_trailing_correlation = ibooker.book2D("Leading Trailing Correlation", title_+";leading trailing corr", 3e2,0,60e-9, 3e2,0,60e-9);
  
  channel_ec_check = ibooker.book2D("optorxEC(8bit) - vfatEC vs optorxEC", title_+";optorxEC-vfatEC", 7e3+1, 0, 7e3, 513,-256,256);

  
  hit_rate = ibooker.book1D("hit rate", title_+";hit rate", 20, -0.5, 1.5);	// Hz?
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

TotemRPDiamondDQMSource::TotemRPDiamondDQMSource(const edm::ParameterSet& ps) :
  verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0))
{
  tokenStatus = consumes<DetSetVector<TotemVFATStatus>>(ps.getParameter<edm::InputTag>("tagStatus"));
  tokenLocalTrack = consumes< DetSetVector<TotemRPLocalTrack> >(ps.getParameter<edm::InputTag>("tagLocalTrack"));
  tokenDigi = consumes< DetSetVector<CTPPSDiamondDigi> >(ps.getParameter<edm::InputTag>("tagDigi"));  
  tokenFEDInfo = consumes<vector<TotemFEDInfo>>(ps.getParameter<edm::InputTag>("tagFEDInfo"));
  tokenDiamondHit = consumes< DetSetVector<CTPPSDiamondRecHit> >(ps.getParameter<edm::InputTag>("tagDiamondRecHits"));  
  tokenDiamondTrack = consumes<DetSetVector<CTPPSDiamondLocalTrack> >(ps.getParameter<edm::InputTag>("tagDiamondLocalTracks"));
  
  excludeMultipleHits = ps.getParameter<bool>( "excludeMultipleHits" );
    
  diff_tmp_582=-500;
  diff_tmp_583=-500;

}

//----------------------------------------------------------------------------------------------------

TotemRPDiamondDQMSource::~TotemRPDiamondDQMSource()
{
}

//----------------------------------------------------------------------------------------------------

void TotemRPDiamondDQMSource::dqmBeginRun(edm::Run const &, edm::EventSetup const &)
{
}

//----------------------------------------------------------------------------------------------------

void TotemRPDiamondDQMSource::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &)
{
  ibooker.cd();
  ibooker.setCurrentFolder("CTPPS");

  // loop over arms
  for (unsigned int arm = 0; arm < 2; ++arm)
  {
    // loop over stations
    for (unsigned int st = 1; st < 2; st += 2)
    {
      // loop over RPs
      for (unsigned int rp = 6; rp < 7; ++rp)
      {
        CTPPSDiamondDetId rpId(arm, st, rp);
        potPlots[rpId] = PotPlots(ibooker, rpId);

        // loop over planes
        for (unsigned int pl = 0; pl < 4; ++pl)
        {
          CTPPSDiamondDetId plId(arm, st, rp, pl);
          planePlots[plId] = PlanePlots(ibooker, plId);
	  
	  // loop over channels
	  for (unsigned int ch = 0; ch < 12; ++ch)
	  {
	    CTPPSDiamondDetId chId(arm, st, rp, pl, ch);
// 	      cout<<"\t"<<arm<<"\t"<<st<<"\t"<<rp<<"\t"<<pl<<"\t"<<ch<<"\t\t"<<chId<<std::endl;
	    channelPlots[chId] = ChannelPlots(ibooker, chId);
	  }  
        }
      }
    }
  }
}


//----------------------------------------------------------------------------------------------------

void TotemRPDiamondDQMSource::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, 
                                            edm::EventSetup const& context) 
{
}

//----------------------------------------------------------------------------------------------------

void TotemRPDiamondDQMSource::analyze(edm::Event const& event, edm::EventSetup const& eventSetup)
{
  // get event data
  Handle< DetSetVector<TotemVFATStatus> > status;
  event.getByToken(tokenStatus, status);
  
  Handle< DetSetVector<TotemRPLocalTrack> > StripTracks;
  event.getByToken(tokenLocalTrack, StripTracks);

  Handle< DetSetVector<CTPPSDiamondDigi> > digi;
  event.getByToken(tokenDigi, digi);
  
  Handle<vector<TotemFEDInfo>> fedInfo;
  event.getByToken(tokenFEDInfo, fedInfo);
  
  Handle<DetSetVector<CTPPSDiamondRecHit>> hits;
  event.getByToken(tokenDiamondHit, hits);
  
  Handle<DetSetVector<CTPPSDiamondLocalTrack>> localTracks;
  event.getByToken(tokenDiamondTrack, localTracks);

  // check validity
  bool valid = true;
  valid &= status.isValid();
  valid &= digi.isValid();
  valid &= fedInfo.isValid();
//   valid &= StripTracks.isValid();

  if (!valid)
  {
    if (verbosity)
    {
      LogProblem("TotemRPDiamondDQMSource") <<
        "ERROR in TotemDQMModuleRP::analyze > some of the required inputs are not valid. Skipping this event.\n"
        << "    status.isValid = " << status.isValid() << "\n"
// 	<< "    StripTracks.isValid = " << StripTracks.isValid() << "\n"
        << "    digi.isValid = " << digi.isValid() << "\n"
	<< "    fedInfo.isValid = " << fedInfo.isValid();
    }

    return;
  }

  
  //------------------------------
  // RP Plots

  //Rnd generator used to "fake" pad width
  TRandom3 randomGen;
  
  
  // EC Errors
  for (DetSetVector<TotemVFATStatus>::const_iterator it = status->begin(); it != status->end(); ++it) 
  {
    CTPPSDiamondDetId detId(it->detId());
    for (DetSet<TotemVFATStatus>::const_iterator dit = it->begin(); dit != it->end(); ++dit)
      {
	CTPPSDiamondDetId detId_pot(it->detId());
	detId_pot.setPlane(0);
	detId_pot.setChannel(0);
	if (potPlots.find(detId_pot) != potPlots.end()) {
	//Check Event Number
	    for (auto &optorx : *fedInfo) 
	    {
		if (detId.arm()==1 && optorx.getFEDId()==582) {
// 		  potPlots[detId_pot].ec_check->Fill(DiamondTmp::HPTDCId(dit->getID()),(int)((optorx.getLV1()& 0xFF)-((unsigned int) dit->getEC() & 0xFF)) & 0xFF);
		  if (((int) ((optorx.getLV1()& 0xFF)-dit->getEC()) != diff_tmp_582) && ((uint8_t) ((optorx.getLV1()& 0xFF)-dit->getEC()) < 128))
		    diff_tmp_582 = (int) (((optorx.getLV1())& 0xFF)-((unsigned int) dit->getEC() & 0xFF));
		  if (diff_tmp_582 != 1 && diff_tmp_582 != -500 && diff_tmp_582<128 && diff_tmp_582>-128)
		    cout << "FED 852: ECError at EV: 0x"<< hex << optorx.getLV1() << "\t\tVFAT EC: 0x"<< (unsigned int) dit->getEC() <<"\twith ID: "<< dec << detId <<"\tdiff: "<<  diff_tmp_582 << endl;
	      }
		else if (detId.arm()==0 && optorx.getFEDId()==583) {
// 		  potPlots[detId_pot].ec_check->Fill(DiamondTmp::HPTDCId(dit->getID()),(int)((optorx.getLV1()& 0xFF)-dit->getEC()) & 0xFF);
		  if (((int) ((optorx.getLV1()& 0xFF)-dit->getEC()) != diff_tmp_583) && ((uint8_t) ((optorx.getLV1()& 0xFF)-dit->getEC()) < 128))
		    diff_tmp_583 = (int) (((optorx.getLV1())& 0xFF)-((unsigned int) dit->getEC() & 0xFF));
		  if (diff_tmp_583 != 1 && diff_tmp_583 != -500 && diff_tmp_583<128 && diff_tmp_583>-128)
		    cout << "FED 853: ECError at EV: 0x"<< hex << optorx.getLV1() << "\t\tVFAT EC: 0x"<< (unsigned int) dit->getEC() <<"\twith ID: "<< dec << detId <<"\tdiff: "<<  diff_tmp_583 << endl;
		}
	    }
	}
      }
  }
  
  //RecHits
  map<unsigned int, set<unsigned int> > planes;
  for (DetSetVector<CTPPSDiamondRecHit>::const_iterator it=hits->begin(); it != hits->end(); ++it) {
    CTPPSDiamondDetId detId_pot(it->detId());
    detId_pot.setPlane(0);
    detId_pot.setChannel(0);
    CTPPSDiamondDetId detId(it->detId());
    for (DetSet<CTPPSDiamondRecHit>::const_iterator hitIt=it->begin(); hitIt != it->end(); ++hitIt) {
      if (excludeMultipleHits and hitIt->getMultipleHits() > 0 ) continue;
      planes[detId_pot].insert(detId.plane());
      if (potPlots.find(detId_pot) != potPlots.end()) {
	double rnd = (2. * randomGen.Rndm()) - 1.;
	potPlots[detId_pot].hitDistribution2d->Fill(detId.plane(), hitIt->getX() + .5*rnd*hitIt->getXWidth());
	potPlots[detId_pot].hitDistribution2dOOT->Fill(detId.plane() + 0.25 * hitIt->getOOTIndex(), hitIt->getX() + .5*rnd*hitIt->getXWidth());
	
	
	potPlots[detId_pot].leadingEdgeCumulativePot->Fill( hitIt->getT() );
	/*if (hitIt->getToT() != 0.0) */potPlots[detId_pot].timeOverThresholdCumulativePot->Fill( hitIt->getToT() );
	/*if (hitIt->getToT() != 0.0) */potPlots[detId_pot].leadingTrailingCorrelationPot->Fill( hitIt->getT(), hitIt->getT() + hitIt->getToT() );
	
	switch (hitIt->getOOTIndex()) {
	  case 0:
	    potPlots[detId_pot].activity_per_bx_minus1->Fill(event.bunchCrossing());
	    potPlots[detId_pot].activity_per_bx_short_minus1->Fill(event.bunchCrossing());
	    break;
	  case 1:
	    potPlots[detId_pot].activity_per_bx->Fill(event.bunchCrossing());
	    potPlots[detId_pot].activity_per_bx_short->Fill(event.bunchCrossing());
	    break;
	  case 2:
	    potPlots[detId_pot].activity_per_bx_plus1->Fill(event.bunchCrossing());
	    potPlots[detId_pot].activity_per_bx_short_plus1->Fill(event.bunchCrossing());
	    break;
	}
      }
      
    }
  }
  
  for (std::map<unsigned int, PotPlots>::iterator it = potPlots.begin(); it != potPlots.end(); it++) {
    it->second.activePlanes->Fill(planes[it->first].size());
  }
  
  //LocalTracks
  for (DetSetVector<CTPPSDiamondLocalTrack>::const_iterator it=localTracks->begin(); it != localTracks->end(); ++it) {
    CTPPSDiamondDetId detId_pot(it->detId());
    detId_pot.setPlane(0);
    detId_pot.setChannel(0);
    CTPPSDiamondDetId detId(it->detId());
    for (DetSet<CTPPSDiamondLocalTrack>::const_iterator trackIt=it->begin(); trackIt != it->end(); ++trackIt) {
      if (excludeMultipleHits and trackIt->getMultipleHits() > 0 ) continue;
      if (potPlots.find(detId_pot) != potPlots.end()) {
	double rnd = (2. * randomGen.Rndm()) - 1.;
	if (trackIt->getOOTIndex() == 1) potPlots[detId_pot].trackDistribution->Fill(trackIt->getX0()+rnd*trackIt->getX0Sigma());
	if (trackIt->getOOTIndex() == 0) potPlots[detId_pot].trackDistribution_mius1->Fill(trackIt->getX0()+rnd*trackIt->getX0Sigma());
	potPlots[detId_pot].trackDistributionOOT->Fill(trackIt->getOOTIndex(), trackIt->getX0()+rnd*trackIt->getX0Sigma());
      }
    }
  }
  
  //For diamond vertical allignment
  for (DetSetVector<CTPPSDiamondRecHit>::const_iterator it=hits->begin(); it != hits->end(); ++it) {
    CTPPSDiamondDetId detId_pot(it->detId());
    detId_pot.setPlane(0);
    detId_pot.setChannel(0);
    CTPPSDiamondDetId detId(it->detId());
    for (DetSet<CTPPSDiamondRecHit>::const_iterator hitIt=it->begin(); hitIt != it->end(); ++hitIt) {
      if (excludeMultipleHits and hitIt->getMultipleHits() > 0 ) continue;
      if ( StripTracks.isValid() ) {
	if (potPlots.find(detId_pot) != potPlots.end()) {
	  for (auto &ds2 : *StripTracks) {
	    for (auto &striplt : ds2) {
	      if (! striplt.isValid()) continue;
	      if (striplt.getTx() > 2e-4 or striplt.getTy() > 2e-4) continue;	//(0.2 mRad -> 1 mm at 5 m)
	      unsigned int arm = striplt.getZ0()>0 ? 1 : 0;
	      if (arm != detId_pot.arm() ) continue;
	      if (abs(striplt.getZ0()) > 207e3) {
		switch (hitIt->getOOTIndex()) {
		  case 0:
			    potPlots[detId_pot].stripTomographyAllFar_minus1->Fill( striplt.getX0() + 50*detId.plane(), striplt.getY0() );
			    break;
		  case 1:
			    potPlots[detId_pot].stripTomographyAllFar->Fill( striplt.getX0() + 50*detId.plane(), striplt.getY0() );
			    break;
		  case 2:
			    potPlots[detId_pot].stripTomographyAllFar_plus1->Fill( striplt.getX0() + 50*detId.plane(), striplt.getY0() );
			    break;
		}
	      }
	      else {
		switch (hitIt->getOOTIndex()) {
		  case 0:
			    potPlots[detId_pot].stripTomographyAllNear_minus1->Fill( striplt.getX0() + 50*detId.plane(), striplt.getY0() );
			    break;
		  case 1:
			    potPlots[detId_pot].stripTomographyAllNear->Fill( striplt.getX0() + 50*detId.plane(), striplt.getY0() );
			    break;
		  case 2:
			    potPlots[detId_pot].stripTomographyAllNear_plus1->Fill( striplt.getX0() + 50*detId.plane(), striplt.getY0() );
			    break;
		}
	      }
	    }
	  }
	}
      }
    }
  }
  
  
  
  for (DetSetVector<CTPPSDiamondDigi>::const_iterator it = digi->begin(); it != digi->end(); ++it)
  {
    CTPPSDiamondDetId detId(it->detId());
   
    for (DetSet<CTPPSDiamondDigi>::const_iterator dit = it->begin(); dit != it->end(); ++dit)
    {
      CTPPSDiamondDetId detId_pot(it->detId());
      detId_pot.setPlane(0);
      detId_pot.setChannel(0);
      if (potPlots.find(detId_pot) != potPlots.end()) { 
        //Leading without trailing investigation
	if (dit->getLeadingEdge() == 0 && dit->getTrailingEdge() == 0) potPlots[detId_pot].leading_without_trailing_cumulative_pot->Fill(1);
	else if (dit->getLeadingEdge() != 0 && dit->getTrailingEdge() == 0) potPlots[detId_pot].leading_without_trailing_cumulative_pot->Fill(2);
	else if (dit->getLeadingEdge() == 0 && dit->getTrailingEdge() != 0) potPlots[detId_pot].leading_without_trailing_cumulative_pot->Fill(3);
	else if (dit->getLeadingEdge() != 0 && dit->getTrailingEdge() != 0) potPlots[detId_pot].leading_without_trailing_cumulative_pot->Fill(4);
	
	
	if (dit->getLeadingEdge() != 0) { 
	  
	  // FED BX monitoring (for MINIDAQ)
          for (auto &fit : *fedInfo) {
	    if ((detId.arm()==1 && fit.getFEDId()==582)||(detId.arm()==0 && fit.getFEDId()==583)) {
// 		if ( dit->getLeadingEdge()>900 && dit->getLeadingEdge()<1450) {
// 		  potPlots[detId_pot].activity_per_fedbx_filtered->Fill(fit.getBX());
// 		  potPlots[detId_pot].activity_per_fedbx_short_filtered->Fill(fit.getBX());
// //                   cout<<"added"<<endl;
// 		}
		potPlots[detId_pot].activity_per_fedbx->Fill(fit.getBX());
		potPlots[detId_pot].activity_per_fedbx_short->Fill(fit.getBX());
	    }
          }
        }
        
        
        // HPTDC Errors
	HPTDCErrorFlags hptdcErrors = dit->getHPTDCErrorFlags();
	for (unsigned short hptdcErrorIndex=1; hptdcErrorIndex<16; ++hptdcErrorIndex) {
	  if (hptdcErrors.getErrorId(hptdcErrorIndex-1)) potPlots[detId_pot].error_flags_cumulative->Fill(hptdcErrorIndex);
	}
	if (dit->getMultipleHit()) potPlots[detId_pot].error_flags_cumulative->Fill(16);
      }
    }
  }

  
  //------------------------------
  // Plane Plots

  map<unsigned int, unsigned int> channelsPerPlane;
  for (DetSetVector<CTPPSDiamondDigi>::const_iterator it = digi->begin(); it != digi->end(); ++it)
  {
    CTPPSDiamondDetId detId(it->detId());
//     unsigned int DetId = DiamondDetId::rawToDecId(it->detId());
    for (DetSet<CTPPSDiamondDigi>::const_iterator dit = it->begin(); dit != it->end(); ++dit)
    {
      CTPPSDiamondDetId detId_plane(it->detId());
      detId_plane.setChannel(0);
      if (planePlots.find(detId_plane) != planePlots.end()) {
	planePlots[detId_plane].threshold_voltage->Fill(detId.channel(), dit->getThresholdVoltage());
	
	if (dit->getLeadingEdge() != 0 /*&& dit->getTrailingEdge() != 0.0*/) planePlots[detId_plane].digi_profile_cumulative->Fill(detId.channel());
// 	if ((dit->getLeadingEdge() != 0 /*&& dit->getTrailingEdge() != 0.0*/) and ( dit->getLeadingEdge()>0 && dit->getLeadingEdge()<1024)) planePlots[detId_plane].hit_profile->Fill(DiamondTmp::XRndPosition(detId));
	if (dit->getLeadingEdge() != 0 /*&& dit->getTrailingEdge() != 0.0*/) {
	  channelsPerPlane[detId_plane]++;
	}
      }
      
    }
  }
  
  for (map<unsigned int, unsigned int>::const_iterator it= channelsPerPlane.begin(); it != channelsPerPlane.end(); ++it) {
    planePlots[it->first].hit_multiplicity->Fill(it->second);
  }
  
  //RecHits
  for (DetSetVector<CTPPSDiamondRecHit>::const_iterator it=hits->begin(); it != hits->end(); ++it) {
    CTPPSDiamondDetId detId_plane(it->detId());
    detId_plane.setChannel(0);
    for (DetSet<CTPPSDiamondRecHit>::const_iterator hitIt=it->begin(); hitIt != it->end(); ++hitIt) {
      if (excludeMultipleHits and hitIt->getMultipleHits() > 0 ) continue;
      if (planePlots.find(detId_plane) != planePlots.end()) {
	double rnd = (2.*randomGen.Rndm()) - 1.;
	if (hitIt->getOOTIndex() == 1) planePlots[detId_plane].hitProfile->Fill( hitIt->getX()+.5*rnd*hitIt->getXWidth() );
      }
    }
  }
  
  //For diamond vertical allignment
  for (DetSetVector<CTPPSDiamondRecHit>::const_iterator it=hits->begin(); it != hits->end(); ++it) {
    CTPPSDiamondDetId detId_plane(it->detId());
    detId_plane.setChannel(0);
    for (DetSet<CTPPSDiamondRecHit>::const_iterator hitIt=it->begin(); hitIt != it->end(); ++hitIt) {
      if (excludeMultipleHits and hitIt->getMultipleHits() > 0 ) continue;
//       if (hitIt->getOOTIndex() == 1) {
	if ( StripTracks.isValid() ) {
	  if (planePlots.find(detId_plane) != planePlots.end()) {
	    for (auto &ds2 : *StripTracks) {
	      for (auto &striplt : ds2) {
		if (! striplt.isValid()) continue;
		if (striplt.getTx() > 2e-4 or striplt.getTy() > 2e-4) continue;	//(0.2 mRad -> 1 mm at 5 m)
		unsigned int arm = striplt.getZ0()>0 ? 1 : 0;
		if (arm != detId_plane.arm() ) continue;
		if (abs(striplt.getZ0()) > 207e3) {
		  planePlots[detId_plane].stripTomography_far->Fill( striplt.getX0(), striplt.getY0() + 30*(hitIt->getOOTIndex()-1) );
		}
		else {
		  planePlots[detId_plane].stripTomography_near->Fill( striplt.getX0(), striplt.getY0() + 30*(hitIt->getOOTIndex()-1) );
		}
	      }
	    }
	  }
	}
//       }
    }
  }
 
  
    //------------------------------
  // Channel Plots

  //Check Event Number
  for (DetSetVector<TotemVFATStatus>::const_iterator it = status->begin(); it != status->end(); ++it) {
    CTPPSDiamondDetId detId(it->detId());
    for (DetSet<TotemVFATStatus>::const_iterator dit = it->begin(); dit != it->end(); ++dit) {
      if (channelPlots.find(detId) != channelPlots.end()) {
	for (auto &optorx : *fedInfo) {
	  if ((detId.arm()==0 && optorx.getFEDId()==583) || (detId.arm()==1 && optorx.getFEDId()==582))
	    channelPlots[detId].channel_ec_check->Fill((optorx.getLV1()& 0xFFF),(int)((optorx.getLV1()& 0xFF)-dit->getEC()) & 0xFF);
	  if ((((optorx.getLV1()& 0xFF)-dit->getEC()) & 0xFF) > 255) cout<<"Strange: "<<(int) (((optorx.getLV1()& 0xFF)-dit->getEC()) & 0xFF) << endl;
	}
      }
    }
  }
  
  
  
  // digi profile cumulative TODO
  for (DetSetVector<CTPPSDiamondDigi>::const_iterator it = digi->begin(); it != digi->end(); ++it)
  {
    CTPPSDiamondDetId detId(it->detId());
    for (DetSet<CTPPSDiamondDigi>::const_iterator dit = it->begin(); dit != it->end(); ++dit)
    {
      if (channelPlots.find(detId) != channelPlots.end()) {
	if (dit->getLeadingEdge() != 0 ) {
            channelPlots[detId].leading_edge_cumulative->Fill(dit->getLeadingEdge() * HPTDC_bin_width);
            if (dit->getTrailingEdge() != 0) channelPlots[detId].time_over_threshold_cumulative->Fill( HPTDC_bin_width * (dit->getLeadingEdge() - dit->getTrailingEdge()));
            if (dit->getTrailingEdge() != 0) channelPlots[detId].leading_trailing_correlation->Fill(dit->getTrailingEdge() * HPTDC_bin_width, dit->getLeadingEdge() * HPTDC_bin_width);
	    
// 	    for (DetSetVector<CTPPSDiamondDigi>::const_iterator it2 = digi->begin(); it2 != digi->end(); ++it2)
// 	    {
// 	      CTPPSDiamondDetId detId2(it2->detId());
// 	      for (DetSet<CTPPSDiamondDigi>::const_iterator dit2 = it2->begin(); dit2 != it2->end(); ++dit2)
// 	      {
// 		channelPlots[detId].time_over_threshold_corrections->Fill()
// 	      }
// 	    }
	    
        }
	
	// HPTDC Errors
	HPTDCErrorFlags hptdcErrors = dit->getHPTDCErrorFlags();
	for (unsigned short hptdcErrorIndex=1; hptdcErrorIndex<16; ++hptdcErrorIndex) {
	  if (hptdcErrors.getErrorId(hptdcErrorIndex-1)) channelPlots[detId].error_flags->Fill(hptdcErrorIndex);
	}
	if (dit->getMultipleHit()) channelPlots[detId].error_flags->Fill(16);
	
	if (dit->getLeadingEdge() == 0 && dit->getTrailingEdge() == 0) channelPlots[detId].leading_without_trailing->Fill(1);
	else if (dit->getLeadingEdge() != 0 && dit->getTrailingEdge() == 0) channelPlots[detId].leading_without_trailing->Fill(2);
	else if (dit->getLeadingEdge() == 0 && dit->getTrailingEdge() != 0) channelPlots[detId].leading_without_trailing->Fill(3);
	else if (dit->getLeadingEdge() != 0 && dit->getTrailingEdge() != 0) channelPlots[detId].leading_without_trailing->Fill(4);
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------

void TotemRPDiamondDQMSource::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) 
{
}

//----------------------------------------------------------------------------------------------------

void TotemRPDiamondDQMSource::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(TotemRPDiamondDQMSource);
