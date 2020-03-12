#ifndef LaserTask_h
#define LaserTask_h

/*
 *	file:			LaserTask.h
 *	Author:			Viktor Khristenko
 *	Date:			16.10.2015
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"
#include "DQM/HcalCommon/interface/ContainerXXX.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle1D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf1D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "FWCore/Framework/interface/Run.h"

class LaserTask : public hcaldqm::DQTask {
public:
  LaserTask(edm::ParameterSet const &);
  ~LaserTask() override {}

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void dqmEndRun(edm::Run const &r, edm::EventSetup const &) override {
    if (_ptype == hcaldqm::fLocal) {
      if (r.runAuxiliary().run() == 1)
        return;
      else
        this->_dump();
    }
  }
  void dqmEndLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) override;

protected:
  //	funcs
  void _process(edm::Event const &, edm::EventSetup const &) override;
  void _resetMonitors(hcaldqm::UpdateFreq) override;
  bool _isApplicable(edm::Event const &) override;
  virtual void _dump();
  void processLaserMon(edm::Handle<QIE10DigiCollection> &col, std::vector<int> &iLaserMonADC);

  //	tags and tokens
  edm::InputTag _tagQIE11;
  edm::InputTag _tagHO;
  edm::InputTag _tagQIE10;
  edm::InputTag _taguMN;
  edm::EDGetTokenT<QIE11DigiCollection> _tokQIE11;
  edm::EDGetTokenT<HODigiCollection> _tokHO;
  edm::EDGetTokenT<QIE10DigiCollection> _tokQIE10;
  edm::EDGetTokenT<HcalUMNioDigi> _tokuMN;

  enum LaserFlag { fBadTiming = 0, fMissingLaserMon = 1, nLaserFlag = 2 };
  std::vector<hcaldqm::flag::Flag> _vflags;

  //	emap
  hcaldqm::electronicsmap::ElectronicsMap _ehashmap;
  hcaldqm::filter::HashFilter _filter_uTCA;
  hcaldqm::filter::HashFilter _filter_VME;
  std::vector<uint32_t> _vhashFEDs;

  //	Cuts and variables
  int _nevents;
  double _lowHBHE;
  double _lowHE;
  double _lowHO;
  double _lowHF;
  uint32_t _laserType;

  //	Compact
  hcaldqm::ContainerXXX<double> _xSignalSum;
  hcaldqm::ContainerXXX<double> _xSignalSum2;
  hcaldqm::ContainerXXX<int> _xEntries;
  hcaldqm::ContainerXXX<double> _xTimingSum;
  hcaldqm::ContainerXXX<double> _xTimingSum2;
  hcaldqm::ContainerXXX<double> _xTimingRefLMSum;  // For computation of channel-by-channel mean timing w.r.t. lasermon
  hcaldqm::ContainerXXX<double> _xTimingRefLMSum2;
  hcaldqm::ContainerXXX<int> _xNBadTimingRefLM;  // Count channels with bad timing
  hcaldqm::ContainerXXX<int> _xNChs;             // number of channels per FED as in emap

  //	1D
  hcaldqm::Container1D _cSignalMean_Subdet;
  hcaldqm::Container1D _cSignalRMS_Subdet;
  hcaldqm::Container1D _cSignalMeanQIE1011_Subdet;
  hcaldqm::Container1D _cSignalRMSQIE1011_Subdet;
  hcaldqm::Container1D _cTimingMean_Subdet;
  hcaldqm::Container1D _cTimingRMS_Subdet;

  hcaldqm::Container1D _cADC_SubdetPM;

  //	Prof1D
  hcaldqm::ContainerProf1D _cShapeCut_FEDSlot;
  hcaldqm::ContainerProf1D _cTimingvsEvent_SubdetPM;
  hcaldqm::ContainerProf1D _cSignalvsEvent_SubdetPM;
  hcaldqm::ContainerProf1D _cTimingvsLS_SubdetPM;
  hcaldqm::ContainerProf1D _cSignalvsLS_SubdetPM;
  hcaldqm::ContainerProf1D _cSignalvsLSQIE1011_SubdetPM;
  hcaldqm::ContainerProf1D _cTimingvsBX_SubdetPM;
  hcaldqm::ContainerProf1D _cSignalvsBX_SubdetPM;
  hcaldqm::ContainerProf1D _cSignalvsBXQIE1011_SubdetPM;

  //	2D timing/signals
  hcaldqm::ContainerProf2D _cSignalMean_depth;
  hcaldqm::ContainerProf2D _cSignalRMS_depth;
  hcaldqm::ContainerProf2D _cSignalMeanQIE1011_depth;
  hcaldqm::ContainerProf2D _cSignalRMSQIE1011_depth;
  hcaldqm::ContainerProf2D _cTimingMean_depth;
  hcaldqm::ContainerProf2D _cTimingRMS_depth;

  hcaldqm::ContainerProf2D _cSignalMean_FEDVME;
  hcaldqm::ContainerProf2D _cSignalMean_FEDuTCA;
  hcaldqm::ContainerProf2D _cTimingMean_FEDVME;
  hcaldqm::ContainerProf2D _cTimingMean_FEDuTCA;
  hcaldqm::ContainerProf2D _cSignalRMS_FEDVME;
  hcaldqm::ContainerProf2D _cSignalRMS_FEDuTCA;
  hcaldqm::ContainerProf2D _cTimingRMS_FEDVME;
  hcaldqm::ContainerProf2D _cTimingRMS_FEDuTCA;

  //	Bad Quality and Missing Channels
  hcaldqm::Container2D _cMissing_depth;
  hcaldqm::Container2D _cMissing_FEDVME;
  hcaldqm::Container2D _cMissing_FEDuTCA;

  // Things for LASERMON
  edm::InputTag _tagLaserMon;
  edm::EDGetTokenT<QIE10DigiCollection> _tokLaserMon;

  std::vector<int> _vLaserMonIPhi;  // Laser mon digis are assigned to CBox=5, IEta=0, IPhi=[23-index] by the emap
  int _laserMonIEta;
  int _laserMonCBox;
  int _laserMonDigiOverlap;
  int _laserMonTS0;
  double _laserMonThreshold;
  std::map<HcalSubdetector, std::pair<double, double>> _thresh_timingreflm;  // Min and max timing (ref. lasermon)
  double _thresh_frac_timingreflm;  // Flag threshold (BAD) on fraction of channels with bad timing
  double _thresh_min_lmsumq;        // Threshold on minimum SumQ from lasermon, if laser is expected
  int _xMissingLaserMon;            // Counter for missing lasermon events

  hcaldqm::ContainerSingle1D _cLaserMonSumQ;
  hcaldqm::ContainerSingle1D _cLaserMonTiming;
  hcaldqm::ContainerSingleProf1D _cLaserMonSumQ_LS;       // Online
  hcaldqm::ContainerSingleProf1D _cLaserMonTiming_LS;     // Online
  hcaldqm::ContainerSingleProf1D _cLaserMonSumQ_Event;    // Local
  hcaldqm::ContainerSingleProf1D _cLaserMonTiming_Event;  // Local
  hcaldqm::Container2D _cTiming_DigivsLaserMon_SubdetPM;
  hcaldqm::ContainerProf2D _cTimingDiffLS_SubdetPM;
  hcaldqm::ContainerProf2D _cTimingDiffEvent_SubdetPM;

  //	Summaries
  hcaldqm::Container2D _cSummaryvsLS_FED;
  hcaldqm::ContainerSingle2D _cSummaryvsLS;
};

#endif
