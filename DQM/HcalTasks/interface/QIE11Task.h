#ifndef QIE11Task_h
#define QIE11Task_h

/*
 *	file:			QIE11Task.h
 *	Author:			Viktor KHristenko
 *	Description:
 *		TestTask of QIE11 Read out
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf1D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle1D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/HashFilter.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"

class QIE11Task : public hcaldqm::DQTask {
public:
  QIE11Task(edm::ParameterSet const &);
  ~QIE11Task() override {}

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void globalEndLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

protected:
  void _process(edm::Event const &, edm::EventSetup const &) override;
  void _resetMonitors(hcaldqm::UpdateFreq) override;
  bool _isApplicable(edm::Event const &) override;

  //	tags
  edm::InputTag _tagQIE11;
  edm::EDGetTokenT<QIE11DigiCollection> _tokQIE11;

  edm::InputTag _taguMN;
  edm::EDGetTokenT<HcalUMNioDigi> _tokuMN;
  edm::ESGetToken<HcalDbService, HcalDbRecord> hcalDbServiceToken_;

  //	cuts/constants from input
  double _cut;
  int _ped;
  int _laserType;
  int _eventType;

  //	filters
  hcaldqm::filter::HashFilter _filter_C34;
  hcaldqm::filter::HashFilter _filter_slot[2];
  hcaldqm::filter::HashFilter _filter_timingChannels[4];

  //	Electronics Maps/Hashes
  hcaldqm::electronicsmap::ElectronicsMap _ehashmap;

  //	hcaldqm::Containers
  hcaldqm::ContainerProf1D _cShapeCut_EChannel[2];
  hcaldqm::Container2D _cLETDCvsADC_EChannel[10][2];
  hcaldqm::Container2D _cLETDCvsTS_EChannel[2];
  hcaldqm::Container1D _cLETDC_EChannel[10][2];
  hcaldqm::Container1D _cLETDCTime_EChannel[2];
  hcaldqm::Container1D _cADC_EChannel[10][2];
  hcaldqm::Container2D _cOccupancy_depth;

  //	hcaldqm::Containers overall
  hcaldqm::ContainerSingleProf1D _cShapeCut;
  hcaldqm::ContainerSingle2D _cLETDCvsADC;
  hcaldqm::ContainerSingle2D _cLETDCTimevsADC;
  hcaldqm::ContainerSingle1D _cLETDC;
  hcaldqm::ContainerSingle1D _cADC;
};

#endif
