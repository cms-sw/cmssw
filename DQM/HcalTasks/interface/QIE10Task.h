#ifndef QIE10Task_h
#define QIE10Task_h

/*
 *	file:			QIE10Task.h
 *	Author:			Viktor KHristenko
 *	Description:
 *		Task for QIE10 Read out
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

class QIE10Task : public hcaldqm::DQTask {
public:
  QIE10Task(edm::ParameterSet const&);
  ~QIE10Task() override {}

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void dqmEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

protected:
  void _process(edm::Event const&, edm::EventSetup const&) override;
  void _resetMonitors(hcaldqm::UpdateFreq) override;

  //	tags
  edm::InputTag _tagQIE10;
  edm::InputTag _tagHF;
  edm::EDGetTokenT<QIE10DigiCollection> _tokQIE10;
  edm::EDGetTokenT<HFDigiCollection> _tokHF;

  //	cuts/constants from input
  double _cut;
  int _ped;

  //	filters
  hcaldqm::filter::HashFilter _filter_slot[36];

  //	Electronics Maps/Hashes
  hcaldqm::electronicsmap::ElectronicsMap _ehashmap;

  //	hcaldqm::Containers
  hcaldqm::ContainerProf1D _cShapeCut_EChannel[36];
  hcaldqm::Container2D _cLETDCvsADC_EChannel[10][36];
  hcaldqm::Container2D _cLETDCvsTS_EChannel[36];
  hcaldqm::Container1D _cLETDC_EChannel[10][36];
  hcaldqm::Container1D _cADC_EChannel[10][36];
  hcaldqm::Container1D _cLETDCTime_EChannel[36];

  //	hcaldqm::Containers overall
  hcaldqm::ContainerSingleProf1D _cShapeCut;
  hcaldqm::ContainerSingle2D _cLETDCTimevsADC;
  hcaldqm::ContainerSingle2D _cLETDCvsADC;
  hcaldqm::ContainerSingle1D _cLETDC;
  hcaldqm::ContainerSingle1D _cADC;

  //occupancy per crate/slot
  hcaldqm::Container2D _cOccupancy_Crate;
  hcaldqm::Container2D _cOccupancy_CrateSlot;

  // Detector coordinates
  hcaldqm::Container2D _cOccupancy_depth;
};

#endif
