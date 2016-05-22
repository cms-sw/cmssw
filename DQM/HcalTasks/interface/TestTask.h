#ifndef TestTask_h
#define TestTask_h

/*
 *	file:			TestTask.h
 *	Author:			Viktor KHristenko
 *	Description:
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf1D.h"

using namespace hcaldqm;
class TestTask : public DQTask
{
	public:
		TestTask(edm::ParameterSet const&);
		virtual ~TestTask(){}

		virtual void bookHistograms(DQMStore::IBooker&,
			edm::Run const&, edm::EventSetup const&);
		virtual void endLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);

	protected:
		virtual void _process(edm::Event const&, edm::EventSetup const&);
		virtual void _resetMonitors(UpdateFreq);

		//	tags
		edm::InputTag	_tagHF;

		//	Hcal Filters
		filter::HashFilter filter_Electronics;

		//	Electronics Map
		HcalElectronicsMap const *_emap;

		//	Containers
		Container1D		_cEnergy_Subdet;
		Container1D		_cTiming_SubdetPMiphi;
		ContainerProf1D	_cEnergyvsiphi_Subdetieta;
		Container2D		_cEnergy_depth;
		ContainerProf2D	_cTiming_depth;
		Container1D		_cTiming_FEDSlot;
		Container1D		_cEnergy_CrateSpigot;
		Container1D		_cEnergy_FED;
		Container1D		_cEt_TTSubdetPM;
		Container1D		_cEt_TTSubdetPMiphi;
		Container1D		_cEt_TTSubdetieta;
		Container2D		_cTiming_FEDuTCA;
		ContainerSingle2D _cSummary;
		ContainerSingleProf1D _cPerformance;

//		ContainerProf1D	_cTiming_fCrateSlot;
//		ContainerProf1D	_cEt_TTSubdetPMiphi;
};

#endif



