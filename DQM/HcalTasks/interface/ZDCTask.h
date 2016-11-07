#ifndef ZDCTask_h
#define ZDCTask_h

/*
 *	file:			ZDCTask.h
 *	Author:			Viktor KHristenko
 *	Description:
 *		Task for ZDC Read out
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

class ZDCTask : public hcaldqm::DQTask
{
	public:
		ZDCTask(edm::ParameterSet const&);
		virtual ~ZDCTask(){}

		virtual void bookHistograms(DQMStore::IBooker&,
			edm::Run const&, edm::EventSetup const&);
		virtual void endLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);

	protected:
		virtual void _process(edm::Event const&, edm::EventSetup const&);
		virtual void _resetMonitors(hcaldqm::UpdateFreq);

		//	tags
		edm::InputTag	_tagQIE10;
		edm::EDGetTokenT<ZDCDigiCollection> _tokQIE10;

		//	cuts/constants from input
		double _cut;
		int _ped;

		//	filters
		hcaldqm::filter::HashFilter _filter_C36;
		hcaldqm::filter::HashFilter _filter_DA;

		//	Electronics Maps/Hashes
		HcalElectronicsMap const* _emap;
		hcaldqm::electronicsmap::ElectronicsMap _ehashmap;
		
		//	hcaldqm::Containers
		hcaldqm::ContainerProf1D	_cShapeCut_EChannel;
		hcaldqm::ContainerProf1D	_cShape_EChannel;
		hcaldqm::Container2D	_cLETDCvsADC_EChannel[10];
		hcaldqm::Container2D	_cTETDCvsADC_EChannel[10];
		hcaldqm::Container1D _cLETDC_EChannel[10];
		hcaldqm::Container1D _cADC_EChannel[10];


		//	hcaldqm::Containers overall
		hcaldqm::ContainerSingleProf1D	_cShapeCut;
		hcaldqm::ContainerSingleProf1D	_cShape;
		hcaldqm::ContainerSingle2D		_cLETDCvsADC;
		hcaldqm::ContainerSingle2D		_cTETDCvsADC;
		hcaldqm::ContainerSingle1D		_cLETDC;
		hcaldqm::ContainerSingle1D		_cADC;
};

#endif



