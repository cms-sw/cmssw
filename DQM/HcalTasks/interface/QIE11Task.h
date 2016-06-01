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

using namespace hcaldqm;
using namespace hcaldqm::filter;
using namespace hcaldqm::electronicsmap;
class QIE11Task : public DQTask
{
	public:
		QIE11Task(edm::ParameterSet const&);
		virtual ~QIE11Task(){}

		virtual void bookHistograms(DQMStore::IBooker&,
			edm::Run const&, edm::EventSetup const&);
		virtual void endLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);

	protected:
		virtual void _process(edm::Event const&, edm::EventSetup const&);
		virtual void _resetMonitors(UpdateFreq);

		//	tags
		edm::InputTag	_tagQIE11;
		edm::EDGetTokenT<QIE11DigiCollection> _tokQIE11;

		//	cuts/constants from input
		double _cut;
		int _ped;

		//	filters
		HashFilter _filter_C36;

		//	Electronics Maps/Hashes
		HcalElectronicsMap const* _emap;
		ElectronicsMap _ehashmap;
		
		//	Containers
		ContainerProf1D	_cShapeCut_EChannel;
		Container2D	_cTDCvsADC_EChannel[10];
		Container1D _cTDC_EChannel[10];
		Container1D _cADC_EChannel[10];
		Container2D _cOccupancy_depth;

		//	Containers overall
		ContainerSingleProf1D	_cShapeCut;
		ContainerSingle2D		_cTDCvsADC;
		ContainerSingle1D		_cTDC;
		ContainerSingle1D		_cADC;
};

#endif



