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

using namespace hcaldqm;
using namespace hcaldqm::filter;
using namespace hcaldqm::electronicsmap;
class QIE10Task : public DQTask
{
	public:
		QIE10Task(edm::ParameterSet const&);
		virtual ~QIE10Task(){}

		virtual void bookHistograms(DQMStore::IBooker&,
			edm::Run const&, edm::EventSetup const&);
		virtual void endLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);

	protected:
		virtual void _process(edm::Event const&, edm::EventSetup const&);
		virtual void _resetMonitors(UpdateFreq);

		//	tags
		edm::InputTag	_tagQIE10;
		edm::EDGetTokenT<QIE10DigiCollection> _tokQIE10;

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
		Container2D	_cLETDCvsADC_EChannel[10];
		Container2D	_cTETDCvsADC_EChannel[10];
		Container1D _cLETDC_EChannel[10];
		Container1D _cADC_EChannel[10];
		Container2D _cOccupancy_depth;

		//	Containers overall
		ContainerSingleProf1D	_cShapeCut;
		ContainerSingle2D		_cLETDCvsADC;
		ContainerSingle2D		_cTETDCvsADC;
		ContainerSingle1D		_cLETDC;
		ContainerSingle1D		_cADC;
};

#endif



