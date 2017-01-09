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

class QIE10Task : public hcaldqm::DQTask
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
		virtual void _resetMonitors(hcaldqm::UpdateFreq);

		//	tags
		edm::InputTag	_tagQIE10;
		edm::InputTag       _tagHF;
		edm::EDGetTokenT<QIE10DigiCollection> _tokQIE10;
		edm::EDGetTokenT<HFDigiCollection>  _tokHF;

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
		hcaldqm::Container2D	_cLETDCvsADC_EChannel[10];
		hcaldqm::Container2D	_cTETDCvsADC_EChannel[10];
		hcaldqm::Container1D _cLETDC_EChannel[10];
		hcaldqm::Container1D _cADC_EChannel[10];
		hcaldqm::Container2D _cOccupancy_depth;

		//	Correlation Plots for 8 vs 10
		hcaldqm::Container2D _cADCCorrelation10vs8_DChannel[10];
		hcaldqm::ContainerSingle2D _cADCCorrelation10vs8;
		hcaldqm::Container2D _cfCCorrelation10vs8_DChannel[10];
		hcaldqm::ContainerSingle2D _cfCCorrelation10vs8;

		//	Correaltion plots for 10 vs 10 - 2 PMTs only
		hcaldqm::ContainerSingle2D _cADCCorrelation10vs10_ieta30[10];
		hcaldqm::ContainerSingle2D _cADCCorrelation10vs10_ieta34[10];
		hcaldqm::ContainerSingle2D _cLETDCCorrelation10vs10_ieta30[10];
		hcaldqm::ContainerSingle2D _cLETDCCorrelation10vs10_ieta34[10];
		hcaldqm::ContainerSingle2D _cADCCorrelation10vs10;
		hcaldqm::ContainerSingle2D _cLETDCCorrelation10vs10;
		hcaldqm::ContainerSingle2D _cfCCorrelation10vs10_ieta30[10];
		hcaldqm::ContainerSingle2D _cfCCorrelation10vs10_ieta34[10];
		hcaldqm::ContainerSingle2D _cfCCorrelation10vs10;

		//	hcaldqm::Containers overall
		hcaldqm::ContainerSingleProf1D	_cShapeCut;
		hcaldqm::ContainerSingle2D		_cLETDCvsADC;
		hcaldqm::ContainerSingle2D		_cTETDCvsADC;
		hcaldqm::ContainerSingle1D		_cLETDC;
		hcaldqm::ContainerSingle1D		_cADC;
};

#endif



