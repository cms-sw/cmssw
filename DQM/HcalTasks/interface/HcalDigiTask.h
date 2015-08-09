#ifndef HCALDIGITASK_H
#define HCALDIGITASK_H

/*
 *	file:			HcalDigiTask.h
 *	Author:			Viktor Khristenko
 *	Start Date:		03/04/2015
 */

#include "DQM/HcalCommon/interface/HcalMECollection.h"
#include "DQM/HcalCommon/interface/HcalDQSource.h"
#include "DQM/HcalCommon/interface/HcalMasters.h"

class HcalDigiTask : public hcaldqm::HcalDQSource
{
	public:
		HcalDigiTask(edm::ParameterSet const&);
		virtual ~HcalDigiTask();

		virtual void doWork(edm::Event const&e,
				edm::EventSetup const& es);

		virtual void beginLuminosityBlock(edm::LuminosityBlock const&,
				edm::EventSetup const&);
		virtual void endLuminosityBlock(edm::LuminosityBlock const&,
				edm::EventSetup const&);

		virtual void reset(int const);
	
	private:
		//	declare the template for specializing
		template<typename Hit>
		void specialize(Hit const& hit, std::string const&, int const wtw=1);
		
		//	define and initialize the collection processors
		DEFPROCESSOR(HBHEDigiCollection, HBHEDataFrame);
		DEFPROCESSOR(HODigiCollection, HODataFrame);
		DEFPROCESSOR(HFDigiCollection, HFDataFrame);

	private:
		//	MEs Collection come from the base class
		//	Here, we only need module specific parameters
		int				_ornMsgTime;
		//	number of digis for HB(0), HE(1), HO(2), HF(3) for an event
		int				_numDigis_wZSCut[hcaldqm::constants::STD_NUMSUBS];
		int				_numDigis_NoZSCut[hcaldqm::constants::STD_NUMSUBS];

		//	Use DcsMaster
		hcaldqm::HcalDcsMaster	_dcs;
};

#endif
