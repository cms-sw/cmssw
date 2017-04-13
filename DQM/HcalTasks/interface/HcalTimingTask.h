#ifndef HCALTIMINGTASK_H
#define HCALTIMINGTASK_H

/*
 *	file:			HcalTimingTask.h
 *	Author:			Viktor Khristenko
 *	Start Date:		03/04/2015
 */

#include "DQM/HcalCommon/interface/HcalMECollection.h"
#include "DQM/HcalCommon/interface/HcalDQSource.h"

class HcalTimingTask : public hcaldqm::HcalDQSource
{
	public:
		HcalTimingTask(edm::ParameterSet const&);
		virtual ~HcalTimingTask();

		virtual void doWork(edm::Event const&e,
				edm::EventSetup const& es);

		virtual void beginLuminosityBlock(edm::LuminosityBlock const&,
				edm::EventSetup const&);
		virtual void endLuminosityBlock(edm::LuminosityBlock const&,
				edm::EventSetup const&);
		virtual void reset(int const);

	private:
		//	MEs Collection come from the base class
		//	Here, we only need module specific parameters
		template<typename Hit>
		void specialize(Hit const& hit, std::string const&,
				int const wtw=1);

		//	Define the processors
		DEFPROCESSOR(HFDigiCollection, HFDataFrame);
		DEFPROCESSOR(HBHEDigiCollection, HBHEDataFrame);

		//	Call HF Phase Scan
		template<typename HIT>
		void hf(HIT const&);
		template<typename HIT>
		void hbhe(HIT const&);
};

#endif
