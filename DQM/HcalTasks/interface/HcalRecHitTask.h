#ifndef HCALRECHITTASK_H
#define HCALRECHITTASK_H

/*
 *	file:			HcalRecHitTask.h
 *	Author:			Viktor Khristenko
 *	Start Date:		03/04/2015
 */

#include "DQM/HcalCommon/interface/HcalMECollection.h"
#include "DQM/HcalCommon/interface/HcalDQSource.h"

class HcalRecHitTask : public hcaldqm::HcalDQSource
{
	public:
		HcalRecHitTask(edm::ParameterSet const&);
		virtual ~HcalRecHitTask();

		virtual void doWork(edm::Event const&e,
				edm::EventSetup const& es);

		virtual void reset(int const);

	private:
		//	MEs Collection come from the base class
		//	Here, we only need module specific parameters
		DEFPROCESSOR(HBHERecHitCollection, HBHERecHit);
		DEFPROCESSOR(HORecHitCollection, HORecHit);
		DEFPROCESSOR(HFRecHitCollection, HFRecHit);

		//	declare a specializer
		template<typename Hit>
		void specialize(Hit const&, std::string const&, int const wtw=1);

		//	Some private Variables
		int		_numRecHits[hcaldqm::constants::STD_NUMSUBS];
};

#endif
