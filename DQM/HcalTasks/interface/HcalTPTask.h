#ifndef HCALTPTASK_H
#define HCALTPTASK_H

/*
 *	file:			HcalTPTask.h
 *	Author:			Viktor Khristenko
 *	Start Date:		03/04/2015
 */

#include "DQM/HcalCommon/interface/HcalMECollection.h"
#include "DQM/HcalCommon/interface/HcalDQSource.h"

class HcalTPTask : public hcaldqm::HcalDQSource
{
	public:
		HcalTPTask(edm::ParameterSet const&);
		virtual ~HcalTPTask();

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

		//	Specializer and checker methods
		//	wtw = what to what to compare
		//	wtw=1 Col1 is iterated for ids
		//	wtw=2 Col2 is iterated for ids
		template<typename Hit>
		void specialize(Hit const& hit1, Hit const& hit2, std::string const&,
				int const);
		template<typename Hit>
		void specialize(Hit const& hit, std::string const& nameRes,
				int const wtw=1);
		template<typename Hit>
		void check(Hit const& hit, std::string const&, int const wtw);

		//	Define and Initialize Comparator for TPs
		DEFTPCOMPARATOR(HcalTrigPrimDigiCollection, HcalTriggerPrimitiveDigi);
		DEFPROCESSOR(HcalTrigPrimDigiCollection, HcalTriggerPrimitiveDigi);
};

#endif
