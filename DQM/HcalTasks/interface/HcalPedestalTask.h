#ifndef HCALPEDESTALTASK_H
#define HCALPEDESTALTASK_H

/*
 *	file:			HcalPedestalTask.h
 *	Author:			Viktor Khristenko
 *	Start Date:		03/04/2015
 */

#include "DQM/HcalCommon/interface/HcalMECollection.h"
#include "DQM/HcalCommon/interface/HcalDQSource.h"

class HcalPedestalTask : public hcaldqm::HcalDQSource
{
	public:
		HcalPedestalTask(edm::ParameterSet const&);
		virtual ~HcalPedestalTask();

		virtual void doWork(edm::Event const&e,
				edm::EventSetup const& es);

		virtual void beginLuminosityBlock(edm::LuminosityBlock const&,
				edm::EventSetup const&);
		virtual void endLuminosityBlock(edm::LuminosityBlock const&,
				edm::EventSetup const&);
		virtual void endRun(const edm::Run& r, const edm::EventSetup& es);

		virtual void reset(int const);
		virtual bool isApplicable(edm::Event const&);
//		virtual bool shouldBook();

	private:
		//	MEs Collection come from the base class
		//	Here, we only need module specific parameters
		template<typename Hit>
		void specialize(Hit const& hit, std::string const&,
				int const wtw=1);
		void publish();

		DEFPROCESSOR(HBHEDigiCollection, HBHEDataFrame);
		DEFPROCESSOR(HODigiCollection, HODataFrame);
		DEFPROCESSOR(HFDigiCollection, HFDataFrame);

		hcaldqm::HcalDQPedData _pedData[hcaldqm::constants::STD_NUMSUBS]
			[hcaldqm::constants::STD_NUMIETAS][hcaldqm::constants::STD_NUMIPHIS]
			[hcaldqm::constants::STD_NUMDEPTHS][hcaldqm::constants::STD_NUMCAPS];
		hcaldqm::packaging::Packager _packager[hcaldqm::constants::STD_NUMSUBS];
};

#endif
