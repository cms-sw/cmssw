#ifndef HCALDQSOURCEEX_H
#define HCALDQSOURCEEX_H

/*
 *	file:			HcalDQSourceEx.h
 *	Author:			Viktor Khristenko
 *	Start Date:		03/04/2015
 */

#include "DQM/HcalCommon/interface/HcalMECollection.h"
#include "DQM/HcalCommon/interface/HcalDQSource.h"


/*
 *	Class HcalDQSourceEx:
 *	Working example of an Hcal DQM Source Module using new Hcal DQM FW
 */
class HcalDQSourceEx : public hcaldqm::HcalDQSource
{
	public:
		HcalDQSourceEx(edm::ParameterSet const&);
		virtual ~HcalDQSourceEx();

		//	To be reimplemented
		//	Entry point to do all the Work
		virtual void doWork(edm::Event const&e, 
				edm::EventSetup const& es);

//	private:
		//	MEs Collection comes from base class
		//	Here, we need only module specific parameters

	};

#endif
