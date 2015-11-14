#ifndef HCALRAWTASK_H
#define HCALRAWTASK_H

/*
 *	file:			HcalRawTask.h
 *	Author:			Viktor Khristenko
 *	Start Date:		03/04/2015
 */

#include "DQM/HcalCommon/interface/HcalMECollection.h"
#include "DQM/HcalCommon/interface/HcalDQSource.h"

class HcalRawTask : public hcaldqm::HcalDQSource
{
	public:
		HcalRawTask(edm::ParameterSet const&);
		virtual ~HcalRawTask();

		virtual void doWork(edm::Event const&e,
				edm::EventSetup const& es);

		virtual void beginLuminosityBlock(
				edm::LuminosityBlock const&, edm::EventSetup const&);

		virtual void endLuminosityBlock(
				edm::LuminosityBlock const&, edm::EventSetup const&);

		virtual void reset(int const);

	private:
		//	declare and implement the raw processors
		DEFRAWPROCESSOR(FEDRawDataCollection);

		//	dump Hcal Unpacker Report Information
		void dumpUnpackerReport(HcalUnpackerReport const&);
		
		//	declare the raw specializer
		void specialize(FEDRawData const&, int ifed);
		
		bool isuTCA(int const ifed) const;
		//	For AMC13/uTCA Cards
		void amc13(hcal::AMC13Header const*, unsigned int const, int const);
		//	For DCC/VME Cards
		void dcc(HcalDCCHeader const*, unsigned int const, int const);

		//	Some Members
		int				_numFEDsUnpackedPerEvent;
};

#endif
