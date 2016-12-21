#ifndef HCALDQSOURCE_H
#define HCALDQSOURCE_H

/*
 *	file:			HcalDQSource.h
 *	Author:			Viktor Khristenko
 *	Start Date:		03/04/2015
 *
 *	TODO:
 *		1) Extracting the Calibration Type
 *		2) Other Source-specific functionality
 */

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DQM/HcalCommon/interface/HcalMECollection.h"
#include "DQM/HcalCommon/interface/HcalDQMonitor.h"
#include "DQM/HcalCommon/interface/HcalDQMBits.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <vector>

namespace hcaldqm
{
	/*
	 *	HcalDQSource Class - Base Class for DQSources
	 */
	class HcalDQSource : public DQMEDAnalyzer, public HcalDQMonitor
	{
		public:
			HcalDQSource(edm::ParameterSet const&);
			virtual ~HcalDQSource();

			//	Genetic doWork function for all DQSources
			//	Note: Different def from DQClients
			virtual void doWork(edm::Event const&e, 
					edm::EventSetup const& es) = 0;

			//	Functions which have to be reimplemented from DQMEDAnalyzer
			virtual void analyze(edm::Event const& e, edm::EventSetup const& es);
			virtual void bookHistograms(DQMStore::IBooker &ib, edm::Run const&,
					edm::EventSetup const&);
			virtual void dqmBeginRun(edm::Run const&, edm::EventSetup const&);

			virtual void beginLuminosityBlock(edm::LuminosityBlock const& ,
					edm::EventSetup const&);
			virtual void endLuminosityBlock(edm::LuminosityBlock const& ,
					edm::EventSetup const&);
			
		protected:
			//	Apply Reset/Update if neccessary
			//	Resets the contents of MEs
			//	periodflag: 0 for Event Reset and 1 for LS Reset
			virtual void reset(int const periodflag);

		protected:
			//	Functions specific for Sources, but generic to all of them
			void extractCalibType(edm::Event const&);
			bool isAllowedCalibType();

		protected:
			HcalMECollection		_mes;
	};
}

//	The use of this macro must be properly controlled!
//	Becaue the COLLECTIONTYPE and HITTYPE must go in accord with each other
//	
//	Desc: If this macro is used to look at TPs, wtw=1 means you are iterating 
//	over Data and wtw=2 means iterating over Emulator
#define DEFPROCESSOR(COLLECTIONTYPE, HITTYPE) \
	void process(COLLECTIONTYPE const& c, std::string const& nameRes,	\
			int const wtw=1) \
	{	\
		for (COLLECTIONTYPE::const_iterator it=c.begin(); it!=c.end(); ++it)	\
		{	\
			this->debug_(" Collection isn't empty");	\
			const HITTYPE hit = (const HITTYPE)(*it);	\
			if ((nameRes=="HB" && hit.id().subdet()!=HcalBarrel)	\
					|| (nameRes=="HE" && hit.id().subdet()!=HcalEndcap)	\
					|| (nameRes=="HF" &&	\
						!hcaldqm::packaging::isHFTrigTower(hit.id().ietaAbs()))	\
					|| (nameRes=="HBHE" &&	\
						!hcaldqm::packaging::isHBHETrigTower(hit.id().ietaAbs())))	\
				continue;	\
			specialize<HITTYPE>(hit, nameRes, wtw);	\
		}	\
	}

//	NOTE: specializer that is being used inside of the process here
//	will not be templated, but should simply be 
//	defined as the specializer(RawData, int) in the inherited class
//
//	Cuts applied are the same ones as applied in the HcalUnpacker
#define DEFRAWPROCESSOR(COLLECTIONTYPE)	\
	void process(COLLECTIONTYPE const& c)	\
	{	\
		for (int ifed=FEDNumbering::MINHCALFEDID;	\
				ifed<=FEDNumbering::MAXHCALuTCAFEDID;	\
				ifed++)	\
		{	\
			if (ifed>FEDNumbering::MAXHCALFEDID &&	\
					ifed<FEDNumbering::MINHCALuTCAFEDID)	\
				continue;	\
			FEDRawData const& raw = c.FEDData(ifed);	\
			if (raw.size() < hcaldqm::constants::RAWDATASIZE_EMPTY)\
				continue;	\
			specialize(raw, ifed);	\
		}	\
	}	

//	The use of this macro must be properly controlled!
//	Becaue the COLLECTIONTYPE and HITTYPE must go in accord with each other
#define DEFCOMPARATOR(COLLECTIONTYPE, HITTYPE) \
	void process(COLLECTIONTYPE const& c1, COLLECTIONTYPE const& c2, \
			std::string const& nameRes, int const wtw) \
	{	\
		for (COLLECTIONTYPE::const_iterator it1=c1.begin();		\
				it1!=c1.end(); ++it1)	\
		{	\
			const HITTYPE hit1 = (const HITTYPE)(*it1);	\
			COLLECTIONTYPE::const_iterator it2=c2.find(hit1.id());	\
			if (it2==c2.end())	\
			{	\
				check<HITTYPE>(hit1, nameRes, wtw);	\
				this->debug_("Didn't find an id with such DetID");	\
				continue;	\
			}	\
			const HITTYPE hit2 = (const HITTYPE)*it2;	\
			if ((nameRes=="HB" && hit1.id().subdet()!=HcalBarrel)	\
					|| (nameRes=="HE" && hit1.id().subdet()!=HcalEndcap))	\
				continue;	\
			specialize<HITTYPE>(hit1, hit2, nameRes, wtw);	\
		}	\
	}

//	The use of this macro must be properly controlled!
//	Becaue the COLLECTIONTYPE and HITTYPE must go in accord with each other
//	
//	Desc:	We deliberately separate HF from HBHE
#define DEFTPCOMPARATOR(COLLECTIONTYPE, HITTYPE) \
	void process(COLLECTIONTYPE const& c1, COLLECTIONTYPE const& c2, \
			std::string const& nameRes, int const wtw) \
	{	\
		for (COLLECTIONTYPE::const_iterator it1=c1.begin();		\
				it1!=c1.end(); ++it1)	\
		{	\
			this->debug_("We have TPs");	\
			const HITTYPE hit1 = (const HITTYPE)(*it1);	\
			if ((nameRes=="HF" && \
						!hcaldqm::packaging::isHFTrigTower(hit1.id().ietaAbs()))	\
					|| (nameRes=="HBHE" &&	\
						!hcaldqm::packaging::isHBHETrigTower(hit1.id().ietaAbs())))	\
				continue;	\
			COLLECTIONTYPE::const_iterator it2=c2.find(	\
					HcalTrigTowerDetId(hit1.id().ieta(), hit1.id().iphi(),	\
						hit1.id().depth()==1 ? 0 : 1));	\
			if (it2==c2.end())	\
			{	\
				check<HITTYPE>(hit1, nameRes, wtw);	\
				this->debug_("Didn't find a matching Tower");	\
				continue;	\
			}	\
			const HITTYPE hit2 = (const HITTYPE)*it2;	\
			specialize<HITTYPE>(hit1, hit2, nameRes, wtw);	\
		}	\
	}


//	Define a specializer
//#define DEFSPECIALIZER()

#endif

