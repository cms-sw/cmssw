#ifndef DQM_HcalCommon_Utilities_h
#define DQM_HcalCommon_Utilities_h

/*
 *	file:			Utilities.h
 *	Author:			Viktor Khristenko
 *
 *	Description:
 *		Utility functions
 */

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"
#include "DQM/HcalCommon/interface/Constants.h"

namespace hcaldqm
{
	using namespace constants;
	namespace utilities
	{
		/*
		 *	Some useful functions for QIE10/11 Data Frames
		 */
		template<typename FRAME>
		double aveTS_v10(FRAME const& frame, double ped=0, int i=0,int j=3)
		{
			double sumQ = 0;
			double sumQT = 0;
			for (int ii=i; ii<=j; ii++)
			{
				double q = constants::adc2fC[frame[ii].adc()]-ped;
				sumQ += q;
				sumQT += (ii+1)*q;
			}

			return sumQ>0 ? sumQT/sumQ-1 : GARBAGE_VALUE;
		}

		template<typename FRAME>
		double sumQ_v10(FRAME const& frame, double ped, int i=0, int j=3)
		{
			double sumQ = 0;
			for (int ii=i; ii<=j; ii++)
				sumQ += constants::adc2fC[frame[ii].adc()]-ped;
			return sumQ;
		}

		/*
		 *	Some useful functions on QIE8 digis
		 */
		template<typename DIGI>
		int maxTS(DIGI const& digi, double ped=0)
		{
			int maxTS = -1;
			double maxQ = -100;
			for (int i=0; i<digi.size(); i++)
				if((digi.sample(i).nominal_fC()-ped)>maxQ)
				{
					maxQ = digi.sample(i).nominal_fC()-ped;
					maxTS = i;
				}
			return maxTS;
		}

		template<typename DIGI>
		double aveTS(DIGI const& digi, double ped=0, int i=0, int j=3)
		{
			double sumQ = 0;
			double sumQT = 0;
			for (int ii=i; ii<=j; ii++)
			{
				sumQ+=digi.sample(ii).nominal_fC()-ped;
				sumQT +=(ii+1)*(digi.sample(ii).nominal_fC()-ped);
			}
			
			return sumQ>0 ? sumQT/sumQ-1 : GARBAGE_VALUE;
		}

		template<typename DIGI>
		double sumQ(DIGI const& digi, double ped, int i=0, int j=3)
		{
			double sum=0;
			for (int ii=i; ii<=j; ii++)
				sum+=(digi.sample(ii).nominal_fC()-ped);
			return sum;
		}

		template<typename DIGI>
		double aveQ(DIGI const& digi, double ped, int i=0, int j=3)
		{
			return sumQ<DIGI>(digi, ped, i, j)/(j-i+1);
		}

		template<typename DIGI>
		double sumADC(DIGI const& digi, double ped, int i=0, int j=3)
		{
			double sum = 0;
			for (int ii=i; ii<=j; ii++)
				sum+=digi.sample(ii).adc()-ped;
			return sum;
		}

		template<typename DIGI>
		double aveADC(DIGI const& digi, double ped, int i=0, int j=3)
		{
			return sumADC<DIGI>(digi, ped, i, j)/(j-i+1);
		}

		/*
		 *	Log Functions
		 */
		template<typename STDTYPE>
		void dqmdebug(STDTYPE const& x, int debug=0)
		{
			if (debug==0)
				return;
			std::cout << "%MSG" << std::endl;
			std::cout << "%MSG-d HCALDQM::" << x;
			std::cout << std::endl;
		}

		/*
		 *	Useful Detector/Electronics/TrigTower Functions. 
		 *	For Fast Detector Validity Check
		 */
		int getTPSubDet(HcalTrigTowerDetId const&);
		int getTPSubDetPM(HcalTrigTowerDetId const&);

		//	returns a list of FEDs sorted.
		std::vector<int> getFEDList(HcalElectronicsMap const*);
		std::vector<int> getFEDVMEList(HcalElectronicsMap const*);
		std::vector<int> getFEDuTCAList(HcalElectronicsMap const*);

        std::pair<uint16_t, uint16_t> fed2crate(int fed);
		uint16_t crate2fed(int crate, int slot);
		bool isFEDHBHE(HcalElectronicsId const&);
		bool isFEDHF(HcalElectronicsId const&);
		bool isFEDHO(HcalElectronicsId const&);

		/**
		 *	This is wrap around in case hashing scheme changes in the future
		 */
		uint32_t hash(HcalDetId const&);
		uint32_t hash(HcalElectronicsId const&);
		uint32_t hash(HcalTrigTowerDetId const&);

		/*
		 *	Orbit Gap Related
		 */	
		std::string ogtype2string(OrbitGapType type);
	}
}

#endif
