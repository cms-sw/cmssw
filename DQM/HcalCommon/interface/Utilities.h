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
	namespace utilities
	{
		/*
		 * adc2fC lookup from conditions
		 * fC values are stored in CaloSamples tool
		 */
		template<class Digi>
		CaloSamples loadADC2fCDB(const edm::ESHandle<HcalDbService>& conditions, const HcalDetId did, const Digi& digi) {
			CaloSamples calo_samples;
			const HcalQIECoder* channelCoder = conditions->getHcalCoder(did);
			const HcalQIEShape* shape = conditions->getHcalShape(channelCoder);
			HcalCoderDb coder(*channelCoder, *shape);
			coder.adc2fC(digi, calo_samples);
			return calo_samples;
		}

		// Get pedestal-subtracted charge
		template<class Digi> 
		double adc2fCDBMinusPedestal(const edm::ESHandle<HcalDbService>& conditions, const CaloSamples& calo_samples, const HcalDetId did, const Digi& digi, unsigned int n) {
			HcalCalibrations calibrations = conditions->getHcalCalibrations(did);
			int capid = digi[n].capid();
			return calo_samples[n] - calibrations.pedestal(capid);
		}

		template<class Digi>
		double aveTSDB(const edm::ESHandle<HcalDbService>& conditions, const CaloSamples& calo_samples, const HcalDetId did, const Digi& digi, unsigned int i_start, unsigned int i_end) {
			double sumQ = 0;
			double sumQT = 0;
			for (unsigned int i = i_start; i <= i_end; ++i) {
				double q = adc2fCDBMinusPedestal(conditions, calo_samples, did, digi, i);
				sumQ += q;
				sumQT += (i+1)*q;
			}
			return (sumQ > 0 ? sumQT/sumQ-1 : constants::GARBAGE_VALUE);
		}

		template<class Digi>
		double sumQDB(const edm::ESHandle<HcalDbService>& conditions, const CaloSamples& calo_samples, const HcalDetId did, const Digi& digi, unsigned int i_start, unsigned int i_end) {
			double sumQ = 0;
			for (unsigned int i = i_start; i <= i_end; ++i) {
				sumQ += adc2fCDBMinusPedestal(conditions, calo_samples, did, digi, i);
			}
			return sumQ;
		}

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

			return sumQ>0 ? sumQT/sumQ-1 : constants::GARBAGE_VALUE;
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
			
			return sumQ>0 ? sumQT/sumQ-1 : constants::GARBAGE_VALUE;
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

		// Get a list of all crates
		std::vector<int> getCrateList(HcalElectronicsMap const* emap);
		std::map<int, uint32_t> getCrateHashMap(HcalElectronicsMap const* emap);

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
		std::string ogtype2string(constants::OrbitGapType type);

		int getRBX(uint32_t iphi);
	}
}

#endif
