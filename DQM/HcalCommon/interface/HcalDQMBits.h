#ifndef HCALDQMERRORBITS_H
#define HCALDQMERRORBITS_H

/*
 *	file:			hcalDQMBits.h
 *	Author:			Viktor Khristenko
 *	StartDate:		03/04/2015
 */

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"

namespace hcaldqm
{
	namespace flags
	{
		enum DigiBits
		{
			//	Generic Error Bits/Flags
			dbDead					= 0,
			dbCapIdRotErr			= 1,
			dbSizeError				= 2,
			dbPreSizeError			= 3,
			dbBadQuality				= 4,

			//	Amplitude Specific
			dbFullAmpBad				= 5,

			//	Timing Specifc
			dbTimingBad				= 6,
			
			//	Counter of those
			nDigiBits				= 7
		};

		enum RecHitBits
		{
			//	Generic Error Bits/Flags
			rhbDead					= 0,
			
			//	Energy Specific
			rhbEnergyBad				= 1,

			//	Timing Specific 
			rhbTimingBad				= 2,

			//	Counter of those
			nRecHitBits				= 3
		};

		enum TPBits
		{
			//	Generic Error Bits/Flags
			tpbMissingData			= 0,
			tpbMissingEmul			= 1,
			tpbSizeError			= 2,
			tpbPreSizeError			= 3,

			//	MisMatching - MM
			tpbMMEt_SOI			= 4,
			tpbMMFG_SOI			= 5,
			tpbMMEt_nonSOI		= 6,
			tpbMMFG_nonSOI		= 7,

			//	Counter of those 
			nTPBits					= 8
		};

		enum RawBits
		{
			//	Generic Error Bits/Flags
			rbMissingSlot			= 0,
			rbMissingSpigot			= 1,
			rbMissingCrate			= 2,
			rbMissingFED			= 3,
			rbMMBcN					= 4,
			rbMMEvN					= 5,
			rbMMOrN					= 6,

			//	
			rbMissingData			= 7,
			rbCRCnotOK				= 8,
			rbSegmentedData			= 9,
			
			//	Generic Flags
			rbIsCapRot				= 10,
			rbFibErr				= 11,
			rbInvalidData			= 12,
			rbInvalidFlavor			= 13,
			rbEmptyEvent			= 14,
			rbOverFlowWarn			= 15,
			rbBusy					= 16,

			//	Counter of those
			nRawBits				= 7
		};

		enum UnpackerBits
		{
			ubErrorFree				= 0,
			ubUnmappedDigis			= 1,
			ubUnmappedTPDigis		= 2,
			ubSpigotFormatErrors	= 3,
			ubBadQualityDigis		= 4,
			ubTotalDigis			= 5,
			ubTotalTPDigis			= 6,
			ubEmptyEventSpigots		= 7,
			ubOFWSpigots			= 8,
			ubBusySpigots			= 9,
			nUnpackerBits			= 10
		};

		enum BitsType
		{
			kDigi			= 0,
			kRecHit			= 1,
			kTP				= 2,
			kRaw			= 3,
			nBitTypes		= 4
		};
	}

	//	Runs with Tasks to control the Status HCAL
	struct HcalDQMChStatus
	{
		HcalDQMChStatus();
		~HcalDQMChStatus();

		uint32_t	_mask;
		bool		_exists;
	};

	class HcalDQMStatusManager
	{
		public:
			HcalDQMStatusManager();
			HcalDQMStatusManager(hcaldqm::flags::BitsType);
			~HcalDQMStatusManager();

			inline void setType(hcaldqm::flags::BitsType t) {_type=t;}
			inline hcaldqm::flags::BitsType getType() {return _type;}
			void setStatus(int sub, int iieta, int iiphi, int id, uint32_t mask)
			{
				_status[sub][iieta][iiphi][id]._mask = mask;
				_status[sub][iieta][iiphi][id]._exists = true;
			}
			void reset()
			{
				for (int s=0; s<hcaldqm::constants::STD_NUMSUBS; s++)
					for (int ie=0; ie<hcaldqm::constants::STD_NUMIETAS; ie++)
						for (int ip=0; ip<hcaldqm::constants::STD_NUMIPHIS; ip++)
							for (int id=0; id<hcaldqm::constants::STD_NUMDEPTHS;
								id++)
							{
								_status[s][ie][ip][id]._mask = 0x0;
								_status[s][ie][ip][id]._exists = false;
							}
			}
			bool exists(int sub, int iieta, int iiphi, int id)
			{
				return _status[sub][iieta][iiphi][id]._exists;
			}
			uint32_t getStatus(int sub, int iieta, int iiphi, int id)
			{
				return _status[sub][iieta][iiphi][id]._mask;
			}
			

		private:
			hcaldqm::flags::BitsType		_type;
			hcaldqm::HcalDQMChStatus		_status
				[hcaldqm::constants::STD_NUMSUBS]
				[hcaldqm::constants::STD_NUMIETAS]
				[hcaldqm::constants::STD_NUMIPHIS]
				[hcaldqm::constants::STD_NUMDEPTHS];
	};
}

#endif



















