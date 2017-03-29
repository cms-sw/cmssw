#ifndef DQM_HcalTasks_DigiRunSummary_h
#define DQM_HcalTasks_DigiRunSummary_h

#include "DQM/HcalCommon/interface/DQClient.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"

namespace hcaldqm
{
	class DigiRunSummary : public DQClient
	{
		public:
			DigiRunSummary(std::string const&, std::string const&,
				edm::ParameterSet const&);
			virtual ~DigiRunSummary() {}

			virtual void beginRun(edm::Run const&, edm::EventSetup const&);
			virtual void endLuminosityBlock(DQMStore::IBooker&,
				DQMStore::IGetter&, edm::LuminosityBlock const&,
				edm::EventSetup const&);
			virtual std::vector<flag::Flag> endJob(
				DQMStore::IBooker&, DQMStore::IGetter&);

		protected:
			std::vector<LSSummary> _vflagsLS;

			double _thresh_unihf;

			electronicsmap::ElectronicsMap _ehashmap;

			std::vector<uint32_t> _vhashVME, _vhashuTCA, _vhashFEDHF;
			std::vector<int> _vFEDsVME, _vFEDsuTCA;
			filter::HashFilter _filter_VME, _filter_uTCA, _filter_FEDHF;

			Container2D _cOccupancy_depth;
			bool _booked;
			MonitorElement *_meNumEvents; // number of events vs LS

			ContainerXXX<uint32_t> _xDead, _xDigiSize, _xUniHF,
				_xUni, _xNChs, _xNChsNominal;

			//	flag enum
			enum DigiLSFlag
			{
				fDigiSize = 0,
				fNChsHF=1,
				nLSFlags=2, // defines the boundayr between lumi based and run
				//	 based flags
				fUniHF=3,
				fDead=4,
				nDigiFlag = 5
			};
	};
}

#endif
