
#include "DQM/HcalCommon/interface/DetectorQuantity.h"

namespace hcaldqm
{
	namespace quantity
	{
		int getValue_iphi(HcalDetId const& did)
		{
			return did.iphi();
		}

		int getValue_ieta(HcalDetId const& did)
		{
			int x = did.ieta();
			if (x<0)
				x = did.subdet()==HcalForward ? x+41 : x+42;
			else
				x = did.subdet()==HcalForward ? x+42 : x+41;
			return x;
		}

		int getValue_depth(HcalDetId const& did)
		{
			return did.depth();
		}

		int getValue_Subdet(HcalDetId const& did)
		{
			return did.subdet()-1;
		}

		int getValue_SubdetPM(HcalDetId const& did)
		{
			return did.ieta()<0 ? 2*(did.subdet()-1) : 
				2*(did.subdet()-1)+1;
		}

		uint32_t getBin_iphi(HcalDetId const& did)
		{
			return (uint32_t)(did.iphi());
		}

		uint32_t getBin_ieta(HcalDetId const& did)
		{
			return (uint32_t)(getValue_ieta(did)+1);
		}

		uint32_t getBin_depth(HcalDetId const& did)
		{
			return (uint32_t)(did.depth());
		}

		uint32_t getBin_Subdet(HcalDetId const& did)
		{
			return (uint32_t)(did.subdet());
		}

		uint32_t getBin_SubdetPM(HcalDetId const& did)
		{
			return ( uint32_t)(getValue_SubdetPM(did)+1);
		}

		HcalDetId getDid_iphi(int v)
		{
			return HcalDetId(HcalBarrel, v, 1, 1);
		}

		HcalDetId getDid_ieta(int v)
		{
			return HcalDetId(HcalBarrel,
				v<=41?(v<=12?v-41:v-42):(v>=71?v-42:v-41), 1, 1);
		}

		HcalDetId getDid_depth(int v)
		{
			return HcalDetId(HcalBarrel, 1, 1, v);
		}

		HcalDetId getDid_Subdet(int v)
		{
			return HcalDetId((HcalSubdetector)(v+1), 1, 1, 1);
		}

		HcalDetId getDid_SubdetPM(int v)
		{
			return HcalDetId((HcalSubdetector)(v/2+1), v%2==0?1:-1, 1, 1);
		}

		std::vector<std::string> getLabels_iphi()
		{
			return std::vector<std::string>();
		}

		std::vector<std::string> getLabels_ieta()
		{
			std::vector<std::string> labels;
			char name[10];
			for (int i=0; i<84; i++)
			{
				sprintf(name, "%d", getDid_ieta(i).ieta());
				labels.push_back(std::string(name));
			}
			return labels;
		}

		std::vector<std::string> getLabels_depth()
		{
			return std::vector<std::string>();
		}

		std::vector<std::string> getLabels_Subdet()
		{
			std::vector<std::string> labels;
			for (int i=0; i<4; i++)
				labels.push_back(constants::SUBDET_NAME[i]);
			return labels;
		}

		std::vector<std::string> getLabels_SubdetPM()
		{
			std::vector<std::string> labels;
			for (int i=0; i<8; i++)
				labels.push_back(constants::SUBDETPM_NAME[i]);
			return labels;
		}
	}
}
