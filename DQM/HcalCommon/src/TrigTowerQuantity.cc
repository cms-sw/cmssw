#include "DQM/HcalCommon/interface/TrigTowerQuantity.h"

namespace hcaldqm
{
	namespace quantity
	{
		int getValue_TTiphi(HcalTrigTowerDetId const& tid)
		{
			return tid.iphi();
		}

		int getValue_TTieta(HcalTrigTowerDetId const& tid)
		{
			return tid.ieta()<0 ? tid.ieta()+41 : tid.ieta()+40;
		}

		int getValue_TTdepth(HcalTrigTowerDetId const& tid)
		{
			return tid.depth();
		}

		int getValue_TTSubdet(HcalTrigTowerDetId const& tid)
		{
			return tid.ietaAbs()<29 ? 0 : 1;
		}

		int getValue_TTSubdetPM(HcalTrigTowerDetId const& tid)
		{
			int x = tid.ietaAbs()<29 ? 0 : 2;
			return tid.ieta()>0 ? x+1 : x;
		}

		int getValue_TTieta2x3(HcalTrigTowerDetId const& tid)
		{
			return tid.ieta()<0?tid.ieta()+32:tid.ieta()-29+4;
		}

		uint32_t getBin_TTiphi(HcalTrigTowerDetId const& tid)
		{
			return (uint32_t)(getValue_TTiphi(tid));
		}

		uint32_t getBin_TTieta(HcalTrigTowerDetId const& tid)
		{
			return (uint32_t)(getValue_TTieta(tid)+1);
		}

		uint32_t getBin_TTdepth(HcalTrigTowerDetId const& tid)
		{
			return (uint32_t)(getValue_TTdepth(tid)+1);
		}

		uint32_t getBin_TTSubdet(HcalTrigTowerDetId const& tid)
		{
			return (uint32_t)(getValue_TTSubdet(tid)+1);
		}

		uint32_t getBin_TTSubdetPM(HcalTrigTowerDetId const& tid)
		{
			return (uint32_t)(getValue_TTSubdetPM(tid)+1);
		}

		uint32_t getBin_TTieta2x3(HcalTrigTowerDetId const& tid)
		{return (uint32_t)(getValue_TTieta2x3(tid)+1);}

		HcalTrigTowerDetId getTid_TTiphi(int v)
		{
			return HcalTrigTowerDetId(1, v);
		}

		HcalTrigTowerDetId getTid_TTieta(int v)
		{
			return HcalTrigTowerDetId(v<41?v-41:v-40, 1);
		}

		HcalTrigTowerDetId getTid_TTdepth(int v)
		{
			return HcalTrigTowerDetId(1, 1, v);
		}

		HcalTrigTowerDetId getTid_TTSubdet(int v)
		{
			return HcalTrigTowerDetId(v==0?1:29, 1);
		}

		HcalTrigTowerDetId getTid_TTSubdetPM(int v)
		{
			return HcalTrigTowerDetId(v%2==0?-(v>=2?29:1):(v>=2?29:1), 1);
		}

		HcalTrigTowerDetId getTid_TTieta2x3(int v)
		{
			//	since numbering goes as
			//	-32 -29 29 32
			//	0   3   4   7
			return HcalTrigTowerDetId(v<4?-(3-v+29):(v-4)+29, 1);
		}

		std::vector<std::string> getLabels_TTiphi()
		{
			return std::vector<std::string>();
		}
		std::vector<std::string> getLabels_TTieta()
		{
			char name[10];
			std::vector<std::string> labels;
			for (int i=0; i<82; i++)
			{
				sprintf(name, "%d", getTid_TTieta(i).ieta());
				labels.push_back(name);
			}
			return labels;
		}

		std::vector<std::string> getLabels_TTieta2x3()
		{
			char name[10];
			std::vector<std::string> labels;
			for (int i=0; i<8; i++)
			{
				sprintf(name, "%d", getTid_TTieta2x3(i).ieta());
				labels.push_back(name);
			}
			return labels;
		}

		std::vector<std::string> getLabels_TTdepth()
		{
			return std::vector<std::string>();
		}

		std::vector<std::string> getLabels_TTSubdet()
		{
			std::vector<std::string> labels;
			for (int i=0; i<2; i++)
				labels.push_back(constants::TPSUBDET_NAME[
					getTid_TTSubdet(i).ietaAbs()<29?0:1]);
			return labels;
		}

		std::vector<std::string> getLabels_TTSubdetPM()
		{
			std::vector<std::string> labels;
			for (int i=0; i<4; i++)
			{
				HcalTrigTowerDetId tid = getTid_TTSubdetPM(i);
				int x = tid.ietaAbs()<29?0:2;
				labels.push_back(constants::TPSUBDETPM_NAME[
					tid.ieta()>0?x+1:x]);
				return labels;
			}
			return labels;
		}
	}
}
