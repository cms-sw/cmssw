#include "DQM/HcalCommon/interface/Utilities.h"

namespace hcaldqm
{
	using namespace constants;
	namespace utilities
	{

		/*
		 *	Useful Detector Functions. For Fast Detector Validity Check
		 */
		bool validDetId(HcalSubdetector sub, int ieta, int iphi, int depth)
		{
			int ie(abs(ieta));
			return ((iphi>=1) && 
					(iphi<=72) &&
					(depth>=1) &&
					(ie>=1) &&
					(((sub==HcalBarrel) &&
					  (((ie<=14) &&
						(depth==1)) ||
					   (((ie==15) || (ie==16)) &&
						(depth<=2)))) ||
					 ((sub==HcalEndcap) &&
					  (((ie==16) &&
						(depth==3)) ||
					   ((ie==17) &&
						(depth==1)) ||
					   ((ie>=18) &&
						(ie<=20) &&
						(depth<=2)) ||
					   ((ie>=21) &&
						(ie<=26) &&
						(depth<=2) &&
						(iphi%2==1)) ||
					   ((ie==29) &&
						(depth<=2) &&
						(iphi%2==1)))) ||
					 ((sub==HcalOuter) &&
					  (ie<=15) &&
					  (depth==4)) ||
					 ((sub==HcalForward) &&
					  (depth<=2) &&
					  (((ie>=29) &&
						(ie<=39) &&
						(iphi%2==1)) ||
					   ((ie>=40) &&
						(ie<=41) &&
						(iphi%4==3))))));
		}

		bool validDetId(HcalDetId const& did)
		{
			return validDetId(did.subdet(), 
				did.ieta(), did.iphi(), did.depth());
		}

		int getTPSubDet(HcalTrigTowerDetId const& tid)
		{	
			return tid.ietaAbs()<29 ? 0 : 1;
		}

		int getTPSubDetPM(HcalTrigTowerDetId const& tid)
		{
			int ieta = tid.ieta();
			if (ieta<0 && ieta>-29)
				return 0;
			else if (ieta>0 && ieta<29)
				return 1;
			else if (ieta<=-29)
				return 2;
			else 
				return 3;
			return 0;
		}

		int getFEDById(int id)
		{
			int fed = 700;
			if (id>=FED_VME_NUM)
				fed = FED_uTCA_MIN + FED_uTCA_DELTA*(id-FED_VME_NUM);
			else 
				fed = FED_VME_MIN + id;
			return fed;
		}
		
		int getIdByFED(int fed)
		{
			int id = 0;
			if (fed>=FED_VME_MIN && fed<=FED_VME_MAX)
				id = fed-FED_VME_MIN;
			else if (fed>=FED_uTCA_MIN && fed<=FED_uTCA_MAX)
				id = FED_VME_NUM + (fed-FED_uTCA_MIN)/FED_uTCA_DELTA;
			return id;
		}
	}
}

