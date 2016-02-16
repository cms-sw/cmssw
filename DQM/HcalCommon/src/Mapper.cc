
#include "DQM/HcalCommon/interface/Mapper.h"

namespace hcaldqm
{
	namespace mapper
	{
		using namespace constants;
		unsigned int generate_fSubDet(Input const& i)
		{
			int subdetector = i.i1;
			if (subdetector>=1)
				return subdetector-1;
			
			return 0;
		}

		unsigned int generate_fiphi(Input const& i)
		{
			int iphi = i.i1;
			unsigned int index = (iphi - IPHI_MIN)/IPHI_DELTA;
			return index;
		}

		//	negative ietas go first always
		unsigned int generate_fieta(Input const &i)
		{
			int ieta = i.i1;
			unsigned int index;
			if (ieta<0)
				index = (abs(ieta) - IETA_MIN)/IETA_DELTA;
			else
				index = (ieta - IETA_MIN)/IETA_DELTA + IETA_NUM/2;

			return index;
		}

		unsigned int generate_fdepth(Input const& i)
		{
			int depth = i.i1;
			unsigned int index = depth-1;
			return index;
		}

		//	packaging goes like this:
		//	all HB iphis, then HE, then HO, then HF
		unsigned int generate_fSubDet_iphi(Input const& i)
		{
			int subdetector = i.i1;
			int iphi = i.i2;
			int totalHB = IPHI_NUM;
			int totalHE = IPHI_NUM;
			int totalHO = IPHI_NUM;
			unsigned int index = 0;

			if (subdetector==HB)
				index = (iphi-IPHI_MIN)/IPHI_DELTA;	
			else if (subdetector==HE)
				index = totalHB + (iphi-IPHI_MIN)/IPHI_DELTA;
			else if (subdetector==HO)
				index = totalHB+totalHE + (iphi-IPHI_MIN)/IPHI_DELTA;
			else // if HF
				index = totalHB+totalHE+totalHO + (iphi-IPHI_MIN)/IPHI_DELTA_HF;

			return index;
		}

		//	packaging goes like this:
		//	all HB-, then HB+, then ...
		unsigned int generate_fSubDet_ieta(Input const& i)
		{
			int subdetector = i.i1;
			int ieta = i.i2;
			int totalHB = IETA_MAX_HB-IETA_MIN_HB+1;
			int totalHE = IETA_MAX_HE-IETA_MIN_HE+1;
			int totalHO = IETA_MAX_HO-IETA_MIN_HO+1;
			int totalHF = IETA_MAX_HF-IETA_MIN_HF+1;
			unsigned int index = 0;

			if (subdetector==HB)
				if (ieta<0)
					index = (abs(ieta)-IETA_MIN_HB);
				else
					index = totalHB + (ieta-IETA_MIN_HB);
			else if (subdetector==HE)
				if (ieta<0)
					index = 2*totalHB + (abs(ieta)-IETA_MIN_HE);
				else
					index = 2*totalHB + totalHE + ieta-IETA_MIN_HE;
			else if (subdetector==HO)
				if (ieta<0)
					index = 2*totalHB + 2*totalHE + (abs(ieta)-IETA_MIN_HO);
				else
					index = 2*totalHB + 2*totalHE + totalHO + 
						ieta-IETA_MIN_HO;
			else // if HF
				if (ieta<0)
					index = 2*totalHB + 2*totalHE + 2*totalHO + 
						(abs(ieta)-IETA_MIN_HF);
				else
					index = 2*totalHB + 2*totalHE + 2*totalHO + totalHF + 
						ieta-IETA_MIN_HF;

			return index;
		}

		unsigned int generate_fFED(Input const& i)
		{
			int fed = i.i1;
			unsigned int index = 0;
			if (fed<=FED_VME_MAX)
				index = (fed-FED_VME_MIN)/FED_VME_DELTA;
			else if (fed>=FED_uTCA_MIN)
				index = FED_VME_NUM + (fed-FED_uTCA_MIN)/FED_uTCA_DELTA;

			return index;
		}	

		unsigned int generate_fCrate(Input const& i)
		{
			int crate = i.i1;
			unsigned int index = 0;

			if (crate<=CRATE_VME_MAX)
				index = (crate - CRATE_VME_MIN)/CRATE_VME_DELTA;
			else if(crate>=CRATE_uTCA_MIN)
				index = CRATE_VME_NUM + (crate - CRATE_uTCA_MIN)/CRATE_uTCA_DELTA;

			return index;
		}

		/*
		 *	Off at the moment. 2 Crates FEDs per 1 VME FED
		 */
		unsigned int generate_fFED_Slot(Input const &i)
		{
			return 0;
		}

		unsigned int generate_fCrate_Slot(Input const&i)
		{
			int crate = i.i1;
			int slot = i.i2;
			unsigned int index = 0;
		
			if (crate<=CRATE_VME_MAX)
			{
				int icrate = (crate-CRATE_VME_MIN)/CRATE_VME_DELTA;
				int sslot = slot<=SLOT_VME_MIN1 ? (slot-SLOT_VME_MIN) : 
					(slot-SLOT_VME_MIN2+SLOT_VME_NUM1);
				index = icrate*SLOT_VME_NUM + sslot;
			}
			else 
				index = CRATE_VME_NUM*SLOT_VME_NUM + 
					(crate-CRATE_uTCA_MIN)*SLOT_uTCA_NUM + 
					(slot-SLOT_uTCA_MIN);

			return index;
		}

		unsigned int generate_fTPSubDet(Input const&i)
		{
			return i.i1<29 ? 0 : 1;
		}

		unsigned int generate_fTPSubDet_iphi(Input const&i)
		{
			unsigned int index = 0;
			if (i.i1<29)
				index = i.i2-IPHI_MIN;
			else
				index = IPHI_NUM + (i.i2 - IPHI_MIN)/IPHI_DELTA_TPHF;

			return index;
		}

		unsigned int generate_fTPSubDet_ieta(Input const&i)
		{
			unsigned int index = 0;
			if (i.i1>=29)
				index = 2*(IETA_MAX_TPHBHE-IETA_MIN+1)+
					IETA_MAX_TPHF-IETA_MIN_HF+1 + i.i1-IETA_MIN_HF;
			else if (i.i1<=-29)
				index = 2*(IETA_MAX_TPHBHE-IETA_MIN+1) - (i.i1+IETA_MIN_HF);
			else if (i.i1>0)
				index = IETA_MAX_TPHBHE-IETA_MIN+1 + i.i1-IETA_MIN;
			else 
				index = -(i.i1+IETA_MIN);

			return index;
		}

		unsigned int generate_fSubDetPM(Input const&i)
		{
			return 2*(i.i1-1)+i.i2;
		}

		unsigned int generate_fSubDetPM_iphi(Input const&i)
		{
			unsigned int index = 0;
			int subdetector = 2*(i.i1-1)+i.i3;
			int iphi = i.i2;

			if (subdetector==2*(HB-1)) //	HBM
				index = (iphi-IPHI_MIN)/IPHI_DELTA;
			else if (subdetector==2*(HB-1)+1) // HBP
				index = IPHI_NUM+(iphi-IPHI_MIN)/IPHI_DELTA;
			else if (subdetector==2*(HE-1))	//	HEM
				index = 2*IPHI_NUM+(iphi-IPHI_MIN)/IPHI_DELTA;
			else if (subdetector==2*(HE-1)+1)	//	HEP
				index = 3*IPHI_NUM+(iphi-IPHI_MIN)/IPHI_DELTA;
			else if (subdetector==2*(HO-1))	//	HOM
				index = 4*IPHI_NUM+(iphi-IPHI_MIN)/IPHI_DELTA;
			else if (subdetector==2*(HO-1)+1)	//	HOP
				index = 5*IPHI_NUM+(iphi-IPHI_MIN)/IPHI_DELTA;
			else if (subdetector==2*(HF-1))	//	HFM
				index = 6*IPHI_NUM+(iphi-IPHI_MIN)/IPHI_DELTA_HF;
			else 
				index = 6*IPHI_NUM+IPHI_NUM/IPHI_DELTA_HF + 
					(iphi-IPHI_MIN)/IPHI_DELTA_HF;

			return index;
		}

		unsigned int generate_fTPSubDetPM(Input const&i)
		{
			unsigned int index = 0;
			if (i.i1>0 && i.i1<29)
				index = 1;
			else if (i.i1<0 && i.i1>-29)
				index = 0;
			else if (i.i1<=-29)
				index = 2;
			else index = 3;

			return index;
		}

		unsigned int generate_fTPSubDetPM_iphi(Input const& i)
		{
			unsigned int index = 0;
			int ieta = i.i1;
			int iphi = i.i2;
			if (ieta<0 && ieta>-29)	//	HBHEM
				index = iphi-IPHI_MIN;
			else if (ieta>0 && ieta<29)	//	HBHEP
				index = IPHI_NUM+iphi-IPHI_MIN;
			else if (ieta<=-29)
				index = IPHI_NUM*2 + (iphi-IPHI_MIN)/IPHI_DELTA_TPHF;
			else 
				index = IPHI_NUM*2+IPHI_NUM_TPHF + 
					(iphi-IPHI_MIN)/IPHI_DELTA_TPHF;

			return index;
		}

		unsigned int generate_fHFPM_iphi(Input const& i)
		{
			return IPHI_NUM_HF*i.i2 + 
				(i.i1-IPHI_MIN)/IPHI_DELTA_HF;
		}

		unsigned int generate_fHBHEPartition(Input const& i)
		{
			unsigned int index = 0;
			if (i.i1>=3 && i.i1<=26)
				index = 0;
			else if (i.i1>=27 && i.i1<=50)
				index = 1;
			else
				index = 2;

			return index;
		}
	}
}

