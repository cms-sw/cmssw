
#include "DQM/HcalCommon/interface/CoordinateAxis.h"

namespace hcaldqm
{
	namespace axis
	{
		CoordinateAxis::CoordinateAxis():
			Axis(), _ctype(fSubDet)
		{}

		CoordinateAxis::CoordinateAxis(AxisType type, CoordinateType ctype,
			int n, double min, double max, std::string title, bool log):
			Axis(title, type, fCoordinate, n, min, max, log), _ctype(ctype)
		{
			this->_setup();
		}

		CoordinateAxis::CoordinateAxis(AxisType type, CoordinateType ctype,
			bool log):
			Axis(ctitle[ctype], type, fCoordinate, cnbins[ctype],
				cmin[ctype], cmax[ctype], log), _ctype(ctype)
		{
			this->_setup();
		}

		/* virtual */ int CoordinateAxis::get(HcalDetId const& did)
		{
			int x;
			switch(_ctype)
			{
				case fSubDet:
					x = did.subdet();
					break;
				case fSubDetPM:
					x = did.ieta()<0 ? 2*(did.subdet()-1) : 
						2*(did.subdet()-1)+1;
					break;
				case fiphi:
					x = did.iphi();
					break;
				case fieta:
					x = did.ieta();
					if (x<0)
						x = did.subdet()==HcalForward ? x+41 : x+42;
					else 
						x = did.subdet()==HcalForward ? x+42 : x+41;
					break;
				case fdepth:
					x = did.depth();
					break;
				default:
					x = -100;
					break;
			}
			return x;
		}

		/* virtual */ int CoordinateAxis::get(HcalTrigTowerDetId const& tid)
		{
			int x;
			switch(_ctype)
			{
				case fiphi:
					x = tid.iphi();
					break;
				case fTPieta:
					x = tid.ieta()<0 ? tid.ieta()+32 : tid.ieta()+31;
					break;
				case fTPSubDet:
					x = tid.ietaAbs()<29 ? 0 : 1;
					break;
				case fTPSubDetPM:
				{
					int ieta = tid.ieta();
					if (ieta<0 && ieta>-29)
						x = 0;
					else if (ieta>0 && ieta<29)
						x = 1;
					else if (ieta<0 && ieta<=-29)
						x = 2;
					else 
						x = 3;
					break;
				}
				default:
					x = -100;
					break;
			}
			return x;
		}

		/* virtual */ int CoordinateAxis::get(HcalElectronicsId const& eid)
		{
			int x;
			switch(_ctype)
			{
				case fCrateVME:
					x = eid.crateId();
					if (x<=CRATE_VME_MAX)
						x = x-CRATE_VME_MIN;
					else 
						x = 100;
					break;
				case fCrateuTCA:
					x = eid.crateId();
					if (x>CRATE_VME_MAX && x<=CRATE_uTCA_MAX)
						x = x-CRATE_uTCA_MIN;
					else 
						x = 100;
					break;
				case fCrateComb:
					x = eid.crateId();
					if (x<=CRATE_VME_MAX)
						x = x-CRATE_VME_MIN;
					else if (x<=CRATE_uTCA_MAX)
						x = CRATE_VME_NUM+ x-CRATE_uTCA_MIN;
					else
						x = 100;
					break;
				case fSlotVME:
					x = eid.slot();
					if (x<=SLOT_VME_MIN1)
						x = x-SLOT_VME_MIN;
					else if (x>=SLOT_VME_MIN2 && x<=SLOT_VME_MAX)
						x = SLOT_VME_NUM1+ x-SLOT_VME_MIN2;
					else 
						x = 100;
					break;
				case fSlotuTCA:
					x = eid.slot();
					if (x<=SLOT_uTCA_MAX)
						x = x-SLOT_uTCA_MIN;
					else
						x = 100;
					break;
				case fSlotComb:
					x = eid.slot();
					if (eid.isVMEid()) // VME
					{
						if (x<=SLOT_VME_MIN1)
							x = x-SLOT_VME_MIN;
						else if (x>=SLOT_VME_MIN2 && x<=SLOT_VME_MAX)
							x = SLOT_VME_NUM1+ x-SLOT_VME_MIN2;	
						else x = 100;
					}
					else // uTCA
					{
						if (x<=SLOT_uTCA_MAX)
							x = x-SLOT_uTCA_MIN;
						else 
							x = 100;
					}
					break;
				case fSpigot:
					x = eid.spigot();
					break;
				case fFiberVME:
					{
						x = eid.fiberIndex();
						int tb = eid.htrTopBottom(); //1 for t
						x = tb*FIBER_VME_NUM + (x-FIBER_VME_MIN);
					}
					break;
				case fFiberuTCA:
					x = eid.fiberIndex()-FIBER_uTCA_MIN;
					break;
				case fFiberComb:
					x = eid.fiberIndex();
					if (eid.isVMEid())
					{
						int tb = eid.htrTopBottom();
						x = tb*FIBER_VME_NUM+(x-FIBER_VME_MIN);
					}
					else
						x = x-FIBER_uTCA_MIN;
					break;
				case fFiberCh:
					x = eid.fiberChanId();
					break;
				default :
					x = -100;
					break;
			}
			return x;
		}

		/* virtual */ int CoordinateAxis::get(int i)
		{
			int x = 0;
			switch(_ctype)
			{
				case fFEDVME:
					x = i-FED_VME_MIN;
					break;
				case fFEDuTCA:
					x = (i-FED_uTCA_MIN)/FED_uTCA_DELTA;
					break;
				case fFEDComb:
					if (i<=FED_VME_MAX)
						x = i-FED_VME_MIN;
					else 
						x = FED_VME_NUM+ (i-FED_uTCA_MIN)/FED_uTCA_DELTA;
					break;
				case fCrateVME:
					x = i-CRATE_VME_MIN;
					break;
				case fCrateuTCA:
					x = i-CRATE_uTCA_MIN;
					break;
				case fCrateComb:
					if (i<CRATE_VME_MAX)
						x = i-CRATE_VME_MIN;
					else 
						x = CRATE_VME_NUM+i-CRATE_uTCA_MIN;
					break;
				default :
					break;
			}
			return x;
		}

		/* virtual */ int CoordinateAxis::getBin(HcalDetId const&)
		{
			return 1;
		}

		/* virtual */ int CoordinateAxis::getBin(HcalElectronicsId const&)
		{
			return 1;
		}

		/* virtual */ int CoordinateAxis::getBin(HcalTrigTowerDetId const&)
		{
			return 1;
		}

		/* virtual */ int CoordinateAxis::getBin(int value)
		{
			int r = 1;
			switch (_ctype)
			{
				case fSubDet:
					r = value+1;
					break;
				case fTPSubDet:
					r = value+1;
					break;
				case fiphi:
					r = value;
					break;
				case fFEDComb:
					r = 1+value;
					break;
				default:
					r = 1;
					break;
			}

			return r;
		}

		/* virtual */ void CoordinateAxis::_setup()
		{
			char name[20];
			switch (_ctype)
			{
				case fSubDet:
					for (int i=HB; i<=HF; i++)
						_labels.push_back(SUBDET_NAME[i-1]);
					break;
				case fSubDetPM:
					for (int i=0; i<2*SUBDET_NUM; i++)
						_labels.push_back(SUBDETPM_NAME[i]);
					break;
				case fTPSubDetPM:
					for (int i=0; i<2*TPSUBDET_NUM; i++)
						_labels.push_back(TPSUBDETPM_NAME[i]);
					break;
				case fieta:
					for (int ieta=-41; ieta<=41; ieta++)
					{
						if (ieta==0)
							continue;
						sprintf(name, "%d", ieta);
						if (ieta==-29 || ieta==29)
							_labels.push_back(std::string(name));
						_labels.push_back(std::string(name));
					}
					break;
				case fFEDVME:
					for (int i=FED_VME_MIN; i<=FED_VME_MAX; i++)
					{
						sprintf(name, "%d", i);
						_labels.push_back(std::string(name));
					}
					break;
				case fFEDuTCA:
					for (int i=FED_uTCA_MIN; i<=FED_uTCA_MAX; i+=2)
					{
						sprintf(name, "%d", i);
						_labels.push_back(std::string(name));
					}
					break;
				case fFEDComb:	// uTCA and VME combined
					for (int i=FED_VME_MIN; i<=FED_VME_MAX; i++)
					{
						sprintf(name, "%d", i);
						_labels.push_back(std::string(name));
					}
					for (int i=FED_uTCA_MIN; i<=FED_uTCA_MAX; i+=2)
					{
						sprintf(name, "%d", i);
						_labels.push_back(std::string(name));
					}
					break;
				case fCrateVME:
					for (int i=CRATE_VME_MIN; i<=CRATE_VME_MAX; i++)
					{
						sprintf(name, "%d", i);
						_labels.push_back(std::string(name));
					}
					break;
				case fCrateuTCA:
					for (int i=CRATE_uTCA_MIN; i<=CRATE_uTCA_MAX; i++)
					{
						sprintf(name, "%d", i);
						_labels.push_back(std::string(name));
					}
					break;
				case fCrateComb:
					for (int i=CRATE_VME_MIN; i<=CRATE_VME_MAX; i++)
					{
						sprintf(name, "%d", i);
						_labels.push_back(std::string(name));
					}
					for (int i=CRATE_uTCA_MIN; i<=CRATE_uTCA_MAX; i++)
					{
						sprintf(name, "%d", i);
						_labels.push_back(std::string(name));
					}
					break;
				case fSlotVME:
					for (int i=SLOT_VME_MIN; i<=SLOT_VME_MIN1; i++)
					{
						sprintf(name, "%d", i);
						_labels.push_back(std::string(name));
					}
					for (int i=SLOT_VME_MIN2; i<=SLOT_VME_MAX; i++)
					{
						sprintf(name, "%d", i);
						_labels.push_back(std::string(name));
					}
					break;
				case fSlotuTCA:
					for (int i=SLOT_uTCA_MIN; i<=SLOT_uTCA_MAX; i++)
					{
						sprintf(name, "%d", i);
						_labels.push_back(std::string(name));
					}
					break;
				case fSlotComb:
					{
						int j = SLOT_uTCA_MIN;
						for (int i=SLOT_VME_MIN; i<=SLOT_VME_MIN1; i++)
						{
							sprintf(name, "%d (%d)", i, j);
							_labels.push_back(std::string(name));
							j++;
						}
						for (int i=SLOT_VME_MIN2; i<=SLOT_VME_MAX; i++)
						{
							sprintf(name, "%d (%d)", i, j);
							_labels.push_back(std::string(name));
							j++;
						}
					}
					break;
				case fFiberVME:
					for (int i=FIBER_VME_MIN; i<=FIBER_VME_MAX; i++)
					{
						sprintf(name, "%d-b", i);
						_labels.push_back(std::string(name));
					}
					for (int i=FIBER_VME_MIN; i<=FIBER_VME_MAX; i++)
					{
						sprintf(name, "%d-t", i);
						_labels.push_back(std::string(name));
					}
					break;
				case fFiberuTCA:
					for (int i=FIBER_uTCA_MIN; i<=FIBER_uTCA_MAX; i++)
					{
						sprintf(name, "%d", i);
						_labels.push_back(std::string(name));
					}
					break;
				case fFiberComb:
					{
						int j = FIBER_uTCA_MIN;
						for (int i=FIBER_VME_MIN; i<=FIBER_VME_MAX; i++)
						{
							sprintf(name, "%d-b (%d)", i, j);
							_labels.push_back(std::string(name));
							j++;
						}
						for (int i=FIBER_VME_MIN; i<=FIBER_VME_MAX; i++)
						{
							sprintf(name, "%d-t (%d)", i, j);
							_labels.push_back(std::string(name));
							j++;
						}
						for (int k=j; k<+FIBER_uTCA_MAX; k++)
						{
							sprintf(name, "%d", k);
							_labels.push_back(std::string(name));
						}
					}
					break;
				case fTPSubDet:
					_labels.push_back(std::string("HBHE"));
					_labels.push_back(std::string("HF"));
					break;
				case fTPieta:
					for (int ieta=-32; ieta<=32; ieta++)
					{
						if (ieta==0)
							continue;
						sprintf(name, "%d", ieta);
						_labels.push_back(std::string(name));
					}
				default:
					break;
			}
		}
	}
}




