#ifndef Mapper_h
#define Mapper_h

/*
 *	file:		Mapper.h
 *	Author:		Viktor Khristenko
 *
 *	Description:
 *		Internal Mapper of vector<ME*> indices. All the possibilities should
 *		be predefined
 */

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"
#include "DQM/HcalCommon/interface/Constants.h"
#include "DQM/HcalCommon/interface/Logger.h"

#include <string>
#include <vector>
#include <sstream>

namespace hcaldqm
{
	namespace mapper
	{
		/*
		 *	Mapper Type enum:
		 *		
		 */
		using namespace hcaldqm::constants;
		enum MapperType
		{
			//	By HCAL subdetector
			fSubDet = 0,
	
			//	By Detector Coordinate
			fiphi = 1,
			fieta = 2,
			fdepth = 3,

			//	Detector Combinations
			fSubDet_iphi = 4,
			fSubDet_ieta = 5,
	
			//	By Electronics Coordinate
			fFED = 6,
			fCrate = 7,

			//	Double Electronics Combinations
			fFED_Slot = 8,
			fCrate_Slot = 9,

			//	TP Mappers
			fTPSubDet = 10,
			fTPSubDet_iphi = 11,
			fTPSubDet_ieta = 12,

			//	Separating Plus Minus
			fSubDetPM = 13,
			fSubDetPM_iphi = 14,
			fTPSubDetPM = 15,
			fTPSubDetPM_iphi = 16,
			fHFPM_iphi = 17,

			fHBHEPartition = 18,

			nMapperType = 19
		};

		/*
		 *	Index Generation Functions - generate index based on input
		 */
		struct Input
		{
			int i1;
			int i2;
			int i3;
			int i4;
		};
		typedef unsigned int(*index_generator)(Input const&);
		unsigned int generate_fSubDet(Input const&);
		unsigned int generate_fiphi(Input const&);
		unsigned int generate_fieta(Input const&); 
		unsigned int generate_fdepth(Input const&);
		unsigned int generate_fSubDet_iphi(Input const&);
		unsigned int generate_fSubDet_ieta(Input const&);
		unsigned int generate_fFED(Input const&);
		unsigned int generate_fCrate(Input const&);
		unsigned int generate_fFED_Slot(Input const&);
		unsigned int generate_fCrate_Slot(Input const&);
		unsigned int generate_fTPSubDet(Input const&);
		unsigned int generate_fTPSubDet_iphi(Input const&);
		unsigned int generate_fTPSubDet_ieta(Input const&);
		unsigned int generate_fSubDetPM(Input const&);
		unsigned int generate_fSubDetPM_iphi(Input const&);
		unsigned int generate_fTPSubDetPM(Input const&);
		unsigned int generate_fTPSubDetPM_iphi(Input const&);
		unsigned int generate_fHFPM_iphi(Input const&);
		unsigned int generate_fHBHEPartition(Input const&);
		index_generator const vindex[nMapperType] = { generate_fSubDet,
			generate_fiphi, generate_fieta, generate_fdepth, 
			generate_fSubDet_iphi, generate_fSubDet_ieta,
			generate_fFED, generate_fCrate, generate_fFED_Slot, 
			generate_fCrate_Slot, generate_fTPSubDet,
			generate_fTPSubDet_iphi, generate_fTPSubDet_ieta,
			generate_fSubDetPM, generate_fSubDetPM_iphi, 
			generate_fTPSubDetPM, generate_fTPSubDetPM_iphi,
			generate_fHFPM_iphi, generate_fHBHEPartition};

		/*
		 *	Mapper Class
		 */
		class Mapper
		{
			public:
				Mapper(): _type(fSubDet) {}
				Mapper(MapperType type) : _type(type)
				{
					this->setSize();
				}
				virtual ~Mapper() {}
	
				virtual void initialize(MapperType type, int debug=0)
				{
					_type = type;
					this->setSize();
					_logger.set("Mapper", debug);
				}
				virtual unsigned int index() {return 0;}
				virtual unsigned int index(double) { return 0;}
				virtual unsigned int index(int x) 
				{
					Input i; i.i1=x;
					if (_type==fFED)
						return vindex[_type](i);
					
					return 0; 
				}

				virtual unsigned int index(HcalDetId const& did)
				{
					Input i;
					if (_type==fSubDet)
						i.i1 = did.subdet();
					else if (_type==fiphi)
						i.i1 = did.iphi();
					else if (_type==fieta)
						i.i1 = did.ieta();
					else if (_type==fdepth)
						i.i1 = did.depth();
					else if (_type==fSubDet_iphi)
					{
						i.i1 = did.subdet();
						i.i2 = did.iphi();
					}
					else if (_type==fSubDet_ieta)
					{
						i.i1 = did.subdet();
						i.i2 = did.ieta();
					}
					else if (_type==fSubDetPM)
					{
						i.i1 = did.subdet();
						i.i2 = did.ieta()>0 ? 1 : 0;
					}
					else if (_type==fSubDetPM_iphi)
					{
						i.i1 = did.subdet();
						i.i2 = did.iphi();
						i.i3 = did.ieta()>0 ? 1 : 0;
					}
					else if (_type==fHFPM_iphi)
					{
						i.i1 = did.iphi();
						i.i2 = did.ieta()>0 ? 1 : 0;
					}
					else if (_type==fHBHEPartition)
					{
						i.i1 = did.iphi();
					}

					return vindex[_type](i);
				}
				virtual unsigned int index(HcalElectronicsId const& eid)
				{
					Input i;
					if (_type==fCrate)
						i.i1 = eid.crateId();
					else if (_type==fCrate_Slot)
					{
						i.i1 = eid.crateId();
						i.i2 = eid.slot();
					}
					return vindex[_type](i);
				}

				virtual unsigned index(HcalTrigTowerDetId const& tid)
				{
					Input i;
					switch(_type)
					{
						case fTPSubDet:
							i.i1 = tid.ietaAbs();
							break;
						case fTPSubDet_iphi:
							i.i1 = tid.ietaAbs();
							i.i2 = tid.iphi();
							break;
						case fTPSubDet_ieta:
							i.i1 = tid.ieta();
							break;
						case fTPSubDetPM:
							i.i1 = tid.ieta();
							break;
						case fTPSubDetPM_iphi:
							i.i1 = tid.ieta();
							i.i2 = tid.iphi();
							break;
						default:
							return 0;
							break;
					}

					return vindex[_type](i);
				}

				virtual std::string buildName(unsigned id)
				{
					_logger.debug(id);
					_logger.debug(_type);
					std::string builtname;
					switch(_type)
					{
						case fSubDet:
							builtname = constants::SUBDET_NAME[id];	
							break;
						case fiphi :
						{
							char name[10];
							sprintf(name, "iphi%d", 
								constants::IPHI_MIN+id*constants::IPHI_DELTA);
							builtname = name;
							break;
						}
						case fHFPM_iphi:
						{
							char name[20];
							if (id>=IPHI_NUM_HF)
								sprintf(name, "HFPiphi%d",
									(id-IPHI_NUM_HF)*IPHI_DELTA_HF + 
									IPHI_MIN);
							else 
								sprintf(name, "HFMiphi%d",
									id*IPHI_DELTA_HF + IPHI_MIN);

							builtname = name;
							break;
						}
						case fieta :
						{
							char name[10];
							int ieta = id<(constants::IETA_NUM/2) ? 
								-(constants::IETA_MIN+id*constants::IETA_DELTA) : 
								(id-constants::IETA_NUM/2)*constants::IETA_DELTA 
								+ constants::IETA_MIN;
							sprintf(name, "ieta%d", ieta);
							builtname = name;
							break;
						}
						case fdepth:
						{
							char name[10];
							sprintf(name, "Depth%d", id+1);
							builtname = name;
							break;
						}
						case fSubDet_iphi:
						{
							char name[20];
							if (id>=IPHI_NUM*3) // HF
								sprintf(name, "HFiphi%d",
									(id-3*constants::IPHI_NUM)*
									constants::IPHI_DELTA_HF+constants::IPHI_MIN);
							else if (id>=2*constants::IPHI_NUM) // HO
								sprintf(name, "HOiphi%d",
									(id-2*constants::IPHI_NUM)*
									constants::IPHI_DELTA+constants::IPHI_MIN);
							else if (id>=constants::IPHI_NUM) // HE
								sprintf(name, "HEiphi%d",
									(id-constants::IPHI_NUM)*
									constants::IPHI_DELTA+constants::IPHI_MIN);
							else 
								sprintf(name, "HBiphi%d",
									id*constants::IPHI_DELTA+constants::IPHI_MIN);
							
							builtname = name;
							break;
						}
						case fSubDet_ieta:
						{
							char name[20];
							unsigned int totalHB = IETA_MAX_HB-IETA_MIN_HB+1;
							unsigned int totalHE = IETA_MAX_HE-IETA_MIN_HE+1;
							unsigned int totalHO = IETA_MAX_HO-IETA_MIN_HO+1;
							unsigned int totalHF = IETA_MAX_HF-IETA_MIN_HF+1;
							if (id>=(2*(totalHB+totalHE+totalHO)+totalHF))
								sprintf(name, "HFPieta%d", 
									(id-2*totalHB-2*totalHE-2*totalHO-totalHF) + 
									IETA_MIN_HF);
							else if (id>=(2*totalHB + 2*totalHE + 2*totalHO))
								sprintf(name, "HFMieta%d",
									-((id-2*totalHB-2*totalHE-2*totalHO) + 
									IETA_MIN_HF));
							else if (id>=(2*totalHB+2*totalHE+totalHO))
								sprintf(name, "HOPieta%d", 
									(id-2*totalHB-2*totalHE-totalHO + 
									 IETA_MIN_HO));
							else if (id>=(2*totalHB+2*totalHE))
								sprintf(name, "HOMieta%d",
									-(id-2*totalHB-2*totalHE + IETA_MIN_HO));
							else if (id>=(2*totalHB+totalHE))
								sprintf(name, "HEPieta%d", 
									(id-2*totalHB-totalHE + IETA_MIN_HE));
							else if (id>=(2*totalHB))
								sprintf(name, "HEMieta%d",
									-(id-2*totalHB+IETA_MIN_HE));
							else if (id>=totalHB)
								sprintf(name, "HBPieta%d",
									id-totalHB+IETA_MIN_HB);
							else 
								sprintf(name, "HBMieta%d",
									-(id+IETA_MIN_HB));

							builtname = name;
							break;
						}
						case fCrate:
						{
							char name[20];
							if (id>=CRATE_VME_NUM)
								sprintf(name, "CRATE%d",
									(id-CRATE_VME_NUM)*CRATE_uTCA_DELTA + 
									CRATE_uTCA_MIN);
							else
								sprintf(name, "CRATE%d",
									id*CRATE_VME_DELTA+CRATE_VME_MIN);

							builtname = name;
							break;
						}
						case fFED:
						{
							char name[20];
							if (id>=FED_VME_NUM)
								sprintf(name, "FED%d", 
									(id-FED_VME_NUM)*FED_uTCA_DELTA+FED_uTCA_MIN);
							else
								sprintf(name, "FED%d",
									id*FED_VME_DELTA+FED_VME_MIN);	
							builtname = name;
							break;
						}
						case fCrate_Slot:
						{
							char name[20];
							if (id>=CRATE_VME_NUM*SLOT_VME_NUM)
							{
								int newid = id - CRATE_VME_NUM*SLOT_VME_NUM;
								int icrate = newid/SLOT_uTCA_NUM;
								int islot = newid%SLOT_uTCA_NUM;
								sprintf(name, "CRATE%dSLOT%d",
									icrate+CRATE_uTCA_MIN, islot+SLOT_uTCA_MIN);
							}
							else 
							{
								int icrate = id/SLOT_VME_NUM;
								int islot = id%SLOT_VME_NUM;
								if (islot>=SLOT_VME_NUM1)
									sprintf(name, "CRATE%dSLOT%d",
										icrate+CRATE_VME_MIN,
										islot-SLOT_VME_NUM1+SLOT_VME_MIN2);
								else 
									sprintf(name, "CRATE%dSLOT%d",
										icrate+CRATE_VME_MIN,
										islot+SLOT_VME_MIN);
							}
							builtname = name;
							break;
						}
						case fTPSubDet:
						{
							builtname = constants::TPSUBDET_NAME[id];
							break;
						}
						case fTPSubDet_iphi:
						{
							char name[20];
							if (id>=IPHI_NUM)
								sprintf(name, "HFiphi%d",
									(id-IPHI_NUM)*IPHI_DELTA_TPHF+
									IPHI_MIN);
							else
								sprintf(name, "HBHEiphi%d",
									id+IPHI_MIN);
							builtname = name;

							break;
						}
						case fTPSubDet_ieta:
						{
							char name[20];
							unsigned int totalHBHE = IETA_MAX_TPHBHE - IETA_MIN+1;
							unsigned int totalHF = IETA_MAX_TPHF - IETA_MIN_HF+1;
							if (id>=(2*totalHBHE+totalHF))
								sprintf(name, "HFPieta%d", 
									id - 2*totalHBHE-totalHF+IETA_MIN_HF);
							else if (id>=2*totalHBHE)
								sprintf(name, "HFMieta%d",
									-(id - 2*totalHBHE + IETA_MIN_HF));
							else if (id>=totalHBHE)
								sprintf(name, "HBHEPieta%d",
									id-totalHBHE+IETA_MIN);
							else
								sprintf(name, "HBHEMieta%d",
									-(id+IETA_MIN));

							builtname = name;
							break;
						}
						case fSubDetPM:
						{
							builtname = constants::SUBDETPM_NAME[id];
							break;
						}
						case fSubDetPM_iphi:
						{
							char name[20];
							if (id>=(IPHI_NUM*6)+IPHI_NUM_HF) // HFP
								sprintf(name, "HFPiphi%d",
									(id-6*IPHI_NUM-IPHI_NUM_HF)*
									constants::IPHI_DELTA_HF+
									constants::IPHI_MIN);
							else if (id>=6*IPHI_NUM) // HFM
								sprintf(name, "HFMiphi%d",
									(id-6*IPHI_NUM)*
									constants::IPHI_DELTA_HF+
									constants::IPHI_MIN);
							else if (id>=5*IPHI_NUM) // HOP
								sprintf(name, "HOPiphi%d",
									(id-5*IPHI_NUM)*
									constants::IPHI_DELTA+constants::IPHI_MIN);
							else if (id>=4*IPHI_NUM) //	HOM
								sprintf(name, "HOMiphi%d",
									(id-4*IPHI_NUM)*
									constants::IPHI_DELTA+constants::IPHI_MIN);
							else if (id>=3*IPHI_NUM) // HEP
								sprintf(name, "HEPiphi%d",
									(id-3*IPHI_NUM)*
									IPHI_DELTA+IPHI_MIN);
							else if (id>=2*IPHI_NUM) //	HEM
								sprintf(name, "HEMiphi%d",
									(id-2*IPHI_NUM)*
									IPHI_DELTA+IPHI_MIN);
							else if (id>=IPHI_NUM) // HBP
								sprintf(name, "HBPiphi%d",
									(id-IPHI_NUM)*IPHI_DELTA+IPHI_MIN);
							else //	HBM
								sprintf(name, "HBMiphi%d",
									id*IPHI_DELTA+IPHI_MIN);

							builtname = name;
							break;
						}
						case fTPSubDetPM:
						{
							builtname = constants::TPSUBDETPM_NAME[id];
							break;
						}
						case fTPSubDetPM_iphi:
						{
							char name[20];
							if (id>=(2*IPHI_NUM+IPHI_NUM_TPHF))
								sprintf(name, "HFPiphi%d",
									(id-2*IPHI_NUM-IPHI_NUM_TPHF)*
									IPHI_DELTA_TPHF+IPHI_MIN);
							else if (id>=2*IPHI_NUM)
								sprintf(name, "HFMiphi%d",
									(id-2*IPHI_NUM)*
									IPHI_DELTA_TPHF+IPHI_MIN);
							else if (id>=IPHI_NUM)
								sprintf(name, "HBHEPiphi%d",
									(id-IPHI_NUM)*IPHI_DELTA+IPHI_MIN);
							else 
								sprintf(name, "HBHEMiphi%d",
									id*IPHI_DELTA+IPHI_MIN);
							builtname = name;
							break;
						}
						case fHBHEPartition:
						{
							if (id==0)
								builtname = "HBHEa";
							else if (id==1)
								builtname = "HBHEb";
							else if (id==2)
								builtname = "HBHEc";
							break;
						}
						default:
						{
							return std::string("UNKNOWN");
							break;
						}
					}
					return builtname;
				}

				inline unsigned int getSize() {return _size;}
	
			protected:
				MapperType			_type;
				unsigned int		_size;
				Logger				_logger;
				void setSize()
				{
					switch (_type)
					{
						case fSubDet : 
							_size = SUBDET_NUM;
							break;
						case fiphi:
							_size = IPHI_NUM;
							break;
						case fieta:
							_size = IETA_NUM;
							break;
						case fdepth:
							_size = DEPTH_NUM;
							break;
						case fSubDet_iphi:
							_size = (SUBDET_NUM-1)*IPHI_NUM + 
								IPHI_NUM/IPHI_DELTA_HF;
							break;
						case fSubDet_ieta:
							_size = 2*(IETA_MAX_HB-IETA_MIN_HB+1) + 
								2*(IETA_MAX_HE-IETA_MIN_HE+1) + 
								2*(IETA_MAX_HO-IETA_MIN_HO+1)+
								2*(IETA_MAX_HF-IETA_MIN_HF+1);
							break;
						case fFED:
							_size = FED_VME_NUM+FED_uTCA_NUM;
							break;
						case fCrate:
							_size = CRATE_VME_NUM+CRATE_uTCA_NUM; 
							break;
						case fCrate_Slot:
							_size = CRATE_VME_NUM*SLOT_VME_NUM + 
								CRATE_uTCA_NUM*SLOT_uTCA_NUM;
							break;
						case fTPSubDet:
							_size = 2;
							break;
						case fTPSubDet_iphi:
							_size = IPHI_NUM+IPHI_NUM/IPHI_DELTA_TPHF;
							break;
						case fTPSubDet_ieta:
							_size = 2*(IETA_MAX_TPHBHE-IETA_MIN+1)+
								2*(IETA_MAX_TPHF-IETA_MIN_HF+1);
							break;
						case fSubDetPM:
							_size = 8;
							break;
						case fSubDetPM_iphi:
							//	6 * 72 + 2*72/2 for the current detecto
							_size = 6*IPHI_NUM+2*IPHI_NUM/IPHI_DELTA_HF;
							break;
						case fTPSubDetPM:
							_size = 4;
							break;
						case fTPSubDetPM_iphi:
							_size = 2*IPHI_NUM + 2*IPHI_NUM/IPHI_DELTA_TPHF;
							break;
						case fHFPM_iphi:
							_size = 72;
							break;
						case fHBHEPartition:
							_size = 3;
							break;
						default:
							_size = 0;
					}
				}
		};
	}
}

#endif








