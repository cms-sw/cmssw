#ifndef L1TMUON_BARREL_PARAMS_HELPER_h
#define L1TMUON_BARREL_PARAMS_HELPER_h

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsRcd.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTLUTFactories.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CondFormats/L1TObjects/interface/L1TriggerLutFile.h"
#include "CondFormats/L1TObjects/interface/DTTFBitArray.h"

#include "L1Trigger/L1TCommon/interface/XmlConfigReader.h"
#include "L1Trigger/L1TCommon/interface/trigSystem.h"
#include "L1Trigger/L1TCommon/interface/setting.h"
#include "L1Trigger/L1TCommon/interface/mask.h"


typedef std::map<short, short, std::less<short> > LUT;

class L1TMuonBarrelParamsHelper 
{
public:
	L1TMuonBarrelParamsHelper() {};
	L1TMuonBarrelParamsHelper(const L1TMuonBarrelParams& barrelParams) ;

	~L1TMuonBarrelParamsHelper() {};

	void configFromPy(std::map<std::string, int>& allInts, std::map<std::string, bool>& allBools, std::map<std::string, std::vector<std::string> > allMasks, unsigned int fwVersion, const std::string& AssLUTpath);
	void configFromDB(l1t::trigSystem& trgSys);
	operator L1TMuonBarrelParams(void) const {return m_params_helper;} ;


private:
	L1TMuonBarrelParams m_params_helper;
	l1t::trigSystem m_trgSys;

	int load_pt(std::vector<LUT>& , std::vector<int>&, unsigned short int, std::string);
	int load_phi(std::vector<LUT>& , unsigned short int, unsigned short int, std::string);
	int load_ext(std::vector<L1TMuonBarrelParams::LUTParams::extLUT>&, unsigned short int, unsigned short int );

};

#endif
