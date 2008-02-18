#ifndef CSCTFConfigProducer_h
#define CSCTFConfigProducer_h

#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "CondFormats/DataRecord/interface/L1MuCSCDTLutRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCDTLut.h"

#include "CondFormats/DataRecord/interface/L1MuCSCPtLutRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCPtLut.h"

#include "CondFormats/DataRecord/interface/L1MuCSCLocalPhiLutRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCLocalPhiLut.h"

#include "CondFormats/DataRecord/interface/L1MuCSCGlobalLutsRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCGlobalLuts.h"

#include "CondFormats/DataRecord/interface/L1MuCSCTFConfigurationRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCTFConfiguration.h"

class CSCTFConfigProducer : public edm::ESProducer {
private:
	std::string ptLUT_path;
	std::string dt1LUT_path;
	std::string dt2LUT_path;
	std::string localPhiLUT_path;
	std::string globalPhi1LUT_path;
	std::string globalPhi2LUT_path;
	std::string globalPhi3LUT_path;
	std::string globalPhi4LUT_path;
	std::string globalPhi5LUT_path;
	std::string globalEta1LUT_path;
	std::string globalEta2LUT_path;
	std::string globalEta3LUT_path;
	std::string globalEta4LUT_path;
	std::string globalEta5LUT_path;

public:
	std::auto_ptr<L1MuCSCPtLut>       produceL1MuCSCPtLutRcd      (const L1MuCSCPtLutRcd& iRecord);
	std::auto_ptr<L1MuCSCDTLut>       produceL1MuCSCDTLutRcd      (const L1MuCSCDTLutRcd& iRecord);
	std::auto_ptr<L1MuCSCLocalPhiLut> produceL1MuCSCLocalPhiLutRcd(const L1MuCSCLocalPhiLutRcd& iRecord);
	std::auto_ptr<L1MuCSCGlobalLuts>  produceL1MuCSCGlobalLutsRcd (const L1MuCSCGlobalLutsRcd&  iRecord);

	std::auto_ptr<L1MuCSCTFConfiguration> produceL1MuCSCTFConfigurationRcd(const L1MuCSCTFConfigurationRcd& iRecord);

	void readLUT(std::string path, unsigned short* lut, unsigned long length);

	CSCTFConfigProducer(const edm::ParameterSet& pset);
	~CSCTFConfigProducer(void){}
};

#endif
