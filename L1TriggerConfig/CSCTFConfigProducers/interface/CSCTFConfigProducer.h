#ifndef CSCTFConfigProducer_h
#define CSCTFConfigProducer_h

#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "CondFormats/DataRecord/interface/L1MuCSCTFConfigurationRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCTFConfiguration.h"

#include "CondFormats/DataRecord/interface/L1MuCSCTFAlignmentRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCTFAlignment.h"

#include "CondFormats/DataRecord/interface/L1MuCSCPtLutRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCPtLut.h"

#include <string>
#include <vector>

class CSCTFConfigProducer : public edm::ESProducer {
private:
  std::string registers[12];
  std::vector<double> alignment;
  std::string ptLUT_path;

public:
  std::unique_ptr<L1MuCSCTFConfiguration> produceL1MuCSCTFConfigurationRcd(const L1MuCSCTFConfigurationRcd& iRecord);
  std::unique_ptr<L1MuCSCTFAlignment> produceL1MuCSCTFAlignmentRcd(const L1MuCSCTFAlignmentRcd& iRecord);
  std::unique_ptr<L1MuCSCPtLut> produceL1MuCSCPtLutRcd(const L1MuCSCPtLutRcd& iRecord);
  void readLUT(std::string path, unsigned short* lut, unsigned long length);

  CSCTFConfigProducer(const edm::ParameterSet& pset);
  ~CSCTFConfigProducer(void) override {}
};

#endif
