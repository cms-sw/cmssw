#include <L1TriggerConfig/CSCTFConfigProducers/interface/CSCTFConfigProducer.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <cerrno>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>


CSCTFConfigProducer::CSCTFConfigProducer(const edm::ParameterSet& pset) {
  const char* name[12] = {"registersSP1",
                          "registersSP2",
                          "registersSP3",
                          "registersSP4",
                          "registersSP5",
                          "registersSP6",
                          "registersSP7",
                          "registersSP8",
                          "registersSP9",
                          "registersSP10",
                          "registersSP11",
                          "registersSP12"};

  for (int sp = 0; sp < 12; sp++) {
    std::vector<std::string> regs = pset.getParameter<std::vector<std::string> >(name[sp]);
    for (std::vector<std::string>::const_iterator line = regs.begin(); line != regs.end(); line++)
      registers[sp] += *line + "\n";
  }

  alignment = pset.getParameter<std::vector<double> >("alignment");
  ptLUT_path = pset.getParameter<std::string>("ptLUT_path");
  setWhatProduced(this, &CSCTFConfigProducer::produceL1MuCSCTFConfigurationRcd);
  setWhatProduced(this, &CSCTFConfigProducer::produceL1MuCSCTFAlignmentRcd);
  setWhatProduced(this, &CSCTFConfigProducer::produceL1MuCSCPtLutRcd);
}

std::unique_ptr<L1MuCSCTFConfiguration> CSCTFConfigProducer::produceL1MuCSCTFConfigurationRcd(
    const L1MuCSCTFConfigurationRcd& iRecord) {
  edm::LogInfo("L1-O2O: CSCTFConfigProducer") << "Producing "
                                              << " L1MuCSCTFConfiguration from PSET";

  std::unique_ptr<L1MuCSCTFConfiguration> config =
      std::make_unique<L1MuCSCTFConfiguration>(registers);
  return config;
}

std::unique_ptr<L1MuCSCTFAlignment> CSCTFConfigProducer::produceL1MuCSCTFAlignmentRcd(
    const L1MuCSCTFAlignmentRcd& iRecord) {
  edm::LogInfo("L1-O2O: CSCTFConfigProducer") << "Producing "
                                              << " L1MuCSCTFAlignment from PSET";

  std::unique_ptr<L1MuCSCTFAlignment> al = std::make_unique<L1MuCSCTFAlignment>(alignment);
  return al;
}

std::unique_ptr<L1MuCSCPtLut> CSCTFConfigProducer::produceL1MuCSCPtLutRcd(const L1MuCSCPtLutRcd& iRecord) {
  edm::LogInfo("L1-O2O: CSCTFConfigProducer") << "Producing "
                                              << " L1MuCSCPtLut from PSET";

  std::unique_ptr<L1MuCSCPtLut> pt_lut = std::make_unique<L1MuCSCPtLut>();

  if (ptLUT_path.length()) {
    readLUT(ptLUT_path, (unsigned short*)pt_lut->pt_lut, 1 << 21);  //CSCBitWidths::kPtAddressWidth
  } else {
    throw cms::Exception("Undefined pT LUT")
        << "CSCTFConfigProducer is unable to generate LUTs on the fly.\n"
           "Specify full LUT file names or just avoid using CSCTFConfigProducer by uncommenting PTLUT "
           "parameter sets in L1Trigger/CSCTrackFinder configuration."
        << std::endl;
  }
  return pt_lut;
}

void CSCTFConfigProducer::readLUT(std::string path, unsigned short* lut, unsigned long length) {
  // Reading
  if (path.find(".bin") != std::string::npos) {  // Binary format
    std::ifstream file(path.c_str(), std::ios::binary);
    file.read((char*)lut, length * sizeof(unsigned short));
    if (file.fail())
      throw cms::Exception("Reading error") << "CSCTFConfigProducer cannot read " << length << " words from " << path
                                            << " (errno=" << errno << ")" << std::endl;
    if ((unsigned int)file.gcount() != length * sizeof(unsigned short))
      throw cms::Exception("Incorrect LUT size")
          << "CSCTFConfigProducer read " << (file.gcount() / sizeof(unsigned short)) << " words from " << path
          << " instead of " << length << " (errno=" << errno << ")" << std::endl;
    file.close();
  } else {
    std::ifstream file(path.c_str());
    if (file.fail())
      throw cms::Exception("Cannot open file")
          << "CSCTFConfigProducer cannot open " << path << " (errno=" << errno << ")" << std::endl;
    unsigned int address = 0;
    for (address = 0; !file.eof() && address < length; address++)
      file >> lut[address];  // Warning: this may throw non-cms like exception
    if (address != length)
      throw cms::Exception("Incorrect LUT size")
          << "CSCTFConfigProducer read " << address << " words from " << path << " instead of " << length << std::endl;
    file.close();
  }
  LogDebug("CSCTFConfigProducer::readLUT") << " read from " << path << " " << length << " words" << std::endl;
}
