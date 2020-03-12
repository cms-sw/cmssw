#include "L1TriggerConfig/CSCTFConfigProducers/interface/L1MuCSCPtLutConfigOnlineProd.h"

std::unique_ptr<L1MuCSCPtLut> L1MuCSCPtLutConfigOnlineProd::newObject(const std::string& objectKey) {
  edm::LogInfo("L1-O2O: L1MuCSCPtLutConfigOnlineProd") << "Producing "
                                                       << "L1MuCSCPtLut "
                                                       << "with key PTLUT_VERSION=" << objectKey;

  // read the Pt_LUT: it is CLOB with 2^21 different values
  //SELECT PT_LUT FROM CMS_CSC_TF.CSCTF_PTLUS WHERE CSCTF_PTLUS.PTLUT_VERSION = objectKey
  l1t::OMDSReader::QueryResults results = m_omdsReader.basicQuery(
      "PT_LUT", "CMS_CSC_TF", "CSCTF_PTLUTS", "CSCTF_PTLUTS.PTLUT_VERSION", m_omdsReader.singleAttribute(objectKey));

  // check if query was successful
  if (results.queryFailed()) {
    edm::LogError("L1-O2O") << "Problem with L1MuCSCPtLutParameters key";
    // return empty object
    return std::unique_ptr<L1MuCSCPtLut>();
  }

  std::string ptlut;
  results.fillVariable(ptlut);

  // if uncommented it will generate a huge output...
  //edm::LogInfo( "L1-O2O: L1MuCSCPtLutConfigOnlineProd" ) << "PtLUT is "
  //                                                       << "ptlut";

  edm::LogInfo("L1-O2O: L1MuCSCPtLutConfigOnlineProd") << "Returning L1MuCSCPtLut";

  auto CSCTFPtLut = std::make_unique<L1MuCSCPtLut>();
  CSCTFPtLut->readFromDBS(ptlut);

  return CSCTFPtLut;
}

DEFINE_FWK_EVENTSETUP_MODULE(L1MuCSCPtLutConfigOnlineProd);
