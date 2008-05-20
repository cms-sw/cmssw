#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "DataFormats/SiStripCommon/test/plugins/test_SiStripFecKey.h"
DEFINE_ANOTHER_FWK_MODULE(testSiStripFecKey);

#include "DataFormats/SiStripCommon/test/plugins/examples_SiStripFecKey.h"
DEFINE_ANOTHER_FWK_MODULE(examplesSiStripFecKey);

#include "DataFormats/SiStripCommon/test/plugins/perf_SiStripFecKey.h"
DEFINE_ANOTHER_FWK_MODULE(perfSiStripFecKey);

#include "DataFormats/SiStripCommon/test/plugins/test_SiStripFedKey.h"
DEFINE_ANOTHER_FWK_MODULE(testSiStripFedKey);

#include "DataFormats/SiStripCommon/test/plugins/test_SiStripNullKey.h"
DEFINE_ANOTHER_FWK_MODULE(testSiStripNullKey);

#include "DataFormats/SiStripCommon/test/plugins/test_SiStripKey.h"
DEFINE_ANOTHER_FWK_MODULE(testSiStripKey);

#include "DataFormats/SiStripCommon/test/plugins/test_SiStripEnumsAndStrings.h"
DEFINE_ANOTHER_FWK_MODULE(testSiStripEnumsAndStrings);

#include "DataFormats/SiStripCommon/test/plugins/test_SiStripHistoTitle.h"
DEFINE_ANOTHER_FWK_MODULE(testSiStripHistoTitle);

#include "DataFormats/SiStripCommon/test/plugins/test_Template.h"
DEFINE_ANOTHER_FWK_MODULE(test_Template);
