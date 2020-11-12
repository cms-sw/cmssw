#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace dd4hep;
using namespace cms;

static long algorithm(Detector&, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  DDAlgoArguments args(ctxt, e);
  vector<string> materials = args.vecStr("MaterialNames");

  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_test_DDTestVectorAlgo, algorithm)
