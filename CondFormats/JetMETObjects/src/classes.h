#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleJetCorrectionUncertainty.h"
#include "CondFormats/JetMETObjects/interface/SimpleJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include <vector>
 
JetCorrectorParameters corr;
JetCorrectorParameters::Definitions def;
JetCorrectorParameters::Record record;
std::vector<JetCorrectorParameters> corrv;
std::vector<JetCorrectorParameters::Record> recordv;
JetCorrectorParametersCollection coll;
JetCorrectorParametersCollection::pair_type pair_type;
JetCorrectorParametersCollection::collection_type colltype;
std::vector<JetCorrectorParametersCollection> collv;
