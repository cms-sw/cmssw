#include "CondFormats/JetMETObjects/src/headers.h"

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
FFTJetCorrectorParameters fftcorr;
QGLikelihoodCategory qgcat;
QGLikelihoodObject qgobj;
QGLikelihoodObject::Entry qgentry;
std::vector< QGLikelihoodObject::Entry > qgentryv;
QGLikelihoodSystematicsObject qgsystobj;
QGLikelihoodSystematicsObject::Entry qgsystentry;
std::vector< QGLikelihoodSystematicsObject::Entry > qgsystentryv;
METCorrectorParameters METcorr;
JME::JetResolutionObject jerobj;
JME::JetResolutionObject::Definition jerdef;
JME::JetResolutionObject::Record jerrecord;
JME::JetResolutionObject::Range jerrange;
std::vector<JME::JetResolutionObject::Record> jerrecordvec;
std::vector<JME::JetResolutionObject::Range> jerrangevec;
