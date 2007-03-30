#include "RecoMuon/MuonIsolation/interface/MuIsoExtractorFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"


MuIsoExtractorFactory::MuIsoExtractorFactory()
  : seal::PluginFactory<muonisolation::MuIsoExtractor*(
        const edm::ParameterSet & p)>("MuIsoExtractorFactory")
{ }

MuIsoExtractorFactory::~MuIsoExtractorFactory()
{ }

MuIsoExtractorFactory * MuIsoExtractorFactory::get()
{
  static MuIsoExtractorFactory theMuIsoExtractorFactory;
  return & theMuIsoExtractorFactory;
}

