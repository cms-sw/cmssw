#ifndef RecoMuon_MuonIsolation_MuonIsolatorFactory_H
#define RecoMuon_MuonIsolation_MuonIsolatorFactory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoMuon/MuonIsolation/interface/MuIsoBaseIsolator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

typedef edmplugin::PluginFactory<muonisolation::MuIsoBaseIsolator* (const edm::ParameterSet&) >  MuonIsolatorFactory;

#endif
