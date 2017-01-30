import FWCore.ParameterSet.Config as cms

def producers_by_type(process, *types):
    "Find all EDProducers in the Process that are instances of the given C++ type."
    return (module for module in process._Process__producers.values() if module._TypedParameterizable__type in types)

def filters_by_type(process, *types):
    "Find all EDFilters in the Process that are instances of the given C++ type."
    return (filter for filter in process._Process__filters.values() if filter._TypedParameterizable__type in types)

def analyzers_by_type(process, *types):
    "Find all EDAnalyzers in the Process that are instances of the given C++ type."
    return (analyzer for analyzer in process._Process__analyzers.values() if analyzer._TypedParameterizable__type in types)

def esproducers_by_type(process, *types):
    "Find all ESProducers in the Process that are instances of the given C++ type."
    return (module for module in process._Process__esproducers.values() if module._TypedParameterizable__type in types)

