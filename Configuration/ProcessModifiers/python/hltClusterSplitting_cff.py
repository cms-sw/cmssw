import FWCore.ParameterSet.Config as cms

# This modifier enables
# - saving pixel vertices at HLT;
# - using those vertices in input for the cluster splitting and ak4CaloJets;

hltClusterSplitting =  cms.Modifier()

