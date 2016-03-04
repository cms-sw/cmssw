import FWCore.ParameterSet.Config as cms

def customiseForRunIInoCCC(process):

    # overwrite parameters which handles the CCC value            
    if hasattr(process,'SiStripClusterChargeCutTiny'):
        setattr(process,'SiStripClusterChargeCutTiny', cms.PSet(value = cms.double( -1.0 ) ) )  #  800.0
    if hasattr(process,'SiStripClusterChargeCutLoose'):
        setattr(process,'SiStripClusterChargeCutLoose', cms.PSet(value = cms.double( -1.0 ) ) ) # 1620.0
    if hasattr(process,'SiStripClusterChargeCutTight'):
        setattr(process,'SiStripClusterChargeCutTight', cms.PSet(value = cms.double( -1.0 ) ) ) # 1945.0
        

    return process
