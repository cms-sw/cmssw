import FWCore.ParameterSet.Config as cms
from RecoHGCal.TICL.ticlDumper_cff import ticlDumper
def customiseTICLForDumper(process, histoName="histo.root"):

    process.ticlDumper = ticlDumper.clone()

    process.TFileService = cms.Service("TFileService",
                                       fileName=cms.string(histoName)
                                       )
    process.FEVTDEBUGHLToutput_step = cms.EndPath(
        process.FEVTDEBUGHLToutput + process.ticlDumper)
    return process
