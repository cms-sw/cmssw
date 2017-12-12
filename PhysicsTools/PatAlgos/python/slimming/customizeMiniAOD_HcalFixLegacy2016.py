import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import MassSearchReplaceAnyInputTagVisitor, addKeepStatement
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask

def cleanPfCandidates(process, verbose=False):
    task = getPatAlgosToolsTask(process)

    #add producer at the beginning of the schedule
    process.load("CommonTools.ParticleFlow.pfCandidatesBadHadRecalibrated_cfi")
    task.add(process.pfCandidatesBadHadRecalibrated)

    replacePFCandidates = MassSearchReplaceAnyInputTagVisitor("particleFlow", "pfCandidatesBadHadRecalibrated", verbose=verbose)
    for everywhere in [ process.producers, process.filters, process.analyzers, process.psets, process.vpsets ]:
        for name,obj in everywhere.iteritems():
            if obj != process.pfCandidatesBadHadRecalibrated:
                replacePFCandidates.doIt(obj, name)

    #add bugged conditions to GT for comparison
    process.GlobalTag.toGet.append(cms.PSet(
        record = cms.string("HcalRespCorrsRcd"),
        label = cms.untracked.string("bugged"),
        tag = cms.string("HcalRespCorrs_v1.02_express") #to be replaced with proper tag name once available
        )
    )



def customizeAll(process, verbose=False):
    print "===>>> customizing the process for legacy rereco 2016"

    cleanPfCandidates(process, verbose)

    addKeepStatement(process,
                     "keep *_pfCandidatesBadHadRecalibrated_*_*",
                     ["keep *_pfCandidatesBadHadRecalibrated_discarded_*"],
                     verbose=verbose)
    

    return process
