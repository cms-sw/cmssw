import FWCore.ParameterSet.Config as cms

class dummy:
    pass

samples=dummy()

##########################################################
# Define which preselections to run                      #
##########################################################

samples.names = ['Wenu',
                 'Zee',
                 'TripleEle',
                 'GammaJet',
                 'DiGamma']
samples.pdgid = [ 11,
                  11,
                  11,
                  22,
                  22]
samples.num   = [1,
                 2,
                 3,
                 1,
                 2]

##########################################################
# produce generated paricles in acceptance               #
##########################################################

genp = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("genParticles"),
    cut = cms.string("isPromptFinalState() & abs(pdgId) = 11")  # replaced in loop
)

fiducial = cms.EDFilter("EtaPtMinCandViewSelector",
    src = cms.InputTag("genp"),
    etaMin = cms.double(-2.5),  # to be replaced in loop ?
    etaMax = cms.double(2.5),   # to be replaced in loop ?
    ptMin = cms.double(2.0)     # to be replaced in loop ?
)

##########################################################
# loop over samples to create modules and sequence       #
##########################################################

tmp = cms.SequencePlaceholder("tmp")
egammaSelectors = cms.Sequence(tmp) # no empty sequences allowed, start with dummy

#loop over samples
for samplenum in range(len(samples.names)):

    # clone genparticles and select correct type
    genpartname = "genpart"+samples.names[samplenum]
    globals()[genpartname] = genp.clone()
    setattr(globals()[genpartname],"cut",cms.string("isPromptFinalState() & abs(pdgId) = "+str(samples.pdgid[samplenum])) ) # set pdgId
    egammaSelectors *= globals()[genpartname]                            # add to sequence

    # clone generator fiducial region
    fiducialname = "fiducial"+samples.names[samplenum]
    globals()[fiducialname] = fiducial.clone()
    setattr(globals()[fiducialname],"src",cms.InputTag(genpartname) ) # set input collection
    egammaSelectors *= globals()[fiducialname]               # add to sequence

egammaSelectors.remove(tmp)  # remove the initial dummy

emdqm = cms.EDAnalyzer('EmDQM',
                           #processname = cms.string("HLT"), # can be obtained from triggerobject
                           autoConfMode = cms.untracked.bool(True),
                           triggerobject = cms.InputTag("hltTriggerSummaryRAW","","HLT"),
                           genEtaAcc = cms.double(2.5),
                           genEtAcc = cms.double(2.0),
                           PtMax = cms.untracked.double(100.0),
                           isData = cms.bool(False),
                           verbosity = cms.untracked.uint32(0),
                           mcMatchedOnly = cms.untracked.bool(True),
                           noPhiPlots = cms.untracked.bool(True),
                           noIsolationPlots = cms.untracked.bool(True)
                          )

# selectors go into separate "prevalidation" sequence
egammaValidationSequence   = cms.Sequence(emdqm)
egammaValidationSequenceFS = cms.Sequence(emdqm)
