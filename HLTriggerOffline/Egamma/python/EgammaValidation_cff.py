import FWCore.ParameterSet.Config as cms

class dummy:
    pass

samples=dummy()
paths=dummy()

##########################################################
# Define which preselections to run                      #
##########################################################

samples.names = ['Wenu',
                 'Zee',
                 'GammaJet',
                 'DiGamma']
samples.pdgid = [ 11,
                  11,
                  22,
                  22]
samples.num   = [1,
                 2,
                 1,
                 2]

#which triggers for which sample

paths.Wenu = ['veryHighEtDQM',
#              'singleElectronRelaxedDQM',
              'singleElectronDQM',
              'looseIsoEle15LWL1RDQM',
              'ele15SWL1RDQM',
              'highEtDQM']

paths.Zee = paths.Wenu + ['doubleElectronRelaxedDQM',
                          'doubleElectronDQM',
                          'doubleEle5SWL1RDQM']

paths.GammaJet = [#'singlePhotonRelaxedDQM',
                  'singlePhotonDQM']

paths.DiGamma  = paths.GammaJet + ['veryHighEtDQM',
                                   'highEtDQM',
                                   'doublePhotonRelaxedDQM',
                                   'doublePhotonDQM']


##########################################################
# produce generated paricles in acceptance               #
##########################################################

genp = cms.EDFilter("PdgIdAndStatusCandViewSelector",
    status = cms.vint32(1),
    src = cms.InputTag("genParticles"),
    pdgId = cms.vint32(11)  # replaced in loop
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
egammaValidationSequence = cms.Sequence(tmp)  # no empty sequences allowed, start with dummy

#loop over samples
for samplenum in range(len(samples.names)):

    # clone genparticles and select correct type
    genpartname = "genpart"+samples.names[samplenum]
    globals()[genpartname] = genp.clone()
    setattr(globals()[genpartname],"pdgId",cms.vint32(samples.pdgid[samplenum]) ) # set pdgId
    egammaValidationSequence *= globals()[genpartname]                            # add to sequence

    # clone generator fiducial region
    fiducialname = "fiducial"+samples.names[samplenum]
    globals()[fiducialname] = fiducial.clone()
    setattr(globals()[fiducialname],"src",cms.InputTag(genpartname) ) # set input collection
    egammaValidationSequence *= globals()[fiducialname]               # add to sequence

    # loop over triggers for each sample
    for trig in getattr(paths,samples.names[samplenum]):
        trigname = trig + samples.names[samplenum]
        #import appropriate config snippet
        filename = "HLTriggerOffline.Egamma."+trig+"_cfi"
        trigdef =__import__( filename )
        import sys
        globals()[trigname] = getattr(sys.modules[filename],trig).clone()    # clone imported config
        setattr(globals()[trigname],"cutcollection",cms.InputTag(fiducialname))        # set preselacted generator collection
        setattr(globals()[trigname],"cutnum",cms.int32( samples.num[samplenum]  )) # cut value for preselection
        setattr(globals()[trigname],"pdgGen",cms.int32( samples.pdgid[samplenum])) #correct pdgId for MC matching
        egammaValidationSequence *= globals()[trigname]                      # add to sequence


egammaValidationSequence.remove(tmp)  # remove the initial dummy
