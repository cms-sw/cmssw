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

paths.Wenu = ['HLT_Ele10_LW_L1RDQM',
              'HLT_Ele15_SW_L1RDQM',
              'HLT_Ele10_LW_EleId_L1RDQM',
              'HLT_Ele15_SiStrip_L1RDQM']

paths.Zee = paths.Wenu + ['HLT_DoubleEle5_SW_L1RDQM']

paths.GammaJet = ['HLT_Photon10_L1R_DQM',
                  'HLT_Photon15_TrackIso_L1R_DQM',
                  'HLT_Photon15_LooseEcalIso_L1R_DQM',
                  'HLT_Photon25_LooseEcalIso_TrackIso_L1R_DQM']

paths.DiGamma  = ['HLT_Photon10_L1R_DQM','HLT_DoublePhoton10_L1R_DQM']

pathlumi = { 'HLT_Ele10_LW_L1RDQM':'8e29',
             'HLT_Ele15_SW_L1RDQM':'1e31',
             'HLT_Ele10_LW_EleId_L1RDQM':'8e29',
             'HLT_Ele15_SiStrip_L1RDQM':'8e29',
             'HLT_DoubleEle5_SW_L1RDQM':'8e29',
             'HLT_Photon10_L1R_DQM':'8e29',
             'HLT_Photon15_TrackIso_L1R_DQM':'8e29',
             'HLT_Photon15_LooseEcalIso_L1R_DQM':'8e29',
             'HLT_DoublePhoton10_L1R_DQM':'8e29',
             'HLT_Photon25_L1R_DQM':'1e31',
             'HLT_Photon25_LooseEcalIso_TrackIso_L1R_DQM':'1e31'}

lumiprocess = { '8e29':'HLT',
                '1e31':'HLT'
                }
    

##########################################################
# produce generated paricles in acceptance               #
##########################################################

genp = cms.EDFilter("PdgIdAndStatusCandViewSelector",
    status = cms.vint32(3),
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
egammaSelectors = cms.Sequence(tmp) # no empty sequences allowed, start with dummy
egammaValidators= cms.Sequence(tmp) # same

#loop over samples
for samplenum in range(len(samples.names)):

    # clone genparticles and select correct type
    genpartname = "genpart"+samples.names[samplenum]
    globals()[genpartname] = genp.clone()
    setattr(globals()[genpartname],"pdgId",cms.vint32(samples.pdgid[samplenum]) ) # set pdgId
    egammaSelectors *= globals()[genpartname]                            # add to sequence

    # clone generator fiducial region
    fiducialname = "fiducial"+samples.names[samplenum]
    globals()[fiducialname] = fiducial.clone()
    setattr(globals()[fiducialname],"src",cms.InputTag(genpartname) ) # set input collection
    egammaSelectors *= globals()[fiducialname]               # add to sequence

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
        getattr(globals()[trigname],'triggerobject').setProcessName( lumiprocess[pathlumi[trig]] )         #set proper process name
        for filterpset in getattr(globals()[trigname],'filters'):
            getattr(filterpset,'HLTCollectionLabels').setProcessName( lumiprocess[pathlumi[trig]] )
            for isocollections in getattr(filterpset,'IsoCollections'):
                isocollections.setProcessName( lumiprocess[pathlumi[trig]])

        egammaValidators *= globals()[trigname]                      # add to sequence


egammaSelectors.remove(tmp)  # remove the initial dummy
egammaValidators.remove(tmp)

# selectors go into separate "prevalidation" sequence
egammaValidationSequence   = cms.Sequence( egammaValidators )
egammaValidationSequenceFS = cms.Sequence( egammaValidators )
