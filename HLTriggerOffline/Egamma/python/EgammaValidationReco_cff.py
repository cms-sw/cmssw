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
              'HLT_Ele10_LW_EleId_L1RDQM'
              ]

paths.Zee = paths.Wenu + ['HLT_DoubleEle5_SW_L1RDQM']

paths.GammaJet = ['HLT_Photon15_TrackIso_L1R_DQM',
                  'HLT_Photon10_L1R_DQM']

paths.DiGamma  = paths.GammaJet 


##########################################################
# loop over samples to create modules and sequence       #
##########################################################

tmp = cms.SequencePlaceholder("tmp")
egammaValidationSequenceReco = cms.Sequence(tmp)  # no empty sequences allowed, start with dummy

#loop over samples
for samplenum in range(len(samples.names)):

    # loop over triggers for each sample
    for trig in getattr(paths,samples.names[samplenum]):
        trigname = trig + samples.names[samplenum]
        #import appropriate config snippet
        filename = "HLTriggerOffline.Egamma."+trig+"_cfi"
        trigdef =__import__( filename )
        import sys
        globals()[trigname] = getattr(sys.modules[filename],trig).clone()    # clone imported config
        setattr(globals()[trigname],"_TypedParameterizable__type","EmDQMReco")
        # setattr(globals()[trigname],"cutcollection",cms.InputTag(fiducialname))       # set preselacted generator collection
        setattr(globals()[trigname],"cutnum",cms.int32( samples.num[samplenum]  )) # cut value for preselection
        setattr(globals()[trigname],"pdgGen",cms.int32( samples.pdgid[samplenum])) #correct pdgId for MC matching
        egammaValidationSequenceReco *= globals()[trigname]                      # add to sequence


egammaValidationSequenceReco.remove(tmp)  # remove the initial dummy
