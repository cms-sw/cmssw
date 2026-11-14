import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQMOffline.RecoB.bTagMiniDQMTaggers import ParticleNetPuppiForwardDiscriminators
from DQMOffline.RecoB.bTagMiniDQMTaggers import UParTDiscriminators

from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff import patJets

# add jets with pfSecondaryVertexTagInfos
patJetsSVInfo = patJets.clone(
    tagInfoSources = ['pfSecondaryVertexTagInfos'],
    addTagInfos = True
)
patJetsSVInfoTask = cms.Task(patJetsSVInfo)


bTagSVDQM = DQMEDAnalyzer('MiniAODSVAnalyzer', cms.PSet(
    JetTag = cms.InputTag('patJetsSVInfo'),
    svTagInfo = cms.string('pfSecondaryVertex'),
    ptMin = cms.double(30.),
    etaMax = cms.double(2.5),
   )
)

bTagMiniDQMGlobal = cms.PSet(
    JetTag = cms.InputTag('slimmedJetsPuppi'),
    MClevel = cms.int32(0),
    differentialPlots = cms.bool(True),
    ptActive = cms.bool(False),
    ptMin = cms.double(30.),
    ptMax = cms.double(40000.),
)
bTagMiniValidationGlobal = bTagMiniDQMGlobal.clone(
    MClevel = 1 # produce flavour plots for b, c ,light (dusg)
)
bTagMiniValidationGlobalUParT = bTagMiniDQMGlobal.clone(
    MClevel = 3 # produce flavour plots for b, c, light split into d, u, s, g (odd: ALL is already booked by DQM)
)

# Eta regions
Etaregions = {
    'Global': cms.PSet(
        etaActive = cms.bool(False),
        etaMin = cms.double(0.),
        etaMax = cms.double(2.5),
    ),
    'Barrel': cms.PSet(
        etaActive = cms.bool(True),
        etaMin = cms.double(0.),
        etaMax = cms.double(1.4),
    ),
    'Endcap': cms.PSet(
        etaActive = cms.bool(True),
        etaMin = cms.double(1.4),
        etaMax = cms.double(2.5),
    ),
}


# For DQM
bTagMiniDQMSource = cms.Sequence()
bTagMiniDQMHarvesting = cms.Sequence()

# For Validation
bTagMiniValidationSource = cms.Sequence()
bTagMiniValidationHarvesting = cms.Sequence()

#####################################################################################
#
# Setup DQM and Validation plots for ParticleNet and UParT taggers' outputs
#
#####################################################################################
def addSequences(Analyzer, Harvester, discriminators, regions, globalPSet, label='bTag'):
    # the jet kinematics, which the diffEff plots of every discriminator divide by, are booked by exactly one
    if sum(d.bookKinematics.value() for d in discriminators.values()) != 1:
        raise RuntimeError(label + ': exactly one discriminator must have bookKinematics = True')
    for discr in discriminators.keys():
        for region in regions.keys():
            name = label + discr + region

            globals()[name + 'Analyzer'] = DQMEDAnalyzer('MiniAODTaggerAnalyzer',    cms.PSet(globalPSet, discriminators[discr], regions[region]))
            globals()[name + 'Harvester'] = DQMEDHarvester('MiniAODTaggerHarvester', cms.PSet(globalPSet, discriminators[discr], regions[region]))

            Analyzer.insert(-1, globals()[name + 'Analyzer'])
            Harvester.insert(-1, globals()[name + 'Harvester'])

taggersToAnalyze = {
    'bTagParticleNetForward': {
        'discriminators': ParticleNetPuppiForwardDiscriminators,
        'regions': {
            'Forward': cms.PSet(
                etaActive = cms.bool(True),
                etaMin = cms.double(2.5),
                etaMax = cms.double(5.0),
            ),
        },
    },
    'bTagUParT': {
        'discriminators': UParTDiscriminators,
        'regions': Etaregions
    }
}

for tagger in taggersToAnalyze:
    # DQM
    addSequences(bTagMiniDQMSource,
                 bTagMiniDQMHarvesting,
                 discriminators=taggersToAnalyze[tagger]['discriminators'],
                 regions=taggersToAnalyze[tagger]['regions'],
                 globalPSet=bTagMiniDQMGlobal,
                 label=tagger+'DQM')

    # Validation
    addSequences(bTagMiniValidationSource,
                 bTagMiniValidationHarvesting,
                 discriminators=taggersToAnalyze[tagger]['discriminators'],
                 regions=taggersToAnalyze[tagger]['regions'],
                 globalPSet=bTagMiniValidationGlobalUParT if "UParT" in tagger else bTagMiniValidationGlobal,
                 label=tagger+'Validation')

#####################################################################################
#
# Setup Validation plots for ParticleNet and UParT taggers' inputs
#
#####################################################################################
# Jets in the tracker-coverage region
patJetsPuppiTagInfoAnalyzerDQM = DQMEDAnalyzer('MiniAODTagInfoAnalyzer', cms.PSet(
    jets = cms.InputTag('updatedPatJetsSlimmedPuppiWithDeepTags'),
    jetTagInfos = cms.vstring(
        "pfUnifiedParticleTransformerAK4TagInfosSlimmedPuppiWithDeepTags",
    ),
    ptMin = cms.double(30.),
    absEtaMin = cms.double(0.0),
    absEtaMax = cms.double(2.5),
    jetPartonFlavour = cms.int32(-1),#Inclusive flavour since DQM is for data
   )
)
bTagMiniDQMSource += patJetsPuppiTagInfoAnalyzerDQM

# Jets outside tracker-coverage region. Only ParticleNet
patJetsPuppiForwardTagInfoAnalyzerDQM = patJetsPuppiTagInfoAnalyzerDQM.clone(
    jetTagInfos = cms.vstring(
        "pfParticleNetFromMiniAODAK4PuppiForwardTagInfosSlimmedPuppiWithDeepTags",
    ),
    absEtaMin = cms.double(2.5),
    absEtaMax = cms.double(5.0),
    jetPartonFlavour = cms.int32(-1),#Inclusive flavour since DQM is for data
)
bTagMiniDQMSource += patJetsPuppiForwardTagInfoAnalyzerDQM

#####################################################################################
#
# Setup Validation plots for ParticleNet and UParT taggers' inputs
#
#####################################################################################
# Jets in the tracker-coverage region (Inclusive flavour)
patJetsPuppiTagInfoAnalyzerValidation = patJetsPuppiTagInfoAnalyzerDQM.clone()
bTagMiniValidationSource += patJetsPuppiTagInfoAnalyzerValidation

# Jets in the tracker-coverage region (B flavour)
patJetsPuppiTagInfoAnalyzerBJetsValidation = patJetsPuppiTagInfoAnalyzerValidation.clone(jetPartonFlavour=5)
bTagMiniValidationSource += patJetsPuppiTagInfoAnalyzerBJetsValidation

# Jets in the tracker-coverage region (C flavour)
patJetsPuppiTagInfoAnalyzerCJetsValidation = patJetsPuppiTagInfoAnalyzerValidation.clone(jetPartonFlavour=4)
bTagMiniValidationSource += patJetsPuppiTagInfoAnalyzerCJetsValidation

# Jets in the tracker-coverage region (L flavour: uds+g)
patJetsPuppiTagInfoAnalyzerLJetsValidation = patJetsPuppiTagInfoAnalyzerValidation.clone(jetPartonFlavour=1)
bTagMiniValidationSource += patJetsPuppiTagInfoAnalyzerLJetsValidation

# Jets outside tracker-coverage region (Inclusive flavour). Only ParticleNet
patJetsPuppiForwardTagInfoAnalyzerValidation = patJetsPuppiForwardTagInfoAnalyzerDQM.clone()
bTagMiniValidationSource += patJetsPuppiForwardTagInfoAnalyzerValidation
#####################################################################################
#
# Setup modifiers here
#
#####################################################################################
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

toExcludeDQMSource_pp_on_AA = [
    patJetsPuppiTagInfoAnalyzerDQM,
    patJetsPuppiForwardTagInfoAnalyzerDQM,
]

toExcludeValidationSource_pp_on_AA = [
    patJetsPuppiTagInfoAnalyzerValidation,
    patJetsPuppiTagInfoAnalyzerBJetsValidation,
    patJetsPuppiTagInfoAnalyzerCJetsValidation,
    patJetsPuppiTagInfoAnalyzerLJetsValidation,
    patJetsPuppiForwardTagInfoAnalyzerValidation
]

_mAOD = (pp_on_AA)
_mAOD.toReplaceWith(bTagMiniDQMSource, bTagMiniDQMSource.copyAndExclude(toExcludeDQMSource_pp_on_AA))
_mAOD.toReplaceWith(bTagMiniValidationSource, bTagMiniValidationSource.copyAndExclude(toExcludeValidationSource_pp_on_AA))

