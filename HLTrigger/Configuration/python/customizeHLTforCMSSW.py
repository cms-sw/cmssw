import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.common import *

# one action function per PR - put the PR number into the name of the function

# example:
# def customiseFor12718(process):
#     for pset in process._Process__psets.values():
#         if hasattr(pset,'ComponentType'):
#             if (pset.ComponentType == 'CkfBaseTrajectoryFilter'):
#                 if not hasattr(pset,'minGoodStripCharge'):
#                     pset.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
#     return process

# Module restructuring for PR #15440
def customiseFor15440(process):
    for producer in producers_by_type(process, "EgammaHLTBcHcalIsolationProducersRegional", "EgammaHLTEcalPFClusterIsolationProducer", "EgammaHLTHcalPFClusterIsolationProducer", "MuonHLTEcalPFClusterIsolationProducer", "MuonHLTHcalPFClusterIsolationProducer"):
        if hasattr(producer, "effectiveAreaBarrel") and hasattr(producer, "effectiveAreaEndcap"):
            if not hasattr(producer, "effectiveAreas") and not hasattr(producer, "absEtaLowEdges"):
                producer.absEtaLowEdges = cms.vdouble( 0.0, 1.479 )
                producer.effectiveAreas = cms.vdouble( producer.effectiveAreaBarrel.value(), producer.effectiveAreaEndcap.value() )
                del producer.effectiveAreaBarrel
                del producer.effectiveAreaEndcap
    return process

# Add quadruplet-specific pixel track duplicate cleaning mode (PR #13753)
def customiseFor13753(process):
    for producer in producers_by_type(process, "PixelTrackProducer"):
        if producer.CleanerPSet.ComponentName.value() == "PixelTrackCleanerBySharedHits" and not hasattr(producer.CleanerPSet, "useQuadrupletAlgo"):
            producer.CleanerPSet.useQuadrupletAlgo = cms.bool(False)
    return process

# Add pixel seed extension (PR #14356)
def customiseFor14356(process):
    for name, pset in process.psets_().iteritems():
        if hasattr(pset, "ComponentType") and pset.ComponentType.value() == "CkfBaseTrajectoryFilter" and not hasattr(pset, "pixelSeedExtension"):
            pset.pixelSeedExtension = cms.bool(False)
    return process

def customiseFor14833(process):
    for producer in esproducers_by_type(process, "DetIdAssociatorESProducer"):
        if (producer.ComponentName.value() == 'MuonDetIdAssociator'):
            if not hasattr(producer,'includeGEM'):
                producer.includeGEM = cms.bool(False)
            if not hasattr(producer,'includeME0'):
                producer.includeME0 = cms.bool(False)
    return process

def customiseFor16670(process):
    for producer in esproducers_by_type(process, "DetIdAssociatorESProducer"):
        if (producer.ComponentName.value() == 'HcalDetIdAssociator'):
            if not hasattr(producer,'hcalRegion'):
                producer.hcalRegion = cms.int32(2)
    return process

def customiseFor15499(process):
    for producer in producers_by_type(process,"HcalHitReconstructor"):
        producer.ts4Max = cms.vdouble(100.0,70000.0)
        if (producer.puCorrMethod.value() == 2):
            producer.timeSigmaHPD = cms.double(5.0)
            producer.timeSigmaSiPM = cms.double(3.5)
            producer.pedSigmaHPD = cms.double(0.5)
            producer.pedSigmaSiPM = cms.double(1.5)
            producer.noiseHPD = cms.double(1.0)
            producer.noiseSiPM = cms.double(2.)
    return process

def customiseFor16569(process):
    for mod in ['hltHbhereco','hltHbherecoMethod2L1EGSeeded','hltHbherecoMethod2L1EGUnseeded','hltHfreco','hltHoreco']:
        if hasattr(process,mod):
            getattr(process,mod).ts4chi2 = cms.vdouble(15.,5000.)

    return process

def customiseFor17094(process):
    for mod in ['hltHbhereco','hltHbherecoMethod2L1EGSeeded','hltHbherecoMethod2L1EGUnseeded','hltHfreco','hltHoreco']:
        if hasattr(process,mod):
            getattr(process,mod).timeSigmaSiPM = cms.double(2.5)
            getattr(process,mod).pedSigmaSiPM = cms.double(0.00065)
            getattr(process,mod).noiseSiPM = cms.double(1)
            getattr(process,mod).ts4Max = cms.vdouble(100.,45000.)
            getattr(process,mod).ts4chi2 = cms.vdouble(15.,15.)

    return process

# Move pixel track fitter, filter, and cleaner to ED/ESProducts (PR #16792)
def customiseFor16792(process):
    def _copy(old, new, skip=[]):
        skipSet = set(skip)
        for key in old.parameterNames_():
            if key not in skipSet:
                setattr(new, key, getattr(old, key))

    from RecoPixelVertexing.PixelTrackFitting.pixelTrackCleanerBySharedHits_cfi import pixelTrackCleanerBySharedHits as _pixelTrackCleanerBySharedHits
    from RecoPixelVertexing.PixelLowPtUtilities.trackCleaner_cfi import trackCleaner as _trackCleaner

    for producer in producers_by_type(process, "PixelTrackProducer"):
        label = producer.label()
        fitterName = producer.FitterPSet.ComponentName.value()
        filterName = producer.FilterPSet.ComponentName.value()

        fitterProducerLabel = label+"Fitter"
        fitterProducerName = fitterName+"Producer"
        fitterProducer = cms.EDProducer(fitterProducerName)
        skip = ["ComponentName"]
        if fitterName == "PixelFitterByHelixProjections":
            skip.extend(["TTRHBuilder", "fixImpactParameter"]) # some HLT producers use these parameters even if they have no effect
        _copy(producer.FitterPSet, fitterProducer, skip=skip)
        setattr(process, fitterProducerLabel, fitterProducer)
        del producer.FitterPSet
        producer.Fitter = cms.InputTag(fitterProducerLabel)

        filterProducerLabel = label+"Filter"
        filterProducerName = filterName+"Producer"
        filterProducer = cms.EDProducer(filterProducerName)
        _copy(producer.FilterPSet, filterProducer, skip=["ComponentName"])
        setattr(process, filterProducerLabel, filterProducer)

        del producer.FilterPSet
        producer.Filter = cms.InputTag(filterProducerLabel)
        if hasattr(producer, "useFilterWithES"): # useFilterWithES has no effect anymore
            del producer.useFilterWithES

        cleanerPSet = producer.CleanerPSet
        del producer.CleanerPSet
        producer.Cleaner = cms.string("")
        if cleanerPSet.ComponentName.value() == "PixelTrackCleanerBySharedHits":
            if cleanerPSet.useQuadrupletAlgo:
                producer.cleaner = "hltPixelTracksCleanerBySharedHitsQuad"
                if not hasattr(process, "hltPixelTracksCleanerBySharedHitsQuad"):
                    process.hltPixelTracksCleanerBySharedHitsQuad = _pixelTrackCleanerBySharedHits.clone(
                        ComponentName = "hltPixelTracksCleanerBySharedHitsQuad",
                        useQuadrupletAlgo=True
                    )
            else:
                producer.Cleaner = "hltPixelTracksCleanerBySharedHits"
                if not hasattr(process, "hltPixelTracksCleanerBySharedHits"):
                    process.hltPixelTracksCleanerBySharedHits = _pixelTrackCleanerBySharedHits.clone(
                        ComponentName = "hltPixelTracksCleanerBySharedHits",
                        useQuadrupletAlgo=False
                    )
        elif cleanerPSet.ComponentName.value() == "TrackCleaner":
            producer.Cleaner = "hltTrackCleaner"
            if not hasattr(process, "hltTrackCleaner"):
                process.hltTrackCleaner = _trackCleaner.clone(
                    ComponentName = "hltTrackCleaner"
                )

        # Modify sequences (also paths to be sure, altough in practice
        # the seeding modules should be only in sequences in HLT?)
        for seqs in [process.sequences_(), process.paths_()]:
            for seqName, seq in seqs.iteritems():
                # cms.Sequence.replace() would look simpler, but it expands
                # the contained sequences if a replacement occurs there.
                try:
                    index = seq.index(producer)
                except:
                    continue
                seq.insert(index, fitterProducer)
                seq.insert(index, filterProducer)

    return process

#
# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

#   only for non-development frozen menus

    import os
    cmsswVersion = os.environ['CMSSW_VERSION']

    if cmsswVersion >= "CMSSW_8_1":
      if menuType == "25ns15e33_v4":
        print "# Applying 81X customization for ",menuType
        process = customiseFor14356(process)
        process = customiseFor13753(process)
        process = customiseFor14833(process)
        process = customiseFor15440(process)
        process = customiseFor15499(process)
        process = customiseFor16569(process)
#       process = customiseFor12718(process)
        process = customiseFor16670(process)
        pass

    if cmsswVersion >= "CMSSW_9_0":
        print "# Applying 90X customization for ",menuType
        process = customiseFor16792(process)
        process = customiseFor17094(process)
        pass

#   stage-2 changes only if needed
    if ("Fake" in menuType):
        return process

    

#    if ( menuType in ("FULL","GRun","PIon")):
#        from HLTrigger.Configuration.CustomConfigs import L1XML
#        process = L1XML(process,"L1Menu_Collisions2016_dev_v3.xml")
#        from HLTrigger.Configuration.CustomConfigs import L1REPACK
#        process = L1REPACK(process)
#
#    _debug = False
#
#   special case
#    for module in filters_by_type(process,"HLTL1TSeed"):
#        label = module._Labelable__label
#        if hasattr(getattr(process,label),'SaveTags'):
#            delattr(getattr(process,label),'SaveTags')
#
#   replace converted l1extra=>l1t plugins which are not yet in ConfDB
#    replaceList = {
#        'EDAnalyzer' : { },
#        'EDFilter'   : {
#            'HLTMuonL1Filter' : 'HLTMuonL1TFilter',
#            'HLTMuonL1RegionalFilter' : 'HLTMuonL1TRegionalFilter',
#            'HLTMuonTrkFilter' : 'HLTMuonTrkL1TFilter',
#            'HLTMuonL1toL3TkPreFilter' : 'HLTMuonL1TtoL3TkPreFilter',
#            'HLTMuonDimuonL2Filter' : 'HLTMuonDimuonL2FromL1TFilter',
#            'HLTEgammaL1MatchFilterRegional' : 'HLTEgammaL1TMatchFilterRegional',
#            'HLTMuonL2PreFilter' : 'HLTMuonL2FromL1TPreFilter',
#            'HLTPixelIsolTrackFilter' : 'HLTPixelIsolTrackL1TFilter',
#            },
#        'EDProducer' : {
#            'CaloTowerCreatorForTauHLT' : 'CaloTowerFromL1TCreatorForTauHLT',
#            'L1HLTTauMatching' : 'L1THLTTauMatching',
#            'HLTCaloJetL1MatchProducer' : 'HLTCaloJetL1TMatchProducer',
#            'HLTPFJetL1MatchProducer' : 'HLTPFJetL1TMatchProducer',
#            'HLTL1MuonSelector' : 'HLTL1TMuonSelector',
#            'L2MuonSeedGenerator' : 'L2MuonSeedGeneratorFromL1T',
#            'IsolatedPixelTrackCandidateProducer' : 'IsolatedPixelTrackCandidateL1TProducer',
#            }
#        }
#    for type,list in replaceList.iteritems():
#        if (type=="EDAnalyzer"):
#            if _debug:
#                print "# Replacing EDAnalyzers:"
#            for old,new in list.iteritems():
#                if _debug:
#                    print '## EDAnalyzer plugin type: ',old,' -> ',new
#                for module in analyzers_by_type(process,old):
#                    label = module._Labelable__label
#                    if _debug:
#                        print '### Instance: ',label
#                    setattr(process,label,cms.EDAnalyzer(new,**module.parameters_()))
#        elif (type=="EDFilter"):
#            if _debug:
#                print "# Replacing EDFilters  :"
#            for old,new in list.iteritems():
#                if _debug:
#                    print '## EDFilter plugin type  : ',old,' -> ',new
#                for module in filters_by_type(process,old):
#                    label = module._Labelable__label
#                    if _debug:
#                        print '### Instance: ',label
#                    setattr(process,label,cms.EDFilter(new,**module.parameters_()))
#        elif (type=="EDProducer"):
#            if _debug:
#                print "# Replacing EDProducers:"
#            for old,new in list.iteritems():
#                if _debug:
#                    print '## EDProducer plugin type: ',old,' -> ',new
#                for module in producers_by_type(process,old):
#                    label = module._Labelable__label
#                    if _debug:
#                        print '### Instance: ',label
#                    setattr(process,label,cms.EDProducer(new,**module.parameters_()))
#                    if (new == 'CaloTowerFromL1TCreatorForTauHLT'):
#                        setattr(getattr(process,label),'TauTrigger',cms.InputTag('hltCaloStage2Digis:Tau'))
#                    if ((new == 'HLTCaloJetL1TMatchProducer') or (new == 'HLTPFJetL1TMatchProducer')):
#                        setattr(getattr(process,label),'L1Jets',cms.InputTag('hltCaloStage2Digis:Jet'))
#                        if hasattr(getattr(process,label),'L1CenJets'):
#                            delattr(getattr(process,label),'L1CenJets')
#                        if hasattr(getattr(process,label),'L1ForJets'):
#                            delattr(getattr(process,label),'L1ForJets')
#                        if hasattr(getattr(process,label),'L1TauJets'):
#                            delattr(getattr(process,label),'L1TauJets')
#                    if (new == 'HLTL1TMuonSelector'):
#                        setattr(getattr(process,label),'InputObjects',cms.InputTag('hltGmtStage2Digis:Muon'))
#                    if (new == 'L2MuonSeedGeneratorFromL1T'):
#                        setattr(getattr(process,label),'GMTReadoutCollection',cms.InputTag(''))            
#                        setattr(getattr(process,label),'InputObjects',cms.InputTag('hltGmtStage2Digis:Muon'))
#                    if (new == 'IsolatedPixelTrackCandidateL1TProducer'):
#                        setattr(getattr(process,label),'L1eTauJetsSource',cms.InputTag('hltCaloStage2Digis:Tau'))
#
#        else:
#            if _debug:
#                print "# Error - Type ',type,' not recognised!"
#
#   Both of the HLTEcalRecHitInAllL1RegionsProducer instances need InputTag fixes
#    for module in producers_by_type(process,'HLTEcalRecHitInAllL1RegionsProducer'):
#        label = module._Labelable__label
#        setattr(getattr(process,label).l1InputRegions[0],'inputColl',cms.InputTag('hltCaloStage2Digis:EGamma'))
#        setattr(getattr(process,label).l1InputRegions[0],'type',cms.string("EGamma"))
#        setattr(getattr(process,label).l1InputRegions[1],'inputColl',cms.InputTag('hltCaloStage2Digis:EGamma'))
#        setattr(getattr(process,label).l1InputRegions[1],'type',cms.string("EGamma"))
#        setattr(getattr(process,label).l1InputRegions[2],'inputColl',cms.InputTag('hltCaloStage2Digis:Jet'))
#        setattr(getattr(process,label).l1InputRegions[2],'type',cms.string("Jet"))
#
#   One of the EgammaHLTCaloTowerProducer instances need InputTag fixes
#    if hasattr(process,'hltRegionalTowerForEgamma'):
#        setattr(getattr(process,'hltRegionalTowerForEgamma'),'L1NonIsoCand',cms.InputTag('hltCaloStage2Digis:EGamma'))
#        setattr(getattr(process,'hltRegionalTowerForEgamma'),'L1IsoCand'   ,cms.InputTag('hltCaloStage2Digis:EGamma'))
#
#   replace remaining l1extra modules with filter returning 'false'
#    badTypes = (
#        'HLTLevel1Activity',
#        )
#    if _debug:
#        print "# Unconverted module types: ",badTypes
#    badModules = [ ]
#    for badType in badTypes:
#        if _debug:
#            print '## Unconverted module type: ',badType
#        for module in analyzers_by_type(process,badType):
#            label = module._Labelable__label
#            badModules += [label]
#            if _debug:
#                print '### analyzer label: ',label
#        for module in filters_by_type(process,badType):
#            label = module._Labelable__label
#            badModules += [label]
#            if _debug:
#                print '### filter   label: ',label
#        for module in producers_by_type(process,badType):
#            label = module._Labelable__label
#            badModules += [label]
#            if _debug:
#                print '### producer label: ',label
#    for label in badModules:
#        setattr(process,label,cms.EDFilter("HLTBool",result=cms.bool(False)))

    return process
