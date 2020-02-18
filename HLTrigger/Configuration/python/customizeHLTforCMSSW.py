import FWCore.ParameterSet.Config as cms

# helper fuctions
from HLTrigger.Configuration.common import *

# add one customisation function per PR
# - put the PR number into the name of the function
# - add a short comment
# for example:

# CCCTF tuning
# def customiseFor12718(process):
#     for pset in process._Process__psets.values():
#         if hasattr(pset,'ComponentType'):
#             if (pset.ComponentType == 'CkfBaseTrajectoryFilter'):
#                 if not hasattr(pset,'minGoodStripCharge'):
#                     pset.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
#     return process

def customiseFor27653(process):
   """ PR27653 : RecoTrackRefSelector has new parameter: invertRapidityCut
                 default value (for back compatibility) : cms.bool(False)
   """
   for prod in producers_by_type(process,"RecoTrackRefSelector"):
      if not hasattr(prod,"invertRapidityCut"):
         setattr(prod,"invertRapidityCut",cms.bool(False))
#      for p in prod.parameterNames_():
#         print p
   return process

def customiseFor2017DtUnpacking(process):
    """Adapt the HLT to run the legacy DT unpacking
    for pre2018 data/MC workflows as the default"""

    if hasattr(process,'hltMuonDTDigis'):
        process.hltMuonDTDigis = cms.EDProducer( "DTUnpackingModule",
            useStandardFEDid = cms.bool( True ),
            maxFEDid = cms.untracked.int32( 779 ),
            inputLabel = cms.InputTag( "rawDataCollector" ),
            minFEDid = cms.untracked.int32( 770 ),
            dataType = cms.string( "DDU" ),
            readOutParameters = cms.PSet(
                localDAQ = cms.untracked.bool( False ),
                debug = cms.untracked.bool( False ),
                rosParameters = cms.PSet(
                    localDAQ = cms.untracked.bool( False ),
                    debug = cms.untracked.bool( False ),
                    writeSC = cms.untracked.bool( True ),
                    readDDUIDfromDDU = cms.untracked.bool( True ),
                    readingDDU = cms.untracked.bool( True ),
                    performDataIntegrityMonitor = cms.untracked.bool( False )
                    ),
                performDataIntegrityMonitor = cms.untracked.bool( False )
                ),
            dqmOnly = cms.bool( False )
        )

    return process

def customiseFor27694(process) :

    for producer in esproducers_by_type(process, "PixelCPETemplateRecoESProducer"):
        if hasattr(producer, "DoCosmics"): del producer.DoCosmics

    for producer in esproducers_by_type(process, "PixelCPEGenericESProducer"):
        if hasattr(producer, "TanLorentzAnglePerTesla"): del producer.TanLorentzAnglePerTesla
        if hasattr(producer, "PixelErrorParametrization"): del producer.PixelErrorParametrization

    return process

def customiseForPFRecHitHcalUpdate(process) :

   listHltPFRecHitHBHE=['hltParticleFlowRecHitHBHE',
                        'hltParticleFlowRecHitHBHEForEgamma',
                        'hltParticleFlowRecHitHBHEForEgammaUnseeded',
                        'hltParticleFlowRecHitHBHEForMuons',
                        'hltParticleFlowRecHitHBHERegForMuons']
   for att in listHltPFRecHitHBHE:
      if hasattr(process,att):
         prod = getattr(process, att)
         pset_navi = prod.navigator
         if hasattr(pset_navi, "sigmaCut"): delattr(pset_navi,'sigmaCut')
         if hasattr(pset_navi, "timeResolutionCalc"): delattr(pset_navi,'timeResolutionCalc')
         pset_navi.detectorEnums = cms.vint32(1,2)

   listHltPFRecHitHF=['hltParticleFlowRecHitHF',
                      'hltParticleFlowRecHitHFForEgammaUnseeded']
   for att in listHltPFRecHitHF:
      if hasattr(process,att):
         prod = getattr(process, att)
         pset_navi = prod.navigator
         if hasattr(pset_navi, "barrel"): delattr(pset_navi,'barrel')
         if hasattr(pset_navi, "endcap"): delattr(pset_navi,'endcap')
         pset_navi.detectorEnums = cms.vint32(4)

   return process

# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)
    process = customiseFor27653(process)

    process = customiseFor27694(process)

    process = customiseForPFRecHitHcalUpdate(process)

    return process
