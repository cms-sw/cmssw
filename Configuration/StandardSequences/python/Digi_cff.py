import FWCore.ParameterSet.Config as cms

#                                                    
# Full-scale Digitization of the simulated hits      
# in all CMS subdets : Tracker, ECAL, HCAl, Muon's;  
# MixingModule (at least in zero-pileup mode) needs  
# to be included to make Digi's operational, since   
# it's required for ECAL/HCAL & Muon's                
# Defined in a separate fragment
#                                                    
# Tracker Digis (Pixel + SiStrips) are now made in the mixing
# module, so the old "trDigi" sequence has been taken out.
#

# Calorimetry Digis (Ecal + Hcal) - * unsuppressed *
# returns sequence "calDigi"
from SimCalorimetry.Configuration.SimCalorimetry_cff import *
# Muon Digis (CSC + DT + RPC)
# returns sequence "muonDigi"
#
from SimMuon.Configuration.SimMuon_cff import *
#
# TrackingParticle Producer is now part of the mixing module, so
# it is no longer run here.
#
from SimGeneral.Configuration.SimGeneral_cff import *

# add updating the GEN information by default
from Configuration.StandardSequences.Generator_cff import *
from GeneratorInterface.Core.generatorSmeared_cfi import *

doAllDigi = cms.Sequence(generatorSmeared*calDigi+muonDigi)
pdigi = cms.Sequence(generatorSmeared*fixGenInfo*cms.SequencePlaceholder("randomEngineStateProducer")*cms.SequencePlaceholder("mix")*doAllDigi*addPileupInfo)
pdigi_valid = cms.Sequence(pdigi)
pdigi_nogen=cms.Sequence(generatorSmeared*cms.SequencePlaceholder("randomEngineStateProducer")*cms.SequencePlaceholder("mix")*doAllDigi*addPileupInfo)
pdigi_valid_nogen=cms.Sequence(pdigi_nogen)

from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    # pretend these digis have been through digi2raw and raw2digi, by using the approprate aliases
    # use an alias to make the mixed track collection available under the usual label
    from FastSimulation.Configuration.DigiAliases_cff import loadDigiAliases
    loadDigiAliases(premix = False)
    from FastSimulation.Configuration.DigiAliases_cff import generalTracks,ecalPreshowerDigis,ecalDigis,hcalDigis,muonDTDigis,muonCSCDigis,muonRPCDigis

#phase 2 common mods
def _modifyDigitizerPhase2Common( theProcess ):
    theProcess.load("CalibCalorimetry/HcalPlugins/Hcal_Conditions_forGlobalTag_cff")
    theProcess.es_hardcode.toGet = cms.untracked.vstring(
                'GainWidths',
                'MCParams',
                'RecoParams',
                'RespCorrs',
                'QIEData',
                'QIETypes',
                'Gains',
                'Pedestals',
                'PedestalWidths',
                'ChannelQuality',
                'ZSThresholds',
                'TimeCorrs',
                'LUTCorrs',
                'LutMetadata',
                'L1TriggerObjects',
                'PFCorrs',
                'ElectronicsMap',
                'CholeskyMatrices',
                'CovarianceMatrices'
                )

    # Special Upgrade trick (if absent - regular case assumed)
    theProcess.es_hardcode.GainWidthsForTrigPrims = cms.bool(True)
    theProcess.es_hardcode.HEreCalibCutoff = cms.double(100.)
    theProcess.mix.digitizers.hcal.HBHEUpgradeQIE = True
    theProcess.mix.digitizers.hcal.hb.siPMCells = cms.vint32([1])
    theProcess.mix.digitizers.hcal.hb.photoelectronsToAnalog = cms.vdouble([10.]*16)
    theProcess.mix.digitizers.hcal.hb.pixels = cms.int32(4500*4*2)
    theProcess.mix.digitizers.hcal.he.photoelectronsToAnalog = cms.vdouble([10.]*16)
    theProcess.mix.digitizers.hcal.he.pixels = cms.int32(4500*4*2)
    theProcess.mix.digitizers.hcal.HFUpgradeQIE = True
    theProcess.mix.digitizers.hcal.HcalReLabel.RelabelHits=cms.untracked.bool(True)
    theProcess.simHcalDigis.useConfigZSvalues=cms.int32(1)
    theProcess.simHcalDigis.HBlevel=cms.int32(16)
    theProcess.simHcalDigis.HElevel=cms.int32(16)
    theProcess.simHcalDigis.HOlevel=cms.int32(16)
    theProcess.simHcalDigis.HFlevel=cms.int32(16)

    theProcess.hcalDigiSequence.remove(theProcess.simHcalTriggerPrimitiveDigis)
    theProcess.hcalDigiSequence.remove(theProcess.simHcalTTPDigis)

#HGCal
def _modifyDigitizerForHGCal( theProcess ):
    theProcess.load('SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi')    
    theProcess.mix.digitizers.hgceeDigitizer=theProcess.hgceeDigitizer
    theProcess.mix.digitizers.hgchebackDigitizer=theProcess.hgchebackDigitizer
    theProcess.mix.digitizers.hgchefrontDigitizer=theProcess.hgchefrontDigitizer
    newFactors = cms.vdouble(
        210.55, 197.93, 186.12, 189.64, 189.63,
        189.96, 190.03, 190.11, 190.18, 190.25,
        190.32, 190.40, 190.47, 190.54, 190.61,
        190.69, 190.83, 190.94, 190.94, 190.94,
        190.94, 190.94, 190.94, 190.94, 190.94,
        190.94, 190.94, 190.94, 190.94, 190.94,
        190.94, 190.94, 190.94, 190.94, 190.94,
        190.94, 190.94, 190.94, 190.94, 190.94)
    theProcess.mix.digitizers.hcal.he.samplingFactors = newFactors
    theProcess.mix.digitizers.hcal.he.photoelectronsToAnalog = cms.vdouble([10.]*len(newFactors))
    # Also need to tell the MixingModule to make the correct collections available from
    # the pileup, even if not creating CrossingFrames.
    theProcess.mix.mixObjects.mixCH.input.append( cms.InputTag("g4SimHits",theProcess.hgceeDigitizer.hitCollection.value()) )
    theProcess.mix.mixObjects.mixCH.input.append( cms.InputTag("g4SimHits",theProcess.hgchebackDigitizer.hitCollection.value()) )
    theProcess.mix.mixObjects.mixCH.input.append( cms.InputTag("g4SimHits",theProcess.hgchefrontDigitizer.hitCollection.value()) )
    theProcess.mix.mixObjects.mixCH.subdets.append( theProcess.hgceeDigitizer.hitCollection.value() )
    theProcess.mix.mixObjects.mixCH.subdets.append( theProcess.hgchebackDigitizer.hitCollection.value() )
    theProcess.mix.mixObjects.mixCH.subdets.append( theProcess.hgchefrontDigitizer.hitCollection.value() )    

from Configuration.StandardSequences.Eras import eras
modifyDigitizerForHGCal_ = eras.phase2_hgcal.makeProcessModifier( _modifyDigitizerForHGCal )
modifyDigitizerPhase2Common_ = eras.phase2_common.makeProcessModifier( _modifyDigitizerPhase2Common )


