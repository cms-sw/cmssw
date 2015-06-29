import FWCore.ParameterSet.Config as cms
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE import customise as customiseBE
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5D import customise as customiseBE5D
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5DPixel10D import customise as customiseBE5DPixel10D
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5DPixel10DLHCC import customise as customiseBE5DPixel10DLHCC
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5DPixel10Ddev import customise as customiseBE5DPixel10Ddev

from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE import l1EventContent as customise_ev_BE
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5D import l1EventContent as customise_ev_BE5D
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5DPixel10D import l1EventContent as customise_ev_BE5DPixel10D

from SLHCUpgradeSimulations.Configuration.phase1TkCustomsPixel10D import customise as customisePhase1TkPixel10D
from SLHCUpgradeSimulations.Configuration.combinedCustoms_TTI import customise as customiseTTI
from SLHCUpgradeSimulations.Configuration.combinedCustoms_TTI import l1EventContent_TTI as customise_ev_l1tracker
from SLHCUpgradeSimulations.Configuration.combinedCustoms_TTI import l1EventContent_TTI_forHLT

from SLHCUpgradeSimulations.Configuration.customise_mixing import customise_NoCrossing
from SLHCUpgradeSimulations.Configuration.phase1TkCustoms import customise as customisePhase1Tk
from SLHCUpgradeSimulations.Configuration.phase1TkCustomsdev import customise as customisePhase1Tkdev
from SLHCUpgradeSimulations.Configuration.HCalCustoms import customise_HcalPhase1, customise_HcalPhase0, customise_HcalPhase2
from SLHCUpgradeSimulations.Configuration.gemCustoms import customise2019 as customise_gem2019
from SLHCUpgradeSimulations.Configuration.gemCustoms import customise2023 as customise_gem2023
from SLHCUpgradeSimulations.Configuration.me0Customs import customise as customise_me0
from SLHCUpgradeSimulations.Configuration.rpcCustoms import customise as customise_rpc
from SLHCUpgradeSimulations.Configuration.fastsimCustoms import customiseDefault as fastCustomiseDefault
from SLHCUpgradeSimulations.Configuration.fastsimCustoms import customisePhase2 as fastCustomisePhase2
from SLHCUpgradeSimulations.Configuration.customise_mixing import customise_noPixelDataloss as cNoPixDataloss
from SLHCUpgradeSimulations.Configuration.customise_ecalTime import cust_ecalTime
from SLHCUpgradeSimulations.Configuration.customise_shashlikTime import cust_shashlikTime
import SLHCUpgradeSimulations.Configuration.aging as aging
import SLHCUpgradeSimulations.Configuration.jetCustoms as jetCustoms


def cust_2023HGCalPandora_common(process):
    process=customisePostLS1(process)
    process=customiseBE5DPixel10D(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE5DPixel10D(process)
    process=customise_gem2023(process)
    process=customise_rpc(process)
    process=jetCustoms.customise_jets(process)
    if hasattr(process,'L1simulation_step'):
    	process.simEcalTriggerPrimitiveDigis.BarrelOnly = cms.bool(True)
    if hasattr(process,'digitisation_step'):
    	process.mix.digitizers.ecal.accumulatorType = cms.string('EcalPhaseIIDigiProducer')
        process.load('SimGeneral.MixingModule.hgcalDigitizer_cfi')
        process.mix.digitizers.hgceeDigitizer=process.hgceeDigitizer
        process.mix.digitizers.hgchebackDigitizer=process.hgchebackDigitizer
        process.mix.digitizers.hgchefrontDigitizer=process.hgchefrontDigitizer
        # Also need to tell the MixingModule to make the correct collections available from
        # the pileup, even if not creating CrossingFrames.
        process.mix.mixObjects.mixCH.input.append( cms.InputTag("g4SimHits",process.hgceeDigitizer.hitCollection.value()) )
        process.mix.mixObjects.mixCH.input.append( cms.InputTag("g4SimHits",process.hgchebackDigitizer.hitCollection.value()) )
        process.mix.mixObjects.mixCH.input.append( cms.InputTag("g4SimHits",process.hgchefrontDigitizer.hitCollection.value()) )
        process.mix.mixObjects.mixCH.subdets.append( process.hgceeDigitizer.hitCollection.value() )
        process.mix.mixObjects.mixCH.subdets.append( process.hgchebackDigitizer.hitCollection.value() )
        process.mix.mixObjects.mixCH.subdets.append( process.hgchefrontDigitizer.hitCollection.value() )
    if hasattr(process,'raw2digi_step'):
        process.ecalDigis.FEDs = cms.vint32(
            # EE-:
            #601, 602, 603, 604, 605,
            #606, 607, 608, 609,
            # EB-:
            610, 611, 612, 613, 614, 615,
            616, 617, 618, 619, 620, 621,
            622, 623, 624, 625, 626, 627,
            # EB+:
            628, 629, 630, 631, 632, 633,
            634, 635, 636, 637, 638, 639,
            640, 641, 642, 643, 644, 645,
            # EE+:
            #646, 647, 648, 649, 650,
            #651, 652, 653, 654
            )
        print "RAW2DIGI only for EB FEDs"
    if hasattr(process,'reconstruction_step'):
        process.particleFlowRecHitHGCNoEB = cms.Sequence(process.particleFlowRecHitHGCEE+process.particleFlowRecHitHGCHEF)
        process.particleFlowClusterHGCNoEB = cms.Sequence(process.particleFlowClusterHGCEE+process.particleFlowClusterHGCHEF)
        process.particleFlowCluster += process.particleFlowRecHitHGCNoEB
        process.particleFlowCluster += process.particleFlowClusterHGCNoEB
        if hasattr(process,'particleFlowSuperClusterECAL'):
            process.particleFlowSuperClusterHGCEE = process.particleFlowSuperClusterECAL.clone()
            process.particleFlowSuperClusterHGCEE.useHGCEmPreID = cms.bool(True)
            process.particleFlowSuperClusterHGCEE.PFClusters = cms.InputTag('particleFlowClusterHGCEE')
            process.particleFlowSuperClusterHGCEE.use_preshower = cms.bool(False)
            process.particleFlowSuperClusterHGCEE.PFSuperClusterCollectionEndcapWithPreshower = cms.string('')
            process.particleFlowCluster += process.particleFlowSuperClusterHGCEE
            if hasattr(process,'ecalDrivenElectronSeeds'):
                process.ecalDrivenElectronSeeds.endcapSuperClusters = cms.InputTag('particleFlowSuperClusterHGCEE')
                process.ecalDrivenElectronSeeds.SeedConfiguration.endcapHCALClusters = cms.InputTag('particleFlowClusterHGCHEF')
                process.ecalDrivenElectronSeeds.SeedConfiguration.hOverEMethodEndcap = cms.int32(3)
                process.ecalDrivenElectronSeeds.SeedConfiguration.hOverEConeSizeEndcap = cms.double(0.087)
                process.ecalDrivenElectronSeeds.SeedConfiguration.maxHOverEEndcaps = cms.double(0.1) 
                process.ecalDrivenElectronSeeds.SeedConfiguration.z2MinB = cms.double(-0.15)
                process.ecalDrivenElectronSeeds.SeedConfiguration.z2MaxB = cms.double(0.15)
                if hasattr(process,'ecalDrivenGsfElectrons'):
                    process.ecalDrivenGsfElectrons.hOverEMethodEndcap = cms.int32(3)
                    process.ecalDrivenGsfElectrons.hOverEConeSizeEndcap = cms.double(0.087)
                    process.ecalDrivenGsfElectrons.maxDeltaEtaEndcaps = cms.double(0.015)
                    process.ecalDrivenGsfElectrons.hcalEndcapClusters = cms.InputTag('particleFlowClusterHGCHEF')
                    if hasattr(process,'gsfElectrons'):
                        process.gsfElectrons.hOverEMethodEndcap = cms.int32(3)
                        process.gsfElectrons.hOverEConeSizeEndcap = cms.double(0.087)
                        process.gsfElectrons.maxDeltaEtaEndcaps = cms.double(0.015)
                        process.gsfElectrons.hcalEndcapClusters = cms.InputTag('particleFlowClusterHGCHEF')  
        # load pandora customization (note we have removed HGC clusters entirely from standard ParticleFlow
        # doing this)                
        process.load('RecoParticleFlow.PandoraTranslator.HGCalTrackCollection_cfi')
        process.load('RecoParticleFlow.PandoraTranslator.runPandora_cfi')
        process.pandorapfanew.pf_electron_output_col = process.particleFlowTmp.pf_electron_output_col
        process.particleFlowBlock.elementImporters[5].source = cms.InputTag('HGCalTrackCollection:TracksNotInHGCal')
        process.pandoraSequence = cms.Sequence(process.HGCalTrackCollection*
                                               process.particleFlowBlock*
                                               process.pandorapfanew)
        process.particleFlowReco.replace(process.particleFlowBlock,process.pandoraSequence)
        process.particleFlowBarrel = process.particleFlowTmp.clone()        
        process.particleFlowTmp = cms.EDProducer(
            "PFCandidateListMerger",
            src = cms.VInputTag("particleFlowBarrel",
                                "pandorapfanew"),
            src1 = cms.VInputTag("particleFlowBarrel:"+str(process.particleFlowTmp.pf_electron_output_col),
                                 "pandorapfanew:"+str(process.particleFlowTmp.pf_electron_output_col)),
            label1 = process.particleFlowTmp.pf_electron_output_col

            )
        process.mergedParticleFlowSequence = cms.Sequence(process.particleFlowBarrel*process.particleFlowTmp)
        process.particleFlowReco.replace(process.particleFlowTmp,process.mergedParticleFlowSequence)

    #mod event content
    process.load('RecoLocalCalo.Configuration.hgcalLocalReco_EventContent_cff')
    if hasattr(process,'FEVTDEBUGHLTEventContent'):
        process.FEVTDEBUGHLTEventContent.outputCommands.extend(process.hgcalLocalRecoFEVT.outputCommands)
        process.FEVTDEBUGHLTEventContent.outputCommands.append('keep *_particleFlowSuperClusterHGCEE_*_*')
        process.FEVTDEBUGHLTEventContent.outputCommands.append('keep *_pandorapfanew_*_*')
        
    if hasattr(process,'RECOSIMEventContent'):
        process.RECOSIMEventContent.outputCommands.extend(process.hgcalLocalRecoFEVT.outputCommands)
        process.RECOSIMEventContent.outputCommands.append('keep *_particleFlowSuperClusterHGCEE_*_*')
        process.RECOSIMEventContent.outputCommands.append('keep *_pandorapfanew_*_*')
    return process

def cust_2023HGCalPandoraMuon(process):
    process = cust_2023HGCalPandora_common(process)
    process = customise_me0(process)
    return process

def mask_layer(mask,layer):
    mask[layer-1] = 0

def mask_layers(layer_mask,masked_layers):
    for det in layer_mask.keys():
        for layer in masked_layers[det]:
            mask_layer(layer_mask[det],layer)

def propagate_layerdropping(process,layer_mask):
    process.pandorapfanew.DoLayerMasking = cms.bool(True)
    process.pandorapfanew.LayerMaskingParams = cms.PSet( EE_layerMask  = cms.vuint32(layer_mask['EE']),
                                                         HEF_layerMask = cms.vuint32(layer_mask['HEF']),
                                                         HEB_layerMask = cms.vuint32(layer_mask['HEB']) )
    for i in range(3): #reset HGC PFRecHit masking
        process.particleFlowRecHitHGCEE.producers[i].qualityTests[1].masking_info = process.pandorapfanew.LayerMaskingParams
    process.particleFlowClusterHGCEE.initialClusteringStep.emEnergyCalibration.masking_info = process.pandorapfanew.LayerMaskingParams
    process.particleFlowClusterHGCEE.initialClusteringStep.hadEnergyCalibration.masking_info = process.pandorapfanew.LayerMaskingParams
    return process

def cust_2023HGCalPandoraMuonScopeDoc_ee28_fh12(process):    
    layer_mask = {}
    layer_mask['EE']  = [1 for x in range(30)]
    layer_mask['HEF'] = [1 for x in range(12)]
    layer_mask['HEB'] = [1 for x in range(12)] 
    
    masked_layers = {}
    masked_layers['EE']  = []
    masked_layers['HEF'] = []
    masked_layers['HEB'] = []
    
    mask_layers(layer_mask,masked_layers)

    process = cust_2023HGCalPandora_common(process)
    process = customise_me0(process)   
    if hasattr(process,'reconstruction_step'):
        process = propagate_layerdropping(process,layer_mask)
    
    return process

def cust_2023HGCalPandoraMuonScopeDoc_ee24_fh11(process):
    layer_mask = {}
    layer_mask['EE']  = [1 for x in range(30)]
    layer_mask['HEF'] = [1 for x in range(12)]
    layer_mask['HEB'] = [1 for x in range(12)] 
    
    masked_layers = {}
    masked_layers['EE']  = [2, 11, 16 , 26, 28]
    masked_layers['HEF'] = [11]
    masked_layers['HEB'] = []
    
    mask_layers(layer_mask,masked_layers)

    process = cust_2023HGCalPandora_common(process)
    process = customise_me0(process)
    if hasattr(process,'reconstruction_step'):
        process = propagate_layerdropping(process,layer_mask)
    
    return process

def cust_2023HGCalPandoraMuonScopeDoc_ee18_fh9(process):
    layer_mask = {}
    layer_mask['EE']  = [1 for x in range(30)]
    layer_mask['HEF'] = [1 for x in range(12)]
    layer_mask['HEB'] = [1 for x in range(12)] 
    
    masked_layers = {}
    masked_layers['EE']  = [2, 4, 6, 8, 11, 13, 16, 19, 21, 24, 26, 28]
    masked_layers['HEF'] = [7, 9, 11]
    masked_layers['HEB'] = []
    
    mask_layers(layer_mask,masked_layers)
    
    process = cust_2023HGCalPandora_common(process)
    process = customise_me0(process)
    if hasattr(process,'reconstruction_step'):
        process = propagate_layerdropping(process,layer_mask)
    
    return process
