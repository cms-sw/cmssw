#!/usr/bin/env python
"""
_Cosmics_

Scenario supporting cosmic data taking

"""

import os
import sys

from Configuration.DataProcessing.Scenario import Scenario
import FWCore.ParameterSet.Config as cms
from Configuration.PyReleaseValidation.ConfigBuilder import ConfigBuilder
from Configuration.PyReleaseValidation.ConfigBuilder import Options
from Configuration.PyReleaseValidation.ConfigBuilder import defaultOptions
from Configuration.PyReleaseValidation.ConfigBuilder import installFilteredStream
    


class Cosmics(Scenario):
    """
    _Cosmics_

    Implement configuration building for data processing for cosmic
    data taking

    """


    def promptReco(self,process, recoOutputModule, aodOutputModule = None,
                   alcaOutputModule = None):
        """
        _promptReco_

        Cosmic data taking prompt reco

        """
        
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "cosmics"
        options.step = 'RAW2DIGI,RECO,DQM'
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        options.magField = 'AutoFromDBCurrent'
        
        cb = ConfigBuilder(options, process = process)
        cb.addStandardSequences()
        cb.addConditions()
        process.load(cb.EVTCONTDefaultCFF)
        recoOutputModule.eventContent = process.RECOEventContent
        if aodOutputModule != None:
            aodOutputModule.eventContent = process.AODEventContent
        if alcaOutputModule != None:
            print "ALCA Output needs to be implemented"
            #alcaOutputModule.eventContent = \
            #     process.ALCAIntermediateEventContent
        return process

    def expressProcessing(self):
        """
        _expressProcessing_

        Implement Cosmics Express processing

        Global Tag needed as arg??

        Based on/Edited from:
        
        ConfigBuilder.py
             step2
             -s RAW2DIGI,RECO:reconstructionCosmics,ALCA:MuAlCalIsolatedMu\
             +RpcCalHLT+TkAlCosmicsHLT+TkAlCosmics0T\
             +MuAlStandAloneCosmics+MuAlGlobalCosmics\
             +HcalCalHOCosmics
             --datatier RECO
             --eventcontent RECO
             --conditions FrontierConditions_GlobalTag,CRAFT_30X::All
             --scenario cosmics
             --no_exec
             --data        
        
        """

        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "cosmics"
        options.step = \
          """RAW2DIGI,RECO:reconstructionCosmics,ALCA:MuAlCalIsolatedMu+RpcCalHLT+TkAlCosmicsHLT+TkAlCosmics0T+MuAlStandAloneCosmics+MuAlGlobalCosmics+HcalCalHOCosmics"""
        options.isMC = False
        options.isData = True
        options.eventcontent = "RECO"
        options.relval = None
        
        process = cms.Process('EXPRESS')
        cb = ConfigBuilder(options, process = process)
        cb.addStandardSequences()
        cb.addConditions()
        process.load(cb.EVTCONTDefaultCFF)

        # import of standard configurations
        process.load('Configuration/StandardSequences/Services_cff')
        process.load('FWCore/MessageService/MessageLogger_cfi')
        process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
        process.load('Configuration/StandardSequences/GeometryIdeal_cff')
        process.load('Configuration/StandardSequences/MagneticField_38T_cff')
        process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
        process.load('Configuration/StandardSequences/ReconstructionCosmics_cff')
        process.load('Configuration/StandardSequences/AlCaRecoStreams_cff')
        process.load('Configuration/StandardSequences/EndOfProcess_cff')
        process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
        process.load('Configuration/EventContent/EventContentCosmics_cff')
        
        process.configurationMetadata = cms.untracked.PSet(
            version = cms.untracked.string('$Revision: 1.2 $'),
            annotation = cms.untracked.string('step2 nevts:1'),
            name = cms.untracked.string('PyReleaseValidation')
        )
        process.options = cms.untracked.PSet(
            Rethrow = cms.untracked.vstring('ProductNotFound')
        )
        # Input source
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring('step2_DIGI2RAW.root')
        )
        
        
        
        # Other statements
        # Path and EndPath definitions
        process.raw2digi_step = cms.Path(process.RawToDigi)
        process.reconstruction_step = cms.Path(process.reconstructionCosmics)
        process.pathALCARECOHcalCalHOCosmics = cms.Path(process.seqALCARECOHcalCalHOCosmics)
        process.pathALCARECOMuAlStandAloneCosmics = cms.Path(process.seqALCARECOMuAlStandAloneCosmics*process.ALCARECOMuAlStandAloneCosmicsDQM)
        process.pathALCARECOTkAlZMuMu = cms.Path(process.seqALCARECOTkAlZMuMu*process.ALCARECOTkAlZMuMuDQM)
        process.pathALCARECOTkAlCosmicsCTF0T = cms.Path(process.seqALCARECOTkAlCosmicsCTF0T*process.ALCARECOTkAlCosmicsCTF0TDQM)
        process.pathALCARECOMuAlBeamHalo = cms.Path(process.seqALCARECOMuAlBeamHalo*process.ALCARECOMuAlBeamHaloDQM)
        process.pathALCARECOTkAlCosmicsRS0THLT = cms.Path(process.seqALCARECOTkAlCosmicsRS0THLT*process.ALCARECOTkAlCosmicsRS0TDQM)
        process.pathALCARECOTkAlCosmicsCTF = cms.Path(process.seqALCARECOTkAlCosmicsCTF*process.ALCARECOTkAlCosmicsCTFDQM)
        process.pathALCARECOHcalCalIsoTrk = cms.Path(process.seqALCARECOHcalCalIsoTrk*process.ALCARECOHcalCalIsoTrackDQM)
        process.pathALCARECOHcalCalHO = cms.Path(process.seqALCARECOHcalCalHO*process.ALCARECOHcalCalHODQM)
        process.pathALCARECOTkAlCosmicsCTFHLT = cms.Path(process.seqALCARECOTkAlCosmicsCTFHLT*process.ALCARECOTkAlCosmicsCTFDQM)
        process.pathALCARECOTkAlCosmicsRS0T = cms.Path(process.seqALCARECOTkAlCosmicsRS0T*process.ALCARECOTkAlCosmicsRS0TDQM)
        process.pathALCARECOTkAlCosmicsCosmicTFHLT = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTFHLT*process.ALCARECOTkAlCosmicsCosmicTFDQM)
        process.pathALCARECOTkAlMuonIsolated = cms.Path(process.seqALCARECOTkAlMuonIsolated*process.ALCARECOTkAlMuonIsolatedDQM)
        process.pathALCARECOTkAlUpsilonMuMu = cms.Path(process.seqALCARECOTkAlUpsilonMuMu*process.ALCARECOTkAlUpsilonMuMuDQM)
        process.pathALCARECOHcalCalDijets = cms.Path(process.seqALCARECOHcalCalDijets*process.ALCARECOHcalCalDiJetsDQM)
        process.pathALCARECOMuAlZMuMu = cms.Path(process.seqALCARECOMuAlZMuMu*process.ALCARECOMuAlZMuMuDQM)
        process.pathALCARECOTkAlBeamHalo = cms.Path(process.seqALCARECOTkAlBeamHalo*process.ALCARECOTkAlBeamHaloDQM)
        process.pathALCARECOSiPixelLorentzAngle = cms.Path(process.seqALCARECOSiPixelLorentzAngle)
        process.pathALCARECOEcalCalElectron = cms.Path(process.seqALCARECOEcalCalElectron*process.ALCARECOEcalCalElectronCalibDQM)
        process.pathALCARECOTkAlCosmicsCTF0THLT = cms.Path(process.seqALCARECOTkAlCosmicsCTF0THLT*process.ALCARECOTkAlCosmicsCTF0TDQM)
        process.pathALCARECOMuAlCalIsolatedMu = cms.Path(process.seqALCARECOMuAlCalIsolatedMu*process.ALCARECOMuAlCalIsolatedMuDQM*process.ALCARECODTCalibrationDQM)
        process.pathALCARECOSiStripCalZeroBias = cms.Path(process.seqALCARECOSiStripCalZeroBias*process.ALCARECOSiStripCalZeroBiasDQM)
        process.pathALCARECOTkAlCosmicsRSHLT = cms.Path(process.seqALCARECOTkAlCosmicsRSHLT*process.ALCARECOTkAlCosmicsRSDQM)
        process.pathALCARECOSiStripCalMinBias = cms.Path(process.seqALCARECOSiStripCalMinBias)
        process.pathALCARECODQM = cms.Path(process.MEtoEDMConverter)
        process.pathALCARECOTkAlLAS = cms.Path(process.seqALCARECOTkAlLAS*process.ALCARECOTkAlLASDQM)
        process.pathALCARECOTkAlMinBias = cms.Path(process.seqALCARECOTkAlMinBias*process.ALCARECOTkAlMinBiasDQM)
        process.pathALCARECOTkAlCosmicsCosmicTF0T = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF0T*process.ALCARECOTkAlCosmicsCosmicTF0TDQM)
        process.pathALCARECOTkAlCosmicsRS = cms.Path(process.seqALCARECOTkAlCosmicsRS*process.ALCARECOTkAlCosmicsRSDQM)
        process.pathALCARECORpcCalHLT = cms.Path(process.seqALCARECORpcCalHLT)
        process.pathALCARECOHcalCalGammaJet = cms.Path(process.seqALCARECOHcalCalGammaJet)
        process.pathALCARECOMuAlBeamHaloOverlaps = cms.Path(process.seqALCARECOMuAlBeamHaloOverlaps*process.ALCARECOMuAlBeamHaloOverlapsDQM)
        process.pathALCARECOTkAlCosmicsCosmicTF0THLT = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF0THLT*process.ALCARECOTkAlCosmicsCosmicTF0TDQM)
        process.pathALCARECOHcalCalNoise = cms.Path(process.seqALCARECOHcalCalNoise)
        process.pathALCARECOMuAlOverlaps = cms.Path(process.seqALCARECOMuAlOverlaps*process.ALCARECOMuAlOverlapsDQM)
        process.pathALCARECOTkAlCosmicsCosmicTF = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF*process.ALCARECOTkAlCosmicsCosmicTFDQM)
        process.pathALCARECOMuAlGlobalCosmics = cms.Path(process.seqALCARECOMuAlGlobalCosmics*process.ALCARECOMuAlGlobalCosmicsDQM)
        process.pathALCARECOTkAlJpsiMuMu = cms.Path(process.seqALCARECOTkAlJpsiMuMu*process.ALCARECOTkAlJpsiMuMuDQM)
        process.endjob_step = cms.Path(process.endOfProcess)
        
        
        # Schedule definition
        process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.pathALCARECORpcCalHLT,process.pathALCARECOHcalCalHOCosmics,process.pathALCARECOMuAlCalIsolatedMu,process.pathALCARECOTkAlCosmicsCTFHLT,process.pathALCARECOTkAlCosmicsCosmicTFHLT,process.pathALCARECOTkAlCosmicsRSHLT,process.pathALCARECOTkAlCosmicsCTF0T,process.pathALCARECOTkAlCosmicsCosmicTF0T,process.pathALCARECOTkAlCosmicsRS0T,process.pathALCARECOMuAlGlobalCosmics,process.pathALCARECOMuAlStandAloneCosmics,process.endjob_step)
        
        
        process.write_Express_StreamExpress_RAW = cms.OutputModule(
            "PoolOutputModule", 
            fileName = cms.untracked.string('write_Express_StreamExpress_RAW.root'), 
            dataset = cms.untracked.PSet( 
            dataTier = cms.untracked.string('RAW'), 
            primaryDataset = cms.untracked.string('StreamExpress') 
            ), 
            compressionLevel = cms.untracked.int32(3), 
            outputCommands = cms.untracked.vstring(
            'drop *',  
            'keep  FEDRawDataCollection_rawDataCollector_*_*',  
            'keep  FEDRawDataCollection_source_*_*',  
            'keep *_gtDigis_*_*',  
            'keep *_l1GtRecord_*_*',  
            'keep *_l1GtObjectMap_*_*',  
            'keep *_l1extraParticles_*_*',  
            'drop *_hlt*_*_*',  
            'keep FEDRawDataCollection_rawDataCollector_*_*',  
            'keep edmTriggerResults_*_*_*',  
            'keep triggerTriggerEvent_*_*_*',  
            'keep *_hltGctDigis_*_*',  
            'keep *_hltGtDigis_*_*',  
            'keep *_hltL1extraParticles_*_*',  
            'keep *_hltL1GtObjectMap_*_*'), 
            fastCloning = cms.untracked.bool(False), 
            logicalFileName = cms.untracked.string('/store/whatever') 
            ) 
        
        return process
    

    def alcaReco(self,*skims):
        """
        _alcaReco_

        AlcaReco processing & skims for cosmics

        Based on:
        Revision: 1.120 
        ConfigBuilder.py 
          step3_V16
          -s ALCA:MuAlStandAloneCosmics+DQM
          --scenario cosmics
          --conditions FrontierConditions_GlobalTag,CRAFT_V16P::All
          --no_exec --data


        Expecting GlobalTag to be provided via API initially although
        this may not be the case

        """
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "cosmics"
        options.step = 'ALCA:MuAlStandAloneCosmics+DQM'
        options.isMC = False
        options.isData = True
        options.conditions = "FrontierConditions_GlobalTag,CRAFT_V16P::All" 
        options.beamspot = None
        options.eventcontent = None
        options.relval = None
        

        
        process = cms.Process('ALCA')
        cb = ConfigBuilder(options, process = process)
        cb.addStandardSequences()
        cb.addConditions()
        process.load(cb.EVTCONTDefaultCFF)
        # import of standard configurations
        process.load('Configuration/StandardSequences/Services_cff')
        process.load('FWCore/MessageService/MessageLogger_cfi')
        process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
        process.load('Configuration/StandardSequences/GeometryIdeal_cff')
        process.load('Configuration/StandardSequences/MagneticField_38T_cff')
        process.load('Configuration/StandardSequences/AlCaRecoStreams_cff')
        process.load('Configuration/StandardSequences/EndOfProcess_cff')
        process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
        process.load('Configuration/EventContent/EventContentCosmics_cff')
        
        process.configurationMetadata = cms.untracked.PSet(
            version = cms.untracked.string('$Revision: 1.2 $'),
            annotation = cms.untracked.string('step3_V16 nevts:1'),
            name = cms.untracked.string('PyReleaseValidation')
        )
        process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32(1)
        )
        process.options = cms.untracked.PSet(
            Rethrow = cms.untracked.vstring('ProductNotFound')
        )
        # Input source
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring('step3_V16_RECO.root')
        )
        
        # Additional output definition
        process.ALCARECOStreamMuAlStandAloneCosmics = cms.OutputModule("PoolOutputModule",
            SelectEvents = cms.untracked.PSet(
                SelectEvents = cms.vstring('pathALCARECOMuAlStandAloneCosmics')
            ),
            outputCommands = cms.untracked.vstring('drop *', 
                'keep *_ALCARECOMuAlStandAloneCosmics_*_*', 
                'keep *_muonCSCDigis_*_*', 
                'keep *_muonDTDigis_*_*', 
                'keep *_muonRPCDigis_*_*', 
                'keep *_dt1DRecHits_*_*', 
                'keep *_dt2DSegments_*_*', 
                'keep *_dt4DSegments_*_*', 
                'keep *_csc2DRecHits_*_*', 
                'keep *_cscSegments_*_*', 
                'keep *_rpcRecHits_*_*'),
            fileName = cms.untracked.string('ALCARECOMuAlStandAloneCosmics.root'),
            dataset = cms.untracked.PSet(
                filterName = cms.untracked.string('StreamALCARECOMuAlStandAloneCosmics'),
                dataTier = cms.untracked.string('ALCARECO')
            )
        )
        
        
        # Path and EndPath definitions
        process.pathALCARECOHcalCalHOCosmics = cms.Path(
            process.seqALCARECOHcalCalHOCosmics)
        process.pathALCARECOMuAlStandAloneCosmics = cms.Path(
            process.seqALCARECOMuAlStandAloneCosmics*process.ALCARECOMuAlStandAloneCosmicsDQM)
        process.pathALCARECOTkAlZMuMu = cms.Path(process.seqALCARECOTkAlZMuMu*process.ALCARECOTkAlZMuMuDQM)
        process.pathALCARECOTkAlCosmicsCTF0T = cms.Path(process.seqALCARECOTkAlCosmicsCTF0T*process.ALCARECOTkAlCosmicsCTF0TDQM)
        process.pathALCARECOMuAlBeamHalo = cms.Path(process.seqALCARECOMuAlBeamHalo*process.ALCARECOMuAlBeamHaloDQM)
        process.pathALCARECOTkAlCosmicsRS0THLT = cms.Path(process.seqALCARECOTkAlCosmicsRS0THLT*process.ALCARECOTkAlCosmicsRS0TDQM)
        process.pathALCARECOTkAlCosmicsCTF = cms.Path(process.seqALCARECOTkAlCosmicsCTF*process.ALCARECOTkAlCosmicsCTFDQM)
        process.pathALCARECOHcalCalIsoTrk = cms.Path(process.seqALCARECOHcalCalIsoTrk*process.ALCARECOHcalCalIsoTrackDQM)
        process.pathALCARECOHcalCalHO = cms.Path(process.seqALCARECOHcalCalHO*process.ALCARECOHcalCalHODQM)
        process.pathALCARECOTkAlCosmicsCTFHLT = cms.Path(process.seqALCARECOTkAlCosmicsCTFHLT*process.ALCARECOTkAlCosmicsCTFDQM)
        process.pathALCARECOTkAlCosmicsRS0T = cms.Path(process.seqALCARECOTkAlCosmicsRS0T*process.ALCARECOTkAlCosmicsRS0TDQM)
        process.pathALCARECOTkAlCosmicsCosmicTFHLT = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTFHLT*process.ALCARECOTkAlCosmicsCosmicTFDQM)
        process.pathALCARECOTkAlMuonIsolated = cms.Path(process.seqALCARECOTkAlMuonIsolated*process.ALCARECOTkAlMuonIsolatedDQM)
        process.pathALCARECOTkAlUpsilonMuMu = cms.Path(process.seqALCARECOTkAlUpsilonMuMu*process.ALCARECOTkAlUpsilonMuMuDQM)
        process.pathALCARECOHcalCalDijets = cms.Path(process.seqALCARECOHcalCalDijets*process.ALCARECOHcalCalDiJetsDQM)
        process.pathALCARECOMuAlZMuMu = cms.Path(process.seqALCARECOMuAlZMuMu*process.ALCARECOMuAlZMuMuDQM)
        process.pathALCARECOTkAlBeamHalo = cms.Path(process.seqALCARECOTkAlBeamHalo*process.ALCARECOTkAlBeamHaloDQM)
        process.pathALCARECOSiPixelLorentzAngle = cms.Path(process.seqALCARECOSiPixelLorentzAngle)
        process.pathALCARECOEcalCalElectron = cms.Path(process.seqALCARECOEcalCalElectron*process.ALCARECOEcalCalElectronCalibDQM)
        process.pathALCARECOTkAlCosmicsCTF0THLT = cms.Path(process.seqALCARECOTkAlCosmicsCTF0THLT*process.ALCARECOTkAlCosmicsCTF0TDQM)
        process.pathALCARECOMuAlCalIsolatedMu = cms.Path(process.seqALCARECOMuAlCalIsolatedMu*process.ALCARECOMuAlCalIsolatedMuDQM*process.ALCARECODTCalibrationDQM)
        process.pathALCARECOSiStripCalZeroBias = cms.Path(process.seqALCARECOSiStripCalZeroBias*process.ALCARECOSiStripCalZeroBiasDQM)
        process.pathALCARECOTkAlCosmicsRSHLT = cms.Path(process.seqALCARECOTkAlCosmicsRSHLT*process.ALCARECOTkAlCosmicsRSDQM)
        process.pathALCARECOSiStripCalMinBias = cms.Path(process.seqALCARECOSiStripCalMinBias)
        process.pathALCARECODQM = cms.Path(process.MEtoEDMConverter)
        process.pathALCARECOTkAlLAS = cms.Path(process.seqALCARECOTkAlLAS*process.ALCARECOTkAlLASDQM)
        process.pathALCARECOTkAlMinBias = cms.Path(process.seqALCARECOTkAlMinBias*process.ALCARECOTkAlMinBiasDQM)
        process.pathALCARECOTkAlCosmicsCosmicTF0T = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF0T*process.ALCARECOTkAlCosmicsCosmicTF0TDQM)
        process.pathALCARECOTkAlCosmicsRS = cms.Path(process.seqALCARECOTkAlCosmicsRS*process.ALCARECOTkAlCosmicsRSDQM)
        process.pathALCARECORpcCalHLT = cms.Path(process.seqALCARECORpcCalHLT)
        process.pathALCARECOHcalCalGammaJet = cms.Path(process.seqALCARECOHcalCalGammaJet)
        process.pathALCARECOMuAlBeamHaloOverlaps = cms.Path(process.seqALCARECOMuAlBeamHaloOverlaps*process.ALCARECOMuAlBeamHaloOverlapsDQM)
        process.pathALCARECOTkAlCosmicsCosmicTF0THLT = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF0THLT*process.ALCARECOTkAlCosmicsCosmicTF0TDQM)
        process.pathALCARECOHcalCalNoise = cms.Path(process.seqALCARECOHcalCalNoise)
        process.pathALCARECOMuAlOverlaps = cms.Path(process.seqALCARECOMuAlOverlaps*process.ALCARECOMuAlOverlapsDQM)
        process.pathALCARECOTkAlCosmicsCosmicTF = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF*process.ALCARECOTkAlCosmicsCosmicTFDQM)
        process.pathALCARECOMuAlGlobalCosmics = cms.Path(process.seqALCARECOMuAlGlobalCosmics*process.ALCARECOMuAlGlobalCosmicsDQM)
        process.pathALCARECOTkAlJpsiMuMu = cms.Path(process.seqALCARECOTkAlJpsiMuMu*process.ALCARECOTkAlJpsiMuMuDQM)
        process.endjob_step = cms.Path(process.endOfProcess)
        process.ALCARECOStreamMuAlStandAloneCosmicsOutPath = cms.EndPath(process.ALCARECOStreamMuAlStandAloneCosmics)
        
        # Schedule definition
        process.schedule = cms.Schedule(process.pathALCARECODQM,process.pathALCARECOMuAlStandAloneCosmics,process.endjob_step,process.ALCARECOStreamMuAlStandAloneCosmicsOutPath)
        

        #  //
        # // Verify and Edit the list of skims to be written out
        #//  by this job
        availableStreams = process.outputModules_().keys()

        #  //
        # // First up: Verify skims are available by output module name
        #//
        for skim in skims:
            if skim not in availableStreams:
                msg = "Skim named: %s not available " % skim
                msg += "in Alca Reco Config:\n"
                msg += "Known Skims: %s\n" % availableStreams
                raise RuntimeError, msg

        #  //
        # // Prune any undesired skims
        #//
        for availSkim in availableStreams:
            if availSkim not in skims:
                self.dropOutputModule(process, availSkim)

        return process
                

        

        


    def dqmHarvesting(self, datasetName, runNumber,  globalTag,
                      *inputFiles, **options):
        """
        _dqmHarvesting_

        Cosmic data taking DQM Harvesting

        """

        #
        # 
        #
        #sys.argv=["cmsDriver.py",
        #          "step3",
        #          "-s", "HARVESTING:dqmHarvesting",
        #          "--conditions", "FrontierConditions_GlobalTag,CRAFT_30X::All",
        #          "--filein", "file:step2_RAW2DIGI_RECO_DQM.root",
        #          "--data",
        #          "--scenario=cosmics"]
        #  //
        # // should be able to translate these into options settngs
        #//
        options = defaultOptions
        options.scenario = "cosmics"
        options.step = "HARVESTING:dqmHarvesting"
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        options.name = "EDMtoMEConvert"
        options.number = -1
        options.conditions = "FrontierConditions_GlobalTag,CRAFT_30X::All"
        options.filein = list(inputFiles)
        options.arguments = ""
        options.evt_type = ""
        options.customisation_file = ""
        configBuilder = ConfigBuilder(options)


        configBuilder.prepare()
        process = configBuilder.process

        #
        # customise process for particular job
        #
        process.source.fileNames = cms.untracked(cms.vstring())
        #for fileName in inputFiles:
        #    process.source.fileNames.append(fileName)
        
        process.maxEvents.input = -1
        process.dqmSaver.workflow = datasetName
        
        return process
