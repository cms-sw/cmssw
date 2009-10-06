#!/usr/bin/env python
"""
_relvalmc_

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
    


class relvalmc(Scenario):
    """
    _relvalmc_

    Implement configuration building for data processing for cosmic
    data taking

    """


    def promptReco(self, globalTag, writeTiers = ['RECO']):
        """
        _promptReco_

        Cosmic data taking prompt reco

        """
        
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "cosmics"
        options.step = 'RAW2DIGI,RECO,VALIDATION,DQM'
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        options.magField = 'AutoFromDBCurrent'
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag

        
        
        process = cms.Process('RECO')
        cb = ConfigBuilder(options, process = process)
        cb.addStandardSequences()
        cb.addConditions()
        process.load(cb.EVTCONTDefaultCFF)

        # Input source
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring()
        )


        if "RECO" in writeTiers:
            process.writeRECO = cms.OutputModule(
                "PoolOutputModule", 
                fileName = cms.untracked.string('writeRECO.root'), 
                dataset = cms.untracked.PSet( 
                dataTier = cms.untracked.string('RECO'), 
                ), 
                )
            
            process.writeRECO.eventContent = process.RECOEventContent
        if "AOD" in writeTiers:
            process.writeAOD = cms.OutputModule(
                "PoolOutputModule", 
                fileName = cms.untracked.string('writeAOD.root'), 
                dataset = cms.untracked.PSet( 
                dataTier = cms.untracked.string('AOD'), 
                ), 
            )

            process.writeAOD.eventContent = process.AODEventContent
        if "ALCA" in writeTiers:
            process.writeALCA = cms.OutputModule(
                "PoolOutputModule", 
                fileName = cms.untracked.string('writeALCA.root'), 
                dataset = cms.untracked.PSet( 
                dataTier = cms.untracked.string('ALCA'), 
                ), 
            )
            process.writeALCA.eventContent = process.ALCAEventContent

        return process

    def expressProcessing(self, globalTag,  writeTiers = [],
                          datasets = [], alcaDataset = None):
        """
        _expressProcessing_

        Implement relvalmc Express processing

        Based on/Edited from:
        
        ConfigBuilder.py
             step2
             -s RAW2DIGI,RECO:reconstructionrelvalmc,ALCA:MuAlCalIsolatedMu\
             +RpcCalHLT+TkAlrelvalmcHLT+TkAlrelvalmc0T\
             +MuAlStandAlonerelvalmc+MuAlGlobalrelvalmc\
             +HcalCalHOrelvalmc
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
          """RAW2DIGI,RECO:reconstructionrelvalmc,ALCA:MuAlCalIsolatedMu+RpcCalHLT+TkAlrelvalmcHLT+TkAlrelvalmc0T+MuAlStandAlonerelvalmc+MuAlGlobalrelvalmc+HcalCalHOrelvalmc"""
        options.isMC = False
        options.isData = True
        options.eventcontent = "RECO"
        options.relval = None
        options.beamspot = None
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
        
        process = cms.Process('EXPRESS')
        cb = ConfigBuilder(options, process = process)
        cb.addStandardSequences()
        cb.addConditions()
        process.load(cb.EVTCONTDefaultCFF)

        #  //
        # // Install the OutputModules for everything but ALCA
        #//
        self.addExpressOutputModules(process, writeTiers, datasets)
        
        #  //
        # // TODO: Install Alca output
        #//
        
        #  //
        # // everything below here could be complete gibberish
        #//
        
        # import of standard configurations
        process.load('Configuration/StandardSequences/Services_cff')
        process.load('FWCore/MessageService/MessageLogger_cfi')
        process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
        process.load('Configuration/StandardSequences/GeometryIdeal_cff')
        process.load('Configuration/StandardSequences/MagneticField_38T_cff')
        process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
        process.load('Configuration/StandardSequences/Reconstructionrelvalmc_cff')
        process.load('Configuration/StandardSequences/AlCaRecoStreams_cff')
        process.load('Configuration/StandardSequences/EndOfProcess_cff')
        process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
        process.load('Configuration/EventContent/EventContentrelvalmc_cff')
        
        process.configurationMetadata = cms.untracked.PSet(
            version = cms.untracked.string('$Revision: 1.1 $'),
            annotation = cms.untracked.string('step2 nevts:1'),
            name = cms.untracked.string('PyReleaseValidation')
        )
        process.options = cms.untracked.PSet(
            Rethrow = cms.untracked.vstring('ProductNotFound')
        )
        # Input source
        process.source = cms.Source(
            "NewEventStreamFileReader",
            fileNames = cms.untracked.vstring()
        )
        
        
        
        # Other statements
        # Path and EndPath definitions
        process.raw2digi_step = cms.Path(process.RawToDigi)
        process.reconstruction_step = cms.Path(process.reconstructionrelvalmc)
        process.pathALCARECOHcalCalHOrelvalmc = cms.Path(process.seqALCARECOHcalCalHOrelvalmc)
        process.pathALCARECOMuAlStandAlonerelvalmc = cms.Path(process.seqALCARECOMuAlStandAlonerelvalmc*process.ALCARECOMuAlStandAlonerelvalmcDQM)
        process.pathALCARECOTkAlZMuMu = cms.Path(process.seqALCARECOTkAlZMuMu*process.ALCARECOTkAlZMuMuDQM)
        process.pathALCARECOTkAlrelvalmcCTF0T = cms.Path(process.seqALCARECOTkAlrelvalmcCTF0T*process.ALCARECOTkAlrelvalmcCTF0TDQM)
        process.pathALCARECOMuAlBeamHalo = cms.Path(process.seqALCARECOMuAlBeamHalo*process.ALCARECOMuAlBeamHaloDQM)
        process.pathALCARECOTkAlrelvalmcRS0THLT = cms.Path(process.seqALCARECOTkAlrelvalmcRS0THLT*process.ALCARECOTkAlrelvalmcRS0TDQM)
        process.pathALCARECOTkAlrelvalmcCTF = cms.Path(process.seqALCARECOTkAlrelvalmcCTF*process.ALCARECOTkAlrelvalmcCTFDQM)
        process.pathALCARECOHcalCalIsoTrk = cms.Path(process.seqALCARECOHcalCalIsoTrk*process.ALCARECOHcalCalIsoTrackDQM)
        process.pathALCARECOHcalCalHO = cms.Path(process.seqALCARECOHcalCalHO*process.ALCARECOHcalCalHODQM)
        process.pathALCARECOTkAlrelvalmcCTFHLT = cms.Path(process.seqALCARECOTkAlrelvalmcCTFHLT*process.ALCARECOTkAlrelvalmcCTFDQM)
        process.pathALCARECOTkAlrelvalmcRS0T = cms.Path(process.seqALCARECOTkAlrelvalmcRS0T*process.ALCARECOTkAlrelvalmcRS0TDQM)
        process.pathALCARECOTkAlrelvalmcCosmicTFHLT = cms.Path(process.seqALCARECOTkAlrelvalmcCosmicTFHLT*process.ALCARECOTkAlrelvalmcCosmicTFDQM)
        process.pathALCARECOTkAlMuonIsolated = cms.Path(process.seqALCARECOTkAlMuonIsolated*process.ALCARECOTkAlMuonIsolatedDQM)
        process.pathALCARECOTkAlUpsilonMuMu = cms.Path(process.seqALCARECOTkAlUpsilonMuMu*process.ALCARECOTkAlUpsilonMuMuDQM)
        process.pathALCARECOHcalCalDijets = cms.Path(process.seqALCARECOHcalCalDijets*process.ALCARECOHcalCalDiJetsDQM)
        process.pathALCARECOMuAlZMuMu = cms.Path(process.seqALCARECOMuAlZMuMu*process.ALCARECOMuAlZMuMuDQM)
        process.pathALCARECOTkAlBeamHalo = cms.Path(process.seqALCARECOTkAlBeamHalo*process.ALCARECOTkAlBeamHaloDQM)
        process.pathALCARECOSiPixelLorentzAngle = cms.Path(process.seqALCARECOSiPixelLorentzAngle)
        process.pathALCARECOEcalCalElectron = cms.Path(process.seqALCARECOEcalCalElectron*process.ALCARECOEcalCalElectronCalibDQM)
        process.pathALCARECOTkAlrelvalmcCTF0THLT = cms.Path(process.seqALCARECOTkAlrelvalmcCTF0THLT*process.ALCARECOTkAlrelvalmcCTF0TDQM)
        process.pathALCARECOMuAlCalIsolatedMu = cms.Path(process.seqALCARECOMuAlCalIsolatedMu*process.ALCARECOMuAlCalIsolatedMuDQM*process.ALCARECODTCalibrationDQM)
        process.pathALCARECOSiStripCalZeroBias = cms.Path(process.seqALCARECOSiStripCalZeroBias*process.ALCARECOSiStripCalZeroBiasDQM)
        process.pathALCARECOTkAlrelvalmcRSHLT = cms.Path(process.seqALCARECOTkAlrelvalmcRSHLT*process.ALCARECOTkAlrelvalmcRSDQM)
        process.pathALCARECOSiStripCalMinBias = cms.Path(process.seqALCARECOSiStripCalMinBias)
        process.pathALCARECODQM = cms.Path(process.MEtoEDMConverter)
        process.pathALCARECOTkAlLAS = cms.Path(process.seqALCARECOTkAlLAS*process.ALCARECOTkAlLASDQM)
        process.pathALCARECOTkAlMinBias = cms.Path(process.seqALCARECOTkAlMinBias*process.ALCARECOTkAlMinBiasDQM)
        process.pathALCARECOTkAlrelvalmcCosmicTF0T = cms.Path(process.seqALCARECOTkAlrelvalmcCosmicTF0T*process.ALCARECOTkAlrelvalmcCosmicTF0TDQM)
        process.pathALCARECOTkAlrelvalmcRS = cms.Path(process.seqALCARECOTkAlrelvalmcRS*process.ALCARECOTkAlrelvalmcRSDQM)
        process.pathALCARECORpcCalHLT = cms.Path(process.seqALCARECORpcCalHLT)
        process.pathALCARECOHcalCalGammaJet = cms.Path(process.seqALCARECOHcalCalGammaJet)
        process.pathALCARECOMuAlBeamHaloOverlaps = cms.Path(process.seqALCARECOMuAlBeamHaloOverlaps*process.ALCARECOMuAlBeamHaloOverlapsDQM)
        process.pathALCARECOTkAlrelvalmcCosmicTF0THLT = cms.Path(process.seqALCARECOTkAlrelvalmcCosmicTF0THLT*process.ALCARECOTkAlrelvalmcCosmicTF0TDQM)
        process.pathALCARECOHcalCalNoise = cms.Path(process.seqALCARECOHcalCalNoise)
        process.pathALCARECOMuAlOverlaps = cms.Path(process.seqALCARECOMuAlOverlaps*process.ALCARECOMuAlOverlapsDQM)
        process.pathALCARECOTkAlrelvalmcCosmicTF = cms.Path(process.seqALCARECOTkAlrelvalmcCosmicTF*process.ALCARECOTkAlrelvalmcCosmicTFDQM)
        process.pathALCARECOMuAlGlobalrelvalmc = cms.Path(process.seqALCARECOMuAlGlobalrelvalmc*process.ALCARECOMuAlGlobalrelvalmcDQM)
        process.pathALCARECOTkAlJpsiMuMu = cms.Path(process.seqALCARECOTkAlJpsiMuMu*process.ALCARECOTkAlJpsiMuMuDQM)
        process.endjob_step = cms.Path(process.endOfProcess)
        
        
        # Schedule definition
        process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.pathALCARECORpcCalHLT,process.pathALCARECOHcalCalHOrelvalmc,process.pathALCARECOMuAlCalIsolatedMu,process.pathALCARECOTkAlrelvalmcCTFHLT,process.pathALCARECOTkAlrelvalmcCosmicTFHLT,process.pathALCARECOTkAlrelvalmcRSHLT,process.pathALCARECOTkAlrelvalmcCTF0T,process.pathALCARECOTkAlrelvalmcCosmicTF0T,process.pathALCARECOTkAlrelvalmcRS0T,process.pathALCARECOMuAlGlobalrelvalmc,process.pathALCARECOMuAlStandAlonerelvalmc,process.endjob_step)
        
        
        ##process.write_Express_StreamExpress_RAW = cms.OutputModule(
##            "PoolOutputModule", 
##            fileName = cms.untracked.string('write_Express_StreamExpress_RAW.root'), 
##            dataset = cms.untracked.PSet( 
##            dataTier = cms.untracked.string('RAW'), 
##            primaryDataset = cms.untracked.string('StreamExpress') 
##            ), 
##            compressionLevel = cms.untracked.int32(3), 
##            outputCommands = cms.untracked.vstring(
##            'drop *',  
##            'keep  FEDRawDataCollection_rawDataCollector_*_*',  
##            'keep  FEDRawDataCollection_source_*_*',  
##            'keep *_gtDigis_*_*',  
##            'keep *_l1GtRecord_*_*',  
##            'keep *_l1GtObjectMap_*_*',  
##            'keep *_l1extraParticles_*_*',  
##            'drop *_hlt*_*_*',  
##            'keep FEDRawDataCollection_rawDataCollector_*_*',  
##            'keep edmTriggerResults_*_*_*',  
##            'keep triggerTriggerEvent_*_*_*',  
##            'keep *_hltGctDigis_*_*',  
##            'keep *_hltGtDigis_*_*',  
##            'keep *_hltL1extraParticles_*_*',  
##            'keep *_hltL1GtObjectMap_*_*'), 
##            fastCloning = cms.untracked.bool(False), 
##            logicalFileName = cms.untracked.string('/store/whatever') 
##            ) 
        
        return process
    

    def alcaReco(self, *skims):
        """
        _alcaReco_

        AlcaReco processing & skims for cosmics

        Based on:
        Revision: 1.120 
        ConfigBuilder.py 
          step3_V16
          -s ALCA:MuAlStandAlonerelvalmc+DQM
          --scenario cosmics
          --conditions FrontierConditions_GlobalTag,CRAFT_V16P::All
          --no_exec --data


        Expecting GlobalTag to be provided via API initially although
        this may not be the case

        """
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "cosmics"
        options.step = 'ALCA:MuAlStandAlonerelvalmc+DQM'
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
        process.load('Configuration/EventContent/EventContentrelvalmc_cff')
        
        process.configurationMetadata = cms.untracked.PSet(
            version = cms.untracked.string('$Revision: 1.1 $'),
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
        process.source = cms.Source(
            "PoolSource",
            fileNames = cms.untracked.vstring()
        )
        
        # Additional output definition
        process.ALCARECOStreamMuAlStandAlonerelvalmc = cms.OutputModule("PoolOutputModule",
            SelectEvents = cms.untracked.PSet(
                SelectEvents = cms.vstring('pathALCARECOMuAlStandAlonerelvalmc')
            ),
            outputCommands = cms.untracked.vstring('drop *', 
                'keep *_ALCARECOMuAlStandAlonerelvalmc_*_*', 
                'keep *_muonCSCDigis_*_*', 
                'keep *_muonDTDigis_*_*', 
                'keep *_muonRPCDigis_*_*', 
                'keep *_dt1DRecHits_*_*', 
                'keep *_dt2DSegments_*_*', 
                'keep *_dt4DSegments_*_*', 
                'keep *_csc2DRecHits_*_*', 
                'keep *_cscSegments_*_*', 
                'keep *_rpcRecHits_*_*'),
            fileName = cms.untracked.string('ALCARECOMuAlStandAlonerelvalmc.root'),
            dataset = cms.untracked.PSet(
                filterName = cms.untracked.string('StreamALCARECOMuAlStandAlonerelvalmc'),
                dataTier = cms.untracked.string('ALCARECO')
            )
        )
        
        
        # Path and EndPath definitions
        process.pathALCARECOHcalCalHOrelvalmc = cms.Path(
            process.seqALCARECOHcalCalHOrelvalmc)
        process.pathALCARECOMuAlStandAlonerelvalmc = cms.Path(
            process.seqALCARECOMuAlStandAlonerelvalmc*process.ALCARECOMuAlStandAlonerelvalmcDQM)
        process.pathALCARECOTkAlZMuMu = cms.Path(process.seqALCARECOTkAlZMuMu*process.ALCARECOTkAlZMuMuDQM)
        process.pathALCARECOTkAlrelvalmcCTF0T = cms.Path(process.seqALCARECOTkAlrelvalmcCTF0T*process.ALCARECOTkAlrelvalmcCTF0TDQM)
        process.pathALCARECOMuAlBeamHalo = cms.Path(process.seqALCARECOMuAlBeamHalo*process.ALCARECOMuAlBeamHaloDQM)
        process.pathALCARECOTkAlrelvalmcRS0THLT = cms.Path(process.seqALCARECOTkAlrelvalmcRS0THLT*process.ALCARECOTkAlrelvalmcRS0TDQM)
        process.pathALCARECOTkAlrelvalmcCTF = cms.Path(process.seqALCARECOTkAlrelvalmcCTF*process.ALCARECOTkAlrelvalmcCTFDQM)
        process.pathALCARECOHcalCalIsoTrk = cms.Path(process.seqALCARECOHcalCalIsoTrk*process.ALCARECOHcalCalIsoTrackDQM)
        process.pathALCARECOHcalCalHO = cms.Path(process.seqALCARECOHcalCalHO*process.ALCARECOHcalCalHODQM)
        process.pathALCARECOTkAlrelvalmcCTFHLT = cms.Path(process.seqALCARECOTkAlrelvalmcCTFHLT*process.ALCARECOTkAlrelvalmcCTFDQM)
        process.pathALCARECOTkAlrelvalmcRS0T = cms.Path(process.seqALCARECOTkAlrelvalmcRS0T*process.ALCARECOTkAlrelvalmcRS0TDQM)
        process.pathALCARECOTkAlrelvalmcCosmicTFHLT = cms.Path(process.seqALCARECOTkAlrelvalmcCosmicTFHLT*process.ALCARECOTkAlrelvalmcCosmicTFDQM)
        process.pathALCARECOTkAlMuonIsolated = cms.Path(process.seqALCARECOTkAlMuonIsolated*process.ALCARECOTkAlMuonIsolatedDQM)
        process.pathALCARECOTkAlUpsilonMuMu = cms.Path(process.seqALCARECOTkAlUpsilonMuMu*process.ALCARECOTkAlUpsilonMuMuDQM)
        process.pathALCARECOHcalCalDijets = cms.Path(process.seqALCARECOHcalCalDijets*process.ALCARECOHcalCalDiJetsDQM)
        process.pathALCARECOMuAlZMuMu = cms.Path(process.seqALCARECOMuAlZMuMu*process.ALCARECOMuAlZMuMuDQM)
        process.pathALCARECOTkAlBeamHalo = cms.Path(process.seqALCARECOTkAlBeamHalo*process.ALCARECOTkAlBeamHaloDQM)
        process.pathALCARECOSiPixelLorentzAngle = cms.Path(process.seqALCARECOSiPixelLorentzAngle)
        process.pathALCARECOEcalCalElectron = cms.Path(process.seqALCARECOEcalCalElectron*process.ALCARECOEcalCalElectronCalibDQM)
        process.pathALCARECOTkAlrelvalmcCTF0THLT = cms.Path(process.seqALCARECOTkAlrelvalmcCTF0THLT*process.ALCARECOTkAlrelvalmcCTF0TDQM)
        process.pathALCARECOMuAlCalIsolatedMu = cms.Path(process.seqALCARECOMuAlCalIsolatedMu*process.ALCARECOMuAlCalIsolatedMuDQM*process.ALCARECODTCalibrationDQM)
        process.pathALCARECOSiStripCalZeroBias = cms.Path(process.seqALCARECOSiStripCalZeroBias*process.ALCARECOSiStripCalZeroBiasDQM)
        process.pathALCARECOTkAlrelvalmcRSHLT = cms.Path(process.seqALCARECOTkAlrelvalmcRSHLT*process.ALCARECOTkAlrelvalmcRSDQM)
        process.pathALCARECOSiStripCalMinBias = cms.Path(process.seqALCARECOSiStripCalMinBias)
        process.pathALCARECODQM = cms.Path(process.MEtoEDMConverter)
        process.pathALCARECOTkAlLAS = cms.Path(process.seqALCARECOTkAlLAS*process.ALCARECOTkAlLASDQM)
        process.pathALCARECOTkAlMinBias = cms.Path(process.seqALCARECOTkAlMinBias*process.ALCARECOTkAlMinBiasDQM)
        process.pathALCARECOTkAlrelvalmcCosmicTF0T = cms.Path(process.seqALCARECOTkAlrelvalmcCosmicTF0T*process.ALCARECOTkAlrelvalmcCosmicTF0TDQM)
        process.pathALCARECOTkAlrelvalmcRS = cms.Path(process.seqALCARECOTkAlrelvalmcRS*process.ALCARECOTkAlrelvalmcRSDQM)
        process.pathALCARECORpcCalHLT = cms.Path(process.seqALCARECORpcCalHLT)
        process.pathALCARECOHcalCalGammaJet = cms.Path(process.seqALCARECOHcalCalGammaJet)
        process.pathALCARECOMuAlBeamHaloOverlaps = cms.Path(process.seqALCARECOMuAlBeamHaloOverlaps*process.ALCARECOMuAlBeamHaloOverlapsDQM)
        process.pathALCARECOTkAlrelvalmcCosmicTF0THLT = cms.Path(process.seqALCARECOTkAlrelvalmcCosmicTF0THLT*process.ALCARECOTkAlrelvalmcCosmicTF0TDQM)
        process.pathALCARECOHcalCalNoise = cms.Path(process.seqALCARECOHcalCalNoise)
        process.pathALCARECOMuAlOverlaps = cms.Path(process.seqALCARECOMuAlOverlaps*process.ALCARECOMuAlOverlapsDQM)
        process.pathALCARECOTkAlrelvalmcCosmicTF = cms.Path(process.seqALCARECOTkAlrelvalmcCosmicTF*process.ALCARECOTkAlrelvalmcCosmicTFDQM)
        process.pathALCARECOMuAlGlobalrelvalmc = cms.Path(process.seqALCARECOMuAlGlobalrelvalmc*process.ALCARECOMuAlGlobalrelvalmcDQM)
        process.pathALCARECOTkAlJpsiMuMu = cms.Path(process.seqALCARECOTkAlJpsiMuMu*process.ALCARECOTkAlJpsiMuMuDQM)
        process.endjob_step = cms.Path(process.endOfProcess)
        process.ALCARECOStreamMuAlStandAlonerelvalmcOutPath = cms.EndPath(process.ALCARECOStreamMuAlStandAlonerelvalmc)
        
        # Schedule definition
        process.schedule = cms.Schedule(process.pathALCARECODQM,process.pathALCARECOMuAlStandAlonerelvalmc,process.endjob_step,process.ALCARECOStreamMuAlStandAlonerelvalmcOutPath)
        

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
                

        

        


    def dqmHarvesting(self, datasetName, runNumber,  globalTag, **options):
        """
        _dqmHarvesting_

        Cosmic data taking DQM Harvesting

        """
        options = defaultOptions
        options.scenario = "cosmics"
        options.step = "HARVESTING:validationHarvesting+dqmHarvesting"
        options.isMC = True
        options.isData = False
        options.beamspot = None
        options.eventcontent = None
        options.name = "EDMtoMEConvert"
        options.number = -1
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
        options.arguments = ""
        options.evt_type = ""
        options.filein = []
        options.gflash = False
	options.himix = False
        options.customisation_file = ""

 
        process = cms.Process("HARVESTING")
        process.source = cms.Source("PoolSource")
        configBuilder = ConfigBuilder(options, process = process)
        configBuilder.prepare()

        #
        # customise process for particular job
        #
        process.source.processingMode = cms.untracked.string('RunsAndLumis')
        process.source.fileNames = cms.untracked(cms.vstring())
        process.maxEvents.input = -1
        process.dqmSaver.workflow = datasetName
        
        return process
