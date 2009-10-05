#!/usr/bin/env python
"""
_RelValMC_

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
    


class RelValMC(Scenario):
    """
    _RelValMC_

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
        options.step = 'RAW2DIGI,RECO,DQM'
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

        Implement RelValMC Express processing

        Based on/Edited from:
        
        ConfigBuilder.py
             step2
             -s RAW2DIGI,RECO:reconstructionRelValMC,ALCA:MuAlCalIsolatedMu\
             +RpcCalHLT+TkAlRelValMCHLT+TkAlRelValMC0T\
             +MuAlStandAloneRelValMC+MuAlGlobalRelValMC\
             +HcalCalHORelValMC
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
          """RAW2DIGI,RECO:reconstructionRelValMC,ALCA:MuAlCalIsolatedMu+RpcCalHLT+TkAlRelValMCHLT+TkAlRelValMC0T+MuAlStandAloneRelValMC+MuAlGlobalRelValMC+HcalCalHORelValMC"""
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
        process.load('Configuration/StandardSequences/ReconstructionRelValMC_cff')
        process.load('Configuration/StandardSequences/AlCaRecoStreams_cff')
        process.load('Configuration/StandardSequences/EndOfProcess_cff')
        process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
        process.load('Configuration/EventContent/EventContentRelValMC_cff')
        
        process.configurationMetadata = cms.untracked.PSet(
            version = cms.untracked.string('$Revision: 1.11 $'),
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
        process.reconstruction_step = cms.Path(process.reconstructionRelValMC)
        process.pathALCARECOHcalCalHORelValMC = cms.Path(process.seqALCARECOHcalCalHORelValMC)
        process.pathALCARECOMuAlStandAloneRelValMC = cms.Path(process.seqALCARECOMuAlStandAloneRelValMC*process.ALCARECOMuAlStandAloneRelValMCDQM)
        process.pathALCARECOTkAlZMuMu = cms.Path(process.seqALCARECOTkAlZMuMu*process.ALCARECOTkAlZMuMuDQM)
        process.pathALCARECOTkAlRelValMCCTF0T = cms.Path(process.seqALCARECOTkAlRelValMCCTF0T*process.ALCARECOTkAlRelValMCCTF0TDQM)
        process.pathALCARECOMuAlBeamHalo = cms.Path(process.seqALCARECOMuAlBeamHalo*process.ALCARECOMuAlBeamHaloDQM)
        process.pathALCARECOTkAlRelValMCRS0THLT = cms.Path(process.seqALCARECOTkAlRelValMCRS0THLT*process.ALCARECOTkAlRelValMCRS0TDQM)
        process.pathALCARECOTkAlRelValMCCTF = cms.Path(process.seqALCARECOTkAlRelValMCCTF*process.ALCARECOTkAlRelValMCCTFDQM)
        process.pathALCARECOHcalCalIsoTrk = cms.Path(process.seqALCARECOHcalCalIsoTrk*process.ALCARECOHcalCalIsoTrackDQM)
        process.pathALCARECOHcalCalHO = cms.Path(process.seqALCARECOHcalCalHO*process.ALCARECOHcalCalHODQM)
        process.pathALCARECOTkAlRelValMCCTFHLT = cms.Path(process.seqALCARECOTkAlRelValMCCTFHLT*process.ALCARECOTkAlRelValMCCTFDQM)
        process.pathALCARECOTkAlRelValMCRS0T = cms.Path(process.seqALCARECOTkAlRelValMCRS0T*process.ALCARECOTkAlRelValMCRS0TDQM)
        process.pathALCARECOTkAlRelValMCCosmicTFHLT = cms.Path(process.seqALCARECOTkAlRelValMCCosmicTFHLT*process.ALCARECOTkAlRelValMCCosmicTFDQM)
        process.pathALCARECOTkAlMuonIsolated = cms.Path(process.seqALCARECOTkAlMuonIsolated*process.ALCARECOTkAlMuonIsolatedDQM)
        process.pathALCARECOTkAlUpsilonMuMu = cms.Path(process.seqALCARECOTkAlUpsilonMuMu*process.ALCARECOTkAlUpsilonMuMuDQM)
        process.pathALCARECOHcalCalDijets = cms.Path(process.seqALCARECOHcalCalDijets*process.ALCARECOHcalCalDiJetsDQM)
        process.pathALCARECOMuAlZMuMu = cms.Path(process.seqALCARECOMuAlZMuMu*process.ALCARECOMuAlZMuMuDQM)
        process.pathALCARECOTkAlBeamHalo = cms.Path(process.seqALCARECOTkAlBeamHalo*process.ALCARECOTkAlBeamHaloDQM)
        process.pathALCARECOSiPixelLorentzAngle = cms.Path(process.seqALCARECOSiPixelLorentzAngle)
        process.pathALCARECOEcalCalElectron = cms.Path(process.seqALCARECOEcalCalElectron*process.ALCARECOEcalCalElectronCalibDQM)
        process.pathALCARECOTkAlRelValMCCTF0THLT = cms.Path(process.seqALCARECOTkAlRelValMCCTF0THLT*process.ALCARECOTkAlRelValMCCTF0TDQM)
        process.pathALCARECOMuAlCalIsolatedMu = cms.Path(process.seqALCARECOMuAlCalIsolatedMu*process.ALCARECOMuAlCalIsolatedMuDQM*process.ALCARECODTCalibrationDQM)
        process.pathALCARECOSiStripCalZeroBias = cms.Path(process.seqALCARECOSiStripCalZeroBias*process.ALCARECOSiStripCalZeroBiasDQM)
        process.pathALCARECOTkAlRelValMCRSHLT = cms.Path(process.seqALCARECOTkAlRelValMCRSHLT*process.ALCARECOTkAlRelValMCRSDQM)
        process.pathALCARECOSiStripCalMinBias = cms.Path(process.seqALCARECOSiStripCalMinBias)
        process.pathALCARECODQM = cms.Path(process.MEtoEDMConverter)
        process.pathALCARECOTkAlLAS = cms.Path(process.seqALCARECOTkAlLAS*process.ALCARECOTkAlLASDQM)
        process.pathALCARECOTkAlMinBias = cms.Path(process.seqALCARECOTkAlMinBias*process.ALCARECOTkAlMinBiasDQM)
        process.pathALCARECOTkAlRelValMCCosmicTF0T = cms.Path(process.seqALCARECOTkAlRelValMCCosmicTF0T*process.ALCARECOTkAlRelValMCCosmicTF0TDQM)
        process.pathALCARECOTkAlRelValMCRS = cms.Path(process.seqALCARECOTkAlRelValMCRS*process.ALCARECOTkAlRelValMCRSDQM)
        process.pathALCARECORpcCalHLT = cms.Path(process.seqALCARECORpcCalHLT)
        process.pathALCARECOHcalCalGammaJet = cms.Path(process.seqALCARECOHcalCalGammaJet)
        process.pathALCARECOMuAlBeamHaloOverlaps = cms.Path(process.seqALCARECOMuAlBeamHaloOverlaps*process.ALCARECOMuAlBeamHaloOverlapsDQM)
        process.pathALCARECOTkAlRelValMCCosmicTF0THLT = cms.Path(process.seqALCARECOTkAlRelValMCCosmicTF0THLT*process.ALCARECOTkAlRelValMCCosmicTF0TDQM)
        process.pathALCARECOHcalCalNoise = cms.Path(process.seqALCARECOHcalCalNoise)
        process.pathALCARECOMuAlOverlaps = cms.Path(process.seqALCARECOMuAlOverlaps*process.ALCARECOMuAlOverlapsDQM)
        process.pathALCARECOTkAlRelValMCCosmicTF = cms.Path(process.seqALCARECOTkAlRelValMCCosmicTF*process.ALCARECOTkAlRelValMCCosmicTFDQM)
        process.pathALCARECOMuAlGlobalRelValMC = cms.Path(process.seqALCARECOMuAlGlobalRelValMC*process.ALCARECOMuAlGlobalRelValMCDQM)
        process.pathALCARECOTkAlJpsiMuMu = cms.Path(process.seqALCARECOTkAlJpsiMuMu*process.ALCARECOTkAlJpsiMuMuDQM)
        process.endjob_step = cms.Path(process.endOfProcess)
        
        
        # Schedule definition
        process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.pathALCARECORpcCalHLT,process.pathALCARECOHcalCalHORelValMC,process.pathALCARECOMuAlCalIsolatedMu,process.pathALCARECOTkAlRelValMCCTFHLT,process.pathALCARECOTkAlRelValMCCosmicTFHLT,process.pathALCARECOTkAlRelValMCRSHLT,process.pathALCARECOTkAlRelValMCCTF0T,process.pathALCARECOTkAlRelValMCCosmicTF0T,process.pathALCARECOTkAlRelValMCRS0T,process.pathALCARECOMuAlGlobalRelValMC,process.pathALCARECOMuAlStandAloneRelValMC,process.endjob_step)
        
        
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
          -s ALCA:MuAlStandAloneRelValMC+DQM
          --scenario cosmics
          --conditions FrontierConditions_GlobalTag,CRAFT_V16P::All
          --no_exec --data


        Expecting GlobalTag to be provided via API initially although
        this may not be the case

        """
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "cosmics"
        options.step = 'ALCA:MuAlStandAloneRelValMC+DQM'
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
        process.load('Configuration/EventContent/EventContentRelValMC_cff')
        
        process.configurationMetadata = cms.untracked.PSet(
            version = cms.untracked.string('$Revision: 1.11 $'),
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
        process.ALCARECOStreamMuAlStandAloneRelValMC = cms.OutputModule("PoolOutputModule",
            SelectEvents = cms.untracked.PSet(
                SelectEvents = cms.vstring('pathALCARECOMuAlStandAloneRelValMC')
            ),
            outputCommands = cms.untracked.vstring('drop *', 
                'keep *_ALCARECOMuAlStandAloneRelValMC_*_*', 
                'keep *_muonCSCDigis_*_*', 
                'keep *_muonDTDigis_*_*', 
                'keep *_muonRPCDigis_*_*', 
                'keep *_dt1DRecHits_*_*', 
                'keep *_dt2DSegments_*_*', 
                'keep *_dt4DSegments_*_*', 
                'keep *_csc2DRecHits_*_*', 
                'keep *_cscSegments_*_*', 
                'keep *_rpcRecHits_*_*'),
            fileName = cms.untracked.string('ALCARECOMuAlStandAloneRelValMC.root'),
            dataset = cms.untracked.PSet(
                filterName = cms.untracked.string('StreamALCARECOMuAlStandAloneRelValMC'),
                dataTier = cms.untracked.string('ALCARECO')
            )
        )
        
        
        # Path and EndPath definitions
        process.pathALCARECOHcalCalHORelValMC = cms.Path(
            process.seqALCARECOHcalCalHORelValMC)
        process.pathALCARECOMuAlStandAloneRelValMC = cms.Path(
            process.seqALCARECOMuAlStandAloneRelValMC*process.ALCARECOMuAlStandAloneRelValMCDQM)
        process.pathALCARECOTkAlZMuMu = cms.Path(process.seqALCARECOTkAlZMuMu*process.ALCARECOTkAlZMuMuDQM)
        process.pathALCARECOTkAlRelValMCCTF0T = cms.Path(process.seqALCARECOTkAlRelValMCCTF0T*process.ALCARECOTkAlRelValMCCTF0TDQM)
        process.pathALCARECOMuAlBeamHalo = cms.Path(process.seqALCARECOMuAlBeamHalo*process.ALCARECOMuAlBeamHaloDQM)
        process.pathALCARECOTkAlRelValMCRS0THLT = cms.Path(process.seqALCARECOTkAlRelValMCRS0THLT*process.ALCARECOTkAlRelValMCRS0TDQM)
        process.pathALCARECOTkAlRelValMCCTF = cms.Path(process.seqALCARECOTkAlRelValMCCTF*process.ALCARECOTkAlRelValMCCTFDQM)
        process.pathALCARECOHcalCalIsoTrk = cms.Path(process.seqALCARECOHcalCalIsoTrk*process.ALCARECOHcalCalIsoTrackDQM)
        process.pathALCARECOHcalCalHO = cms.Path(process.seqALCARECOHcalCalHO*process.ALCARECOHcalCalHODQM)
        process.pathALCARECOTkAlRelValMCCTFHLT = cms.Path(process.seqALCARECOTkAlRelValMCCTFHLT*process.ALCARECOTkAlRelValMCCTFDQM)
        process.pathALCARECOTkAlRelValMCRS0T = cms.Path(process.seqALCARECOTkAlRelValMCRS0T*process.ALCARECOTkAlRelValMCRS0TDQM)
        process.pathALCARECOTkAlRelValMCCosmicTFHLT = cms.Path(process.seqALCARECOTkAlRelValMCCosmicTFHLT*process.ALCARECOTkAlRelValMCCosmicTFDQM)
        process.pathALCARECOTkAlMuonIsolated = cms.Path(process.seqALCARECOTkAlMuonIsolated*process.ALCARECOTkAlMuonIsolatedDQM)
        process.pathALCARECOTkAlUpsilonMuMu = cms.Path(process.seqALCARECOTkAlUpsilonMuMu*process.ALCARECOTkAlUpsilonMuMuDQM)
        process.pathALCARECOHcalCalDijets = cms.Path(process.seqALCARECOHcalCalDijets*process.ALCARECOHcalCalDiJetsDQM)
        process.pathALCARECOMuAlZMuMu = cms.Path(process.seqALCARECOMuAlZMuMu*process.ALCARECOMuAlZMuMuDQM)
        process.pathALCARECOTkAlBeamHalo = cms.Path(process.seqALCARECOTkAlBeamHalo*process.ALCARECOTkAlBeamHaloDQM)
        process.pathALCARECOSiPixelLorentzAngle = cms.Path(process.seqALCARECOSiPixelLorentzAngle)
        process.pathALCARECOEcalCalElectron = cms.Path(process.seqALCARECOEcalCalElectron*process.ALCARECOEcalCalElectronCalibDQM)
        process.pathALCARECOTkAlRelValMCCTF0THLT = cms.Path(process.seqALCARECOTkAlRelValMCCTF0THLT*process.ALCARECOTkAlRelValMCCTF0TDQM)
        process.pathALCARECOMuAlCalIsolatedMu = cms.Path(process.seqALCARECOMuAlCalIsolatedMu*process.ALCARECOMuAlCalIsolatedMuDQM*process.ALCARECODTCalibrationDQM)
        process.pathALCARECOSiStripCalZeroBias = cms.Path(process.seqALCARECOSiStripCalZeroBias*process.ALCARECOSiStripCalZeroBiasDQM)
        process.pathALCARECOTkAlRelValMCRSHLT = cms.Path(process.seqALCARECOTkAlRelValMCRSHLT*process.ALCARECOTkAlRelValMCRSDQM)
        process.pathALCARECOSiStripCalMinBias = cms.Path(process.seqALCARECOSiStripCalMinBias)
        process.pathALCARECODQM = cms.Path(process.MEtoEDMConverter)
        process.pathALCARECOTkAlLAS = cms.Path(process.seqALCARECOTkAlLAS*process.ALCARECOTkAlLASDQM)
        process.pathALCARECOTkAlMinBias = cms.Path(process.seqALCARECOTkAlMinBias*process.ALCARECOTkAlMinBiasDQM)
        process.pathALCARECOTkAlRelValMCCosmicTF0T = cms.Path(process.seqALCARECOTkAlRelValMCCosmicTF0T*process.ALCARECOTkAlRelValMCCosmicTF0TDQM)
        process.pathALCARECOTkAlRelValMCRS = cms.Path(process.seqALCARECOTkAlRelValMCRS*process.ALCARECOTkAlRelValMCRSDQM)
        process.pathALCARECORpcCalHLT = cms.Path(process.seqALCARECORpcCalHLT)
        process.pathALCARECOHcalCalGammaJet = cms.Path(process.seqALCARECOHcalCalGammaJet)
        process.pathALCARECOMuAlBeamHaloOverlaps = cms.Path(process.seqALCARECOMuAlBeamHaloOverlaps*process.ALCARECOMuAlBeamHaloOverlapsDQM)
        process.pathALCARECOTkAlRelValMCCosmicTF0THLT = cms.Path(process.seqALCARECOTkAlRelValMCCosmicTF0THLT*process.ALCARECOTkAlRelValMCCosmicTF0TDQM)
        process.pathALCARECOHcalCalNoise = cms.Path(process.seqALCARECOHcalCalNoise)
        process.pathALCARECOMuAlOverlaps = cms.Path(process.seqALCARECOMuAlOverlaps*process.ALCARECOMuAlOverlapsDQM)
        process.pathALCARECOTkAlRelValMCCosmicTF = cms.Path(process.seqALCARECOTkAlRelValMCCosmicTF*process.ALCARECOTkAlRelValMCCosmicTFDQM)
        process.pathALCARECOMuAlGlobalRelValMC = cms.Path(process.seqALCARECOMuAlGlobalRelValMC*process.ALCARECOMuAlGlobalRelValMCDQM)
        process.pathALCARECOTkAlJpsiMuMu = cms.Path(process.seqALCARECOTkAlJpsiMuMu*process.ALCARECOTkAlJpsiMuMuDQM)
        process.endjob_step = cms.Path(process.endOfProcess)
        process.ALCARECOStreamMuAlStandAloneRelValMCOutPath = cms.EndPath(process.ALCARECOStreamMuAlStandAloneRelValMC)
        
        # Schedule definition
        process.schedule = cms.Schedule(process.pathALCARECODQM,process.pathALCARECOMuAlStandAloneRelValMC,process.endjob_step,process.ALCARECOStreamMuAlStandAloneRelValMCOutPath)
        

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
        options.step = "HARVESTING:dqmHarvesting"
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        options.name = "EDMtoMEConvert"
        options.number = -1
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
        options.arguments = ""
        options.evt_type = ""
        options.filein = []
        options.gflash = False
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
