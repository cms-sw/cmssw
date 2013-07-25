# Auto generated configuration file
# using: 
# Revision: 1.173 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step2 -s RAW2DIGI,L1Reco,RECO,ALCA:MuAlCalIsolatedMu+DtCalib,VALIDATION,DQM --datatier GEN-SIM-RECO --eventcontent RECOSIM --geometry DB --conditions auto:startup -n 100 --customise Geometry/RPCGeometry/customise_RPCgeom37X.py --fileout file:reco.root --filein file:raw.root
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.MixingNoPileUp_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.AlCaRecoStreamsMC_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('step2 nevts:100'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.options = cms.untracked.PSet(

)
# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:raw.root')
)

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    fileName = cms.untracked.string('file:reco.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    )
)

# Additional output definition
process.ALCARECOStreamMuAlCalIsolatedMu = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlCalIsolatedMu')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOMuAlCalIsolatedMu_*_*', 
        'keep *_muonCSCDigis_*_*', 
        'keep *_muonDTDigis_*_*', 
        'keep *_muonRPCDigis_*_*', 
        'keep *_dt1DRecHits_*_*', 
        'keep *_dt2DSegments_*_*', 
        'keep *_dt4DSegments_*_*', 
        'keep *_csc2DRecHits_*_*', 
        'keep *_cscSegments_*_*', 
        'keep *_rpcRecHits_*_*', 
        'keep L1AcceptBunchCrossings_*_*_*', 
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', 
        'keep *_TriggerResults_*_*', 
        'keep DcsStatuss_scalersRawToDigi_*_*', 
        'keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('MuAlCalIsolatedMu.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('MuAlCalIsolatedMu'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamDtCalib = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECODtCalib')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_dt4DSegments_*_*', 
        'keep *_dt4DSegmentsNoWire_*_*', 
        'keep *_muonDTDigis_*_*', 
        'keep *_dttfDigis_*_*', 
        'keep *_gtDigis_*_*', 
        'keep *_TriggerResults_*_*', 
        'keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('DtCalib.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('DtCalib'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)

# Other statements
process.mix.playback = True
process.GlobalTag.globaltag = 'START37_V1::All'

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.pathALCARECOHcalCalHOCosmics = cms.Path(process.seqALCARECOHcalCalHOCosmics)
process.pathALCARECOMuAlStandAloneCosmics = cms.Path(process.seqALCARECOMuAlStandAloneCosmics*process.ALCARECOMuAlStandAloneCosmicsDQM)
process.pathALCARECOTkAlZMuMu = cms.Path(process.seqALCARECOTkAlZMuMu*process.ALCARECOTkAlZMuMuDQM)
process.pathALCARECOTkAlCosmicsCTF0T = cms.Path(process.seqALCARECOTkAlCosmicsCTF0T*process.ALCARECOTkAlCosmicsCTF0TDQM)
process.pathALCARECOMuAlBeamHalo = cms.Path(process.seqALCARECOMuAlBeamHalo*process.ALCARECOMuAlBeamHaloDQM)
process.pathALCARECOTkAlCosmicsCTF = cms.Path(process.seqALCARECOTkAlCosmicsCTF*process.ALCARECOTkAlCosmicsCTFDQM)
process.pathALCARECOHcalCalHO = cms.Path(process.seqALCARECOHcalCalHO*process.ALCARECOHcalCalHODQM)
process.pathALCARECOTkAlCosmicsCTFHLT = cms.Path(process.seqALCARECOTkAlCosmicsCTFHLT*process.ALCARECOTkAlCosmicsCTFDQM)
process.pathALCARECODtCalib = cms.Path(process.seqALCARECODtCalib*process.ALCARECODTCalibSynchDQM)
process.pathALCARECOTkAlCosmicsCosmicTFHLT = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTFHLT*process.ALCARECOTkAlCosmicsCosmicTFDQM)
process.pathALCARECOHcalCalMinBias = cms.Path(process.seqALCARECOHcalCalMinBias*process.ALCARECOHcalCalPhisymDQM)
process.pathALCARECOTkAlMuonIsolated = cms.Path(process.seqALCARECOTkAlMuonIsolated*process.ALCARECOTkAlMuonIsolatedDQM)
process.pathALCARECOTkAlUpsilonMuMu = cms.Path(process.seqALCARECOTkAlUpsilonMuMu*process.ALCARECOTkAlUpsilonMuMuDQM)
process.pathALCARECOHcalCalDijets = cms.Path(process.seqALCARECOHcalCalDijets*process.ALCARECOHcalCalDiJetsDQM)
process.pathALCARECOMuAlZMuMu = cms.Path(process.seqALCARECOMuAlZMuMu*process.ALCARECOMuAlZMuMuDQM)
process.pathALCARECOEcalCalPi0Calib = cms.Path(process.seqALCARECOEcalCalPi0Calib*process.ALCARECOEcalCalPi0CalibDQM)
process.pathALCARECOTkAlBeamHalo = cms.Path(process.seqALCARECOTkAlBeamHalo*process.ALCARECOTkAlBeamHaloDQM)
process.pathALCARECOSiPixelLorentzAngle = cms.Path(process.seqALCARECOSiPixelLorentzAngle)
process.pathALCARECOTkAlCosmicsCosmicTF0T = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF0T*process.ALCARECOTkAlCosmicsCosmicTF0TDQM)
process.pathALCARECOEcalCalElectron = cms.Path(process.seqALCARECOEcalCalElectron*process.ALCARECOEcalCalElectronCalibDQM)
process.pathALCARECOTkAlCosmicsCTF0THLT = cms.Path(process.seqALCARECOTkAlCosmicsCTF0THLT*process.ALCARECOTkAlCosmicsCTF0TDQM)
process.pathALCARECOMuAlCalIsolatedMu = cms.Path(process.seqALCARECOMuAlCalIsolatedMu*process.ALCARECOMuAlCalIsolatedMuDQM*process.ALCARECODTCalibrationDQM)
process.pathALCARECOSiStripCalZeroBias = cms.Path(process.seqALCARECOSiStripCalZeroBias*process.ALCARECOSiStripCalZeroBiasDQM)
process.pathALCARECOEcalCalEtaCalib = cms.Path(process.seqALCARECOEcalCalEtaCalib*process.ALCARECOEcalCalEtaCalibDQM)
process.pathALCARECOHcalCalIsoTrk = cms.Path(process.seqALCARECOHcalCalIsoTrk*process.ALCARECOHcalCalIsoTrackDQM)
process.pathALCARECOSiStripCalMinBias = cms.Path(process.seqALCARECOSiStripCalMinBias*process.ALCARECOSiStripCalMinBiasDQM)
process.pathALCARECODQM = cms.Path(process.MEtoEDMConverter)
process.pathALCARECOTkAlLAS = cms.Path(process.seqALCARECOTkAlLAS*process.ALCARECOTkAlLASDQM)
process.pathALCARECOTkAlMinBias = cms.Path(process.seqALCARECOTkAlMinBias*process.ALCARECOTkAlMinBiasDQM)
process.pathALCARECORpcCalHLT = cms.Path(process.seqALCARECORpcCalHLT)
process.pathALCARECOHcalCalGammaJet = cms.Path(process.seqALCARECOHcalCalGammaJet)
process.pathALCARECOMuAlBeamHaloOverlaps = cms.Path(process.seqALCARECOMuAlBeamHaloOverlaps*process.ALCARECOMuAlBeamHaloOverlapsDQM)
process.pathALCARECOTkAlCosmicsCosmicTF0THLT = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF0THLT*process.ALCARECOTkAlCosmicsCosmicTF0TDQM)
process.pathALCARECOHcalCalNoise = cms.Path(process.seqALCARECOHcalCalNoise)
process.pathALCARECOMuAlOverlaps = cms.Path(process.seqALCARECOMuAlOverlaps*process.ALCARECOMuAlOverlapsDQM)
process.pathALCARECOTkAlCosmicsCosmicTF = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF*process.ALCARECOTkAlCosmicsCosmicTFDQM)
process.pathALCARECOEcalCalPhiSym = cms.Path(process.seqALCARECOEcalCalPhiSym*process.ALCARECOEcalCalPhisymDQM)
process.pathALCARECOMuAlGlobalCosmics = cms.Path(process.seqALCARECOMuAlGlobalCosmics*process.ALCARECOMuAlGlobalCosmicsDQM)
process.pathALCARECOTkAlJpsiMuMu = cms.Path(process.seqALCARECOTkAlJpsiMuMu*process.ALCARECOTkAlJpsiMuMuDQM)
process.validation_step = cms.Path(process.validation)
process.dqmoffline_step = cms.Path(process.DQMOffline)
process.endjob_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.output)
process.ALCARECOStreamMuAlCalIsolatedMuOutPath = cms.EndPath(process.ALCARECOStreamMuAlCalIsolatedMu)
process.ALCARECOStreamDtCalibOutPath = cms.EndPath(process.ALCARECOStreamDtCalib)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.pathALCARECOMuAlCalIsolatedMu,process.pathALCARECODtCalib,process.validation_step,process.dqmoffline_step,process.endjob_step,process.out_step,process.ALCARECOStreamMuAlCalIsolatedMuOutPath,process.ALCARECOStreamDtCalibOutPath)


# Automatic addition of the customisation function
def customise(process):
    process.GlobalTag.toGet = cms.VPSet(
        cms.PSet(record = cms.string("PCastorRcd"),
                 tag = cms.string("CASTORRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PZdcRcd"),
                 tag = cms.string("ZDCRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PCaloTowerRcd"),
                 tag = cms.string("CTRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PEcalEndcapRcd"),
                 tag = cms.string("EERECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("CSCRecoDigiParametersRcd"),
                 tag = cms.string("CSCRECODIGI_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("CSCRecoGeometryRcd"),
                 tag = cms.string("CSCRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PEcalBarrelRcd"),
                 tag = cms.string("EBRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_IdealGFlash_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY"),
                 label = cms.untracked.string("IdealGFlash")
                 ),
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_Ideal_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY"),
                 label = cms.untracked.string("Ideal")
                 ),
        cms.PSet(record = cms.string("RPCRecoGeometryRcd"),
                 tag = cms.string("RPCRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("DTRecoGeometryRcd"),
                 tag = cms.string("DTRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PEcalPreshowerRcd"),
                 tag = cms.string("EPRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_ExtendedGFlash_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY"),
                 label = cms.untracked.string("ExtendedGFlash")
                 ),
        cms.PSet(record = cms.string("GeometryFileRcd"),
                 tag = cms.string("XMLFILE_Geometry_Extended_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY"),
                 label = cms.untracked.string("Extended")
                 ),
        cms.PSet(record = cms.string("IdealGeometryRecord"),
                 tag = cms.string("TKRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 ),
        cms.PSet(record = cms.string("PHcalRcd"),
                 tag = cms.string("HCALRECO_Geometry_Tag38YV0"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_34X_GEOMETRY")
                 )
        )
    return(process)



# End of customisation function definition

process = customise(process)
