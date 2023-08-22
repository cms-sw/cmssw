import FWCore.ParameterSet.Config as cms
import CondTools.Ecal.db_credentials as auth
import FWCore.ParameterSet.VarParsing as VarParsing


from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel


#process = cms.Process("ProdTPGParam")
process = cms.Process('DIGI',Phase2C17I13M9,phase2_ecal_devel)

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D88_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC14TeV_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')  TO BE FIXED 
process.load('CalibCalorimetry.EBPhase2TPGTools.ecalEBPhase2TPParamProducer_cfi')
"""
options = VarParsing.VarParsing('tpg')

options.register ('outFile',
                  'testtest.txt', 
                  VarParsing.VarParsing.multiplicity.singleton, 
                  VarParsing.VarParsing.varType.string,         
                  "Output file")

options.parseArguments()
"""
# Calo geometry service model
#process.load("Configuration.StandardSequences.GeometryDB_cff")

# ecal mapping
process.eegeom = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalMappingRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

# Get hardcoded conditions the same used for standard digitization before CMSSW_3_1_x
## process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")
# or Get DB parameters
# process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("CondCore.CondDB.CondDB_cfi")

process.CondDB.connect = 'frontier://FrontierProd/CMS_CONDITIONS'
process.CondDB.DBParameters.authenticationPath = '/nfshome0/popcondev/conddb' ###P5 stuff

"""
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toGet = cms.VPSet(
              cms.PSet(
            record = cms.string('EcalPedestalsRcd'),
                    #tag = cms.string('EcalPedestals_v5_online')
                    #tag = cms.string('EcalPedestals_2009runs_hlt') ### obviously diff w.r.t previous
                    tag = cms.string('EcalPedestals_hlt'), ### modif-alex 22/02/2011
                 ),
              cms.PSet(
                record = cms.string('EcalMappingElectronicsRcd'),
                    tag = cms.string('EcalMappingElectronics_EEMap_v1_mc')
                 )
               )
             )
"""

#########################
process.source = cms.Source("EmptySource",
       ##firstRun = cms.untracked.uint32(100000000) ### need to use latest run to pick-up update values from DB
       firstRun = cms.untracked.uint32(161310)
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)



process.p = cms.Path(process.EBPhase2TPGParamProducer)
