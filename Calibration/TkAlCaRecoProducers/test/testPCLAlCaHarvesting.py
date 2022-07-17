from __future__ import print_function
import calendar
import CondCore.Utilities.conddblib as conddb

#___________________________________________________________________
def findRunStopTime(run_number):
    con = conddb.connect(url = conddb.make_url("pro"))
    session = con.session()
    RunInfo = session.get_dbtype(conddb.RunInfo)
    bestRun = session.query(RunInfo.run_number,RunInfo.start_time, RunInfo.end_time).filter(RunInfo.run_number >= run_number).first()
    if bestRun is None:
        raise Exception("Run %s can't be matched with an existing run in the database." % run_number)

    start= bestRun[1]
    stop = bestRun[2]

    bestRunStartTime = calendar.timegm( bestRun[1].utctimetuple() ) << 32
    bestRunStopTime  = calendar.timegm( bestRun[2].utctimetuple() ) << 32

    print("run start time:",start,"(",bestRunStartTime,")")
    print("run stop time: ",stop,"(",bestRunStopTime,")")

    return bestRunStopTime

import optparse
parser = optparse.OptionParser(usage = 'Usage: %prog [options] <file> [<file> ...]\n')
parser.add_option('-G', '--inputGT',
                  dest = 'inputGT',
                  default = 'auto:run2_data',
                  help = 'Global Tag to get conditions')

parser.add_option('-r', '--inputRun',
                  dest = 'inputRun',
                  default = 325022,
                  help = 'run to be used')

parser.add_option('-t', '--inputTime',
                  dest = 'inputTime',
                  default = 6614916085915320320,
                  help = 'time to be used')

parser.add_option('-e', '--enableJobReport',
                  dest = 'empty',
                  default = None,
                  help = 'unused')

(options, arguments) = parser.parse_args()

import FWCore.ParameterSet.Config as cms
process = cms.Process('ALCAHARVEST')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.AlCaHarvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

##
## configure the source with an random run
##
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.inputRun),
                            numberEventsInRun = cms.untracked.uint32(1),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1),
                            firstTime = cms.untracked.uint64(options.inputTime),
                            timeBetweenEvents = cms.untracked.uint64(1)
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

##
## Define the tags to write
##
process.PoolDBOutputService.toPut.append(process.ALCAHARVESTSiStripQuality_dbOutput)
process.PoolDBOutputService.toPut.append(process.ALCAHARVESTSiStripGains_dbOutput)
process.PoolDBOutputService.toPut.append(process.ALCAHARVESTSiStripGainsAAG_dbOutput )
process.PoolDBOutputService.toPut.append(process.ALCAHARVESTSiStripHitEff_dbOutput)
process.PoolDBOutputService.toPut.append(process.ALCAHARVESTSiPixelAli_dbOutput)
process.PoolDBOutputService.toPut.append(process.ALCAHARVESTSiPixelAliHG_dbOutput)
process.PoolDBOutputService.toPut.append(process.ALCAHARVESTSiPixelLA_dbOutput)
process.PoolDBOutputService.toPut.extend(process.ALCAHARVESTSiPixelQuality_dbOutput)
process.PoolDBOutputService.toPut.append(process.ALCAHARVESTBeamSpotByRun_dbOutput)
process.PoolDBOutputService.toPut.append(process.ALCAHARVESTBeamSpotByLumi_dbOutput)
process.PoolDBOutputService.toPut.append(process.ALCAHARVESTBeamSpotHPByRun_dbOutput)
process.PoolDBOutputService.toPut.append(process.ALCAHARVESTBeamSpotHPByLumi_dbOutput)

##
## change the output sqlite file in order to avoid concurrent writing from other unit tests
##
process.PoolDBOutputService.connect = cms.string('sqlite_file:testPCLAlCaHarvesting.db')

##
## Define the file metadatas
##
process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVESTSiStripQuality_metadata)
process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVESTSiStripGains_metadata )
process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVESTSiStripGainsAAG_metadata)
process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVESTSiStripHitEff_metadata)
process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVESTSiPixelAli_metadata)
process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVESTSiPixelAliHG_metadata)
process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVESTSiPixelLA_metadata)
process.pclMetadataWriter.recordsToMap.extend(process.ALCAHARVESTSiPixelQuality_metadata)
process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVESTBeamSpotByRun_metadata)
process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVESTBeamSpotByLumi_metadata)
process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVESTBeamSpotHPByRun_metadata)
process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVESTBeamSpotHPByLumi_metadata)


process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.inputGT, '')

process.SiStripQuality  = cms.Path(process.ALCAHARVESTSiStripQuality)
process.alcaSiStripQualityHarvester.CalibrationThreshold = cms.untracked.uint32(0)

process.SiStripGains    = cms.Path(process.ALCAHARVESTSiStripGains)
process.alcaSiStripGainsHarvester.DQMdir=''
process.alcaSiStripGainsHarvester.minNrEntries=0
process.alcaSiStripGainsHarvester.GoodFracForTagProd=0
process.alcaSiStripGainsHarvester.NClustersForTagProd=0

process.SiStripGainsAAG = cms.Path(process.ALCAHARVESTSiStripGainsAAG)
process.alcaSiStripGainsAAGHarvester.minNrEntries=0
process.alcaSiStripGainsAAGHarvester.minNrEntries=0
process.alcaSiStripGainsAAGHarvester.GoodFracForTagProd=0
process.alcaSiStripGainsAAGHarvester.NClustersForTagProd=0

process.SiStripHitEff = cms.Path(process.ALCAHARVESTSiStripHitEfficiency)

process.SiPixelAli      = cms.Path(process.ALCAHARVESTSiPixelAli)
process.SiPixelAliMilleFileExtractor.outputBinaryFile = cms.string('')
process.SiPixelAliPedeAlignmentProducer.algoConfig.mergeBinaryFiles=[]

process.SiPixelAliHG      = cms.Path(process.ALCAHARVESTSiPixelAliHG)
process.SiPixelAliMilleFileExtractorHG.outputBinaryFile = cms.string('')
process.SiPixelAliPedeAlignmentProducerHG.algoConfig.mergeBinaryFiles=[]

process.SiPixelLA      = cms.Path(process.ALCAHARVESTSiPixelLorentzAngle)

process.SiPixelQuality  = cms.Path(process.ALCAHARVESTSiPixelQuality)

process.ALCAHARVESTDQMSaveAndMetadataWriter = cms.Path(process.dqmSaver+process.pclMetadataWriter)

process.BeamSpotByRun  = cms.Path(process.ALCAHARVESTBeamSpotByRun)
process.BeamSpotByLumi = cms.Path(process.ALCAHARVESTBeamSpotByLumi)
process.BeamSpotHPByRun  = cms.Path(process.ALCAHARVESTBeamSpotHPByRun)
process.BeamSpotHPByLumi = cms.Path(process.ALCAHARVESTBeamSpotHPByLumi)

process.schedule = cms.Schedule(process.SiStripQuality,
                                process.SiStripGains,    
                                process.SiStripGainsAAG, 
                                process.SiStripHitEff,
                                process.SiPixelAli,
                                process.SiPixelAliHG,
                                process.SiPixelLA,
                                process.SiPixelQuality,
                                process.BeamSpotByRun,
                                process.BeamSpotByLumi,
                                process.BeamSpotHPByRun,
                                process.BeamSpotHPByLumi,
                                process.ALCAHARVESTDQMSaveAndMetadataWriter)

from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
