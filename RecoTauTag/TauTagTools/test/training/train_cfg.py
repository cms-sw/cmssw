'''

TaNC MVA trainer

Author: Evan K. Friis (UC Davis)

'''

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os
import sys

options = VarParsing.VarParsing ('analysis')

# Register options
options.register(
    'xml', '',
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "XML file with MVA configuration")

options.parseArguments()

# Map the XML file name to a nice computer name
_computer_name = os.path.basename(os.path.splitext(options.xml)[0])
print _computer_name

decay_mode_map = {
    '1prong0pi0' : 0,
    '1prong1pi0' : 1,
    '1prong2pi0' : 2,
    '3prong0pi0' : 10,
}
_decay_mode = decay_mode_map[_computer_name]

process = cms.Process("TrainMVA")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# Input files
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles),
    skipBadFiles = cms.untracked.bool(True),
)

#######################################################
# Database BS
#######################################################

process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet( messageLevel = cms.untracked.int32(0) ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:%s' % options.outputFile.replace('.root','')),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('TauTagMVAComputerRcd'),
        tag = cms.string('Train')
    ))
)

process.MVATrainerSave = cms.EDAnalyzer(
    "TauMVATrainerSave",
    toPut = cms.vstring(_computer_name),
    toCopy = cms.vstring()
)

setattr(process.MVATrainerSave, _computer_name,
        cms.string(_computer_name + ".mva"))

process.looper = cms.Looper(
    "TauMVATrainerLooper",
    trainers = cms.VPSet(cms.PSet(
        calibrationRecord = cms.string(_computer_name),
        saveState = cms.untracked.bool(True),
        trainDescription = cms.untracked.string(options.xml),
        loadState = cms.untracked.bool(False),
        doMonitoring = cms.bool(True),
    ))
)

###############################################################################
# Define signal and background paths Each path only gets run if the appropriate
# colleciton is present.  This allows separation of signal and background
# events.
###############################################################################

# Finally, pass our selected sig/bkg taus to the MVA trainer
from RecoTauTag.RecoTau.RecoTauDiscriminantConfiguration import \
        discriminantConfiguration

discriminantConfiguration.FlightPathSignificance.discSrc = cms.VInputTag(
    "hpsTancTausDiscriminationByFlightPathSignal",
    "hpsTancTausDiscriminationByFlightPathBackground",
)

process.trainer = cms.EDAnalyzer(
    "RecoTauMVATrainer",
    signalSrc = cms.InputTag(
        "selectedHpsTancTrainTausDecayMode%iSignal" % _decay_mode),
    backgroundSrc = cms.InputTag(
        "selectedHpsTancTrainTausDecayMode%iBackground" % _decay_mode),
    computerName = cms.string(_computer_name),
    dbLabel = cms.string("trainer"),
    discriminantOptions = discriminantConfiguration
)

process.trainPath = cms.Path(process.trainer)

process.outpath = cms.EndPath(process.MVATrainerSave)

process.schedule = cms.Schedule(
    process.trainPath,
    process.outpath
)

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
