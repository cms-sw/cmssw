import FWCore.ParameterSet.Config as cms
import os
import json

process = cms.Process("READ")

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

import FWCore.ParameterSet.VarParsing as VarParsing
from Configuration.AlCa.GlobalTag import GlobalTag

options = VarParsing.VarParsing()
options.register('lumisPerRun',
                1,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "the number of lumis to be processed per-run.")
options.register('firstRun',
                290550,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "the first run number be processed")
options.register('lastRun',
                325175,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "the run number to stop at")
options.register('config',
                 default = None,
                 mult    = VarParsing.VarParsing.multiplicity.singleton,
                 mytype  = VarParsing.VarParsing.varType.string,
                 info    = 'JSON config with information about the GT, Alignments, etc.')
options.register('unitTest',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "is it a unit test?")

defaultFirstRun    = options.firstRun
defaultLastRun     = options.lastRun
defaultLumisPerRun = options.lumisPerRun

options.parseArguments()

if(options.config is None):
    configuration = {
        "alignments": {
            "prompt": {
                "globaltag": "140X_dataRun3_Prompt_v4",
                "conditions": {"TrackerAlignmentRcd": {"tag":"TrackerAlignment_PCL_byRun_v2_express"}}
            },
            "EOY": {
                "conditions": {"TrackerAlignmentRcd": {"tag":"TrackerAlignment_v24_offline"}}
            },
            "rereco": {
                "conditions": {"TrackerAlignmentRcd": {"tag":"TrackerAlignment_v29_offline"}}
            }
        },
        "validation": {}
    }
else:
    # Load configuration from file
    with open(options.config) as f:
        configuration = json.load(f)

# The priority for the options is:
# 1. Value specified on command line
# 2. Value in the config
# 3. Default value in the parser
if(options.firstRun != defaultFirstRun):
    firstRun = options.firstRun
else:
    firstRun = configuration["validation"].get('firstRun', defaultFirstRun)

if(options.lastRun != defaultLastRun):
    lastRun = options.lastRun
else:
    lastRun = configuration["validation"].get('lastRun', defaultLastRun)

if(options.lumisPerRun != defaultLumisPerRun):
    lumisPerRun = options.lumisPerRun
else:
    lumisPerRun = configuration["validation"].get('lumisPerRun', defaultLumisPerRun)

process.load("FWCore.MessageService.MessageLogger_cfi")

# Test that the configuration is complete
if(lastRun < firstRun):
    raise ValueError("The last run is smaller than the first")

process.MessageLogger.cerr.FwkReport.reportEvery = lumisPerRun*1000   # do not clog output with I/O

if options.unitTest:
    numberOfRuns = 10
else:
    numberOfRuns = lastRun - firstRun + 1

print("INFO: Runs: {:d} - {:d} --> number of runs: {:d}".format(firstRun, lastRun, numberOfRuns))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.lumisPerRun*numberOfRuns) )

####################################################################
# Empty source 
####################################################################
#import FWCore.PythonUtilities.LumiList as LumiList
#DCSJson='/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/DCSOnly/json_DCSONLY.txt'

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(firstRun),
                            firstLuminosityBlock = cms.untracked.uint32(1),           # probe one LS after the other
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1),  # probe one event per LS
                            numberEventsInRun = cms.untracked.uint32(lumisPerRun),           # a number of events > the number of LS possible in a real run (5000 s ~ 32 h)
                            )

####################################################################
# Load and configure analyzer
####################################################################
bcLabels_ = cms.untracked.vstring("")
bsLabels_ = cms.untracked.vstring("")

alignments = configuration.get('alignments', None) # NOTE: aligments is plural
if alignments is None:
    align = configuration['alignment'] # NOTE: alignment is singular
    label = configuration['alignment'].get('label', align['title'].split()[0])
    alignments = {label: align}

for label, align in alignments.items():
    if(align.get('globaltag')):
        if(len(process.GlobalTag.globaltag.value()) > 0):
            if(process.GlobalTag.globaltag.value() != align['globaltag']):
                print('ERROR: another GT has already been specified: "{}". Ignoring GT "{}" from alignment "{}"'.format(
                    process.GlobalTag.globaltag.value(), align['globaltag'], label))
        else:
            # Assign this GlobalTag to the process
            process.GlobalTag = GlobalTag(process.GlobalTag, align['globaltag'])
            print('INFO: GlobalTag:', process.GlobalTag.globaltag.value())

    conditions = align.get('conditions')
    if(conditions is None):
        print('INFO: No conditions specified for alignment "{}": skipping'.format(label))
        continue

    bcLabels_.append(label)
    print(f'TrackerAlignment: {label=} {align=}')

    for record, condition in conditions.items():
        condition.setdefault('connect', 'frontier://FrontierProd/CMS_CONDITIONS')
        if  (record == 'TrackerAlignmentRcd'):
            condition.setdefault('tag', 'Alignments')
        elif(record == 'TrackerSurfaceDeformationRcd'):
            condition.setdefault('tag', 'Deformations')
        elif(record == 'TrackerAlignmentErrorsExtendedRcd'): # Errors should not affect the barycentre
            condition.setdefault('tag', 'AlignmentErrors')

        process.GlobalTag.toGet.append(
            cms.PSet(
                record  = cms.string(record),
                label   = cms.untracked.string(label),
                tag     = cms.string(condition['tag']),
                connect = cms.string(condition['connect'])
            )
        )


for label, beamspot in configuration['validation'].get("beamspots", {}).items() :
    bsLabels_.append(label)
    print(f'BeamSpot        : {label=} {beamspot=}')

    process.GlobalTag.toGet.append(
        cms.PSet(
            record = cms.string("BeamSpotObjectsRcd"),
            label = cms.untracked.string(label),
            tag = cms.string(beamspot["tag"]),
            connect = cms.string(beamspot.get("connect", "frontier://FrontierProd/CMS_CONDITIONS"))
        )
    )


from Alignment.OfflineValidation.pixelBaryCentreAnalyzer_cfi import pixelBaryCentreAnalyzer as _pixelBaryCentreAnalyzer

process.PixelBaryCentreAnalyzer = _pixelBaryCentreAnalyzer.clone(
    usePixelQuality = False,
    tkAlignLabels = bcLabels_,
    beamSpotLabels = bsLabels_
)

process.PixelBaryCentreAnalyzerWithPixelQuality = _pixelBaryCentreAnalyzer.clone(
    usePixelQuality = True,
    tkAlignLabels = bcLabels_,
    beamSpotLabels = bsLabels_
)

####################################################################
# Output file
####################################################################
outfile = os.path.join(configuration.get("output", os.getcwd()), 'PixelBaryCentre.root')

process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string(outfile)
                                   ) 
print('INFO: output in', outfile)

# Put module in path:
process.p = cms.Path(process.PixelBaryCentreAnalyzer
#*process.PixelBaryCentreAnalyzerWithPixelQuality
)
-- dummy change --
