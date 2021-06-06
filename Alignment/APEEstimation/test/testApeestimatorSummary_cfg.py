from __future__ import print_function
import os

import FWCore.ParameterSet.Config as cms





##
## Setup command line options
##
import FWCore.ParameterSet.VarParsing as VarParsing
import sys
options = VarParsing.VarParsing ('standard')
options.register('sample', 'wlnu', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "wlnu")
options.register('isTest', True, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Test run")

# get and parse the command line arguments
if( hasattr(sys, "argv") ):
    for args in sys.argv :
        arg = args.split(',')
        for val in arg:
            val = val.split('=')
            if(len(val)==2):
                setattr(options,val[0], val[1])

print("Input sample: ", options.sample)
print("Test run: ", options.isTest)



##
## Process definition
##
process = cms.Process("ApeEstimatorSummary")



##
## Message Logger
##
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.CalculateAPE=dict()
#process.MessageLogger.ApeEstimatorSummary=dict()
process.MessageLogger.cerr.INFO.limit = 0
process.MessageLogger.cerr.default.limit = -1
process.MessageLogger.cerr.CalculateAPE = cms.untracked.PSet(limit = cms.untracked.int32(-1))
#process.MessageLogger.cerr.ApeEstimatorSummary = cms.untracked.PSet(limit = cms.untracked.int32(-1))

#process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
#    reportEvery = cms.untracked.int32(100),  # every 100th only
#    limit = cms.untracked.int32(10),         # or limit to 10 printouts...
#))
process.MessageLogger.cerr.FwkReport.reportEvery = 1000 ## really show only every 1000th



##
## Process options
##
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
)



##
## Input sample definition
##
isData1 = isData2 = False
isData = False
isQcd = isWlnu = isZmumu = isZtautau = isZmumu10 = isZmumu20 = False
isMc = False
if options.sample == 'data1':
    isData1 = True
    isData = True
elif options.sample == 'data2':
    isData2 = True
    isData = True
elif options.sample == 'qcd':
    isQcd = True
    isMc = True
elif options.sample == 'wlnu':
    isWlnu = True
    isMc = True
elif options.sample == 'zmumu':
    isZmumu = True
    isMc = True
elif options.sample == 'ztautau':
    isZtautau = True
    isMc = True
elif options.sample == 'zmumu10':
    isZmumu10 = True
    isMc = True
elif options.sample == 'zmumu20':
    isZmumu20 = True
    isMc = True
else:
    print('ERROR --- incorrect data sammple: ', options.sample)
    exit(8888)



##
## Input Files
##
process.source = cms.Source("EmptySource")



##
## Number of Events
##
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )



##
## ApeEstimatorSummary
##
from Alignment.APEEstimation.ApeEstimatorSummary_cff import *
process.ApeEstimatorSummary1 = ApeEstimatorSummaryBaseline.clone(
    InputFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/test_' + options.sample + '.root',
    ResultsFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/test_' + options.sample + '_resultsFile1.root',
    BaselineFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/test_' + options.sample + '_baselineApe.root',
)
process.ApeEstimatorSummary2 = ApeEstimatorSummaryIter.clone(
    correctionScaling = 0.6,
    InputFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/test_' + options.sample + '.root',
    ResultsFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/test_' + options.sample + '_resultsFile2.root',
    BaselineFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/test_' + options.sample + '_baselineApe.root',
    IterationFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/test_' + options.sample + '_iterationApe2.root',
    ApeOutputFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/test_' + options.sample + '_apeOutput2.txt',
)
process.ApeEstimatorSummary3 = ApeEstimatorSummaryIter.clone(
    correctionScaling = 0.6,
    InputFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/test_' + options.sample + '.root',
    ResultsFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/test_' + options.sample + '_resultsFile3.root',
    BaselineFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/test_' + options.sample + '_baselineApe.root',
    IterationFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/test_' + options.sample + '_iterationApe3.root',
    ApeOutputFile = os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/test_' + options.sample + '_apeOutput3.txt',
)



process.p = cms.Path(
    process.ApeEstimatorSummary1*
    process.ApeEstimatorSummary2
    #~ *process.ApeEstimatorSummary3
)



