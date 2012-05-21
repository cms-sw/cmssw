import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQM_cfg import *

from DQM.L1TMonitor.environment_file_cfi import *


#RunType, and Runkey selection from RCMS
import sys
from FWCore.ParameterSet.VarParsing import VarParsing
from DQM.Integration.test.dqmPythonTypes import *

runParameters = VarParsing ('analysis')
runParameters.register ('runtype',
  'pp_run',
  VarParsing.multiplicity.singleton,
  VarParsing.varType.string,
  "Type of Run in CMS")

runParameters.register ('runkey',
  'pp_run',
  VarParsing.multiplicity.singleton,
  VarParsing.varType.string,
  "Run Keys of CMS")

# Fix to allow scram to compile
if len(sys.argv) > 1:
  runParameters.parseArguments()

runType = RunType(['pp_run','cosmic_run','hi_run','hpu_run'])
if not runParameters.runkey.strip():
  runParameters.runkey = 'pp_run'

runType.setRunType(runParameters.runkey.strip())


