#
# $Id: edm2me_cfg.py,v 1.3 2009/05/07 12:54:54 argiro Exp $
# Author Stefano Argiro
#
# Extract histos from EDM root file
#
# Usage:
# cmsRun edm2me_cfg.py files=file:file.root
#
# Todo: specify output file name from command line
#

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing


process = cms.Process("EDMtoMEConvert")
process.load("DQMServices.Components.EDMtoMEConverter_cff")

# setup 'standard'  options
options = VarParsing.VarParsing ('standard')
options.maxEvents = -1 # -1 means all events

# get and parse the command line arguments
options.parseArguments()

# Use the options

process.source = cms.Source ("PoolSource",
                              fileNames = cms.untracked.vstring (options.files)
                             )
                                                          
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32 (options.maxEvents)
        )


process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
        )

                           

process.p = cms.Path(process.EDMtoMEConverter*process.dqmSaver)
process.EDMtoMEConverter.Verbosity = 1
process.EDMtoMEConverter.Frequency = 1
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/A/B/C'
