#!/usr/bin/env python3
"""
_RunRepack_

Test/Debugging harness for the repack configuration builder

"""
from __future__ import print_function

import sys
import getopt

from Configuration.DataProcessing.Repack import repackProcess


class RunRepack:

    def __init__(self):
        self.selectEvents = None
        self.inputLFN = None
        self.dataTier = None

    def __call__(self):
        if self.inputLFN == None:
            msg = "No --lfn specified"
            raise RuntimeError(msg)
        allowedDataTiers = ["RAW", "HLTSCOUT", "L1SCOUT"]
        if self.dataTier == None: 
            self.dataTier = "RAW"
        elif self.dataTier not in allowedDataTiers:
            msg = f"{self.dataTier} isn't an allowed datatier for repacking. Allowed data tiers: {allowedDataTiers}"
            raise RuntimeError(msg)

        outputs = []
        outputs.append( { 'moduleLabel' : f"write_PrimDS1_{self.dataTier}" } )
        outputs.append( { 'moduleLabel' : f"write_PrimDS2_{self.dataTier}" } )
        if self.selectEvents != None:
            outputs[0]['selectEvents'] = self.selectEvents.split(',')
            outputs[1]['selectEvents'] = self.selectEvents.split(',')

        try:
            process = repackProcess(outputs = outputs, dataTier = self.dataTier)
        except Exception as ex:
            msg = "Error creating process for Repack:\n"
            msg += str(ex)
            raise RuntimeError(msg)

        process.source.fileNames.append(self.inputLFN)

        import FWCore.ParameterSet.Config as cms

        process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

        psetFile = open("RunRepackCfg.py", "w")
        psetFile.write(process.dumpPython())
        psetFile.close()
        cmsRun = "cmsRun -e RunRepackCfg.py"
        print("Now do:\n%s" % cmsRun)
        
                


if __name__ == '__main__':
    valid = ["select-events=", "lfn=", "data-tier="]
             
    usage = \
"""
RunRepack.py <options>

Where options are:
 --select-events (option, event selection based on trigger paths)
 --lfn=/store/input/lfn
 --data-tier=RAW|HLTSCOUT|L1SCOUT

Example:
python RunRepack.py --select-events HLT:path1,HLT:path2 --lfn /store/whatever --data-tier RAW|HLTSCOUT|L1SCOUT

"""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", valid)
    except getopt.GetoptError as ex:
        print(usage)
        print(str(ex))
        sys.exit(1)


    repackinator = RunRepack()

    for opt, arg in opts:
        if opt == "--select-events":
            repackinator.selectEvents = arg
        if opt == "--lfn" :
            repackinator.inputLFN = arg
        if opt == "--data-tier" :
            repackinator.dataTier = arg

    repackinator()


