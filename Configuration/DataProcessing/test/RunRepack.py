#!/usr/bin/env python
"""
_RunRepack_

Test/Debugging harness for the repack configuration builder

"""

import sys
import getopt

from Configuration.DataProcessing.Repack import repackProcess


class RunRepack:

    def __init__(self):
        self.selectEvents = None
        self.inputLFN = None

    def __call__(self):
        if self.inputLFN == None:
            msg = "No --lfn specified"
            raise RuntimeError, msg

        outputs = []
        outputs.append( { 'moduleLabel' : "write_PrimDS1_RAW" } )
        outputs.append( { 'moduleLabel' : "write_PrimDS2_RAW" } )
        if self.selectEvents != None:
            outputs[0]['selectEvents'] = self.selectEvents.split(',')
            outputs[1]['selectEvents'] = self.selectEvents.split(',')

        try:
            process = repackProcess(outputs = outputs)
        except Exception, ex:
            msg = "Error creating process for Repack:\n"
            msg += str(ex)
            raise RuntimeError, msg

        process.source.fileNames.append(self.inputLFN)

        import FWCore.ParameterSet.Config as cms

        process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

        psetFile = open("RunRepackCfg.py", "w")
        psetFile.write(process.dumpPython())
        psetFile.close()
        cmsRun = "cmsRun -e RunRepackCfg.py"
        print "Now do:\n%s" % cmsRun
        
                


if __name__ == '__main__':
    valid = ["select-events=", "lfn="]
             
    usage = \
"""
RunRepack.py <options>

Where options are:
 --select-events (option, event selection based on trigger paths)
 --lfn=/store/input/lfn

Example:
python RunRepack.py --select-events HLT:path1,HLT:path2 --lfn /store/whatever

"""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", valid)
    except getopt.GetoptError, ex:
        print usage
        print str(ex)
        sys.exit(1)


    repackinator = RunRepack()

    for opt, arg in opts:
        if opt == "--select-events":
            repackinator.selectEvents = arg
        if opt == "--lfn" :
            repackinator.inputLFN = arg

    repackinator()
