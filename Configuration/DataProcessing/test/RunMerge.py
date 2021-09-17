#!/usr/bin/env python3
"""
_RunMerge_

Test/Debugging harness for the merge configuration builder

"""
from __future__ import print_function



import sys
import getopt

from Configuration.DataProcessing.Merge import mergeProcess


class RunMerge:

    def __init__(self):
        self.processName = "Merge"
        self.outputFile = "Merged.root"
        self.outputLFN = None
        self.inputFiles = []
        self.newDQMIO = False
        self.mergeNANO = False
        self.bypassVersionCheck = False
        

    def __call__(self):
        if self.inputFiles == []:
            msg = "No Input Files provided"
            raise RuntimeError(msg)

        try:
            process = mergeProcess(
                self.inputFiles,
                process_name = self.processName,
                output_file = self.outputFile,
                output_lfn = self.outputLFN,
                newDQMIO = self.newDQMIO,
                mergeNANO = self.mergeNANO,
                bypassVersionCheck = self.bypassVersionCheck)
        except Exception as ex:
            msg = "Error creating process for Merge:\n"
            msg += str(ex)
            raise RuntimeError(msg)

        psetFile = open("RunMergeCfg.py", "w")
        psetFile.write(process.dumpPython())
        psetFile.close()
        cmsRun = "cmsRun -j FrameworkJobReport.xml RunMergeCfg.py"
        print("Now do:\n%s" % cmsRun)
        
                


if __name__ == '__main__':
    valid = ["input-files=", "output-file=", "output-lfn=", "dqmroot", "mergeNANO", "bypassVersionCheck" ]
             
    usage = """RunMerge.py <options>"""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", valid)
    except getopt.GetoptError as ex:
        print(usage)
        print(str(ex))
        sys.exit(1)


    merger = RunMerge()

    for opt, arg in opts:
        if opt == "--input-files":
            merger.inputFiles = [
                x for x in arg.split(',') if x.strip() != '' ]
            
        if opt == "--output-file" :
            merger.outputFile = arg
        if opt == "--output-lfn" :
            merger.outputLFN = arg
        if opt == "--dqmroot" :
            merger.newDQMIO = True
        if opt == "--mergeNANO" :
            merger.mergeNANO = True
        if opt == "--bypassVersionCheck" :
            merger.bypassVersionCheck = True

    merger()
