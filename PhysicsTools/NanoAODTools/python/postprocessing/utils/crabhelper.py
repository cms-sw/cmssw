import os
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import *
import sys
import re
import PSet


def inputFiles():
    print("ARGV: " + str(sys.argv))
    JobNumber = sys.argv[1]
    crabFiles = PSet.process.source.fileNames
    print(crabFiles)
    firstInput = crabFiles[0]
    tested = False
    forceaaa = False
    print("--------- using edmFileUtil to convert PFN to LFN --------------")
    for i in range(0, len(crabFiles)):
        if os.getenv("GLIDECLIENT_Group", "") != "overflow" and os.getenv("GLIDECLIENT_Group", "") != "overflow_conservative" and not forceaaa:
            print("Data is local")
            pfn = os.popen("edmFileUtil -d %s" % (crabFiles[i])).read()
            pfn = re.sub("\n", "", pfn)
            print(str(crabFiles[i]) + "->" + str(pfn))
            if not tested:
                print("Testing file open")
                import ROOT
                testfile = ROOT.TFile.Open(pfn)
                if testfile and testfile.IsOpen():
                    print("Test OK")
                    crabFiles[i] = pfn
                    testfile.Close()
                    # tested=True
                else:
                    print("Test open failed, forcing AAA")
                    crabFiles[i] = "root://cms-xrd-global.cern.ch/" + \
                        crabFiles[i]
                    forceaaa = True
            else:
                crabFiles[i] = pfn

        else:
            print("Data is not local, using AAA/xrootd")
            crabFiles[i] = "root://cms-xrd-global.cern.ch/" + crabFiles[i]
    return crabFiles


def runsAndLumis():
    if hasattr(PSet.process.source, "lumisToProcess"):
        lumis = PSet.process.source.lumisToProcess
        runsAndLumis = {}
        for l in lumis:
            if "-" in l:
                start, stop = l.split("-")
                rstart, lstart = start.split(":")
                rstop, lstop = stop.split(":")
            else:
                rstart, lstart = l.split(":")
                rstop, lstop = l.split(":")
            if rstart != rstop:
                raise Exception(
                    "Cannot convert '%s' to runs and lumis json format" % l)
            if rstart not in runsAndLumis:
                runsAndLumis[rstart] = []
            runsAndLumis[rstart].append([int(lstart), int(lstop)])
        print("Runs and Lumis: " + str(runsAndLumis))
        return runsAndLumis
    return None
