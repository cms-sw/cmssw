#!/usr/bin/env python

import json
import argparse
import subprocess
import multiprocessing
from pprint import pprint
from dbs.apis.dbsClient import DbsApi
from random import shuffle
import time
import os

def parser():
    parser = argparse.ArgumentParser(description='Create json config files for your defined IOV')
    
    parser.add_argument("--json-input", type = str, help = "Input json file", default = {})
    parser.add_argument("--data-txt", type = str, help = "Txt file with data set names", required = True)
    parser.add_argument("--N-max-IOV", type = int, help = "Maximum number of events per IOV", default = 1e20) 
    parser.add_argument("--rm-bad-runs", type = str, help = "Remove bad runs from json config")
    parser.add_argument("--iov-txt", type = str, help = "Txt file with IOV boundaries", default = [])
    parser.add_argument("--out-data", type = str, help = "Name of skimmed file with list of data file names", default = "skimmed_dataset")
    parser.add_argument("--out-dir", type = str, help = "Output dir name", default = "configs_" + "_".join([str(time.localtime()[i]) for i in range(6)]))


    return parser.parse_args()


##Called in fillJson function in parallel
def getFileInfo(filename):
    print "Processing: {}".format(filename)

    ##Get file info
    try:
        edmFileUtilArgs = ['edmFileUtil', '-f', filename, '--eventsInLumis']
        fullRunInfo = subprocess.check_output(edmFileUtilArgs).split()[14:]
        runInfo = [tuple(fullRunInfo[index:index+3]) for index in range(0, len(fullRunInfo), 3)]

    ##File not at CERN
    except:
        print "Not at CERN {}".format(filename)
        runInfo = filename

    return runInfo

##Called in paralell in the main function
def getFileList(dataset):
##File list which will contain skimmed file names
    filelist = []
    emptyfiles = []
    nEvents = 0

    ##Find files in dataset
    dbs = DbsApi('https://cmsweb.cern.ch/dbs/prod/global/DBSReader')

    print "Processing: {}".format(dataset)
    sites = subprocess.check_output(["dasgoclient", "--query", "site dataset={}".format(dataset)]).split()

    if "T2_CH_CERN" in sites:
        for f in dbs.listFileArray(dataset=dataset.replace("\n", ""), detail=1):
            filename = f['logical_file_name']
            nevents = f['event_count']

            if nevents != 0:    
                filelist.append(filename)
                nEvents += f['event_count']

            else:
                emptyfiles.append(filename)

    else:
        print "Not at CERN {}".format(dataset)

    return filelist, emptyfiles, nEvents


def fillJson(runJson, listIOV, filelist, nMax, outDir):
    ##Function for finding run in IOV intervall
    sort = lambda lower, run, upper: lower < int(run) < upper

    ##Boundaries of IOVS
    if listIOV:
        lowerBoundaries = [int(run) for run in listIOV[:-1]]
        upperBoundaries = [int(run)-1 for run in listIOV[1:]]

    else:
        lowerBoundaries = [0.]
        upperBoundaries = [1e20]
   
    ##Get file information (run number, events) in paralell
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = [pool.apply_async(getFileInfo, (filename,)) for filename in filelist]
    output = [result.get() for result in results]

    fileInfo = [result for result in output if type(result) == list]
    notAtCern = [result for result in output if type(result) == str]

    ##Write out files which are not at CERN
    with open("{}/filesNotAtCern.txt".format(outDir), "w") as filesNotCern:
        for filename in notAtCern:
            filesNotCern.write(filename)
            filesNotCern.write("\n")

    runDic = {}

    ##Fill dic like {runNumber: {lumi: (events, filenames), lumi2: (...)}}
    for (runInfo, filename) in zip(fileInfo, filelist):
        for (run, lumi, events) in runInfo:
            if events not in ["Events", "Lumi"]:
                try:
                    runDic[int(run)][int(lumi)] = (int(events), filename)

                except KeyError:
                    runDic[int(run)] = {int(lumi): (int(events), filename)}

    ##json configs for IOV
    jsonAlign = [{} for index in lowerBoundaries]
    jsonVali = [{} for index in lowerBoundaries]
    eventsInTotal = [0 for index in lowerBoundaries]
    eventsInAlign = [0 for index in lowerBoundaries]
    eventsInVali = [0 for index in lowerBoundaries]

    ##Shuffle runJson to have random run number position
    if runJson:
        runJson = runJson.items()
        shuffle(runJson)
        filelist = {}

    else:
        return jsonAlign, jsonVali, set(filelist)

    ##Loop over json input file
    for (run, value) in runJson:
        try:
            ##Check if run is in IOV boundaries and check in which IOV
            index = [sort(lower, run, upper) for (lower, upper) in zip(lowerBoundaries, upperBoundaries)].index(True)
        
            ##Check if run is one of files
            if int(run) in runDic:
                alignLumi = [[]]
                valiLumi = [[]]

                ##Loop over all lumi section of a run
                for (lumi, lumiInfo) in runDic[int(run)].iteritems():
                    eventsInTotal[index] += lumiInfo[0]

                    ##Add events from lumi section
                    if eventsInAlign[index] < nMax:
                        if not True in [sort(lower, lumi, upper) for lower, upper in value]:
                            if len(alignLumi[-1]) != 0:
                                alignLumi.append([])
                            continue

                        eventsInAlign[index] += lumiInfo[0]
                        filelist.setdefault(index, set()).add(lumiInfo[1])

                        if len(alignLumi[-1]) == 0:
                            alignLumi[-1] = [lumi, lumi]

                        else:
                            alignLumi[-1][1] = lumi

                    else:
                        if not True in [sort(lower, lumi, upper) for lower, upper in value]:
                            if len(valiLumi[-1]) != 0:
                                valiLumi.append([])
                            continue

                        eventsInVali[index] += lumiInfo[0]
                        if len(valiLumi[-1]) == 0:
                            valiLumi[-1] = [lumi, lumi]

                        else:
                            valiLumi[-1][1] = lumi

                alignLumi = [element for element in alignLumi if len(element) != 0]
                valiLumi = [element for element in valiLumi if len(element) != 0]

                if len(alignLumi) != 0:
                    jsonAlign[index][str(run)] = alignLumi

                if len(valiLumi) != 0:
                    jsonVali[index][str(run)] = valiLumi
                        

        except ValueError:
            ##run of json file is not in IOV boundaries
            pass

    
    ##Write out events for Alignment/Validation
    with open("{}/eventsUsed.txt".format(outDir), "w") as eventsUsed:
        for index in range(len(eventsInTotal)):
            eventsUsed.write("Events used in Total for IOV {}: {}".format(lowerBoundaries[index], eventsInTotal[index]) + "\n")
            eventsUsed.write("Events used for Alignment for IOV {}: {}".format(lowerBoundaries[index], eventsInAlign[index]) + "\n")
            eventsUsed.write("Events used for Validation for IOV {}: {}".format(lowerBoundaries[index], eventsInVali[index]) + "\n")

    return jsonAlign, jsonVali, filelist
        

def main():
    ##Get parser arguments
    args = parser()

    ##create dir for all the configs
    os.system("mkdir -p {}".format(args.out_dir))

    ##Read out files from datasets which are at CERN in parallel
    filelist = []
    emptyfiles = []
    nEvents = []
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    with open(args.data_txt, "r") as datasets:
        results = [pool.apply_async(getFileList, (dataset.replace("\n", ""),)) for dataset in datasets.readlines()]
    
    for result in results:
        files, empties, events = result.get()
        filelist.extend(files)
        emptyfiles.extend(empties)
        nEvents.append(events)

    with open("{}/emptyFiles.txt".format(args.out_dir), "w") as empty:
        for emptyFile in emptyfiles:
            empty.write(emptyFile + '\n')

    ##Load IOV boundaries
    if args.iov_txt:
        with open(args.iov_txt) as fIOV:
            listIOV = [line.strip() for line in fIOV]

    else:
        listIOV = args.iov_txt

    ##Load json file   
    if args.json_input:
        with open(args.json_input) as fJson:
            runJson = json.load(fJson)

    else:
        runJson = args.json_input

    ##Fill json configs
    jsonAlign, jsonVali, filelist = fillJson(runJson, listIOV, filelist, args.N_max_IOV, args.out_dir)

    ##Remove bad runs if wished
    if args.rm_bad_runs != None:
        with open(args.rm_bad_runs, "r") as badRuns:
            for badRun in badRuns:
                for dic in jsonAlign:
                    dic.pop(int(badRun), None)

                for dic in jsonVali:
                    dic.pop(int(badRun), None)


    ##Template for python configuration files with file names for each IOV
    pyTempl = """import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList

lumiSecs = cms.untracked.VLuminosityBlockRange()
goodLumiSecs = LumiList.LumiList(filename = '{json}').getCMSSWString().split(',')
readFiles = cms.untracked.vstring()
source = cms.Source("PoolSource",
                            lumisToProcess = lumiSecs,
                            fileNames = readFiles)
readFiles.extend([
    {filenames}
])
lumiSecs.extend(goodLumiSecs)
maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
    """

    ##Write out skimmed file set:
    if not args.iov_txt:
        with open("{}/{}.txt".format(args.out_dir, args.out_data), "w") as outData:
            for filename in filelist:
                outData.write(filename + '\n')
    
    ##Write json IOV files if wished
    if args.iov_txt and args.json_input:
        for index, (jsonContent, runNumber) in enumerate(zip(jsonAlign, [int(run) for run in listIOV[:-1]])):
            with open("{}/IOV_Align_{}.json".format(args.out_dir, runNumber), "w") as fAlignJson:
                json.dump(jsonContent, fAlignJson, sort_keys=True, indent=4, separators=(',', ': '))

        for (jsonContent, runNumber) in zip(jsonVali, [int(run) for run in listIOV[:-1]]):
            with open("{}/IOV_Vali_{}.json".format(args.out_dir, runNumber), "w") as fValiJson:
                json.dump(jsonContent, fValiJson, sort_keys=True, indent=4, separators=(',', ': '))

            with open("{}/{}_since{}_cff.py".format(args.out_dir, args.out_data, runNumber), "w") as outData:
                outData.write(pyTempl.format(json=os.path.abspath("{}/IOV_Vali_{}.json".format(args.out_dir, runNumber)), filenames=",\n".join(["'{}'".format(filename) for filename in filelist[index]])))

    if args.json_input:
        mergeJsonAlign = {}
        [mergeJsonAlign.update(jsonDic) for jsonDic in jsonAlign]

        mergeJsonVali = {}
        [mergeJsonVali.update(jsonDic) for jsonDic in jsonVali]

        with open("{}/Align.json".format(args.out_dir, runNumber), "w") as fAlignJson:
            json.dump(mergeJsonAlign, fAlignJson, sort_keys=True, indent=4, separators=(',', ': '))

        with open("{}/Vali.json".format(args.out_dir, runNumber), "w") as fValiJson:
            json.dump(mergeJsonVali, fValiJson, sort_keys=True, indent=4, separators=(',', ': '))

    if not os.path.exists("{}/eventsUsed.txt".format(args.out_dir)):
        with open("{}/eventsUsed.txt".format(args.out_dir), "w") as eventsUsed:
            eventsUsed.write("Events used for Alignment: {}".format(sum(nEvents)) + "\n")
            eventsUsed.write("Events used for Validation: {}".format(0) + "\n")

if __name__ == "__main__":
    main()
