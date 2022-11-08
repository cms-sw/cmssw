#!/usr/bin/env python3

import Utilities.General.cmssw_das_client as das_client
import json
import os
import sys
import subprocess
import argparse

##############################################
def parseArguments():
##############################################
    """Parse the control line arguments"""

    parser = argparse.ArgumentParser(description = "Tool to find which runs are included in files. Used to generate input dataset for JetHT validation tool in case of run based splitting for condor jobs.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--input", action="store", help="Name of the input file list. Has one file name in each line.", required=True)
    parser.add_argument("-o", "--output", action = "store", help ="Name of the output file in which the produced file list is stored", default = "myFileListWithRuns.txt")

    return parser.parse_args()


##############################################
def check_proxy():
##############################################
    """Check if GRID proxy has been initialized."""

    try:
        with open(os.devnull, "w") as dump:
            subprocess.check_call(["voms-proxy-info", "--exists"],
                                  stdout = dump, stderr = dump)
    except subprocess.CalledProcessError:
        return False
    return True

##############################################
def findInJson(jsondict, strings):
##############################################
    """ Find string from json file. Code copy-pasted from dataset.py  """

    if isinstance(strings, str):
        strings = [ strings ]

    if len(strings) == 0:
        return jsondict
    if isinstance(jsondict,dict):
        if strings[0] in jsondict:
            try:
                return findInJson(jsondict[strings[0]], strings[1:])
            except KeyError:
                pass
    else:
        for a in jsondict:
            if strings[0] in a:
                try:
                    return findInJson(a[strings[0]], strings[1:])
                except (TypeError, KeyError):  #TypeError because a could be a string and contain strings[0]
                    pass
    #if it's not found
    raise KeyError("Can't find " + strings[0])

##############################################
def getData( dasQuery, dasLimit = 0 ):
##############################################
    """ Get data from DAS query. Code copy-pasted from dataset.py """

    dasData = das_client.get_data(dasQuery, dasLimit)
    if isinstance(dasData, str):
        jsondict = json.loads( dasData )
    else:
        jsondict = dasData
    # Check, if the DAS query fails
    try:
        error = findInJson(jsondict,["data","error"])
    except KeyError:
        error = None
    if error or findInJson(jsondict,"status") != 'ok' or "data" not in jsondict:
        try:
            jsonstr = findInJson(jsondict,"reason")
        except KeyError: 
            jsonstr = str(jsondict)
        if len(jsonstr) > 10000:
            jsonfile = "das_query_output_%i.txt"
            i = 0
            while os.path.lexists(jsonfile % i):
                i += 1
            jsonfile = jsonfile % i
            theFile = open( jsonfile, "w" )
            theFile.write( jsonstr )
            theFile.close()
            msg = "The DAS query returned an error.  The output is very long, and has been stored in:\n" + jsonfile
        else:
            msg = "The DAS query returned a error.  Here is the output\n" + jsonstr
        msg += "\nIt's possible that this was a server error.  If so, it may work if you try again later"
        raise KeyError(msg)
    return findInJson(jsondict,"data")

##############################################
def main():
##############################################
    """ Main program """

    # Before doing anything, check that grip proxy exists
    if not check_proxy():
        print("Grid proxy is required to connect to DAS. Cannot run the tool without it.")
        print("Please create a proxy via 'voms-proxy-init -voms cms'.")
        sys.exit(1)

    # Read the command line argument
    commandLineArguments = parseArguments()

    # Read the file list from the input file
    inputFile = open(commandLineArguments.input,"r")
    inputFileList = inputFile.readlines()
    inputFile.close()

    # Find which runs are included in each of the files in the file list
    runDictionary = {}  # Dictionary telling which files contain each run
    for rawInputFile in inputFileList:

        inputFile = rawInputFile.rstrip()
        myData = getData("run file={}".format(inputFile))

        myRunsArray = []
        for dasInstance in myData:
            myRunsArray.append(findInJson(dasInstance,"run"))

        for innerArray in myRunsArray:
            for jsonDictionary in innerArray:
                runNumber = jsonDictionary["run_number"]
                if runNumber in runDictionary:
                    runDictionary[runNumber].append(inputFile)
                else:
                    runDictionary[runNumber] = [inputFile]


    # Create an output file indicating which runs can be found from each file
    outputFileName = commandLineArguments.output
    outputFile = open(outputFileName, "w")

    for runNumber in runDictionary:
        for fileName in runDictionary[runNumber]:
            outputFile.write("{} {}\n".format(runNumber, fileName))

    outputFile.close()

##############################################
if __name__ == "__main__":
##############################################
    main()
