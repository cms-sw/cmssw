#!/usr/bin/env python3
import sys
argv = sys.argv
sys.argv = argv[:1]

import argparse
import os
import json
import yaml
import csv
import ROOT

def log(log_type="",text=""):
    #########################################################################################################################
    #Logger:
    #  INFO    = Informative text
    #  WARNING = Notify user about unpredictable changes or missing files which do not result in abort
    #  ERROR   = Error in logic results in abort. Can be fixed by user (missing input, settings clash ...)
    #  FATAL   = Fatal error results in abort. Cannot be fixed by user (the way how input is produced has changed or bug ...)
    #########################################################################################################################

    v = int(sys.version_info[0])
    source = "mkLumiAveragedPlots:       "
    text = str(text)
    if v == 3:
        if "i" in log_type:
            print(source,"[INFO]     ",text)
        elif "n" in log_type:
            print("                  ",text)
        elif "w" in log_type:
            print(source,"[WARNING]  ",text)
        elif "e" in log_type:
            print(source,"[ERROR]    ",text)
        elif "f" in log_type:
            print(source,"[FATAL]    ",text)
        else:
            print(text) 

def isNumber(this):
    try:
        int(this)
        return True
    except ValueError:
        return False 

def decodeLine(line,index,type):
    if "txt" in type:
        return line.strip('\n').split(" ")[index]
    elif "csv" in type:
        return line.strip('\r\n').replace("\t"," ").split(" ")[index]

def getLumiPerIoV(start_run,end_run = 0):
    #################################
    #Returns luminosity per IoV
    lumi = 0.0
    foundStartRun = False
    if end_run > 0:
        for lumifile in lumiPerRun:
            f = open(lumifile, "r")
            for line in f:
                if int(decodeLine(line,0,lumifile)) in range(start_run,end_run):
                    #log(line.strip('\n').split(" ")[0],lumi,line.strip('\n').split(" ")[1])
                    lumi += float(decodeLine(line,1,lumifile))
                if int(decodeLine(line,0,lumifile)) == start_run:
                    foundStartRun = True 
            f.close()
    elif end_run < 0:
        for lumifile in lumiPerRun:
            f = open(lumifile, "r")
            for line in f:
                if int(decodeLine(line,0,lumifile)) >= start_run:
                    #log(line.strip('\n').split(" ")[0],lumi,line.strip('\n').split(" ")[1])
                    lumi += float(decodeLine(line,1,lumifile)) 
                if int(decodeLine(line,0,lumifile)) == start_run:
                    foundStartRun = True   
            f.close()  
    elif end_run == 0: 
        for lumifile in lumiPerIoV:
            f = open(lumifile, "r")
            for line in f:
                first = decodeLine(line,0,lumifile)
                try:
                    if int(first) == start_run:
                        lumi = float(decodeLine(line,1,lumifile))
                        foundStartRun = True 
                except ValueError:
                    if str(first) == str(start_run):
                        lumi = float(decodeLine(line,1,lumifile))
                        foundStartRun = True
            f.close()

    if not foundStartRun:
        if lumi == 0.0:
            log("w","Lumi per IoV: "+str(start_run)+" not found.")
            return 11112
    return lumi

def getLumiPerRun(run):
    ###############################################################
    #Return luminosity per run as stored in given lumiPerRun files.
    #Return code 11112 if run number was not found in the list.
    ###############################################################

    lumi = 0.0
    foundRun = False
    for lumifile in lumiPerRun:
        f = open(lumifile, "r")
        for line in f:
            if int(decodeLine(line,0,lumifile)) == run:
                #log(line.strip('\n').split(" ")[0],lumi,line.strip('\n').split(" ")[1])
                lumi = float(decodeLine(line,1,lumifile)) 
                foundRun = True
        f.close() 
    if not foundRun: 
        log("w","Lumi per run: "+str(run)+" not found.")
        return 11112
    return lumi 

def getTuples(inputDir, filterNumbers=[]):
    ########################################
    #This applies for DATA only.
    #PV:
    #DMR:
    ######################################## 

    tuples = []
    if isDMR:
        valid_files = 0
        for dirpath,dirs,files in os.walk(inputDir):
            if len(dirs) == 0: dirs = [""] #needed for finalization jobs 
            for n_dir in dirs:
                if len(filterNumbers)!=0 and (n_dir not in filterNumbers or not isNumber(n_dir)): continue
                _number = 0
                if isNumber(n_dir):
                    _number = int(n_dir)
                    n_dir = n_dir+"/"
                else:
                    if len(n_dir) > 0: 
                        _number = n_dir  
                        n_dir = n_dir+"/" 
                    else: 
                        _number = os.path.join(dirpath,"OfflineValidationSummary.root")
                if os.path.isfile(os.path.join(dirpath,n_dir+"OfflineValidationSummary.root")): 
                    test_root = ROOT.TFile(os.path.join(dirpath,n_dir+"OfflineValidationSummary.root"),"READ")
                    if test_root.IsZombie():
                        log("w",os.path.join(dirpath,n_dir+"OfflineValidationSummary.root")+" is ZOMBIE!")
                        tuples.append({'file' : "invalid.root",
                                       'number' : _number,
                                       'lumi' : 11111
                                     })
                        continue
                    test_root.Close()
                    tuples.append({'file' : os.path.join(dirpath,n_dir+"OfflineValidationSummary.root"),
                                   'number' : _number,
                                   'lumi' : 0 
                                 })
                    valid_files += 1
                else:
                    log("w",os.path.join(dirpath,n_dir+"OfflineValidationSummary.root")+" NOT found! Directory is empty?")
                    tuples.append({'file' : "invalid.root",
                                   'number' : _number,
                                   'lumi' : 11111
                                 }) 
            if (valid_files < 2 and config['mode'] != "finalize") or (valid_files < 1 and config['mode'] == "finalize"):
                log("f","Check input directory. Less than two valid files to merge recognized.")
                sys.exit(0)   
    
    elif isPV:
        for dirpath,dirs,files in os.walk(inputDir):
            for n_file in files:
                if "PVValidation" in n_file: #if pass this condition then file exists TODO: check for zombie
                    if isNumber((n_file.split("_")[-1]).split(".")[0]) and os.path.isfile(os.path.join(dirpath,n_file)):
                        tuples.append({'file' : os.path.join(dirpath,n_file),
                                       'number' : int((n_file.split("_")[-1]).split(".")[0]),
                                       'lumi' : 0
                                     })
                    else:
                        log("f","Format for run by run PV results NOT recognised!")
                        sys.exit(0)
       
    #sort list based on IoV number and calculate luminosity per IoV (from lumiPerRun or directly from lumiPerIoV)
    if config['mode'] != "finalize":
        tuples.sort(key=lambda tuple: tuple['number'])
    if len(lumiPerRun)!=0:
        if isDMR:
            valid_tuples = []
            for ituple,tuple in enumerate(tuples):
                if ituple != len(tuples)-1 or len(config['validation']['firstFromNext']) != 0:
                    if int(tuple['lumi']) != 11111: #input is not empty/zombie
                        if ituple == len(tuples)-1:
                            tuple['lumi'] += getLumiPerIoV(tuples[ituple]['number'],int(config['validation']['firstFromNext'][0]))
                        else:  
                            tuple['lumi'] += getLumiPerIoV(tuples[ituple]['number'],tuples[ituple+1]['number'])  
                    else:
                        _ituple = ituple
                        while _ituple > 0 and int(tuples[_ituple-1]['lumi']) in [11111,11112]:
                            _ituple -= 1
                        if _ituple != 0:
                            if ituple == len(tuples)-1:
                                dLumi = getLumiPerIoV(tuples[ituple]['number'],int(config['validation']['firstFromNext'][0]))
                            else: 
                                dLumi = getLumiPerIoV(tuples[ituple]['number'],tuples[ituple+1]['number'])
                            if int(dLumi) not in [11111,11112]: #TODO loop until dLumi is OK 
                                tuples[_ituple-1]['lumi'] += dLumi
                            else:
                                dLumi = 0. 
                            log("w","Effectively adding luminosity of IoV: "+str(tuples[ituple]['number'])+"(missing file with "+str(dLumi)+") to IoV: "+str(tuples[_ituple-1]['number']))
                        else:
                            __ituple = ituple
                            while __ituple < len(tuples)-1 and int(tuples[__ituple+1]['lumi']) in [11111,11112]:
                                __ituple += 1
                            if __ituple !=  len(tuples)-1:
                                dLumi = getLumiPerIoV(tuples[ituple]['number'],tuples[ituple+1]['number'])
                                if int(dLumi) not in [11111,11112]: #TODO loop until dLumi is OK
                                    tuples[__ituple+1]['lumi'] += dLumi
                                else:
                                    dLumi = 0. 
                                log("w","Effectively adding luminosity of IoV: "+str(tuples[ituple]['number'])+"(missing file with "+str(dLumi)+") to IoV: "+str(tuples[__ituple+1]['number']))
                else:
                    if skipLast:
                        log("i","User requested to skip IoV: "+str(tuples[ituple]['number']))
                    else:
                        if int(tuple['lumi']) != 11111: #input is not empty/zombie
                            tuple['lumi'] += getLumiPerIoV(tuples[ituple]['number'],-1)
                        else:
                            _ituple = ituple
                            while _ituple > 0 and int(tuples[_ituple-1]['lumi']) in [11111,11112]:
                                _ituple -= 1
                            if _ituple != 0:
                                dLumi = getLumiPerIoV(tuples[ituple]['number'],-1) 
                                if int(dLumi) not in [11111,11112]: #TODO loop until dLumi is OK
                                    tuples[_ituple-1]['lumi'] += dLumi
                                else:
                                    dLumi = 0.  
                                log("w","Effectively adding luminosity of IoV: "+str(tuples[ituple]['number'])+"(missing file with "+str(dLumi)+") to IoV: "+str(tuples[_ituple-1]['number']))
                            else:
                                log("w","No more IOVs in the list to add luminosity from missing IOV"+str(tuples[ituple]['number'])+".")

            if skipLast:
                valid_tuples = [ tuple for ituple,tuple in enumerate(tuples) if int(tuple['lumi']) not in [11111,11112] and ituple != len(tuples)-1 ]
            else:
                valid_tuples = [ tuple for tuple in tuples if int(tuple['lumi']) not in [11111,11112] ]   
            tuples = valid_tuples
 
        elif isPV:
            for ituple,tuple in enumerate(tuples):
                tuple['lumi'] = getLumiPerRun(tuple['number'])
           
    elif len(lumiPerIoV)!=0 and isDMR:
        #This will work for finalization jobs as well  
        valid_tuples = []
        for ituple,tuple in enumerate(tuples):
            if tuple['lumi'] != 11111: #empty input will not contribute to total average
                tuple['lumi'] = getLumiPerIoV(tuple['number'])
                valid_tuples.append(tuple)
        tuples = valid_tuples 

    return tuples

def getTuplesMC(inputDir):
    #####################################################################
    #This applies for MC only.
    #Define different periods inside input object directory:
    #PV: PVValidation_<period>_<part>.root
    #DMR: OfflineValidationSummary.root (period hidden in subdirectory)
    #####################################################################

    tuples = {}
    if isPV:
        for dirpath,dirs,files in os.walk(inputDir):
            for file in files:
                if "PVValidation" in file:
                    if len(file.split("_")) == 3:
                        period = file.split("_")[1]
                        if period not in tuples: 
                            tuples[period] = []
                            tuples[period].append({ 'file' : os.path.join(dirpath,file),
                                                    'lumi' : 1
                                                 })
                        else:
                            tuples[period].append({ 'file' : os.path.join(dirpath,file),
                                                    'lumi' : 1
                                                 }) 
                    elif len(file.split("_")) == 2:
                        if ".root" in file.split("_")[1]:
                            period = file.split("_")[1].replace(".root","")
                            if period not in tuples:
                                tuples[period] = []  
                                tuples[period].append({ 'file' : os.path.join(dirpath,file),
                                                        'lumi' : 1
                                                     })
                            else:
                                tuples[period].append({ 'file' : os.path.join(dirpath,file),
                                                        'lumi' : 1
                                                     }) 
                        else:
                            log("e","No extension found for PV MC input files.")
                            sys.exit(0)    
                    else:
                        log("w","Incorrect format for <period> tag inside PV MC input-file names. Char \"_\" is not allowed inside tag.")
    elif isDMR:
        inputFile = os.path.join(inputDir,"OfflineValidationSummary.root")
        period = (inputDir.split("/")[-2])
        tuples[period] = []
        if os.path.isfile(inputFile): 
            tuples[period].append({ 'file' : inputFile, 
                                   'lumi' : 1})
        else:
            tuples[period].append({ 'file' : "invalid.root", 
                                    'lumi' : 11111}) 

    return tuples    
            

def makeAveragedFile(tuples,intLumi,objName=""):
    ####################################################################
    #Use (compiled) haddws executable to weight every histogram in given
    #rootfile by particular weight = lumi/intLumi and merge histograms 
    #for each run/IoV. Skip files with 0.0 lumi (uncertified runs).
    ####################################################################

    fileArgument = ""
    weightArgument = ""
    haddwsDir = ""
    #cmd_bare = ""

    countTest = 0.0
    for ituple,tuple in enumerate(tuples):
        if str(tuple['lumi']) not in ['11111','11112']: #will not include 11111=non-existing input,11112=non-existing lumi record
            if tuple['lumi']/intLumi > -1.:#1e-6: #will not include 'almost zero' lumi input (not digestible by haddws)  
                fileArgument += tuple['file']+" "
                weightArgument += str(format(tuple['lumi']/intLumi,'.20f'))+" "
                #cmd_bare += tuple['file']+" "+str(format(tuple['lumi']/intLumi,'.20f'))+"\n" 
                countTest += tuple['lumi']/intLumi
        else:
            log("i","Not including IOV "+str(tuple['number'])+" for weighting: "+str(tuple['lumi']))

    if countTest < 1.0:
        log("w","Normalization factor: "+str(format(countTest,'.20f'))+" is less than 1.")
    if len(weightArgument.split(" ")) != len(fileArgument.split(" ")):
        log("e","There is "+str(len(fileArgument.split(" ")))+"rootfiles but "+str(len(weightArgument.split(" ")))+" weights.")
    else:
        log("i","Real number of files to merge is: "+str(len(fileArgument.split(" "))-1)) 

    #f = open("haddws_command.txt", "w")
    #f.write(cmd_bare)
    #f.close()
    #sys.exit(0)

    cmd = haddwsDir+"haddws "+fileArgument+weightArgument+" > haddws.out 2> haddws.err"
    log("i","Running haddws.C")
    os.system(cmd)

    #Store result file in output
    outFileName = ""
    if isDMR:
        if objName != "" and not objName.startswith("_"): objName = "_"+objName 
        outFileName = os.path.join(config['output'],"OfflineValidationSummary"+objName+".root") 
    elif isPV:
        outFileName = os.path.join(config['output'],"result"+objName+".root") 
    if os.path.isfile("./result.root"): 
        os.system("mv result.root "+outFileName)

def getIntLumi(tuples):
    ################################
    #Calculate integrated luminosity
    ################################

    intLumi = 0.0
    for tuple in tuples:
        if tuple['lumi'] not in [11111,11112]:  
            intLumi += tuple['lumi'] 
    return intLumi

def parser():
    sys.argv = argv
    parser = argparse.ArgumentParser(description = "run the python plots for the AllInOneTool validations", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("config", metavar='config', type=str, action="store", help="Averager AllInOneTool config (json/yaml format)")
    parser.add_argument("-b", "--batch", action = "store_true", help ="Batch mode")

    #sys.argv.append('-b')
    #ROOT.gROOT.SetBatch()
    return parser.parse_args()
  
if __name__ == '__main__':
    ############################################################
    #Main code:
    #  - parse local config
    #  - run sanity checks
    #  - prepare average file (PV or DMR) using external C macro
    #  - (and) or run plotting macros
    ############################################################

    args = parser()
    with open(args.config, "r") as configFile:
        if args.config.split(".")[-1] == "json":
            config = json.load(configFile)
        elif args.config.split(".")[-1] == "yaml":
            config = yaml.load(configFile, Loader=yaml.Loader)
        else:
            raise Exception("Unknown config extension '{}'. Please use json/yaml format!".format(args.config.split(".")[-1]))

    log(' ----- All-in-one Averager -----')
    log(' type:   '+config['type'])
    log(' mode:   '+config['mode'])
    log(' isData: '+str(config['isData']))
    log(' isMC:   '+str(not config['isData']))
    if config['mode'] == "finalize":
        nFiles = len(config['validation']['mergeFile'])
    elif config['mode'] == "plot":
        nFiles = len(config['plot']['inputData'])+len(config['plot']['inputMC'])
    else: 
        nFiles = len(config['validation']['IOV'])  
    log(' nFiles: '+str(nFiles))
    log(' -------------------------------')

    #BASIC SANITY CHECKS
    ######################## 
    #Input directory checks
    ########################
    inputDirData = []
    inputDirMC   = []
    IOVs         = []
    if config['mode'] == "merge":
        #DATA
        if config['isData']:
            if len(config['validation']['IOV']) == 0:
                log("f","No input DATA found. List of IOVs needs to contain at least one number.")
                sys.exit(0)
            elif len(config['validation']['IOV']) != 0:
                _dir = config['validation']['mergeFile'].replace("{}","")
                if os.path.isdir(_dir):
                    log("i","Will average "+str(len(config['validation']['IOV']))+" DATA files from directory(ies): ")
                    log("i",_dir)
                subDirMissing = False
                for IOV in config['validation']['IOV']:
                    IOVs.append(str(IOV))
                    if not os.path.isdir(config['validation']['mergeFile'].replace("{}",str(IOV))):
                        log("f","Subdir not found "+str(IOV)+"!")
                        subDirMissing = True
                if subDirMissing:
                    sys.exit(0)
                else:
                    inputDirData.append(_dir)
        #MC 
        elif not config['isData']:
            if len(config['validation']['IOV']) != 1 \
            or (len(config['validation']['IOV']) == 1 and str(config['validation']['IOV'][0]) != "1"):
                log("f","MC validation configuration error: IOV!=1")
                sys.exit(0)
            else:
                IOVs.append(str(config['validation']['IOV'][0]))
            if len(config['validation']['mergeFile']) == 0:
                log("f", "No input MC found.")
                sys.exit(0)
            else:
                log("i","Will scale and merge "+str(len(config['validation']['mergeFile']))+" MC directory(ies): ")
                for _dir in config['validation']['mergeFile']:
                    basedir = _dir.replace("{}","")
                    subdir = _dir.replace("{}",str(config['validation']['IOV'][0]))
                    if os.path.isdir(basedir) and os.path.isdir(subdir):
                        log("i",subdir)
                        inputDirMC.append(subdir)
                    else:
                        log("f","Directory not found "+subdir)
                        sys.exit(0)
    elif config['mode'] == "finalize":
        #DATA FINALIZE
        if config['isData']:
            if len(config['validation']['mergeFile']) == 0:
                log("f", "No files found to finalize.")
                sys.exit(0)
            log("i","Will finalize average job for "+str(len(config['validation']['mergeFile']))+" parts:")
            for partFile in config['validation']['mergeFile']:
                if os.path.isdir(partFile): 
                    inputDirData.append(partFile)
                    log("i","---> "+partFile)  
                else: log("w","Missing partial input: "+partFile)
            if len(inputDirData) != len(config['validation']['mergeFile']):
                log("e","Some input was not found for DATA finalization job.")
                sys.exit(0)   
        #NO FINALIZE FOR MC 
        elif not config['isData']:
            log("f", "Nothing to finalize for MC.")
            sys.exit(0)
    elif config['mode'] == "plot":
        if len(config['plot']['inputData'])+len(config['plot']['inputMC']) == 0:
                log("f", "No files found for plotting!")
                sys.exit(0)
        log("i", "Will attempt to plot objects from "+str(nFiles)+" source files.") 
        for inputDir in config['plot']['inputData']:
            if os.path.isdir(inputDir):
                inputDirData.append(inputDir)
        for inputDir in config['plot']['inputMC']:
            if os.path.isdir(inputDir):
                inputDirMC.append(inputDir)  
                  
                
    #######################
    # JSON lumifiles checks
    #######################
    isPV  = (config['type'] == "PV")
    isDMR = (config['type'] == "DMR")
    if config['mode'] != "plot":
        lumiPerRun = config['validation']['lumiPerRun']
        lumiPerIoV = config['validation']['lumiPerIoV']
        lumiMC     = config['validation']['lumiMC']
    else:
        lumiPerRun = []
        lumiPerIoV = []
        lumiMC     = []
    if config['mode'] != "plot":
        if isPV and len(lumiPerIoV)!=0: 
            log("w","Are you sure you have run PV validation in IoV by IoV mode?")
        if (isPV or isDMR) and len(lumiPerRun)==0 and len(lumiPerIoV)==0 and len(inputDirData) != 0:
            log("e","Use option 'lumiPerRun' or 'lumiPerIoV' to specify luminosity files.")
            sys.exit(0)
        if (isPV or isDMR) and len(lumiPerIoV)==0:
            if len(lumiPerRun) != 0:
                if config['mode'] == "finalize":
                    log("i","Integrated Luminosity per intermediate files found.")
                else:
                    log("w","Lumi per run list will be processed to get Lumi per IoV list (applies for DATA).")
            elif len(inputDirMC) != 0 and len(inputDirData) == 0:
                log("i", "MC will be scaled per period to given integrated luminosity.") 
        if len(inputDirMC) != 0 and len(lumiMC) == 0:
            log("w","MC object(s) found on input but lumi per period not specified.")
        _lumiMC = {}
        for formatLumi in lumiMC:
            if "::" in formatLumi:
                try:
                    float(formatLumi.split("::")[-1])
                except ValueError:
                    log("e","Wrong lumi per period for MC formatting. USAGE: <object>::<merge>::<lumi> or <merge>::<lumi> for single MC object (alignment).")
                    sys.exit(0)
                if len(formatLumi.split("::")) == 2:
                    _lumiMC[formatLumi.split("::")[0]] = { 'lumi' : float(formatLumi.split("::")[-1]), 'group' : 0}
                elif len(formatLumi.split("::")) == 3:
                    _lumiMC[formatLumi.split("::")[1]] = { 'lumi' : float(formatLumi.split("::")[-1]), 'group' : formatLumi.split("::")[0]}
            else:
                log("e","Wrong lumi per period for MC formatting. USAGE: <object>::<merge>::<lumi> or <merge>::<lumi> for single MC object (alignment).")
                sys.exit(0)
        lumiMC = _lumiMC
        skipLast = False
        if isDMR and config['isData']: 
            if len(config['validation']['firstFromNext']) == 0: skipLast = True 

    ######################
    #Style optional checks
    ######################
    if config['mode'] == "plot": 
        if len(config['plot']['colors']) < len(config['plot']['objects']):
            log("e","Please specify color code for each object.")
            sys.exit(0)
        if len(config['plot']['styles']) < len(config['plot']['objects']):
            log("e","Please specify line style for each object.")
            sys.exit(0) 
        if len(config['plot']['objects']) != 0:
            for obj in config['plot']['objects']:
                if len(obj.split(" ")) != 1:
                    log("e","No space in object name is allowed. Use \"_\" instead.")
                    sys.exit(0)
        if len(config['plot']['objects']) != 0 and len(config['plot']['labels']) == 0:
            log("i","Object labels will be generated automatically.")
        if 'plotGlobal' not in config.keys():
            log("w","Global plotting settings not found. Fallback to default.")
            config['plotGlobal'] = {}  
 
    ##########################
    #Stat plot settings checks
    ##########################
    if config['mode'] == "plot":
        if config['plot']['showMeanError'] and not config['plot']['showMean']:
            log("w","Cannot show mean error without showing mean.")
            config['plot']['showMeanError'] = False
        if config['plot']['showRMSError'] and not config['plot']['showRMS']:
            log("w","Cannot show RMS error without showing RMS.")
            config['plot']['showRMSError'] = False
        if config['plot']['useFitError'] and not config['plot']['useFit']:
            log("w","Cannot show fit parameters error without fitting.")
            config['plot']['useFitError'] = False

    #################
    #Printout details
    #################
    whichValidation = ""
    if isDMR: whichValidation = "DMR"
    elif isPV: whichValidation = "PV"

    ##########################################################################################
    #DATA: PERFORM INITIAL MERGE OR FINALIZE
    #PV: EACH FOLDER CONSIDERED TO BE ONE OBJECT, DMR: (MULTIPLE) OBJECTS DECIDED IMPLICITELY
    ##########################################################################################
    tuples_total = []
    if config['mode'] == "merge": 
        for inputDir in inputDirData:
            ###################################################################################
            #Get list of python dictionaries:  
            #{ 'file' : file name, 'number' : IoV/run number, 'lumi' : corresponding luminosity} 
            ####################################################################################  
            tuples = getTuples(inputDir,IOVs)
            intLumi = getIntLumi(tuples)
            if len(tuples) == 0:
                log("e","Zero "+whichValidation+" files to merge detected.")
                sys.exit(0)
            log("i","Attempting to average "+str(len(tuples))+" "+whichValidation+" files with (partial) integrated luminosity "+str(intLumi)+".")
            outFileName = ""
            objName = ""
            if isDMR:
                outFileName = os.path.join(config['output'],"OfflineValidationSummary"+objName+".root")
            elif isPV:
                objName = inputDir.split("/")[-1] 
                outFileName = os.path.join(config['output'],"result"+objName+".root")
            tuples_total.append({ 'file' : outFileName,
                                  'lumi' : intLumi 
                               })
            makeAveragedFile(tuples,intLumi,objName)
    elif config['mode'] == "finalize":
        tuples = []
        intLumi = 0.0 
        for inputDir in inputDirData:
            ####################################################################################################
            #Get list of python dictionaries:  
            #{ 'file' : previously merged file, 'number' : [], 'lumi' : corresponding lumi from previous jobs} 
            ####################################################################################################
            tupleFinal = getTuples(inputDir)
            tuples += tupleFinal
            intLumi += getIntLumi(tupleFinal)
        if len(tuples) == 0:
                log("e","Zero final "+whichValidation+" files to merge detected.")
                sys.exit(0)
        log("i","Attempting to average final "+str(len(tuples))+" "+whichValidation+" files with overall integrated luminosity "+str(intLumi)+".")
        objName = ""
        makeAveragedFile(tuples,intLumi,objName)

    ###################################################
    #DATA: CREATE LUMI FILE FOR FINALIZATION JOB
    ###################################################
    if config['mode'] == "merge":
        with open(os.path.join(config['output'],'lumiPerFile.csv'), 'a') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ')
            for tuple in tuples_total:
                csvwriter.writerow([tuple['file'],str(tuple['lumi'])])

    #################################################################
    #MC: FIRST DEFINE MC GROUPS 
    #    THEN MERGE ALL MC IN GROUP WITH CORRESPONDING LUMI SCALE
    #################################################################
    tupleGroupsMC = {}
    if config['mode'] == "merge":
        for inputDir in inputDirMC:
            tuples = getTuplesMC(inputDir)
            for period, _list in tuples.items():
                tuple = _list[0] 
                if period in lumiMC.keys(): 
                    tuple['lumi'] = lumiMC[period]['lumi']   
                    log("i","Group N."+str(lumiMC[period]['group'])+" <-- "+str(tuple['lumi']))  
                    if lumiMC[period]['group'] not in tupleGroupsMC.keys():
                        tupleGroupsMC[lumiMC[period]['group']] = []
                        tupleGroupsMC[lumiMC[period]['group']].append(tuple)
                    else:
                        tupleGroupsMC[lumiMC[period]['group']].append(tuple)
                else:
                    log("w","Period "+str(period)+" not recognised in lumiMC list.")
        for group, tuples in tupleGroupsMC.items():
            log("i","Detected MC N."+str(group)+" group to be merged.")
            intLumi = getIntLumi(tuples)
            makeAveragedFile(tuples,intLumi,"_merged"+str(group))

    ##############################
    #PLOT:
    ##############################
    if isDMR and config['mode'] == "plot":
        #import plotting class  
        from Alignment.OfflineValidation.TkAlAllInOneTool.DMRplotter import DMRplotter
 
        #initialize plotting class with proper options
        plotInfo = {}
        plotInfo['outputDir'] = config['output']
        for key in ['objects','labels','colors','styles', \
                    'useFit','useFitError','showMean','showMeanError','showRMS','showRMSError']: 
            plotInfo[key] = config['plot'][key]
        if 'plotGlobal' in config.keys():
            if 'CMSlabel' in config['plotGlobal'].keys():
                plotInfo['CMSlabel'] = config['plotGlobal']['CMSlabel']
            else:
                plotInfo['CMSlabel'] = ""
            if 'Rlabel' in config['plotGlobal'].keys():
                plotInfo['Rlabel'] = config['plotGlobal']['Rlabel']
            else:
                plotInfo['Rlabel'] = "single muon (2016+2017+2018)" 
        plotter = DMRplotter(plotInfo)

        #add input files
        for inputDir in inputDirData:
            plotter.addDATA(inputDir)
        for inputDir in inputDirMC:
            plotter.addDirMC(inputDir)

        #plot&save
        plotter.plot()

    log("i","All done.") 
