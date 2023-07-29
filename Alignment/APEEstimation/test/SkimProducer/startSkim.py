from __future__ import print_function
import sys
sys.path.append("../SkimProducer")
# for switching to CAF queue (condor submit)
sys.path.append("../autoSubmitter") 
import os
import subprocess
import signal
import time
import argparse
import multiprocessing as mp

def replaceAllRanges(string):
    if "[" in string and "]" in string:
        strings = []
        posS = string.find("[")
        posE = string.find("]")
        nums = string[posS+1:posE].split(",")
        expression = string[posS:posE+1]
        
        nums = string[string.find("[")+1:string.find("]")]
        for interval in nums.split(","):
            interval = interval.strip()
            if "-" in interval:
                lowNum = int(interval.split("-")[0])
                upNum = int(interval.split("-")[1])
                for i in range(lowNum, upNum+1):
                    newstring = string[0:posS]+str(i)+string[posE+1:]
                    newstring = replaceAllRanges(newstring)
                    strings += newstring
            else:
                newstring = string[0:posS]+interval+string[posE+1:]
                newstring = replaceAllRanges(newstring)
                strings += newstring
        return strings
    else:
        return [string,]


def condorSubmitSkim(sample, caf=False):
    from helpers import enableCAF
    enableCAF(caf)
    path = "{base}/src/Alignment/APEEstimation/test/SkimProducer".format(base=os.environ['CMSSW_BASE'])
    
    from skimTemplates import skimScript
    scriptfile = "{path}/workingArea/skim_{name}.tcsh".format(path=path, name=sample)
    with open(scriptfile, "w") as fi:
        fi.write(skimScript.format(base=os.environ['CMSSW_BASE']))
    
    if caf:
        from skimTemplates import condorSubTemplateCAF as condorSubScript
    else:
        from skimTemplates import condorSubTemplate as condorSubScript
    subfile = "{path}/workingArea/skim_{name}.sub".format(path=path, name=sample)
    with open(subfile, "w") as fi:
        fi.write(condorSubScript.format(path=path, name=sample))
    
    print(subfile)
    subprocess.call("condor_submit {subfile}".format(subfile=subfile), shell=True)
    
def localStartSkim(sample):
    base = os.environ['CMSSW_BASE']    
    
    execString = "cmsRun {base}/src/Alignment/APEEstimation/test/SkimProducer/skimProducer_cfg.py sample={sample}".format(sample=sample, base=base)
    print(execString)
    toExec = execString.split(" ")
    
    outFileName = None
    outFilePath = None
    
    # start cmsRun
    proc = subprocess.Popen(toExec, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    def get_output(proc):
        while True:
            line = proc.stdout.readline().rstrip().decode()
            if not line:
                break
            yield line

    
    # print output in shell while program runs, also extract output filename
    try:
        for line in get_output(proc):
            if "Using output name" in line:
                outFileName = line.split("output name ")[1].split(".root")[0]
            if "Using output path" in line:
                outFilePath = line.split("output path ")[1]
            print(sample+": "+line)
    except KeyboardInterrupt:
        #this way, the current file is closed and the skimming is finished in a way that the last opened file is actually usable
        
        print("Interrupted")
        proc.send_signal(signal.SIGINT)
        proc.wait()
        
    
    time.sleep(1)    
    print("Finished with %s, renaming files now"%(sample))

    ## Rename output files to match _n.root naming scheme    
    targetFiles = []
    if outFileName:
        for fi in os.listdir("."):
            if fi.split(".root")[0].startswith(outFileName):
                if fi.split(".root")[0] == outFileName:
                    newFileName = "%s_1.root"%(outFileName)
                    os.rename(fi, newFileName)
                    targetFiles.append(newFileName)
                else:
                    fileNoString = fi.split(".root")[0].split(outFileName)[1]
                    try:
                        fileNo = int(fileNoString)
                        # For (most) weird naming conventions to not mess up renaming
                        if len(fileNoString) != 3:
                            continue
                        
                        newFileName = "%s_%d.root"%(outFileName, fileNo+1)
                        os.rename(fi, newFileName)
                        targetFiles.append(newFileName)
                    except ValueError: 
                        # Catching files from previous skim with same name that were already renamed and not removed before next skim
                        # and files with longer names but identical parts
                        continue
    
    for fi in targetFiles:
        print(fi)

    if outFilePath:
        print("Copying files to desired path")
        if not os.path.isdir(outFilePath):
            os.makedirs(outFilePath)
        for fi in targetFiles:
            if not subprocess.call("xrdcp %s %s/"%(fi, outFilePath), shell=True):
                os.remove(fi)    

def main(argv):
    if not 'CMSSW_BASE' in os.environ:
        print("CMSSW evironment is not set up, do that first")
        exit(1)
    
    parser = argparse.ArgumentParser(description='Define which samples to skim')
    parser.add_argument("-s", "--sample", action="append", dest="samples", default=[],
                          help="Name of sample as defined in skimProducer_cfg.py. Multiple inputs possible")
    parser.add_argument("-c", "--condor", action="store_true", dest="condor", default=False,
                          help="Submit to condor, if False, the skim will be done locally on lxplus")
    parser.add_argument("-C", "--caf", action="store_true", dest="caf", default=False,
                          help="Submit to CAF queue for faster execution")
    parser.add_argument("-n", "--ncores", action="store", dest="ncores", default=-1, type=int,
                          help="Set maximum number of parallel skims to run if skimming is done locally")
    
    args = parser.parse_args()
    
    if len(args.samples) == 0:
        print("Usage: python startSkim.py -s <sample>")
        sys.exit(1)
    
    finalSamples = []
    for sample in args.samples:
        parsedSamples = replaceAllRanges(sample)
        finalSamples += parsedSamples
    
    args.samples = finalSamples
    
    if args.ncores<0 or args.ncores > len(args.samples):
        args.ncores = len(args.samples)
    
    if args.condor:
        # Every skim gets its own condor job. One could also submit one
        # job and add all jobs as arguments with small changes.
        for sample in args.samples:
            condorSubmitSkim(sample, args.caf)
    else:
        if len(args.samples) == 1:
            for sample in args.samples:
                localStartSkim(sample) 
        else:
            try:
                # Not recommended for a large number of jobs
                # They will get killed by admins if they overload lxplus
                pool = mp.Pool(args.ncores)
                pool.map_async(localStartSkim, args.samples)
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                pass # The keyboard interrupt will be forwarded to the subprocesses anyway, stopping them without terminating them immediately
if __name__ == "__main__":
    main(sys.argv)
