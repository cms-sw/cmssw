import helpers
import os
import shutil
import argparse
import glob
import condorTemplates

def main():
    parser = argparse.ArgumentParser(description="Automatically run APE measurements")
    parser.add_argument("--workingArea", action="store", dest="workingArea",
                          help="Config file that configures measurement")
    parser.add_argument("--dataDir", action="store", dest="dataDir",
                          help="Path of files")
    parser.add_argument("--fileName", action="store", dest="fileName",
                          help="Filenames")
    parser.add_argument("--measName", action="store", dest="measName",
                          help="Measurement name")
    parser.add_argument("--maxIterations", action="store", dest="maxIterations", default=15, type=int,
                          help="Max Iterations")
    parser.add_argument("--maxEvents", action="store", dest="maxEvents", default=-1, type=int,
                          help="Max Events")
    parser.add_argument("--isBaseline", action="store", dest="isBaseline",
                          help="Baseline measurement")
    parser.add_argument("--baselineName", action="store", dest="baselineName",
                          help="Baseline measurement")
    parser.add_argument("--isCosmics", action="store", dest="isCosmics", 
                          help="Is cosmics dataset")
    parser.add_argument("--globalTag", action="store", dest="globalTag", 
                          help="global Tag")
                          
    
    args = parser.parse_args()
    # argparser can't handle bool arguments
    args.isBaseline = (args.isBaseline == "True")
    args.isCosmics = (args.isCosmics == "True")
    
    
    files = glob.glob(os.path.join(args.dataDir, "{}_*.root".format(args.fileName)))
    files.sort()
    numFiles = len(files)
    
    base = os.environ['CMSSW_BASE']
    
    workingFolder = os.path.join(args.workingArea,args.measName)
    dag_name = "measurement.dag"
    if args.isBaseline:
        dag_name = "baseline.dag"
        
    # set up dag script
    with open(os.path.join(workingFolder, dag_name), "w") as dag_script:
        for iteration in range(0, args.maxIterations+1):
            # in the last iteration (usually iteration 15) additional plots are produced
            firstIter=(iteration==0)
            lastIter=(iteration==args.maxIterations)
            refitJobs = []
            dag_script.write("#iteration {}\n".format(iteration))
            
            # refit jobs
            for fileNumber in range(1, numFiles+1):
                # for each file, a refit job is started, the results are merged in the next step once all refit jobs are finished
                refitJob = "refit_{}_iter{}_{}".format(args.measName, iteration, fileNumber)
                refitJobs.append(refitJob)
                refitJobFile = os.path.join(workingFolder,"refit_iter{}_{}.sub".format(iteration, fileNumber))
                fileLocation = os.path.join(args.dataDir, "{}_{}.root".format(args.fileName,fileNumber))
                with open(refitJobFile, "w") as refitJobSub:
                    refitJobSub.write(condorTemplates.refitterSubTemplate.format(
                                                            base=base, 
                                                            workingArea=args.workingArea,
                                                            fileLocation=fileLocation,
                                                            globalTag=args.globalTag,
                                                            measName=args.measName,
                                                            fileNumber=fileNumber,
                                                            iteration=iteration,
                                                            lastIter=lastIter,
                                                            isCosmics=args.isCosmics,
                                                            maxEvents=args.maxEvents))
                    
                dag_script.write("JOB {} {}\n".format(refitJob, refitJobFile))
            
            # If this is not the first iteration, the refit jobs need to wait for the previous iteration to finish
            if not firstIter:
                iterationJobPrev = "iteration_{}_iter{}".format(args.measName, (iteration-1))
                dag_script.write("PARENT {} CHILD {}\n".format(iterationJobPrev, " ".join(refitJobs)))
            
            # finish iteration job
            iterationJob = "iteration_{}_iter{}".format(args.measName, iteration)
            iterationJobFile = os.path.join(workingFolder, "iteration_iter{}.sub".format(iteration))
            with open(iterationJobFile, "w") as iterationJobSub:
                iterationJobSub.write(condorTemplates.iterationSubTemplate.format(
                                                            base=base,
                                                            workingArea=args.workingArea,
                                                            measName=args.measName,
                                                            numFiles=numFiles,
                                                            iteration=iteration,
                                                            isBaseline=args.isBaseline,
                                                            baselineName=args.baselineName
                                                            ))
            
            dag_script.write("JOB {} {}\n".format(iterationJob, iterationJobFile))
            dag_script.write("PARENT {} CHILD {}\n".format(" ".join(refitJobs),iterationJob))
            dag_script.write("\n")
    
   


if __name__ == "__main__":
    main()
-- dummy change --
