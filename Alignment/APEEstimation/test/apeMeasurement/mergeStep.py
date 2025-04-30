import os
import shutil
import argparse
import subprocess
import helpers


def main():
    parser = argparse.ArgumentParser(description="Automatically run APE measurements")
    parser.add_argument("--iteration", action="store", dest="iteration", type=int,
                      help="Current iteration number")
    parser.add_argument("--numFiles", action="store", dest="numFiles", type=int,
                      help="Number of files")
    parser.add_argument("--workingArea", action="store", dest="workingArea", 
                      help="Working Area folder")
    parser.add_argument("--measName", action="store", dest="measName", 
                      help="Measurement name.")                      
    parser.add_argument("--isBaseline", action="store", dest="isBaseline",
                      help="Is baseline measurement")
    # create iteration folders, then merge outputs and remove files
    args = parser.parse_args()
    args.isBaseline = (args.isBaseline == "True")
    
    fileNames = [os.path.join(args.workingArea, args.measName, "out{}.root".format(i)) for i in range(1, args.numFiles+1)  ]    
    
    if not args.isBaseline:
        helpers.newIterFolder(args.workingArea, args.measName, "iter"+str(args.iteration))
        targetName = os.path.join(args.workingArea, args.measName, "iter"+str(args.iteration), "allData.root")
    else:
        helpers.newIterFolder(args.workingArea, args.measName, "baseline")
        targetName = os.path.join(args.workingArea, args.measName, "baseline", "allData.root")
    
    subprocess.call("hadd {} {}".format(targetName, " ".join(fileNames)), shell=True)
    
    for name in fileNames:
        os.remove(name)
        
    if args.iteration>0 and not args.isBaseline: 
        # copy over file from last iteration, so that new results are appended in each iteration
        shutil.copyfile( os.path.join(args.workingArea, args.measName, "iter"+str(args.iteration-1), "allData_iterationApe.root"), 
                         os.path.join(args.workingArea, args.measName, "iter"+str(args.iteration),   "allData_iterationApe.root"))
    
    
if __name__ == "__main__":
    main()
-- dummy change --
