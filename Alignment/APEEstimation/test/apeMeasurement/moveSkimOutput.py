import os 
import shutil
import helpers
import argparse
import glob
import os



def main():
    parser = argparse.ArgumentParser(description="Rename and move output files")
    parser.add_argument("-f", "--fileName", action="store", dest="fileName",
                          help="Base file name")
    parser.add_argument("-s", action="store", dest="source",
                          help="Source folder")
    parser.add_argument("-t", action="store", dest="target",
                          help="Target folder")
    args = parser.parse_args()
    
    
    # create target directory
    
    helpers.ensurePathExists(args.target)
    
    # remove files in target directory if they exist
    
    
    # before: files have the naming scheme {baseName}.root and {baseName}00X.root
    # after: files have the nameing scheme {baseName}_X.root and are in the target folder
    
    files = glob.glob("{}/{}*.root".format(args.source, args.fileName))
    
    for file in files:
        if file == "{}/{}.root".format(args.source, args.fileName):
            shutil.move(file, "{}/{}_1.root".format(args.target, args.fileName))
        else:
            path, fn = os.path.split(file)
            num = int(fn[len(args.fileName):-5])+1 # add +1 as _1 is already occupied by the first file, so 001 becomes _2 etc
            shutil.move(file, "{}/{}_{}.root".format(args.target, args.fileName, num))
    

if __name__ == "__main__":
    main()
-- dummy change --
