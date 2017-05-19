#!/usr/bin/python
#

import sys,string,time,os

### parameters ###

FilesPerCfg=1
CFGBASE="MinimumBias_"

####

def usage():
    """ Usage: CreateFileList <FileList> <OutputDir>
    
 Splits a large file list into a set of smaller lists with <FilesPerCfg> files
    """
    pass

def OpenFile(file_in,iodir):
    """  file_in -- Input file name
         iodir   -- 'r' readonly  'r+' read+write """
    try:
        ifile=open(file_in, iodir)
        # print "Opened file: ",file_in," iodir ",iodir
    except:
        print "Could not open file: ",file_in
        sys.exit(1)
    return ifile

def CloseFile(ifile):
    ifile.close()

def ReadFile(file):

    
    infile=OpenFile(file,'r')
    iline=0

    x = infile.readline()

    files=[]
    while x != "":
        iline+=1
        xx=string.strip(x)
        if (len(xx)>0):
            files.append(xx)
                    
        x = infile.readline() 
    CloseFile(infile)

    return files

def createCFG(i,filelist):

    CFGFILE=CFGBASE + str(i) +".py"
    print i, CFGFILE
    file = open(CFGFILE,'w')
    file.write("import FWCore.ParameterSet.Config as cms \n")
    file.write("\n")
    file.write("maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )\n")
    file.write("readFiles = cms.untracked.vstring()\n")
    file.write("secFiles = cms.untracked.vstring()\n")        
    file.write("source = cms.Source (\"PoolSource\",fileNames = readFiles, secondaryFileNames = secFiles)\n")        
    file.write("readFiles.extend( [\n")

    i=0
    for infile in filelist:
       i=i+1
       if i<len(filelist):
           endstr="',"
       else:
           endstr="' ]);"
       outstring="    " + infile + endstr
       file.write(outstring + "\n")

    file.write("\n")            
    file.write("secFiles.extend( [ ])\n")        
    CloseFile(file)
    
if __name__ == '__main__':


    narg=len(sys.argv)
    if narg < 3 :
        print usage.__doc__
        sys.exit(1)


    InputFile=sys.argv[1]
    OutputDir=sys.argv[2]
    
    if not os.path.exists(OutputDir):
        os.mkdir(OutputDir)
    
    filelist = ReadFile(InputFile)
    print "Number of files in input filelist: ", len(filelist)

    os.chdir(OutputDir)

    i=0
    ibatch=0
    cfglist=[]
    for file in filelist:
        i=i+1
        #if (i>MAXFILES): break
        filename=file
        cfglist.append(filename)
        if i%FilesPerCfg == 0 or i==len(filelist):
            createCFG(ibatch,cfglist)
            ibatch=ibatch+1
            cfglist=[]
            
        #print i%FilesPerCfg, i, filename

