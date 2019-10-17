#!/usr/bin/env python
#

from __future__ import print_function
import sys,string,time,os

### parameters ###
istart=0
NJOBS=41
##################


INPUTSTARTSWITH="INPUTFILE="
searchInput="xxx"

OUTPUTSTARTSWITH="    OUTPUTFILE="
#OUTPUTSTARTSWITH="SkimmedOutput="
searchOutput=".root"



def usage():
    """ Usage: CreateCFGS <cmsCFGFile> <outputDir>
    """
    pass

def OpenFile(file_in,iodir):
    """  file_in -- Input file name
         iodir   -- 'r' readonly  'r+' read+write """
    try:
        ifile=open(file_in, iodir)
        # print "Opened file: ",file_in," iodir ",iodir
    except:
        print("Could not open file: ",file_in)
        sys.exit(1)
    return ifile

def CloseFile(ifile):
    ifile.close()

def createCFGFiles(i,orgFile,basename,dir):

    newFile=basename + "_" + str(i) + ".py"
    newFile=os.path.join(dir,newFile)
    print(newFile)
    outFile = open(newFile,'w')
    
    for iline in orgFile:
        indx=string.find(iline,INPUTSTARTSWITH)
        if (indx == 0):
            indx2=string.find(iline,searchInput)
            if (indx2 < 0):
                print("Problem")
                sys.exit(1)
            else:
                iline=string.replace(iline,searchInput,str(i))
            
        indx=string.find(iline,OUTPUTSTARTSWITH)
        if (indx == 0):
            indx2=string.find(iline,searchOutput)
            if (indx2 < 0):
                print("Problem")
                sys.exit(1)
            else:
                replString="_" + str(i) + searchOutput
                iline=string.replace(iline,searchOutput,replString)
            
        outFile.write(iline + "\n")
    CloseFile(outFile)
    
    return newFile

def createLSFScript(cfgfile):

    file=os.path.basename(cfgfile)
    absDir=os.path.abspath(os.path.dirname(cfgfile))
    
    
    outScript="runlsf_" + string.replace(file,".py",".csh")
    outLog="runlsf_" + string.replace(file,".py",".log")

    inScript=os.path.join(file)
    outScript=os.path.join(absDir,outScript)
    outLog=os.path.join(absDir,outLog)

    oFile = open(outScript,'w')
    oFile.write("#!/bin/csh" + "\n")
    #oFile.write("\n")    
    oFile.write("cd " + absDir + "\n")
    oFile.write("eval `scram runtime -csh`" +  "\n")
    #oFile.write("\n")
    oFile.write("cmsRun " + inScript + "\n")    
    #oFile.write("date" + "\n")
    
    oFile.close()
    
    return

def ReadFile(file):
    
    infile=OpenFile(file,'r')
    iline=0
    
    x = infile.readline()

    file=[]
    while x != "":
        iline+=1
        xx=string.rstrip(x)
        file.append(xx)        
        x = infile.readline()
        
    CloseFile(infile)

    return file

    
if __name__ == '__main__':


    narg=len(sys.argv)
    if narg < 3 :
        print(usage.__doc__)
        sys.exit(1)


    InputFile=sys.argv[1]
    basename=string.replace(InputFile,".py","")
    
    OutputDir=sys.argv[2]
    if not os.path.exists(OutputDir):
        os.mkdir(OutputDir)
                
    infile=ReadFile(InputFile)

    # Create the cmsRun configuration files
    cfglist=[]
    for i in range(istart,istart+NJOBS):
        # print i
        file=createCFGFiles(i,infile,basename,OutputDir)
        cfglist.append(file)
        
   # Create the LSF submit scripts
    for cfgfile in cfglist:
        createLSFScript(cfgfile)




        
        
