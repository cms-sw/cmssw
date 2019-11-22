#!/usr/bin/env python

from __future__ import print_function
import sys, os, string

def usage():
    """ Usage: fixTrue <indir> <outdir> 
    Reads .out files in directory <indir> to determine L1 skimming efficiency
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

def FixFile(ifile,ofile):
    
    infile =OpenFile(ifile,'r')
    outfile=OpenFile(ofile,'w')
    iline=0

    x = infile.readline()

    nFailed=-1
    nPassed=-1
    
    while x != "":
        iline+=1
        xx=string.rstrip(x)
        if xx.find("doMuonCuts = [true];")>-1:
            xx=" doMuonCuts = [ABCD];"
        if xx.find("doElecCuts = [false];")>-1:
            xx=" doElecCuts = [EFGH];"
        if xx.find("20130919_QCD")>-1:
            xx="    versionTag = \"VERSIONTAG\";"
        if xx.find("dcache:/pnfs/cms/WAX/11/store/user/lpctrig/ingabu/TMDNtuples/YYYY/")>-1:
            xx=" paths = [\"BASENAME\"];"
        outfile.write(xx + "\n")
        x = infile.readline() 

    outfile.write("\n")
    CloseFile(infile)
    CloseFile(outfile)

    return

if __name__ == '__main__':

    narg=len(sys.argv)
    if narg < 3 :
        print(usage.__doc__)
        sys.exit(1)


    Dir1=sys.argv[1]
    Dir2=sys.argv[2]

    filelist=os.listdir(Dir1)
    for thefile in filelist:
        print(thefile)
        ifile=os.path.join(Dir1,thefile)
        ofile=os.path.join(Dir2,thefile)
        print("diff ",ifile,ofile)
        FixFile(ifile,ofile)
