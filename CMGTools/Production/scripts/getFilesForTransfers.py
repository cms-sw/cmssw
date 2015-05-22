#!/usr/bin/env python
## Author: Peter Meckiffe
## @ CERN, Meyrin
## March 27th 2012

import os, sys, re, optparse
import CMGTools.Production.eostools as eostools
import CMGTools.Production.castorBaseDir as castorBaseDir

def burrow(LFN):
        returnFiles = []
        
        files = eostools.matchingFiles(LFN,"[!histo]*.")
        
        for i in files:
            if re.search(i, "histo"):
                del(files[i])
        for file in files:
            if not re.search("\.",file):
                for i in burrow(file): returnFiles.append(i)
            else:
                returnFiles.append(file)
        
        return returnFiles
if __name__ == '__main__':
    parser = optparse.OptionParser()
    
    parser.usage = """
%prog [options] <filename>
filename should be a .txt file with the names or samples you want transferred
use script to get LFN of all files needed for transfer.

Each line should be in the form:
fileowner%dataset
"""
    
    
    
    
    (options, args) = parser.parse_args()
    
    # Allow no more than one argument
    if len(args)!=1:
        parser.print_help()
        sys.exit(1)
    
    
    
    # For multiple file input
    
    file = open(args[0], 'r')
    lines = file.readlines()
    for line in lines:
        line = re.sub("\s+", " ", line)	
        fileown = line.split("%")[0].lstrip().rstrip()
        dataset = line.split("%")[1].lstrip().rstrip()
        files = []
        if re.search('group',fileown):
            castor = eostools.lfnToEOS(castorBaseDir.castorBaseDir(fileown))+dataset
            castor2 = eostools.lfnToEOS(castorBaseDir.castorBaseDir(fileown.strip("_group")))+dataset
        else:
            castor = eostools.lfnToEOS(castorBaseDir.castorBaseDir(fileown))+dataset
            castor2 = eostools.lfnToEOS(castorBaseDir.castorBaseDir(fileown+"_group"))+dataset
        LFN = eostools.eosToLFN(castor)
        LFN2 = eostools.eosToLFN(castor2)

        if eostools.isDirectory(castor):
            files = burrow(LFN)
            
        elif eostools.isDirectory(castor2):
            files = burrow(LFN2)
            
        print dataset
        for i in files: print "\t"+i
        
            #print "Dataset: "+dataset+" not found"
        

            
