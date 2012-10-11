#!/usr/bin/env python

# Add version numbers for PD stuff...

import os, string, sys, posix, tokenize, array, getopt

def main(argv):
    menufilename = "hltmenu_extractedhltmenu_2012_cdaq_5e33_v4p6_V4_unprescaled2.cfg"
    #    pdfilename = "PathsByPhysicsGroup_5E33_2012.list"
    #    versionedpdfilename = "Versioned_PathsByPhysicsGroup_5E33_2012.list"
    pdfilename = "Datasets_5E33_2012.list"
    versionedpdfilename = "Versioned_Datasets_5E33_2012.list"

    pdfile = open(pdfilename)
    versionedpdfile = open(versionedpdfilename, 'w')

    menufile = open(menufilename)
    menufilelines = menufile.readlines()
    foundpath = 0
    for menufileline in menufilelines:
        if(menufileline.find('# dataset') != -1):
            thepd = (menufileline.split('# dataset')[1]).split(' #')[0]
            thepd = thepd.lstrip().rstrip()
            versionedpdfile.write("\n"+thepd+":"+"\n")
        if(menufileline.find('"HLT_') != -1 or menufileline.find('"AlCa_') != -1):
            if(menufileline.lstrip().startswith("#")):
               continue
            menufiletokens = menufileline.split('"')
            menupath = menufiletokens[1]
            versionedpdfile.write("  "+menupath+"\n")
    menufile.close()

    versionedpdfile.close()
    
if __name__ == "__main__":
    main(sys.argv[1:])
    
