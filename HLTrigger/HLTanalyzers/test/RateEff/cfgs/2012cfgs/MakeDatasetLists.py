#!/usr/bin/env python

# Add version numbers for PD stuff...

import os, string, sys, posix, tokenize, array, getopt

def main(argv):
    menufilename = "hltmenu_extractedhltmenu_2012_online_8e33_v1p0_8e33column_NoParking.cfg"
    #    pdfilename = "PathsByPhysicsGroup_5E33_2012.list"
    #    versionedpdfilename = "Versioned_PathsByPhysicsGroup_5E33_2012.list"
    #    pdfilename = "Datasets_8E33_GRun_V32_2012.list"
    versionedpdfilename = "Versioned_Datasets_8E33_online_v1p0_2012_NoParking.list"

    #    pdfile = open(pdfilename)
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
    
