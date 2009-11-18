#!/usr/bin/env python

# ConfdbToOpenHLT.py
#
# Generate OpenHLT code/configs from a menu in ConfDB


import os, string, sys, posix, tokenize, array, getopt, operator

sys.path.append(os.environ.get("CMS_PATH") + "/slc4_ia32_gcc345/external/py2-cx-oracle/4.2/lib/python2.4/site-packages/") 
 
import cx_Oracle 

def main(argv):

    input_verbose = 0
    input_dbuser = "CMS_HLTDEV_WRITER"
    input_dbpwd = ""
    input_host = "CMS_ORCOFF_PROD" 
    input_notech = 0
    input_fakel1 = 0

    input_config = "/online/beamhalo/week47/HLT/V4"

    opts, args = getopt.getopt(sys.argv[1:], "c:v:d:u:s:o:n:fh", ["config=","verbose=","dbname=","user=","password=","dbtype=","hostname=","notechnicaltriggers=","fakel1seeds="])

    for o, a in opts:
        if o in ("-c","config="):
            input_config = str(a)
            print "Using config name " + input_config
        if o in ("-d","dbname="):
            input_dbname = str(a)
            print "Using DB named " + input_dbname
        if o in ("-u","user="):
            input_dbuser = str(a)
            print "Connecting as user " + input_dbuser
        if o in ("-s","password="):
            input_dbpwd = str(a)
            print "Use DB password " + input_dbpwd
        if o in ("-o","hostname="):
            input_host = str(a)
            print "Use hostname " + input_host
        if o in ("-v","verbose="):
            input_verbose = int(a)
            print "Verbosity = " + str(input_verbose)
        if o in ("-n","notechnicaltriggers="):
            print "Paths seeded by technical triggers excluded"
            input_notech = 1
        if o in ("-f","fakel1seeds="):
            print "Will use Open L1 seeding"
            input_fakel1 = 1

    confdbjob = ConfdbToOpenHLT(input_config,input_verbose,input_dbuser,input_dbpwd,input_host,input_notech,input_fakel1)
    confdbjob.BeginJob()

class ConfdbToOpenHLT:
    def __init__(self,cliconfig,cliverbose,clidbuser,clidbpwd,clihost,clinotech,clifakel1):
        
        self.dbname = ''
        self.dbuser = clidbuser
        self.verbose = int(cliverbose)
        self.dbpwd = clidbpwd
        self.dbhost = clihost
        self.verbose = cliverbose
        self.configname = cliconfig
        self.notech = clinotech
        self.fakel1 = clifakel1
        
        # Track CVS tags
        self.tagtuple = []
        self.alreadyadded = []
        
        # Get a Conf DB connection here. Only need to do this once at the
        # beginning of a job.
        print "Connecting as " + self.dbuser+"@"+self.dbhost+"/"+self.dbpwd
        self.connection = cx_Oracle.connect(self.dbuser+"/"+self.dbpwd+"@"+self.dbhost) 
        self.dbcursor = self.connection.cursor()  

    def BeginJob(self):
        theconfdbpaths = []
        thefullhltpaths = []
        theopenhltpaths = []
        theintbits = []
        thebranches = []
        theaddresses = [] 
        themaps = []

        rateeffhltcfgfile = open("hltmenu_extractedhltmenu.cfg",'w')
        rateeffopenhltcfgfile = open("openhltmenu_extractedhltmenu.cfg",'w')
        rateefflibfile = open("OHltTree_FromConfDB.h",'w')

        self.dbcursor.execute("SELECT Configurations.configId FROM Configurations WHERE (configDescriptor = '" + self.configname + "')")
        tmpconfid = self.dbcursor.fetchone()
        if(tmpconfid):
            tmpconfid = tmpconfid[0]
        else:
            print 'Could not find the configuration ' + str(self.configname) + ' - exiting'
            return
            
        self.dbcursor.execute("SELECT Paths.name FROM Paths JOIN ConfigurationPathAssoc ON ConfigurationPathAssoc.pathId = Paths.pathId JOIN Configurations ON ConfigurationPathAssoc.configId = Configurations.configId WHERE (Configurations.configId = " + str(tmpconfid) + ")") 
        
        thepaths = self.dbcursor.fetchall()

        crazygenerall1select = "SELECT STRINGPARAMVALUES.VALUE FROM STRINGPARAMVALUES JOIN PARAMETERS ON PARAMETERS.PARAMID = STRINGPARAMVALUES.PARAMID JOIN SUPERIDPARAMETERASSOC ON SUPERIDPARAMETERASSOC.PARAMID = PARAMETERS.PARAMID JOIN MODULES ON MODULES.SUPERID = SUPERIDPARAMETERASSOC.SUPERID JOIN MODULETEMPLATES ON MODULETEMPLATES.SUPERID = MODULES.TEMPLATEID JOIN PATHMODULEASSOC ON PATHMODULEASSOC.MODULEID = MODULES.SUPERID JOIN PATHS ON PATHS.PATHID = PATHMODULEASSOC.PATHID JOIN CONFIGURATIONPATHASSOC ON CONFIGURATIONPATHASSOC.PATHID = PATHS.PATHID JOIN CONFIGURATIONS ON CONFIGURATIONS.CONFIGID = CONFIGURATIONPATHASSOC.CONFIGID WHERE MODULETEMPLATES.NAME = 'HLTLevel1GTSeed' AND PARAMETERS.NAME = 'L1SeedsLogicalExpression' AND CONFIGURATIONS.CONFIGDESCRIPTOR = "


        for thepathname in thepaths:
            if((thepathname[0]).startswith("HLT_") or (thepathname[0]).startswith("AlCa_")):
                l1select = crazygenerall1select + "'" + str(self.configname) + "' AND Paths.name = '" + str(thepathname[0]) + "'"
                if(self.fakel1 == 0):
                    self.dbcursor.execute(l1select)
                    l1bits = self.dbcursor.fetchone()
                    if(l1bits):
                        if((l1bits[0]).find("L1_") != -1 or self.notech == 0):
                            theconfdbpaths.append((thepathname[0],l1bits[0]))
                else:
                    theconfdbpaths.append((thepathname[0],'"OpenL1_ZeroBias"'))

                theintbits.append('  Int_t           ' + thepathname[0] + ';')
                thebranches.append('  TBranch        *b_' + thepathname[0] + ';   //!')
                theaddresses.append('  fChain->SetBranchAddress("' + thepathname[0] + '", &' + thepathname[0] + ', &b_' + thepathname[0] + ');')
                themaps.append('  fChain->SetBranchAddress("' + thepathname[0] + '", &map_BitOfStandardHLTPath["' + thepathname[0] + '"], &b_' + thepathname[0] + ');')
                

        npaths = len(theconfdbpaths)
        pathcount = 1
        
        for hltpath, seed in theconfdbpaths:
            if(hltpath.startswith("HLT_")):
               fullpath = '   ("' + str(hltpath) + '", ' + seed + ', 1, 0.15)'
               openpath = '   ("Open' + str(hltpath) + '", ' + seed + ', 1, 0.15)'
               if(pathcount < npaths):
                   fullpath = fullpath + ','
                   openpath = openpath + ','
               
            if(hltpath.startswith("AlCa_")):
               fullpath = '   ("' + str(hltpath) + '", ' + seed + ', 1, 0.)'
               openpath = '   ("Open' + str(hltpath) + '", ' + seed + ', 1, 0.)'
               if(pathcount < npaths):
                   fullpath = fullpath + ','
                   openpath = openpath + ','
                                                        
            thefullhltpaths.append(fullpath)
            theopenhltpaths.append(openpath)
            pathcount = pathcount + 1
            
        for rateeffpath in thefullhltpaths:
            rateeffhltcfgfile.write(rateeffpath + "\n")
        for rateeffpath in theopenhltpaths:
            rateeffopenhltcfgfile.write(rateeffpath + "\n")

        # Now write .h file    
        rateefflibfile.write("// The following lines should be included once and only once in OHltTree.h\n\n") 
        for intbit in theintbits:
            rateefflibfile.write(intbit + "\n")
        rateefflibfile.write("\n")    
        for branch in thebranches:
            rateefflibfile.write(branch + "\n")
        rateefflibfile.write("\n")
        for address in theaddresses:
            rateefflibfile.write(address + "\n")
        rateefflibfile.write("\n")
        for mapping in themaps:
            rateefflibfile.write(mapping + "\n")
        
        rateeffhltcfgfile.close()
        rateeffopenhltcfgfile.close()
        rateefflibfile.close()

        self.connection.commit() 
        self.connection.close() 
            
if __name__ == "__main__": 
    main(sys.argv[1:]) 
