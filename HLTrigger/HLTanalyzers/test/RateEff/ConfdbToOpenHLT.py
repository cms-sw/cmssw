#!/usr/bin/env python

# GetL1BitsForSkim.py
#
# Simple script for getting all L1 seeds used by a particular HLT menu

import os, string, sys, posix, tokenize, array, getopt
from pkgutil import extend_path
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.Modules as mod


def main(argv):

    input_notech = 0
    input_fakel1 = 0
    input_orcoffmenu = 0
    input_userefprescales = 0
    input_referencecolumn = 0
    input_useconfdbprescales = 0
    input_pdsort = 0
    input_config = "/cdaq/physics/firstCollisions10/v5.0/HLT/V2"
    input_refconfig = "/cdaq/physics/firstCollisions10/v5.1/HLT_900GeV/V3"

    opts, args = getopt.getopt(sys.argv[1:], "c:onfdp:s:r:h", ["config=","orcoff","notechnicaltriggers","fakel1seeds","datasetsorting","addreferenceprescales=","confdbprescales=","referenceconfig=","help"])

    for o, a in opts:
        if o in ("-c","config="):
            input_config = str(a)
        if o in ("-n","notechnicaltriggers="):
            print "Paths seeded by technical triggers excluded"
            input_notech = 1
        if o in ("-f","fakel1seeds="):
            print "Will use Open L1 seeding"
            input_fakel1 = 1
        if o in ("-d","datasetsorting="):
            print "Will sort by PD's"
            input_pdsort = 1
        if o in ("-p","addreferenceprescales"):
            input_userefprescales = 1
            input_referencecolumn = a
            print "Will use reference prescales from column " + str(a)
        if o in ("-s","confdbprescales"):
            input_useconfdbprescales = 1
            input_referencecolumn = a
            print "Will use ConfDB prescales from column " + str(a)
        if o in ("-r","referenceconfig="):
            input_refconfig = str(a)
        if o in ("-o","orcoff"):
            input_orcoffmenu = 1
        if o in ("-h","help"):
            print "-c <HLT key> (HLT configuration to generate the menu from)"
            print "-o (Take the configuration from ORCOFF instead of HLTDEV)"
            print "-n (Don't include paths seeded by technical triggers)"
            print "-f (Use the fake OpenL1_ZeroBias seed instead of rechecking L1's)"
            print "-p <column> (Include prescales that were previously applied online. Default is the first column (0))"
            print "-s <column> (Apply prescales from the ConfDB menu. Default is the first column (0))"
            print "-r <HLT key> (HLT configuration to take the previously applied prescales from)" 
            print "-d (Sort triggers in the cfg by Primary Dataset rather than the order they appear in the menu" 
            print "-h (Print the help menu)"
            return

    confdbjob = ConfdbToOpenHLT(input_config,input_orcoffmenu,input_notech,input_fakel1,input_userefprescales,input_refconfig,input_referencecolumn,input_pdsort,input_useconfdbprescales)
    confdbjob.BeginJob()
    os.system("mv OHltTree_FromConfDB.h OHltTree.h")
    
            
class ConfdbToOpenHLT:
    def __init__(self,cliconfig,cliorcoff,clinotech,clifakel1,clirefprescales,clirefconfig,clireferencecolumn,clipdsort,cliconfdbprescales):

        self.configname = cliconfig
        self.orcoffmenu = cliorcoff 
        self.notech = clinotech
        self.fakel1 = clifakel1
        self.refconfigname = clirefconfig
        self.userefprescales = clirefprescales
        self.hltl1seedmap = {}
        self.l1aliasmap = {}
        self.hltprescalemap = {}
        self.confdbhltprescalemap = {}
        self.l1modulenameseedmap = {}
        self.hltl1modulemap = {}
        self.referencecolumn = int(clireferencecolumn)
        self.pdsort = clipdsort
        self.confdbprescales = cliconfdbprescales

    def BeginJob(self):

        thepaths = []
        theconfdbpaths = []
        thefullhltpaths = []
        theopenhltpaths = []
        theintbits = []
        thebranches = []
        theaddresses = [] 
        themaps = []
        theintprescls = []
        thepresclbranches = []
        theprescladdresses = []
        thepresclmaps = []
        intbitstoadd = []
        branchestoadd = []
        addressestoadd = []
        mapstoadd = []
        intpresclstoadd = []
        presclbranchestoadd = []
        prescladdressestoadd = []
        presclmapstoadd = []
        usedl1bits = [] 
        hltpdmap = []
        thepds = []
        
        rateeffhltcfgfile = open("hltmenu_extractedhltmenu.cfg",'w')
        rateeffopenhltcfgfile = open("openhltmenu_extractedhltmenu.cfg",'w')
        rateefflibfile = open("OHltTree_FromConfDB.h",'w')
        rateefforiglibfile = open("OHltTree.h")

        # Use edmConfigFromDB to get a temporary HLT configuration
        configcommand = ""
        if(self.orcoffmenu == 1):
            configcommand = "edmConfigFromDB --orcoff --configName " + self.configname + " --cff >& temphltmenu.py"
        else:
            configcommand = "edmConfigFromDB --configName " + self.configname + " --cff >& temphltmenu.py"
        #configcommand = "edmConfigFromDB --orcoff --configName " + self.configname + " --cff >& temphltmenu.py"
            
        os.system(configcommand)

        # Use edmConfigFromDB to get a temporary HLT configuration that determines the prescales applied online 
        # We assume this is always from ORCOFF...
        referencemenuconfigcommand = "edmConfigFromDB --orcoff --configName " + self.refconfigname + " --cff >& refhltmenu.py"
        #        referencemenuconfigcommand = "edmConfigFromDB --configName " + self.configname + " --cff >& refhltmenu.py"
        os.system(referencemenuconfigcommand)
    
        # Setup a fake process and load the HLT configuration for the reference menu
        refimportcommand = "import refhltmenu"
        refprocess = cms.Process("MyReferenceProcess")
        exec refimportcommand
        reftheextend = "refprocess.extend(refhltmenu)"
        exec reftheextend

        # Get HLT reference prescales
        myservices = refprocess.services_()
        for servicename, servicevalue in myservices.iteritems():
            if(servicename == "PrescaleService"):
                serviceparams = servicevalue.parameters_()
                for serviceparamname, serviceparamval in serviceparams.iteritems():
                    if(serviceparamname == "prescaleTable"):
                        for vpsetentry in serviceparamval:
                            hltname = ""
                            hltprescale = "1"
                            prescalepsetparams = vpsetentry.parameters_()
                            for paramname, paramval in prescalepsetparams.iteritems():
                                if(paramname == "prescales"):
                                    hltprescale = str(paramval.value()).strip('[').strip(']')
                                    # Deal with prescale columns
                                    if(hltprescale.find(',') != -1):
                                        hltprescalevector = hltprescale.split(',')
                                        hltprescale = hltprescalevector[self.referencecolumn]
                                if(paramname == "pathName"):
                                    hltname = str(paramval.value())
                            self.hltprescalemap[hltname] = hltprescale

        # Setup a fake process and load the HLT configuration for the new menu
        importcommand = "import temphltmenu"
        process = cms.Process("MyProcess")
        exec importcommand
        theextend = "process.extend(temphltmenu)"
        exec theextend

        # JH Get HLT prescales from a new ConfDB menu
        myconfdbservices = process.services_()
        for confdbservicename, confdbservicevalue in myconfdbservices.iteritems():
            if(confdbservicename == "PrescaleService"):
                confdbserviceparams = confdbservicevalue.parameters_()
                for confdbserviceparamname, confdbserviceparamval in confdbserviceparams.iteritems():
                    if(confdbserviceparamname == "prescaleTable"):
                        for vpsetentry in confdbserviceparamval:
                            hltname = ""
                            hltprescale = "1"
                            prescalepsetparams = vpsetentry.parameters_()
                            for paramname, paramval in prescalepsetparams.iteritems():
                                if(paramname == "prescales"):
                                    hltprescale = str(paramval.value()).strip('[').strip(']')
                                    # Deal with prescale columns
                                    if(hltprescale.find(',') != -1):
                                        hltprescalevector = hltprescale.split(',')
                                        hltprescale = hltprescalevector[self.referencecolumn]
                                if(paramname == "pathName"):
                                    hltname = str(paramval.value())
                            self.confdbhltprescalemap[hltname] = hltprescale
        #JH

        # Get L1 Seeds
        myfilters = process.filters_()
        for name, value in myfilters.iteritems():
            myfilter = value.type_()
            myfilterinstance = name
            if(myfilter == "HLTLevel1GTSeed"):
                # Get parameters of the L1 seeding module and look for the L1
                params = value.parameters_()
                for paramname, paramval in params.iteritems():
                    if(paramname == "L1SeedsLogicalExpression"):
                        seeds = paramval.value()
                        self.l1modulenameseedmap[myfilterinstance] = seeds

                        # Record all L1 seeds for the L1 menu
                        tmpindividualseeds = seeds.split(' OR ') 
 
                        for tmpindividualseed in tmpindividualseeds: 
                            if(tmpindividualseed.find(' AND ') == -1): 
                                theseed = str(tmpindividualseed).strip('NOT ').strip('(').strip(')') 
                                if (not theseed in usedl1bits): 
                                    usedl1bits.append(theseed) 
                            else: 
                                individualseeds = tmpindividualseed.split(' AND ') 
                                for individualseed in individualseeds: 
                                    theseed = str(individualseed).strip('NOT ').strip('(').strip(')') 
                                    if (not theseed in usedl1bits): 
                                        usedl1bits.append(theseed) 

        # Lookup L1 alias names 
        l1file = open("tmpl1menu.table")
        l1lines = l1file.readlines()
        for l1line in l1lines:
            aliasname = (l1line.split("|")[1]).strip("!").lstrip().rstrip()
            bitnumber = l1line.split("|")[2].lstrip().rstrip()
            self.l1aliasmap[bitnumber] = aliasname

        # Get PrimaryDatasets
        mydatasets = process.datasets
        mydataset = mydatasets.parameters_()
        for datasetname, datasetval in mydataset.iteritems():
            if(datasetname.find("Monitor") != -1):
                continue
            if(datasetname.find("Test") != -1):
                continue
            if(datasetname.find("Express") != -1):
                continue
            thepds.append(datasetname)
            for datasetmember in datasetval:
                hltpdmap.append((datasetmember, datasetname)) 

        # Get ordering from the schedule
        myschedule = process.HLTSchedule
        for path in myschedule:
            pathinschedname = path.label_()
            mypaths = process.paths_()
            for name, value in mypaths.iteritems():
                if name == pathinschedname:
                    # Get L1-seeding modules of HLT paths
                    aliasedseeds = []
                    thepaths.append(name)
                    thingsinpath = str(value).split('+')
                    countl1seedmodules = 0
                    for thinginpath in thingsinpath:
                        if(thinginpath.startswith('hltL1s')):
                            countl1seedmodules = countl1seedmodules + 1

                            # If we have >1 L1 seed module in the path, add an AND
                            # since both are required to be satisfied 
                            if(countl1seedmodules > 1):
                                aliasedseeds.append("AND")
                                                                
                            self.hltl1modulemap[name] = thinginpath
                            if thinginpath not in self.l1modulenameseedmap:
                                continue
                                  
                            seedexpression = self.l1modulenameseedmap[thinginpath]
                            splitseedexpression = seedexpression.split()

                            for splitseed in splitseedexpression:
                                if(splitseed in self.l1aliasmap):
                                    splitseed = self.l1aliasmap[splitseed]
                                aliasedseeds.append(splitseed)


                        reconstructedseedexpression = "" 
                        for aliasedseed in aliasedseeds:
                            reconstructedseedexpression = reconstructedseedexpression + " " + aliasedseed

                        self.hltl1seedmap[name] = reconstructedseedexpression.lstrip().rstrip()
                        if(name in self.hltprescalemap):
                            theprescale = self.hltprescalemap[name]

                            
        # Now we have all the information, construct any configuration/branch changes
        for thepathname in thepaths:
            if((thepathname.startswith("HLT_")) or (thepathname.startswith("AlCa_"))):     
                if(self.fakel1 == 0):
                    if(thepathname in self.hltl1seedmap):
                        l1expression = self.hltl1seedmap[thepathname]
                    else:
                        l1expression = ''
                    theconfdbpaths.append((thepathname,l1expression))
                else:
                    theconfdbpaths.append((thepathname,'"OpenL1_ZeroBias"'))

                theintbits.append('   Int_t ' + thepathname + ';')
                thebranches.append('   TBranch *b_' + thepathname + '; //!')
                theaddresses.append('   fChain->SetBranchAddress("' + thepathname + '", &' + thepathname + ', &b_' + thepathname + ');')
                themaps.append('   fChain->SetBranchAddress("' + thepathname + '", &map_BitOfStandardHLTPath["' + thepathname + '"], &b_' + thepathname + ');')
                theintprescls.append('   Int_t ' + thepathname + '_Prescl;')
                thepresclbranches.append('   TBranch *b_' + thepathname + '_Prescl; //!')
                theprescladdresses.append('   fChain->SetBranchAddress("' + thepathname + '_Prescl", &' + thepathname + '_Prescl, &b_' + thepathname + '_Prescl);')
                thepresclmaps.append('   fChain->SetBranchAddress("' + thepathname + '_Prescl", &map_RefPrescaleOfStandardHLTPath["' + thepathname + '"], &b_' + thepathname + '_Prescl);')
                
        npaths = len(theconfdbpaths)
        pathcount = 1
        
        for hltpath, seed in theconfdbpaths:
            refprescaleval = ''
            confdbprescaleval = ', 1'

            # Check if we need to use the L1 alias
            if(seed in self.l1aliasmap):
                seed = self.l1aliasmap[seed]

            if(self.userefprescales == 1):
                if(hltpath in self.hltprescalemap):
                    refprescaleval = ', ' + self.hltprescalemap[hltpath]
                else:
                    refprescaleval = ', 1'

            if(self.confdbprescales == 1):
                if(hltpath in self.confdbhltprescalemap):
                    confdbprescaleval = ', ' + self.confdbhltprescalemap[hltpath]
                else:
                    confdbprescaleval = ', 1'
                                                                
            if(hltpath.startswith("HLT_")):
                #               fullpath = '   ("' + str(hltpath) + '", "' + seed + '", 1, 0.15' + refprescaleval + ')'
                #               openpath = '   ("Open' + str(hltpath) + '", "' + seed + '", 1, 0.15' + refprescaleval + ')'
                fullpath = '   ("' + str(hltpath) + '", "' + seed + '"' + confdbprescaleval + ', 0.15' + refprescaleval + ')'
                openpath = '   ("Open' + str(hltpath) + '", "' + seed + '"' + confdbprescaleval + ', 0.15' + refprescaleval + ')'
                if(pathcount < npaths):
                    fullpath = fullpath + ','
                    openpath = openpath + ','
               
            if(hltpath.startswith("AlCa_")):
                #               fullpath = '   ("' + str(hltpath) + '", "' + seed + '", 1, 0.' + refprescaleval + ')'
                #               openpath = '   ("Open' + str(hltpath) + '", "' + seed + '", 1, 0.' + refprescaleval + ')'
                fullpath = '   ("' + str(hltpath) + '", "' + seed + '"' + confdbprescaleval + ', 0.15' + refprescaleval + ')'
                openpath = '   ("Open' + str(hltpath) + '", "' + seed + '"' + confdbprescaleval + ', 0.15' + refprescaleval + ')'
                if(pathcount < npaths):
                    fullpath = fullpath + ','
                    openpath = openpath + ','
                                                        
            thefullhltpaths.append(fullpath)
            theopenhltpaths.append(openpath)
            pathcount = pathcount + 1

        rateeffhltcfgfile.write("  # (TriggerName, Prescale, EventSize)" + "\n" + " triggers = (" + "\n" + "#" + "\n") 
        rateeffopenhltcfgfile.write("  # (TriggerName, Prescale, EventSize)" + "\n" + " triggers = (" + "\n" + "#" + "\n")


        # Sort triggers by Primary Dataset
        if(self.pdsort == 1):
            for thepd in thepds:
                rateeffhltcfgfile.write('############# dataset ' + str(thepd) + ' ###############\n')
                rateeffopenhltcfgfile.write('############# dataset ' + str(thepd) + ' ###############\n')
                for rateeffpath in thefullhltpaths:
                    therateeffpath = (rateeffpath.split(',')[0]).split('"')[1]
                    for pdtrigname, pd, in hltpdmap:
                        if (pdtrigname == therateeffpath):
                            if(pd == thepd):
                                rateeffhltcfgfile.write(rateeffpath + "\n")
                for rateeffpath in theopenhltpaths:
                    therateeffpath = (rateeffpath.split(',')[0]).split('"')[1]
                    if(therateeffpath.find("OpenHLT") != -1):
                        therateeffpath = therateeffpath.replace("OpenHLT","HLT")
                    if(therateeffpath.find("OpenAlCa") != -1):
                        therateeffpath = therateeffpath.replace("OpenAlCa","AlCa")
                    for pdtrigname, pd, in hltpdmap:
                        if (pdtrigname == therateeffpath):
                            if(pd == thepd):
                                rateeffopenhltcfgfile.write(rateeffpath + "\n")

        # Sort triggers by the order they appear in the menu 
        else:
            for rateeffpath in thefullhltpaths:
                rateeffhltcfgfile.write(rateeffpath + "\n")
            for rateeffpath in theopenhltpaths:
                rateeffopenhltcfgfile.write(rateeffpath + "\n")

        rateeffhltcfgfile.write("# " + "\n" + " );" + "\n\n")
        rateeffopenhltcfgfile.write("# " + "\n" + " );" + "\n\n")
        rateeffhltcfgfile.write(" # For L1 prescale preloop to be used in HLT mode only" + "\n" + " L1triggers = ( " + "\n" + "#" + "\n")
        rateeffopenhltcfgfile.write(" # For L1 prescale preloop to be used in HLT mode only" + "\n" + " L1triggers = ( " + "\n" + "#" + "\n")

        nbits = len(usedl1bits)
        bitcount = 1

        for usedl1bit in usedl1bits:
            if(usedl1bit in self.l1aliasmap):
                usedl1bit = self.l1aliasmap[usedl1bit]

            if(bitcount < nbits):
                rateeffhltcfgfile.write('  ("' + usedl1bit + '", 1),' + '\n')
                rateeffopenhltcfgfile.write('  ("' + usedl1bit + '", 1),' + '\n')
            else:
                rateeffhltcfgfile.write('  ("' + usedl1bit + '", 1)' + '\n')
                rateeffopenhltcfgfile.write('  ("' + usedl1bit + '", 1)' + '\n')
            bitcount = bitcount+1

        rateeffhltcfgfile.write("# " + "\n" + " );")
        rateeffopenhltcfgfile.write("# " + "\n" + " );")

        linestomerge = rateefforiglibfile.readlines()    
        foundintbit = False
        foundbranch = False
        foundaddress = False
        foundmapping = False

        # Now update the library with any newly-added trigger bit names 
        for intbit in theintbits:
            foundintbit = False
            for linetomerge in linestomerge:
                if(intbit in linetomerge):
                    foundintbit = True
            if foundintbit == False:
                intbitstoadd.append(intbit) 
        for branch in thebranches:
            foundbranch = False
            for linetomerge in linestomerge:
                if(branch in linetomerge):
                    foundbranch = True
            if foundbranch == False:
                branchestoadd.append(branch)
        for address in theaddresses:
            foundaddress = False
            for linetomerge in linestomerge:
                if(address in linetomerge):
                    foundaddress = True
            if foundaddress == False:
                addressestoadd.append(address)
        for mapping in themaps:        
            foundmapping = False
            for linetomerge in linestomerge: 
                if(mapping in linetomerge):
                    foundmapping = True
            if foundmapping == False:
                mapstoadd.append(mapping)

        # Now update the library with any newly-added trigger bit prescale names
        for intpresclbit in theintprescls:
            foundintpresclbit = False
            for linetomerge in linestomerge:
                if(intpresclbit in linetomerge):
                    foundintpresclbit = True
            if foundintpresclbit == False:
                intpresclstoadd.append(intpresclbit)
        for presclbranch in thepresclbranches:
            foundpresclbranch = False
            for linetomerge in linestomerge:
                if(presclbranch in linetomerge):
                    foundpresclbranch = True
            if foundpresclbranch == False:
                presclbranchestoadd.append(presclbranch)
        for prescladdress in theprescladdresses:
            foundprescladdress = False
            for linetomerge in linestomerge:
                if(prescladdress in linetomerge):
                    foundprescladdress = True
            if foundprescladdress == False:
                prescladdressestoadd.append(prescladdress)
        for presclmapping in thepresclmaps:
            foundpresclmapping = False
            for linetomerge in linestomerge:
                if(presclmapping in linetomerge):
                    foundpresclmapping = True
            if foundpresclmapping == False:
                presclmapstoadd.append(presclmapping)

        for linetomerge in linestomerge:
            rateefflibfile.write(linetomerge)
            if(linetomerge.find("Autogenerated from ConfDB - Int_t") != -1):
                for intbittoadd in intbitstoadd:
                    rateefflibfile.write(intbittoadd + "\n")
                    print "Adding trigger " + str(intbittoadd.split("Int_t")[1]).lstrip().rstrip()
            if(linetomerge.find("Autogenerated from ConfDB - TBranch") != -1):
                for branchtoadd in branchestoadd:
                    rateefflibfile.write(branchtoadd + "\n")
            if(linetomerge.find("Autogenerated from ConfDB - SetBranchAddressBits") != -1):
                for addresstoadd in addressestoadd:
                    rateefflibfile.write(addresstoadd + "\n")
            if(linetomerge.find("Autogenerated from ConfDB - SetBranchAddressMaps") != -1):
                for maptoadd in mapstoadd:
                    rateefflibfile.write(maptoadd + "\n")
            if(linetomerge.find("Autogenerated from ConfDB - Prescale Int_t") != -1):
                for intpresclbittoadd in intpresclstoadd:
                    rateefflibfile.write(intpresclbittoadd + "\n")
                    print "Adding prescale bit for trigger " + str(intpresclbittoadd.split("Int_t")[1]).lstrip().rstrip()
            if(linetomerge.find("Autogenerated from ConfDB - Prescale TBranch") != -1):
                for presclbranchtoadd in presclbranchestoadd:
                    rateefflibfile.write(presclbranchtoadd + "\n")
            if(linetomerge.find("Autogenerated from ConfDB - Prescale SetBranchAddressBits") != -1):
                for prescladdresstoadd in prescladdressestoadd:
                    rateefflibfile.write(prescladdresstoadd + "\n")
            if(linetomerge.find("Autogenerated from ConfDB - Prescale SetBranchAddressMaps") != -1):
                for presclmaptoadd in presclmapstoadd:
                    rateefflibfile.write(presclmaptoadd + "\n")
                                                                                                                                                
                    
        rateeffhltcfgfile.close()
        rateeffopenhltcfgfile.close()
        rateefflibfile.close()
        rateefforiglibfile.close()
                
if __name__ == "__main__":
    main(sys.argv[1:])
    
