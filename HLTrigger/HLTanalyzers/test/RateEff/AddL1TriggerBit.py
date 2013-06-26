#!/usr/bin/env python

# GetL1BitsForSkim.py
#
# Simple script for getting all L1 seeds used by a particular HLT menu

import os, string, sys, posix, tokenize, array, getopt
from pkgutil import extend_path
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.Modules as mod


def main(argv):

    input_l1name = ""

    opts, args = getopt.getopt(sys.argv[1:], "l:h", ["l1name=","help"])

    for o, a in opts:
        if o in ("-l","l1name="):
            input_l1name = str(a)
        if o in ("-h","help"):
            print "-l <L1 bit> (Name of the L1 bit to add)"
            print "-h (Print the help menu)"
            return

    confdbjob = AddL1TriggerBit(input_l1name)
    bailout = confdbjob.BeginJob()
    if(bailout == -2):
        print "Bailing out!"
        return


    os.system("mv OHltTree_AddL1.h OHltTree.h")
    
            
class AddL1TriggerBit:
    def __init__(self,clil1name):

        self.l1name = clil1name

    def BeginJob(self):

        thepaths = []
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
        
        rateefflibfile = open("OHltTree_AddL1.h",'w')
        rateefforiglibfilescan = open("OHltTree.h")
        rateefforiglibfile = open("OHltTree.h")

        # First check if the L1 already exists...
        scanlines = rateefforiglibfilescan.readlines()
        scanl1 = self.l1name + ';'
        print "Looking for " + scanl1
        for scanline in scanlines:
            if(scanline.find(scanl1) != -1):
                print self.l1name + " already exists - exiting!"
                return -2 
                                                    

        # Now we have all the information, construct any configuration/branch changes
        theintbits.append('  Int_t           ' + self.l1name + ';')
        thebranches.append('  TBranch        *b_' + self.l1name + ';   //!')
        theaddresses.append('  fChain->SetBranchAddress("' + self.l1name + '", &' + self.l1name + ', &b_' + self.l1name + ');')
        themaps.append('  fChain->SetBranchAddress("' + self.l1name + '", &map_BitOfStandardHLTPath["' + self.l1name + '"], &b_' + self.l1name + ');')
        theintprescls.append('  Int_t           ' + self.l1name + '_Prescl;')
        thepresclbranches.append('  TBranch        *b_' + self.l1name + '_Prescl;   //!')
        theprescladdresses.append('  fChain->SetBranchAddress("' + self.l1name + '_Prescl", &' + self.l1name + '_Prescl, &b_' + self.l1name + '_Prescl);')
        thepresclmaps.append('  fChain->SetBranchAddress("' + self.l1name + '_Prescl", &map_RefPrescaleOfStandardHLTPath["' + self.l1name + '"], &b_' + self.l1name + '_Prescl);')
                
        pathcount = 1

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
            if(linetomerge.find("Autogenerated L1 - Int_t") != -1):
                for intbittoadd in intbitstoadd:
                    rateefflibfile.write(intbittoadd + "\n")
                    print "Adding trigger " + str(intbittoadd.split("Int_t")[1]).lstrip().rstrip()
            if(linetomerge.find("Autogenerated L1 - TBranch") != -1):
                for branchtoadd in branchestoadd:
                    rateefflibfile.write(branchtoadd + "\n")
            if(linetomerge.find("Autogenerated L1 - SetBranchAddressBits") != -1):
                for addresstoadd in addressestoadd:
                    rateefflibfile.write(addresstoadd + "\n")
            if(linetomerge.find("Autogenerated L1 - SetBranchAddressMaps") != -1):
                for maptoadd in mapstoadd:
                    rateefflibfile.write(maptoadd + "\n")
            if(linetomerge.find("Autogenerated L1 - Prescale Int_t") != -1):
                for intpresclbittoadd in intpresclstoadd:
                    rateefflibfile.write(intpresclbittoadd + "\n")
                    print "Adding prescale bit for trigger " + str(intpresclbittoadd.split("Int_t")[1]).lstrip().rstrip()
            if(linetomerge.find("Autogenerated L1 - Prescale TBranch") != -1):
                for presclbranchtoadd in presclbranchestoadd:
                    rateefflibfile.write(presclbranchtoadd + "\n")
            if(linetomerge.find("Autogenerated L1 - Prescale SetBranchAddressBits") != -1):
                for prescladdresstoadd in prescladdressestoadd:
                    rateefflibfile.write(prescladdresstoadd + "\n")
            if(linetomerge.find("Autogenerated L1 - Prescale SetBranchAddressMaps") != -1):
                for presclmaptoadd in presclmapstoadd:
                    rateefflibfile.write(presclmaptoadd + "\n")
                                                                                                                                                
        rateefflibfile.close()
        rateefforiglibfile.close()
                
if __name__ == "__main__":
    main(sys.argv[1:])
    
