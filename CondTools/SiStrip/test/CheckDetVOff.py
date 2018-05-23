#!/usr/bin/env python

'''
This script checks the outputs from SiStripDetVOffFakeBuilder and reader. It compares the status of all detIds
both for low and high voltage and it checks that the values written in the database are correctly read back.
'''

import os

def getDetIds(fileName, vType):
    return os.popen("cat "+fileName+" | grep \"detid\" | grep \""+vType+"\" | grep \"OFF\" | awk \'{print $2}\' ", "r")


def compare(vType):
    #builderChild = os.popen("cat SiStripDetVOffFakeBuilder.log | grep \"detid\" | grep \"HV\" | awk \'{print $2}\' ", "r")
    builderChild = getDetIds("SiStripDetVOffFakeBuilder.log", vType)
    builderOutput = builderChild.read()

    #readerChild = os.popen("cat SiStripDetVOffReader.log | grep \"detid\" | grep \"HV\" | grep \"OFF\" | awk \'{print $2}\' ", "r")
    readerChild = getDetIds("SiStripDetVOffReader.log", vType)
    readerOutput = readerChild.read()

    builderDetIds = builderOutput.split('\n')
    readerDetIds = readerOutput.split('\n')

    builderDetIds = sorted(builderDetIds)

    #builderLine = popen

    i = 0
    printNum = 5
    print "Checking", vType, ":"
    print "printing the first ", printNum, " for comparison"
    for detId in builderDetIds:
        #if( i < 1000 ):
        # print "detId = ", detId
        builderDetId = detId
        #print "DetId = ", builderDetIds[i].split()
        readerDetId = readerDetIds[i]
        # print "builderDetId = ", readerDetId
        if( builderDetId ):
            readerDetId = readerDetIds[i]
            # builderDetId = builderDetIds[i].split()[0]           
            if( readerDetId != builderDetId ):
                print "does not match: builder = ", detId, " reader = ", readerDetIds[i]
            if( i < printNum ):
                print "builder = ", detId
                print "reader = ", readerDetIds[i]
            i += 1
    print


builderFile = open("SiStripDetVOffFakeBuilder.log")
readerFile = open("SiStripDetVOffReader.log")

compare("HV")
compare("LV")
compare(" V")
