#! /usr/bin/env python

""" This macro can be used to merge the information from the accurate HV
map derived from pedestal runs studies with the old db map into a new
accurate map.
"""

import sys

print "Reading psu-detId map from map.txt"

inputMap = open("map.txt", "r")

outputFile = open("newMap.txt", "w")

for line in inputMap:
    # 369120285         cms_trk_dcs_05:CAEN/CMS_TRACKER_SY1527_4/branchController09/easyCrate1/easyBoard14/channel000
    # print dir(line)
    splittedLine = line.rsplit("/", 1)
    # print splittedLine[0]+"/"
    detId = splittedLine[0].split()[0]

    channel = ""

    HVmap = open("HVmap.txt", "r")
    for HVline in HVmap:
        if detId in HVline:
            channel = HVline.split()[1].strip()

    if channel == "" or "undefined" in channel:
        if channel == "":
            print "channel not found for detId = ", detId
        else:
            print "channel is undefined in HV map ",
        print "leaving channel 0"
        outputFile.write(line)
        # break
    else:
        # print splittedLine[0]+"/"+channel
        outputFile.write(splittedLine[0]+"/"+channel+"\n")

    # break
