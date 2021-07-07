#!/usr/bin/env python3

# This script compares the writer output with the reader output.
# - run the SiStripBadComponentsDQMService.py > logReader
# - open the log and format it so that it only has lines of the form:
# -- detid = INT, flag = INT
# - run the SiStripBadStripReader_cfg.py > logReader
# - again format the log file
# Run this script passing the two log files as input parameters.
# The output will be all the matching lines and a final count of them
# The check is positive if the total number of lines matches the total number of detids in the log

from __future__ import print_function
import sys

fileIN = open(sys.argv[1], "r")
line = fileIN.readline()

matchCount = 0

while line:
    # print line.split()[2].strip(',')
    # print line.split()[5].strip(',')

    fileIN2 = open(sys.argv[2], "r")
    line2 = fileIN2.readline()
    detId = int(line.split()[2].strip(','))
    flag = int(line.split()[5])
    matching = 0
    while line2:
        if( detId == int(line2.split()[2].strip(',')) ):
            if( flag == int(line2.split()[5]) ):
                print("matching:", end=' ')
                print("detId1 = ", detId, " detId2 = ", line2.split()[2].strip(','), end=' ')
                print("flag1 = ", flag, "flag2 = ", line2.split()[5])
                matching = 1
                matchCount += 1
                break
        line2 = fileIN2.readline()
    if( matching == 0 ):
        print("no match found")


    line = fileIN.readline()

print("MatchCount = ", matchCount)
