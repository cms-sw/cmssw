#!/usr/bin/env python

import sys
import re
import optparse

if __name__ == '__main__':


    parser = optparse.OptionParser('Usage: %prog lumi.csv')

    (options, args) = parser.parse_args()

    if len (args) != 1:
        print "You must provide a CSV file\n"
        parser.print_help()
        sys.exit()

    sepRE = re.compile (r'[\s,;:]+')
    totDelivered = totRecorded = 0.
    events = open (args[0], 'r')
    minRun  = maxRun  = 0
    minLumi = maxLumi = 0
    for line in events:
        pieces = sepRE.split (line.strip())
        if len (pieces) < 4:
                continue
        try:
            run,       lumi     = int  ( pieces[0] ), int  ( pieces[1] )
            delivered, recorded = float( pieces[2] ), float( pieces[3] )
        except:
            continue
        if not minRun or run < minRun:
            minRun  = run
            minLumi = lumi
        if run == minRun and lumi < minLumi:
            minLumi = lumi
        if not maxRun or run > maxRun:
            maxRun  = run
            maxLumi = lumi
        if run == maxRun and lumi > maxLumi:
            maxLumi = lumi
        totDelivered += delivered
        totRecorded  += recorded
    print "Runs (%d, %d) to (%d, %d)" % (minRun, minLumi, maxRun, maxLumi)
    unit = "ub"
    if totRecorded > 1000.:
        totRecorded  /= 1000.
        totDelivered /= 1000.
        unit = "nb"
    if totRecorded > 1000.:
        totRecorded  /= 1000.
        totDelivered /= 1000.
        unit = "pb"
    if totRecorded > 1000.:
        totRecorded  /= 1000.
        totDelivered /= 1000.
        unit = "fb"
    print "Total Delivered %.1f 1/%s  Total Recorded %.1f 1/%s" % \
          (totDelivered, unit, totRecorded, unit)

