#!/usr/bin/env python
from Page1Parser import Page1Parser
import sys
import os
import cPickle as pickle
import getopt
from TablePrint import *

WBMPageTemplate = "http://cmswbm/cmsdb/servlet/TriggerMode?KEY=l1_hlt_collisions/v%s"

def usage():
    print "%s [Options] KeyVersion" % sys.argv[0]
    print "--IgnoreCols=<cols>               List of columns to ignore from the prescale checker (format is 1,2,3,4 etc.)"

    
def main():
    try:
        opt, args = getopt.getopt(sys.argv[1:],"",["IgnoreCols="])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(2)

    if len(args)<1:
        usage()
        sys.exit(2)

    IgnoreCols=[]
    for o,a in opt:
        if o == "--IgnoreCols":
            tmp = a.split(',')
            try:
                for e in tmp:
                    IgnoreCols.append(int(e))
            except:
                print "Invalid argument to '--IgnoreCols' "
                sys.exit(2)
        else:
            print "Invalid option "+o
            usage()
            sys.exit(0)
            
    WBMPage = WBMPageTemplate % args[0]
    ## Parse the key page
    Parser = Page1Parser()
    Parser._Parse(WBMPage)
    Parser.ParseTrigModePage()
    Parser.ComputeTotalPrescales()

    Header=["Path Name", "L1 Seed"]+Parser.ColumnLumi
    ColWidths=[70,30]+[10]*len(Parser.ColumnLumi)
    print """
    TOTAL L1*HLT PRESCALE TABLE:
    """
    PrettyPrintTable(Header,Parser.TotalPrescaleTable,ColWidths)
    
    print """
    Weird Looking L1*HLT Prescales

    WARNING: paths seeded by the OR of several L1 bits may not be calculated properly (they assume an L1 prescale of 1 in all columns)
    """

    PrettyPrintTable(Header,findAnomalies(Parser.TotalPrescaleTable,IgnoreCols),ColWidths)
    ## OK, we need some more checks here, but to first order this is useful

def TrendingWithLumi(ColLumi,PrescaleTable):
    RatioTable=[]
    for line in PrescaleTable:
        name = line[0]
        l1 = line[1]
        prescales = line[2:]
        ratios=[]
        for lumi,ps in zip(ColLumi,prescales):
            if ps>0:
                ratios.append(lumi/ps)
            else:
                ratios.append(0)
        RatioTable.append([name,l1]+ratios)
    return RatioTable

def isMonotonic(array, ignoreCols):  # return 0 if not, 1 if true and 2 if the array is constant
    lastEntry = array[0]
    returnVal=2
    for entry,i in zip(array[0:],range(len(array[0:]))):
        if i in ignoreCols:
            continue
        if lastEntry<entry and lastEntry!=0:
            return 0
        if lastEntry!=entry:
            returnVal=1
        lastEntry=entry
    return returnVal

def findAnomalies(PrescaleTable,ignoreCols):
    anomalies=[]
    for line in PrescaleTable:
        ps = line[2:]
        if not isMonotonic(ps,ignoreCols):
            anomalies.append(line)
    return anomalies
if __name__=='__main__':
    main()
