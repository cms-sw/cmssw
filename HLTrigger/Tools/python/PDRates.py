#! /usr/bin/env python
from __future__ import print_function
import sys, imp
import os
import commands

from optparse import OptionParser


import six
# --
# -- Usage :
# -- Rate within a given PD :
# --       ./PDRates.py  --runNumber 135525 --PD Mu
# -- Plot the rate of a PD versus the LS number:
# --       ./PDRates.py  --runNumber 135525 --PD Mu --perLS
# -- Rates in all PDs:
# --       ./PDRates.py  --runNumber 135525
# --


parser = OptionParser(usage="Example: ./PDRates.py --runNumber 146644 --PD Mu --perLS")
parser.add_option("--runNumber",  dest="RunNumber",  help="run number", type="int", metavar="RunNumber")
parser.add_option("--PD",dest="PrimaryDataset",help="PD name",metavar="PrimaryDataset")
parser.add_option("--perLS",action="store_true",default=False,dest="perLS",help="plot the rate vs the LS number",metavar="perLS")
parser.add_option("--logy",action="store_true",default=False,dest="logy",help="log scale for the y axis",metavar="logy")
parser.add_option("--saveplot",action="store_true",default=False,dest="saveplot",help="save the plot as a tmp.eps",metavar="saveplot")
(options, args) = parser.parse_args()


def PrimaryDatasets(Run):
    # -- List of Primary Datasets:

    dbs_cmd = """ dbs search --query='find primds.name 
          	where run=%d and dataset like */RAW' """ % (Run)
    rows = commands.getoutput(dbs_cmd)
    lines = rows.split("\n")
    j=0
    print("\nThe primary datasets for this run are: \n")
    for Line in lines:
        j=j+1
        if j <=4:
            continue
        print(Line)
        line=Line.split()
        Datasets.append(line[0])
    print(" ")



def RateInPD(Run,PrimaryDataset,lsMin,lsMax,printLS=False):
    dbs_cmd = """ dbs search --query='find file,lumi,file.numevents, file.size
		where run=%d and dataset like /%s/*/RAW 
		and lumi >= %d and lumi <= %d
		and file.status=VALID '""" % (Run, PrimaryDataset,lsMin, lsMax)
    rows = commands.getoutput(dbs_cmd)
    lines = rows.split("\n")
    j=0
    LumiSections = []
    Files = []
    Evts = 0
    Size = 0
    LSinFile = {}
    NumberOfLSInFile = {}
    for Line in lines:
        j=j+1
        if j <=4:
            continue
        line=Line.split()
        LS = line[1]
        file = line[0]
        Nevts = int(line[2])
        size = int(line[3])
        LSinFile[LS] = file
        if LumiSections.count(LS) == 0:
            LumiSections.append(LS)
        if Files.count(file) == 0:
            Files.append(file)
            Evts += Nevts
            Size += size
            NumberOfLSInFile[file] =1
        else:
            NumberOfLSInFile[file] = NumberOfLSInFile[file]+1
        RatePerLS[LS] = Nevts

    Number_of_LS = len(LumiSections)
    LS_Length = 23.3
    if Run < 125100:
        LS_Length = 93.3
    rate = Evts / (Number_of_LS * LS_Length)
    if Evts > 0:
        size_per_event = (Size / Evts) / 1000.
    else:
        size_per_event=-1
    print("Rate in \t",PrimaryDataset,"\t is : \t",rate," Hz", " \t size is : \t",size_per_event, "kB / event ")

    lsmin=9999999
    lsmax=-1
    for (LS,file) in six.iteritems(LSinFile):
        nls = NumberOfLSInFile[file]
        RatePerLS[LS] = RatePerLS[LS] / nls
        RatePerLS[LS] = RatePerLS[LS] / LS_Length
        if int(LS) > lsmax:
            lsmax=int(LS)
        if int(LS) < lsmin:
            lsmin=int(LS)
    if printLS:
        print("lsmin lsmax",lsmin,lsmax)
        for ls in range(lsmin,lsmax):
            if not repr(ls) in RatePerLS.keys():
                RatePerLS[LS] = 0
                print("Missing LS ",ls)


if __name__ == "__main__":

    if not options.RunNumber:
        print("wrong usage")
        exit(2)

    lsMin = -1
    lsMax = 999999

# -- does not work yet.
#  if options.lsMin:
#	lsMin = options.lsMin
#  if options.lsMax:
#	lsMax = options.lsMax


    print("\nRun Number: ",options.RunNumber)
    if options.PrimaryDataset:
        Run = options.RunNumber
        PrimaryDataset = options.PrimaryDataset
        RatePerLS = {}
        if not options.perLS:
            RateInPD(Run,PrimaryDataset,lsMin, lsMax, False)
        if options.perLS:
            average = 0
            nLS_within_range = 0
            RateInPD(Run,PrimaryDataset,lsMin, lsMax, True)
            RatesTmp = open("rates_tmp.txt","w")
            #RatesTmpSort = open("rates_tmp_sort.txt","w")
            for (LS, rate) in six.iteritems(RatePerLS):
                RatesTmp.write(LS+"\t"+repr(rate)+"\n")
                #if int(LS) >=  lsMin and int(LS) <= lsMax:
                #nLS_within_range =nLS_within_range +1
                #average = average + rate
            #print "Average rate within ",options.lsMin," and ",options.lsMax," is: ",average/nLS_within_range
            #if os.path.exists("./rates_tmp_sort.txt"):
                #os.system("rm rates_tmp_sort.txt")
            #os.system("sort -n rates_tmp.txt > rates_tmp_sort.txt")
            RatesTmp.close()

            TempFile = open("tmp.gnuplot.txt","w")
            if options.logy:
                TempFile.write("set logscale y \n")
            if options.saveplot:
                TempFile.write("set term postscript eps enhanced \n")
                TempFile.write("set output \"tmp.eps\" \n")
            st_title = " \"Rates in PrimaryDataset " + PrimaryDataset+ " in Run " + repr(Run)+ "\" "
            TempFile.write("set title " + st_title + "\n")
            TempFile.write(" set pointsize 2. \n")
            TempFile.write(" set nokey \n")
            TempFile.write(" set xlabel \"LS number\" \n")
            TempFile.write(" set ylabel \"Rate (Hz)\" \n")
            TempFile.write(" plot \"rates_tmp.txt\" using 1:2 title 'Rate per LS' \n")
            if not options.saveplot:
                TempFile.write(" pause -1")
            else:
                print("The plot is saved under tmp.eps")
            TempFile.close()

            os.system("gnuplot tmp.gnuplot.txt")


    else:
        Run = options.RunNumber
        Datasets = []
        PrimaryDatasets(Run)
        for PrimaryDataset in Datasets:
            RatePerLS = {}
            RateInPD(Run,PrimaryDataset,lsMin,lsMax)



