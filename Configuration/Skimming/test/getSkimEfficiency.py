#!/usr/bin/env python

import subprocess as sp
import sys
import os
import optparse
import re
import string

parser = optparse.OptionParser(description='Calculate Skim Efficiency')
parser.add_option ("--input_dir", "-i", dest = "input_dir", type=str, help = "input directory")
parser.add_option ("--number_events"    , "-n", dest = "number_events"    , type=str, help = "number of events processed")
(options, args) = parser.parse_args()

if not options.input_dir        : parser.print_help(); sys.exit()
if not options.number_events    : parser.print_help(); sys.exit()

input_dir = options.input_dir
number_events = options.number_events


#------------------------------------------------
# Step 1: loop over PD folders in input directory
#------------------------------------------------

ls_command = "ls " + input_dir

listPDs_output = ""
listPDs_output, listPDs_stderr  = sp.Popen ( ls_command , shell=True, stdout=sp.PIPE ).communicate()

#print listPDs_output

if listPDs_output == "":
    print "\n"
    print "ERROR! Can't find the directory or the directory is empty:", input_dir
    print "\n"
    sys.exit()


#-------------------------------------------------------
# create matrix
#-------------------------------------------------------
skimMatrix = {}
column=[]
lineCounter = int(0)

listPDs_output_lines = listPDs_output.split("\n")
#print listPDs_output_lines
del listPDs_output_lines[-1]
#print listPDs_output_lines

for l, PD in enumerate(listPDs_output_lines):
    PD = PD.strip()
    print "PD:",  l, PD
    skimMatrix[PD] = {}

    ls_command_two = "ls " + input_dir + "/" + PD + " | grep .root"
    listSkims_output, listSkims_stderr  = sp.Popen ( ls_command_two , shell=True, stdout=sp.PIPE ).communicate()
    listSkims_output_files = listSkims_output.split("\n")
    del listSkims_output_files[-1]
    #print listSkims_output_files
    
    for f, file in enumerate(listSkims_output_files):
        skim = string.split(file,".root")[0]
        #print "  SKIM:", f, skim

        ## search for number of events in skim
        file_path = input_dir + "/" + PD + "/" + file
        edm_command = "edmEventSize -v " + file_path 
        #print edm_command
        edmCommand_output, edmCommand_stderr  = sp.Popen ( edm_command , shell=True, stdout=sp.PIPE , stderr=sp.PIPE ).communicate()
        #print edmCommand_output
        edmCommand_output_lines = edmCommand_output.split("\n")
        #print edmCommand_output_lines
        if "contains no Events" in edmCommand_stderr :
            numberOfEventsInSkim = 0
        else:
            numberOfEventsInSkim = (float(string.split(edmCommand_output_lines[1]," ")[3]) / float(number_events))*100
        #print numberOfEventsInSkim
        skimMatrix[PD][skim] = numberOfEventsInSkim

    print skimMatrix[PD] , "\n"

#print skimMatrix

