import subprocess as sp
import sys
import os
import optparse

parser = optparse.OptionParser(description='Compare output of edm command and a txt file')
parser.add_option ("--txt_file", "-t", dest = "txt_file_name", type=str, help = "txt file with trigger info")
parser.add_option ("--menu"    , "-m", dest = "menu_name"    , type=str, help = "HLT menu name")
parser.add_option ("--db"    , "-d", dest = "db_name"    , type=str, help = "DB type: put orcoff or hltdev")
parser.add_option ("--stream_name"    , "-s", dest = "stream_name"    , type=str, help = "Stream name: A, B, etc..")
(options, args) = parser.parse_args()

if not options.txt_file_name: parser.print_help(); sys.exit()
if not options.menu_name    : parser.print_help(); sys.exit()
if not options.db_name      : parser.print_help(); sys.exit()
if not options.stream_name  : parser.print_help(); sys.exit()

txt_file_name = options.txt_file_name
menu_name = options.menu_name
db_name = options.db_name
stream_name = options.stream_name

if not os.path.isfile ( txt_file_name ): 
    print "ERROR! This file does not exist:", txt_file_name
    sys.exit() 

#-------------------------------------------------------
# Step 1: get output from shell command via subprocess.Popen 
#-------------------------------------------------------

edmConfigFromDB_output = ""
#edmConfigFromDB_command = "edmConfigFromDB --"+ db_name + " --configName " + menu_name + " --streams A | hltDumpStream"
edmConfigFromDB_command = "edmConfigFromDB --"+ db_name + " --configName " + menu_name + " --streams " + stream_name + " | hltDumpStream"
edmConfigFromDB_output, edmConfigFromDB_stderr  = sp.Popen ( edmConfigFromDB_command, shell=True, stdout=sp.PIPE ).communicate()

if edmConfigFromDB_output == "":
    print "\n"
    print "ERROR! Can't retrieve information about this menu:", menu_name
    print "       I used this command:", edmConfigFromDB_command
    print "\n"
    sys.exit()

#-------------------------------------------------------
# split output into lines and loop over them
#-------------------------------------------------------

edmConfigFromDB_output_lines = edmConfigFromDB_output.split("\n") 

all_edmCommand_streams = []
all_edmCommand_datasets = [] 
all_edmCommand_triggers = []

last_stream = ""
last_dataset =""
last_trigger = ""

d_edmCommand_stream_dataset_triggers = {} 

for line in edmConfigFromDB_output_lines:
    if "*** missing paths in the output module ***" in line: break
    
    # Get rid of extra white space in the line
    line = line.strip()

    #-------------------------------------------------------
    # Deal with streams
    #-------------------------------------------------------
    
    if line[0:6] == "stream" : 
        last_stream = line.split("stream")[1].strip()
        
        all_edmCommand_streams.append (last_stream)
        
        if last_stream not in d_edmCommand_stream_dataset_triggers.keys():
            d_edmCommand_stream_dataset_triggers[last_stream] = {} 
        else:
            print "ERROR!  Streams should not appear more than once, but I found this stream twice in the edm command:", last_stream
            sys.exit() 

    #-------------------------------------------------------
    # Deal with datasets
    #-------------------------------------------------------

    elif line[0:7] == "dataset":
        last_dataset = line.split("dataset")[1].strip()
        
        all_edmCommand_datasets.append ( last_dataset )
        
        if last_dataset not in d_edmCommand_stream_dataset_triggers[last_stream].keys():
            d_edmCommand_stream_dataset_triggers[last_stream][last_dataset] = []
        else:
            print "ERROR!  Datasets should not appear more than once within a stream, but I found this dataset twice in the edm command:", last_dataset
            print "        This was in stream", last_stream
            sys.exit() 

    #-------------------------------------------------------
    # Deal with triggers
    #-------------------------------------------------------

    elif line[0:3] == "HLT":
        trigger_name_fields = line.split()[0].split("_")
        trigger_name = ""
        for trigger_name_field in trigger_name_fields[:-1]:
            trigger_name = trigger_name + trigger_name_field + "_"
        trigger_name = trigger_name[:-1]
        last_trigger = trigger_name.strip()

        all_edmCommand_triggers.append ( last_trigger ) 
        
        if last_trigger not in d_edmCommand_stream_dataset_triggers[last_stream][last_dataset]:
            d_edmCommand_stream_dataset_triggers[last_stream][last_dataset].append ( last_trigger ) 
        else: 
            print "ERROR!  Triggers should not appear more than once within a dataset, but I found this trigger twice in the edm command:", last_trigger
            print "        This was in dataset", last_dataset
            print "        This was in stream", last_stream
            sys.exit()
            
#-------------------------------------------------------
# Now analyze the txt file
#-------------------------------------------------------

txt_file = open ( txt_file_name, "r" )

d_txtFile_dataset_triggers = {} 
all_txtFile_datasets = []
all_txtFile_triggers = []

last_stream = ""
last_dataset =""
last_trigger = ""

for line in txt_file:
    line = line.strip()
    if line == "" : continue
    if "#" in line: continue

    
    #-------------------------------------------------------
    # Deal with datasets
    #-------------------------------------------------------

    if ":" in line:
        last_dataset = line.split(":")[0].strip()
        all_txtFile_datasets.append ( last_dataset )

        if last_dataset not in d_txtFile_dataset_triggers.keys():
            d_txtFile_dataset_triggers[last_dataset] = []
        else:
            print "ERROR!  Datasets should not appear more than once within a stream, but I found this dataset twice in the txt file:", last_dataset
            sys.exit() 

    #-------------------------------------------------------
    # Deal with triggers
    #-------------------------------------------------------
    
    elif "HLT_" in line:
        trigger_name_fields = line.split()[0].split("_")
        trigger_name = ""
        for trigger_name_field in trigger_name_fields[:-1]:
            trigger_name = trigger_name + trigger_name_field + "_"
        trigger_name = trigger_name[:-1]
        last_trigger = trigger_name.strip()

        if last_trigger not in d_txtFile_dataset_triggers[last_dataset]:
            d_txtFile_dataset_triggers[last_dataset].append ( last_trigger ) 
        else: 
            print "ERROR!  Triggers should not appear more than once within a dataset, but I found this trigger twice in the txt file:", last_trigger
            print "        This was in dataset", last_dataset
            sys.exit()
        
#-------------------------------------------------------
# Now we have a dictionary that maps stream + dataset -> list of triggers from the txt file:
#   -  d_txtFile_dataset_triggers
# We also have the full lists of streams, datasets, and triggers from the txt file:
#   -  all_txtFile_streams
#   -  all_txtFile_datasets
#   -  all_txtFile_triggers
# 
# 
# 
# Now we have a dictionary that maps stream + dataset -> list of triggers from the edm commnad:
#   -  d_edmCommand_stream_dataset_triggers
# We also have the full lists of streams, datasets, and triggers from the edm command:
#   -  all_edmCommand_streams
#   -  all_edmCommand_datasets
#   -  all_edmCommand_triggers
#-------------------------------------------------------

#-------------------------------------------------------
# Look for datasets that don't match
#-------------------------------------------------------

print "\n"
print "***********************************************************"
print "*** This is a summary of the datasets that don't match *** "
print "***********************************************************"
print "\n"

datasets_in_txtFile_notIn_edmCommand = [] 
datasets_notIn_txtFile_in_edmCommand = [] 

print "I found these", len(all_txtFile_datasets), "datasets in the txt file:"

for txtFile_dataset in all_txtFile_datasets:
    print "\t", txtFile_dataset
    if txtFile_dataset not in all_edmCommand_datasets:
        datasets_in_txtFile_notIn_edmCommand.append ( txtFile_dataset ) 

print "I found these", len(all_edmCommand_datasets), "datasets in the edm command output:"

for edmCommand_dataset in all_edmCommand_datasets:
    print "\t", edmCommand_dataset
    if edmCommand_dataset not in all_txtFile_datasets:
        datasets_notIn_txtFile_in_edmCommand.append ( edmCommand_dataset )

print "These", len ( datasets_in_txtFile_notIn_edmCommand ), "files in the txt file that were not in the edm command output:"
for dataset_in_txtFile_notIn_edmCommand in datasets_in_txtFile_notIn_edmCommand:
    print "\t", dataset_in_txtFile_notIn_edmCommand

print "These", len ( datasets_notIn_txtFile_in_edmCommand ), "files in the edm command that were not in the txt file"
for dataset_notIn_txtFile_in_edmCommand in datasets_notIn_txtFile_in_edmCommand:
    print "\t", dataset_notIn_txtFile_in_edmCommand

#-------------------------------------------------------
# Look for triggers that don't match
#-------------------------------------------------------

print "\n"
print "************************************************************************************"
print "*** These triggers were in the txt file, but they were not in the edm command:   ***"
print "************************************************************************************"
print "\n"

for txt_file_dataset in all_txtFile_datasets:
    
    if txt_file_dataset not in all_edmCommand_datasets:
        print "\t", txt_file_dataset, "(whole dataset present in txt file, missing from edm command): "
        for trigger in d_txtFile_dataset_triggers[txt_file_dataset]:
            print "\t\t", trigger

    else:
        txt_file_dataset_triggers    = d_txtFile_dataset_triggers           [txt_file_dataset]
#        edm_command_dataset_triggers = d_edmCommand_stream_dataset_triggers ["A"][txt_file_dataset]
        edm_command_dataset_triggers = d_edmCommand_stream_dataset_triggers [stream_name][txt_file_dataset]
        
        found_trigger = False

        for txt_file_dataset_trigger in txt_file_dataset_triggers:
            if txt_file_dataset_trigger not in edm_command_dataset_triggers: 
                found_trigger = True
        
        if found_trigger:
            print "\t", txt_file_dataset
            for txt_file_dataset_trigger in txt_file_dataset_triggers:
                if txt_file_dataset_trigger not in edm_command_dataset_triggers: 
                    print "\t\t", txt_file_dataset_trigger
        else:
            print "\t", txt_file_dataset, "(all triggers found in edm command output)"


print "\n"
print "************************************************************************************"
print "*** These triggers were in the edm command, but they were not in the txt file:   ***"
print "************************************************************************************"
print "\n"


for edmCommand_dataset in all_edmCommand_datasets:
    
    if edmCommand_dataset not in all_txtFile_datasets:
        print "\t", edmCommand_dataset, "(whole dataset present in edm command, missing from txt file): "
        #for trigger in d_edmCommand_stream_dataset_triggers["A"][edmCommand_dataset]:
        for trigger in d_edmCommand_stream_dataset_triggers[stream_name][edmCommand_dataset]:
            print "\t\t", trigger

    else:
        txt_file_dataset_triggers    = d_txtFile_dataset_triggers           [edmCommand_dataset]
        #edm_command_dataset_triggers = d_edmCommand_stream_dataset_triggers ["A"][edmCommand_dataset]
        edm_command_dataset_triggers = d_edmCommand_stream_dataset_triggers [stream_name][edmCommand_dataset]
        
        found_trigger = False

        for edm_command_dataset_trigger in edm_command_dataset_triggers:
            if edm_command_dataset_trigger not in txt_file_dataset_triggers: 
                found_trigger = True
        
        if found_trigger:
            print "\t", edmCommand_dataset
            for edm_command_dataset_trigger in edm_command_dataset_triggers:
                if edm_command_dataset_trigger not in txt_file_dataset_triggers: 
                    print "\t\t", edm_command_dataset_trigger
        else:
            print "\t", edmCommand_dataset, "(all triggers found in txt file)"

