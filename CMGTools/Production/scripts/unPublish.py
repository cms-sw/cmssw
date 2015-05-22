#!/usr/bin/env python
## Author: Peter Meckiffe
## @ CERN, Meyrin
## April 4th 2011

import os, getpass, sys, re, optparse
from DBSAPI.dbsProcessedDataset import DbsProcessedDataset
from DBSAPI.dbsPrimaryDataset import DbsPrimaryDataset
from datetime import *
from CMGTools.Production.unPublish import unPublish
from optparse import *



if __name__ == '__main__':
    parser = optparse.OptionParser()
    
    parser.usage = """
%prog [options] <sampleName>

Use this script to close dataset tasks on CmgDB and savannah.
Example:
unPublish.py -F cbern /VBF_HToTauTau_M-120_7TeV-powheg-pythia6-tauola/Summer11-PU_S4_START42_V11-v1/AODSIM/V2/PAT_CMG_V2_5_0_Test_v2
"""
    
    group = OptionGroup(parser, "unPublish Options", """These options affect the way you publish to Savannah and CMGDB""")
    genGroup = OptionGroup(parser, "Login Options", """These options apply to your login credentials""")
    
    	
    
    # If user is not specified default is current user
    # This option will be used to find dataset on castor, and assign dataset on savannah
    group.add_option("-F", "--fileown", 
                      dest="fileown",
                      help="User who is the files owner on Castor." ,
                      default=os.environ['USER'] )
    # If specified is used to log in to savannah (only required if user that created the dataset,
    # is different to user publishing it)
    genGroup.add_option("-u", "--username",
                      action = "store",
                      dest="username",
                      help="""Specify the username to access both the DBS and savannah servers. 
Default is $USER.""",
                      default=os.environ['USER'] )

    # If user wants to add multiple datasets from file
    group.add_option("-M", "--multi",
                      action = "store_true",
                      dest="multi",
                      help="""Argument is now LFN to location of .txt file
							Entries in the file should be on independant lines in the form: DatasetName Fileowner
							Comment is not compulsory, and if fileowner is not entered, $USER will be used as default.
							Comment MUST be enclosed in speech marks
							E.g.
							/MuHad/Run2011A-05Aug2011-v1/AOD/V2 cmgtools""",
                      default = False)

    genGroup.add_option("-d", "--dev",
                      action = "store_true",
                      dest="development",
                      help=""".""",
                      default=False )

    parser.add_option_group(genGroup)
    parser.add_option_group(group)
    
    (options, args) = parser.parse_args()
    
    # Allow no more than one argument
    if len(args)!=1:
        parser.print_help()
        sys.exit(1)
        
    # For multiple file input
    if options.multi:
        file = open(args[0], 'r')
        lines = file.readlines()
        for line in lines:
            line = re.sub("\s+", " ", line)
            try:
            	
                dataset = line.split(" ")[0].lstrip().rstrip()
                fileown = options.fileown
                if re.search("%", line):
            		fileown = line.split("%")[0].lstrip().rstrip()
            		dataset = line.split("%")[1].split(" ")[0].lstrip().rstrip()
                elif not re.search("---", dataset):
                	if len(line.lstrip().rstrip().split(" ")) ==1:
                		
                		dataset = line.rstrip("\n").lstrip().rstrip()
                		fileown = options.fileown
                	if len(line.lstrip().rstrip().split(" ")) >1 and re.search("'",line) is None and re.search('"',line) is None:
                		fileown = line.split(" ")[1].rstrip("\n").lstrip().rstrip()
                	elif re.search("'",line):
                		preComment = line.split("'")[0]
                		if len(preComment.lstrip().rstrip().split(" ")) == 2:
                			fileown = preComment.lstrip().rstrip().split(" ")[1]
                		else:
                			fileown = options.fileown
                	elif re.search('"',line):
                		preComment = line.split('"')[0]
                		if len(preComment.lstrip().rstrip().split(" ")) == 2:
                			fileown = preComment.lstrip().rstrip().split(" ")[1]
                		else:
                			fileown = options.fileown
                	else:
                		fileown = options.fileown
                	
                dataset.rstrip("/")
                unPublish(dataset,fileown,options.username,options.development)
            except NameError as err:
                print err.args, "\nDataset not published"
    # For singular file input
    else:
        dataset = args[0].rstrip("/")
        unPublish(dataset,options.fileown,options.username,options.development)

