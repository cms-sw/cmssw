#!/usr/bin/env python
import re
import argparse
import math
import fileinput

# Set up argrument parser
helpEpilog = ''''''

parser = argparse.ArgumentParser(description='Take card file, blank all INFI directives and insert the INFI directives from the modifier file instead.', 
                                 epilog=helpEpilog, 
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('inCfg', action='store',
                    help='name of the config-template')
parser.add_argument('modCfg', action='store',
                    help='name of the modifier file from mps_split')
parser.add_argument('outCfg', action='store',
                    help='name of modified output file')
parser.add_argument('isn', action='store',
                    help='number of the job (three digit number with leading zeros)')

# Parse arguments
args = parser.parse_args()
inCfg  = args.inCfg
modCfg = args.modCfg
outCfg = args.outCfg
isn    = args.isn


# open input file
with open(inCfg, 'r') as INFILE:
	body = INFILE.read()

# read modifier file
with open(modCfg, 'r') as MODFILE:
	mods = MODFILE.read()
mods = mods.strip()

# prepare the new fileNames directive. Delete first line if necessary.
fileNames = mods.split('\n')
if 'CastorPool=' in fileNames[0]:
	del fileNames[0]

# replace ISN number (input is a string of three digit number with leading zeros though)
body = re.sub(re.compile('ISN',re.M), isn, body)

# print to outCfg
with open(outCfg, 'w') as OUTFILE:
	OUTFILE.write(body)

# Number of total files and number of extends for cms.vstring are needed
numberOfFiles   = len(fileNames)
numberOfExtends = int(math.ceil(numberOfFiles/255.))

# Create and insert the readFile.extend lines
for j in xrange(numberOfExtends):
	insertBlock = "readFiles.extend([\n    "	
	i=0
	currentStart = j*255
	while (i<255) and ((currentStart+i)<numberOfFiles):
		entry = fileNames[currentStart+i].strip()
		if (i==254) or ((currentStart+i+1)==numberOfFiles):
		    insertBlock += "\'"+entry+"\'])\n"
		else:
			insertBlock += "\'"+entry+"\',\n    "
		i+=1
	
	for line in fileinput.input(outCfg, inplace=1):
		if re.match('readFiles\s*=\s*cms.untracked.vstring()',line):
			print line,
			print insertBlock,
		else:
			print line,








