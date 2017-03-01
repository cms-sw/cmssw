#!/bin/python
import sys

# Parse the output of dump_l1t_configs.py, printing the nonempty records and hash values
# Author: Dustin Burns Oct 22, 2015

# Check inputs
if len(sys.argv) != 2: 
  print('Usage: python parse_dump_l1t_configs.py log_file.txt')
  sys.exit()

# Open log file
f = open(sys.argv[1])

# Loop through log file, printing hashes for records of interest
ind=0
rec = ' '
for i, line in enumerate(f):
  if 'Rcd' in line or 'Record' in line:
    ind = i
    rec = line.split()[0]
  if i == ind+3 and len(line.split())>3:
    print rec + ' ' + ':' + ' ' + line.split()[4]
