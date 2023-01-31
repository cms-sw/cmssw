#!/usr/bin/python3
import argparse
import pprint
import json
import os
parser = argparse.ArgumentParser()
parser.add_argument('version',action="store",help="Software version")
results = parser.parse_args()
FileName = '/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/'+results.version+'/src/DPGAnalysis/HcalTools/scripts/rmt/mycurrentlist'
FileNameSorted = '/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMScript/'+results.version+'/src/DPGAnalysis/HcalTools/scripts/rmt/currentlist'
with open(FileName,'r') as first_file:
    rows = first_file.readlines()
    sorted_rows = sorted(rows, key=lambda x: int(x.split('PEDESTAL_')[1]), reverse=True)
    with open(FileNameSorted,'w') as second_file:
        for row in sorted_rows:
            second_file.write(row)
