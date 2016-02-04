import optparse
import os

usage="--list"
parser = optparse.OptionParser(usage)
parser.add_option("--GT")
parser.add_option("--option",default="")
(options,args)=parser.parse_args()

from Configuration.Skimming.autoSkim import autoSkim
for PD in autoSkim:
    com='cmsDriver.py skim -s SKIM:%s --data --conditions %s --python_filenam skim_%s.py --magField AutoFromDBCurrent --no_exec %s'%(autoSkim[PD],options.GT,PD,options.option)            
    if 'cosmic' in PD.lower():
        com+=' --scenario cosmics'
    print com
    os.system(com)

