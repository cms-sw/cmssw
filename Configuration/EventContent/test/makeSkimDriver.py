#! /usr/bin/env python

r'''
The Wrapper for makeSkim.py, the general config for cmsRun.
'''

import optparse
import os

def _green(string):
    return '%s%s%s' %('\033[1;32m',string,'\033[1;0m') 

# To parse commandline args

usage='%prog -i inrootfile -o outrootfile -n number_of_events --outputcommands name_of_block'

parser=optparse.OptionParser(usage)


parser.add_option("-n", "--number",
                   help="The number of evts. The default is 50.",
                   default="50",
                   dest="nevts")

parser.add_option("-i",
                   help="The infile name",
                   default="",
                   dest="infilename")                  
                   
parser.add_option("-o",
                   help="The outfile name",
                   default="",
                   dest="outfilename")       
                   
parser.add_option("--outputcommands",
                   help='The outputcommands (i.e. RECOSIMEventContent, '+\
                   'AODSIMEventContent, FEVTSIMEventContent, '+\
                   'AODEventContent and all blocks in EventContent.cff)',
                   default="",
                   dest="outputCommands")

options,args=parser.parse_args() 

if '' in (options.infilename,
          options.outfilename,
          options.outputCommands):
    raise ('Incomplete list of arguments!')
                                      
# Construct and dump the metaconfiguration on disk as a python file
metaconfig_content='nevts=%s\n' %options.nevts+\
                   'outputCommands="%s"\n' %options.outputCommands+\
                   'infile="%s"\n' %options.infilename+\
                   'outfile="%s"\n' %options.outfilename

metaconfig_file=open('metaconfig.py','w')
metaconfig_file.write(metaconfig_content) 
metaconfig_file.close()

# print it to screen!
print _green('\nThe Metaconfiguration:\n')
print metaconfig_content


# and now execute!
command='cmsRun ./makeSkim.py'
print _green('\nAnd now run %s ...\n' %command)
os.environ['PYTHONPATH']+=':./' # to find the metaconfig..
os.system(command)



                      
                                                            
