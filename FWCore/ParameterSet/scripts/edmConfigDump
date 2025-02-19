#! /usr/bin/env python

from optparse import OptionParser
import sys
import os 
import imp

parser = OptionParser()
parser.usage = "%prog <file> : expand this python configuration"

(options,args) = parser.parse_args()

if len(args)!=1:
    parser.print_help()
    sys.exit(1)

filename = args[0]
handle = open(filename, 'r')
cfo = imp.load_source("pycfg", filename, handle)
cmsProcess = cfo.process
handle.close()

print cmsProcess.dumpPython()
