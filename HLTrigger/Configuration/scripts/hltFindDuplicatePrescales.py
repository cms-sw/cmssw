#! /usr/bin/env python

import sys
import imp

if len(sys.argv) != 2:
    print "usage: %s menu.py" % sys.argv[0]
    sys.exit(1)

# load the menu and get the "process" object
menu = imp.new_module('menu')
name = sys.argv[1]
execfile(name, globals(), menu.__dict__)
process = menu.process

# get all paths
paths = process.paths_()

# keep track of precaler names, and of duplicates
prescalerNames = set()
duplicateNames = set()

# loop over all paths and look for duplicate prescaler modules
for path in paths:
    for module in paths[path].moduleNames():
        if module in process.filters_(): # found a filter
            if process.filters_()[module].type_() == 'HLTPrescaler': # it's a prescaler
                label = process.filters_()[module].label()
                if label in prescalerNames:
                    duplicateNames.add(label)
                else:
                    prescalerNames.add(label)

# print the duplicate prescales, and the affected paths
for label in duplicateNames:
    print 'ERROR: prescale  "%s"  is shared by the paths' % label
    for path in paths:
        if label in paths[path].moduleNames():
            print '\t%s' % path
    print

