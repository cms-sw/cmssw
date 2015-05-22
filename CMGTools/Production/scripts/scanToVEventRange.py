#!/usr/bin/env python

from CMGTools.Production.scanToVEventRange import *

if __name__ == '__main__':
    # eventRanges = scanToVEventRange( testString )

    print 'paste your TTree::Scan text below. It should like:'
    print testString

    lines = []
    while input!='':
        input = raw_input()
        lines.append( input )

    lines.pop() # removing last empty line
    
    #    print 'got input'
    # import pprint 
    # pprint.pprint(lines)

    eventRanges = scanToVEventRange( lines )
    
    print eventRanges
