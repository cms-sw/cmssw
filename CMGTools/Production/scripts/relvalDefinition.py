#!/usr/bin/env python

from CMGTools.Production.relvalDefinition import *

if __name__ == '__main__':
    import sys
    dataset = sys.argv[1]
    rd = relvalDefinition( dataset )
    print rd
