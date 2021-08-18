#!/usr/bin/env python3
'''
'''

__author__ = 'Giacomo Govi'

import sys
import os

import CondCore.Utilities.o2olib as o2olib
import optparse
import argparse
    
def main( argv ):

    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=str, help="wrapper for O2O jobs")
    parser.add_argument("-n","--name", type=str, help="the O2O job name" )
    parser.add_argument("--db", type=str, help="the target database: pro ( for prod ) or dev ( for prep ). default=pro")
    parser.add_argument("-a","--auth", type=str,  help="path of the authentication file")
    parser.add_argument("-v","--verbose", action="count", help="job output mirrored to screen (default=logfile only)")
    args = parser.parse_args()  

    tool = o2olib.O2OTool()
    tool.setup(args)
    return tool.run()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
