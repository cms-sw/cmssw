#!/usr/bin/env python

from __future__ import print_function
import logging
import argparse
import sys
import os
import re


if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description="Add numbers to lhe")
    parser.add_argument("input_file", type=str,
                        help="Input LHE file paths separated by commas. Shell-type wildcards are supported.")
    parser.add_argument("-o", "--output-file", default='output.lhe', type=str,
                        help="Output LHE file path.")
    parser.add_argument("--debug", action='store_true',
                        help="Use the debug mode.")
    args = parser.parse_args()

    logging.basicConfig(
        format='[%(levelname)s] %(message)s',
        level=logging.INFO if not args.debug else DEBUG)
    logging.info('>>> launch mergeLHE.py in %s' % os.path.abspath(os.getcwd()))

    logging.info('>>> Input file: [%s]' % args.input_file)
    logging.info('>>> Write to output: %s ' % args.output_file)

    if not os.path.exists(os.path.dirname(os.path.realpath(args.output_file))):
        os.makedirs(os.path.dirname(os.path.realpath(args.output_file)))

    nevent = 0
    with open(args.output_file, 'w') as fw:
        with open(args.input_file, 'r') as ftmp:
            for line in ftmp:
                if re.search('\s*</event>', line):
                    nevent += 1
                    fw.write("<event num> " + str(nevent) +  "</event num>\n")
                fw.write(line)
