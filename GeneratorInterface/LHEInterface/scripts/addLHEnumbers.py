#!/usr/bin/env python3

from __future__ import print_function
import logging
import argparse
import sys
import os
import re


def number_events(input_file, output_file=None, offset=0):
    if output_file is None:
        output_file = input_file
    if not os.path.exists(os.path.dirname(os.path.realpath(output_file))):
        os.makedirs(os.path.dirname(os.path.realpath(output_file)))

    nevent = offset
    with open('tmp.txt', 'w') as fw:
        with open(input_file, 'r') as ftmp:
            for line in ftmp:
                if re.search('\s*</event>', line):
                    nevent += 1
                    fw.write('<event_num num="' + str(nevent) +  '"> ' + str(nevent) + '</event_num>\n')
                fw.write(line)
    if output_file is not None:
        os.rename("tmp.txt", output_file)
    else:
        os.rename("tmp.txt", input_file)
    return nevent


if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description="Add numbers to lhe")
    parser.add_argument("input_file", type=str,
                        help="Input LHE file path.")
    parser.add_argument("-o", "--output-file", default=None, type=str,
                        help="Output LHE file path. If not specified, output to input file")
    args = parser.parse_args()

    logging.info('>>> launch addLHEnumbers.py in %s' % os.path.abspath(os.getcwd()))

    logging.info('>>> Input file: [%s]' % args.input_file)
    logging.info('>>> Write to output: %s ' % args.output_file)

    number_events(args.input_file, args.output_file)