#!/usr/bin/env python3
import os
import argparse
import subprocess
from multiprocessing import Pool


def gunzip(file):
    with open(file + '.gunzipped', 'wb') as out:
        command = ['gunzip', '-c', '-S', '.pb', file]
        process = subprocess.Popen(command, stdout=out)
        process.wait()


def concatenate(files, output, gzip):
    """ Protobuf files can just be concatenated one after the other.

    For details information on how protocol buffers format works please read the comments in:
    DQMServices/DQMGUI/python/protobuf/protobuf_parser.py
    """

    # Un gzip files first if needed
    if gzip:
        pool = Pool(len(files))
        pool.map(gunzip, files)
        files = [x + '.gunzipped' for x in files]

    # Clear the output file
    command = 'cat /dev/null > %s' % output
    subprocess.call(command, shell=True)

    # Concatenate file
    command = 'cat %s >> %s' % (' '.join(files), output)
    subprocess.call(command, shell=True)

    # Remove ungzipped files
    if gzip:
        command = 'rm %s' % ' '.join(files)
        subprocess.call(command, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
        '''Utility to concatenate DQM Protobuf files saved in DQMServices/DQMGUI/python/protobuf/ROOTFilePB.proto format.
        This utility DOES NOT merge the histograms!!! If histograms with the same name appear in multiple input files, all of them
        will also appear in the resulting file as duplicates.
        It is assumed that input files are not gzipped. If input files are gzipped, pass --gzip option.
        Resulting file is saved not gzipped.''')
    parser.add_argument('files', nargs='+', help='PB files to be concatenated.')
    parser.add_argument('-o', '--output', default='concatenated_streamDQMHistograms.pb', help='Name of the resulting file.')
    parser.add_argument('-g', '--gzip', action='store_true', help='If set, input files will be ungzipped.')
    args = parser.parse_args()

    files = args.files
    output = args.output
    gzip = args.gzip

    concatenate(files, output, gzip)
