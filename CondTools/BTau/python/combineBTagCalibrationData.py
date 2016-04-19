#!/usr/bin/env python

import os
import sys
import itertools
import checkBTagCalibrationConsistency as checker


def check_csv_data(csv_data):
    res = checker.run_check_csv(csv_data, False, False, False)
    if not all(res):
        print 'Checks on csv data failed. Exit.'
        exit(-1)


def main():
    if len(sys.argv) < 4:
        print 'Need at least two input- and one output-filename. Exit.'
        exit(-1)
    if os.path.exists(sys.argv[-1]):
        print 'Output file exists. Exit.'
        exit(-1)

    all_csv_data = dict()
    header = None
    tagger = None
    for fname in sys.argv[1:-1]:
        with open(fname) as f:
            all_csv_data[fname] = f.readlines()
        header = all_csv_data[fname].pop(0)
        tggr = header.split('/')[0]
        if tagger and tggr != tagger:
            print 'Found different taggers: %s vs. %s Exit.' % (tagger, tggr)
            exit(-1)
        else:
            tagger = tggr

    print '\n' + '='*80
    print 'Checking consistency of individual input files...'
    print '='*80
    for fname, csv_data in all_csv_data.iteritems():
        print '\nChecking file:', fname
        print '='*80
        check_csv_data(csv_data)

    print '\n' + '='*80
    print 'Checking consistency of combinations...'
    print '='*80
    for one, two in itertools.combinations(all_csv_data.iteritems(), 2):
        print '\nChecking combination:', one[0], two[0]
        print '='*80
        check_csv_data(one[1] + two[1])

    print '\nCombining data...'
    print '='*80
    with open(sys.argv[-1], 'w') as f:
        f.write(header)
        for csv_data in all_csv_data.itervalues():
            f.write('\n')
            f.writelines(csv_data)

    print 'Done.'


if __name__ == '__main__':
    main()
