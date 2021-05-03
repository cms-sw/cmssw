#!/usr/bin/env python

from __future__ import print_function
import argparse
import glob
import sys
import os

def main(argv = None):
    """Main routine of the script.

    Arguments:
    - `argv`: arguments passed to the main routine
    """

    if argv == None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Merge specified input LHE files")
    parser.add_argument("-i", "--input", dest="input",
                        type=str, help="input LHE path(s) separated by comma. Shell-type wildcards are supported.")
    parser.add_argument("-o", "--output", dest="output",
                        default='output.lhe', type=str, help="output LHE path.")
    parser.add_argument("--maxEvents", dest="maxEvents",
                        default=-1, type=int,
                        help="Maximum number of events to process.")
    args = parser.parse_args(argv)

    print('>>> launch mergeLHE.py', os.path.abspath(os.getcwd()))
    inputfiles = []
    for path in args.input.split(','):
        findfiles = glob.glob(path)
        if len(findfiles) == 0:
            print('Warning: cannot find files in %s' % path)
        inputfiles += findfiles
    print('>>> Merge %d files: [%s]' % (len(inputfiles), ', '.join(inputfiles)))

    run_script = '''
import FWCore.ParameterSet.Config as cms

process = cms.Process("LHE")

process.source = cms.Source("LHESource",
	fileNames = cms.untracked.vstring({paths})
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32({maxEvents}))

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)

process.writer = cms.EDAnalyzer("LHEWriter")

process.p = cms.Path(process.writer)
    '''.format(paths=','.join(['"file:%s"'%p for p in inputfiles]), maxEvents=args.maxEvents)

    with open('mergeLHE_cff.py', 'w') as f:
        f.write(run_script)

    os.system('cmsRun mergeLHE_cff.py')
    os.rename('writer.lhe', args.output)
    os.remove('mergeLHE_cff.py')

if __name__=="__main__":
    main()
