#!/usr/bin/env python

import os
from os.path import join


def main():
    path_cms = join(os.environ['CMSSW_BASE'], 'src')
    path_tools = join(path_cms, 'CondTools/BTau/test')

    #  headers
    file_h = join(path_tools, 'BTagCalibrationStandalone.h')
    print 'Creating', file_h
    with open(file_h, 'w') as fout:
        for fname in ['CondFormats/BTauObjects/interface/BTagEntry.h',
                      'CondFormats/BTauObjects/interface/BTagCalibration.h',
                      'CondTools/BTau/interface/BTagCalibrationReader.h']:
            with open(join(path_cms, fname)) as fin:
                for line in fin:
                    if (line.startswith('#include "Cond') or
                        'COND_SERIALIZABLE' in line):
                        continue
                    fout.write(line)
            fout.write('\n\n')

    # implementation
    file_cc = join(path_tools, 'BTagCalibrationStandalone.cpp')
    print 'Creating', file_cc
    with open(file_cc, 'w') as fout:
        fout.write('#include "BTagCalibrationStandalone.h"\n')
        fout.write('#include <iostream>\n')
        fout.write('#include <exception>\n')
        for fname in ['CondFormats/BTauObjects/src/BTagEntry.cc',
                      'CondFormats/BTauObjects/src/BTagCalibration.cc',
                      'CondTools/BTau/src/BTagCalibrationReader.cc']:
            with open(join(path_cms, fname)) as fin:
                err_on_line = -3
                for line_no, line in enumerate(fin):
                    if (line.startswith('#include "Cond') or
                        line.startswith('#include "FWCore')):
                        continue

                    # cms exceptions cannot be used in the standalone version
                    # in the cpp source, exceptions must be formatted to span three lines, where
                    # the first and last lines are replaced.
                    elif 'throw cms::Exception' in line:
                        line = 'std::cerr << "ERROR in BTagCalibration: "\n'
                        err_on_line = line_no
                    elif line_no == err_on_line + 2:
                        line += 'throw std::exception();\n'
                    fout.write(line)
            fout.write('\n\n')


if __name__ == '__main__':
    main()
