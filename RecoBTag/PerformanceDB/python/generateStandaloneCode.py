#!/usr/bin/env python

import os
from os.path import join


def main():
    path_formats = join(os.environ['CMSSW_BASE'],
                        'src/CondFormats/BTauObjects')
    path_tools = join(os.environ['CMSSW_BASE'],
                      'src/RecoBTag/PerformanceDB/test')

    #  headers
    file_h = join(path_tools, 'BTagCalibrationStandalone.h')
    print 'Creating', file_h
    with open(file_h, 'w') as fout:
        for fname in ['BTagEntry.h',
                      'BTagCalibration.h',
                      'BTagCalibrationReader.h']:
            with open(join(path_formats, 'interface', fname)) as fin:
                for line in fin:
                    if (line.startswith('#include "CondFormats') or
                        'COND_SERIALIZABLE' in line):
                        continue
                    fout.write(line)
            fout.write('\n\n')

    # implementation
    file_cc = join(path_tools, 'BTagCalibrationStandalone.cc')
    print 'Creating', file_cc
    with open(file_cc, 'w') as fout:
        fout.write('#include "BTagCalibrationStandalone.h"\n')
        fout.write('#include <iostream>\n')
        fout.write('#include <exception>\n')
        for fname in ['BTagEntry.cc',
                      'BTagCalibration.cc',
                      'BTagCalibrationReader.cc']:
            with open(join(path_formats, 'src', fname)) as fin:
                err_on_line = -3
                for line_no, line in enumerate(fin):
                    if (line.startswith('#include "CondFormats') or
                        line.startswith('#include "FWCore')):
                        continue
                    elif 'throw cms::Exception' in line:
                        line = 'std::cerr << "ERROR in BTagCalibration: "\n'
                        err_on_line = line_no
                    elif line_no == err_on_line + 2:
                        line += 'throw std::exception();\n'
                    fout.write(line)
            fout.write('\n\n')


if __name__ == '__main__':
    main()
