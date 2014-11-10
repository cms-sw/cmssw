import os
from os.path import join

def main():
    path_formats = join(os.environ['CMSSW_BASE'],
                        'src/CondFormats/BTagObjects')
    path_tools = join(os.environ['CMSSW_BASE'],
                      'src/CondTools/BTag/test')

    #  headers
    print 'Creating CondTools/BTag/test/BTagCalibrationStandalone.h'
    with open(join(path_tools, 'BTagCalibrationStandalone.h'), 'w') as fout:
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
    print 'Creating CondTools/BTag/test/BTagCalibrationStandalone.cc'
    with open(join(path_tools, 'BTagCalibrationStandalone.cc'), 'w') as fout:
        fout.write('#include "BTagCalibrationStandalone.h"\n')
        for fname in ['BTagEntry.cc',
                      'BTagCalibration.cc',
                      'BTagCalibrationReader.cc']:
            with open(join(path_formats, 'src', fname)) as fin:
                for line in fin:
                    if line.startswith('#include "CondFormats'):
                        continue
                    fout.write(line)
            fout.write('\n\n')


if __name__ == '__main__':
    main()
