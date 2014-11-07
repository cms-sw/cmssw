import os


def main():
    #  headers
    print 'Creating test/BTagCalibrationStandalone.h'
    with open('test/BTagCalibrationStandalone.h', 'w') as fout:
        for fname in ['BTagEntry.h',
                      'BTagCalibration.h',
                      'BTagCalibrationReader.h']:
            with open('interface/'+fname) as fin:
                for line in fin:
                    if (line.startswith('#include "CondFormats') or
                        'COND_SERIALIZABLE' in line):
                        continue
                    fout.write(line)
            fout.write('\n\n')

    # implementation
    print 'Creating test/BTagCalibrationStandalone.cc'
    with open('test/BTagCalibrationStandalone.cc', 'w') as fout:
        fout.write('#include "BTagCalibrationStandalone.h"\n')
        for fname in ['BTagEntry.cc',
                      'BTagCalibration.cc',
                      'BTagCalibrationReader.cc']:
            with open('src/'+fname) as fin:
                for line in fin:
                    if line.startswith('#include "CondFormats'):
                        continue
                    fout.write(line)
            fout.write('\n\n')


if __name__ == '__main__':
    if not (os.path.exists('interface/BTagEntry.h') and
            os.path.exists('src/BTagEntry.cc')):
        print 'This script must be executed in CondFormats/BTagObjects. Exit.'
    main()
