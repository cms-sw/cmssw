#!/usr/bin/env python

from __future__ import print_function
import argparse
import math
import glob
import sys
import os

class BaseLHEMerger(object):
    """Base class of the LHE merge schemes"""
    def __init__(self, inputFiles, outputFileName):
        self.inputFiles = inputFiles
        self.outputFileName = outputFileName

    def merge(self):
        """Output the merged LHE"""
        pass

class DefaultLHEMerger(BaseLHEMerger):
    """Default LHE merge scheme that copies the header of the first LHE file, merges and outputs the init block, then concatenates all event blocks."""
    def __init__(self, inputFiles, outputFileName):
        super(DefaultLHEMerger, self).__init__(inputFiles, outputFileName)
        self._f = [self.file_iterator(name) for name in self.inputFiles] # line-by-line iterator for each input file
        self._init_str = [] # initiated blocks for each input file
        self._nevent = []   # number of events for each input file

    def file_iterator(self, path):
        """Line-by-line iterator of a txt file"""
        with open(path, 'r') as f:
            for line in f:
                yield line

    def merge_init_blocks(self):
        """Calculate the output init blocks by merging the input init block info
        Formula used (same with the MadGraph5LHEMerger scheme):
            XSECUP = sum(xsecup * no.events) / tot.events
            XERRUP = sqrt( sum(sigma^2 * no.events^2) ) / tot.events
            XMAXUP = max(xmaxup)
        """
        # Initiate collected init block info. Will be in format of {iprocess: [xsecup, xerrup, xmaxup]}
        new_init_block = {}
        old_init_block = [{} for _ in self._init_str]

        # Read init block from input LHEs
        for i, bl in enumerate(self._init_str): # loop over files
            nline = int(bl.split('\n')[0].strip().split()[-1])
            for bl_line in bl.split('\n')[1:nline + 1]: # loop over lines in <init> block
                bl_line_sp = bl_line.split()
                old_init_block[i][int(bl_line_sp[3])] = [float(bl_line_sp[0]), float(bl_line_sp[1]), float(bl_line_sp[2])]
            # After reading all subprocesses info, store the rest content in <init> block for the first file
            if i == 0:
                info_after_subprocess = bl.strip().split('\n')[nline + 1:]
            print('Input file: %s' % self.inputFiles[i])
            for ipr in sorted(list(old_init_block[i].keys()), reverse=True): # reverse order: follow the MG5 custom
                print('  xsecup, xerrup, xmaxup, lprup: %.6E, %.6E, %.6E, %d' % tuple(old_init_block[i][ipr] + [ipr]))

        # Calculate merged init block
        for i in range(len(self._f)):
            for ipr in old_init_block[i]:
                # Initiate the subprocess for the new block if it is found for the first time in one input file
                if ipr not in new_init_block:
                    new_init_block[ipr] = [0., 0., 0.]
                new_init_block[ipr][0] += old_init_block[i][ipr][0] * self._nevent[i] # xsec
                new_init_block[ipr][1] += old_init_block[i][ipr][1]**2 * self._nevent[i]**2 # xerrup
                new_init_block[ipr][2] = max(new_init_block[ipr][2], old_init_block[i][ipr][2]) # xmaxup
        tot_nevent = sum([self._nevent[i] for i in range(len(self._f))])

        # Write first line of the <init> block (modify the nprocess at the last)
        self._merged_init_str = self._init_str[0].split('\n')[0].strip().rsplit(' ', 1)[0] + ' ' + str(len(new_init_block)) + '\n'
        # Form the merged init block
        print('\nOutput file: %s' % self.outputFileName)
        for ipr in sorted(list(new_init_block.keys()), reverse=True): # reverse order: follow the MG5 custom
            new_init_block[ipr][0] /= tot_nevent
            new_init_block[ipr][1] = math.sqrt(new_init_block[ipr][1]) / tot_nevent
            print('  xsecup, xerrup, xmaxup, lprup: %.6E, %.6E, %.6E, %d' % tuple(new_init_block[ipr] + [ipr]))
            self._merged_init_str += '%.6E %.6E %.6E %d\n' % tuple(new_init_block[ipr] + [ipr])
        self._merged_init_str += '\n'.join(info_after_subprocess)
        if len(info_after_subprocess):
            self._merged_init_str += '\n'

        return self._merged_init_str

    def merge(self):
        with open(self.outputFileName, 'w') as fw:
            # Read the header for the first file then write to the output
            while True:
                line = next(self._f[0])
                if line.startswith('<init>'):
                    break
                else:
                    fw.write(line)

            # Let all inputFiles reach the start of <init> block
            for i in range(1, len(self._f)):
                while(not next(self._f[i]).startswith('<init>')):
                    pass

            # Read <init> blocks for all inputFiles
            for i in range(len(self._f)):
                init_str = ''
                while True:
                    line = next(self._f[i])
                    if line.startswith('</init>'):
                        break
                    else:
                        init_str += line
                self._init_str.append(init_str)

            # Iterate over all events file-by-file and write events temporarily to .tmp.lhe
            with open('.tmp.lhe', 'w') as _fwtmp:
                for i in range(len(self._f)):
                    nevent = 0
                    while True:
                        line = next(self._f[i])
                        if line.startswith('</event>'):
                            nevent += 1
                        if line.startswith('</LesHouchesEvents>'):
                            break
                        else:
                            _fwtmp.write(line)
                    self._nevent.append(nevent)
                    self._f[i].close()

            # Merge the init blocks and write to the output
            fw.write('<init>\n' + self.merge_init_blocks() + '</init>\n')

            # Write event blocks in .tmp.lhe back to the output
            with open('.tmp.lhe', 'r') as _ftmp:
                for _line in _ftmp:
                    fw.write(_line)
            fw.write('</LesHouchesEvents>\n')
            os.remove('.tmp.lhe')


class MadGraph5LHEMerger(BaseLHEMerger):
    """Use the merger script in genproductions dedicated for MadGraph5-produced LHEs"""
    def __init__(self, inputFiles, outputFileName):
        super(MadGraph5LHEMerger, self).__init__(inputFiles, outputFileName)
        self._merger_script_url = 'https://raw.githubusercontent.com/cms-sw/genproductions/5c1e865a6fbe3a762a28363835d9a804c9cf0dbe/bin/MadGraph5_aMCatNLO/Utilities/merge.pl'

    def merge(self):
        print('Info: use the merger script in genproductions dedicated for MadGraph5-produced LHEs')
        os.system('curl -L %s | perl - %s %s.gz .banner.txt' % (self._merger_script_url, ' '.join(self.inputFiles), self.outputFileName))
        os.system('gzip -df %s.gz' % self.outputFileName)


def main(argv = None):
    """Main routine of the script.

    Arguments:
    - `argv`: arguments passed to the main routine
    """

    if argv == None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Merge specified input LHE files (three available modes).")
    parser.add_argument("-i", "--inputPaths", dest="inputPaths",
                        required=True, type=str, help="input LHE path(s) separated by comma. Shell-type wildcards are supported. E.g. -i 'a/*.lhe,b/*.lhe'")
    parser.add_argument("-o", "--outputFileName", dest="outputFileName",
                        default='output.lhe', type=str, help="output LHE name (should not be a path).")
    parser.add_argument("--forceMadGraph5LHEMerger", dest="forceMadGraph5LHEMerger",
                        action='store_true',
                        help="Force to use the merger script in genproductions dedicated for MadGraph5-produced LHEs.")
    args = parser.parse_args(argv)

    print('>>> launch mergeLHE.py', os.path.abspath(os.getcwd()))
    inputFiles = []
    for path in args.inputPaths.split(','):
        findFiles = glob.glob(path)
        if len(findFiles) == 0:
            print('Warning: cannot find files in %s' % path)
        inputFiles += findFiles
    inputFiles.sort()
    print('>>> Merge %d files: [%s]' % (len(inputFiles), ', '.join(inputFiles)))

    # Check arguments
    assert len(inputFiles) > 1, 'Input LHE files should be more than 1.'
    assert '/' not in args.outputFileName, "outputFileName should not be a path."

    # Determine the merging scheme
    if args.forceMadGraph5LHEMerger:
        lhe_merger = MadGraph5LHEMerger(inputFiles, args.outputFileName)
    else:
        lhe_merger = DefaultLHEMerger(inputFiles, args.outputFileName)

    # Do merging
    lhe_merger.merge()

if __name__=="__main__":
    main()
