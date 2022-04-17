#!/usr/bin/env python3

from __future__ import print_function
import logging
import argparse
import math
import glob
import sys
import os
import re

class BaseLHEMerger(object):
    """Base class of the LHE merge schemes"""

    def __init__(self, input_files, output_file):
        self.input_files = input_files
        self.output_file = output_file

    def merge(self):
        """Output the merged LHE"""
        pass

class DefaultLHEMerger(BaseLHEMerger):
    """Default LHE merge scheme that copies the header of the first LHE file,
    merges and outputs the init block, then concatenates all event blocks."""

    def __init__(self, input_files, output_file, **kwargs):
        super(DefaultLHEMerger, self).__init__(input_files, output_file)

        self.bypass_check = kwargs.get('bypass_check', False)
        # line-by-line iterator for each input file
        self._f = [self.file_iterator(name) for name in self.input_files]
        self._header_str = []
        self._is_mglo = False
        self._xsec_combined = 0.
        self._uwgt = 0.
        self._init_str = [] # initiated blocks for each input file
        self._nevent = [] # number of events for each input file

    def file_iterator(self, path):
        """Line-by-line iterator of a txt file"""
        with open(path, 'r') as f:
            for line in f:
                yield line

    def check_header_compatibility(self):
        """Check if all headers for input files are consistent."""

        if self.bypass_check:
            return

        inconsistent_error_info = ("Incompatibility found in LHE headers: %s. "
                                  "Use -b/--bypass-check to bypass the check.")
        allow_diff_keys = [
            'nevent', 'numevts', 'iseed', 'Seed', 'Random', '.log', '.dat', '.lhe',
            'Number of Events', 'Integrated weight'
            ]
        self._header_lines = [header.split('\n') for header in self._header_str]

        # Iterate over header lines for all input files and check consistency
        logging.debug('header line number: %s' \
            % ', '.join([str(len(lines)) for lines in self._header_lines]))
        assert all([
            len(self._header_lines[0]) == len(lines) for lines in self._header_lines]
            ), inconsistent_error_info % "line number not matches"
        inconsistent_lines_set = [set() for _ in self._header_lines]
        for line_zip in zip(*self._header_lines):
            if any([k in line_zip[0] for k in allow_diff_keys]):
                logging.debug('Captured \'%s\', we allow difference in this line' % line_zip[0])
                continue
            if not all([line_zip[0] == line for line in line_zip]):
                # Ok so meet inconsistency in some lines, then temporarily store them
                for i, line in enumerate(line_zip):
                    inconsistent_lines_set[i].add(line)
        # Those inconsistent lines still match, meaning that it is only a change of order
        assert all([inconsistent_lines_set[0] == lset for lset in inconsistent_lines_set]), \
            inconsistent_error_info % ('{' + ', '.join(inconsistent_lines_set[0]) + '}')

    def merge_headers(self):
        """Merge the headers of input LHEs. Need special handle for the MG5 LO case."""

        self._is_mglo = all(['MGGenerationInfo' in header for header in self._header_str])
        if self._is_mglo and not self.bypass_check:
            # Special handling of MadGraph5 LO LHEs
            match_geninfo = [
                re.search(
                    (r"\<MGGenerationInfo\>\s+#\s*Number of Events\s*\:\s*(\S+)\s+"
                    r"#\s*Integrated weight \(pb\)\s*\:\s*(\S+)\s+\<\/MGGenerationInfo\>"),
                    header
                    ) for header in self._header_str
                ]
            self._xsec_combined = sum(
                [float(info.group(2)) * nevt for info, nevt in zip(match_geninfo, self._nevent)]
                ) / sum(self._nevent)
            geninfo_combined = ("<MGGenerationInfo>\n"
                                "#  Number of Events        : %d\n"
                                "#  Integrated weight (pb)  : %.10f\n</MGGenerationInfo>") \
                                    % (sum(self._nevent), self._xsec_combined)
            logging.info('Detected: MG5 LO LHEs. Input <MGGenerationInfo>:\n\tnevt\txsec')
            for info, nevt in zip(match_geninfo, self._nevent):
                logging.info('\t%d\t%.10f' % (nevt, float(info.group(2))))
            logging.info('Combined <MGGenerationInfo>:\n\t%d\t%.10f' \
                % (sum(self._nevent), self._xsec_combined))

            header_combined = self._header_str[0].replace(match_geninfo[0].group(), geninfo_combined)
            return header_combined

        else:
            # No need to merge the headers
            return self._header_str[0]

    def merge_init_blocks(self):
        """If all <init> blocks are identical, return the same <init> block
        (in the case of Powheg LHEs); otherwise, calculate the output <init>
        blocks by merging the input blocks info using formula (same with the
        MG5LOLHEMerger scheme):
            XSECUP = sum(xsecup * no.events) / tot.events
            XERRUP = sqrt( sum(sigma^2 * no.events^2) ) / tot.events
            XMAXUP = max(xmaxup)
        """

        if self.bypass_check:
            # If bypass the consistency check, simply use the first LHE <init>
            # block as the output
            return self._init_str[0]

        # Initiate collected init block info. Will be in format of
        # {iprocess: [xsecup, xerrup, xmaxup]}
        new_init_block = {}
        old_init_block = [{} for _ in self._init_str]

        # Read the xsecup, xerrup, and xmaxup info from the <init> block for
        # all input LHEs
        for i, bl in enumerate(self._init_str): # loop over files
            nline = int(bl.split('\n')[0].strip().split()[-1])

            # loop over lines in <init> block
            for bl_line in bl.split('\n')[1:nline + 1]:
                bl_line_sp = bl_line.split()
                old_init_block[i][int(bl_line_sp[3])] = [
                    float(bl_line_sp[0]), float(bl_line_sp[1]), float(bl_line_sp[2])]

            # After reading all subprocesses info, store the rest content in
            # <init> block for the first file
            if i == 0:
                info_after_subprocess = bl.strip().split('\n')[nline + 1:]

            logging.info('Input file: %s' % self.input_files[i])
            for ipr in sorted(list(old_init_block[i].keys()), reverse=True):
                # reverse order: follow the MG5 custom
                logging.info('  xsecup, xerrup, xmaxup, lprup: %.6E, %.6E, %.6E, %d' \
                    % tuple(old_init_block[i][ipr] + [ipr]))

        # Adopt smarter <init> block merging method
        # If all <init> blocks from input files are identical, return the same block;
        # otherwise combine them based on MG5LOLHEMerger scheme
        if all([old_init_block[i] == old_init_block[0] for i in range(len(self._f))]):
            # All <init> blocks are identical
            logging.info(
                'All input <init> blocks are identical. Output the same "<init> block.')
            return self._init_str[0]

        # Otherwise, calculate merged init block
        for i in range(len(self._f)):
            for ipr in old_init_block[i]:
                # Initiate the subprocess for the new block if it is found for the
                # first time in one input file
                if ipr not in new_init_block:
                    new_init_block[ipr] = [0., 0., 0.]
                new_init_block[ipr][0] += old_init_block[i][ipr][0] * self._nevent[i] # xsecup
                new_init_block[ipr][1] += old_init_block[i][ipr][1]**2 * self._nevent[i]**2 # xerrup
                new_init_block[ipr][2] = max(new_init_block[ipr][2], old_init_block[i][ipr][2]) # xmaxup
        tot_nevent = sum([self._nevent[i] for i in range(len(self._f))])

        # Write first line of the <init> block (modify the nprocess at the last)
        self._merged_init_str = self._init_str[0].split('\n')[0].strip().rsplit(' ', 1)[0] \
            + ' ' + str(len(new_init_block)) + '\n'
        # Form the merged init block
        logging.info('Output file: %s' % self.output_file)
        for ipr in sorted(list(new_init_block.keys()), reverse=True):
            # reverse order: follow the MG5 custom
            new_init_block[ipr][0] /= tot_nevent
            new_init_block[ipr][1] = math.sqrt(new_init_block[ipr][1]) / tot_nevent
            logging.info('  xsecup, xerrup, xmaxup, lprup: %.6E, %.6E, %.6E, %d' \
                % tuple(new_init_block[ipr] + [ipr]))
            self._merged_init_str += '%.6E %.6E %.6E %d\n' % tuple(new_init_block[ipr] + [ipr])
        self._merged_init_str += '\n'.join(info_after_subprocess)
        if len(info_after_subprocess):
            self._merged_init_str += '\n'

        return self._merged_init_str

    def merge(self):
        with open(self.output_file, 'w') as fw:
            # Read the header for the all input files
            for i in range(len(self._f)):
                header = []
                line = next(self._f[i])
                while not re.search('\s*<init(>|\s)', line):
                    header.append(line)
                    line = next(self._f[i])
                # 'header' includes all contents before reaches <init>
                self._header_str.append(''.join(header))
            self.check_header_compatibility()

            # Read <init> blocks for all input_files
            for i in range(len(self._f)):
                init = []
                line = next(self._f[i])
                while not re.search('\s*</init>', line):
                    init.append(line)
                    line = next(self._f[i])
                # 'init_str' includes all contents inside <init>...</init>
                self._init_str.append(''.join(init))

            # Iterate over all events file-by-file and write events temporarily
            # to .tmp.lhe
            with open('.tmp.lhe', 'w') as _fwtmp:
                for i in range(len(self._f)):
                    nevent = 0
                    while True:
                        line = next(self._f[i])
                        if re.search('\s*</event>', line):
                            nevent += 1
                        if re.search('\s*</LesHouchesEvents>', line):
                            break
                        _fwtmp.write(line)
                    self._nevent.append(nevent)
                    self._f[i].close()

            # Merge the header and init blocks and write to the output
            fw.write(self.merge_headers())
            fw.write('<init>\n' + self.merge_init_blocks() + '</init>\n')

            # Write event blocks in .tmp.lhe back to the output
            # If is MG5 LO LHE, will recalculate the weights based on combined xsec
            # and nevent read from <MGGenerationInfo>, and the 'event_norm' mode
            if self._is_mglo and not self.bypass_check:
                event_norm = re.search(
                    r'\s(\w+)\s*=\s*event_norm\s',
                    self._header_str[0]).group(1)
                if event_norm == 'sum':
                    self._uwgt = self._xsec_combined / sum(self._nevent)
                elif event_norm == 'average':
                    self._uwgt = self._xsec_combined
                logging.info(("MG5 LO LHE with event_norm = %s detected. Will "
                             "recalculate weights in each event block.\n"
                             "Unit weight: %+.7E") % (event_norm, self._uwgt))

                # Modify event wgt when transfering .tmp.lhe to the output file
                event_line = -999
                with open('.tmp.lhe', 'r') as ftmp:
                    sign = lambda x: -1 if x < 0 else 1
                    for line in ftmp:
                        event_line += 1
                        if re.search('\s*<event.*>', line):
                            event_line = 0
                        if event_line == 1:
                            # modify the XWGTUP appeared in the first line of the
                            # <event> block
                            orig_wgt = float(line.split()[2])
                            fw.write(re.sub(r'(^\s*\S+\s+\S+\s+)\S+(.+)', r'\g<1>%+.7E\g<2>' \
                                % (sign(orig_wgt) * self._uwgt), line))
                        elif re.search('\s*<wgt.*>.*</wgt>', line):
                            addi_wgt_str = re.search(r'\<wgt.*\>\s*(\S+)\s*\<\/wgt\>', line).group(1)
                            fw.write(line.replace(
                                addi_wgt_str, '%+.7E' % (float(addi_wgt_str) / orig_wgt * self._uwgt)))
                        else:
                            fw.write(line)
            else:
                # Simply transfer all lines
                with open('.tmp.lhe', 'r') as ftmp:
                    for line in ftmp:
                        fw.write(line)
            fw.write('</LesHouchesEvents>\n')
            os.remove('.tmp.lhe')


class MG5LOLHEMerger(BaseLHEMerger):
    """Use the merger script dedicated for MG5 LO LHEs, as introduced in
    https://github.com/cms-sw/genproductions/blob/master/bin/MadGraph5_aMCatNLO/Utilities/merge.pl
    """

    def __init__(self, input_files, output_file, **kwargs):
        super(MG5LOLHEMerger, self).__init__(input_files, output_file)
        self._merger_script_url = \
            'https://raw.githubusercontent.com/cms-sw/genproductions/5c1e865a6fbe3a762a28363835d9a804c9cf0dbe/bin/MadGraph5_aMCatNLO/Utilities/merge.pl'

    def merge(self):
        logging.info(
            ('Use the merger script in genproductions dedicated for '
            'MadGraph5-produced LHEs'))
        os.system('curl -s -L %s | perl - %s %s.gz banner.txt' \
            % (self._merger_script_url, ' '.join(self.input_files), self.output_file))
        os.system('gzip -df %s.gz' % self.output_file)
        os.system('rm banner.txt')


class ExternalCppLHEMerger(BaseLHEMerger):
    """Use the external mergeLheFiles.cpp file to merge LHE files, as introduced in
    https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideSubgroupMC#1_2_Using_pLHE_campaigns
    """

    def __init__(self, input_files, output_file, **kwargs):
        super(ExternalCppLHEMerger, self).__init__(input_files, output_file)
        self._merger_script_url = \
            'https://twiki.cern.ch/twiki/bin/viewfile/CMSPublic/SWGuideSubgroupMC?filename=mergeLheFiles.cpp;rev=2'

    def merge(self):
        logging.info(
            ('Use the external mergeLheFiles.cpp file to merge LHE files.'))
        os.system('curl -s -o mergeLheFiles.cpp %s' % self._merger_script_url)
        with open('mergeLheFiles.cpp') as f:
            script_str = f.read()
        with open('mergeLheFiles.cpp', 'w') as fw:
            fw.write(script_str.replace('/tmp/covarell/out.lhe', self.output_file))
        with open('input_files.txt', 'w') as fw:
            fw.write('\n'.join(self.input_files) + '\n')

        os.system('g++ -Wall -o mergeLheFiles mergeLheFiles.cpp')
        os.system('./mergeLheFiles input_files.txt')
        os.system('rm mergeLheFiles* input_files.txt')


def main(argv = None):
    """Main routine of the script.

    Arguments:
    - `argv`: arguments passed to the main routine
    """

    if argv == None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description=("A universal script that merges multiple LHE files for all possible conditions and in the most "
                    "natural way.\n"
                    "A detailed description of the merging step (in the default mode):\n"
                    "  1. Header:\n"
                    "    a. assert consistency of the headers (allow difference for the info of e.g. #event, seed);\n"
                    "    b. if not MG LO LHEs, will simply use the header from the first LHE; otherwise, reset the "
                    "<MGGenerationInfo> from the headers by merging the #event & xsec info;\n"
                    "  2. Init block: if all <init> blocks are the same, use the same as output; otherwise (the MG LO "
                    "case), merge them by recalculating the # of subprocess (LRPUP) and XSECUP, XERRUP, XMAXUP per "
                    "each subprocess.\n"
                    "  3. Event block: concatenate all event blocks. If for MG LO LHEs, recalculate the per-event "
                    "XWGTUP and all <wgt> tags based on the new XSECUP, #event, and 'event_norm' read from the MG "
                    "run card.\n"
                    "For further development of this script please always validate the merging result on the test "
                    "routines: https://github.com/colizz/mergelhe_validate\n"
                    "Example usage:\n"
                    "    mergeLHE.py -i 'thread*/*.lhe,another_file/another.lhe' -o output.lhe"),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--input-files", type=str,
                        help="Input LHE file paths separated by commas. Shell-type wildcards are supported.")
    parser.add_argument("-o", "--output-file",
                        default='output.lhe', type=str,
                        help="Output LHE file path.")
    parser.add_argument("--force-mglo-merger", action='store_true',
                        help=("Force to use the merger script dedicated for MG5 LO LHEs, as introduced in "
                             "https://github.com/cms-sw/genproductions/blob/master/bin/MadGraph5_aMCatNLO/Utilities/merge.pl"))
    parser.add_argument("--force-cpp-merger", action='store_true',
                        help=("Force to use the external mergeLheFiles.cpp file to merge LHE files, as introduced in "
                             "https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideSubgroupMC#1_2_Using_pLHE_campaigns"))
    parser.add_argument("-b", "--bypass-check", action='store_true',
                        help=("Bypass the compatibility check for the headers. If true, the header and init block "
                             "will be just a duplicate from the first input file, and events are concatenated without "
                             "modification."))
    parser.add_argument("--debug", action='store_true',
                        help="Use the debug mode.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        format='[%(levelname)s] %(message)s',
        level=logging.INFO if not args.debug else DEBUG)
    logging.info('>>> launch mergeLHE.py in %s' % os.path.abspath(os.getcwd()))

    # Extract input LHE files from the path
    assert len(args.input_files), \
        ('Please specify your input LHE files by -i/--input-files. '
        'Run \'mergeLHE.py -h\' for details.')
    input_files = [] # each individual input file
    for path in args.input_files.split(','):
        find_files = glob.glob(path)
        if len(find_files) == 0:
            logging.info('Warning: cannot find files in %s' % path)
        input_files += find_files
    input_files.sort()
    logging.info('>>> Merge %d files: [%s]' % (len(input_files), ', '.join(input_files)))
    logging.info('>>> Write to output: %s ' % args.output_file)

    if not os.path.exists(os.path.dirname(os.path.realpath(args.output_file))):
        os.makedirs(os.path.dirname(os.path.realpath(args.output_file)))

    # Check arguments
    assert len(input_files) > 0, 'Input LHE files should be more than 0.'
    if len(input_files) == 1:
        logging.warning('Input LHE only has 1 file. Will copy this file to the destination.')
        import shutil
        shutil.copy(input_files[0], args.output_file)
        return
    assert [args.force_mglo_merger, args.force_cpp_merger].count(True) <= 1, \
        "Can only specify at most one from --force-mglo-merger or --force-cpp-merger."

    # Determine the merging scheme
    if args.force_mglo_merger:
        lhe_merger = MG5LOLHEMerger(input_files, args.output_file)
    elif args.force_cpp_merger:
        lhe_merger = ExternalCppLHEMerger(input_files, args.output_file)
    else:
        lhe_merger = DefaultLHEMerger(input_files, args.output_file, bypass_check=args.bypass_check)

    # Do merging
    lhe_merger.merge()


if __name__=="__main__":
    main()
