#!/usr/bin/env python3

# Original author: Joerg Behr
# Translation from Perl to Python: Gregor Mittag
#
# This script reads the histogram file produced by Pede and it extracts the plot
# showing the average chi2/ndf per Mille binary number.  After reading the MPS
# database, for which the file name has to be provided, an output file called
# chi2pedehis.txt is produced where the first column corresponds to the
# associated name, the second column corresponds to the Mille binary number, and
# the last column is equal to <chi2/ndf>. As further argument this scripts
# expects the file name of the Pede histogram file -- usually millepede.his. The
# last required argument represents the location of the Python config which was
# used by CMSSW.
#
# Use createChi2ndfplot.C to plot the output of this script.

from __future__ import print_function
import os
import sys
import re
import argparse

import Alignment.MillePedeAlignmentAlgorithm.mpslib.tools as mps_tools
import Alignment.MillePedeAlignmentAlgorithm.mpslib.Mpslibclass as mpslib


################################################################################
def main(argv = None):
    """Main routine of the script.

    Arguments:
    - `argv`: arguments passed to the main routine
    """

    if argv == None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Analysis pede histogram file")
    parser.add_argument("-d", "--mps-db", dest="mps_db", required=True,
                        metavar="PATH", help="MPS database file ('mps.db')")
    parser.add_argument("--his", dest="his_file", required=True,
                        metavar="PATH", help="pede histogram file")
    parser.add_argument("-c", "--cfg", dest="cfg", metavar="PATH", required=True,
                        help="python configuration file of pede job")
    parser.add_argument("-b", "--no-binary-check", dest="no_binary_check",
                        default=False, action="store_true",
                        help=("skip check for existing binaries "
                              "(possibly needed if used interactively)"))
    args = parser.parse_args(argv)


    for input_file in (args.mps_db, args.his_file, args.cfg):
        if not os.path.exists(input_file):
            print("Could not find input file:", input_file)
            sys.exit(1)

    ids, names = get_all_ids_names(args.mps_db)
    used_binaries = get_used_binaries(args.cfg, args.no_binary_check)
    his_data = get_his_data(args.his_file)

    if len(his_data) != len(used_binaries):
        print("The number of used binaries is", len(used_binaries), end=' ')
        print("whereas in contrast, however, the <chi2/ndf> histogram in Pede has", end=' ')
        print(len(his_data), "bins (Pede version >= rev92 might help if #bins < #binaries).", end=' ')
        print("Exiting.")
        sys.exit(1)

    with open("chi2pedehis.txt", "w") as f:
        for i, b in enumerate(used_binaries):
            index = ids.index(b)
            name = names[index]
            f.write(" ".join([name, "{:03d}".format(b), his_data[i]])+"\n")


################################################################################
def get_all_ids_names(mps_db):
    """Returns two lists containing the mille job IDs and the associated names.
    
    Arguments:
    - `mps_db`: path to the MPS database file
    """

    lib = mpslib.jobdatabase()
    lib.read_db(mps_db)

    ids = lib.JOBNUMBER[:lib.nJobs]
    names = lib.JOBSP3[:lib.nJobs]

    return ids, names


def get_used_binaries(cfg, no_binary_check):
    """Returns list of used binary IDs.
    
    Arguments:
    - `cfg`: python config used to run the pede job
    - `no_binary_check`: if 'True' a check for file existence is skipped
    """

    cms_process = mps_tools.get_process_object(cfg)

    binaries = cms_process.AlignmentProducer.algoConfig.mergeBinaryFiles
    if no_binary_check:
        used_binaries = binaries
    else:
        # following check works only if 'args.cfg' was run from the same directory:
        used_binaries = [b for b in binaries
                         if os.path.exists(os.path.join(os.path.dirname(cfg), b))]

    used_binaries = [int(re.sub(r"milleBinary(\d+)\.dat", r"\1", b))
                     for b in used_binaries]

    return used_binaries


def get_his_data(his_file):
    """Parse the pede histogram file.
    
    Arguments:
    - `his_file`: pede histogram file
    """
    
    his_data = []
    with open(his_file, "r") as his:
        found_chi2_start = False;
        
        for line in his:
            if r"final <Chi^2/Ndf> from accepted local fits vs file number" in line:
                found_chi2_start = True
            if not found_chi2_start:
                continue
            else:
                if r"end of xy-data" in line: break
                if not re.search("\d", line): continue
                if re.search(r"[a-z]", line): continue
                splitted = line.split()
                his_data.append(splitted[-1])

    return his_data

    
################################################################################
if __name__ == "__main__":
    main()
