#!/usr/bin/env python3

import sys
import argparse
import Alignment.MillePedeAlignmentAlgorithm.mpslib.tools as mps_tools
import six

################################################################################
def main(argv = None):
    """Main routine of the script.

    Arguments:
    - `argv`: arguments passed to the main routine
    """

    if argv == None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Prepare input db file for MillePede workflow.")
    parser.add_argument("-g", "--global-tag", dest="global_tag", required=True,
                        metavar="TAG",
                        help="global tag to extract the alignment payloads")
    parser.add_argument("-r", "--run-number", dest="run_number", required=True,
                        metavar="INTEGER", type=int,
                        help="run number to select IOV")
    parser.add_argument("-o", "--output-db", dest="output_db",
                        default="alignment_input.db", metavar="PATH",
                        help="name of the output file (default: '%(default)s')")
    args = parser.parse_args(argv)

    inputs = mps_tools.get_tags(args.global_tag,
                                ["TrackerAlignmentRcd",
                                 "TrackerSurfaceDeformationRcd",
                                 "TrackerAlignmentErrorExtendedRcd"])
    for inp in six.itervalues(inputs):
        inp["iovs"] = mps_tools.get_iovs(inp["connect"], inp["tag"])
    mps_tools.create_single_iov_db(inputs, args.run_number, args.output_db)


################################################################################
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
