if __name__ == '__main__':
    from optparse import OptionParser
    from PhysicsTools.HeppyCore.framework.heppy_loop import main

    parser = OptionParser()
    parser.usage = """
    %prog <name> <analysis_cfg>
    For each component, start a Loop.
    'name' is whatever you want.
    """

    parser.add_option("-N", "--nevents",
                      dest="nevents",
                      type="int",
                      help="number of events to process",
                      default=None)
    parser.add_option("-p", "--nprint",
                      dest="nprint",
                      help="number of events to print at the beginning",
                      default=5)
    parser.add_option("-e", "--iEvent", 
                      dest="iEvent",
                      help="jump to a given event. ignored in multiprocessing.",
                      default=None)
    parser.add_option("-f", "--force",
                      dest="force",
                      action='store_true',
                      help="don't ask questions in case output directory already exists.",
                      default=False)
    parser.add_option("-i", "--interactive", 
                      dest="interactive",
                      action='store_true',
                      help="stay in the command line prompt instead of exiting",
                      default=False)
    parser.add_option("-t", "--timereport", 
                      dest="timeReport",
                      action='store_true',
                      help="Make a report of the time used by each analyzer",
                      default=False)
    parser.add_option("-v", "--verbose",
                      dest="verbose",
                      action='store_true',
                      help="increase the verbosity of the output (from 'warning' to 'info' level)",
                      default=False)
    parser.add_option("-q", "--quiet",
                      dest="quiet",
                      action='store_true',
                      help="do not print log messages to screen.",
                      default=False)
    parser.add_option("-o", "--option",
                      dest="extraOptions",
                      type="string",
                      action="append",
                      default=[],
                      help="Save one extra option (either a flag, or a key=value pair) that can be then accessed from the job config file")

    (options,args) = parser.parse_args()

    loop = main(options, args, parser)
    if not options.interactive:
        exit() # trigger exit also from ipython
