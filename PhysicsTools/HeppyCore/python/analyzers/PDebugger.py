import logging
import sys
from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
import PhysicsTools.HeppyCore.utils.pdebug as pdebug


class PDebugger(Analyzer):
    '''Analyzer which turns on the physics debug output which
          (1) sets up the pdebugging tool, a separate logger for physics
          (2) logs for each event the number of event

    The pdebugging module should be used wherever the user wants a physics debug output.
    The physics debug output documents creation of clusters, tracks
    particles and aspects of simulation and reconstruction and can
    be used to trace errors or to follow the simulation and reconstruction code
    It was built in order to allow verification of C++ code vs python code

    This analyszer can be used to decide whether pysics output goes to either/both
        log file,
        stdout

    Example:
    from PhysicsTools.HeppyCore.analyzers.PDebugger import PDebugger
    pdebug = cfg.Analyzer(
    PDebugger,
    output_to_stdout = False, #optional
    debug_filename = os.getcwd()+'/python_physics_debug.log' #optional argument
    )
    '''
    def __init__(self, *args, **kwargs):
        super(PDebugger, self).__init__(*args, **kwargs)

        #no output will occur unless one or both of the following is requested.

        #turn on output to stdout if requested
        #note that both the main log leve and the stdout log level must be set in order to
        # obtain output at the info level
        if hasattr(self.cfg_ana, 'output_to_stdout') and self.cfg_ana.output_to_stdout:
            pdebug.set_stream(sys.stdout,level=logging.INFO)
            pdebug.pdebugger.setLevel(logging.INFO)

        #turn on output to file if requested
        if hasattr(self.cfg_ana, 'debug_filename'):
            pdebug.set_file(self.cfg_ana.debug_filename)
            pdebug.pdebugger.setLevel(logging.INFO)

    def process(self, event):
        pdebug.pdebugger.info(str('Event: {}'.format(event.iEv)))
