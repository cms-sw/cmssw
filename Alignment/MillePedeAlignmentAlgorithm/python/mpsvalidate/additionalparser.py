##########################################################################
# Parse the alignment_merge.py file for additional information
#

import logging
import Alignment.MillePedeAlignmentAlgorithm.mpslib.tools as mps_tools


class AdditionalData:
    """ stores the additional information of the alignment_merge.py file
    """

    def __init__(self):
        self.pede_steerer_method  = ""
        self.pede_steerer_options = []
        self.pede_steerer_command = ""

        self.selectors = {}
        self.iov_definition = ""


    def parse(self, config, path):
        logger = logging.getLogger("mpsvalidate")

        # extract process object from aligment_merge.py file
        try:
            process = mps_tools.get_process_object(path)
        except ImportError:
            logger.error("AdditionalData: {0} does not exist".format(path))
            return

        # find alignable selectors
        param_builder = process.AlignmentProducer.ParameterBuilder
        for index,sel in enumerate(param_builder.parameterTypes):
            selector_name = sel.split(",")[0].strip()
            self.selectors[index] = {
                "name": selector_name,
                "selector": getattr(param_builder, selector_name),
                }

        # find IOV definition
        if len(process.AlignmentProducer.RunRangeSelection) > 0:
            self.iov_definition = \
                process.AlignmentProducer.RunRangeSelection.dumpPython()

        # find pede steerer configuration
        pede_steerer = process.AlignmentProducer.algoConfig.pedeSteerer
        self.pede_steerer_method  = pede_steerer.method.value()
        self.pede_steerer_options = pede_steerer.options.value()
        self.pede_steerer_command = pede_steerer.pedeCommand.value()
