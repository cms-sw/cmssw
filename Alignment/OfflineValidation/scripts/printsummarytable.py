#!/usr/bin/env python

import os
import sys

if len(sys.argv) > 1:
    folder = sys.argv[1]
else:
    folder = "."

from Alignment.OfflineValidation.TkAlAllInOneTool.genericValidation import *
from Alignment.OfflineValidation.TkAlAllInOneTool.helperFunctions import recursivesubclasses
from Alignment.OfflineValidation.TkAlAllInOneTool.offlineValidation import *
from Alignment.OfflineValidation.TkAlAllInOneTool.primaryVertexValidation import *
from Alignment.OfflineValidation.TkAlAllInOneTool.trackSplittingValidation import *
from Alignment.OfflineValidation.TkAlAllInOneTool.zMuMuValidation import *

subclasses = recursivesubclasses(ValidationWithPlotsSummaryBase)
subclasses = [subcls for subcls in subclasses if not subcls.__abstractmethods__]
printedanything = False
tried = []
for subcls in subclasses:
    tried += ["{}Summary.txt".format(subcls.__name__), os.path.join(subcls.plotsdirname(), "{}Summary.txt".format(subcls.__name__))]
    if os.path.exists(os.path.join(folder, "{}Summary.txt".format(subcls.__name__))):
        printedanything = True
        subcls.printsummaryitems(folder=folder)
    elif os.path.exists(os.path.join(folder, subcls.plotsdirname(), "{}Summary.txt".format(subcls.__name__))):
        subcls.printsummaryitems(folder=os.path.join(folder, subcls.plotsdirname()))
        printedanything = True

if not printedanything:
    raise ValueError("Didn't find any *ValidationSummary.txt.  Maybe try somewhere else?\nPaths searched:\n" + "\n".join(tried))
