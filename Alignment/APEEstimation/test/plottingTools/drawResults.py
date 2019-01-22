# Implementation to draw all iterations of an APE measurement
# to check convergence

import ROOT
from resultPlotter import *
from systematics import *
from granularity import *

try:
    base = os.environ['CMSSW_BASE']+"/src/Alignment/APEEstimation"
except KeyError:
    base = ""

plot = ResultPlotter()
plot.setOutputPath(base+"/workingArea/") 
# internal name (used for example when adding systematic errors), path to file, label, color (optional)
plot.addInputFile("placeholder1", base+"/workingArea/iter14/allData_iterationApe.root", "measurement A")
plot.addInputFile("placeholder2", base+"/workingArea2/iter14/allData_iterationApe.root", "measurement B", ROOT.kRed)
plot.setTitle("")
plot.setGranularity(standardGranularity)
plot.draw()
