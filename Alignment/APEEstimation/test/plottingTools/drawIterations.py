# Implementation to draw all iterations of an APE measurement
# to check convergence

from iterationsPlotter import *
from granularity import *

try:
    base = os.environ['CMSSW_BASE']+"/src/Alignment/APEEstimation"
except KeyError:
    base = ""

plot = IterationsPlotter()
plot.setOutputPath(base+"/workingArea/")
plot.setInputFile(base+"/workingArea/iter14/allData_iterationApe.root")
plot.setTitle("")
plot.setGranularity(standardGranularity)
plot.draw()
