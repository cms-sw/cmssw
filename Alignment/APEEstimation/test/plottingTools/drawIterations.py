# Implementation to draw all iterations of an APE measurement
# to check convergence
from iterationsPlotter import *
from granularity import *
import os

try:
    base = os.environ['CMSSW_BASE']+"/src/Alignment/APEEstimation"
except KeyError:
    base = ""

plot = IterationsPlotter()
plot.setOutputPath(base+"/hists/iterations/")
plot.setInputFile(base+"/src/Alignment/APEEstimation/hists/workingArea/iter14/allData_iterationApe.root")
plot.setTitle("Title")
plot.setGranularity(standardGranularity)
plot.draw()
