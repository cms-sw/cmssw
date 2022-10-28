# Implementation to draw all iterations of an APE measurement
# to check convergence
from resultPlotter import *
from systematicErrors import *
from granularity import *
import os
try:
    base = os.environ['CMSSW_BASE']+"/src/Alignment/APEEstimation"
except KeyError:
    base = ""

plot = ResultPlotter()
plot.setOutputPath(base+"/hists/workingArea/") 
# label(also used as name when adding systematic errors), inputFile, color (optional, automatic by default), 
# marker (optional, 20 by default, 0 is line), hitNumbers (optional, file number of hits in each sector, allData.root)
plot.addInputFile("label", base+"/hists/workingArea/iter14/allData_iterationApe.root",  color = ROOT.kGray+2)
plot.setGranularity(standardGranularity)
plot.draw()
