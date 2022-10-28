# Implementation to draw results of an APE measurement
# to check convergence
from validationPlotter import *
from granularity import *
import ROOT
import os
try:
    base = os.environ['CMSSW_BASE']+"/src/Alignment/APEEstimation"
except KeyError:
    base = ""

plot = ValidationPlotter()
plot.setOutputPath(base+"/hists/workingArea/validation") 
# label(also used as name when adding systematic errors), inputFile, color (optional, automatic by default), 
# marker (optional, 20 by default, 0 is line)
# Multiple inputs possible, in which case the plots will be normalized
# Remember that the folder has to be either iter0 or iter15 or baseline
plot.addInputFile("mp3401", "{base}/hists/workingArea/iter15/allData.root".format(base=base),color=ROOT.kBlack)
plot.addInputFile("Design", "{base}/hists/Design/baseline/allData.root".format(base=base),color=ROOT.kRed,marker=0)
plot.setGranularity(standardGranularity)
plot.draw()
