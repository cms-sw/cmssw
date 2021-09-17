# Implementation to draw APE trends
# to check convergence

import ROOT
from trendPlotter import *
from granularity import *
import os

try:
    base = os.environ['CMSSW_BASE']
except KeyError:
    base = "../../../../.."

plot = TrendPlotter()
plot.setOutputPath(base+"/src/Alignment/APEEstimation/trends/") 
plot.setTitle("Title")
plot.setGranularity(standardGranularity)

# The x-axis range is chosen depending on which year's APE are drawn.
# One can draw APE trends over multiple years

# List of tuples for one trend
trendList = []
# The last run is for example the first run of the next IOV minus 1
#~ trendsList.append( (firstRun, lastRun, inputFile) )

# label, trendList, color (optional, automatic), marker (optional, 0) 
# other options: dashed(optional, false)
plot.addTrend("label", trendList, color = ROOT.kBlack, marker=0)

# if this is set to false, the plot is made with runs on the x-axis
plot.doLumi = True

# if this is set to true, a logarithmic y-axis is used
plot.log = False

plot.draw()
