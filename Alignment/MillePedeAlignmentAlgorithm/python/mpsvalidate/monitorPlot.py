#!/usr/bin/env python

##########################################################################
# Draw the plots saved in the millePedeMonitor_merge.root file
#

import logging
import os

from ROOT import TH1F, TCanvas, TFile, TImage, gStyle

from Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.classes import MonitorData, OutputData
from Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.style import setstatsize


def plot(config):
    logger = logging.getLogger("mpsvalidate")
    
    # adjust the plot style
    # show the skewness in the legend
    gStyle.SetOptStat("emrs")
    gStyle.SetPadLeftMargin(0.07)

    # loop over all millepedemonitor_X.root files
    for filename in os.listdir("{0}".format(config.jobDataPath)):
        if (filename.endswith(".root") and filename.startswith("millepedemonitor_")):
            # get X out of millepedemonitor_X.root files
            inputname = filename[17:-5]

            # open file
            rootfile = TFile("{0}/{1}".format(config.jobDataPath, filename))

            plotPaths = ["usedTrackHists/usedptTrack", "usedTrackHists/usedetaTrack",
                         "usedTrackHists/usedphiTrack", "usedTrackHists/usednHitTrack"]

            # loop over plots which should be plotted
            for plotNumber, plotPath in enumerate(plotPaths):
                # get plotname
                plotName = plotPath.split("/")[1]
                # get plot
                plot = rootfile.Get(plotPath)

                if (plotNumber == 0):
                    # get number of used tracks
                    ntracks = int(plot.GetEntries())
                    MonitorData(inputname.replace("_", " "), ntracks)

                # create canvas
                canvas = TCanvas("canvas{0}_{1}".format(
                    inputname, plotName), "Monitor", 300, 0, 800, 600)
                canvas.cd()

                # set statistics size
                setstatsize(canvas, plot, config)

                # draw
                plot.Draw()

                # save as pdf
                canvas.Print(
                    "{0}/plots/pdf/monitor_{1}_{2}.pdf".format(config.outputPath, inputname.replace(".","_"), plotName))

                # export as png
                image = TImage.Create()
                image.FromPad(canvas)
                image.WriteImage(
                    "{0}/plots/png/monitor_{1}_{2}.png".format(config.outputPath, inputname.replace(".","_"), plotName))

                # add to output list
                output = OutputData(plottype="monitor", name=inputname.replace("_", " "), number=plotName, filename="monitor_{1}_{2}".format(
                    config.outputPath, inputname.replace(".","_"), plotName))
                config.outputList.append(output)

    # reset the plot style
    gStyle.SetOptStat(0)
    gStyle.SetPadLeftMargin(0.17)
