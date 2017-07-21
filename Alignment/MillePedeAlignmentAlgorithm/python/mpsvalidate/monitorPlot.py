##########################################################################
# Draw the plots saved in the millePedeMonitor_merge.root file
#

import logging
import os
import cPickle

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch()

import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.style as mpsv_style
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.classes as mpsv_classes


def plot(config):
    logger = logging.getLogger("mpsvalidate")
    
    # adjust the plot style
    # show the skewness in the legend
    ROOT.gStyle.SetOptStat("emrs")
    ROOT.gStyle.SetPadLeftMargin(0.07)

    # retrieve the weights of the different datasets
    with open(os.path.join(config.jobDataPath, ".weights.pkl"), "rb") as f:
        weight_conf = cPickle.load(f)

    # loop over all millepedemonitor_X.root files
    for filename in os.listdir("{0}".format(config.jobDataPath)):
        if (filename.endswith(".root") and filename.startswith("millepedemonitor_")):
            # get X out of millepedemonitor_X.root files
            inputname = os.path.splitext(filename.split("millepedemonitor_")[-1])[0]

            # open file
            rootfile = ROOT.TFile(os.path.join(config.jobDataPath, filename))

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
                    weight = [item[1]
                              for item in weight_conf
                              if item[0] == inputname][0]
                    mpsv_classes.MonitorData(inputname.replace("_", " "), ntracks, weight)

                # create canvas
                canvas = ROOT.TCanvas("canvas{0}_{1}".format(
                    inputname, plotName), "Monitor", 300, 0, 800, 600)
                canvas.cd()

                # set statistics size
                mpsv_style.setstatsize(canvas, plot, config)

                # draw
                plot.Draw()

                # save as pdf
                canvas.Print(
                    "{0}/plots/pdf/monitor_{1}_{2}.pdf".format(config.outputPath, inputname.replace(".","_"), plotName))

                # export as png
                image = ROOT.TImage.Create()
                image.FromPad(canvas)
                image.WriteImage(
                    "{0}/plots/png/monitor_{1}_{2}.png".format(config.outputPath, inputname.replace(".","_"), plotName))

                # add to output list
                output = mpsv_classes.OutputData(plottype="monitor", name=inputname.replace("_", " "), number=plotName, filename="monitor_{1}_{2}".format(
                    config.outputPath, inputname.replace(".","_"), plotName))
                config.outputList.append(output)

    # reset the plot style
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPadLeftMargin(0.17)
