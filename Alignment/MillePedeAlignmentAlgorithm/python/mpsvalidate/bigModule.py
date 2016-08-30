#!/usr/bin/env python

##########################################################################
# Creates histograms of the modules of one structure. and returns them as
# a list of PlotData objects.
##

import logging

from ROOT import (TH1F, TCanvas, TImage, TPaveLabel, TPaveText, TTree, gROOT,
                  gStyle)

from Alignment.MillePedeAlignmentAlgorithm.mpsvalidate import subModule
from Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.classes import PedeDumpData, OutputData, PlotData
from Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.geometry import Alignables, Structure
from Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.style import identification, setstatsize


def plot(MillePedeUser, alignables, config):
    logger = logging.getLogger("mpsvalidate")

    alignables.create_list(MillePedeUser)

    # number of bins to start
    numberOfBins = 10000

    ######################################################################
    # initialize data hierarchy
    # plots[mode][struct]
    #

    plots = []
    # loop over mode
    for modeNumber, mode in enumerate(["xyz", "rot", "dist"]):
        plots.append([])
        # loop over structures
        for structNumber, struct in enumerate(alignables.structures):
            plots[modeNumber].append(PlotData(mode))

    # initialize histograms
    for modeNumber, mode in enumerate(["xyz", "rot", "dist"]):
        for structNumber, struct in enumerate(alignables.structures):
            # use a copy for shorter name
            plot = plots[modeNumber][structNumber]

            for i in range(3):
                if (mode == "xyz"):
                    plot.histo.append(TH1F("{0} {1} {2}".format(struct.get_name(), plot.xyz[
                                      i], mode), "Parameter {0}".format(plot.xyz[i]), numberOfBins, -1000, 1000))
                else:
                    plot.histo.append(TH1F("{0} {1} {2}".format(struct.get_name(), plot.xyz[
                                      i], mode), "Parameter {0}".format(plot.xyz[i]), numberOfBins, -0.1, 0.1))

                plot.histo[i].SetXTitle(plot.unit)
                plot.histo[i].GetXaxis().SetTitleOffset(0.85)
                plot.histoAxis.append(plot.histo[i].GetXaxis())

            # add labels
            plot.title = TPaveLabel(
                0.1, 0.8, 0.9, 0.9, "Module: {0} {1}".format(struct.get_name(), mode))
            plot.text = TPaveText(0.05, 0.1, 0.95, 0.75)
            plot.text.SetTextAlign(12)
            plot.text.SetTextSizePixels(20)

            # save copy
            plots[modeNumber][structNumber] = plot

    ######################################################################
    # fill histogram
    #

    for line in MillePedeUser:
        # is module ?
        if (line.ObjId == 1):
            for modeNumber, mode in enumerate(["xyz", "rot", "dist"]):
                for structNumber, struct in enumerate(alignables.structures):
                    # use a copy for shorter name
                    plot = plots[modeNumber][structNumber]

                    # module in struct ?
                    if (struct.contains_detid(line.Id)):
                        for i in range(3):
                            if (abs(line.Par[plot.data[i]]) != 999999):
                                # transform xyz data from cm to #mu m
                                if (mode == "xyz"):
                                    plot.histo[i].Fill(
                                        10000 * line.Par[plot.data[i]])
                                else:
                                    plot.histo[i].Fill(line.Par[plot.data[i]])

                    # save copy
                    plots[modeNumber][structNumber] = plot

    ######################################################################
    # find the best range
    #

    for modeNumber, mode in enumerate(["xyz", "rot", "dist"]):
        for structNumber, struct in enumerate(alignables.structures):
            # use a copy for shorter name
            plot = plots[modeNumber][structNumber]

            for i in range(3):
                # get first and last bin with content and chose the one which
                # has a greater distance to the center
                if (abs(numberOfBins / 2 - plot.histo[i].FindFirstBinAbove()) > abs(plot.histo[i].FindLastBinAbove() - numberOfBins / 2)):
                    plot.maxBinShift[i] = abs(
                        numberOfBins / 2 - plot.histo[i].FindFirstBinAbove())
                    # set the maxShift value
                    plot.maxShift[i] = plot.histo[i].GetBinCenter(
                        plot.histo[i].FindFirstBinAbove())
                else:
                    plot.maxBinShift[i] = abs(
                        plot.histo[i].FindLastBinAbove() - numberOfBins / 2)
                    # set the maxShift value
                    plot.maxShift[i] = plot.histo[i].GetBinCenter(
                        plot.histo[i].FindLastBinAbove())
                # skip empty histogram
                if (abs(plot.maxBinShift[i]) == numberOfBins / 2 + 1):
                    plot.maxBinShift[i] = 0

            # three types of ranges

            # 1. multiple of standard dev
            if (config.rangemode == "stddev"):
                for i in range(3):
                    if (plot.histo[i].GetEntries() != 0 and plot.histo[i].GetStdDev() != 0):
                        # if the plotrange is much bigger than the standard
                        # deviation use config.widthstdev * StdDev als Range
                        if (max(plot.maxShift) / plot.histo[i].GetStdDev() > config.defpeak):
                            # corresponding bin config.widthstdev*StdDev
                            binShift = int(plot.histo[i].FindBin(
                                config.widthstddev * plot.histo[i].GetStdDev()) - numberOfBins / 2)
                        else:
                            binShift = max(plot.maxBinShift)

                        # save used binShift
                        plot.binShift[i] = binShift

            # 2. show all
            if (config.rangemode == "all"):
                for i in range(3):
                    plot.binShift[i] = plot.maxBinShift[i]

            # 3. use given ranges
            if (config.rangemode == "given"):
                for i in range(3):
                    if (mode == "xyz"):
                        valuelist = config.rangexyzM
                    if (mode == "rot"):
                        valuelist = config.rangerotM
                    if (mode == "dist"):
                        valuelist = config.rangedistM

                    for value in valuelist:
                        # maximum smaller than given value
                        if (abs(plot.maxShift[i]) < value):
                            binShift = value
                            break
                    # if not possible, force highest
                    if (abs(plot.maxShift[i]) > valuelist[-1]):
                        binShift = valuelist[-1]
                    # calculate binShift
                    plot.binShift[i] = int(
                        binShift / plot.histo[i].GetBinWidth(1))

            # all plot the same range
            if (config.samerange == 1):
                for i in range(3):
                    plot.binShift[i] = max(plot.binShift)

            # save used range
            for i in range(3):
                plot.usedRange[i] = plot.binShift[i]

            # count entries which are not shown anymore
            for i in range(3):
                # bin 1 to begin of histogram
                for j in range(1, numberOfBins / 2 - plot.binShift[i]):
                    plot.hiddenEntries[i] += plot.histo[i].GetBinContent(j)
                # from the end of shown bins to the end of histogram
                for j in range(numberOfBins / 2 + plot.binShift[i], plot.histo[i].GetNbinsX()):
                    plot.hiddenEntries[i] += plot.histo[i].GetBinContent(j)

            # apply new range
            for i in range(3):
                if (plot.histo[i].GetEntries() != 0):
                    # merge bins, ca. 100 should be visible in the resulting
                    # plot
                    mergeNumberBins = plot.binShift[i]
                    # skip empty histogram
                    if (mergeNumberBins != 0):
                        # the 2*maxBinShift bins should shrink to 100 bins
                        mergeNumberBins = int(
                            2. * mergeNumberBins / config.numberofbins)
                        # the total number of bins should be dividable by the
                        # bins shrinked together
                        if (mergeNumberBins == 0):
                            mergeNumberBins = 1
                        while (numberOfBins % mergeNumberBins != 0 and mergeNumberBins != 1):
                            mergeNumberBins -= 1

                        # Rebin and save new created histogram and axis
                        plot.histo[i] = plot.histo[i].Rebin(mergeNumberBins)
                        plot.histoAxis[i] = plot.histo[i].GetXaxis()

                        # set view range. it is important to note that the number of bins have changed with the rebinning
                        # the total number and the number of shift must be
                        # corrected with / mergeNumberBins
                        plot.histoAxis[i].SetRange(int(numberOfBins / (2 * mergeNumberBins) - plot.binShift[
                                                   i] / mergeNumberBins), int(numberOfBins / (2 * mergeNumberBins) + plot.binShift[i] / mergeNumberBins))

            # error if shift is bigger than limit
            limit = config.limit[mode]
            for i in range(3):
                # skip empty
                if (plot.histo[i].GetEntries() > 0):
                    plot.text.AddText("max. shift {0}: {1:.2}".format(
                        plot.xyz[i], plot.maxShift[i]))
                    if (abs(plot.maxShift[i]) > limit):
                        plot.text.AddText(
                            "! {0} shift bigger than {1} !".format(plot.xyz[i], limit))
                    if (plot.hiddenEntries[i] != 0):
                        plot.text.AddText("! {0} {1} outlier !".format(
                            plot.xyz[i], int(plot.hiddenEntries[i])))

            # save copy
            plots[modeNumber][structNumber] = plot

    ######################################################################
    # make the plots
    #

    # show the skewness in the legend
    gStyle.SetOptStat("emrs")

    for modeNumber, mode in enumerate(["xyz", "rot", "dist"]):
        for structNumber, struct in enumerate(alignables.structures):
            # use a copy for shorter name
            plot = plots[modeNumber][structNumber]

            canvas = TCanvas("canvasModules{0}_{1}".format(
                struct.get_name(), mode), "Parameter", 300, 0, 800, 600)
            canvas.Divide(2, 2)

            canvas.cd(1)
            plot.title.Draw()
            plot.text.Draw()

            # draw identification
            ident = identification(config)
            ident.Draw()

            # is there any plot?
            plotNumber = 0

            # loop over coordinates
            for i in range(3):
                if(plot.histo[i].GetEntries() > 0):
                    plotNumber += 1
                    canvas.cd(i + 2)
                    setstatsize(canvas, plot.histo[i], config)
                    plot.histo[i].DrawCopy()

            if (plotNumber == 0):
                break

            canvas.Update()

            # save as pdf
            canvas.Print(
                "{0}/plots/pdf/modules_{1}_{2}.pdf".format(config.outputPath, mode, struct.get_name()))

            # export as png
            image = TImage.Create()
            image.FromPad(canvas)
            image.WriteImage(
                "{0}/plots/png/modules_{1}_{2}.png".format(config.outputPath, mode, struct.get_name()))

            # add to output list
            output = OutputData(plottype="mod", name=struct.get_name(),
                                parameter=mode, filename="modules_{0}_{1}".format(mode, struct.get_name()))
            config.outputList.append(output)

    ######################################################################
    # make plots with substructure
    #

    if (config.showsubmodule == 1):
        alignables.create_children_list()
        for modeNumber, mode in enumerate(["xyz", "rot", "dist"]):
            for structNumber, struct in enumerate(alignables.structures):
                # use a copy for shorter name
                plot = plots[modeNumber][structNumber]

                subModule.plot(MillePedeUser, alignables,
                               mode, struct, plot, config)
