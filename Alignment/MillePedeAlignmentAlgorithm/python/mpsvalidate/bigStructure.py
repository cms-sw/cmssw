##########################################################################
# Creates a histogram where the the names of the structures are present
# as humanreadable text.
##

from builtins import range
import logging

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch()

import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.style as mpsv_style
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.classes as mpsv_classes


def plot(MillePedeUser, alignables, config):
    logger = logging.getLogger("mpsvalidate")
    
    # more space for labels
    ROOT.gStyle.SetPadBottomMargin(0.25)
    ROOT.gStyle.SetOptStat("emrs")

    for mode in ["xyz", "rot"]:
        big = mpsv_classes.PlotData(mode)

        # count number of needed bins and max shift
        for line in MillePedeUser:
            if (line.ObjId != 1):
                for i in range(3):
                    if (abs(line.Par[big.data[i]]) != 999999):
                        if (mode == "xyz"):
                            line.Par[big.data[i]] *= 10000
                        big.numberOfBins[i] += 1
                        if (abs(line.Par[big.data[i]]) > abs(big.maxShift[i])):
                            big.maxShift[i] = line.Par[big.data[i]]

        # initialize histograms
        for i in range(3):
            big.histo.append(ROOT.TH1F("Big Structure {0} {1}".format(big.xyz[i], mode), "", big.numberOfBins[i], 0, big.numberOfBins[i]))
            if (big.unit!=""):
                big.histo[i].SetYTitle("#Delta"+big.xyz[i]+" ["+big.unit+"]")
            else:
                big.histo[i].SetYTitle("#Delta"+big.xyz[i])
            big.histo[i].SetStats(0)
            big.histo[i].SetMarkerStyle(21)
            big.histoAxis.append(big.histo[i].GetXaxis())
            # bigger labels for the text
            big.histoAxis[i].SetLabelSize(0.06)
            big.histo[i].GetYaxis().SetTitleOffset(1.6)

        # add labels
        big.title = ROOT.TPaveLabel(
            0.1, 0.8, 0.9, 0.9, "High Level Structures {0}".format(mode))
        big.text = ROOT.TPaveText(0.05, 0.1, 0.95, 0.75)
        big.text.SetTextAlign(12)

        # error if shift is bigger than limit
        limit = config.limit[mode]
        for i in range(3):
            if (big.unit!=""):
                big.text.AddText("max. shift {0}: {1:.2} {2}".format(big.xyz[i], float(big.maxShift[i]), big.unit))
                if (abs(big.maxShift[i]) > limit):
                    big.text.AddText("! {0} shift bigger than {1} {2}".format(big.xyz[i], limit, big.unit))
            else:
                big.text.AddText("max. shift {0}: {1:.2}".format(big.xyz[i], float(big.maxShift[i])))
                if (abs(big.maxShift[i]) > limit):
                    big.text.AddText("! {0} shift bigger than {1}".format(big.xyz[i], limit))

        # fill histograms with value and name
        for line in MillePedeUser:
            if (line.ObjId != 1):
                for i in range(3):
                    if (abs(line.Par[big.data[i]]) != 999999):
                        # set name of the structure
                        big.histoAxis[i].SetBinLabel(
                            big.binPosition[i],
                            str(line.Name) if len(line.Name) <= 13 else str(line.Name)[:12]+".")
                        # fill with data, big.data[i] xyz or rot data
                        # transform xyz data from cm to #mu m
                        if (mode == "xyz"):
                            big.histo[i].SetBinContent(
                                big.binPosition[i], 10000 * line.Par[big.data[i]])
                        else:
                            big.histo[i].SetBinContent(
                                big.binPosition[i], line.Par[big.data[i]])
                        big.binPosition[i] += 1

        # rotate labels
        for i in range(3):
            big.histoAxis[i].LabelsOption("v")

        # reset y range
        # two types of ranges

        # 1. show all
        if (config.rangemodeHL == "all"):
            for i in range(3):
                big.usedRange[i] = big.maxShift[i]

        # 2. use given values
        if (config.rangemodeHL == "given"):
            # loop over coordinates
            for i in range(3):
                if (mode == "xyz"):
                    valuelist = config.rangexyzHL
                if (mode == "rot"):
                    valuelist = config.rangerotHL
                # loop over given values
                # without last value
                for value in valuelist:
                    # maximum smaller than given value
                    if (abs(big.maxShift[i]) < value):
                        big.usedRange[i] = value
                        break
                    # if not possible, force highest
                if (abs(big.maxShift[i]) > valuelist[-1]):
                    big.usedRange[i] = valuelist[-1]

        # all the same range
        if (config.samerangeHL == 1):
            # apply new range
            for i in range(3):
                big.usedRange[i] = max(map(abs, big.usedRange))

        # count outlieres
        if (config.rangemodeHL == "given"):
            for i in range(3):
                for binNumber in range(1, big.numberOfBins[i] + 1):
                    if (abs(big.histo[i].GetBinContent(binNumber)) > big.usedRange[i]):
                        big.hiddenEntries[i] += 1

            # add number of outlieres to text
            for i in range(3):
                if (big.hiddenEntries[i] != 0):
                    big.text.AddText("! {0}: {1} outlier !".format(
                        big.xyz[i], int(big.hiddenEntries[i])))

        # create canvas
        cBig = ROOT.TCanvas("canvasBigStrucutres_{0}".format(
            mode), "Parameter", 300, 0, 800, 600)
        cBig.Divide(2, 2)

        # draw histograms
        cBig.cd(1)
        big.title.Draw()
        big.text.Draw()

        # draw identification
        ident = mpsv_style.identification(config)
        ident.Draw()

        # TGraph copy to hide outlier
        copy = 3 * [None]

        # loop over coordinates
        for i in range(3):
            cBig.cd(i + 2)
            # option "AXIS" to only draw the axis
            big.histo[i].SetLineColor(0)
            big.histo[i].Draw("AXIS")
            # set new range
            big.histo[i].GetYaxis().SetRangeUser(-1.1 *
                                                 abs(big.usedRange[i]), 1.1 * abs(big.usedRange[i]))

            # TGraph object to hide outlier
            copy[i] = ROOT.TGraph(big.histo[i])
            # set the new range
            copy[i].SetMaximum(1.1 * abs(big.usedRange[i]))
            copy[i].SetMinimum(-1.1 * abs(big.usedRange[i]))
            # draw the data
            copy[i].Draw("PSAME")

        cBig.Update()

        # save as pdf
        cBig.Print(
            "{0}/plots/pdf/structures_{1}.pdf".format(config.outputPath, mode))

        # export as png
        image = ROOT.TImage.Create()
        image.FromPad(cBig)
        image.WriteImage(
            "{0}/plots/png/structures_{1}.png".format(config.outputPath, mode))

        # add to output list
        output = mpsv_classes.OutputData(plottype="big", parameter=mode,
                                         filename="structures_{0}".format(mode))
        config.outputList.append(output)

    # reset BottomMargin
    ROOT.gStyle.SetPadBottomMargin(0.1)
