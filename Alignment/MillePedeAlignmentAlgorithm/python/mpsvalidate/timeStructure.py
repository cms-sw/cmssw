##########################################################################
# Creates a histogram where the the names of the structures are present
# as humanreadable text. Multiple MillePedeUser TTrees are used to
# get a time dependent plot.
##

import logging

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch()

import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.style as mpsv_style
import Alignment.MillePedeAlignmentAlgorithm.mpsvalidate.classes as mpsv_classes


def plot(treeFile, alignables, config):
    logger = logging.getLogger("mpsvalidate")

    for mode in ["xyz", "rot"]:

        time = mpsv_classes.PlotData(mode)

        # list of all avaible TTrees
        listMillePedeUser = []
        MillePedeUser = []
        for i in range(config.firsttree, 101):
            if (treeFile.GetListOfKeys().Contains("MillePedeUser_{0}".format(i))):
                listMillePedeUser.append(i)

        # load MillePedeUser_X TTrees
        for i in listMillePedeUser:
            MillePedeUser.append(treeFile.Get("MillePedeUser_{0}".format(i)))

        ######################################################################
        # remove TTrees without results
        #

        # check if there is a TTree without any results
        # therefor search for the first alignable
        first = 0
        newlistMillePedeUser = []
        # find first alignable
        for line in MillePedeUser[0]:
            if (line.ObjId != 1 and any(abs(line.Par[time.data[i]]) != 999999 for i in [0, 1, 2])):
                first = line.Id
                newlistMillePedeUser.append(config.firsttree)
                break

        # check the following TTrees
        for ttreeNumber, ttree in enumerate(MillePedeUser[1:]):
            for line in ttree:
                if (line.Id == first):
                    if (any(abs(line.Par[time.data[i]]) != 999999 for i in [0, 1, 2])):
                        # note that the first tree was checked
                        newlistMillePedeUser.append(
                            ttreeNumber + config.firsttree + 1)
                    break

        listMillePedeUser = newlistMillePedeUser

        # reload MillePedeUser_X TTrees
        MillePedeUser = []
        for i in listMillePedeUser:
            MillePedeUser.append(treeFile.Get("MillePedeUser_{0}".format(i)))

        if not listMillePedeUser:
            logger.error("Timeplots: no TTrees found")
            return

        if not MillePedeUser:
            logger.error("Timeplots: no TTree could be opened")
            return

        ######################################################################
        # initialize data hierarchy
        #

        plots = []
        # objids which were found in the TTree
        objids = []
        obj_names = []

        # loop over first tree to initialize
        for line in MillePedeUser[0]:
            if (line.ObjId != 1 and any(abs(line.Par[time.data[i]]) != 999999 for i in [0, 1, 2])):
                plots.append(mpsv_classes.PlotData(mode))

                # new objid?
                if (line.ObjId not in objids):
                    objids.append(line.ObjId)
                    obj_names.append(str(line.Name))

                # initialize histograms
                for i in range(3):
                    plots[-1].histo.append(ROOT.TH1F("Time Structure {0} {1} {2} {3}".format(mode, str(line.Name),
                        len(plots), i), "", len(listMillePedeUser), 0, len(listMillePedeUser)))
                    plots[-1].label = line.Id
                    plots[-1].objid = line.ObjId
                                           
                    if (time.unit!=""):
                        plots[-1].histo[i].SetYTitle("#Delta"+time.xyz[i]+" ["+time.unit+"]")
                    else:
                        plots[-1].histo[i].SetYTitle("#Delta"+time.xyz[i])
                    plots[-1].histo[i].SetXTitle("IOV")
                    plots[-1].histo[i].SetStats(0)
                    plots[-1].histo[i].SetMarkerStyle(21)
                    # bigger labels for the text
                    plots[-1].histo[i].GetXaxis().SetLabelSize(0.08)
                    plots[-1].histo[i].GetYaxis().SetTitleOffset(1.6)

        ######################################################################
        # fill histogram
        #

        # loop over TTrees
        for treeNumber, tree in enumerate(MillePedeUser):
            for line in tree:
                if (line.ObjId != 1 and any(abs(line.Par[time.data[i]]) != 999999 for i in [0, 1, 2])):
                    # find the right plot
                    for plot in plots:
                        if (plot.label == line.Id):
                            for i in range(3):
                                # note that the first bin is referenced by 1
                                plot.histo[i].GetXaxis().SetBinLabel(
                                    treeNumber + 1, str(listMillePedeUser[treeNumber]))
                                # transform xyz data from cm to #mu m
                                if (mode == "xyz"):
                                    plot.histo[i].SetBinContent(
                                        treeNumber + 1, 10000 * line.Par[plot.data[i]])
                                else:
                                    plot.histo[i].SetBinContent(
                                        treeNumber + 1, line.Par[plot.data[i]])

        ######################################################################
        # find maximum/minimum
        #

        maximum = [[0, 0, 0] for x in range(len(objids))]
        minimum = [[0, 0, 0] for x in range(len(objids))]

        for index, objid in enumerate(objids):
            for plot in plots:
                if (plot.objid == objid):
                    for i in range(3):
                        # maximum
                        if (plot.histo[i].GetMaximum() > maximum[index][i]):
                            maximum[index][i] = plot.histo[i].GetMaximum()
                        # minimum
                        if (plot.histo[i].GetMinimum() < minimum[index][i]):
                            minimum[index][i] = plot.histo[i].GetMinimum()

        ######################################################################
        # make the plots
        #

        # loop over all objids
        for index, objid in enumerate(objids):

            canvas = ROOT.TCanvas("canvasTimeBigStrucutres_{0}_{1}".format(
                mode, obj_names[index]), "Parameter", 300, 0, 800, 600)
            canvas.Divide(2, 2)

            # add text
            title = ROOT.TPaveLabel(0.1, 0.8, 0.9, 0.9, "{0} over time {1}".format(
                obj_names[index], mode))

            legend = ROOT.TLegend(0.05, 0.1, 0.95, 0.75)

            # draw on canvas
            canvas.cd(1)
            title.Draw()

            # draw identification
            ident = mpsv_style.identification(config)
            ident.Draw()

            # TGraph copies to hide outlier
            copy = []

            # reset y range of first plot
            # two types of ranges
            for i in range(3):
                for plot in plots:
                    if (plot.objid == objid):
                        # 1. show all
                        if config.rangemodeHL == "all":
                            plot.usedRange[i] = max(
                                abs(maximum[index][i]), abs(minimum[index][i]))

                        # 2. use given values
                        if (config.rangemodeHL == "given"):
                            # loop over coordinates
                            if mode == "xyz":
                                valuelist = config.rangexyzHL
                            if mode == "rot":
                                valuelist = config.rangerotHL
                            # loop over given values
                            # without last value
                            for value in valuelist:
                                # maximum smaller than given value
                                if max(abs(maximum[index][i]), abs(minimum[index][i])) < value:
                                    plot.usedRange[i] = value
                                    break
                            # if not possible, force highest
                            if (max(abs(maximum[index][i]), abs(minimum[index][i])) > valuelist[-1]):
                                plot.usedRange[i] = valuelist[-1]

            # draw plots on canvas
            for i in range(3):
                canvas.cd(2 + i)

                number = 1

                for plot in plots:
                    if (plot.objid == objid):
                        # all the same range
                        if (config.samerangeHL == 1):
                            plot.usedRange[i] = max(map(abs, plot.usedRange))

                        # set new range
                        plot.histo[i].GetYaxis(
                        ).SetRangeUser(-1.2 * abs(plot.usedRange[i]), 1.2 * abs(plot.usedRange[i]))

                        plot.histo[i].SetLineColorAlpha(number + 2, 0.5)
                        plot.histo[i].SetMarkerColorAlpha(number + 2, 1)

                        # option "AXIS" to only draw the axis
                        plot.histo[i].SetLineColor(0)
                        plot.histo[i].Draw("PSAME")

                        # TGraph object to hide outlier
                        copy.append(ROOT.TGraph(plot.histo[i]))
                        # set the new range
                        copy[-1].SetMaximum(1.2 * abs(plot.usedRange[i]))
                        copy[-1].SetMinimum(-1.2 * abs(plot.usedRange[i]))
                        # draw the data
                        copy[-1].SetLineColorAlpha(number + 2, 0.5)
                        copy[-1].Draw("LPSAME")

                        if (i == 0):
                            legend.AddEntry(
                                plot.histo[i], "{0}".format(number))
                        number += 1

            canvas.cd(1)
            legend.Draw()

            canvas.Update()

            # save as pdf
            canvas.Print("{0}/plots/pdf/timeStructures_{1}_{2}.pdf".format(
                config.outputPath, mode, obj_names[index]))

            # export as png
            image = ROOT.TImage.Create()
            image.FromPad(canvas)
            image.WriteImage("{0}/plots/png/timeStructures_{1}_{2}.png".format(
                config.outputPath, mode, obj_names[index]))

            # add to output list
            output = mpsv_classes.OutputData(plottype="time", name=obj_names[index],
                                             parameter=mode, filename="timeStructures_{0}_{1}".format(mode, obj_names[index]))
            config.outputList.append(output)
