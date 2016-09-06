#!/usr/bin/env python

##########################################################################
##
# Set the style of the output
##

from ROOT import TPaveText, gStyle


######################################################################
# creates the identification text in the top left corner
#


def identification(config):
    text = TPaveText(0.0, 0.95, 1.0, 1.0, "blNDC")
    text.AddText(config.message)
    text.SetBorderSize(0)
    text.SetTextAlign(12)
    text.SetTextSizePixels(10)
    text.SetTextFont(82)
    text.SetFillColor(0)
    return text

######################################################################
# statistics size
#


def setstatsize(canvas, plot, config):
    # statistics size
    gStyle.SetStatW(0.3)
    gStyle.SetStatH(0.3)
    plot.Draw()
    canvas.Update()

    # set the size of the statistics box
    stat = plot.FindObject("stats")
    stat.SetX1NDC(1 - config.statboxsize)
    stat.SetY1NDC(1 - config.statboxsize)


######################################################################
# set gstyle
# by https://github.com/mschrode/AwesomePlots/blob/master/Style.cc
#


def setgstyle():
    # Zero horizontal error bars
    gStyle.SetErrorX(0)

    # For the canvas
    gStyle.SetCanvasBorderMode(0)
    gStyle.SetCanvasColor(0)
    gStyle.SetCanvasDefH(800)  # Height of canvas
    gStyle.SetCanvasDefW(800)  # Width of canvas
    gStyle.SetCanvasDefX(0)  # Position on screen
    gStyle.SetCanvasDefY(0)

    # For the frame
    gStyle.SetFrameBorderMode(0)
    gStyle.SetFrameBorderSize(1)
    gStyle.SetFrameFillColor(1)
    gStyle.SetFrameFillStyle(0)
    gStyle.SetFrameLineColor(1)
    gStyle.SetFrameLineStyle(0)
    gStyle.SetFrameLineWidth(1)

    # For the Pad
    gStyle.SetPadBorderMode(0)
    gStyle.SetPadColor(0)
    gStyle.SetPadGridX(False)
    gStyle.SetPadGridY(False)
    gStyle.SetGridColor(0)
    gStyle.SetGridStyle(3)
    gStyle.SetGridWidth(1)

    # Margins
    gStyle.SetPadTopMargin(0.08)
    gStyle.SetPadBottomMargin(0.19)
    gStyle.SetPadLeftMargin(0.17)
    #gStyle.SetPadRightMargin(0.07)

    # For the histo:
    gStyle.SetHistLineColor(1)
    gStyle.SetHistLineStyle(0)
    gStyle.SetHistLineWidth(2)
    gStyle.SetMarkerSize(1.4)
    gStyle.SetEndErrorSize(4)

    # For the statistics box:
    gStyle.SetOptStat(0)

    # For the axis
    gStyle.SetAxisColor(1, "XYZ")
    gStyle.SetTickLength(0.03, "XYZ")
    gStyle.SetNdivisions(510, "XYZ")
    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
    gStyle.SetStripDecimals(False)

    # For the axis labels and titles
    gStyle.SetTitleColor(1, "XYZ")
    gStyle.SetLabelColor(1, "XYZ")
    gStyle.SetLabelFont(42, "XYZ")
    gStyle.SetLabelOffset(0.007, "XYZ")
    gStyle.SetLabelSize(0.045, "XYZ")
    gStyle.SetTitleFont(42, "XYZ")
    gStyle.SetTitleSize(0.06, "XYZ")

    # For the legend
    gStyle.SetLegendBorderSize(0)
