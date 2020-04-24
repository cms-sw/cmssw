##########################################################################
##
# Set the style of the output
##

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch()


######################################################################
# creates the identification text in the top left corner
#


def identification(config):
    text = ROOT.TPaveText(0.0, 0.95, 1.0, 1.0, "blNDC")
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
    ROOT.gStyle.SetStatW(0.3)
    ROOT.gStyle.SetStatH(0.3)
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
    ROOT.gStyle.SetErrorX(0)

    # For the canvas
    ROOT.gStyle.SetCanvasBorderMode(0)
    ROOT.gStyle.SetCanvasColor(0)
    ROOT.gStyle.SetCanvasDefH(800)  # Height of canvas
    ROOT.gStyle.SetCanvasDefW(800)  # Width of canvas
    ROOT.gStyle.SetCanvasDefX(0)  # Position on screen
    ROOT.gStyle.SetCanvasDefY(0)

    # For the frame
    ROOT.gStyle.SetFrameBorderMode(0)
    ROOT.gStyle.SetFrameBorderSize(1)
    ROOT.gStyle.SetFrameFillColor(1)
    ROOT.gStyle.SetFrameFillStyle(0)
    ROOT.gStyle.SetFrameLineColor(1)
    ROOT.gStyle.SetFrameLineStyle(0)
    ROOT.gStyle.SetFrameLineWidth(1)

    # For the Pad
    ROOT.gStyle.SetPadBorderMode(0)
    ROOT.gStyle.SetPadColor(0)
    ROOT.gStyle.SetPadGridX(False)
    ROOT.gStyle.SetPadGridY(False)
    ROOT.gStyle.SetGridColor(0)
    ROOT.gStyle.SetGridStyle(3)
    ROOT.gStyle.SetGridWidth(1)

    # Margins
    ROOT.gStyle.SetPadTopMargin(0.08)
    ROOT.gStyle.SetPadBottomMargin(0.19)
    ROOT.gStyle.SetPadLeftMargin(0.17)
    #ROOT.gStyle.SetPadRightMargin(0.07)

    # For the histo:
    ROOT.gStyle.SetHistLineColor(1)
    ROOT.gStyle.SetHistLineStyle(0)
    ROOT.gStyle.SetHistLineWidth(2)
    ROOT.gStyle.SetMarkerSize(1.4)
    ROOT.gStyle.SetEndErrorSize(4)

    # For the statistics box:
    ROOT.gStyle.SetOptStat(0)

    # For the axis
    ROOT.gStyle.SetAxisColor(1, "XYZ")
    ROOT.gStyle.SetTickLength(0.03, "XYZ")
    ROOT.gStyle.SetNdivisions(510, "XYZ")
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetStripDecimals(False)

    # For the axis labels and titles
    ROOT.gStyle.SetTitleColor(1, "XYZ")
    ROOT.gStyle.SetLabelColor(1, "XYZ")
    ROOT.gStyle.SetLabelFont(42, "XYZ")
    ROOT.gStyle.SetLabelOffset(0.007, "XYZ")
    ROOT.gStyle.SetLabelSize(0.045, "XYZ")
    ROOT.gStyle.SetTitleFont(42, "XYZ")
    ROOT.gStyle.SetTitleSize(0.06, "XYZ")

    # For the legend
    ROOT.gStyle.SetLegendBorderSize(0)
