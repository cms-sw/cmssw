from ROOT import kBlack, TPaveText

def officialStyle(style):
    style.SetCanvasColor     (0)
    style.SetCanvasBorderSize(10)
    style.SetCanvasBorderMode(0)
    style.SetCanvasDefH      (700)
    style.SetCanvasDefW      (700)
    style.SetCanvasDefX      (100)
    style.SetCanvasDefY      (100)

    # color palette for 2D temperature plots
    # style.SetPalette(1,0)

    # Pads
    style.SetPadColor       (0)
    style.SetPadBorderSize  (10)
    style.SetPadBorderMode  (0)
    style.SetPadBottomMargin(0.13)
    style.SetPadTopMargin   (0.08)
    style.SetPadLeftMargin  (0.15)
    style.SetPadRightMargin (0.05)
    style.SetPadGridX       (0)
    style.SetPadGridY       (0)
    style.SetPadTickX       (1)
    style.SetPadTickY       (1)

    # Frames
    style.SetLineWidth(3)
    style.SetFrameFillStyle ( 0)
    style.SetFrameFillColor ( 0)
    style.SetFrameLineColor ( 1)
    style.SetFrameLineStyle ( 0)
    style.SetFrameLineWidth ( 2)
    style.SetFrameBorderSize(10)
    style.SetFrameBorderMode( 0)

    # Histograms
    style.SetHistFillColor(2)
    style.SetHistFillStyle(0)
    style.SetHistLineColor(1)
    style.SetHistLineStyle(0)
    style.SetHistLineWidth(3)
    style.SetNdivisions(505)

    # Functions
    style.SetFuncColor(1)
    style.SetFuncStyle(0)
    style.SetFuncWidth(2)

    # Various
    style.SetMarkerStyle(20)
    style.SetMarkerColor(kBlack)
    style.SetMarkerSize (1.4)

    style.SetTitleBorderSize(0)
    style.SetTitleFillColor (0)
    style.SetTitleX         (0.2)

    style.SetTitleSize  (0.055,"X")
    style.SetTitleOffset(1.200,"X")
    style.SetLabelOffset(0.005,"X")
    style.SetLabelSize  (0.050,"X")
    style.SetLabelFont  (42   ,"X")

    style.SetStripDecimals(False)
    style.SetLineStyleString(11,"20 10")

    style.SetTitleSize  (0.055,"Y")
    style.SetTitleOffset(1.600,"Y")
    style.SetLabelOffset(0.010,"Y")
    style.SetLabelSize  (0.050,"Y")
    style.SetLabelFont  (42   ,"Y")

    style.SetTextSize   (0.055)
    style.SetTextFont   (42)

    style.SetStatFont   (42)
    style.SetTitleFont  (42)
    style.SetTitleFont  (42,"X")
    style.SetTitleFont  (42,"Y")

    style.SetOptStat    (0)


def CMSPrelim(dataset, channel, lowX, lowY):
    cmsprel  =  TPaveText(lowX, lowY+0.06, lowX+0.30, lowY+0.16, "NDC")
    cmsprel.SetBorderSize(   0 )
    cmsprel.SetFillStyle(    0 )
    cmsprel.SetTextAlign(   12 )
    cmsprel.SetTextSize ( 0.04 )
    cmsprel.SetTextColor(    1 )
    cmsprel.SetTextFont (   62 )
    cmsprel.AddText(dataset)
    
##     lumi     =  TPaveText(lowX+0.38, lowY+0.061, lowX+0.45, lowY+0.161, "NDC")
##     lumi.SetBorderSize(   0 )
##     lumi.SetFillStyle(    0 )
##     lumi.SetTextAlign(   12 )
##     lumi.SetTextSize ( 0.04 )
##     lumi.SetTextColor(    1 )
##     lumi.SetTextFont (   62 )
##     lumi.AddText(dataset)
    
    chan     =  TPaveText(lowX+0.68, lowY+0.061, lowX+0.80, lowY+0.161, "NDC")
    chan.SetBorderSize(   0 )
    chan.SetFillStyle(    0 )
    chan.SetTextAlign(   12 )
    chan.SetTextSize ( 0.05 )
    chan.SetTextColor(    1 )
    chan.SetTextFont (   62 )
    chan.AddText(channel)
    
    return cmsprel, chan

