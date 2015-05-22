from ROOT import TH1, kViolet, kMagenta, kOrange, kRed, kBlue

class Style:
    def __init__(self,
                 markerStyle = 8,
                 markerColor = 1,
                 markerSize = 1,
                 lineStyle = 1,
                 lineColor = 1,
                 lineWidth = 2,
                 fillColor = None,
                 fillStyle = 1001 ):
        self.markerStyle = markerStyle
        self.markerColor = markerColor
        self.markerSize = markerSize
        self.lineStyle = lineStyle
        self.lineColor = lineColor
        self.lineWidth = lineWidth
        if fillColor is None:
            self.fillColor = lineColor
        else:
            self.fillColor = fillColor
        self.fillStyle = fillStyle

    def formatHistoAxis( self, hist ):
        hist.GetXaxis().SetTitleSize(0.05)
        hist.GetYaxis().SetTitleSize(0.05)
        hist.GetYaxis().SetTitleOffset(1.7)        
        
    def formatHisto( self, hist, title=None):
        hist.SetMarkerStyle( self.markerStyle )
        hist.SetMarkerColor( self.markerColor )
        hist.SetMarkerSize( self.markerSize )
        hist.SetLineStyle( self.lineStyle )
        hist.SetLineColor( self.lineColor )
        hist.SetLineWidth( self.lineWidth )
        hist.SetFillColor( self.fillColor )
        hist.SetFillStyle( self.fillStyle )
        self.formatHistoAxis( hist )
        if title!=None:
            hist.SetTitle( title )
        return hist

def formatPad( pad ):
    pad.SetLeftMargin(0.15)
    pad.SetBottomMargin(0.15)
    #pad.SetLeftMargin(0.)
    #pad.SetBottomMargin(0.)


# the following standard files are defined and ready to be used.
# more standard styles can be added on demand.
# user defined styles can be created in the same way in any python module

sBlack  = Style()
sData   = Style(fillStyle=0, markerSize=1.3)
sBlue   = Style(markerColor=4, fillColor=4)
sGreen  = Style(markerColor=8, fillColor=8)
sRed    = Style(markerColor=2, fillColor=2)
sYellow = Style(lineColor=1, markerColor=5, fillColor=5)
sViolet = Style(lineColor=1, markerColor=kViolet, fillColor=kViolet)

qcdcol      = kMagenta - 10
sHTT_QCD    = Style(lineColor=1, markerColor=qcdcol, fillColor = qcdcol)
dycol       = kOrange - 4 
sHTT_DYJets = Style(lineColor=1, markerColor=dycol , fillColor = dycol)
wcol        = kRed+2 
sHTT_WJets  = Style(lineColor=1, markerColor=wcol  , fillColor = wcol)
ttcol       = kBlue-8
sHTT_TTJets = Style(lineColor=1, markerColor=ttcol , fillColor = ttcol)
sHTT_Higgs  = Style(lineColor=4, markerColor=2, lineStyle=2 , fillColor = 0)
zlcol       = kBlue
sHTT_ZL     = Style(lineColor=1, markerColor=zlcol , fillColor = zlcol)


sBlackSquares = Style( markerStyle = 21)
sBlueSquares  = Style( lineColor=4, markerStyle = 21, markerColor=4 )
sGreenSquares = Style( lineColor=8, markerStyle = 21, markerColor=8 )
sRedSquares   = Style( lineColor=2, markerStyle = 21, markerColor=2 )


styleSet = [sBlue, sGreen, sRed, sYellow, sViolet, sBlackSquares, sBlueSquares, sGreenSquares, sRedSquares]
iStyle = 0

def nextStyle():
    global iStyle
    style = styleSet[iStyle]
    iStyle = iStyle+1
    if iStyle>=len(styleSet):
        iStyle = 0
    return style
