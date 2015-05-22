import copy
from ROOT import THStack, gPad, kGray
from CMGTools.RootTools.Style import sBlue,sBlack

class Stack:
    '''Attempt to overcome the deficiencies of the THStack class.

    Contains:
    - hists    : a list of Histogram (from this package,
    we\'re not taking about ROOT histograms here
    - integral : the integral of the stack.
    - totalHist, the sum of all histograms in the stack with a layer < 1000.
    
    '''

    STAT_ERRORS = True
    STYLE = copy.copy(sBlack)
    STYLE.markerStyle = 1 
    
    def __init__(self, name, ytitle=None):
        self.name = name
        self.hists = []
        self.integral = 0
        self.totalHist = None
        self.statErrors = Stack.STAT_ERRORS
        self.style = Stack.STYLE
        self.ytitle = ytitle
        self.axisWasSet = False
        
    def Add(self, hist):
        '''Add an Histogram.'''
        # one MUST do a deepcopy here. 
        self.hists.append( copy.deepcopy(hist) )
        self.integral += hist.Yield( weighted=True )
        if self.totalHist is None:
            self.totalHist = copy.deepcopy( hist )
        elif hist.layer<1000:
            self.totalHist.Add( hist )
        
    def Draw(self, opt='',
             xmin=None, xmax=None, ymin = None, ymax=None):
        '''Draw the stack. opt are the ROOT options'''
        if len( self.hists )==0:
            return
        self.obj = THStack(self.name,'')
        self.totalHist = None
        for hist in self.hists:
            self.obj.Add(hist.weighted)
            if self.totalHist is None:
                self.totalHist = copy.deepcopy( hist )
            elif hist.layer<1000.:
                self.totalHist.Add( hist )
        self._Draw( opt, self.hists, xmin, xmax, ymin, ymax)

    def _Draw( self, opt, hists, xmin, xmax, ymin, ymax):
        self.SetStyle(self.style)
        # drawing the first histogram in the stack
        # as a support histo.
        # otherwise, can't change y axis range (ROOT!@#!)
        # we draw it as hist so that the markers don't appear.
        # hist = hists[0]
        hist = self.totalHist
        # hist.Draw('hist')
        hist.Draw(opt)  #### riccardo
        if self.ytitle is not None:
            hist.GetYaxis().SetTitle( self.ytitle )
        self.obj.Draw( opt+'same' )  ### riccardo
        #self.obj.Draw( opt )  ### riccardo
        # need to redraw the axes, which are now "under"
        # the stacked histograms. 
        hist.Draw('axissame')
        if ymin is None:
            ymin = 0.1
        if ymax is None:
            ymax = hist.GetMaximum()*1.3
        if not self.axisWasSet:
            self.axisWasSet = True
            hist.GetYaxis().SetRangeUser( ymin, ymax )
        if xmin is not None and xmax is not None:
            # without the little offset,
            # the range includes the bin over xmax !@(*
            xmax = xmax - 1e-9
            hist.GetXaxis().SetRangeUser( xmin, xmax )
        self._DrawStatErrors()
        # self._updateTitles( hist )
       

    def _DrawStatErrors(self):
        '''Draw statistical errors if statErrors is True.'''
        if self.statErrors is False:
            return
        #self.totalHist.weighted.SetFillColor(kGray)
        self.totalHist.weighted.SetFillColor(1)
        self.totalHist.weighted.SetFillStyle(3544)
        self.totalHist.Draw('samee2')
        
    def SetStyle(self, style ):
        '''Set the style for the total histogram'''
        self.style = style 
        self.totalHist.SetStyle( self.style )
        if gPad:
            gPad.Update()

    def DrawNormalized(self, opt='',
                       xmin=None, xmax=None, ymin = None, ymax=None ):
        '''Draw a normalized version of the stack (integral=1).'''
        if len( self.hists )==0:
            return
        integral = 0
        self.normHists = []
        self.obj = THStack(self.name,'')
        self.totalHist = None
        for hist in self.hists:
            normHist = copy.deepcopy(hist)
            normHist.Scale(1/self.integral)
            self.normHists.append( normHist )
            self.obj.Add( normHist.weighted)
            if self.totalHist is None:
                self.totalHist = copy.deepcopy( normHist )
            else: 
                self.totalHist.Add( normHist )
        # self.obj.Draw( opt )
        # self._updateTitles()
        self._Draw( opt, self.normHists, xmin, xmax, ymin, ymax)
        
    def Normalize(self):
        '''Normalize the stack.

        All histograms in the stack are scaled
        so that the integral of the stack is 1.'''
        for hist in self.hists:
            hist.weighted.Scale( 1/self.integral )        

    def Divide(self, otherHist):
        '''Divide the stack by an histogram.'''
        for hist in self.hists:
            hist.weighted.Divide(otherHist)
    
    def _updateTitles( self,hist ):
        '''Update the axis titles of the stack to the titles of the first histogram in the stack.'''
        print 'update', hist.GetYaxis().GetTitle()
        self.obj.GetXaxis().SetTitle( hist.GetXaxis().GetTitle() )
        self.obj.GetYaxis().SetTitle( hist.GetYaxis().GetTitle() )
        
    def Blind(self,minx, maxx):
        for hist in self.hists:
            hist.Blind(minx, maxx)
        # self.totalHist.Blind(minx, maxx)
