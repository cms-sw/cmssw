import copy


class Histogram( object ):
    '''Histogram + a few things.

    This class does not inherit from a ROOT class as we could want to use it
    with a TH1D, TH1F, and even a 2D at some point.

    Histogram contains the original ROOT histogram, obj, and a weighted version,
    weigthed, originally set equal to obj (weight == 1).
    - layer : can be used to order histograms
    - stack : to decide whether the histogram
    should be stacked or not (see the Stack class for more information)
    - name  : user defined histogram. Useful when manipulating several histograms with
    the same GetName(), coming from different TDirectories.
    '''


    def __init__(self, name, obj, layer=0., legendLine=None, stack=True):
        # name is a user defined name
        self.name = name
        self.realName = name # can be different if an alias is set
        if legendLine is None:
            self.legendLine = name
        else:
            self.legendLine = legendLine
        self.obj = obj
        # self.weighted = copy.deepcopy(self.obj)
        self.layer = layer
        self.stack = stack
        self.on = True
        self.style = None
        # after construction, weighted histogram = base histogram
        self.SetWeight(1)


    def Clone(self, newName):
        newHist = copy.deepcopy(self)
        newHist.name = newName
        newHist.legendLine = newName
        return newHist
        
    def __str__(self):
        fmt = '{self.name:<10} / {hname:<50},\t Layer ={self.layer:8.1f}, w = {weighted:8.1f}, u = {unweighted:8.1f}'
        tmp = fmt.format(self=self,
                         hname = self.realName,
                         weighted = self.Yield(weighted=True),
                         unweighted = self.Yield(weighted=False) )
        return tmp

    def Yield(self, weighted=True):
        '''Returns the weighted number of entries in the histogram
        (under and overflow not counted).
        
        Use weighted=False if you want the unweighted number of entries'''
        hist = self.weighted
        if not weighted:
            hist = self.obj
        return hist.Integral( 0, hist.GetNbinsX()+1)

    def Rebin(self, factor):
        '''Rebins by factor'''
        self.obj.Rebin( factor )
        self.weighted.Rebin(factor)

    def Divide(self, other):
        self.obj.Divide( other.obj)
        self.weighted.Divide( other.weighted )
    
    def NormalizeToBinWidth(self):
        '''Divides each bin content and error by the bin size'''
        for i in range (1,self.obj.GetNbinsX()+1) :
           self.obj.SetBinContent(i, self.obj.GetBinContent(i) / self.obj.GetBinWidth(i))
           self.obj.SetBinError  (i, self.obj.GetBinError(i)   / self.obj.GetBinWidth(i))
        for i in range (1,self.weighted.GetNbinsX()+1) :
           self.weighted.SetBinContent(i, self.weighted.GetBinContent(i) / self.weighted.GetBinWidth(i))
           self.weighted.SetBinError  (i, self.weighted.GetBinError(i)   / self.weighted.GetBinWidth(i))
    
    def SetWeight(self, weight):
        '''Set the weight and create the weighted histogram.'''
        self.weighted = copy.deepcopy(self.obj)
        self.weight = weight
        self.weighted.Scale(weight)

    def Scale(self, scale):
        '''Scale the histogram (multiply the weight by scale)'''
        self.SetWeight( self.weight * scale )

    def SetStyle(self, style):
        '''Set the style for the original and weighted histograms.'''
        if style is None:
            return 
        style.formatHisto( self.obj )
        style.formatHisto( self.weighted )
        self.style = style

    def AddEntry(self, legend, legendLine=None):
        '''By default the legend entry is set to self.legendLine of the histogram.'''
        if legendLine == None:
            legendLine = self.legendLine
        opt = 'f'
        if self.weighted.GetFillStyle()==0:
            opt = 'p'
        legend.AddEntry(self.obj, legendLine, opt)

    def Draw(self, opt='', weighted=True):
        '''Draw the weighted (or original) histogram.'''
        if weighted is True:
            self.weighted.Draw(opt)
        else:
            self.obj.Draw(opt)

    def GetXaxis(self, opt='', weighted=True):
        '''All these functions could be written in a clever and compact way'''
        if weighted is True:
            return self.weighted.GetXaxis()
        else:
            return self.obj.GetXaxis()  

    def GetYaxis(self, opt='', weighted=True):
        '''All these functions could be written in a clever and compact way'''
        if weighted is True:
            return self.weighted.GetYaxis()
        else:
            return self.obj.GetYaxis()  

    def GetMaximum(self, opt='', weighted=True):
        '''All these functions could be written in a clever and compact way'''
        if weighted is True:
            return self.weighted.GetMaximum()
        else:
            return self.obj.GetMaximum()  
       
    def Add(self, other, coeff=1):
        '''Add another histogram.
        Provide the optional coeff argument for the coefficient factor (e.g. -1 to subtract)
        '''
        self.obj.Add( other.obj, coeff )
        self.weighted.Add( other.weighted, coeff )
        integral = self.obj.Integral(0, self.obj.GetNbinsX())
        if integral > 0.:
            self.weight = self.weighted.Integral(0, self.weighted.GetNbinsX()+1)/integral
        return self

    def Integral(self, weighted=True, xmin=None, xmax=None ):
        '''
        Returns the weighted or unweighted integral of this histogram.
        If xmin and xmax are None, underflows and overflows are included.
        '''
        if type( weighted ) is not bool:
            raise ValueError('weighted should be a boolean')
        if xmin is not None:
            bmin = self.obj.FindFixBin( xmin )
        else:
            bmin = None
        if xmax is not None:
            bmax = self.obj.FindFixBin( xmax ) - 1 
        else:
            bmax = None
        hist = self.weighted
        if weighted is False:
            hist = self.obj
        if bmin is None and bmax is None:
            return hist.Integral(0, hist.GetNbinsX()+1)
        elif bmin is not None and bmax is not None:
            # import pdb; pdb.set_trace()
            if (xmax - xmin) % self.obj.GetBinWidth(1) != 0:
                raise ValueError('boundaries should define an integer number of bins. nbins=%d, xmin=%3.3f, xmax=%3.3f' % (self.obj.GetNbinsX(), self.obj.GetXaxis().GetXmin(), self.obj.GetXaxis().GetXmax()) )
            return hist.Integral(bmin, bmax)
        else:
            raise ValueError('if specifying one boundary, you must specify the other')


    def DrawNormalized(self):
        '''Draw a normalized version of this histogram.

        The original and weighted histograms stay untouched.'''
        self.obj.DrawNormalized()

    def Normalize(self):
        '''Sets the weight to normalize the weighted histogram to 1.

        In other words, the original histogram stays untouched.'''
        self.Scale( 1/self.Integral() )

    def RemoveNegativeValues(self, hist=None):
        # what about errors??
        if hist is None:
            self.RemoveNegativeValues(self.weighted)
            self.RemoveNegativeValues(self.obj)
        else:
            for ibin in range(1, hist.GetNbinsX()+1):
                if hist.GetBinContent(ibin)<0:
                    hist.SetBinContent(ibin, 0)
                
    def Blind(self, minx, maxx):
        whist = self.weighted
        uwhist = self.weighted
        minbin = whist.FindBin(minx)
        maxbin = min( whist.FindBin(maxx), whist.GetNbinsX())
        for bin in range(minbin, maxbin):
            whist.SetBinContent(bin,0)
            whist.SetBinError(bin,0)
            uwhist.SetBinContent(bin,0)
            uwhist.SetBinError(bin,0)
            
