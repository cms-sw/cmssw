from ROOT import TCanvas, TPad, TH1F, TLegend


class HistComparator(object):
    '''Comparison of two histograms.

    The histograms are superimposed in the top pad, and their ratio
    is shown in the bottom pad.

    This class manipulates histograms that behave like a TH1.
    (TH1F, TH1D, and probably TProfile and RooDataSets).
    '''

    CAN_NUM = 0
    
    def __init__(self, name, h1, h2, title1=None, title2=None):
        '''Constructor.

        h1, h2: two TH1.
        title1 and title2: will be used in the legend.
        name: will be used on the X axis.
        '''
        self.set(name, h1, h2, title1, title2)
        # self.can, self.pad_main, self.pad_ratio = self.buildCanvas()
        self.can = None
        self.pad_main = None
        self.pad_ratio = None
        self.can_simple = None
        self.simple_state = False

    def __del__(self):
        self.closeCanvas()

    def closeCanvas(self):
        self.can.Close()
        self.can = None
        
    def set(self, name, h1, h2, title1=None, title2=None):
        '''Change the histograms, in case we want to keep the same canvas
        to draw different histograms.'''
        self.name = name 
        self.h1 = h1
        self.h2 = h2
        if title1 is None:
            self.title1 = h1.GetTitle()
        else:
            self.title1 = title1
        if title2 is None:
            self.title2 = h2.GetTitle()
        else:
            self.title2 = title2            
        self.h1.SetTitle('')
        self.h2.SetTitle('')
        self.ratio = h1.Clone( '_'.join([name, 'ratio']))
        self.ratio.Divide(self.h2)
        self.ratio.SetStats(0)


    def draw(self, simple=False, opt2='', opt1=''):
        '''The canvas is created if needed.'''
        # import pdb; pdb.set_trace()
        if simple!=self.simple_state:
             self.simple_state=simple
             if self.can:
                 self.closeCanvas()
        if type(self.can) is not TCanvas:
            self.can, self.pad_main, self.pad_ratio = self.buildCanvas(simple)
        self.can.Draw()

        if not simple:
            self.pad_main.Draw()
            self.pad_ratio.Draw()
            self.pad_main.cd()
        self.h2.SetStats(0)
        self.h2.Draw(opt2)
        self.h1.Draw(opt1+'same')
        self.h2.GetYaxis().SetRangeUser(1e-3,
                                        self.ymax(self.h1, self.h2)*1.2)
        self.h2.GetYaxis().SetLabelSize(0.045)
        self.legend = TLegend(0.6, 0.7, 0.9, 0.9)
        self.legend.AddEntry(self.h1, self.title1, 'lpf')
        self.legend.AddEntry(self.h2, self.title2, 'lpf')
        self.legend.SetFillColor(0)
        self.legend.Draw('same')
        if not simple:
            self.pad_ratio.cd()
            self.ratio.Draw()
            self.ratio.GetYaxis().SetNdivisions(5)
            self.ratio.GetYaxis().SetTitle('{t1}/{t2}'.format(t1=self.title1,
                                                              t2=self.title2))
            self.ratio.GetYaxis().SetTitleSize(0.1)
            self.ratio.GetYaxis().SetTitleOffset(0.5)
            self.ratio.GetXaxis().SetTitle(self.h1.GetXaxis().GetTitle())
            self.ratio.GetXaxis().SetTitleSize(0.13)
            self.ratio.GetXaxis().SetTitleOffset(0.9)
            rls = 0.075
            self.ratio.GetYaxis().SetLabelSize(rls)
            self.ratio.GetXaxis().SetLabelSize(rls)
            self.ratio.GetYaxis().SetRangeUser(0.5, 1.5)
        self.can.Modified()
        self.can.Update()
        self.can.cd()

    def save(self, fname=None):
        if fname is None:
            fname = self.name
        self.can.SaveAs( self.name + '.png' )

    def buildCanvas(self, simple=False):
        can = TCanvas('can_{num}'.format(num=self.__class__.CAN_NUM),
                      self.name,600,600)
        self.__class__.CAN_NUM += 1
        can.cd()
        can.Draw()
        pad = None
        padr = None
        if not simple: 
            sep = 0.35
            pad = TPad('pad','',0.01,sep,0.99, 0.99)
            pad.SetBottomMargin(0.04)
            padr = TPad('padr','',0.01, 0.01, 0.99, sep+0.02)
            padr.SetTopMargin(0.04)
            padr.SetBottomMargin(0.3)
            padr.SetGridy()
        return can, pad, padr

    def ymax(self, h1=None, h2=None):
        '''Returns the best y axis maximum so that h1 and h2 are both visible.'''
        def getmax(h):
            return  h.GetBinContent(h.GetMaximumBin())
        if h1 is None: h1 = self.h1
        if h2 is None: h2 = self.h2
        ymax = max(getmax(h1), getmax(h2))
        if ymax == 0:
            ymax = 1
        return ymax
        
        
        
    
