import ROOT as rt
import numpy as np

class PuJetIDWP:
    def __init__(self):

        # tuning in 76X
        self.xbins = np.array([0.00,2.50,2.75,3.00,5.00])
        self.ybins = np.array([0,10,20,30,50])
        grid = rt.TH2F('grid','',4,self.xbins,4,self.ybins)

        loose = grid.Clone('loose')
        loose.SetBinContent(1,1,-0.96); loose.SetBinContent(1,2,-0.96); loose.SetBinContent(1,3,-0.96); loose.SetBinContent(1,4,-0.93);
        loose.SetBinContent(2,1,-0.62); loose.SetBinContent(2,2,-0.62); loose.SetBinContent(2,3,-0.62); loose.SetBinContent(2,4,-0.52);
        loose.SetBinContent(3,1,-0.53); loose.SetBinContent(3,2,-0.53); loose.SetBinContent(3,3,-0.53); loose.SetBinContent(3,4,-0.39);
        loose.SetBinContent(4,1,-0.49); loose.SetBinContent(4,2,-0.49); loose.SetBinContent(4,3,-0.49); loose.SetBinContent(4,4,-0.31);

        medium = grid.Clone('medium')
        medium.SetBinContent(1,1,-0.58); medium.SetBinContent(1,2,-0.58); medium.SetBinContent(1,3,-0.58); medium.SetBinContent(1,4,-0.20);
        medium.SetBinContent(2,1,-0.52); medium.SetBinContent(2,2,-0.52); medium.SetBinContent(2,3,-0.52); medium.SetBinContent(2,4,-0.39);
        medium.SetBinContent(3,1,-0.40); medium.SetBinContent(3,2,-0.40); medium.SetBinContent(3,3,-0.40); medium.SetBinContent(3,4,-0.24);
        medium.SetBinContent(4,1,-0.36); medium.SetBinContent(4,2,-0.36); medium.SetBinContent(4,3,-0.36); medium.SetBinContent(4,4,-0.19);

        tight = grid.Clone('tight')
        tight.SetBinContent(1,1,0.09); tight.SetBinContent(1,2,0.09); tight.SetBinContent(1,3,0.09); tight.SetBinContent(1,4,0.52);
        tight.SetBinContent(2,1,-0.37); tight.SetBinContent(2,2,-0.37); tight.SetBinContent(2,3,-0.37); tight.SetBinContent(2,4,-0.19);
        tight.SetBinContent(3,1,-0.24); tight.SetBinContent(3,2,-0.24); tight.SetBinContent(3,3,-0.24); tight.SetBinContent(3,4,-0.06);
        tight.SetBinContent(4,1,-0.21); tight.SetBinContent(4,2,-0.21); tight.SetBinContent(4,3,-0.21); tight.SetBinContent(4,4,-0.03);

        self.wps = {}
        self.wps['loose'] = loose
        self.wps['medium'] = medium
        self.wps['tight'] = tight

    def getBin(self, bvec, val):
        return int(bvec.searchsorted(val, side="right")) - 1

    def passWP(self, jet, wp='loose'):
        if wp not in self.wps: return False
        wph = self.wps[wp]
        #if jet is a simple class with attributes
        if isinstance(getattr(jet, "pt"), float):
            pt   = getattr(jet, "pt")
            aeta = abs(getattr(jet, "eta"))
        #if jet is a heppy Jet object
        else:
            pt   = jet.pt()
            aeta = abs(jet.eta())

        binx = self.getBin(self.xbins,aeta)
        biny = self.getBin(self.ybins,pt)

        if binx == wph.GetNbinsX(): return True
        if biny == wph.GetNbinsY(): return True        

        cut = wph.GetBinContent(binx,biny)
        return jet.puMva > cut
