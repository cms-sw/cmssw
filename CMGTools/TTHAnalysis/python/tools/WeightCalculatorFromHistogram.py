class WeightCalculatorFromHistogram(object):
    def __init__(self,histogram):
        self.histo = histogram
    
    
    def getWeight(self,x, y = None):
        if y is None: #1D
            bin = self.histo.GetXaxis().FindBin(x)
            if bin > self.histo.GetXaxis().GetNbins():
                bin = self.histo.GetXaxis().GetNbins()
            return self.histo.GetBinContent(bin)
        else:
            binx = self.histo.GetXaxis().FindBin(x)
            biny = self.histo.GetYaxis().FindBin(y)
            if binx > self.histo.GetXaxis().GetNbins():
                binx = self.histo.GetXaxis().GetNbins()
            if biny > self.histo.GetYaxis().GetNbins():
                biny = self.histo.GetYaxis().GetNbins()
            return self.histo.GetBinContent(binx,biny)


    def getWeightErr(self,x, y = None):
        if y is None: #1D
            bin = self.histo.GetXaxis().FindBin(x)
            if bin > self.histo.GetXaxis().GetNbins():
                bin = self.histo.GetXaxis().GetNbins()
            return self.histo.GetBinError(bin)
        else:
            binx = self.histo.GetXaxis().FindBin(x)
            biny = self.histo.GetYaxis().FindBin(y)
            if binx > self.histo.GetXaxis().GetNbins():
                binx = self.histo.GetXaxis().GetNbins()
            if biny > self.histo.GetYaxis().GetNbins():
                biny = self.histo.GetYaxis().GetNbins()
            return self.histo.GetBinError(binx,biny)
            
