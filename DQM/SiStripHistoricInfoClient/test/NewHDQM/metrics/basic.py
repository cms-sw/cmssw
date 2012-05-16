class BaseMetric:
    "baseclass for all metrics. should not be used on its own"
    def __init__(self):
        self._reference = None
        self._threshold = 1

    def setCache(self, cache):
        self.__cache = cache
    def setReference(self, histo): 
        self._reference = histo
    def setThreshold(self, threshold): 
        self._threshold = threshold
    def setCacheLocation(self, serverUrl, runNr, dataset, histoPath):
         self.__cacheLocation = (serverUrl, runNr, dataset, histoPath)


    def __call__(self, histo, cacheLocation=None):
        if not cacheLocation == None and not self.__cache == None and cacheLocation in self.__cache:
            result, entries = self.__cache[cacheLocation]
        else:
            assert (not histo==None), "reading from cache failed but no histo givento compute metric!"
            result = self.calculate(histo)
            entries = histo.GetEntries()
            if not self.__cache == None:
                self.__cache[cacheLocation] = (result, entries)
        if entries < self._threshold:
            raise StandardError," Number of entries (%s) is below threshold (%s) using '%s'"%(entries, self._threshold, self.__class__.__name__) #, histo.GetName())
        if not len(result) == 2:
            raise StandardError, "calculate needs to return a tuple with the value and the error of the metric!"
        if not "__iter__" in dir(result[1]):
            result = (result[0], (result[1],result[1]))
        else:
            if not len(result[1]) == 2:
                raise StandardError, "when you use assymmetric errors you need to specify exactly two values. You gave: '%s'"%result[1]

        return result

    def calculate(self, histo):
        raise StandardError, "you should not use the baseclass as a metric. Use the derived classes!"
        
        
class SummaryMapPartition(BaseMetric):
    def __init__(self,  binx, nbinsy):
        self.__binx = binx
        self.__nbinsy = nbinsy

    def calculate(self, histo):
        value = 0
        for ybin in range(self.__nbinsy):
            ybin=ybin+1
            value += histo.GetBinContent(self.__binx , ybin)
            value1=value
        value /= self.__nbinsy
        return (value, 0)

class Mean(BaseMetric):
    def calculate(self, histo):
        return (histo.GetMean(), histo.GetMeanError())

class MeanY(BaseMetric):
    def calculate(self, histo):
        sumw     = histo.GetSumOfWeights()
        nentries = histo.GetEntries()
        return (sumw/nentries if nentries else 0, 0) 

#class WeightedMeanY(BaseMetric):
#    def calculate(self, histo):

class MeanDiff(BaseMetric):
    def calculate(self, histo):
        from math import sqrt
        return (histo.GetMean() - self._reference.GetMean(), 
                sqrt(histo.GetMeanError()**2 + self._reference.GetMeanError()**2))    

class Count(BaseMetric):
    def calculate(self, histo):
        return ( histo.GetEntries(), 0)

class MaxBin(BaseMetric):
    def calculate(self, histo):
        bin = histo.GetMaximumBin()
        return ( histo.GetBinCenter(bin), 0) 

class BinCount(BaseMetric):
    def __init__(self,  name, noError = False):
        self.__name = name
        self.__noError = noError

    def calculate(self, histo):
        from math import sqrt
        binNr = self.__name
        if type(self.__name) == type(""):
            binNr = hist.GetXaxis().FindBin(self.__name)
        error = 0
        if not self.__noError:
            error = sqrt(histo.GetBinContent(binNr))
        return ( histo.GetBinContent(binNr), error)

class BinsCount(BaseMetric):
    def __init__(self, startBin):
        self.__loBin = startBin

    def calculate(self, histo):
        from math import sqrt
        sum=float(0.0)
        for bin in range(self.__loBin,histo.GetNbinsX()+1) : sum+=histo.GetBinContent(bin)
        return ( sum, sqrt(1/sum)*sum if sum else 0)   

class NormBinCount(BaseMetric):
    def __init__(self,  name, norm = None):
        self.__name = name
        self.__norm = norm
        self.__iWeightHisto = 0

    def calculate(self, histo):        
        from ROOT import TGraphAsymmErrors
        frac = self.__getWeightOneHisto(histo, self.__name)
        total = self.__getWeightOneHisto( histo, self.__norm)

        if(frac.GetEntries() > total.GetEntries()):
            raise StandardError," comparing '%s' to '%s' in '%s' makes no sense eff > 1!"%(self.__name, self.__norm, histo.GetName())
        
        eff = TGraphAsymmErrors(1)
        eff.BayesDivide(frac, total)
        if eff.GetN() < 1: 
            raise StandardError,"Efficiency cannot be calculated '%s' in '%s'"%(self.__name, histo.GetName())
        return ( eff.GetY()[0], (eff.GetEYlow()[0],eff.GetEYhigh()[0]) )

    def __getWeightOneHisto(self, hist, name):
        """return histo with one bin filled with entries of weight on to match hist at name
        if name == None use integral of hist."""
        from ROOT import TH1D
        from math import sqrt
        result = TH1D( ("%s"%name) + "%s"%self.__iWeightHisto,
                       ("%s"%name) + "%s"%self.__iWeightHisto,1,0,1)
        self.__iWeightHisto+=1
        bin = hist.GetSumOfWeights()
        if not name == None:
            binNr = name
            if type(name) == type(""):
                binNr = hist.GetXaxis().FindBin(name)
            bin =  hist.GetBinContent(binNr)
        result.Sumw2()
        result.SetBinContent(1, bin )
        result.SetBinError(1, sqrt(bin))
        return result
    


class Ratio(BaseMetric):
    def __init__(self,  low, high):
        self.__low = low
        self.__high = high

    def calculate(self, histo):
        from math import sqrt
        s = histo.Integral(histo.FindBin( self.__low),
                           histo.FindBin( self.__high))
        T = histo.Integral()
        B = T-s
        return (  s / B if B else 0,
                  sqrt( s + s*s/B ) / B if s and B else 1/B if B else 0 )

class Ratio1(BaseMetric):
    def __init__(self,  low, high):
        self.__low = low
        self.__high = high

    def calculate(self, histo):
        from math import sqrt
        s = histo.Integral(histo.FindBin( self.__low),
                           histo.FindBin( self.__high))
        Nbins = histo.GetSize()
        T = histo.Integral(0,Nbins)
        B = T-s
        return (  B / s if s else 0,
                  sqrt( B + B*B/s ) / s if s and B else 1/s if s else 0 )

class Fraction(BaseMetric):
    def __init__(self, low, high):
        self.__low = low
        self.__high = high

    def calculate(self, histo):
        from math import sqrt
        s = histo.Integral(histo.FindBin( self.__low),
                           histo.FindBin( self.__high))
        T = histo.Integral()
        B = T-s
        return ( s/T if T else 0,
                 sqrt( s*s*B + B*B*s ) / (T*T) if s and B else 1/T if T else 0)

class Fraction1(BaseMetric): 
    def __init__(self, low, high):
        self.__low = low
        self.__high = high

    def calculate(self, histo):
        from math import sqrt
        s = histo.Integral(histo.FindBin( self.__low),
                           histo.FindBin( self.__high))
        print "AAA",self.__high,self.__high+1
        Nbins = histo.GetSize()
#        T = histo.Integral(0,self.__high+1)
        T = histo.Integral(0,Nbins)
        B = T-s
        return ( B/T if T else 0,
                 sqrt( s*s*B + B*B*s ) / (T*T) if s and B else 1/T if T else 0)

    
class FractionInBin(BaseMetric):
    def __init__(self, bin):
        self.__bin = bin

    def calculate(self, histo):
        from math import sqrt
        s = histo.GetBinContent(self.__bin)
        T = histo.GetEntries()
        return ( s/T if T else 0,
                 sqrt(1/T + 1/s)*s/T if T else 0)
    
class FractionInBinArray(BaseMetric):
    def __init__(self, binsnum, binsden):
        self.__binsnum = binsnum
        self.__binsden = binsden
        
    def calculate(self, histo):
        from math import sqrt
        num=float(0.0)
        den=float(0.0)
        
        for bn in self.__binsnum : num+=histo.GetBinContent(bn)
        for bd in self.__binsden : den+=histo.GetBinContent(bd)
        return ( num/den if den else 0,
                 sqrt(1/num + 1/den)*num/den if den and num else 0)    

class Quantile(BaseMetric):
    def __init__(self,  frac = 0.95):
        self.__frac = float(frac)
        from ROOT import gROOT
        gROOT.LoadMacro( 'Quantile.h+' )

    def calculate(self, histo):
        from ROOT import Quantile
        q = Quantile(histo)
        "frac is the fraction from the left"
        quant, quantErr = q.fromHead(self.__frac)
	if quantErr == 0.:
          raise StandardError," Quantile cannot be calculated!"
        return (quant,quantErr)

#--- statistical Tests            
class Kolmogorov(BaseMetric):
    def calculate(self, histo):
        k = histo.KolmogorovTest( self._reference)
        return (k, 0)

class Chi2(BaseMetric):
    def calculate(self, histo):
        chi2 = histo.Chi2Test( self._reference, "UUNORMCHI2")
        return (chi2, 0 )

class NormChi2(BaseMetric):
    def calculate(self, histo):
        chi2 = histo.Chi2Test( self._reference, "UUNORMCHI2/NDF")
        return (chi2, 0 )
