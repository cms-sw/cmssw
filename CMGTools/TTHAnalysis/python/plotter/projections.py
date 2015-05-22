from math import *
import re

class Projections:
    def __init__(self,name,options):
        self._scenario = 2 if "scenario2" in name else (3 if "scenario3" in name else 1)
        self._tolumi   = options.lumi
        lm = re.search(r"(\d+(\.\d+)?)fb", name)
        if lm: self._tolumi = float(lm.group(1))
        self._energy   = 14   if "14TeV"  in name else 8
        self._lumi     = options.lumi
    def scaleYield(self,process):
        factor = self._tolumi / self._lumi;
        if self._energy == 14:
            if process in [ "QF_data", "FR_data", "TT", "TTG" ]:
                factor *= 946.6/248.9
            elif "ttH" in process:
                factor *=  0.6113/0.1271
            elif process == "TTW":
                factor *= 0.769/0.208
            elif process in [ "TTZ", "TTGStar", "TTWW" ]:
                factor *= 0.958/0.207
            elif process in [ "ZZ", ]:
                factor *= 2.64949/1.21944
            elif process in [ "WZ", ]:
                factor *= 2.79381/1.23344
            elif process in [ "VVV", "WWW", "WWZ"]:
                factor *=  2.5 ## MG5:  WWW = 2.46, WWZ = 2.67
            elif process == "TBZ":
                factor *=  2.8 ## MG5 
            elif process == "WWqq":
                factor *=  2.7 ## MG5 
            else:
                factor = 3 ## FIXME
        return factor 
    def scaleReport(self,report):
        for key in report.keys():
            sf = self.scaleYield(key)
            for i,(cut,(n,e)) in enumerate(report[key]):
                report[key][i][1][0] *= sf
                report[key][i][1][1] *= sf
    def scalePlots(self,plots):
        for key in plots.keys():
            sf = self.scaleYield(key)
            plots[key].Scale(sf)
    def scaleSyst(self,name,value):
        if self._scenario == 1:
            return value
        if self._scenario == 2:
            for N in [ "QCDscale_", "pdf_", "thu_" ]:
                if N in name: return sqrt(value)
        return pow(value, 1.0 / sqrt(self._tolumi / self._lumi))
    def scaleSystTemplate(self,name,nominal,alternate):
        for b in xrange(1,nominal.GetNbinsX()+1):
            y0 = nominal.GetBinContent(b)
            yA = alternate.GetBinContent(b)
            if yA != 0 and y0 != 0:
                alternate.SetBinContent(b, y0 * self.scaleSyst(name,yA/y0))
