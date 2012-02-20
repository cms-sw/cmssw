from basic import BaseMetric

class Landau(BaseMetric):
    def __init__(self, diseredParameter, minVal, maxVal, paramDefaults):
        BaseMetric.__init__(self)
        self.range = [minVal, maxVal]
        self.parameters = paramDefaults
        assert diseredParameter in [0,1,2], "can only get parameter 0, 1 or 2 not '%s'"%desiredParameter
        self.desired = diseredParameter
        
    def calculate(self, histo):
        from ROOT import TF1
        fit = TF1("landau","[2]*TMath::Landau(x,[0],[1],0)", *(self.range))
        fit.SetParameters(*(self.parameters))
        histo.Fit(fit,"QOR")
        result = (fit.GetParameter(self.desired), fit.GetParError(self.desired))
        del fit
        return result

