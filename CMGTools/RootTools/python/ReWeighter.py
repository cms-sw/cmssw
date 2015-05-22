class ReWeighter(object):

    def __init__(self, name, hto, hfrom):
        self.name = name
        self.hto = hto
        self.hfrom = hfrom
        self.hto.Scale(1/hto.Integral())
        self.hfrom.Scale(1/hfrom.Integral())
        self.weights = self.hto.Clone( '_'.join(['weights',name]))
        self.weights.Divide(hfrom)
        
    def weightStr(self, var=None):
        if var is None:
            var = self.name
        cuts = []
        for i in range(1, self.weights.GetNbinsX()+1):
            xmin = self.weights.GetBinLowEdge(i)
            xmax = self.weights.GetBinLowEdge(i) + self.weights.GetBinWidth(i)
            weight = self.weights.GetBinContent(i)
            print i, xmin, xmax, weight
            cut = '({var}>={xmin} && {var}<{xmax})*{weight}'.format(var=var, xmin=xmin, xmax=xmax, weight=weight)
            cuts.append(cut)
        return ' + '.join(cuts)
