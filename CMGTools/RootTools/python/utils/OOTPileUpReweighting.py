import copy
from CMGTools.RootTools.RootInit import *
from ROOT import TFile, TH2F

class OOTPileUpReweighting(object):
    def __init__(self, fnam1, fnam2):
        self.files = []
        self.hist1 = self.load( fnam1, 'h1')
        self.hist2 = self.load( fnam2, 'h2')
        self.pdf1 = self.buildPdf(self.hist1)
        self.pdf2 = self.buildPdf(self.hist2)
        
    def load(self, fileName, ext):
        file = TFile( fileName )
        oorhist = file.Get('MUON_Denom/MUON_Denom_h_pup_VS_pu')
        oorhist.SetName('_'.join([oorhist.GetName(), ext]))
        print 'loading', oorhist.GetName()
        self.files.append( file )
        return oorhist
    
    def buildPdf(self, oorhist):
        xproj = oorhist.ProjectionX()
        pdf = oorhist.Clone('_'.join( [oorhist.GetName(), 'pdf'] ))
        pdf.Reset()
        for binx in range(1, oorhist.GetNbinsX()+1):
            bincont = float(xproj.GetBinContent(binx))
            # print binx, bincont
            if bincont:
                for biny in range(1, oorhist.GetNbinsY()+1):
                    newval = oorhist.GetBinContent(binx, biny) / bincont
                    # print '2D', binx, biny, oorhist.GetBinContent(binx, biny), newval
                    pdf.SetBinContent( binx, biny, newval )
                    # print 'get', pdf.GetBinContent(binx, biny)
        return pdf
    
    def getWeight(self, npu, npup, oneToTwo=True):
        pdf1 = self.pdf1
        pdf2 = self.pdf2
        bin = pdf1.FindBin( npu, npup)
        p1 = pdf1.GetBinContent(bin)
        p2 = pdf2.GetBinContent(bin)
        if oneToTwo:
            if p1:
                return p2/p1
            else:
                return 0.
        else:
            if p2:
                return p1/p2
            else:
                return 0.
            
    def drawSlice(self, npu):
        self.p1 = self.pdf1.ProjectionY('_'.join([self.pdf1.GetName(), 'py']),
                                        npu+1, npu+1, "")
        self.p2 = self.pdf2.ProjectionY('_'.join([self.pdf2.GetName(), 'py']),
                                        npu+1, npu+1, "")
        self.p1.Draw()
        self.p2.Draw('same')



base = '/'.join([os.environ['CMSSW_BASE'],'/src/CMGTools/RootTools/data/Reweight'])

ootPUReweighter = OOTPileUpReweighting(
    '/'.join([base, 'DYJetsChamonix/EfficiencyAnalyzer/EfficiencyAnalyzer.root']),
    '/'.join([base, 'DYJetsFall11/EfficiencyAnalyzer/EfficiencyAnalyzer.root']),    
    )


if __name__ == '__main__':
    oot = None
    if len(sys.argv)>2:
        fnam1 = sys.argv[1]
        fnam2 = sys.argv[2]
        oot = OOTPileUpReweighting(fnam1, fnam2)
