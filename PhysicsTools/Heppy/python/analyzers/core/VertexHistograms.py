from ROOT import TFile, TH1F

class VertexHistograms(object):

    def __init__(self, filename ):
        self.file = TFile(filename, 'recreate')
        self.hist = TH1F('pileup','', 70, 0, 70)
        self.mindist = TH1F('mindist','Minimum distance between 2 vertices', 100, 0, 0.1)

    def write(self):
        self.file.Write()
