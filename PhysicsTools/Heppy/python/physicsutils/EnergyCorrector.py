from ROOT import TH1F, TH2F, TFile


class EnergyCorrector( object ):
    """Generic energy corrector"""
    
    def __init__(self, fnam, histnam='h_cor'):
        """
        fnam is a root file containing a 1D histogram giving
        the correction factor as a function of eta.
        """
        self.file = TFile(fnam)
        if self.file.IsZombie():
            raise ValueError(fnam+' cannot be opened')
        self.hist = self.file.Get(histnam)
        if self.hist==None:
            raise ValueError('{h} cannot be found in {f}'.format(h=histnam,
                                                                 f=fnam))
            

    def correct_p4(self, p4):
        """
        returns the corrected 4-momentum.
        The 4 momentum is expected to behave as the one of the Candidate class
        """
        eta = p4.eta()
        pt = p4.pt()
        return pt*self.correction_factor(pt, eta)

    def correction_factor(self, pt, eta):
        """
        returns the correction factor.
        takes also pt as this class could be generalized for a 2D calibration.
        """
        etabin = self.hist.FindBin(eta)
        shift = self.hist.GetBinContent(etabin)/100.
        return shift
    
        
if __name__ == '__main__':

    import sys
    c = JetEnergyCorrector( sys.argv[1] )
    etas = [-5, -4.5, -4, -3, -2.5, -2, -1, 0, 1, 2, 2.5, 3, 4, 4.5, 5]
    pt = 20.
    print pt
    for eta in etas:
        print eta, c.correction_factor(pt, eta)
        
