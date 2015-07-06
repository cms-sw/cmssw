class RecEffCorrection(object):
    def effCor(self, pt, eta ):
        if self.inBarrel(eta):
            return self.effCorBarrel( pt )
        else:
            return self.effCorEndcaps( pt )


class RefEffCorrectionEle( RecEffCorrection ):
    
    def effCorBarrel(self, pt):
        # if pt<10: return (1,0)
        # elif pt<15: return (1.11, 0.08)
        # elif pt<20: return (0.98, 0.02)
        # else: return (0.990, 0.002)
        if pt<20: return (1.,0.)
        elif pt<30: return (0.9359,0.)
        else: return (1.0273, 0.)

    def effCorEndcaps(self, pt):
        # if pt<10: return (1,0)
        # elif pt<15: return (1.19, 1.13)
        # elif pt<20: return (1.06, 0.05)
        # else: return (1.05, 0.01)
        if pt<20: return (1,0)
        elif pt<30: return (0.9070,0.)
        else: return (0.9662, 0.01)

    def inBarrel(self, eta):
        if abs(eta)<1.479: return True
        else: return False

        
class RefEffCorrectionMu( RecEffCorrection ):
    
    def effCorBarrel(self, pt):
        if pt<10: return (1,0)
        elif pt<15: return (0.92, 0.01)
        elif pt<20: return (0.948, 0.005)
        else: return (0.9933, 0.0003)

    def effCorEndcaps(self, pt):
        if pt<10: return (1,0)
        elif pt<15: return (0.98, 0.01)
        elif pt<20: return (0.962, 0.008)
        else: return (0.9982, 0.0004)

    def inBarrel(self, eta):
        #FIXME is it the correct barrel endcaps limit? must be in phase with T&P
        if abs(eta)<1.2: return True
        else: return False

recEffMapMu = RefEffCorrectionMu()
recEffMapEle = RefEffCorrectionEle()

        
if __name__ == '__main__':

    pts = [9,11,19,21]
    etas = [0, 1.1, 1.4, 1.7]

    def printCorr( corrMap, pts, etas):
        for pt in pts:
            for eta in etas:
                corr, err = corrMap.effCor(pt, eta)
                print 'pt={pt: 4.2f}  eta={eta: 4.2f}  corr={corr: 4.4f}  err={err: 4.4f}'.format(
                    pt=pt, eta=eta, corr=corr, err=err
                    )
    print 'muons'
    printCorr( recEffMapMu, pts, etas)

    print 'electrons'
    printCorr( recEffMapEle, pts, etas)
 
            
