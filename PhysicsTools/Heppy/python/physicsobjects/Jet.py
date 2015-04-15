from PhysicsTools.Heppy.physicsobjects.PhysicsObject import *

loose_WP = [
    (0, 2.5, -0.8),
    (2.5, 2.75, -0.74),
    (2.75, 3.0, -0.68),
    (3.0, 5.0, -0.77),
    ]

# Working point 2 May 2013 (Phil via H2tau list)
loose_53X_WP = [
    (0, 2.5, -0.63),
    (2.5, 2.75, -0.60),
    (2.75, 3.0, -0.55),
    (3.0, 5.2, -0.45),
    ]

_btagWPs = {
    "TCHEL": ("pfTrackCountingHighEffBJetTags", 1.7),
    "TCHEM": ("pfTrackCountingHighEffBJetTags", 3.3),
    "TCHPT": ("pfTrackCountingHighPurBJetTags", 3.41),
    "JPL": ("pfJetProbabilityBJetTags", 0.275),
    "JPM": ("pfJetProbabilityBJetTags", 0.545),
    "JPT": ("pfJetProbabilityBJetTags", 0.790),
    "CSVL": ("combinedSecondaryVertexBJetTags", 0.244),
    "CSVM": ("combinedSecondaryVertexBJetTags", 0.679),
    "CSVT": ("combinedSecondaryVertexBJetTags", 0.898),
    "CSVv2IVFL": ("pfCombinedInclusiveSecondaryVertexV2BJetTags", 0.423),
    "CSVv2IVFM": ("pfCombinedInclusiveSecondaryVertexV2BJetTags", 0.814),
    "CSVv2IVFT": ("pfCombinedInclusiveSecondaryVertexV2BJetTags", 0.941),
    "CMVAL": ("pfCombinedMVABJetTags", 0.630), # for same b-jet efficiency of CSVv2IVFL on ttbar MC, jet pt > 30
    "CMVAM": ("pfCombinedMVABJetTags", 0.732), # for same b-jet efficiency of CSVv2IVFM on ttbar MC, jet pt > 30
    "CMVAT": ("pfCombinedMVABJetTags", 0.813), # for same b-jet efficiency of CSVv2IVFT on ttbar MC, jet pt > 30

}

class Jet(PhysicsObject):   
    def __init__(self, *args, **kwargs):
        super(Jet, self).__init__(*args, **kwargs)
        self._physObjInit()

    def _physObjInit(self):
        self._rawFactorMultiplier = 1.0
        self._leadingTrack = None
        self._leadingTrackSearched = False

    def jetID(self,name=""):
        if not self.isPFJet():
            raise RuntimeError, "jetID implemented only for PF Jets"
        eta = abs(self.eta());
        energy = (self.p4()*self.rawFactor()).energy();
        chf = self.chargedHadronEnergy()/energy;
        nhf = self.neutralHadronEnergy()/energy;
        phf = self.neutralEmEnergy()/energy;
        muf = self.muonEnergy()/energy;
        elf = self.chargedEmEnergy()/energy;
        chm = self.chargedHadronMultiplicity();
        npr = self.chargedMultiplicity() + self.neutralMultiplicity();
        #if npr != self.nConstituents():
        #    import pdb; pdb.set_trace()
        if name == "POG_PFID":  
            if   self.jetID("POG_PFID_Tight")  : return 3;
            elif self.jetID("POG_PFID_Medium") : return 2;
            elif self.jetID("POG_PFID_Loose")  : return 1;
            else                               : return 0;
        
        if name == "POG_PFID_Loose":    return (npr>1 and phf<0.99 and nhf<0.99 and muf < 0.8) and (eta>2.4 or (elf<0.99 and chf>0 and chm>0));
        if name == "POG_PFID_Medium":   return (npr>1 and phf<0.95 and nhf<0.95 and muf < 0.8) and (eta>2.4 or (elf<0.99 and chf>0 and chm>0));
        if name == "POG_PFID_Tight":    return (npr>1 and phf<0.90 and nhf<0.90 and muf < 0.8) and (eta>2.4 or (elf<0.90 and chf>0 and chm>0));
        if name == "VBFHBB_PFID_Loose":  return (npr>1 and phf<0.99 and nhf<0.99);
        if name == "VBFHBB_PFID_Medium": return (npr>1 and phf<0.99 and nhf<0.99) and ((eta<=2.4 and nhf<0.9 and phf<0.9 and elf<0.99 and muf<0.99 and chf>0 and chm>0) or eta>2.4);
        if name == "VBFHBB_PFID_Tight":  return (npr>1 and phf<0.99 and nhf<0.99) and ((eta<=2.4 and nhf<0.9 and phf<0.9 and elf<0.70 and muf<0.70 and chf>0 and chm>0) or eta>2.4);
        raise RuntimeError, "jetID '%s' not supported" % name

    def looseJetId(self):
        '''PF Jet ID (loose operation point) [method provided for convenience only]'''
        return self.jetID("POG_PFID_Loose")

    def puMva(self, label="pileupJetId:fullDiscriminant"):
        return self.userFloat(label)

    def puJetId(self, label="pileupJetId:fullDiscriminant"):
        '''Full mva PU jet id'''

        puMva = self.puMva(label)
        wp = loose_53X_WP
        eta = abs(self.eta())
        
        for etamin, etamax, cut in wp:
            if not(eta>=etamin and eta<etamax):
                continue
            return puMva>cut
        
    def rawFactor(self):
        return self.jecFactor('Uncorrected') * self._rawFactorMultiplier
    def setRawFactor(self, factor):
        self._rawFactorMultiplier = factor/self.jecFactor('Uncorrected')

    def btag(self,name):
        ret = self.bDiscriminator(name)
        if ret == -1000 and name.startswith("pf"):
            ret = self.bDiscriminator(name[2].lower()+name[3:])
        return ret
 
    def btagWP(self,name):
        global _btagWPs
        (disc,val) = _btagWPs[name]
        return self.btag(disc) > val

    def leadingTrack(self):
        if self._leadingTrackSearched :
            return self._leadingTrack
        self._leadingTrackSearched = True
        self._leadingTrack =  max( self.daughterPtrVector() , key = lambda x : x.pt() if  x.charge()!=0 else 0. )
        if self._leadingTrack.charge()==0: #in case of "all neutral"
            self._leadingTrack = None
        return self._leadingTrack

    def leadTrackPt(self):
        lt=self.leadingTrack()
        if lt :
             return lt.pt()
        else :
             return 0. 

class GenJet( PhysicsObject):
    pass

