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
    "TCHEL": ("trackCountingHighEffBJetTags", 1.7),
    "TCHEM": ("trackCountingHighEffBJetTags", 3.3),
    "TCHPT": ("trackCountingHighPurBJetTags", 3.41),
    "JPL": ("jetProbabilityBJetTags", 0.275),
    "JPM": ("jetProbabilityBJetTags", 0.545),
    "JPT": ("jetProbabilityBJetTags", 0.790),
    "CSVL": ("combinedSecondaryVertexBJetTags", 0.244),
    "CSVM": ("combinedSecondaryVertexBJetTags", 0.679),
    "CSVT": ("combinedSecondaryVertexBJetTags", 0.898),
}

class Jet(PhysicsObject):   
    def jetID(self,name=""):
        if not self.isPFJet():
            raise RuntimeError, "jetID implemented only for PF Jets"
        eta = abs(self.eta());
        chf = self.chargedHadronEnergyFraction();
        nhf = self.neutralHadronEnergyFraction();
        phf = self.neutralEmEnergyFraction();
        muf = self.muonEnergyFraction();
        elf = self.chargedEmEnergyFraction();
        chm = self.chargedHadronMultiplicity();
        npr = self.chargedMultiplicity() + self.neutralMultiplicity();
        #if npr != self.nConstituents():
        #    import pdb; pdb.set_trace()
        if name == "POG_PFID":  
            if   self.jetID("POG_PFID_Tight")  : return 3;
            elif self.jetID("POG_PFID_Medium") : return 2;
            elif self.jetID("POG_PFID_Loose")  : return 1;
            else                               : return 0;
        
        if name == "POG_PFID_Loose":    return (npr>1 and phf<0.99 and nhf<0.99) and (eta>2.4 or (elf<0.99 and chf>0 and chm>0));
        if name == "POG_PFID_Medium":   return (npr>1 and phf<0.95 and nhf<0.95) and (eta>2.4 or (elf<0.99 and chf>0 and chm>0));
        if name == "POG_PFID_Tight":    return (npr>1 and phf<0.90 and nhf<0.90) and (eta>2.4 or (elf<0.99 and chf>0 and chm>0));
        if name == "VBFHBB_PFID_Loose":  return (npr>1 and phf<0.99 and nhf<0.99);
        if name == "VBFHBB_PFID_Medium": return (npr>1 and phf<0.99 and nhf<0.99) and ((eta<=2.4 and nhf<0.9 and phf<0.9 and elf<0.99 and muf<0.99 and chf>0 and chm>0) or eta>2.4);
        if name == "VBFHBB_PFID_Tight":  return (npr>1 and phf<0.99 and nhf<0.99) and ((eta<=2.4 and nhf<0.9 and phf<0.9 and elf<0.70 and muf<0.70 and chf>0 and chm>0) or eta>2.4);
        raise RuntimeError, "jetID '%s' not supported" % name

    def looseJetId(self):
        '''PF Jet ID (loose operation point) [method provided for convenience only]'''
        return self.jetID("POG_PFID_Loose")

    def puMva(self):
        return self.userFloat("pileupJetId:fullDiscriminant")

    def puJetId(self):
        '''Full mva PU jet id'''

        puMva = self.puMva()
        wp = loose_53X_WP
        eta = abs(self.eta())
        
        for etamin, etamax, cut in wp:
            if not(eta>=etamin and eta<etamax):
                continue
            return puMva>cut
        
    def rawFactor(self):
        return self.jecFactor('Uncorrected')

    def btag(self,name):
        return self.bDiscriminator(name) 
 
    def btagWP(self,name):
        global _btagWPs
        (disc,val) = _btagWPs[name]
        return self.bDiscriminator(disc) > val
        

class GenJet( PhysicsObject):
    pass

