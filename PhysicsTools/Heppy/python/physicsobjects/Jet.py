from PhysicsTools.Heppy.physicsobjects.PhysicsObject import *
from PhysicsTools.HeppyCore.utils.deltar import deltaPhi
import math

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
###https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation74X50ns
    "CMVAL": ("pfCombinedMVABJetTags", 0.630), # for same b-jet efficiency of CSVv2IVFL on ttbar MC, jet pt > 30
    "CMVAM": ("pfCombinedMVABJetTags", 0.732), # for same b-jet efficiency of CSVv2IVFM on ttbar MC, jet pt > 30
    "CMVAT": ("pfCombinedMVABJetTags", 0.813), # for same b-jet efficiency of CSVv2IVFT on ttbar MC, jet pt > 30
    "CMVAv2M": ("pfCombinedMVAV2BJetTags", 0.185), # for same b-jet efficiency of CSVv2IVFM on ttbar MC, jet pt > 30
###https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation80X
    "CSVv2IVFL": ("pfCombinedInclusiveSecondaryVertexV2BJetTags", 0.460),
    "CSVv2IVFM": ("pfCombinedInclusiveSecondaryVertexV2BJetTags", 0.800),
    "CSVv2IVFT": ("pfCombinedInclusiveSecondaryVertexV2BJetTags", 0.935),
    "CMVAv2L": ("pfCombinedMVAV2BJetTags", -0.715), # for same b-jet efficiency of CSVv2IVFL on ttbar MC, jet pt > 30
    "CMVAv2M": ("pfCombinedMVAV2BJetTags", 0.185),  # for same b-jet efficiency of CSVv2IVFM on ttbar MC, jet pt > 30
    "CMVAv2T": ("pfCombinedMVAV2BJetTags", 0.875),  # for same b-jet efficiency of CSVv2IVFT on ttbar MC, jet pt > 30

}

class Jet(PhysicsObject):   
    def __init__(self, *args, **kwargs):
        super(Jet, self).__init__(*args, **kwargs)
        self._physObjInit()

    def _physObjInit(self):
        self._rawFactorMultiplier = 1.0
        self._recalibrated = False
        self._leadingTrack = None
        self._leadingTrackSearched = False

    def rawEnergy(self):
        return self.energy() * self.rawFactor()

    # these energy fraction methods need to be redefined here 
    # because the pat::Jet's currentJECLevel data member cannot be update easily by the calibrator 
    # and then the C++ methods would be broken when new jet corrections are applied in Heppy
    def chargedEmEnergyFraction(self):
        return self.chargedEmEnergy()/self.rawEnergy()

    def chargedHadronEnergyFraction(self):
        return self.chargedHadronEnergy()/self.rawEnergy()

    def chargedMuEnergyFraction(self):
        return self.chargedMuEnergy()/self.rawEnergy()

    def electronEnergyFraction(self):
        return self.electronEnergy()/self.rawEnergy()

    def muonEnergyFraction(self):
        return self.muonEnergy()/self.rawEnergy()

    def neutralEmEnergyFraction(self):
        return self.neutralEmEnergy()/self.rawEnergy()

    def neutralHadronEnergyFraction(self):
        return self.neutralHadronEnergy()/self.rawEnergy()

    def photonEnergyFraction(self):
        return self.photonEnergy()/self.rawEnergy()


    def HFHadronEnergyFraction(self):
        return self.HFHadronEnergy()/self.rawEnergy()

    def HFEMEnergyFraction(self):
        return self.HFEMEnergy()/self.rawEnergy()

    def hoEnergyFraction(self):
        return self.hoEnergy()/self.rawEnergy()



    def jetID(self,name=""):
        if not self.isPFJet():
            raise RuntimeError("jetID implemented only for PF Jets")
        eta = abs(self.eta());
        energy = self.rawEnergy();
        chf = self.chargedHadronEnergy()/energy;
        nhf = self.neutralHadronEnergy()/energy;
        phf = self.neutralEmEnergy()/energy;
        muf = self.muonEnergy()/energy;
        elf = self.chargedEmEnergy()/energy;
        chm = self.chargedHadronMultiplicity();
        npr = self.chargedMultiplicity() + self.neutralMultiplicity();
        npn = self.neutralMultiplicity();
        #if npr != self.nConstituents():
        #    import pdb; pdb.set_trace()
        if name == "POG_PFID":  

            if   self.jetID("PAG_monoID_Tight") and self.jetID("POG_PFID_Tight") : return 5;
            if   self.jetID("PAG_monoID_Loose") and self.jetID("POG_PFID_Tight") : return 4;
            if   self.jetID("POG_PFID_Tight")  : return 3;
            #elif self.jetID("POG_PFID_Medium") : return 2;  commented this line because this working point doesn't exist anymore (as 12/05/15)
            elif self.jetID("POG_PFID_Loose")  : return 1;
            else                               : return 0;
        
        # jetID from here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID#Recommendations_for_13_TeV_data
        if name == "POG_PFID_Loose":    return ((eta<3.0 and ((npr>1 and phf<0.99 and nhf<0.99) and (eta>2.4 or (elf<0.99 and chf>0 and chm>0)))) or (eta>3.0 and (phf<0.90 and npn>10)));
        if name == "POG_PFID_Medium":   return (npr>1 and phf<0.95 and nhf<0.95 and muf < 0.8) and (eta>2.4 or (elf<0.99 and chf>0 and chm>0));
        if name == "POG_PFID_Tight":    return ((eta<3.0 and ((npr>1 and phf<0.90 and nhf<0.90) and (eta>2.4 or (elf<0.99 and chf>0 and chm>0)))) or (eta>3.0 and (phf<0.90 and npn>10)));
        if name == "VBFHBB_PFID_Loose":  return (npr>1 and phf<0.99 and nhf<0.99);
        if name == "VBFHBB_PFID_Medium": return (npr>1 and phf<0.99 and nhf<0.99) and ((eta<=2.4 and nhf<0.9 and phf<0.9 and elf<0.99 and muf<0.99 and chf>0 and chm>0) or eta>2.4);
        if name == "VBFHBB_PFID_Tight":  return (npr>1 and phf<0.99 and nhf<0.99) and ((eta<=2.4 and nhf<0.9 and phf<0.9 and elf<0.70 and muf<0.70 and chf>0 and chm>0) or eta>2.4);
        if name == "PAG_monoID_Loose":    return (eta<3.0 and chf>0.05 and nhf<0.7 and phf<0.8);
        if name == "PAG_monoID_Tight":    return (eta<3.0 and chf>0.2 and nhf<0.7 and phf<0.7);

        raise RuntimeError, "jetID '%s' not supported" % name

    def looseJetId(self):
        '''PF Jet ID (loose operation point) [method provided for convenience only]'''
        return self.jetID("POG_PFID_Loose")

    def puMva(self, label="pileupJetId:fullDiscriminant"):
        if self.hasUserFloat(label):
            return self.userFloat(label)
        return -99

    def puJetId(self, label="pileupJetId:fullDiscriminant"):
        '''Full mva PU jet id'''

        puMva = self.puMva(label)
        wp = loose_53X_WP
        eta = abs(self.eta())
        
        for etamin, etamax, cut in wp:
            if not(eta>=etamin and eta<etamax):
                continue
            return puMva>cut
        return -99
        
    def rawFactor(self):
        return self.jecFactor('Uncorrected') * self._rawFactorMultiplier

    def setRawFactor(self, factor):
        self._rawFactorMultiplier = factor/self.jecFactor('Uncorrected')

    def corrFactor(self):
        return 1.0/self.rawFactor()

    def setCorrP4(self,newP4):
        self._recalibrated = True
        corr = newP4.Pt() / (self.pt() * self.rawFactor()) 
        self.setP4(newP4);
        self.setRawFactor(1.0/corr);

    def l1corrFactor(self):
        if hasattr(self, 'CorrFactor_L1'):
            return self.CorrFactor_L1
        if self._recalibrated:
            raise RuntimeError("The jet was recalibrated, but without calculateSeparateCorrections. L1 is not available")
        jecLevels = self.physObj.availableJECLevels()
        for level in jecLevels:
            if "L1" in level:
                return self.physObj.jecFactor(level)/self.physObj.jecFactor('Uncorrected')
        return 1.0 # Jet does not have any L1 correction

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
    def qgl(self) :
       if not hasattr(self,"qgl_value") :
	  if hasattr(self,"qgl_rho") : #check if qgl calculator is configured
              self.computeQGvars()
              self.qgl_value=self.qgl_calc(self,self.qgl_rho)
	  else :
              self.qgl_value=-1. #if no qgl calculator configured
		  
       return self.qgl_value

    def computeQGvars(self):
       #return immediately if qgvars already computed or if qgl is disabled
       if not hasattr(self,"qgl_rho") or getattr(self,"hasQGVvars",False) :
	  return self
       self.hasQGvars = True
	 
       jet = self
       jet.mult = 0
       sum_weight = 0.
       sum_pt = 0.    
       sum_deta = 0.  
       sum_dphi = 0.  
       sum_deta2 = 0. 
       sum_detadphi = 0.
       sum_dphi2 = 0.   



       for ii in range(0, jet.numberOfDaughters()) :

         part = jet.daughter(ii)

         if part.charge() == 0 : # neutral particles 

           if part.pt() < 1.: continue

         else : # charged particles

           if part.trackHighPurity()==False: continue
           if part.fromPV()<=1: continue             


         jet.mult += 1

         deta = part.eta() - jet.eta()
         dphi = deltaPhi(part.phi(), jet.phi())
         partPt = part.pt()                    
         weight = partPt*partPt                
         sum_weight += weight                  
         sum_pt += partPt                      
         sum_deta += deta*weight               
         sum_dphi += dphi*weight               
         sum_deta2 += deta*deta*weight         
         sum_detadphi += deta*dphi*weight      
         sum_dphi2 += dphi*dphi*weight         




       a = 0.
       b = 0.
       c = 0.

       if sum_weight > 0 :
         jet.ptd = math.sqrt(sum_weight)/sum_pt
         ave_deta = sum_deta/sum_weight        
         ave_dphi = sum_dphi/sum_weight        
         ave_deta2 = sum_deta2/sum_weight      
         ave_dphi2 = sum_dphi2/sum_weight      
         a = ave_deta2 - ave_deta*ave_deta     
         b = ave_dphi2 - ave_dphi*ave_dphi     
         c = -(sum_detadphi/sum_weight - ave_deta*ave_dphi)
       else: jet.ptd = 0.                                  

       delta = math.sqrt(math.fabs((a-b)*(a-b)+4.*c*c))

       if a+b-delta > 0: jet.axis2 = -math.log(math.sqrt(0.5*(a+b-delta)))
       else: jet.axis2 = -1.                                              
       return jet	
   


class GenJet( PhysicsObject):
    pass

