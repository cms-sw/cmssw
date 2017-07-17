import math
from ROOT import TLorentzVector       
from PhysicsTools.HeppyCore.utils.deltar import deltaPhi, deltaR2

class VBF( object ):
    '''Computes and holds VBF quantities'''
    def __init__(self, jets, diLepton, vbfMvaCalc, cjvPtCut):
        '''jets: jets cleaned from the diLepton legs.
        diLepton: the di-tau, for example. Necessary to compute input variables for MVA selection
        '''
        self.cjvPtCut = cjvPtCut
        self.vbfMvaCalc = vbfMvaCalc
        self.jets = jets
        # the MET is taken from the di-lepton, because it can depend on it
        # e.g. recoil corrections, mva met
        self.met = diLepton.met()
        self.leadJets = jets[:2] # taking first 2 jets (leading ones)
        self.otherJets = jets[2:]
        self.centralJets = self.findCentralJets( self.leadJets, self.otherJets )

        # delta eta
        self.deta = self.leadJets[0].eta() - self.leadJets[1].eta()

        # below, the variables for the MVA selection
        # delta phi
        self.dphi = deltaPhi(self.leadJets[0].phi(), self.leadJets[1].phi())
        dijetp4 = self.leadJets[0].p4() + self.leadJets[1].p4()
        # mass of the di-jet system
        self.mjj = dijetp4.M()
        # pt of di-jet system
        self.dijetpt = dijetp4.pt()
        # phi of di-jet system
        self.dijetphi = dijetp4.phi()
        # higgs momentum (defined as the di-lepton momentum + the met momentum)
        # don't access longitudinal quantities!
        self.higgsp4 = diLepton.p4() + self.met.p4()
        # delta phi between dijet system and higgs system
        self.dphidijethiggs = deltaPhi( self.dijetphi, self.higgsp4.phi() )
        # ? 
        visDiLepton = diLepton.leg1 ().p4 () + diLepton.leg2 ().p4 ()
        self.visjeteta = min (
            abs (self.leadJets[0].eta () - visDiLepton.eta ()), 
            abs (self.leadJets[1].eta () - visDiLepton.eta ()))
        # visible higgs pt = di-lepton pt
        self.ptvis = visDiLepton.pt()
        ## self.ptvis = diLepton.pt()
        # new VBF MVA, based on 4 variables
        if self.vbfMvaCalc is not None:
            self.mva = self.vbfMvaCalc.val( self.mjj,
                                            abs(self.deta),
                                            self.visjeteta,
                                            self.ptvis )
        else:
            self.mva = -99.

#  double mjj      , // the invariant mass of the two tag jets
#  double dEta     , // the pseudorapidity difference between the two tag jets
#  double dPhi     , // the phi difference between the two tag jets
#  double ditau_pt , // the vector sum of the pT of the tau + electron/muon + MET
#  double dijet_pt , // the vector sum of the pT of the two tag jets
#  double dPhi_hj  , // the phi difference between the di-tau vector and the di-jet vector
#  double C1       , // the pseudorapidity difference between the *visible* di-tau vector and the closest tag jet
#  double C2         // the *visible* pT of the di-tau
        
        
    def findCentralJets( self, leadJets, otherJets ):
        '''Finds all jets between the 2 leading jets, for central jet veto.'''
        if not len(otherJets):
            return []
        etamin = leadJets[0].eta()
        etamax = leadJets[1].eta()
        if etamin > etamax:
            etamin, etamax = etamax, etamin
        def isCentral( jet ):
            if jet.pt()<self.cjvPtCut:
                return False
            eta = jet.eta()
            if etamin < eta and eta < etamax:
                return True
            else:
                return False
        centralJets = filter( isCentral, otherJets )
        return centralJets

    def calcP4(self, jets):
        '''returns the sum p4 of a collection of objects.
        FIXME: remove this function, which is a bit stupid
        '''
        p4 = TLorentzVector()
        for jet in jets:
            p4 += TLorentzVector(jet.px(), jet.py(), jet.pz(), jet.energy())
        return p4

    def __str__(self):
        header = 'VBF : deta={deta:4.2f}, Mjj={mjj:4.2f}, #centjets={ncentjets}'
        header = header.format( deta=self.deta, mjj=self.mjj, ncentjets=len(self.centralJets))
        leadJets = map( str, self.leadJets )
        centralJets = map( str, self.centralJets)
        tmp = [header]
        tmp.append('MVA input variables: dphi={dphi:4.2f}, dijetpt={dijetpt:4.2f}, dijetphi={dijetphi:4.2f}, dphidijethiggs={dphidijethiggs:4.2f}, visjeteta={visjeteta:4.2f}, ptvis={ptvis:4.2f}'.format(
            dphi = self.dphi,
            dijetpt = self.dijetpt,
            dijetphi = self.dijetphi,
            dphidijethiggs = self.dphidijethiggs,
            visjeteta = self.visjeteta,
            ptvis = self.ptvis
            ))
        tmp.append('Leading Jets:')
        tmp.extend( leadJets )
        tmp.append('Central Jets:')
        tmp.extend( centralJets )
        return '\n'.join( tmp )
