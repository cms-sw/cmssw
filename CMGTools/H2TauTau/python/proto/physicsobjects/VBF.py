import math
from ROOT import TLorentzVector       

class VBF( object ):

    def __init__(self, jets ):
        self.jets = jets 
        self.leadJets = jets[:2] # taking first 2 jets (leading ones)
        self.otherJets = jets[2:]
        self.centralJets = self.findCentralJets( self.leadJets, self.otherJets )
        self.mjj = self.calcP4( self.leadJets ).M()
        self.deta = self.leadJets[0].eta() - self.leadJets[1].eta()

    def findCentralJets( self, leadJets, otherJets ):
        etamin = leadJets[0].eta()
        etamax = leadJets[1].eta()
        if etamin > etamax:
            etamin, etamax = etamax, etamin
        def isCentral( jet ):
            #COLIN: shouln't I take a margin? 
            eta = jet.eta()
            if etamin < eta and eta < etamax:
                return True
            else:
                return False
        centralJets = filter( isCentral, otherJets )
        return centralJets

    def calcP4(self, jets):
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
        tmp.append('Leading Jets:')
        tmp.extend( leadJets )
        tmp.append('Central Jets:')
        tmp.extend( centralJets )
        return '\n'.join( tmp )
