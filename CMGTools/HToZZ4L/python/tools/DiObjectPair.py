from ROOT import TLorentzVector
from CMGTools.HToZZ4L.tools.DiObject import DiObject
class DiObjectPair( TLorentzVector ):
    '''Class used for A->VV'''
    def __init__(self, leg1, leg2,leg3,leg4):

        a=DiObject(leg1,leg2, doSort=False)
        b=DiObject(leg3,leg4, doSort=False)
        lv=a+b
        super( DiObjectPair, self).__init__(lv)
        self.leg1=a
        self.leg2=b


    def px(self):
         return self.Px()
    def py(self):
         return self.Py()
    def pz(self):
         return self.Pz()
    def energy(self):
         return self.Energy()

    def eta(self):
         return self.Eta()
    def phi(self):
         return self.Phi()

    def pt(self):
         return self.Pt()

    def mass(self):
         return self.M()



    def sortedPtLeg(self,N):
        ''' Gives the Nth highest pt lepton. 0 is the highest'''
        leptons=sorted([self.leg1.leg1, \
                        self.leg1.leg2, \
                        self.leg2.leg1, \
                        self.leg2.leg2], \
                       key=lambda x: x.pt(), \
                       reverse=True)
        return leptons[N]
        

    def charge(self):
        return self.leg1.charge() + self.leg2.charge()

    def PdgId(self):
        '''Dummy, needed to fill the tree'''
        return 25



    def sortedMassPairs(self,onlyOS = False):
        pairs=[DiObject(self.leg1.leg1,self.leg1.leg2),
               DiObject(self.leg2.leg1,self.leg2.leg2),
               DiObject(self.leg1.leg1,self.leg2.leg1),
               DiObject(self.leg1.leg1,self.leg2.leg2),
               DiObject(self.leg1.leg2,self.leg2.leg1),
               DiObject(self.leg1.leg2,self.leg2.leg2)]

        if onlyOS:
            pairs=filter(lambda x: x.charge()==0,pairs)

        pairs=sorted(pairs,key=lambda x: x.mass())
        return pairs

    def minPairMass(self):
        sortedPairs=self.sortedMassPairs()
        return sortedPairs[0].mass()

    def minOSPairMass(self):
        sortedPairs=self.sortedMassPairs(True)
        if len(sortedPairs)>0:
            return sortedPairs[0].mass()
        else:
            return 999.

    def __str__(self):
        return ', '.join( ['DiObjectPair:', str(self.leg1), str(self.leg2)] )


    def fsrUncorrected(self):
        return self.leg1.fsrUncorrected()+self.leg2.fsrUncorrected()

    def hasFSR(self):
        return self.leg1.hasFSR() or self.leg2.hasFSR()


    def updateP4(self):
        
        z1     = TLorentzVector(self.leg1.Px(),self.leg1.Py(),self.leg1.Pz(),self.leg1.Energy())
        z2     = TLorentzVector(self.leg2.Px(),self.leg2.Py(),self.leg2.Pz(),self.leg2.Energy())
        new=z1+z2
        self.SetPxPyPzE(new.Px(),new.Py(),new.Pz(),new.Energy())



    def daughterLeptons(self):
        return [self.leg1.leg1,self.leg1.leg2,self.leg2.leg1,self.leg2.leg2]

    def daughterPhotons(self):
        return self.leg1.daughterPhotons()+self.leg2.daughterPhotons()



        ###MELA#########################################


        ###MELA#########################################

        
