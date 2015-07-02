from ROOT import TLorentzVector
from math import pi,acos,asin
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi

class DiObject( TLorentzVector ):
    '''Class used for Zs, and also for Higgs candidates'''
    def __init__(self, leg1, leg2,doSort = True):
        if (leg2.pt() > leg1.pt()) and doSort:
            leg2, leg1 = leg1, leg2
        lv1 = TLorentzVector( leg1.px(), leg1.py(), leg1.pz(), leg1.energy() )
        lv2 = TLorentzVector( leg2.px(), leg2.py(), leg2.pz(), leg2.energy() )
        lv1 += lv2 
        super( DiObject, self).__init__( lv1 )
        self.leg1 = leg1
        self.leg2 = leg2




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



#    def __getattr__(self, name):
#        '''Trick to preserve the interface in use in CMSSW.'''
#        if name.lower() == 'mass':
#            name = 'M'
        # changing the first letter of the function name to upper case. 
#        capName = ''.join( [name[0].capitalize(), name[1:]] ) 
#        return getattr( self, capName )

    def PdgId(self):
        '''Dummy, needed to fill the tree'''
        return 23

    def Sip3D(self):
        '''Dummy, needed to fill the tree'''
        return -1

    def RelIso(self, dBetaCor):
        '''Sum of the relative isolation (dbeta corrected) of the 2 legs'''
        return self.leg1.relIso( dBetaCor ) + self.leg2.relIso(dBetaCor )

        
    def charge(self):
        return self.leg1.charge() + self.leg2.charge()
        
    def __str__(self):
        return ', '.join( ['DiObject:', str(self.leg1), str(self.leg2)] )

    def daughterLeptons(self):
        return [self.leg1,self.leg2]

    def daughterPhotons(self):
        if hasattr(self,'fsrPhoton'):
            return [self.fsrPhoton]
        else:
            return []
    



############FSR variables
    def fsrUncorrected(self):
        if not hasattr(self,'fsrPhoton'):
            return self
        else:
            gamma = TLorentzVector( self.fsrPhoton.px(), self.fsrPhoton.py(), self.fsrPhoton.pz(), self.fsrPhoton.energy() )
            return self-gamma
    
    def setFSR(self,photon):
        self.fsrPhoton=photon
        gamma = TLorentzVector( photon.px(), photon.py(), photon.pz(), photon.energy() )
        z     = TLorentzVector(self.Px(),self.Py(),self.Pz(),self.Energy())
        new=gamma+z
        self.SetPxPyPzE(new.Px(),new.Py(),new.Pz(),new.Energy())
        

    def hasFSR(self):
        return hasattr(self,'fsrPhoton')

    def fsrTheta1(self):
        if hasattr(self,'fsrPhoton'):
            photon=self.fsrPhoton
            return acos(round(self.leg1.p4().Vect().Dot(photon.p4().Vect())/(self.leg1.p4().P()*photon.p4().P()),5))*180/pi
        else:
            return -99


    def fsrTheta2(self):
        if hasattr(self,'fsrPhoton'):
            photon=self.fsrPhoton
            return acos(round(self.leg2.p4().Vect().Dot(photon.p4().Vect())/(self.leg2.p4().P()*photon.p4().P()),5))*180/pi
        else:
            return -99

    def fsrDR1(self):
        if hasattr(self,'fsrPhoton'):
            photon=self.fsrPhoton
            return deltaR(self.leg1.eta(),self.leg1.phi(),photon.eta(),photon.phi())

    def fsrDR2(self):
        if hasattr(self,'fsrPhoton'):
            photon=self.fsrPhoton
            return deltaR(self.leg2.eta(),self.leg2.phi(),photon.eta(),photon.phi())


    def fsrThetaStar(self):
        if hasattr(self,'fsrPhoton'):
            photon=self.fsrPhoton
            plane = (self.leg1.p4().Vect().Cross(self.leg2.p4().Vect())).unit()
            angle = asin(round(plane.Dot(photon.p4().Vect())/(photon.p4().P()),5))*180/pi
            return abs(angle)

    def fsrDRStar(self):
        if hasattr(self,'fsrPhoton'):
            photon=self.fsrPhoton
            plane = self.leg1.p4().Vect().Cross(self.leg2.p4().Vect()).unit()
            return deltaR(plane.eta(),plane.phi(),photon.eta(),photon.phi())


        

        
