from PhysicsTools.Heppy.physicsobjects.Electron import Electron
from ROOT import gSystem, AutoLibraryLoader
gSystem.Load("libFWCoreFWLite")
AutoLibraryLoader.enable()
gSystem.Load("libDataFormatsRecoCandidate.so")
 
from ROOT import reco

class HTauTauElectron( Electron ):

    def __init__(self, *args, **kwargs):
        super(HTauTauElectron, self).__init__(*args, **kwargs)
        self.photonIsoCache = None
        self.chargedAllIsoCache = None

    # JAN FIXME - replace our old tau-tau-specific vetoes
    # I hope they are not needed anymore!
 
    # def photonIso(self):
    #     if self.photonIsoCache is None:
    #         myVetoes = reco.IsoDeposit.Vetos()
    #         pfGammaIsoType = 6
    #         iso = self.sourcePtr().isoDeposit(pfGammaIsoType).depositWithin(0.4,myVetoes,True)
    #         iso_veto = self.sourcePtr().isoDeposit(pfGammaIsoType).depositWithin(0.08,myVetoes,True)
    #         iso -= iso_veto
    #         self.photonIsoCache = iso
    #     return self.photonIsoCache

    def photonIso(self):
        return super(HTauTauElectron, self).photonIso(0.3)

    # JAN FIXME - replace our old tau-tau-specific vetoes
    # I hope they are not needed anymore!
    # def chargedAllIso(self):
    #     if self.chargedAllIsoCache is None:
    #         # chargedAllIsoType = 13
    #         # myVetoes = reco.IsoDeposit.Vetos()
    #         # iso = self.sourcePtr().isoDeposit(chargedAllIsoType).depositWithin(0.4,
    #         #                                                                    myVetoes,True)
    #         # vetoSize = 0.01
    #         # if self.sourcePtr().isEE():
    #         #     vetoSize = 0.015
    #         # iso_veto = self.sourcePtr().isoDeposit(chargedAllIsoType).depositWithin(vetoSize,
    #         #                                                                         myVetoes,True)
    #         # iso -= iso_veto
    #         # self.chargedAllIsoCache = iso
    #     return self.chargedAllIsoCache

    def chargedAllIso(self):
        return self.physObj.pfIsolationVariables().sumChargedParticlePt 
    
    def relaxedIdForEleTau(self):
        """Relaxing conversion cuts for sideband studies
        """
        eta = abs( self.superCluster().eta() )
        if   eta<0.8:   lmvaID = 0.925
        elif eta<1.479: lmvaID = 0.975
        else :          lmvaID = 0.985
        result = self.mvaNonTrigV0()  > lmvaID
        return result

    def tightIdForEleTau(self):
        """reference numbers from the Htautau twiki

        https://twiki.cern.ch/twiki/bin/view/CMS/HiggsToTauTauWorking2012#2012_Baseline_Selection
        """
        
        # JAN FIXME - do we need this cut?
        # if self.numberOfHits() != 0: return False
        if not self.passConversionVeto(): return False        
        eta = abs( self.superCluster().eta() )

        # Update Jan: Deleted the old numbers which are for e-mu, the ones below
        # are for e-tau starting from pt>20
        if 1:
            if   eta<0.8:   lmvaID = 0.925
            elif eta<1.479: lmvaID = 0.975
            else :          lmvaID = 0.985
        result = self.mvaNonTrigV0()  > lmvaID
        #self.tightIdResult = result
        return result
    

    def looseIdForTriLeptonVeto(self):
        '''To be used in the tri-lepton veto for both the etau and mutau channels.
        Agreed at the CMS center with Josh, Andrew, Valentina, Jose on the 22nd of October
        '''
        # JAN FIXME - do we need this cut?
        # if self.numberOfHits() != 0: return False
        if not self.passConversionVeto(): return False
        eta = abs( self.superCluster().eta() )
        #Colin no eta cut should be done here.
        #        if eta > 2.1 : return False
        lmvaID = -99999 # identification
        if self.pt() < 20 :
            if   eta<0.8:   lmvaID = 0.925
            elif eta<1.479: lmvaID = 0.915
            else :          lmvaID = 0.965
        else:
            if   eta<0.8:   lmvaID = 0.905
            elif eta<1.479: lmvaID = 0.955
            else :          lmvaID = 0.975
        result = self.mvaNonTrigV0()  > lmvaID
        return result
        

    def tightId( self ):
        return self.tightIdForEleTau()
        

    def looseIdForEleTau(self):
        """Loose electron selection, for the lepton veto, 
        according to Phil sync prescription for the sync exercise 18/06/12
        """
        #COLIN inner hits and conversion veto not on the twiki
        # nInnerHits = self.numberOfHits()
        # if nInnerHits != 0 : return False
        # if self.passConversionVeto() == False   : return False
        #COLIN: we might want to keep the vertex constraints separated
        #COLIN: in the twiki there is no cut on dxy
        # if abs(self.dxy())             >= 0.045 : return False
        if abs(self.dz())              >= 0.2   : return False
        # Below, part of WP95 without vertex constraints (applied above)
        hoe = self.hadronicOverEm()
        deta = abs(self.deltaEtaSuperClusterTrackAtVtx())
        dphi = abs(self.deltaPhiSuperClusterTrackAtVtx())
        sihih = self.sigmaIetaIeta()
        # print sihih
        if self.isEB() :
            if sihih >= 0.010     : return False
            if dphi  >= 0.80      : return False 
            if deta  >= 0.007     : return False
            if hoe   >= 0.15      : return False
        elif self.isEE() :
            if sihih >= 0.030     : return False
            if dphi  >= 0.70      : return False 
            if deta  >= 0.010     : return False
    #            if hoe   >= 0.07      : return False
        else : return False #PG is this correct? does this take cracks into consideration?
        return True    

    def __str__(self):
        base = [super(HTauTauElectron, self).__str__()]
        spec = [
            'vertex    : dxy = {dxy}, dz = {dz}'.format(dxy=self.dxy(), dz=self.dz()),
            # 'mva       = {mva}'.format(mva=self.mvaNonTrigV0()),
            # 'nmisshits = {nhits}'.format(nhits=self.numberOfHits()),
            'conv veto = {conv}'.format(conv=self.passConversionVeto()),
            'tight ID  = {id}'.format(id=self.tightId()),
            '3-veto ID = {id}'.format(id=self.looseIdForTriLeptonVeto()),
            '2-veto ID = {id}'.format(id=self.looseIdForEleTau()),
            ]
        return '\n\t'.join( base + spec )
