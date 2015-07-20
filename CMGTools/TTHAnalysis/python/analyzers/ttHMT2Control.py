import operator
import itertools
import copy
import types

from math import *

from ROOT import TLorentzVector

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle

from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi

def mtw(x1,x2):
    return sqrt(2*x1.pt()*x2.pt()*(1-cos(x1.phi()-x2.phi())))

class ttHMT2Control( Analyzer ):


    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHMT2Control,self).__init__(cfg_ana,cfg_comp,looperName)
        self.jetPt = cfg_ana.jetPt

    def makeMT(self, event):
        #        print '==> INSIDE THE PRINT MT'
        #        print 'MET=',event.met.pt()
        
        if len(event.selectedLeptons)>0:
            for lepton in event.selectedLeptons:
                event.mtw = mtw(lepton, event.met)
                    
        if len(event.selectedTaus)>0:
            for myTau in event.selectedTaus:
                event.mtwTau = mtw(myTau, event.met)
                foundTau = True

        if hasattr(event, 'selectedIsoCleanTrack'):
            if len(event.selectedIsoTrack)>0:
                for myTrack in event.selectedIsoTrack:
                    event.mtwIsoTrack = mtw(myTrack, event.met)      
##number of PF leptons (e,mu) with pt > 5, reliso < 0.2, MT < 100 
#number of PF hadrons with pt > 10, reliso < 0.1, MT < 100                        
                    if event.mtwIsoTrack < 100:
                        if abs(myTrack.pdgId()) == 11 or abs(myTrack.pdgId()) == 13:
                            if myTrack.pt()>5 and myTrack.absIso/myTrack.pt()<0.2:
                                event.nPFLep5LowMT += 1                           
                        if abs(myTrack.pdgId()) == 211:
                            if myTrack.pt()>10 and myTrack.absIso/myTrack.pt()<0.1:
                                event.nPFHad10LowMT += 1                            

        leptons = []
        if hasattr(event, 'selectedLeptons'):
            leptons = [ l for l in event.selectedLeptons if l.pt() > 10]
        if hasattr(event, 'selectedIsoCleanTrack'):
            leptons = leptons[:] + event.selectedIsoCleanTrack
        if len(leptons)>0:
            for lepton in leptons:
                mtwLep = mtw(lepton, event.met)
                if mtwLep < 100:
                    event.nLepLowMT +=1

    def makeGammaObjects(self, event):

        import ROOT

        ## with Central Jets                                                                                                                                                                                                      
        gamma_objects25 = [ j for j in event.gamma_cleanJets if j.pt() > 25 ] + event.selectedLeptons
        gamma_objects30 = [ j for j in event.gamma_cleanJets if j.pt() > 30 ] + event.selectedLeptons
        gamma_objects40  = [ j for j in event.gamma_cleanJets if j.pt() > 40 and abs(j.eta()) < 2.5 ] + [ l for l in event.selectedLeptons if l.pt() > 10 and abs(l.eta()) < 2.5 ]
        gamma_objectsX  = [ j for j in event.gamma_cleanJets if j.pt() > self.jetPt and abs(j.eta()) < 2.5 ] + [ l for l in event.selectedLeptons if l.pt() > 10 and abs(l.eta()) < 2.5 ]
        # for the gamma + jets variables we use do not take care about the leptons, we consider jets that do not overlap with the first jets
        gamma_objects40j = [ j for j in event.gamma_cleanJets if j.pt() > 40 and abs(j.eta()) < 2.5 ]
        gamma_objects40ja = [ j for j in event.gamma_cleanJetsAll if j.pt() > 40 ]
        gamma_objectsXj = [ j for j in event.gamma_cleanJets if j.pt() > self.jetPt and abs(j.eta()) < 2.5 ]
        gamma_objectsXja = [ j for j in event.gamma_cleanJetsAll if j.pt() > self.jetPt ]
       
        event.gamma_htJet25 = sum([x.pt() for x in gamma_objects25])
        event.gamma_mhtJet25vec = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in gamma_objects25])) , -1.*(sum([x.py() for x in gamma_objects25])), 0, 0 )
        event.gamma_mhtPhiJet25 = event.mhtJet25vec.phi()
        event.gamma_mhtJet25 = event.mhtJet25vec.pt()

        event.gamma_htJet30 = sum([x.pt() for x in gamma_objects30])
        event.gamma_mhtJet30vec = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in gamma_objects30])) , -1.*(sum([x.py() for x in gamma_objects30])), 0, 0 )
        event.gamma_mhtJet30 = event.mhtJet30vec.pt()
        event.gamma_mhtPhiJet30 = event.mhtJet30vec.phi()

        event.gamma_htJet40 = sum([x.pt() for x in gamma_objects40])
        event.gamma_mhtJet40vec = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in gamma_objects40])) , -1.*(sum([x.py() for x in gamma_objects40])), 0, 0 )
        event.gamma_mhtJet40 = event.gamma_mhtJet40vec.pt()
        event.gamma_mhtPhiJet40 = event.gamma_mhtJet40vec.phi()

        event.gamma_htJetX = sum([x.pt() for x in gamma_objectsX])
        event.gamma_mhtJetXvec = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in gamma_objectsX])) , -1.*(sum([x.py() for x in gamma_objectsX])), 0, 0 )
        event.gamma_mhtJetX = event.gamma_mhtJetXvec.pt()
        event.gamma_mhtPhiJetX = event.gamma_mhtJetXvec.phi()

        event.gamma_htJet40j = sum([x.pt() for x in gamma_objects40j])
        event.gamma_mhtJet40jvec = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in gamma_objects40j])) , -1.*(sum([x.py() for x in gamma_objects40j])), 0, 0 )
        event.gamma_mhtJet40j = event.gamma_mhtJet40jvec.pt()
        event.gamma_mhtPhiJet40j = event.gamma_mhtJet40jvec.phi()

        event.gamma_htJetXj = sum([x.pt() for x in gamma_objectsXj])
        event.gamma_mhtJetXjvec = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in gamma_objectsXj])) , -1.*(sum([x.py() for x in gamma_objectsXj])), 0, 0 )
        event.gamma_mhtJetXj = event.gamma_mhtJetXjvec.pt()
        event.gamma_mhtPhiJetXj = event.gamma_mhtJetXjvec.phi()

        event.gamma_htJet40ja = sum([x.pt() for x in gamma_objects40ja])
        event.gamma_mhtJet40javec = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in gamma_objects40ja])) , -1.*(sum([x.py() for x in gamma_objects40ja])), 0, 0 )
        event.gamma_mhtJet40ja = event.gamma_mhtJet40javec.pt()
        event.gamma_mhtPhiJet40ja = event.gamma_mhtJet40javec.phi()

        event.gamma_htJetXja = sum([x.pt() for x in gamma_objectsXja])
        event.gamma_mhtJetXjavec = ROOT.reco.Particle.LorentzVector(-1.*(sum([x.px() for x in gamma_objectsXja])) , -1.*(sum([x.py() for x in gamma_objectsXja])), 0, 0 )
        event.gamma_mhtJetXja = event.gamma_mhtJetXjavec.pt()
        event.gamma_mhtPhiJetXja = event.gamma_mhtJetXjavec.phi()


         # MET + photon                                                                                                                                                                                                           
        event.gamma_met = ROOT.reco.Particle.LorentzVector( event.met.px(), event.met.py(), 0, 0 )
        event.gamma_metNoPU = ROOT.reco.Particle.LorentzVector( event.metNoPU.px(), event.metNoPU.py(), 0, 0 )
        for gamma in event.selectedPhotons:
            event.gamma_met = ROOT.reco.Particle.LorentzVector( event.gamma_met.px() + gamma.px(), event.gamma_met.py() + gamma.py() , 0, 0 )
            event.gamma_metNoPU = ROOT.reco.Particle.LorentzVector( event.gamma_metNoPU.px() + gamma.px(), event.gamma_metNoPU.py() + gamma.py() , 0, 0 )
            break # only lead photon

        # look for minimal deltaPhi between MET and four leading jets with pt>40 and eta<2.4                                                                                                                                      
        event.gamma_deltaPhiMin_had = 999.
        for n,j in enumerate(gamma_objects40ja):
            if n>3:  break
            thisDeltaPhi = abs( deltaPhi( j.phi(), event.gamma_met.phi() ) )
            if thisDeltaPhi < event.gamma_deltaPhiMin_had : event.gamma_deltaPhiMin_had = thisDeltaPhi

        event.gamma_deltaPhiMin_Xj_had = 999.
        for n,j in enumerate(gamma_objectsXja):
            if n>3:  break
            thisDeltaPhi = abs( deltaPhi( j.phi(), event.gamma_met.phi() ) )
            if thisDeltaPhi < event.gamma_deltaPhiMin_Xj_had : event.gamma_deltaPhiMin_Xj_had = thisDeltaPhi


        # absolute value of the vectorial difference between met and mht                                                                                                                                                          
        gamma_diffMetMht_had_vec = ROOT.reco.Particle.LorentzVector(event.gamma_mhtJet40jvec.px()-event.gamma_met.px(), event.gamma_mhtJet40jvec.py()-event.gamma_met.py(), 0, 0 )
        event.gamma_diffMetMht_had = sqrt( gamma_diffMetMht_had_vec.px()*gamma_diffMetMht_had_vec.px() + gamma_diffMetMht_had_vec.py()*gamma_diffMetMht_had_vec.py() )

        gamma_diffMetMht_vec = ROOT.reco.Particle.LorentzVector(event.gamma_mhtJet40vec.px()-event.gamma_met.px(), event.gamma_mhtJet40vec.py()-event.gamma_met.py(), 0, 0 )
        event.gamma_diffMetMht = sqrt( gamma_diffMetMht_vec.px()*gamma_diffMetMht_vec.px() + gamma_diffMetMht_vec.py()*gamma_diffMetMht_vec.py() )

        gamma_diffMetMht_Xj_had_vec = ROOT.reco.Particle.LorentzVector(event.gamma_mhtJetXjvec.px()-event.gamma_met.px(), event.gamma_mhtJetXjvec.py()-event.gamma_met.py(), 0, 0 )
        event.gamma_diffMetMht_Xj_had = sqrt( gamma_diffMetMht_Xj_had_vec.px()*gamma_diffMetMht_Xj_had_vec.px() + gamma_diffMetMht_Xj_had_vec.py()*gamma_diffMetMht_Xj_had_vec.py() )

        gamma_diffMetMht_Xj_vec = ROOT.reco.Particle.LorentzVector(event.gamma_mhtJetXvec.px()-event.gamma_met.px(), event.gamma_mhtJetXvec.py()-event.gamma_met.py(), 0, 0 )
        event.gamma_diffMetMht_Xj = sqrt( gamma_diffMetMht_Xj_vec.px()*gamma_diffMetMht_Xj_vec.px() + gamma_diffMetMht_Xj_vec.py()*gamma_diffMetMht_Xj_vec.py() )



    def makeZllObjects(self, event):

        import ROOT

        vetoLeptons = [ l for l in event.selectedLeptons if l.pt() > 10 and abs(l.eta()) < 2.5 ]

        # MET + zll                                                                                                                                                                                                               
        event.zll_ht = -999.
        event.zll_deltaPhiMin = -999.
        event.zll_met_pt = -999.
        event.zll_met_phi = -999.
        event.zll_diffMetMht = -999.
        event.zll_mhtJet40j = -999.
        event.zll_mhtPhiJet40j = -999.
        event.zll_p4 = ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 )
                      
        event.zll_ht_Xj = -999.
        event.zll_deltaPhiMin_Xj = -999.
        event.zll_diffMetMht_Xj = -999.
        event.zll_mhtJetXj = -999.
        event.zll_mhtPhiJetXj = -999.

        if len(vetoLeptons)==2:
            event.zll_met = ROOT.reco.Particle.LorentzVector( event.met.px(), event.met.py(), 0, 0 )
            for l in vetoLeptons:
                event.zll_met = ROOT.reco.Particle.LorentzVector( event.zll_met.px() + l.px(), event.zll_met.py() + l.py() , 0, 0 )

            event.zll_met_pt = event.zll_met.pt()
            event.zll_met_phi = event.zll_met.phi()

            # defining mht as hadronic mht                                                                                                                                                                                        
            event.zll_mhtJet40j = event.mhtJet40j
            event.zll_mhtPhiJet40j = event.mhtPhiJet40j

            event.zll_mhtJetXj = event.mhtJetXj
            event.zll_mhtPhiJetXj = event.mhtPhiJetXj

            # look for minimal deltaPhi between MET and four leading jets with pt>40 and |eta|<2.4                                                                                                                                
            event.zll_deltaPhiMin = 999.
            objects40jc = [ j for j in event.cleanJets if j.pt() > 40 and abs(j.eta())<2.5 ]
            objects40ja = [ j for j in event.cleanJets if j.pt() > 40]
            event.zll_ht = sum([x.pt() for x in objects40jc])
            for n,j in enumerate(objects40ja):
                if n>3:  break
                thisDeltaPhi = abs( deltaPhi( j.phi(), event.zll_met.phi() ) )
                if thisDeltaPhi < event.zll_deltaPhiMin : event.zll_deltaPhiMin = thisDeltaPhi

            event.zll_deltaPhiMin_Xj = 999.
            objectsXjc = [ j for j in event.cleanJets if j.pt() > self.jetPt and abs(j.eta())<2.5 ]
            objectsXja = [ j for j in event.cleanJets if j.pt() > self.jetPt]
            event.zll_ht_Xj = sum([x.pt() for x in objectsXjc])
            for n,j in enumerate(objectsXja):
                if n>3:  break
                thisDeltaPhi = abs( deltaPhi( j.phi(), event.zll_met.phi() ) )
                if thisDeltaPhi < event.zll_deltaPhiMin_Xj : event.zll_deltaPhiMin_Xj = thisDeltaPhi

            # absolute value of the vectorial difference between met and mht                                                                                                                                                      
            zll_diffMetMht_vec = ROOT.reco.Particle.LorentzVector(event.mhtJet40jvec.px()-event.zll_met.px(), event.mhtJet40jvec.py()-event.zll_met.py(), 0, 0 )
            event.zll_diffMetMht = sqrt( zll_diffMetMht_vec.px()*zll_diffMetMht_vec.px() + zll_diffMetMht_vec.py()*zll_diffMetMht_vec.py() )

            zll_diffMetMht_Xj_vec = ROOT.reco.Particle.LorentzVector(event.mhtJetXjvec.px()-event.zll_met.px(), event.mhtJetXjvec.py()-event.zll_met.py(), 0, 0 )
            event.zll_diffMetMht_Xj = sqrt( zll_diffMetMht_Xj_vec.px()*zll_diffMetMht_Xj_vec.px() + zll_diffMetMht_Xj_vec.py()*zll_diffMetMht_Xj_vec.py() )

            # di-lepton invariant mass                                                                                                                                                                                            
            for l in vetoLeptons:
                event.zll_p4 += l.p4()

            ##event.zll_invmass = zll_p4.M()

    def process(self, event):

        self.readCollections( event.input )
        self.makeGammaObjects(event)                                                                                                                                                                                             
        self.makeZllObjects(event)

        event.nLepLowMT =0
        event.nPFLep5LowMT = 0
        event.nPFHad10LowMT = 0
        event.mtw=-999 
        event.mtwTau=-999
        event.mtwIsoTrack=-999               


        self.makeMT(event)

        return True
