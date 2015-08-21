import operator
import itertools
import copy
import types

from ROOT import TLorentzVector

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.Lepton import Lepton
from PhysicsTools.Heppy.physicsobjects.Tau import Tau
from PhysicsTools.Heppy.physicsobjects.IsoTrack import IsoTrack

from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi, bestMatch , matchObjectCollection3

import PhysicsTools.HeppyCore.framework.config as cfg

from ROOT import heppy




def mtw(x1,x2):
    import math
    return math.sqrt(2*x1.pt()*x2.pt()*(1-math.cos(x1.phi()-x2.phi())))

def makeNearestLeptons(leptons,track, event):

    minDeltaR = 99999
    
    nearestLepton = []
    ibest=-1
    for i,lepton in enumerate(leptons):
        minDeltaRtemp=deltaR(lepton.eta(),lepton.phi(),track.eta(),track.phi())
        if minDeltaRtemp < minDeltaR:
            minDeltaR = minDeltaRtemp
            ibest=i

    if len(leptons) > 0 and ibest!=-1:
        nearestLepton.append(leptons[ibest])

    return nearestLepton
 
class IsoTrackAnalyzer( Analyzer ):

    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(IsoTrackAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        self.IsoTrackIsolationComputer = heppy.IsolationComputer(self.cfg_ana.isoDR)

        self.doIsoAnulus = getattr(cfg_ana, 'doIsoAnulus', False)
        if self.doIsoAnulus:
            self.isoAnPUCorr = self.cfg_ana.isoAnPUCorr
            self.anDeltaR = self.cfg_ana.anDeltaR
            self.IsoTrackIsolationComputer = heppy.IsolationComputer()



    #----------------------------------------
    # DECLARATION OF HANDLES OF LEPTONS STUFF   
    #----------------------------------------
    def declareHandles(self):
        super(IsoTrackAnalyzer, self).declareHandles()
        self.handles['met'] = AutoHandle( 'slimmedMETs', 'std::vector<pat::MET>' )
        self.handles['packedCandidates'] = AutoHandle( 'packedPFCandidates', 'std::vector<pat::PackedCandidate>')

    def beginLoop(self, setup):
        super(IsoTrackAnalyzer,self).beginLoop(setup)
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')
        count.register('has >=1 selected Track')
        count.register('has >=1 selected Iso Track')

    #------------------
    # MAKE LIST
    #------------------
    def makeIsoTrack(self, event):

        event.selectedIsoTrack = []
        event.selectedIsoCleanTrack = []
        #event.preIsoTrack = []

        patcands = self.handles['packedCandidates'].product()

        charged = [ p for p in patcands if ( p.charge() != 0 and abs(p.dz())<=self.cfg_ana.dzMax ) ]

        self.IsoTrackIsolationComputer.setPackedCandidates(patcands, -1, self.cfg_ana.dzPartMax, 9999., True)

        alltrack = map( IsoTrack, charged )


        for track in alltrack:

            if ( (abs(track.pdgId())!=11) and (abs(track.pdgId())!=13) and (track.pt() < self.cfg_ana.ptMin) ): continue
            if ( track.pt() < self.cfg_ana.ptMinEMU ): continue

            foundNonIsoTrack = False

## ===> require is not the leading lepton and opposite to the leading lepton 
            if( (self.cfg_ana.doSecondVeto) and len(event.selectedLeptons)>0) : 
               if( deltaR(event.selectedLeptons[0].eta(), event.selectedLeptons[0].phi(), track.eta(), track.phi()) <0.01) : continue
               if ( (abs(track.pdgId())!=11) and (abs(track.pdgId())!=13) and (track.charge()*event.selectedLeptons[0].charge()) ): continue


## ===> Redundant:: require the Track Candidate with a  minimum dz
            track.associatedVertex = event.goodVertices[0] if len(event.goodVertices)>0 else event.vertices[0]

## ===> compute the isolation and find the most isolated track

            isoSum = self.IsoTrackIsolationComputer.chargedAbsIso(track.physObj, self.cfg_ana.isoDR, 0., self.cfg_ana.ptPartMin)
            
            if self.cfg_ana.doRelIsolation:
                relIso = (isoSum-track.pt())/track.pt()
                if ( (abs(track.pdgId())!=11) and (abs(track.pdgId())!=13) and (relIso > self.cfg_ana.MaxIsoSum) ): continue
                elif((relIso > self.cfg_ana.MaxIsoSumEMU)): continue
            else:
                if(isoSum > (self.cfg_ana.maxAbsIso + track.pt())): continue

            if self.doIsoAnulus:
                self.attachIsoAnulus(track)


            #if abs(track.pdgId())==211 :
            track.absIso = isoSum - track.pt() 

            #### store a preIso track
            #event.preIsoTrack.append(track)
            
#            if (isoSum < minIsoSum ) :
            if self.cfg_ana.doRelIsolation or (track.absIso < min(0.2*track.pt(), self.cfg_ana.maxAbsIso)): 
                event.selectedIsoTrack.append(track)

                if self.cfg_ana.doPrune:
                    myMet = self.handles['met'].product()[0]
                    mtwIsoTrack = mtw(track, myMet)
                    if mtwIsoTrack < 100:
                        if abs(track.pdgId()) == 11 or abs(track.pdgId()) == 13:
                            if track.pt()>5 and track.absIso/track.pt()<0.2:

                                myLeptons = [ l for l in event.selectedLeptons if l.pt() > 10 ] 
                                nearestSelectedLeptons = makeNearestLeptons(myLeptons,track, event)
                                if len(nearestSelectedLeptons) > 0:
                                    for lep in nearestSelectedLeptons:
                                        if deltaR(lep.eta(), lep.phi(), track.eta(), track.phi()) > 0.1:
                                            event.selectedIsoCleanTrack.append(track)
                                else: 
                                    event.selectedIsoCleanTrack.append(track)



##        alltrack = map( IsoTrack, charged )

##        for track in alltrack:
##
##            foundNonIsoTrack = False
##
#### ===> require Track Candidate above some pt and charged
##            if ( (abs(track.pdgId())!=11) and (abs(track.pdgId())!=13) and (track.pt() < self.cfg_ana.ptMin) ): continue
##            if ( track.pt() < self.cfg_ana.ptMinEMU ): continue
##
##
#### ===> require is not the leading lepton and opposite to the leading lepton 
##            if( (self.cfg_ana.doSecondVeto) and len(event.selectedLeptons)>0) : 
##               if( deltaR(event.selectedLeptons[0].eta(), event.selectedLeptons[0].phi(), track.eta(), track.phi()) <0.01) : continue
##               if ( (abs(track.pdgId())!=11) and (abs(track.pdgId())!=13) and (track.charge()*event.selectedLeptons[0].charge()) ): continue
##
#### ===> Redundant:: require the Track Candidate with a  minimum dz
##            track.associatedVertex = event.goodVertices[0]
##
#### ===> compute the isolation and find the most isolated track
##
##            othertracks = [ p for p in charged if( deltaR(p.eta(), p.phi(), track.eta(), track.phi()) < self.cfg_ana.isoDR and p.pt()>self.cfg_ana.ptPartMin ) ]
##            #othertracks = alltrack
##
##            isoSum=0
##            for part in othertracks:
##                #### ===> skip pfcands with a pt min (this should be 0)
##                #if part.pt()<self.cfg_ana.ptPartMin : continue
##                #### ===> skip pfcands outside the cone (this should be 0.3)
##                #if deltaR(part.eta(), part.phi(), track.eta(), track.phi()) > self.cfg_ana.isoDR : continue
##                isoSum += part.pt()
##                ### break the loop to save time
##                if(isoSum > (self.cfg_ana.maxAbsIso + track.pt())):
##                    foundNonIsoTrack = True
##                    break
##
##            if foundNonIsoTrack: continue
##
##               ## reset
##               #isoSum=0
##               #for part in othertracks :
##               #### ===> skip pfcands with a pt min (this should be 0)
##               #    if part.pt()<self.cfg_ana.ptPartMin : continue
##               #### ===> skip pfcands outside the cone (this should be 0.3)
##               #    if deltaR(part.eta(), part.phi(), track.eta(), track.phi()) > self.cfg_ana.isoDR : continue
##               #    isoSum += part.pt()
##
##            #    ###            isoSum = isoSum/track.pt()  ## <--- this is for relIso
##
##            ### ===> the sum should not contain the track candidate
##
##            track.absIso = isoSum - track.pt()
##
##            #### store a preIso track
##            #event.preIsoTrack.append(track)
##            
###            if (isoSum < minIsoSum ) :
##            if(track.absIso < min(0.2*track.pt(), self.cfg_ana.maxAbsIso)): 
##                event.selectedIsoTrack.append(track)
##
##                if self.cfg_ana.doPrune:
##                    myMet = self.handles['met'].product()[0]
##                    mtwIsoTrack = mtw(track, myMet)
##                    if mtwIsoTrack < 100:
##                        if abs(track.pdgId()) == 11 or abs(track.pdgId()) == 13:
##                            if track.pt()>5 and track.absIso/track.pt()<0.2:
##
##                                myLeptons = [ l for l in event.selectedLeptons if l.pt() > 10 ] 
##                                nearestSelectedLeptons = makeNearestLeptons(myLeptons,track, event)
##                                if len(nearestSelectedLeptons) > 0:
##                                    for lep in nearestSelectedLeptons:
##                                        if deltaR(lep.eta(), lep.phi(), track.eta(), track.phi()) > 0.1:
##                                            event.selectedIsoCleanTrack.append(track)
##                                else: 
##                                    event.selectedIsoCleanTrack.append(track)

        event.selectedIsoTrack.sort(key = lambda l : l.pt(), reverse = True)
        event.selectedIsoCleanTrack.sort(key = lambda l : l.pt(), reverse = True)

        self.counters.counter('events').inc('all events')
        #if(len(event.preIsoTrack)): self.counters.counter('events').inc('has >=1 selected Track') 
        if(len(event.selectedIsoTrack)): self.counters.counter('events').inc('has >=1 selected Iso Track')

    
    def attachIsoAnulus(self, mu):

        mu.absIsoAnCharged = self.IsoTrackIsolationComputer.chargedAbsIso(mu.physObj, self.cfg_ana.anDeltaR,  self.cfg_ana.isoDR, 0.0);

        if self.isoAnPUCorr == None: puCorr = 'deltaBeta'
        else: puCorr = self.isoAnPUCorr

        mu.absIsoAnPho  = self.IsoTrackIsolationComputer.photonAbsIsoRaw( mu.physObj, self.cfg_ana.anDeltaR,  self.cfg_ana.isoDR, 0.0)
        mu.absIsoAnNHad = self.IsoTrackIsolationComputer.neutralHadAbsIsoRaw(mu.physObj, self.cfg_ana.anDeltaR, self.cfg_ana.isoDR, 0.0)
        mu.absIsoAnNeutral = mu.absIsoAnPho + mu.absIsoAnNHad
        if puCorr == "rhoArea":
            mu.absIsoAnNeutral = max(0.0, mu.absIsoAnNeutral - mu.rho * mu.EffectiveArea03 * (self.cfg_ana.anDeltaR/0.3)**2)
        elif puCorr == "deltaBeta":
            mu.absIsoAnPU = self.IsoTrackIsolationComputer.puAbsIso(mu.physObj, self.cfg_ana.anDeltaR, self.cfg_ana.isoDR, 0.0);
            mu.absIsoAnNeutral = max(0.0, mu.absIsoAnNeutral - 0.5*mu.absIsoAnPU)
        elif puCorr != 'raw':
            raise RuntimeError, "Unsupported miniIsolationCorr name '" + puCorr +  "'! For now only 'rhoArea', 'deltaBeta', 'raw' are supported."

        mu.absIsoAn = mu.absIsoAnCharged + mu.absIsoAnNeutral
        mu.relIsoAn = mu.absIsoAn/mu.pt()


    def matchIsoTrack(self, event):
        matchTau = matchObjectCollection3(event.selectedIsoTrack, event.gentaus + event.gentauleps + event.genleps, deltaRMax = 0.5)
        for lep in event.selectedIsoTrack:
            gen = matchTau[lep]
            lep.mcMatchId = 1 if gen else 0


    def printInfo(self, event):
        print 'event to Veto'
        print '----------------'

        if len(event.selectedIsoTrack)>0:
            print 'lenght: ',len(event.selectedIsoTrack)
            print 'track candidate pt: ',event.selectedIsoTrack[0].pt()
            print 'track candidate eta: ',event.selectedIsoTrack[0].eta()
            print 'track candidate phi: ',event.selectedIsoTrack[0].phi()
            print 'track candidate mass: ',event.selectedIsoTrack[0].mass()
            print 'pdgId candidate : ',event.selectedIsoTrack[0].pdgId()
            print 'dz: ',event.selectedIsoTrack[0].dz()
            print 'iso: ',event.selectedIsoTrack[0].absIso
            print 'matchId: ',event.selectedIsoTrack[0].mcMatchId 
                
#        for lepton in event.selectedLeptons:
#            print 'good lepton type: ',lepton.pdgId()
#            print 'pt: ',lepton.pt()
            
#        for tau in event.selectedTaus:
#            print 'good lepton type: ',tau.pdgId()
#            print 'pt: ',tau.pt()
            
        print '----------------'


    def process(self, event):

        if self.cfg_ana.setOff:
            return True

        self.readCollections( event.input )
        self.makeIsoTrack(event)

        if len(event.selectedIsoTrack)==0 : return True

##        event.pdgIdIsoTrack.append(event.selectedIsoTrack[0].pdgId())
##        event.isoIsoTrack.append(minIsoSum)
##        event.dzIsoTrack.append(abs(dz(event.selectedIsoTrack[0])))

### ===> do matching
        
        if not self.cfg_comp.isMC:
            return True

        if hasattr(event, 'gentaus') and hasattr(event, 'gentauleps') and hasattr(event, 'genleps') and self.cfg_ana.do_mc_match :
            self.matchIsoTrack(event)        

###        self.printInfo(event)
        
### ===> do veto if needed

#        if (self.cfg_ana.doSecondVeto and (event.selectedIsoTrack[0].pdgId()!=11) and (event.selectedIsoTrack[0].pdgId()!=12) and event.isoIsoTrack < self.cfg_ana.MaxIsoSum ) :
###            self.printInfo(event)
#            return False

#        if ((self.cfg_ana.doSecondVeto and event.selectedIsoTrack[0].pdgId()==11 or event.selectedIsoTrack[0].pdgId()==12) and event.isoIsoTrack < self.cfg_ana.MaxIsoSumEMU ) :
##            self.printInfo(event)
#            return False


        return True


setattr(IsoTrackAnalyzer,"defaultConfig",cfg.Analyzer(
    class_object=IsoTrackAnalyzer,
    setOff=True,
    #####
    candidates='packedPFCandidates',
    candidatesTypes='std::vector<pat::PackedCandidate>',
    ptMin = 5, # for pion 
    ptMinEMU = 5, # for EMU
    dzMax = 0.1,
    #####
    isoDR = 0.3,
    ptPartMin = 0,
    dzPartMax = 0.1,
    maxAbsIso = 8,
    #####
    doRelIsolation = False,
    MaxIsoSum = 0.1, ### unused
    MaxIsoSumEMU = 0.2, ### unused
    doSecondVeto = False,
    #####
    doIsoAnulus = False,
    anDeltaR = 0.4,
    isoAnPUCorr = 'deltaBeta',
    ###
    doPrune = True,
    do_mc_match = True, # note: it will in any case try it only on MC, not on data
  )
)
