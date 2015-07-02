from math import *
from CMGTools.HToZZ4L.analyzers.FourLeptonAnalyzerBase import *

        
class FourLeptonAnalyzer( FourLeptonAnalyzerBase ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(FourLeptonAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        self.tag = cfg_ana.tag
    def declareHandles(self):
        super(FourLeptonAnalyzer, self).declareHandles()

    def beginLoop(self, setup):
        super(FourLeptonAnalyzer,self).beginLoop(setup)
        self.counters.addCounter('FourLepton')
        count = self.counters.counter('FourLepton')
        count.register('all events')


    def process(self, event):
        self.readCollections( event.input )

        subevent = EventBox()
        setattr(event,'fourLeptonAnalyzer'+self.tag,subevent)

        #startup counter
        self.counters.counter('FourLepton').inc('all events')

        #create a cut flow
        cutFlow = CutFlowMaker(self.counters.counter("FourLepton"),subevent,event.selectedLeptons)

        passed = cutFlow.applyCut(lambda x:True,'At least four loose leptons',4,'looseLeptons')

        #Ask for four goodleptons
        passed = cutFlow.applyCut(self.leptonID,'At least four good non isolated Leptons',4,'goodLeptons')


        #Create Four Lepton Candidates
        subevent.fourLeptonPreCands = self.findOSSFQuads(cutFlow.obj1,event.fsrPhotons)
        cutFlow.setSource1(subevent.fourLeptonPreCands)

        #Apply isolation on all legs
        passed=cutFlow.applyCut(self.fourLeptonIsolation,'At least four OSSF Isolated leptons   ',1,'fourLeptonsIsolated')

        #Apply minimum Z mass
        passed=cutFlow.applyCut(self.fourLeptonMassZ1Z2,'Z masses between 12 and 120 GeV',1,'fourLeptonsZMass')

        #Apply ghost suppression
        passed=cutFlow.applyCut(self.ghostSuppression,'Ghost suppression ',1,'fourLeptonsGhostSuppressed')

        #Pt Thresholds
        passed=cutFlow.applyCut(self.fourLeptonPtThresholds,'Pt 20 and 10 GeV',1,'fourLeptonsPtThresholds')

        #QCD suppression
        passed=cutFlow.applyCut(self.qcdSuppression,'QCD suppression',1,'fourLeptonsPtThresholds')

        #Z1 mass
        passed=cutFlow.applyCut(self.fourLeptonMassZ1,'Z1 Mass cut',1,'fourLeptonsMass')

        #smart cut
        passed=cutFlow.applyCut(self.stupidCut,'Smart cut',1,'fourLeptonsFinal')

        #compute MELA
        passed=cutFlow.applyCut(self.fillMEs,'Fill MEs',1,'fourLeptonsWithME')

        #Save the best
        if len(subevent.fourLeptonsFinal)>0:
            subevent.fourLeptonsFinal=sorted(subevent.fourLeptonsFinal,key=lambda x: x.leg2.leg1.pt()+x.leg2.leg2.pt(),reverse=True)
            subevent.fourLeptonsFinal=sorted(subevent.fourLeptonsFinal,key=lambda x: abs(x.leg1.M()-91.1876))
            setattr(event,'bestFourLeptons'+self.tag,subevent.fourLeptonsFinal[:getattr(self.cfg_ana,'maxCand',1)])
        else:    
            setattr(event,'bestFourLeptons'+self.tag,[])

        #FSR test
        passedFSR=cutFlow.applyCut(lambda x: x.hasFSR(),'FSR tagged',1,'fourLeptonsFSR')
        if passedFSR:
            for c in subevent.fourLeptonsFSR:
                #print 'Mass' ,c.fsrUncorrected().M(),c.M()
                pass

        return True        
