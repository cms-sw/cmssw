from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
import PhysicsTools.HeppyCore.framework.config as cfg
from math import *
from DataFormats.FWLite import Events, Handle

class LHEAnalyzer( Analyzer ):
    """    """
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(LHEAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        self.lheh=Handle('LHEEventProduct')

    def declareHandles(self):
        super(LHEAnalyzer, self).declareHandles()
#        self.mchandles['lhestuff'] = AutoHandle( 'externalLHEProducer','LHEEventProduct')

    def beginLoop(self, setup):
        super(LHEAnalyzer,self).beginLoop(setup)

    def process(self, event):

        # if not MC, nothing to do
        if not self.cfg_comp.isMC: 
            return True
        event.lheHT=0
        event.lheHTIncoming=0 #restrict HT computation to particles that have status<0 mothers
        event.lheNj=0
        event.lheNb=0
        event.lheNc=0
        event.lheNl=0
        event.lheNg=0
        event.lheV_pt = 0
        try:
          event.input.getByLabel( 'externalLHEProducer',self.lheh)
        except :
          return True
        if not  self.lheh.isValid() :
            return True
        self.readCollections( event.input )
        hepeup=self.lheh.product().hepeup()
        pup=hepeup.PUP
        l=None
        lBar=None
        nu=None
        nuBar=None 
        for i in xrange(0,len(pup)):
          id=hepeup.IDUP[i]
          status = hepeup.ISTUP[i]
          idabs=abs(id)

          mothIdx = max(hepeup.MOTHUP[i][0]-1,0) #first and last mother as pair; first entry has index 1 in LHE; incoming particles return motherindex 0
          mothIdxTwo = max(hepeup.MOTHUP[i][1]-1,0) 
          
          mothStatus  = hepeup.ISTUP[mothIdx] 
          mothStatusTwo  = hepeup.ISTUP[mothIdxTwo] 

          hasIncomingAsMother = mothStatus<0 or mothStatusTwo<0
          
          if status == 1 and ( ( idabs == 21 ) or (idabs > 0 and idabs < 7) ) : # gluons and quarks
              pt = sqrt( pup[i][0]**2 + pup[i][1]**2 ) # first entry is px, second py
              event.lheHT += pt
              if hasIncomingAsMother: event.lheHTIncoming += pt
              event.lheNj +=1
              if idabs==5:
                event.lheNb += 1
              if idabs==4:
                event.lheNc += 1
              if idabs in [1,2,3]:
                event.lheNl += 1
              if idabs==21:
                event.lheNg += 1
          if idabs in [12,14,16] :  
              if id > 0 :
                nu = i
              else :
                nuBar = i
          if idabs in [11,13,15] :  
              if id > 0 :
                l = i
              else :
                lBar = i
          v=None
          if l and lBar : #Z to LL
              v=(l,lBar)
          elif l and nuBar : #W 
              v=(l,nuBar)
          elif lBar and nu : #W 
              v=(nu,lBar)
          elif nu and nuBar : #Z to nn 
              v=(nu,nuBar)
          if v :
            event.lheV_pt = sqrt( (pup[v[0]][0]+pup[v[1]][0])**2 +  (pup[v[0]][1]+pup[v[1]][1])**2 )

        return True

setattr(LHEAnalyzer,"defaultConfig",
    cfg.Analyzer(LHEAnalyzer,
    )
)
