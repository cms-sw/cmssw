import copy
from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.AutoHandle import AutoHandle
import random

def pv(vc):
    print 'x = {x:5.4f},  y = {y:5.4f},  z = {z:5.4f}'.format(x=vc.X(),
                                                              y=vc.Y(),
                                                              z=vc.Z())

class MetAnalyzer( Analyzer ):
    '''Analyze MET in Z+jet events.
    Need a to provide a module creating event.diLepton
    earlier in the sequence.
    '''

    def declareHandles(self):
        super(MetAnalyzer, self).declareHandles()
        self.handles['met'] =  AutoHandle(
            self.cfg_ana.metCol,
            self.cfg_ana.metType
            )


    def beginLoop(self, setup):
        super(MetAnalyzer,self).beginLoop(setup)

       
    def process(self, iEvent, event):
        self.readCollections( iEvent )
        event.met = self.handles['met'].product()[0]
        met = event.met
        # here, do pure met stuff
        
        if not hasattr(event, 'diLepton'):
            return False
        
        diL = event.diLepton

##         rnd = random.random()
##         leg = diL.leg1()
##         if rnd>0.5:
##             leg = diL.leg2()


        # here, do recoil stuff
        
        mVect = met.p4().Vect()
        mVect.SetZ(0.)
        vVect = diL.p4().Vect()
        vVect.SetZ(0.)
        recoilVect = copy.deepcopy(mVect)
        recoilVect -= vVect
        
        uvVect = vVect.Unit()
        zAxis = type(vVect)(0,0,1)
        uvVectPerp = vVect.Cross(zAxis).Unit()

        u1 = - recoilVect.Dot(uvVect)
        u2 = recoilVect.Dot(uvVectPerp)

        event.u1 = u1
        event.u2 = u2
        
        if self.cfg_ana.verbose:
            print 'met', met.pt()
            print 'diL', diL
            print 'vVect'
            pv(vVect)
            print 'uvVect'
            pv(uvVect)
            print 'uvVectPerp'
            pv(uvVectPerp)
            print u1, u2
        
        return True




##     def write(self, setup):
##         super(MetAnalyzer, self).write(setup)
