from CMGTools.RootTools.analyzers.TreeAnalyzer import TreeAnalyzer
import random

class Particle(object):
    def pt(self): return -1000
    def eta(self): return -1000
    def phi(self): return -1000
    def charge(self): return -1000

dummyParticle = Particle()         

class DeltaTreeAnalyzer( TreeAnalyzer ):
    '''Just an example. You should create your analyzer on this model.

    One useful technique is to use other analyzers to fill the event with
    what you need. In your TreeAnalyzer, you can simply read the event
    and fill the tree.'''
    def declareVariables(self):

        def var( varName ):
            self.tree.addVar('float', varName)

        def particleVars( pName ):
            var('{pName}Pt'.format(pName=pName))
            var('{pName}Eta'.format(pName=pName))
            var('{pName}Phi'.format(pName=pName))
            var('{pName}Charge'.format(pName=pName))
            var('{pName}Sel'.format(pName=pName))
            var('{pName}Iso'.format(pName=pName))
            # var('{pName}Iso'.format(pName=pName))

        particleVars('gen')
        particleVars('col1')
        particleVars('col2')
        # particleVars('col1G')
        # particleVars('gen2')
        # particleVars('col2G')
        self.tree.book()

    def process(self, iEvent, event):
        
        def fill( varName, value ):
            setattr( self.tree.s, varName, value )

        def fParticleVars( pName, particle ):
            fill('{pName}Pt'.format(pName=pName), particle.pt() )
            fill('{pName}Eta'.format(pName=pName), particle.eta() )
            fill('{pName}Phi'.format(pName=pName), particle.phi() )
            fill('{pName}Charge'.format(pName=pName), particle.charge() )
            if hasattr( particle, 'selected'):
                fill('{pName}Sel'.format(pName=pName), particle.selected )
            if hasattr( particle, 'relIso'):
                fill('{pName}Iso'.format(pName=pName), particle.relIso(0.5) )

        
        assert( len(event.pairsG1) == len(event.pairsG2) )
        
        for (gen, col1), (gen, col2) in zip( event.pairsG1.iteritems(), event.pairsG2.iteritems()):
            fParticleVars('gen', gen)
            if col1 is None: col1 = dummyParticle
            if col2 is None: col2 = dummyParticle
            fParticleVars('col1', col1)
            fParticleVars('col2', col2)
            # one entry per gen particle 
            self.tree.fill()

        return True 

        # we get a -1000 when a dummy particle is put in a matching
        #   no matching found, probably because no particle found in other collection

        # gen eta = -99 means that the tree initialized itself to -99
        #   probably: no gen muon found? 

##         for p1, p2 in event.pairs.iteritems():
##             fParticleVars('col1', p1)
##             if p2 is None: p2 = dummyParticle
##             fParticleVars('col2', p2)

##         for gen, col1 in event.pairsG1.iteritems():
##             if col1 is None: col1 = dummyParticle 
##             fParticleVars('gen1', gen)
##             fParticleVars('col1G', col1)

##         for gen, col2 in event.pairsG2.iteritems():
##             if col2 is None: col2 = dummyParticle 
##             fParticleVars('gen2', gen)
##             fParticleVars('col2G', col2)

            
##         if len(event.col1)>0:
##             self.tree.fill()
                
