from CMGTools.RootTools.fwlite.Analyzer import Analyzer
from CMGTools.RootTools.fwlite.AutoHandle import AutoHandle
from CMGTools.RootTools.physicsobjects.PhysicsObjects import PhysicsObject, GenParticle
from CMGTools.RootTools.utils.DeltaR import cleanObjectCollection, matchObjectCollection



class DeltaAnalyzer( Analyzer ):
    '''Matches 2 collections of any objects.
    Can for example put a TreeAnalyzer behind
    '''

    def declareHandles(self):
        super(DeltaAnalyzer, self).declareHandles()

        self.handles['col1'] =  AutoHandle(
            self.cfg_ana.col1_instance,
            self.cfg_ana.col1_type 
            )
        
        self.handles['col2'] =  AutoHandle(
            self.cfg_ana.col2_instance,
            self.cfg_ana.col2_type 
            )

        self.mchandles['gen'] =  AutoHandle(
            self.cfg_ana.gen_instance,
            self.cfg_ana.gen_type 
            )


    def process(self, iEvent, event ):
        self.readCollections(iEvent)

        event.gen = []
        for genp in self.mchandles['gen'].product():
            if abs(genp.pdgId())!= self.cfg_ana.gen_pdgId:
                continue
            event.gen.append( GenParticle(genp) )

        event.col1 =  map(PhysicsObject, self.handles['col1'].product())
        event.col2 =  map(PhysicsObject, self.handles['col2'].product())

        for p in event.col1:
            if hasattr( self.cfg_ana, 'sel1'):
                p.selected = self.cfg_ana.sel1( p )
            else:
                p.selected = True
                
        for p in event.col2:
            if hasattr( self.cfg_ana, 'sel2'):
                p.selected = self.cfg_ana.sel2( p )
            else:
                p.selected = True
                
                
            
        # first collection is taken as a pivot.
        # will store in the tree all instances in col1, and each time,
        # the closest object in col2

        event.pairs = matchObjectCollection( event.col1, event.col2,
                                             self.cfg_ana.deltaR )

        event.pairsG1 = matchObjectCollection( event.gen, event.col1,
                                               self.cfg_ana.deltaR )

        event.pairsG2 = matchObjectCollection( event.gen, event.col2,
                                               self.cfg_ana.deltaR )
        
        # yeah, that's it
        
