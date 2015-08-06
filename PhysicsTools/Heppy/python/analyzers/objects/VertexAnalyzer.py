import itertools

from PhysicsTools.Heppy.analyzers.core.VertexHistograms import VertexHistograms
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.HeppyCore.statistics.average import Average
from PhysicsTools.Heppy.physicsutils.PileUpSummaryInfo import PileUpSummaryInfo
import PhysicsTools.HeppyCore.framework.config as cfg

class VertexAnalyzer( Analyzer ):
    """selects a list of good primary vertices,
    and optionally add a pile-up weight to MC events.

    The list of good primary vertices is put in event.goodVertices.
    if no good vertex is found, the process function returns False.

    The weight is put in event.vertexWeight, and is multiplied to
    the global event weight, event.eventWeight. 

    Example:
    
    vertexAna = cfg.Analyzer(
      'VertexAnalyzer',
      goodVertices = 'goodPVFilter',
      vertexWeight = 'vertexWeightFall112011AB',
      # uncomment the following line if you want a vertex weight = 1 (no weighting)
      # fixedWeight = 1, 
      verbose = False
      )

    If fixedWeight is set to None, the vertex weight is read from the EDM collection with module name
    'vertexWeightFall112011AB'.
    Otherwise, the weight is set to fixedWeight.

    The vertex weight collection was at some point produced in the PAT+CMG step,
    and could directly be accessed from the PAT or CMG tuple. 
    In the most recent versions of the PAT+CMG tuple, this collection is not present anymore,
    and an additional full framework process must be ran to produce this collection,
    so that this analyzer can read it. An example cfg to do that can be found here:
    http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/UserCode/CMG/CMGTools/H2TauTau/prod/vertexWeight2011_cfg.py?view=markup

    
    """

    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(VertexAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)

        self.doHists=True
        if (hasattr(self.cfg_ana,'makeHists')) and (not self.cfg_ana.makeHists):
            self.doHists=False
        if self.doHists:    
            self.pileup = VertexHistograms('/'.join([self.dirName,
                                                     'pileup.root']))
        
        self.allVertices = self.cfg_ana.allVertices if (hasattr(self.cfg_ana,'allVertices')) else "_AUTO_"

    def declareHandles(self):
        super(VertexAnalyzer, self).declareHandles()
        if self.allVertices == '_AUTO_':
          self.handles['vertices'] =  AutoHandle( "offlineSlimmedPrimaryVertices", 'std::vector<reco::Vertex>', fallbackLabel="offlinePrimaryVertices" )
        else:
          self.handles['vertices'] =  AutoHandle( self.allVertices, 'std::vector<reco::Vertex>' )
        self.fixedWeight = None
        if self.cfg_comp.isMC:
            if hasattr( self.cfg_ana, 'fixedWeight'):
                self.fixedWeight = self.cfg_ana.fixedWeight
            else:
                self.mchandles['vertexWeight'] = AutoHandle( self.cfg_ana.vertexWeight,
                                                             'double' )

        self.mchandles['pusi'] =  AutoHandle(
            'addPileupInfo',
            'std::vector<PileupSummaryInfo>' 
            )        

        self.handles['rho'] =  AutoHandle(
            ('fixedGridRhoFastjetAll',''),
            'double' 
            )        
        self.handles['sigma'] =  AutoHandle(
            ('fixedGridSigmaFastjetAll',''),
            'double',
            mayFail=True
            )

    def beginLoop(self, setup):
        super(VertexAnalyzer,self).beginLoop(setup)
        self.averages.add('vertexWeight', Average('vertexWeight') )
        self.counters.addCounter('GoodVertex')
        self.count = self.counters.counter('GoodVertex')
        self.count.register('All Events')
        self.count.register('Events With Good Vertex')

        
    def process(self,  event):
        self.readCollections(event.input )
        event.rho = self.handles['rho'].product()[0]
        event.sigma = self.handles['sigma'].product()[0] if self.handles['sigma'].isValid() else -999
        event.vertices = self.handles['vertices'].product()
        event.goodVertices = filter(self.testGoodVertex,event.vertices)


        self.count.inc('All Events')

        
        event.vertexWeight = 1
        if self.cfg_comp.isMC:
            event.pileUpInfo = map( PileUpSummaryInfo,
                                    self.mchandles['pusi'].product() )
            if self.fixedWeight is None:
                event.vertexWeight = self.mchandles['vertexWeight'].product()[0]
            else:
                event.vertexWeight = self.fixedWeight
        event.eventWeight *= event.vertexWeight
            
        self.averages['vertexWeight'].add( event.vertexWeight )
        if self.verbose:
            print 'VertexAnalyzer: #vert = ', len(event.vertices), \
                  ', weight = ', event.vertexWeight

        # Check if events needs to be skipped if no good vertex is found (useful for generator level studies)
        keepFailingEvents = False
        if hasattr( self.cfg_ana, 'keepFailingEvents'):
            keepFailingEvents = self.cfg_ana.keepFailingEvents
        if len(event.goodVertices)==0:
            event.passedVertexAnalyzer=False
            if not keepFailingEvents:
                return False
        else:
            event.passedVertexAnalyzer=True

        if self.doHists:
            self.pileup.hist.Fill( len(event.goodVertices) )
#A.R. mindist is one of the slowest functions, default commented
#           self.pileup.mindist.Fill( self.mindist(event.goodVertices) )

        self.count.inc('Events With Good Vertex')
        return True


    def testGoodVertex(self,vertex):
        if vertex.isFake():
            return False
        if vertex.ndof()<=4:
            return False
        if abs(vertex.z())>24:
            return False
        if vertex.position().Rho()>2:
            return False
     
        return True

    def mindist(self, vertices):
        mindist = 999999
        for comb in itertools.combinations(vertices, 2):
            dist = abs(comb[0].z() - comb[1].z())
            if dist<mindist:
                mindist = dist
        return mindist
                                                                 
    def write(self, setup):
        super(VertexAnalyzer, self).write(setup)
        if self.doHists:
            self.pileup.write()

setattr(VertexAnalyzer,"defaultConfig",cfg.Analyzer(
    class_object=VertexAnalyzer,
    vertexWeight = None,
    fixedWeight = 1,
    verbose = False
   )
)
