from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from math import floor
import re
        
class susyParameterScanAnalyzer( Analyzer ):
    """Get information for susy parameter scans    """
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(susyParameterScanAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        self.susyParticles = {
            100001 : 'Squark',
            (1000000 + 21) : 'Gluino',
            (1000000 + 39) : 'Gravitino',
            (1000000 + 5) : 'Sbottom',
            (2000000 + 5) : 'Sbottom2',
            (1000000 + 6) : 'Stop',
            (2000000 + 6) : 'Stop2',
            (1000000 + 22) : 'Neutralino',
            (1000000 + 23) : 'Neutralino2',
            (1000000 + 25) : 'Neutralino3',
            (1000000 + 35) : 'Neutralino4',
            (1000000 + 24) : 'Chargino',
            (1000000 + 37) : 'Chargino2',
        }
    #---------------------------------------------
    # DECLARATION OF HANDLES OF GEN LEVEL OBJECTS 
    #---------------------------------------------
        

    def declareHandles(self):
        super(susyParameterScanAnalyzer, self).declareHandles()

        #mc information
        self.mchandles['genParticles'] = AutoHandle( 'prunedGenParticles',
                                                     'std::vector<reco::GenParticle>' )
        if self.cfg_ana.doLHE:
            self.mchandles['lhe'] = AutoHandle( 'source', 'LHEEventProduct', mayFail = True, lazy = False )
        
    def beginLoop(self, setup):
        super(susyParameterScanAnalyzer,self).beginLoop(setup)


    def findSusyMasses(self,event):
        masses = {}
        for p in event.genParticles:
            id = abs(p.pdgId())
            if (id / 1000000) % 10 in [1,2]:
                particle = None
                if id % 100 in [1,2,3,4]:
                    particle = "Squark"
                elif id in self.susyParticles:
                    particle = self.susyParticles[id]
                if particle != None:
                    if particle not in masses: masses[particle] = []
                    masses[particle].append(p.mass())
        for p,ms in masses.iteritems():
            avgmass = floor(sum(ms)/len(ms)+0.5)
            setattr(event, "genSusyM"+p, avgmass)

    def readLHE(self,event):
        if not self.mchandles['lhe'].isValid():
            if not hasattr(self,"warned_already"):
                print "ERROR: Missing LHE header in file"
                self.warned_already = True
            return
        lheprod = self.mchandles['lhe'].product()
        scanline = re.compile(r"#\s*model\s+([A-Za-z0-9]+)_((\d+\.?\d*)(_\d+\.?\d*)*)\s+(\d+\.?\d*)\s*")
        for i in xrange(lheprod.comments_size()):
            comment = lheprod.getComment(i)
            m = re.match(scanline, comment) 
            if m:
                event.susyModel = m.group(1)
                masses = [float(x) for x in m.group(2).split("_")]
                if len(masses) >= 1: event.genSusyMScan1 = masses[0]
                if len(masses) >= 2: event.genSusyMScan2 = masses[1]
                if len(masses) >= 3: event.genSusyMScan3 = masses[2]
                if len(masses) >= 4: event.genSusyMScan4 = masses[3]
            elif "model" in comment:
                if not hasattr(self,"warned_already"):
                    print "ERROR: I can't understand the model: ",comment
                    self.warned_already = True

    def process(self, event):
        # if not MC, nothing to do
        if not self.cfg_comp.isMC: 
            return True

        self.readCollections( event.input )

        # create parameters
        event.susyModel = None
        for id,X in self.susyParticles.iteritems():
            setattr(event, "genSusyM"+X, -99.0)
        event.genSusyMScan1 = 0.0
        event.genSusyMScan2 = 0.0
        event.genSusyMScan3 = 0.0
        event.genSusyMScan4 = 0.0

        # do MC level analysis
        if self.cfg_ana.doLHE:
            self.readLHE(event)
        self.findSusyMasses(event)
        return True
