
from CMGTools.RootTools.fwlite.Analyzer import Analyzer
from CMGTools.RootTools.fwlite.AutoHandle import AutoHandle
from CMGTools.RootTools.statistics.Average import Average
from CMGTools.RootTools.statistics.Histograms import Histograms
from CMGTools.RootTools.physicsobjects.PhysicsObjects import GenParticle,Jet, GenJet
from CMGTools.RootTools.utils.DeltaR import cleanObjectCollection, matchObjectCollection, matchObjectCollection2, deltaR2
from CMGTools.RootTools.utils.PileupJetHistograms import PileupJetHistograms
## from CMGTools.RootTools.RootTools import loadLibs

from ROOT import TNtuple, TH1F, TH2F, TFile, THStack, TF1, TGraphErrors


class SimpleJetNTupler (Analyzer) :
    '''dump very few quantities into a TNtuple for jet resolution studies.'''
    ### def __init__(self,cfg_ana, cfg_comp, looperName):
    ###     loadLibs()
    ###     super (SimpleJetNTupler, self).__init__(cfg_ana, cfg_comp, looperName)

    def declareHandles (self) :
        super (SimpleJetNTupler, self).declareHandles ()
        self.handles['jets'] =  AutoHandle (
            *self.cfg_ana.jetCollection
            )
        if self.cfg_ana.useGenLeptons: 
            self.mchandles['genParticlesPruned'] =  AutoHandle (
                'genParticlesPruned',
                'std::vector<reco::GenParticle>'
                )
        else:
            self.mchandles['genParticles'] =  AutoHandle (
                'prunedGen',
                'std::vector<reco::GenParticle>'
                )
            
        self.mchandles['genJets'] =  AutoHandle (
            *self.cfg_ana.genJetsCollection
           )
        self.handles['vertices'] =  AutoHandle (
            'offlinePrimaryVertices',
            'std::vector<reco::Vertex>'
          )

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def beginLoop (self) :
        super (SimpleJetNTupler,self).beginLoop ()
        self.file = TFile ('/'.join ([self.looperName, 'testJetsNT.root']),
                           'recreate')
        if self.cfg_ana.applyPFLooseId:
            from ROOT import PFJetIDSelectionFunctor 
            self.isPFLooseFunc = PFJetIDSelectionFunctor(0,PFJetIDSelectionFunctor.LOOSE)
            ## Workaround: for some reason PyROOT does not bind nor PFJetIDSelectionFunctor(Jet)PFJetIDSelectionFunctor.getBitsTemplates 
            from ROOT import pat        
            self.isPFLooseFunc.bits = pat.strbitset()
            for i in "CHF","NHF","CEF","NEF","NCH","nConstituents": self.isPFLooseFunc.bits.push_back(i) 
            ## /Workaround
            self.isPFLoose = lambda x : self.isPFLooseFunc(x,self.isPFLooseFunc.bits)
        else:
            self.isPFLoose = lambda x : True

        self.myntuple = TNtuple (self.cfg_ana.ntupleName, 
                                 self.cfg_ana.ntupleName, 'genPt:recoPt:genEta:recoEta:genPhi:recoPhi:nvtx')


# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....
    def process (self, iEvent, event) :
        #read all the handles defined beforehand
        self.readCollections (iEvent)
        
        jetEtaCut = 4.5 
        
        # get the vertexes
        event.vertices = self.handles['vertices'].product ()
#        self.h_nvtx.Fill (len (event.vertices))
        
        # get the jets in the jets variable
        jets = self.handles['jets'].product ()
        # filter jets with some selections
        event.jets = [ jet for jet in jets if ( abs(jet.eta()) < jetEtaCut and jet.pt()>self.cfg_ana.ptCut and self.isPFLoose(jet) ) ]
        
        # get status 2 leptons
        if 'genParticlesPruned' in self.mchandles:
            event.genLeptons = [ lep for lep in self.mchandles['genParticlesPruned'].product() if lep.status() == 2 and (abs(lep.pdgId()) == 11 or abs(lep.pdgId()) == 13 or abs(lep.pdgId()) == 15) ]  
        else:
            event.genLeptons = [ lep for lep in self.mchandles['genParticles'].product() if lep.status() == 3 and (abs(lep.pdgId()) == 11 or abs(lep.pdgId()) == 13 or abs(lep.pdgId()) == 15) ]  
# @ Pasquale: why level 3 and not level 2?
#        event.selGenLeptons = [GenParticle (lep) for lep in event.genLeptons if (lep.pt ()>self.cfg_ana.ptCut and abs (lep.eta ()) < jetEtaCut)]
        
        # get genJets
        event.genJets = map (GenJet, self.mchandles['genJets'].product ())
        # filter genjets as for reco jets
        event.selGenJets = [GenJet (jet) for jet in event.genJets if (jet.pt ()>self.cfg_ana.genPtCut)]
        
        #FIXME why are there cases in which there's 4 or 6 leptons?
        if len (event.genLeptons) != 2 :
            return
        # in case I want to filter out taus
        # 11, 13, 15 : e, u, T
#        event.genOneLepton = [GenParticle (part) for part in event.genLeptons if abs (part.pdgId ()) == 15]
        # remove leptons from jets if closer than 0.2
        event.cleanJets = cleanObjectCollection (event.jets, event.genLeptons, 0.2)
        event.matchingCleanJets = matchObjectCollection2 (event.cleanJets, event.selGenJets, 0.25)
        # assign to each jet its gen match (easy life :))
        for jet in event.cleanJets :
            jet.gen = event.matchingCleanJets[ jet ]

        
        event.matchedCleanJets = [jet for jet in event.matchingCleanJets if jet.gen != None]
        for jet in event.matchedCleanJets:
            self.myntuple.Fill (jet.gen.pt (), jet.pt (), jet.gen.eta (), jet.eta (), jet.gen.phi (), jet.phi (), len (event.vertices))

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def write (self):
        from ROOT import gROOT
        gROOT.SetBatch(True)
        self.file.cd ()
        self.myntuple.Write ()
        self.file.Close()
        
