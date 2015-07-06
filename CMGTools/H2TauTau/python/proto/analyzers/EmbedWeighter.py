from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.HeppyCore.statistics.average import Average
from PhysicsTools.Heppy.utils.cmsswRelease import cmsswIs52X
from PhysicsTools.Heppy.physicsobjects import GenParticle

class EmbedWeighter( Analyzer ):
    '''Gets lepton efficiency weight and puts it in the event.
    Applies additional cuts:
    - m(tau tau) > 50 GeV
    - Require matched generated tau
    '''

    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(EmbedWeighter,self).__init__(cfg_ana, cfg_comp, looperName)

            
    def beginLoop(self, setup):
        print self, self.__class__
        super(EmbedWeighter,self).beginLoop(setup)
        self.averages.add('weight', Average('weight') )
        self.counters.addCounter('EmbedWeighter')
        count = self.counters.counter('EmbedWeighter')
        count.register('all events')
        count.register('gen Z mass > 50')


    def declareHandles(self):
        super(EmbedWeighter,self).declareHandles()
        #import pdb ; pdb.set_trace()
        if self.cfg_comp.isEmbed:
            isRHEmbedded = self.cfg_ana.isRecHit
            if 'PFembedded' in self.cfg_comp.name and isRHEmbedded :
              'WARNING: in the cfg you set RecHit, but this appears to be PF embedded'
            if 'RHembedded' in self.cfg_comp.name and not isRHEmbedded :
              'WARNING: in the cfg you set PF, but this appears to be RecHit embedded'
            if cmsswIs52X():
                self.embhandles['minVisPtFilter'] = AutoHandle(
                    ('generator', 'minVisPtFilter'),
                    'GenFilterInfo'
                    )
                self.embhandles['genpart'] =  AutoHandle(
                        'genParticles',
                        'std::vector<reco::GenParticle>'
                        )
                if isRHEmbedded:
                    self.embhandles['TauSpinnerReco'] = AutoHandle(
                        ('TauSpinnerReco', 'TauSpinnerWT'),
                        'double'
                        )
                    self.embhandles['ZmumuEvtSelEffCorrWeightProducer'] = AutoHandle(
                        ('ZmumuEvtSelEffCorrWeightProducer', 'weight'),
                        'double'
                        )
                    self.embhandles['muonRadiationCorrWeightProducer'] = AutoHandle(
                        ('muonRadiationCorrWeightProducer', 'weight'),
                        'double'
                        )
                    self.embhandles['genTau2PtVsGenTau1Pt'] = AutoHandle(
                        ('embeddingKineReweightRECembedding', 'genTau2PtVsGenTau1Pt'),
                        'double'
                        )
                    self.embhandles['genTau2EtaVsGenTau1Eta'] = AutoHandle(
                        ('embeddingKineReweightRECembedding', 'genTau2EtaVsGenTau1Eta'),
                        'double'
                        )
                    self.embhandles['genDiTauMassVsGenDiTauPt'] = AutoHandle(
                        ('embeddingKineReweightRECembedding', 'genDiTauMassVsGenDiTauPt'),
                        'double'
                        )
                    
            else:
                self.embhandles['minVisPtFilter'] = AutoHandle(
                    ('generator', 'weight'),
                    'double'
                    )

                

    def process(self, event):
        self.readCollections( event.input )
        self.weight = 1
        isRHEmbedded = False
        event.genfilter                = 1.
        event.tauspin                  = 1.
        event.zmumusel                 = 1.
        event.muradcorr                = 1.
        event.genTau2PtVsGenTau1Pt     = 1.
        event.genTau2EtaVsGenTau1Eta   = 1.
        event.genDiTauMassVsGenDiTauPt = 1.
        if self.cfg_comp.isEmbed:
            try: 
                genfilter = self.embhandles['minVisPtFilter'].product()
                if isRHEmbedded:
                    tauspin                  = self.embhandles['TauSpinnerReco'].product()
                    zmumusel                 = self.embhandles['ZmumuEvtSelEffCorrWeightProducer'].product()
                    muradcorr                = self.embhandles['muonRadiationCorrWeightProducer'].product()
                    genTau2PtVsGenTau1Pt     = self.embhandles['genTau2PtVsGenTau1Pt'].product()
                    genTau2EtaVsGenTau1Eta   = self.embhandles['genTau2EtaVsGenTau1Eta'].product()
                    genDiTauMassVsGenDiTauPt = self.embhandles['genDiTauMassVsGenDiTauPt'].product()
            except RuntimeError:
                print 'WARNING EmbedWeighter, cannot find the weight in the event'
                return False
            if cmsswIs52X():
                self.weight = genfilter.filterEfficiency()
                if isRHEmbedded:
                    self.weight *= tauspin[0]
                    self.weight *= zmumusel[0]
                    self.weight *= muradcorr[0]
                    self.weight *= genTau2PtVsGenTau1Pt[0]
                    self.weight *= genTau2EtaVsGenTau1Eta[0]
                    self.weight *= genDiTauMassVsGenDiTauPt[0]

                event.genfilter                = genfilter.filterEfficiency()
            
                if isRHEmbedded:
                  event.tauspin                  = tauspin[0]
                  event.zmumusel                 = zmumusel[0]
                  event.muradcorr                = muradcorr[0]
                  event.genTau2PtVsGenTau1Pt     = genTau2PtVsGenTau1Pt[0]
                  event.genTau2EtaVsGenTau1Eta   = genTau2EtaVsGenTau1Eta[0]
                  event.genDiTauMassVsGenDiTauPt = genDiTauMassVsGenDiTauPt[0]

                self.counters.counter('EmbedWeighter').inc('all events')

                event.genParticles = map( GenParticle, self.embhandles['genpart'].product() )
                genTaus = [p for p in event.genParticles if abs(p.pdgId()) == 15]
                if len(genTaus) != 2:
                    print 'WARNING EmbedWeighter, not 2 gen taus in the event'
                genZMass = (genTaus[0].p4() + genTaus[1].p4()).mass()
                # import pdb; pdb.set_trace()
                
                if genZMass < 50.:
                    return False
                self.counters.counter('EmbedWeighter').inc('gen Z mass > 50')
            else: 
                self.weight = genfilter[0]
        if self.cfg_ana.verbose:
            print self.name, 'efficiency =', self.weight
        event.eventWeight *= self.weight
        event.embedWeight = self.weight
        self.averages['weight'].add( self.weight )
        return True
                
