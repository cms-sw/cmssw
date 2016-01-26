import os 
from PhysicsTools.Heppy.analyzers.core.VertexHistograms import VertexHistograms
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.HeppyCore.statistics.average import Average
from PhysicsTools.Heppy.physicsutils.PileUpSummaryInfo import PileUpSummaryInfo
import PhysicsTools.HeppyCore.framework.config as cfg

from ROOT import TFile, TH1F

class PileUpAnalyzer( Analyzer ):
    '''Computes pile-up weights for MC from the pile up histograms for MC and data.
    These histograms should be set on the components as
    puFileData, puFileMC attributes, as is done here:
    
    http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/UserCode/CMG/CMGTools/H2TauTau/Colin/test_tauMu_2012_cfg.py?view=markup

    THESE HISTOGRAMS MUST BE CONSISTENT, SEE
    https://twiki.cern.ch/twiki/bin/view/CMS/CMGToolsPileUpReweighting#Generating_pile_up_distributions

    If the component is not MC, or if the puFileData and puFileMC are not
    set for the component, the reweighting is not done. 
    
    The analyzer sets event.vertexWeight.
    This weight is multiplied to the global event weight, event.eventWeight.
    When using this analyzer, make sure that the VertexAnalyzer is disabled,
    as you would be reweighting the MC PU distribution twice!
    
    Additionally, this analyzer writes in the output an histogram containing the unweighting MC
    pile-up distribution, to be used in input of the weighting for a later pass. 
    
    Example of use: 
    
    puAna = cfg.Analyzer(
      "PileUpAnalyzer",
      # build unweighted pu distribution using number of pile up interactions if False
      # otherwise, use fill the distribution using number of true interactions
      true = True
      )
    '''

    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(PileUpAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)

        self.doHists=True

        if (hasattr(self.cfg_ana,'makeHists')) and (not self.cfg_ana.makeHists):
            self.doHists=False

        self.allVertices = self.cfg_ana.allVertices if (hasattr(self.cfg_ana,'allVertices')) else "_AUTO_"

        if self.cfg_comp.isMC and self.doHists:
            self.rawmcpileup = VertexHistograms('/'.join([self.dirName,
                                                          'rawMCPU.root']))
        self.enable = True
        ## if component is embed return (has no trigger obj)
        if self.cfg_comp.isEmbed :
          self.cfg_comp.puFileMC   = None
          self.cfg_comp.puFileData = None
          
        if self.cfg_comp.isMC or self.cfg_comp.isEmbed:
            if not hasattr(self.cfg_comp,"puFileMC") or (self.cfg_comp.puFileMC is None and self.cfg_comp.puFileData is None):
                self.enable = False
            else:
                assert( os.path.isfile(os.path.expandvars(self.cfg_comp.puFileMC)) )
                assert( os.path.isfile(os.path.expandvars(self.cfg_comp.puFileData)) )

                self.mcfile = TFile( self.cfg_comp.puFileMC )
                self.mchist = self.mcfile.Get('pileup')
                self.mchist.Scale( 1 / self.mchist.Integral() )

                self.datafile = TFile( self.cfg_comp.puFileData )
                self.datahist = self.datafile.Get('pileup')
                self.datahist.Scale( 1 / self.datahist.Integral() )
                # import pdb; pdb.set_trace()
                if self.mchist.GetNbinsX() != self.datahist.GetNbinsX():
                    raise ValueError('data and mc histograms must have the same number of bins')
                if self.mchist.GetXaxis().GetXmin() != self.datahist.GetXaxis().GetXmin():
                    raise ValueError('data and mc histograms must have the same xmin')
                if self.mchist.GetXaxis().GetXmax() != self.datahist.GetXaxis().GetXmax():
                    raise ValueError('data and mc histograms must have the same xmax')

    def declareHandles(self):
        super(PileUpAnalyzer, self).declareHandles()
        self.mchandles['pusi'] =  AutoHandle(
            'slimmedAddPileupInfo',
            'std::vector<PileupSummaryInfo>',
            fallbackLabel="addPileupInfo"
            ) 

        if self.allVertices == '_AUTO_':
            self.handles['vertices'] =  AutoHandle( "offlineSlimmedPrimaryVertices", 'std::vector<reco::Vertex>', fallbackLabel="offlinePrimaryVertices" ) 
        else:
            self.handles['vertices'] =  AutoHandle( self.allVertices, 'std::vector<reco::Vertex>' ) 

    def beginLoop(self, setup):
        super(PileUpAnalyzer,self).beginLoop(setup)
        self.averages.add('puWeight', Average('puWeight') )


    def process(self, event):
        self.readCollections( event.input )
        ## if component is embed return (has no trigger obj)
        if self.cfg_comp.isEmbed :
          return True

        event.puWeight = 1
        event.nPU = None
        event.pileUpVertex_z = []
        event.pileUpVertex_ptHat = []
        if self.cfg_comp.isMC:
            event.pileUpInfo = map( PileUpSummaryInfo,
                                    self.mchandles['pusi'].product() )
            for puInfo in event.pileUpInfo:
                if puInfo.getBunchCrossing()==0:
                    # import pdb; pdb.set_trace()
                    if self.cfg_ana.true is False:
                        event.nPU = puInfo.nPU()
                    else:
                        event.nPU = puInfo.nTrueInteractions()

                    if self.doHists:
                        self.rawmcpileup.hist.Fill( event.nPU )

                    ##get z position of on-time pile-up sorted by pt-hat
                    ptHat_zPositions = zip(puInfo.getPU_pT_hats(),puInfo.getPU_zpositions())
                    ptHat_zPositions.sort(reverse=True)
                    for ptHat_zPosition in ptHat_zPositions:
                        event.pileUpVertex_z.append(ptHat_zPosition[1])
                        event.pileUpVertex_ptHat.append(ptHat_zPosition[0])
            
            if event.nPU is None:
                raise ValueError('nPU cannot be None! means that no pu info has been found for bunch crossing 0.')
        elif self.cfg_comp.isEmbed:
            vertices = self.handles['vertices'].product()
            event.nPU = len(vertices)
        else:
            return True

        if self.enable:
            bin = self.datahist.FindBin(event.nPU)
            if bin<1 or bin>self.datahist.GetNbinsX():
                event.puWeight = 0
            else:
                data = self.datahist.GetBinContent(bin)
                mc = self.mchist.GetBinContent(bin)
                #Protect 0 division!!!!
                if mc !=0.0:
                    event.puWeight = data/mc
                else:
                    event.puWeight = 1
                
        event.eventWeight *= event.puWeight
        self.averages['puWeight'].add( event.puWeight )
        return True
        
    def write(self, setup):
        super(PileUpAnalyzer, self).write(setup)
        if self.cfg_comp.isMC and self.doHists:
            self.rawmcpileup.write()


setattr(PileUpAnalyzer,"defaultConfig", cfg.Analyzer(
    class_object = PileUpAnalyzer,
    true = True,  # use number of true interactions for reweighting
    makeHists=False
)
)

