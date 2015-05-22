
from CMGTools.RootTools.fwlite.Analyzer import Analyzer
from CMGTools.RootTools.fwlite.AutoHandle import AutoHandle
from CMGTools.RootTools.statistics.Average import Average
from CMGTools.RootTools.statistics.Histograms import Histograms
from CMGTools.RootTools.physicsobjects.PhysicsObjects import GenParticle,Jet, GenJet
from CMGTools.RootTools.utils.DeltaR import cleanObjectCollection, matchObjectCollection, matchObjectCollection2, deltaR2, deltaR
from CMGTools.RootTools.utils.PileupJetHistograms import PileupJetHistograms
## from CMGTools.RootTools.RootTools import loadLibs

from ROOT import TH1F, TH2F, TFile, THStack, TF1, TGraphErrors
import math

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

class ResolutionJetHistograms (Histograms) :
    ''' energy resolution as a function of the jet eta for different number of vertexes'''
    def __init__ (self, name, maxVtx = 50, vtxBinning = 5) :
        self.maxVtx = maxVtx
        self.vtxBinning = vtxBinning
        self.listLen = int (self.maxVtx) / int (self.vtxBinning)
        self.histosEta = []
        self.histosPt = []
        for i in range (self.listLen) : 
            self.histosEta.append (TH2F (name + '_h_dpt_eta_' + str (i), '', 24, -6, 6, 200, -2, 6))
            self.histosPt.append (TH2F (name + '_h_dpt_pt_' + str (i), '', 20, 0, 200, 200, -2, 6))
        super (ResolutionJetHistograms, self).__init__ (name)

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def fillJet (self, jet, nVtx) :
        if nVtx < self.maxVtx : 
            index = int (nVtx) / int (self.vtxBinning)
            self.histosEta[index].Fill (jet.gen.eta (), jet.pt () / jet.gen.pt ())
            self.histosPt[index].Fill (jet.gen.pt (), jet.pt () / jet.gen.pt ())
        else : print 'the vertex number: ' + str (nVtx) + ' is too high'

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def fillEvent (self, jets, nVtx) :
        for jet in jets :
            self.fillJet (jet, nVtx)

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def summary (self) :
        '''to be run after the event loop, before saving'''
        self.ptResolutions = []
        self.etaResolutions = []
        for i in range (self.listLen) :
            self.ptResolutions.append (self.GetSigmaGraph (self.histosPt[i]))
            self.etaResolutions.append (self.GetSigmaGraph (self.histosEta[i]))

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def Write (self, dir) :
        '''overloads mother function, to save lists contents'''
        self.dir = dir.mkdir( self.name )
        self.dir.cd ()
        for i in range (len (self.ptResolutions)) : self.ptResolutions[i].Write ('pt_graph_' + str (i))
        for i in range (len (self.etaResolutions)) : self.etaResolutions[i].Write ('eta_graph_' + str (i))
        for i in range (len (self.histosPt)) : self.histosPt[i].Write ()
        for i in range (len (self.histosEta)) : self.histosEta[i].Write ()
        dir.cd ()

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def doubleFit (self, histo, k = 1.) :            
        '''double iterative gaussian fit'''
        #FIXME put in into an external module
        min = histo.GetMean () - 1.5 * k * histo.GetRMS () # FIXME 1.5 is maybe useless?
        max = histo.GetMean () + 1.5 * k * histo.GetRMS () # FIXME 1.5 is maybe useless?
        self.func = TF1 ('gauss','gaus', 0, 2) #FIXME do I want "self" here? the variable is local...
        histo.Fit (self.func, 'Q', '', min, max)
        min = self.func.GetParameter (1) - k * self.func.GetParameter (2)
        max = self.func.GetParameter (1) + k * self.func.GetParameter (2)
        histo.Fit (self.func, '+Q', '', min, max)
        return [self.func.GetParameter (1), self.func.GetParameter (2), self.func.GetParError (1), self.func.GetParError (2)] #FIXME would it be better to return a tuple?

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def GetSigmaGraph (self, histo) :
        '''more sophisticated FitSlicesY'''
        # if the first gaus fit is not easy, I could try to start from the TProfile of the TH2F
        # or from the default FitSlicesY
        #FIXME put in into an external module
        # create the resulting histo
        self.resol_vs_var = TGraphErrors ()
        self.minEntriesNum = 10 #FIXME want it configurable

        # get the sigmas
        for xBin in range (1, histo.GetNbinsX ()) :
            # get a slice of the histogram
            #FIXME do I want the slide to be integrated possibly on more than a single bin?
            aSlice = histo.ProjectionY ('temp', xBin, xBin, 'e') 
            if aSlice.GetEntries () < self.minEntriesNum : 
                self.resol_vs_var.SetPoint (xBin, histo.GetXaxis ().GetBinCenter (xBin), 0.)
                self.resol_vs_var.SetPointError (xBin, 0., 0.)
            else :                
                # fit the slice w/a gaussian within the range of the gaussian
                res = self.doubleFit (aSlice)
                # get the sigma
                self.resol_vs_var.SetPoint (xBin, histo.GetXaxis ().GetBinCenter (xBin), res[1])
                self.resol_vs_var.SetPointError (xBin, 0., res[3])
        return self.resol_vs_var


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


# what the jet components are
# http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/UserCode/CMG/AnalysisDataFormats/CMGTools/interface/PFJet.h?revision=1.4&view=markup

#    enum ParticleType {
#              X=0,     // undefined
#                    h,       // charged hadron
#                    e,       // electron
#                    mu,      // muon
#                    gamma,   // photon
#                    h0,      // neutral hadron
#                    h_HF,        // HF tower identified as a hadron
#                    egamma_HF    // HF tower identified as an EM particle
#                  };


class FractionJetHistograms (Histograms) :
    '''eta distribution of the energy fraction per component'''
    def __init__ (self, name) :
        self.histos = []
        for i in range (8) : # NB here we start from 0 on purpose, for simplicity
            self.histos.append (TH2F (name + '_' + str (len (self.histos)), '', 240, -6, 6, 100, 0, 1))
        super (FractionJetHistograms, self).__init__ (name)

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def fillJet (self, jet) :
        try:
            for i in range (1, 8) :
                self.histos[i].Fill (jet.eta (), jet.component (i).fraction ())
        except:
            pass
            
# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def fillEvent (self, jets) :
        for jet in jets:
            self.fillJet (jet)

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def summary (self) :
        '''to be run after the event loop, before saving'''
        self.summ = THStack (self.name + '_summ', 'total energy')
        self.mean = []
        for i in range (1, 8) :
#            self.mean.append (self.histos[i].ProfileX ())
            self.mean.append (self.fromProfileToHisto (self.histos[i].ProfileX (), 10 + i * 2))
            self.summ.Add (self.mean[len (self.mean) - 1])

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def fromProfileToHisto (self, profile, color = 0) :
        thename = profile.GetName ().replace('pfx', 'ave')
        histo = TH1F (thename, '', profile.GetNbinsX (), profile.GetXaxis ().GetXmin (), profile.GetXaxis ().GetXmax ())
        histo.SetFillColor (color)
        for iBin in range (1, profile.GetNbinsX () + 1) :
            histo.SetBinContent (iBin, profile.GetBinContent (iBin))
            histo.SetBinError (iBin, profile.GetBinError (iBin))
        return histo

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def Write (self, dir) :
        '''overloads mother function, to save lists contents'''
        self.dir = dir.mkdir( self.name )
        self.dir.cd ()
        self.summ.Write ()
        for i in range (1, 8) :
            self.histos[i].Write ()
            self.mean[i-1].Write ()
        dir.cd ()


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


class JetHistograms (Histograms):
    '''general histograms for jets'''
    def __init__ (self, name) :
        self.h_pt = TH1F (name + '_h_pt', '', 100, 0, 200)
        self.h_genpt = TH1F (name + '_h_genpt', '', 100, 0, 200)
        self.h_geneta = TH1F (name + '_h_geneta', '', 240, -6, 6)
        self.h_dpt = TH1F (name + '_h_dpt', '', 200, -2, 6)
        self.h_eta = TH1F (name + '_h_eta', '', 240, -6, 6)
        self.h_comp = TH1F (name + '_h_comp', '', 10, 0, 10)
        self.h_deltaEleMatch = TH1F (name + '_h_deltaEleMatch', '', 1000, 0, 6)
        self.h_deltaJetMatch = TH1F (name + '_h_deltaJetMatch', '', 1000, 0, 6)
        self.h_numGen_numReco = TH2F (name + '_h_numGen_numReco', '', 20, 0, 20, 20, 0, 20)
        self.h_dpt_pt = TH2F (name + '_h_dpt_pt', '', 100, 0, 200, 200, -2, 6)
        self.h_dpt_eta = TH2F (name + '_h_dpt_eta', '', 240, -6, 6, 200, -2, 6)
        self.h_phi_eta = TH2F (name + '_h_phi_eta', '', 240, -6, 6, 360, -3.14, 3.14)
        self.h_dpt_dR2 = TH2F (name + '_h_dpt_dR2', '', 100, 0, 6, 200, -2, 6)
        self.h_ptr_ptg = TH2F (name + '_h_ptr_ptg', '', 100, 0, 200, 100, 0, 200)
        self.h_dR2_ptr = TH2F (name + '_h_dR2_ptr', '', 100, 0, 200, 100, 0, 6)
        self.h_dR2_eta = TH2F (name + '_h_dR2_eta', '', 240, -6, 6, 100, 0, 6)
        self.h_frac_com = TH2F (name + '_h_frac_com', '', 8, 0, 8, 10, 0, 1) # fraction, component
        super (JetHistograms, self).__init__ (name) #FIXME check that the super has to be called within __init__

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def fillJet (self, jet) :
        # import pdb
        # pdb.set_trace ()
        self.fillFrac (jet)
        self.h_pt.Fill (jet.pt ())
        if jet.pt () > 10 : self.h_eta.Fill (jet.eta ())
        self.h_phi_eta.Fill (jet.eta (), jet.phi ()) 
        if hasattr (jet, 'gen') and jet.gen is not None:
            dR2 = deltaR2 (jet.gen.eta (), jet.gen.phi (), jet.eta (), jet.phi ())
            self.h_deltaJetMatch.Fill (dR2)
            self.h_dpt_dR2.Fill (dR2, jet.pt () / jet.gen.pt ())
            self.h_dR2_ptr.Fill (jet.gen.pt (), dR2)
            self.h_dR2_eta.Fill (jet.gen.eta (), dR2)
            if dR2 < 0.3 :
                self.h_genpt.Fill (jet.gen.pt ())
                self.h_geneta.Fill (jet.gen.pt ())
                self.h_dpt.Fill (jet.pt () / jet.gen.pt ())
                self.h_dpt_pt.Fill (jet.gen.pt (), jet.pt () / jet.gen.pt ()) 
                self.h_dpt_eta.Fill (jet.gen.eta (), jet.pt () / jet.gen.pt ()) 
                self.h_ptr_ptg.Fill (jet.gen.pt (), jet.pt ()) 

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def fillFrac (self, jet) :
        try:
            for i in range (1, 8) :
                self.h_frac_com.Fill (i, jet.component (i).fraction ())
        except:
            pass
# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def fillEvent (self, jets) :
        for jet in jets :
            self.fillJet (jet)

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def fillStats (self, ngj, nrj) :
        self.h_numGen_numReco.Fill (ngj, nrj)

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def fillEleMatch (self, delta) :
        self.h_deltaEleMatch.Fill (delta)

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def summary (self) :
        pass


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


class SimpleJetAnalyzer (Analyzer) :
    '''A simple jet analyzer for Pietro.'''
    ### def __init__(self,cfg_ana, cfg_comp, looperName):
    ###     loadLibs()
    ###     super (SimpleJetAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)

    def declareHandles (self) :
        super (SimpleJetAnalyzer, self).declareHandles ()
        self.handles['jets'] =  AutoHandle (
            *self.cfg_ana.jetCollection
            )
        if self.cfg_ana.useGenLeptons: 
            self.mchandles['genParticlesPruned'] =  AutoHandle (
                *self.cfg_ana.GenParticlesCollection
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
            *self.cfg_ana.VtxCollection
           )

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def beginLoop (self) :
        super (SimpleJetAnalyzer,self).beginLoop ()
        self.file = TFile ('/'.join ([self.looperName, 'testJets.root']),
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

        # general histograms
        self.jetHistos = JetHistograms ('Jets')
        self.cleanJetHistos = JetHistograms ('CleanJets')
        self.matchedCleanJetHistos = JetHistograms ('MatchedCleanJets')
        self.matchedCleanJetHistos_barrel = JetHistograms ('MatchedCleanJets_barrel')
        self.matchedCleanJetHistos_endtk = JetHistograms ('MatchedCleanJets_endtk')
        self.matchedCleanJetHistos_endNOtk = JetHistograms ('MatchedCleanJets_endNOtk')
        self.matchedCleanJetHistos_fwd = JetHistograms ('MatchedCleanJets_fwd')
        self.LPtmatchedCleanJetHistos = JetHistograms ('LPtMatchedCleanJets') 
        self.HPtmatchedCleanJetHistos = JetHistograms ('HPtMatchedCleanJets')
        self.unmatchedCleanJetHistos = JetHistograms ('UnmatchedCleanJets')
        self.LPtUnmatchedCleanJetHistos = JetHistograms ('LPtUnmatchedCleanJets') 
        self.HPtUnmatchedCleanJetHistos = JetHistograms ('HPtUnmatchedCleanJets')

        # histograms of the components fraction
        self.matchedCleanJetHistosComponents = FractionJetHistograms ('MatchedCleanJetsCompontents')
        self.unmatchedCleanJetHistosComponents = FractionJetHistograms ('UnmatchedCleanJetsCompontents')

        # histograms for the resolution of matched jets
        self.matchedCleanJetHistosResolution = ResolutionJetHistograms ('MatchedCleanJetsResolution', 50, 1)
        self.matchedCleanJetHistosResolution_barrel = ResolutionJetHistograms ('MatchedCleanJetsResolution_barrel', 50, 1)
        self.matchedCleanJetHistosResolution_endtk = ResolutionJetHistograms ('MatchedCleanJetsResolution_endtk', 50, 1)
        self.matchedCleanJetHistosResolution_endNOtk = ResolutionJetHistograms ('MatchedCleanJetsResolution_endNOtk', 50, 1)
        self.matchedCleanJetHistosResolution_fwd = ResolutionJetHistograms ('MatchedCleanJetsResolution_fwd', 50, 1)

        print 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
        self.doJetIdHisto = False
        if self.cfg_ana.doJetIdHisto:
            print 'doing jet ID'
            # histograms for pileup jet identification variables
            self.vtxBins   = (0,5,10,15,20,30) ## (0,2,4,6,10,15,20,30,35)
            self.ptBins    = (20,30,50) ## (20,30,40,50,100)
            self.etaBins   = (0,1.4,2.5,3.0)
            self.puEtaLables = ["_barrel","_endtk","_endNOtk","_fwd"]
            reweight_f = TF1("f","pol2(0)+expo(3)")
            reweight_f.SetParameters(0.1955298,-0.003830591,1.944794e-05,4.649755,-0.1722024)
            self.reweight = ("pt", reweight_f)
            self.doJetIdHisto = True
            self.gluCleanHistosId = PileupJetHistograms("GluonMatchedCleanHistosId",self.vtxBins,self.ptBins,self.etaBins,etalabels=self.puEtaLables,reweight=self.reweight,
                                                        jetIdMva=self.cfg_ana.jetIdMva)
            self.quarkCleanHistosId = PileupJetHistograms("QuarkMatchedCleanHistosId",self.vtxBins,self.ptBins,self.etaBins,etalabels=self.puEtaLables,reweight=self.reweight,
                                                          jetIdMva=self.cfg_ana.jetIdMva)
            self.reweiMatchedCleanHistosId = PileupJetHistograms("ReweiMatchedCleanHistosId",self.vtxBins,self.ptBins,self.etaBins,etalabels=self.puEtaLables,reweight=self.reweight,
                                                                 jetIdMva=self.cfg_ana.jetIdMva)
            self.unmatchedCleanHistosId = PileupJetHistograms("UnmatchedCleanHistosId",self.vtxBins,self.ptBins,self.etaBins,etalabels=self.puEtaLables,
                                                              jetIdMva=self.cfg_ana.jetIdMva)
            
        self.h_nvtx = TH1F ("h_nvtx", "" ,50, 0, 50)
        self.h_genjetspt = TH1F ("h_genjetspt", "" ,500, 0, 500)
        self.h_secondClosestVsPtratio = TH2F ("h_secondClosestVsPtratio", "" ,100, 0, 2, 100, 0, 6)
        self.h_avedistanceVSNvtx = TH2F ("h_avedistanceVSNvtx", "" ,50, 0, 50, 100, 0, 6)
        self.h_PTRatioVSgenEta = TH2F ("h_PTRatioVSgenEta", "" ,150, -5, 5, 100, 0, 2)
        self.h_PTRatioVSgenPt = TH2F ("h_PTRatioVSgenPt", "" ,200, 0, 100, 100, 0, 2)
        self.h_matchDR = TH1F ("h_matchDR", "" ,60, 0, 0.30)
        self.h_relPtVSmatchDR = TH2F ("h_relPtVSmatchDR", "" ,60, 0, 0.30, 100, 0, 2)
        self.h_relPtVSchFrac = TH2F ("h_relPtVSchFrac", "" ,100, 0, 1, 100, 0, 2)

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....
    def process (self, iEvent, event) :
        #read all the handles defined beforehand
        self.readCollections (iEvent)
        
        jetEtaCut = 4.5 
        
        # get the vertexes
        event.vertices = self.handles['vertices'].product ()
        self.h_nvtx.Fill (len (event.vertices))
        event.vertexBin = int (len (event.vertices))
        
        # get the jets in the jets variable
        jets = self.handles['jets'].product ()
        # filter jets with some selections
        event.jets = [ jet for jet in jets if ( abs(jet.eta()) < jetEtaCut and jet.pt()>self.cfg_ana.ptCut and self.isPFLoose(jet) ) ]
        self.jetHistos.fillEvent (event.jets)
        
        # get status 2 leptons
        if 'genParticlesPruned' in self.mchandles:
            event.genLeptons = [ lep for lep in self.mchandles['genParticlesPruned'].product() if lep.status() == 3 and (abs(lep.pdgId()) == 11 or abs(lep.pdgId()) == 13 or abs(lep.pdgId()) == 15) ]
        else:
            event.genLeptons = [ lep for lep in self.mchandles['genParticles'].product() if lep.status() == 3 and (abs(lep.pdgId()) == 11 or abs(lep.pdgId()) == 13 or abs(lep.pdgId()) == 15) ]  
# @ Pasquale: why level 3 and not level 2?
#        event.selGenLeptons = [GenParticle (lep) for lep in event.genLeptons if (lep.pt ()>self.cfg_ana.ptCut and abs (lep.eta ()) < jetEtaCut)]
        
        # get genJets
        event.genJets = map (GenJet, self.mchandles['genJets'].product ())
        # filter genjets as for reco jets
        event.myGenJets = [GenJet (jet) for jet in event.genJets if (jet.pt ()>self.cfg_ana.genPtCut)]
        event.selGenJets = cleanObjectCollection (event.myGenJets, event.genLeptons, 0.2)
        # event.selGenJets = event.genJets
        for jet in event.selGenJets : 
            self.h_genjetspt.Fill (jet.pt ())
        
        event.noNegJets = [ jet for jet in event.jets if (jet.jecFactor(0) > 0) ]
#        event.noNegJets = [ jet for jet in event.jets]
        
        # first stats plots
        # print 'genLeptons : ' + repr (len (event.genLeptons)) + ' | genJets : ' + repr (len (event.genJets)) + ' | recoJets : ' + repr (len (event.jets))
        self.jetHistos.fillStats (len (event.selGenJets), len (event.noNegJets))
        
        #FIXME why are there cases in which there's 4 or 6 leptons?
        if len (event.genLeptons) > 2 :
            return
        # in case I want to filter out taus
        # 11, 13, 15 : e, u, T
#        event.genOneLepton = [GenParticle (part) for part in event.genLeptons if abs (part.pdgId ()) == 15]
        # remove leptons from jets if closer than 0.2
        event.cleanJets = cleanObjectCollection (event.noNegJets, event.genLeptons, 0.2)
#        event.cleanJets = event.noNegJets
        self.cleanJetHistos.fillEvent (event.cleanJets)
        
#        print len (jets),len (event.jets), len (event.noNegJets), len (event.cleanJets), len (event.genLeptons),"-->",(len (event.noNegJets) - len (event.cleanJets) - len (event.genLeptons))

        event.matchingCleanJets = matchObjectCollection2 (event.cleanJets, event.selGenJets, 0.25)
        # assign to each jet its gen match (easy life :))
        for jet in event.cleanJets :
            jet.gen = event.matchingCleanJets[ jet ]
        # FIXME next step might be to put this in the matching and remove the boolean flags

        event.matchedCleanJets = [jet for jet in event.matchingCleanJets if jet.gen != None]
        event.cleanGluJets = []
        event.cleanQuarkJets = []
        for jet in event.matchedCleanJets:
            flav = abs(jet.partonFlavour()) 
            if flav == 21:
                event.cleanGluJets.append(jet)
            elif flav > 0 and flav <= 3:
                event.cleanQuarkJets.append(jet)

        event.LPtmatchedCleanJets = [jet for jet in event.matchingCleanJets if jet.gen != None and jet.pt () <= 30]
        event.HPtmatchedCleanJets = [jet for jet in event.matchingCleanJets if jet.gen != None and jet.pt () > 30]

        self.matchedCleanJetHistos.fillEvent (event.matchedCleanJets)
        self.LPtmatchedCleanJetHistos.fillEvent (event.LPtmatchedCleanJets)
        self.HPtmatchedCleanJetHistos.fillEvent (event.HPtmatchedCleanJets)

        event.unmatchedCleanJets = [jet for jet in event.matchingCleanJets if jet.gen == None]
        event.LPtunmatchedCleanJets = [jet for jet in event.matchingCleanJets if jet.gen == None and jet.pt () <= 30]
        event.HPtunmatchedCleanJets = [jet for jet in event.matchingCleanJets if jet.gen == None and jet.pt () > 30]
        
        self.unmatchedCleanJetHistos.fillEvent (event.unmatchedCleanJets)
        self.LPtUnmatchedCleanJetHistos.fillEvent (event.LPtunmatchedCleanJets)
        self.HPtUnmatchedCleanJetHistos.fillEvent (event.HPtunmatchedCleanJets)
        
        self.matchedCleanJetHistosComponents.fillEvent (event.matchedCleanJets)
        
        self.unmatchedCleanJetHistosComponents.fillEvent (event.unmatchedCleanJets)
        
        self.matchedCleanJetHistosResolution.fillEvent (event.matchedCleanJets, len (event.vertices))
        
        for jet in event.matchedCleanJets :
            if abs (jet.gen.eta ()) < 1.4 :
                self.matchedCleanJetHistosResolution_barrel.fillJet (jet, len (event.vertices))
                self.matchedCleanJetHistos_barrel.fillJet (jet)
            elif 1.6 < abs (jet.gen.eta ()) < 2.5 :    
                self.matchedCleanJetHistosResolution_endtk.fillJet (jet, len (event.vertices))
                self.matchedCleanJetHistos_endtk.fillJet (jet)
            elif 2.6 < abs (jet.gen.eta ()) < 2.9 :    
                self.matchedCleanJetHistosResolution_endNOtk.fillJet (jet, len (event.vertices))
                self.matchedCleanJetHistos_endNOtk.fillJet (jet)
            elif 3.1 < abs (jet.gen.eta ()) :    
                self.matchedCleanJetHistosResolution_fwd.fillJet (jet, len (event.vertices))
                self.matchedCleanJetHistos_fwd.fillJet (jet)

        ##PG debugging for tails
        #for jet in event.matchedCleanJets :
            #deltaRR = deltaR( jet.eta (), jet.phi (), jet.gen.eta (), jet.gen.phi ())
            #self.h_matchDR.Fill (deltaRR)
            #self.h_relPtVSmatchDR.Fill (deltaRR, jet.pt () / jet.gen.pt ())
            #if abs (jet.gen.eta ()) > 2.5 and abs (jet.gen.eta ()) < 3 :
                #self.h_relPtVSchFrac.Fill (jet.chargedHadronEnergyFraction (), jet.pt () / jet.gen.pt ())

            #if jet.gen.pt () > 20 and jet.gen.pt () < 40 :
                #self.h_PTRatioVSgenEta.Fill (jet.gen.eta (), jet.pt () / jet.gen.pt ())
            #if abs (jet.gen.eta ()) > 1.6 :
                #self.h_PTRatioVSgenPt.Fill (jet.gen.pt (), jet.pt () / jet.gen.pt ())

            #minDelta = 10
            #secondClosest = jet
            #for recojet in event.cleanJets :
                #if recojet == jet : continue
                #dr2 = deltaR2( jet.gen.eta (), jet.gen.phi (), recojet.eta (), recojet.phi ())
                #if dr2 < minDelta :
                    #minDelta = dr2
                    #secondClosest = recojet 
            ##if len(event.vertices) < 10 or abs (jet.gen.eta ()) < 1.6: continue
            #self.h_secondClosestVsPtratio.Fill (jet.pt () / jet.gen.pt (), math.sqrt (minDelta))
            #if jet.pt () / jet.gen.pt () < 0.2 and jet.gen.pt () > 20 and abs (jet.gen.eta ()) < 3 and abs (jet.gen.eta ()) > 2.5 :
                #print '============',len(event.genLeptons)
                #print jet.pt (), jet.eta (), jet.phi (), jet.jecFactor (0)
                #print jet.gen.pt (), jet.gen.eta (), jet.gen.phi ()
                #print '------------ leptons:'
                #for lept in event.genLeptons :
                    #print lept.pt (), lept.eta (), lept.phi ()
                #print '------------'
                #for recojet in event.cleanJets :
                    #print "RECO",recojet.pt (), recojet.eta (), recojet.phi (), recojet.jecFactor (0)
                #for genjet in event.selGenJets :
                    #print "GEN ",genjet.pt (), genjet.eta (), genjet.phi ()

        #aveDeltaR = 0
        #num = 0
        #for recojet1 in event.cleanJets :
            #minDelta = 10
            #closest = recojet1
            #for recojet2 in event.cleanJets :
                #if recojet1 == recojet2 : continue
                #dr2 = deltaR2( recojet1.eta (), recojet1.phi (), recojet2.eta (), recojet2.phi ())
                #if dr2 < minDelta :
                    #minDelta = dr2
                    #closest = recojet2
            #if minDelta == 10 : continue
            #aveDeltaR = aveDeltaR + math.sqrt (minDelta)
            #num = num + 1
        #if num > 0 :
            #aveDeltaR = aveDeltaR / num
            #self.h_avedistanceVSNvtx.Fill (len(event.vertices), aveDeltaR)


        if self.doJetIdHisto:
            self.gluCleanHistosId.fillEvent(event.cleanGluJets,event.vertices)
            self.quarkCleanHistosId.fillEvent(event.cleanQuarkJets,event.vertices)
            ### self.matchedCleanHistosId.fillEvent(event.matchedCleanJets,event.vertices)
            self.reweiMatchedCleanHistosId.fillEvent(event.matchedCleanJets,event.vertices)
            self.unmatchedCleanHistosId.fillEvent(event.unmatchedCleanJets,event.vertices)
        

# .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... ....

    def write (self):
        from ROOT import gROOT
        gROOT.SetBatch(True)
        self.jetHistos.Write (self.file)
        self.cleanJetHistos.Write (self.file)
        self.matchedCleanJetHistos.Write (self.file)
        self.matchedCleanJetHistos_barrel.Write (self.file)
        self.matchedCleanJetHistos_endtk.Write (self.file)
        self.matchedCleanJetHistos_endNOtk.Write (self.file)
        self.matchedCleanJetHistos_fwd.Write (self.file)
        
        self.LPtmatchedCleanJetHistos.Write (self.file)
        self.HPtmatchedCleanJetHistos.Write (self.file)
        self.LPtUnmatchedCleanJetHistos.Write (self.file)
        self.HPtUnmatchedCleanJetHistos.Write (self.file)

        self.unmatchedCleanJetHistos.Write (self.file)

        self.matchedCleanJetHistosComponents.summary ()
        self.matchedCleanJetHistosComponents.Write (self.file)
        self.unmatchedCleanJetHistosComponents.summary ()
        self.unmatchedCleanJetHistosComponents.Write (self.file)

        self.matchedCleanJetHistosResolution.summary ()
        self.matchedCleanJetHistosResolution.Write (self.file)
        
        self.matchedCleanJetHistosResolution_barrel.summary ()
        self.matchedCleanJetHistosResolution_barrel.Write (self.file)
        
        self.matchedCleanJetHistosResolution_endtk.summary ()
        self.matchedCleanJetHistosResolution_endtk.Write (self.file)
        
        self.matchedCleanJetHistosResolution_endNOtk.summary ()
        self.matchedCleanJetHistosResolution_endNOtk.Write (self.file)
        
        self.matchedCleanJetHistosResolution_fwd.summary ()
        self.matchedCleanJetHistosResolution_fwd.Write (self.file)

        if self.doJetIdHisto:
            self.gluCleanHistosId.summary()
            self.gluCleanHistosId.Write(self.file)
            
            self.quarkCleanHistosId.summary()
            self.quarkCleanHistosId.Write(self.file)
            
            self.reweiMatchedCleanHistosId.summary()
            self.reweiMatchedCleanHistosId.Write(self.file)
        
            ### self.matchedCleanHistosId.Write(self.file)
            self.unmatchedCleanHistosId.Write(self.file)
        
        self.file.cd ()
        self.h_nvtx.Write ()
        self.h_genjetspt.Write ()
        self.h_secondClosestVsPtratio.Write ()
        self.h_avedistanceVSNvtx.Write ()
        self.h_PTRatioVSgenEta.Write ()
        self.h_PTRatioVSgenPt.Write ()
        self.h_matchDR.Write ()
        self.h_relPtVSmatchDR.Write ()
        self.h_relPtVSchFrac.Write ()
        
        self.file.Close()
        
