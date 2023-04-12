import FWCore.ParameterSet.Config as cms
import copy

from PhysicsTools.NanoAOD.nanoDQM_cfi import nanoDQM
from PhysicsTools.NanoAOD.nanoDQM_tools_cff import *
from PhysicsTools.NanoAOD.nano_eras_cff import *

nanojmeDQM = nanoDQM.clone()

#============================================
#
# Add more variables for AK4 Puppi jets
#
#============================================
_ak4puppiplots = cms.VPSet(
    Count1D('_size', 20, -0.5, 19.5, 'AK4 PF Puppi jets with JECs applied.')
)
for plot in nanojmeDQM.vplots.Jet.plots:
    if plot.name.value()=="_size": continue
    _ak4puppiplots.append(plot)

_ak4puppiplots.extend([
    Plot1D('nConstChHads','nConstChHads', 10,  0, 40,'number of charged hadrons in the jet'),
    Plot1D('nConstNeuHads','nConstNeuHads', 10,  0, 40,'number of neutral hadrons in the jet'),
    Plot1D('nConstPhotons','nConstPhotons', 10,  0, 40,'number of photons in the jet'),
    Plot1D('nConstElecs','nConstElecs', 5, 0, 10,'number of electrons in the jet'),
    Plot1D('nConstMuons','nConstMuons', 5, 0, 10,'number of muons in the jet'),
    Plot1D('nConstHFEMs','nConstHFEMs', 5, 0, 10,'number of HF EMs in the jet'),
    Plot1D('nConstHFHads','nConstHFHads', 5,  0, 10,'number of HF Hadrons in the jet'),
    Plot1D('puId_dR2Mean','puId_dR2Mean',20, 0, 0.2,"pT^2-weighted average square distance of jet constituents from the jet axis (PileUp ID BDT input variable)"),
    Plot1D('puId_majW','puId_majW',10, 0, 0.5, "major axis of jet ellipsoid in eta-phi plane (PileUp ID BDT input variable)"),
    Plot1D('puId_minW','puId_minW',10, 0, 0.5, "minor axis of jet ellipsoid in eta-phi plane (PileUp ID BDT input variable)"),
    Plot1D('puId_frac01','puId_frac01',10, 0, 1, "fraction of constituents' pT contained within dR <0.1 (PileUp ID BDT input variable)"),
    Plot1D('puId_frac02','puId_frac02',10, 0, 1, "fraction of constituents' pT contained within 0.1< dR <0.2 (PileUp ID BDT input variable)"),
    Plot1D('puId_frac03','puId_frac03',10, 0, 1, "fraction of constituents' pT contained within 0.2< dR <0.3 (PileUp ID BDT input variable)"),
    Plot1D('puId_frac04','puId_frac04',10, 0, 1, "fraction of constituents' pT contained within 0.3< dR <0.4 (PileUp ID BDT input variable)"),
    Plot1D('puId_ptD','puId_ptD',10, 0, 1, "pT-weighted average pT of constituents (PileUp ID BDT input variable)"),
    Plot1D('puId_beta','puId_beta',10, 0, 1, "fraction of pT of charged constituents associated to PV (PileUp ID BDT input variable)"),
    Plot1D('puId_pull','puId_pull',10, 0, 0.05, "magnitude of pull vector (PileUp ID BDT input variable)"),
    Plot1D('puId_jetR','puId_jetR',10, 0, 1, "fraction of jet pT carried by the leading constituent (PileUp ID BDT input variable)"),
    Plot1D('puId_jetRchg','puId_jetRchg',10, 0, 1, "fraction of jet pT carried by the leading charged constituent (PileUp ID BDT input variable)"),
    Plot1D('puId_nCharged','puId_nCharged',10, 0, 40, "number of charged constituents (PileUp ID BDT input variable)"),
    Plot1D('qgl_axis2','qgl_axis2',10, 0, 0.4, "ellipse minor jet axis (Quark vs Gluon likelihood input variable)"),
    Plot1D('qgl_ptD','qgl_ptD',10, 0, 1, "pT-weighted average pT of constituents (Quark vs Gluon likelihood input variable)"),
    Plot1D('qgl_mult','qgl_mult', 10, 0, 50, "PF candidates multiplicity (Quark vs Gluon likelihood input variable)"),
    Plot1D('btagDeepFlavG','btagDeepFlavG',20, -1, 1, "DeepFlavour gluon tag raw score"),
    Plot1D('btagDeepFlavUDS','btagDeepFlavUDS',20, -1, 1, "DeepFlavour uds tag raw score"),
    Plot1D('particleNetAK4_B','particleNetAK4_B',20, -1, 1, "ParticleNetAK4 tagger b vs all (udsg, c) discriminator"),
    Plot1D('particleNetAK4_CvsL','particleNetAK4_CvsL',20, -1, 1,"ParticleNetAK4 tagger c vs udsg discriminator"),
    Plot1D('particleNetAK4_CvsB','particleNetAK4_CvsB',20, -1, 1,"ParticleNetAK4 tagger c vs b discriminator"),
    Plot1D('particleNetAK4_QvsG','particleNetAK4_QvsG',20, -1, 1,"ParticleNetAK4 tagger uds vs g discriminator"),
    Plot1D('particleNetAK4_G','particleNetAK4_G',20, -1, 1, "ParticleNetAK4 tagger g raw score"),
    Plot1D('particleNetAK4_puIdDisc','particleNetAK4_puIdDisc',20, -1, 1,"ParticleNetAK4 tagger pileup jet discriminator"),
    Plot1D('hfEmEF','hfEmEF', 20, 0, 1,'electromagnetic energy fraction in HF'),
    Plot1D('hfHEF','hfHEF', 20, 0, 1,'hadronic energy fraction in HF'),
])

#============================================
#
# Setup for AK4 CHS jets
#
#============================================
_ak4chsplots = cms.VPSet(
    Count1D('_size', 20, -0.5, 19.5, 'AK4 PF CHS jets with JECs applied.')
)
for plot in _ak4puppiplots:
    if plot.name.value()=="_size": continue
    _ak4chsplots.append(plot)
    _ak4chsplots.extend([
        Plot1D('chFPV1EF', 'chFPV1EF', 20, 0, 2, 'charged fromPV==1 Energy Fraction (component of the total charged Energy Fraction).'),
        Plot1D('chFPV2EF', 'chFPV2EF', 20, 0, 2, 'charged fromPV==2 Energy Fraction (component of the total charged Energy Fraction).'),
        Plot1D('chFPV3EF', 'chFPV3EF', 20, 0, 2, 'charged fromPV==3 Energy Fraction (component of the total charged Energy Fraction).'),
    ])

#============================================
#
# Setup all extra AK4 collections. Will remove
# collection depending on era.
#
#============================================
nanojmeDQM.vplots.Jet.plots = _ak4puppiplots #Puppi is default "Jet collection" for Run-3
nanojmeDQM.vplots.JetPuppi = cms.PSet( # This is for the Run-2 extra "JetPuppi" collection
    sels = nanojmeDQM.vplots.Jet.sels,
    plots = _ak4puppiplots
)
nanojmeDQM.vplots.JetCHS = cms.PSet( # This is for the Run-3 extra "JetCHS" collection
    sels = nanojmeDQM.vplots.Jet.sels,
    plots = _ak4chsplots
)



#============================================
#
# Add more variables for AK8 Puppi jets
#
#============================================
nanojmeDQM.vplots.FatJet.plots.extend([
    Plot1D('nConstChHads','nConstChHads',10,0,40,'number of charged hadrons in the jet'),
    Plot1D('nConstNeuHads','nConstNeuHads',10,0,40,'number of neutral hadrons in the jet'),
    Plot1D('nConstPhotons','nConstPhotons',10,0,40,'number of photons in the jet'),
    Plot1D('nConstElecs','nConstElecs',5,0,10,'number of electrons in the jet'),
    Plot1D('nConstMuons','nConstMuons',5,0,10,'number of muons in the jet'),
    Plot1D('nConstHFEMs','nConstHFEMs',5,0,10,'number of HF EMs in the jet'),
    Plot1D('nConstHFHads','nConstHFHads',5,0,10,'number of HF Hadrons in the jet'),
    Plot1D('neEmEF','neEmEF',20, 0, 1,'neutral Electromagnetic Energy Fraction'),
    Plot1D('neHEF','neHEF',20, 0, 1,'neutral Hadron Energy Fraction'),
])

#============================================
#
# Setup for AK8 Puppi jets for JEC studies
#
#============================================
nanojmeDQM.vplots.FatJetForJEC = cms.PSet(
    sels = cms.PSet(
        CentralPt30 = cms.string('abs(eta) < 2.4 && pt > 30'),
        ForwardPt30 = cms.string('abs(eta) > 2.4 && pt > 30')
    ),
    plots = cms.VPSet(
        Count1D('_size', 20, -0.5, 19.5, 'AK8 PF Puppi jets with JECs applied. Reclustered for JEC studies so only minimal info stored.'),
        Plot1D('area', 'area', 20, 0.2, 0.8, 'jet catchment area, for JECs'),
        Plot1D('eta', 'eta', 20, -6, 6, 'eta'),
        Plot1D('jetId', 'jetId', 8, -0.5, 7.5, 'Jet ID flags bit1 is loose (always false in 2017 since it does not exist), bit2 is tight, bit3 is tightLepVeto'),
        Plot1D('mass', 'mass', 20, 0, 200, 'mass'),
        Plot1D('phi', 'phi', 20, -3.14159, 3.14159, 'phi'),
        Plot1D('pt', 'pt', 20, 0, 400, 'pt'),
        Plot1D('rawFactor', 'rawFactor', 20, -0.5, 0.5, '1 - Factor to get back to raw pT'),
        Plot1D('nConstChHads','nConstChHads', 10,  0, 40,'number of charged hadrons in the jet'),
        Plot1D('nConstNeuHads','nConstNeuHads', 10,  0, 40,'number of neutral hadrons in the jet'),
        Plot1D('nConstPhotons','nConstPhotons', 10,  0, 40,'number of photons in the jet'),
        Plot1D('nConstElecs','nConstElecs', 5, 0, 10,'number of electrons in the jet'),
        Plot1D('nConstMuons','nConstMuons', 5, 0, 10,'number of muons in the jet'),
        Plot1D('nConstHFEMs','nConstHFEMs', 5, 0, 10,'number of HF EMs in the jet'),
        Plot1D('nConstHFHads','nConstHFHads', 5,  0, 10,'number of HF Hadrons in the jet'),
        Plot1D('nElectrons', 'nElectrons', 5, -0.5, 4.5, 'number of electrons in the jet'),
        Plot1D('nMuons', 'nMuons', 4, -0.5, 3.5, 'number of muons in the jet'),
        Plot1D('hadronFlavour', 'hadronFlavour', 6, -0.5, 5.5, 'flavour from hadron ghost clustering'),
        Plot1D('partonFlavour', 'partonFlavour', 40, -9.5, 30.5, 'flavour from parton matching'),
        Plot1D('chEmEF', 'chEmEF', 20, 0, 1, 'charged Electromagnetic Energy Fraction'),
        Plot1D('chHEF', 'chHEF', 20, 0, 2, 'charged Hadron Energy Fraction'),
        Plot1D('neEmEF', 'neEmEF', 20, 0.3, 0.4, 'neutral Electromagnetic Energy Fraction'),
        Plot1D('neHEF', 'neHEF', 20, 0.01, 0.2, 'neutral Hadron Energy Fraction'),
        Plot1D('hfEmEF', 'hfEmEF', 20, 0, 1, 'electromagnetic energy fraction in HF'),
        Plot1D('hfHEF', 'hfHEF', 20, 0, 1, 'hadronic energy fraction in HF'),
        Plot1D('muEF', 'muEF', 20, -1, 1, 'muon Energy Fraction'),
        NoPlot('genJetIdx'),
    ),
)

#============================================
#
# Setup for AK8 CHS jets
#
#============================================
_ak8chsplots = cms.VPSet(
    Count1D('_size', 20, -0.5, 19.5, 'AK8 CHS jets with JECs applied.')
)
for plot in nanojmeDQM.vplots.FatJetForJEC.plots:
    if plot.name.value()=="_size": continue
    _ak8chsplots.append(plot)

nanojmeDQM.vplots.FatJetCHS = cms.PSet(
    sels = nanojmeDQM.vplots.FatJetForJEC.sels,
    plots = _ak8chsplots,
)

#============================================
#
# Setup for AK4 Calo jets
#
#============================================
nanojmeDQM.vplots.JetCalo = cms.PSet(
    sels = cms.PSet(
        CentralPt30 = cms.string('abs(eta) < 2.4 && pt > 30'),
        ForwardPt30 = cms.string('abs(eta) > 2.4 && pt > 30')
    ),
    plots = cms.VPSet(
        Count1D('_size', 20, -0.5, 19.5, 'AK4 Calo jets (slimmedCaloJets)'),
        Plot1D('area', 'area', 20, 0.2, 0.8, 'jet catchment area'),
        Plot1D('eta', 'eta', 20, -6, 6, 'eta'),
        Plot1D('mass', 'mass', 20, 0, 200, 'mass'),
        Plot1D('phi', 'phi', 20, -3.14159, 3.14159, 'phi'),
        Plot1D('pt', 'pt', 20, 0, 400, 'pt'),
        Plot1D('rawFactor', 'rawFactor', 20, -0.5, 0.5, '1 - Factor to get back to raw pT'),
        Plot1D('emf', 'emf', 20, 0, 1, 'electromagnetic energy fraction'),
        Plot1D('hadronFlavour', 'hadronFlavour', 6, -0.5, 5.5, 'flavour from hadron ghost clustering'),
        Plot1D('partonFlavour', 'partonFlavour', 40, -9.5, 30.5, 'flavour from parton matching'),
        NoPlot('genJetIdx'),
    ),
)

##MC
nanojmeDQMMC = nanojmeDQM.clone()
#nanojmeDQMMC.vplots.Electron.sels.Prompt = cms.string("genPartFlav == 1")
nanojmeDQMMC.vplots.LowPtElectron.sels.Prompt = cms.string("genPartFlav == 1")
nanojmeDQMMC.vplots.Muon.sels.Prompt = cms.string("genPartFlav == 1")
nanojmeDQMMC.vplots.Photon.sels.Prompt = cms.string("genPartFlav == 1")
nanojmeDQMMC.vplots.Tau.sels.Prompt = cms.string("genPartFlav == 5")
nanojmeDQMMC.vplots.Jet.sels.Prompt = cms.string("genJetIdx != 1")
nanojmeDQMMC.vplots.Jet.sels.PromptB = cms.string("genJetIdx != 1 && hadronFlavour == 5")

#============================================
#
# Era dependent customization
#
#============================================
#
# Run 3
#
(~run2_nanoAOD_ANY).toModify(
    nanojmeDQM.vplots.Jet, 
    plots = _ak4puppiplots,
).toModify(
    nanojmeDQM.vplots, 
    JetPuppi = None # Remove "JetPuppi" from DQM
)
(~run2_nanoAOD_ANY).toModify(
    nanojmeDQMMC.vplots.JetCHS.sels,
    Prompt = nanojmeDQMMC.vplots.Jet.sels.Prompt,
    PromptB = nanojmeDQMMC.vplots.Jet.sels.PromptB
)
#
# Run 2
#
run2_nanoAOD_ANY.toModify(
    nanojmeDQM.vplots.Jet, 
    plots = _ak4chsplots, #
).toModify(
    nanojmeDQM.vplots, 
    JetCHS = None # Remove "JetCHS" from DQM
)
run2_nanoAOD_ANY.toModify(
    nanojmeDQMMC.vplots.JetPuppi.sels, 
    Prompt = nanojmeDQMMC.vplots.Jet.sels.Prompt,
    PromptB = nanojmeDQMMC.vplots.Jet.sels.PromptB
)

from DQMServices.Core.DQMQualityTester import DQMQualityTester
nanoDQMQTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('PhysicsTools/NanoAOD/test/dqmQualityTests.xml'),
    prescaleFactor = cms.untracked.int32(1),                               
    testInEventloop = cms.untracked.bool(False),
    qtestOnEndLumi = cms.untracked.bool(False),
    verboseQT =  cms.untracked.bool(True)
)

nanojmeHarvest = cms.Sequence( nanoDQMQTester )
