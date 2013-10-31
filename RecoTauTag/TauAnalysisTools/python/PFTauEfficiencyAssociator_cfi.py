'''
        PFTauEficiencyAssociator_cfi

        Author: Evan K. Friis, UC Davis 

        The PFTauEfficiencyAssociator produces ValueMap<pat::LookupTableRecord>s
        associating an expected efficiency (or fake rate, from QCD) for a reco::PFTau
        given its kinematics.  The default configuration parameterizes the eff/fake rate
        by pt, eta and jet width and stores the information in a TH3.

'''
import FWCore.ParameterSet.Config as cms
from RecoTauTag.TauAnalysisTools.tools.ntupleDefinitions import pftau_expressions, common_expressions
from RecoTauTag.TauAnalysisTools.fakeRate.histogramConfiguration import makeCuts 
from RecoTauTag.TauAnalysisTools.fakeRate.associatorTools import *

# Get the histogram selection definitions.  We don't need to worry about what
# the denominator is here, we just need the names
disc_configs = makeCuts(denominator="1")

protoEffciencyAssociator = cms.EDProducer("PFTauEfficiencyAssociatorFromTH3",
       PFTauProducer = cms.InputTag("shrinkingConePFTauProducer"),
       xAxisFunction = pftau_expressions.jetPt,
       yAxisFunction = cms.string("abs(%s)" % pftau_expressions.jetEta.value()),
       zAxisFunction = pftau_expressions.jetWidth,
       efficiencySources = cms.PSet(
           filename = cms.string("/afs/cern.ch/user/f/friis/public/TauPeformance_QCD_BCtoMu.root"),
           # each efficiency source needs to be defined as a separate PSet
       )
)

# Build the list of efficiency sources from the histogram production
# configuration disc_configs
MuEnrichedQCDEffSources = add_eff_sources(prefix="fr",
        disc_configs=disc_configs.keys(), suffix="MuEnrichedQCDsim")

shrinkingConeMuEnrichedQCDAssociator = protoEffciencyAssociator.clone()
shrinkingConeMuEnrichedQCDAssociator.efficiencySources = cms.PSet(
        MuEnrichedQCDEffSources,
        filename=cms.string("/afs/cern.ch/user/f/friis/public/TauFakeRateApr29_VersusJetPt/ppmux_histograms.root")
)

DiJetHighPtEffSources = add_eff_sources(prefix="fr",
        disc_configs=disc_configs.keys(), suffix="DiJetHighPtsim")

shrinkingConeDiJetHighPt = protoEffciencyAssociator.clone()
shrinkingConeDiJetHighPt.efficiencySources = cms.PSet(
        DiJetHighPtEffSources,
        filename=cms.string("/afs/cern.ch/user/f/friis/public/TauFakeRateApr29_VersusJetPt/dijet_highpt_histograms.root")
)

DiJetSecondPtEffSources = add_eff_sources(prefix="fr",
        disc_configs=disc_configs.keys(), suffix="DiJetSecondPtsim")

shrinkingConeDiJetSecondPt = protoEffciencyAssociator.clone()
shrinkingConeDiJetSecondPt.efficiencySources = cms.PSet(
        DiJetSecondPtEffSources,
        filename=cms.string("/afs/cern.ch/user/f/friis/public/TauFakeRateApr29_VersusJetPt/dijet_secondpt_histograms.root")
)

WJetsEffSources = add_eff_sources(prefix="fr",
        disc_configs=disc_configs.keys(), suffix="WJetssim")

shrinkingConeWJets = protoEffciencyAssociator.clone()
shrinkingConeWJets.efficiencySources = cms.PSet(
        WJetsEffSources,
        filename=cms.string("/afs/cern.ch/user/f/friis/public/TauFakeRateApr29_VersusJetPt/wjets_histograms.root")
)

ZTTEffSimSources = add_eff_sources(prefix="eff",
        disc_configs=disc_configs.keys(), suffix="ZTTsim")

shrinkingConeZTTEffSimAssociator = protoEffciencyAssociator.clone()
shrinkingConeZTTEffSimAssociator.efficiencySources = cms.PSet(
        ZTTEffSimSources,
        filename=cms.string("/afs/cern.ch/user/f/friis/public/TauFakeRateApr29_VersusJetPt/ztt_histograms.root")
)

associateTauFakeRates = cms.Sequence(shrinkingConeZTTEffSimAssociator*
                                     shrinkingConeWJets*
                                     shrinkingConeMuEnrichedQCDAssociator*
                                     shrinkingConeDiJetHighPt*
                                     shrinkingConeDiJetSecondPt)
   
if __name__ == '__main__':
   # Print all the available efficiencies
   fake_rates = [shrinkingConeZTTEffSimAssociator,
                 shrinkingConeWJets,
                 shrinkingConeMuEnrichedQCDAssociator,
                 shrinkingConeDiJetHighPt,
                 shrinkingConeDiJetSecondPt]

   my_pat_effs = cms.PSet()
   # Loop by index to avoid funny namespace stuff
   for i in range(len(fake_rates)):
       build_pat_efficiency_loader(
           fake_rates[i], namespace=None, append_to=my_pat_effs)
   print my_pat_effs

