import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

PFAnalyzer = DQMEDAnalyzer("PFAnalyzer",
    pfJetCollection        = cms.InputTag("ak4PFJetsCHS"),
    pfCandidates             = cms.InputTag("particleFlow"),
    PVCollection             = cms.InputTag("offlinePrimaryVerticesWithBS"),

    pfAnalysis = cms.PSet(

      # A list of observables for which plots should be made.
      # The format should be a list of semicolon-separated values.
      # The first is the observable name, corresponding to a key in m_funcMap 
      # in PFAnalysis. The second is the TLatex string that serves as the x-axis
      # title (which must not include a semicolon).
      # The last values are the bins. If three values are given, then the values,
      # in order, are the number of bins, the lowest, and the highest values.
      # If any other number is given, this is just a list of bins for the histogram.
      observables     = cms.vstring('pt;p_{T,PFC};50.;0.;300.', 
                                    'hcalE;E_{hcal};50;0;300', 
                                   ),

      # This is a list of multidimensional cuts that are applied for the plots.
      # In the case of multiple bins, every combination of bin is tested.
      # The format should be a list of semicolon-separated values.
      # The first is the observable name, corresponding to a key in m_funcMap 
      # in PFAnalysis. The last values are the bins, following the same
      # conventions as the observables.
      cutList     = cms.vstring('eta;2;0;5',
                                'phi;2;-3;3'
                               ),

      # This is a list of multidimensional cuts on the jets that are applied for the plots.
      # The format should be a list of semicolon-separated values.
      # The first is the observable name, corresponding to a key in m_jetFuncMap 
      # in PFAnalysis. The last values are the bins, following the same
      # conventions as the observables.
      jetCutList     = cms.vstring('pt;150;10000'),
    )


)
