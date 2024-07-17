import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

PFAnalyzer = DQMEDAnalyzer("PFAnalyzer",
    pfJetCollection        = cms.InputTag("ak4PFJetsCHS"),
    pfCandidates             = cms.InputTag("particleFlow"),
    PVCollection             = cms.InputTag("offlinePrimaryVerticesWithBS"),

    TriggerResultsLabel        = cms.InputTag("TriggerResults::HLT"),


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
                                    'eta;#eta;50;-5;5',
                                    'phi;#phi;50;-3.14;3.14',
                                    'energy;E;50;0;300',
                                    'RawHCal_E;Raw E_{hcal};50;0;300', 
                                    'HCal_E;E_{hcal};50;0;300', 
                                    'PFHad_calibration;E_{Hcal,calib} / E_{Hcal, raw};50;0;4',
                                   ),

      # A list of event- or jet-wide observables for which plots should be made.
      # The format should be a list of semicolon-separated values.
      # The first is the observable name, corresponding to a key in m_funcMap 
      # in PFAnalysis. The second is the TLatex string that serves as the x-axis
      # title (which must not include a semicolon).
      # The last values are the bins. If three values are given, then the values,
      # in order, are the number of bins, the lowest, and the highest values.
      # If any other number is given, this is just a list of bins for the histogram.
      # Since the binning may be very different for events and jets, the first list
      # of bins is for the event histograms, and the second set is for the jets
      eventObservables     = cms.vstring(
                                         'NPFC;N_{PFC};500;0;10000;100;0;100',
                                        ),


      pfInJetObservables     = cms.vstring(
                                         'PFSpectrum;E_{PF}/E_{jet};50;0;1',
                                        ),


      # This is a list of multidimensional cuts that are applied for the plots.
      # In the case of multiple bins, every combination of bin is tested.
      # The format should be a list of semicolon-separated values.
      # The first is the observable name, corresponding to a key in m_funcMap 
      # in PFAnalysis. The last values are the bins, following the same
      # conventions as the observables.
      #
      # Since we may want to test multiple sets of cuts simultaneously, 
      # these are separated by '['
      #cutList     = cms.vstring('pt;0;1;5;10;100',
      cutList     = cms.vstring(
                                '[pt;1;0;10000]',
                                '[pt;1;0;10000][eta;1;-5;5]',
                                '[pt;1;0;1;2;4;10;10000]',
                               ),

      # This is a list of multidimensional cuts on the jets that are applied for the plots.
      # The format should be a list of semicolon-separated values.
      # The first is the observable name, corresponding to a key in m_jetFuncMap 
      # in PFAnalysis. The last values are the bins, following the same
      # conventions as the observables.
      jetCutList     = cms.vstring('pt;20;50;100;10000'),
    )


)
