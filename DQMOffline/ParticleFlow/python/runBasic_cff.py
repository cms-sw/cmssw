import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from DQMOffline.ParticleFlow.pfAnalyzer_cfi import pfAnalyzer

DQMOfflinePFAnalyzer = DQMEDAnalyzer("PFAnalyzer",
    pfCandidates             = cms.InputTag("particleFlow"),
    pfJetCollection        = cms.InputTag("ak4PFJetsPuppiCorrected"),
    PVCollection             = cms.InputTag("offlinePrimaryVertices"),

    TriggerResultsLabel        = cms.InputTag("TriggerResults::HLT"),
    TriggerNames = cms.vstring("HLT_PFJet450"),
    puppiWeight  = cms.InputTag("puppi"),
    #eventSelection = cms.string("dijet"),
    eventSelection = cms.string("nocut"),
    isMiniAOD = cms.bool(False),
    runNumber = cms.uint32(0),

    pfAnalysis = cms.PSet(
      pfNames = cms.vstring("allPFC", "neutralHadPFC", "chargedHadPFC", "electronPFC", "muonPFC", "gammaPFC", "hadHFPFC", "emHFPFC"),
      # Bins of NPV for plots
      NPVBins = cms.vdouble(0,200),


      # A list of observables for which plots should be made.
      # The format should be a list of semicolon-separated values.
      # The first is the observable name, corresponding to a key in m_funcMap 
      # in PFAnalysis. The second is the TLatex string that serves as the x-axis
      # title (which must not include a semicolon).
      # The last values are the bins. If three values are given, then the values,
      # in order, are the number of bins, the lowest, and the highest values.
      # If any other number is given, this is just a list of bins for the histogram.
      observables     = cms.vstring('pt;p_{T,PFC};50.;0.;350.', 
                                    'eta;#eta;30;-5;5',
                                    'phi;#phi;30;-3.14;3.14',
                                    'puppi;E;30;0;1',
                                    'PFHad_calibration;E_{Hcal,calib} / E_{Hcal, raw};30;0;4',
                                    'HCalE_depth1;HCal E, depth 1;30;0;1',
                                    'HCalE_depth2;HCal E, depth 2;30;0;1',
                                    'HCalE_depth3;HCal E, depth 3;30;0;1',
                                    'HCalE_depth4;HCal E, depth 4;30;0;1',
                                    'HCalE_depth5;HCal E, depth 5;30;0;1',
                                    'HCalE_depth6;HCal E, depth 6;30;0;1',
                                    'HCalE_depth7;HCal E, depth 7;30;0;1',
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
      binList2D     = cms.vstring(
                                '[eta;30;-5;5][phi;30;-3.14;3.14]',
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
      # For example, for 
      # cutList     = cms.vstring('[pt;1;0;10000]'),
      # there is one histogram made for PFCs with 0 < pT < 10000.
      # Similarly, for cutList     = cms.vstring('[pt;1;0;10000]', '[pt;1;05;10000][eta;1;-5;5]'),
      # there is one histogram made for PFCs with 0 < pT < 10000,
      # and one histogram made for PFCs with 5 < pT < 10000 and -5 < eta < 5.
     
      cutList     = cms.vstring(
                                '[pt;1;0;10000]',
                                #'[pt;0;1;2;4;6;10;20;40;60;100][abseta;0;1.5;2.0;2.5;2.8;2.85;2.9;2.95;3]',
                                '[pt;0;2;10;100]',
                                '[abseta;0;1;2;2.5;2.7;3;3.5;5.0]',
                               ),

      # This is a list of multidimensional cuts on the jets that are applied for the plots.
      # The format should be a list of semicolon-separated values.
      # The first is the observable name, corresponding to a key in m_jetFuncMap 
      # in PFAnalysis. The last values are the bins, following the same
      # conventions as the observables.
      #
      # Just like for cutList, multiple sets of cuts can be applied, using the same formulation.
      jetCutList     = cms.vstring(
                                   '[pt;20;50;100;1000]',
                                  ),
    )


)
