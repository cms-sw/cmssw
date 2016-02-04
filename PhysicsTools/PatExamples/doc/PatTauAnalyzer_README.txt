/** class PhysicsTools/PatExamples/PatTauAnalyzer
 *
 * Tutorial on pat::Taus
 *
 * author: Christian Veelken, UC Davis
 *
 * NOTE: The tutorial has been prepared and tested using CMSSW_2_2_13.
 *       In order to run the tutorial, you need to check-out newer versions 
 *       of the following packages from cvs:
 *
 *         cvs co -r CMSSW_2_2_13 PhysicsTools/PatAlgos
 *         cvs up -r 1.2.2.6 PhysicsTools/PatAlgos/python/tools/tauTools.py
 *         cvs co -r global_PFTau_22X_V00-02-03 RecoTauTag
 *         cvs co -r V00-12-01-02 DataFormats/TauReco
 *
 *       After you compiled these packages (by executing 'scramv1 b'),
 *       you can produce the pat::Tau plots by executing:
 *
 *         cmsRun analyzePatTau_fromAOD_cfg.py
 *         root patTau_idEfficiency.C
 *         .q
 *
 *       This will produce .eps file showing the tau id. efficiency as function of Pt, Eta and Phi.
 *
 *       You can look at the other plots produced by executing 'cmsRun analyzePatTau_fromAOD_cfg.py' by doing:
 * 
 *         root patTauHistograms.root
 *         new TBrowser
 *         
 *       and then click on "ROOT Files" --> patTauHistograms.root --> analyzePatTau 
 *       and finally the name of the histogram you wish to look at.
 *
 */
