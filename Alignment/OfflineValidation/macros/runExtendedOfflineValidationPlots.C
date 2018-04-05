// template to run Extended Offline Validation Plots plots
void runExtendedOfflineValidationPlots()
{
  // load framework lite just to find the CMSSW libs...
  gSystem->Load("libFWCoreFWLite");
  FWLiteEnabler::enable();
  //compile the makro
  gROOT->ProcessLine(".L PlotAlignmentValidation.C++");


  ///CRAFT////
  ///add file that you want to overlay following the syntax below
  ///(std::string "fileName", std::string legName="", int color=1, int style=1)
     PlotAlignmentValidation p("/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/CRAFT_Note/draft0_27may2009/data/FinalIdeal.root","Design",1,1);
     p.loadFileList("/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/CRAFT_Note/draft0_27may2009/data/FinalHIP.root","HIP",3,1);
     p.loadFileList("/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/CRAFT_Note/draft0_27may2009/data/FinalMP.root","Millepede",6,1);
     p.loadFileList("/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/CRAFT_Note/draft0_27may2009/data/FinalSHMPmerged.root","SuperHipMPmerged",4,1);
    
  p.setOutputDir("$TMPDIR");
  //  p.setTreeBaseDir("TrackHitFilter");  //if the tree TkOffValidation is not in TrackerOfflineValidation directory
  //  p.useFitForDMRplots(true); //if the width adn mean value of the DMR shall be taken from fit
  p.plotDMR("medianY",30);
}
