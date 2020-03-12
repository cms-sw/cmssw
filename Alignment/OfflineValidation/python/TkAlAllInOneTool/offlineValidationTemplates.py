######################################################################
######################################################################
offlineTemplate = """
process.oneGoodVertexFilter = cms.EDFilter("VertexSelector",
                                           src = cms.InputTag("offlinePrimaryVertices"),
                                           cut = cms.string("!isFake && ndof > 4 && abs(z) <= 15 && position.Rho <= 2"), # tracksSize() > 3 for the older cut
                                           filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
                                           )



process.FilterGoodEvents=cms.Sequence(process.oneGoodVertexFilter)


process.noScraping= cms.EDFilter("FilterOutScraping",
                                 src=cms.InputTag(".oO[TrackCollection]Oo."),
                                 applyfilter = cms.untracked.bool(True),
                                 debugOn = cms.untracked.bool(False), ## Or 'True' to get some per-event info
                                 numtrack = cms.untracked.uint32(10),
                                 thresh = cms.untracked.double(0.25)
                                 )
####################################


 ##
 ## Load and Configure OfflineValidation and Output File
 ##
process.load("Alignment.OfflineValidation.TrackerOfflineValidation_.oO[offlineValidationMode]Oo._cff")
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..Tracks = 'FinalTrackRefitter'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..trajectoryInput = 'FinalTrackRefitter'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..moduleLevelHistsTransient = .oO[offlineModuleLevelHistsTransient]Oo.
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..moduleLevelProfiles = .oO[offlineModuleLevelProfiles]Oo.
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..stripYResiduals = .oO[stripYResiduals]Oo.
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..maxTracks = .oO[maxtracks]Oo./ .oO[parallelJobs]Oo.
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..chargeCut = .oO[chargeCut]Oo.
"""

OfflineValidationSequence = "process.seqTrackerOfflineValidation.oO[offlineValidationMode]Oo."


######################################################################
######################################################################
mergeOfflineParJobsTemplate="""
#include "Alignment/OfflineValidation/scripts/merge_TrackerOfflineValidation.C"

int TkAlOfflineJobsMerge(TString pars, TString outFile)
{
// load framework lite just to find the CMSSW libs...
gSystem->Load("libFWCoreFWLite");
FWLiteEnabler::enable();

return hadd(pars, outFile);
}
"""


######################################################################
######################################################################
offlineFileOutputTemplate = """
process.TFileService.fileName = '.oO[outputFile]Oo.'
"""


######################################################################
######################################################################
offlineDqmFileOutputTemplate = """
process.DqmSaverTkAl.workflow = '.oO[workflow]Oo.'
process.DqmSaverTkAl.dirName = '.oO[workdir]Oo./.'
process.DqmSaverTkAl.forceRunNumber = .oO[firstRunNumber]Oo.
"""


######################################################################
######################################################################
extendedValidationExecution="""
#run extended offline validation scripts
echo -e "\n\nRunning extended offline validation"

cp .oO[extendedValScriptPath]Oo. .
root -x -b -q -l TkAlExtendedOfflineValidation.C

"""


######################################################################
######################################################################
extendedValidationTemplate="""
#include "Alignment/OfflineValidation/macros/PlotAlignmentValidation.C"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"

void TkAlExtendedOfflineValidation()
{
  TkAlStyle::legendheader = ".oO[legendheader]Oo.";
  TkAlStyle::legendoptions = ".oO[legendoptions]Oo.";
  TkAlStyle::set(.oO[publicationstatus]Oo., .oO[era]Oo., ".oO[customtitle]Oo.", ".oO[customrighttitle]Oo.");
  bool bigtext = .oO[bigtext]Oo.;
  gStyle->SetTitleH        ( 0.07 );
  gStyle->SetTitleW        ( 1.00 );
  gStyle->SetTitleFont     (  132 );
  // load framework lite just to find the CMSSW libs...
  gSystem->Load("libFWCoreFWLite");
  FWLiteEnabler::enable();

  PlotAlignmentValidation p(bigtext);
.oO[PlottingInstantiation]Oo.
  p.setOutputDir(".oO[datadir]Oo./.oO[PlotsDirName]Oo.");
  p.useFitForDMRplots(.oO[usefit]Oo.);
  p.setTreeBaseDir(".oO[OfflineTreeBaseDir]Oo.");
  p.plotDMR(".oO[DMRMethod]Oo.",.oO[DMRMinimum]Oo.,".oO[DMROptions]Oo.");
  p.plotSurfaceShapes(".oO[SurfaceShapes]Oo.");
  p.plotChi2("root://eoscms//eos/cms/store/group/alca_trackeralign/AlignmentValidation/.oO[eosdir]Oo./.oO[validationId]Oo._result.root");
  vector<int> moduleids = {.oO[moduleid]Oo.};
  for (auto moduleid : moduleids) {
  	p.residual_by_moduleID(moduleid);
  }
}
"""
