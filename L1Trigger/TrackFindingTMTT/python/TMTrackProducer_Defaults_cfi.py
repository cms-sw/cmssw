import FWCore.ParameterSet.Config as cms

#---------------------------------------------------------------------------------------------------------
# This describes the full TMTT track reconstruction chain with 3 GeV threshold, where:
# the GP divides the tracker into 18 eta sectors (each sub-divided into 2 virtual eta subsectors);  
# the HT uses a  32x18 array followed by 2x2 mini-HT array, with transverese HT readout & multiplexing, 
# followed by the KF (or optionally SF+SLR) track fit; duplicate track removal (Algo50) is run.
#---------------------------------------------------------------------------------------------------------

TMTrackProducer_params = cms.PSet(

  tpInputTag = cms.InputTag("mix", "MergedTrackTruth"),
  stubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"),
  stubTruthInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
  clusterTruthInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
  genJetInputTag = cms.InputTag("ak4GenJets", ""),

  # Enable all use of MC truth info (disable to save CPU)
  EnableMCtruth = cms.bool(True),
  # Enable output histograms & job tracking performance summary (disable to save CPU)
  EnableHistos = cms.bool(True),

  #=== Cuts on MC truth particles (i.e., tracking particles) used for tracking efficiency measurements.

  GenCuts = cms.PSet(
     GenMinPt         = cms.double(3.0),
     GenMaxAbsEta     = cms.double(2.4),
     GenMaxVertR      = cms.double(1.0), # Max distance of particle production vertex from centre of CMS.
     GenMaxVertZ      = cms.double(30.0),
     GenMaxD0         = cms.double(5.0), # Max transverse impact parameter.
     GenMaxZ0         = cms.double(999.0), # Max transverse impact parameter.
     GenPdgIds        = cms.vuint32(), # Only particles with these PDG codes used for efficiency measurement.

     # Additional cut on MC truth tracks used for algorithmic tracking efficiency measurements.
     # You should usually set this equal to value of L1TrackDef.MinStubLayers below, unless L1TrackDef.MinPtToReduceLayers
     # is < 10000, in which case, set it equal to (L1TrackDef.MinStubLayers - 1).
     GenMinStubLayers = cms.uint32(4)
  ),

  #=== Cuts applied to stubs before arriving in L1 track finding board.

  StubCuts = cms.PSet(
     # Reduce number of bits used by front-end chips to store stub bend info? 
     # = 0 (no); = 1 (yes using official recipe); = 2 (yes using TMTT method)
     DegradeBendRes = cms.uint32(2),
     # Don't use stubs with eta beyond this cut, since the tracker geometry makes it impossible to reconstruct tracks with them.
     MaxStubEta     = cms.double(2.4),
     # Don't use stubs whose measured Pt from bend info is significantly below HTArraySpec.HoughMinPt, where "significantly" means allowing for resolution in q/Pt derived from stub bend resolution specified below.
     KillLowPtStubs = cms.bool(True),
     # Print stub windows corresponding to KillLowPtStubs, in python cfg format used by CMSSW.
     PrintStubWindows = cms.bool(False),
     # Bend resolution assumed by bend filter in units of strip pitch. Also used when assigning stubs to sectors if EtaPhiSectors.CalcPhiTrkRes=True. And by the bend filter if HTFillingRphi.UseBendFilter=True.
     # Suggested value: 1.19 if DegradeBendRes = 0, or 1.249 if it > 0.
     # N.B. Avoid 1/4-integer values due to rounding error issues.
     BendResolution = cms.double(1.249),
     # Additional contribution to bend resolution from its encoding into a reduced number of bits.
     # This number is the assumed resolution relative to the naive guess of its value.
     # It is ignored in DegradeBendRes = 0.
     BendResolutionExtra = cms.double(0.0),
     # Order stubs by bend in DTC, such that highest Pt stubs are transmitted first.
     OrderStubsByBend = cms.bool(True)
  ),

  #=== Optional Stub digitization.

  StubDigitize = cms.PSet(
     EnableDigitize  = cms.bool(True),  # Digitize stub coords? If not, use floating point coords.
     #
     #--- Parameters available in MP board. (And in case of Hybrid used internally in KF)
     #
     PhiSectorBits   = cms.uint32(6),    # Bits used to store phi sector number -- NOT USED
     PhiSBits        = cms.uint32(14),   # Bits used to store phiS coord. (13 enough?)
     PhiSRange       = cms.double(0.698131700),  # Range phiS coord. covers in radians.
     RtBits          = cms.uint32(12),   # Bits used to store Rt coord.
      RtRange        = cms.double(91.652837), # Range Rt coord. covers in units of cm.
     ZBits           = cms.uint32(14),   # Bits used to store z coord.
      ZRange          = cms.double(733.2227), # Range z coord. covers in units of cm.
     #
     #--- Parameters available in GP board (excluding any in common with MP specified above).
     #
     PhiOBits        = cms.uint32(15),      # Bits used to store PhiO parameter.
     PhiORange      = cms.double(1.3962634), # Range PhiO parameter covers.
     BendBits        = cms.uint32(6)        # Bits used to store stub bend.
  ),

  #=== Configuration of Geometric Processor.

  GeometricProc = cms.PSet(
     # Use an FPGA-friendly approximation to determine track angle dphi from bend in GP?
     UseApproxB        = cms.bool(True),       # Use approximation for B
     # Params of approximation if used.
     BApprox_gradient  = cms.double(0.886454), # Gradient term of linear equation for approximating B
     BApprox_intercept = cms.double(0.504148)  # Intercept term of linear equation for approximating B
  ),

  #=== Division of Tracker into phi sectors.

  PhiSectors = cms.PSet(
     NumPhiNonants      = cms.uint32(9),    # Divisions of Tracker at DTC
     NumPhiSectors      = cms.uint32(18),   # Divisions of Tracker at GP.
     ChosenRofPhi       = cms.double(67.240), # Use phi of track at this radius for assignment of stubs to phi sectors & also for one of the axes of the r-phi HT. If ChosenRofPhi=0, then use track phi0. - Should be an integer multiple of the stub r digitisation granularity.
     #--- You can set one or both the following parameters to True.
     UseStubPhi         = cms.bool(True),  # Require stub phi to be consistent with track of Pt > HTArraySpec.HoughMinPt that crosses HT phi axis?
     UseStubPhiTrk      = cms.bool(True),  # Require stub phi0 (or phi65 etc.) as estimated from stub bend, to lie within HT phi axis, allowing tolerance(s) specified below?
     AssumedPhiTrkRes   = cms.double(0.5), # Tolerance in stub phi0 (or phi65) assumed to be this fraction of phi sector width. (N.B. If > 0.5, then stubs can be shared by more than 2 phi sectors).
     CalcPhiTrkRes      = cms.bool(True),  # If true, tolerance in stub phi0 (or phi65 etc.) will be reduced below AssumedPhiTrkRes if stub bend resolution specified in StubCuts.BendResolution suggests it is safe to do so.
     HandleStripsPhiSec = cms.bool(False)  # If True, adjust algorithm to allow for uncertainty in stub (r,z) coordinate caused by length of 2S module strips when assigning stubs to phi sectors.
  ),

  #=== Division of Tracker into eta sectors

  EtaSectors = cms.PSet(
# Eta boundaries for 18 eta regions
#     EtaRegions = cms.vdouble(-2.4,-2.16,-1.95,-1.7,-1.43,-1.16,-0.89,-0.61,-0.31,0.0,0.31,0.61,0.89,1.16,1.43,1.7,1.95,2.16,2.4), 
# Eta boundaries for 16 eta regions
     EtaRegions = cms.vdouble(-2.4,-2.08,-1.68,-1.26,-0.90,-0.62,-0.41,-0.20,0.0,0.20,0.41,0.62,0.90,1.26,1.68,2.08,2.4), 
     ChosenRofZ  = cms.double(50.),        # Use z of track at this radius for assignment of tracks to eta sectors & also for one of the axes of the r-z HT. Do not set to zero!
     BeamWindowZ = cms.double(15),         # Half-width of window assumed to contain beam-spot in z.
     HandleStripsEtaSec = cms.bool(False), # If True, adjust algorithm to allow for uncertainty in stub (r,z) coordinate caused by length of 2S module strips when assigning stubs to eta sectors.
     AllowOver2EtaSecs = cms.bool(True)    # If True, the code will not throw an error if a stub is assigned to 3 or more eta sectors.
  ),

  #=== r-phi Hough transform array specifications.

  HTArraySpecRphi = cms.PSet(
     HoughMinPt      = cms.double(3.0), # Min track Pt that Hough Transform must find. Also used by StubCuts.KillLowPtStubs and by EtaPhiSectors.UseStubPhi.
     # If MiniHTstage = True, these refers to mini cells in whole HT array.
     HoughNbinsPt    = cms.uint32(32),  # HT array dimension in track q/Pt. Ignored if HoughNcellsRphi > 0. (If MiniHTstage = True, this refers to mini cells in whole HT array).
     HoughNbinsPhi   = cms.uint32(64),  # HT array dimension in track phi0 (or phi65 or any other track phi angle. Ignored if HoughNcellsRphi > 0. (If MiniHTstage = True, this refers to mini cells in whole HT array).
     HoughNcellsRphi = cms.int32(-1),   # If > 0, then parameters HoughNbinsPt and HoughNbinsPhi will be calculated from the constraints that their product should equal HoughNcellsRphi and their ratio should make the maximum |gradient|" of stub lines in the HT array equal to 1. If <= 0, then HoughNbinsPt and HoughNbinsPhi will be taken from the values configured above.
     EnableMerge2x2  = cms.bool(False), # Groups of neighbouring 2x2 cells in HT will be treated as if they are a single large cell? N.B. You can only enable this option if your HT array has even numbers of bins in both dimensions. And this cfg param ignored if MiniHTstage = True.  HISTORIC OPTION. SUGGEST NOT USING!
     MaxPtToMerge2x2 = cms.double(3.5), # but only cells with pt < MaxPtToMerge2x2 will be merged in this way (irrelevant if EnableMerge2x2 = false).
     NumSubSecsEta   = cms.uint32(2),   # Subdivide each sector into this number of subsectors in eta within r-phi HT.
     Shape           = cms.uint32(0),   # cell shape: 0 for square, 1 for diamond, 2 hexagon (with vertical sides), 3 square with alternate rows shifted by 0.5*cell_width.
     MiniHTstage       = cms.bool(True), # Run 2nd stage HT with mini cells inside each 1st stage normal HT cell..
     MiniHoughNbinsPt  = cms.uint32(2),   # Number of mini cells along q/Pt axis inside each normal HT cell.
     MiniHoughNbinsPhi = cms.uint32(2),   # Number of mini cells along phi axis inside each normal HT cell.
     MiniHoughMinPt    = cms.double(3.0), # Below this Pt threshold, the mini HT will not be used, to reduce sensitivity to scattering, with instead tracks found by 1st stage coarse HT sent to output. (HT cell numbering remains as if mini HT were in use everywhere).
     MiniHoughDontKill = cms.bool(False), # If true, allows tracks found by 1st stage coarse HT to be output if 2nd stage mini HT finds no tracks.
     MiniHoughDontKillMinPt = cms.double(8.0), # If MiniHoughDontKill=True, this option restricts it to keep 1st stage HT tracks only if their Pt is exceeds this cut. (Used to improve electron tracking above this threshold).
     MiniHoughLoadBalance = cms.uint32(2) # Load balancing disabled = 0; static load balancing of output links = 1; dynamic load balancing of output links = 2.
  ),

  #=== Rules governing how stubs are filled into the r-phi Hough Transform array.

  HTFillingRphi = cms.PSet(
     # If True, adjust algorithm to allow for uncertainty in stub (r,z) coordinate caused by length of 2S module strips when filling r-phi HT with stubs.
     HandleStripsRphiHT   = cms.bool(False),
     # Take all cells in r-phi HT array crossed by line corresponding to each stub (= 0) or take only some to reduce rate at cost
     # of efficiency ( > 0). If this option is > 0, it can be 1 or 2, corresponding to different algorithms for rejecting
     # some of the cells. "1" is an algorithm invented by Ian, whereas "2" corresponds to Thomas' 1st firmware implementation which only handled 1 cell per HT column.
     # Suggest setting KillSomeHTCellsRphi=1 (=0) if HTArraySpec.ChosenRofPhi=0 (>0)
     KillSomeHTCellsRphi  = cms.uint32(0),
     # Use filter in each r-phi HT cell, filling it only with stubs that have consistent bend information?
     # The assumed bend resolution is specified in StubCuts.BendResolution.
     UseBendFilter        = cms.bool(True),
     # Use filter in each HT cell, preventing more than the specified number of stubs being stored in the cell. (Reflecting memory limit of hardware). N.B. Results depend on assumed order of stubs.
     # N.B. If mini-HT is in use, then this cut applies to coarse-HT.
     #MaxStubsInCell       = cms.uint32(99999), # Setting this to anything more than 99 disables this option
     MaxStubsInCell          = cms.uint32(32),    # set it equal to value used in hardware.
     MaxStubsInCellMiniHough = cms.uint32(16),    # Same type of cut for mini-HT (if in use)
     # If BusySectorKill = True, and more than BusySectorNumStubs stubs are assigned to tracks by an r-phi HT array, then the excess tracks are killed, with lowest Pt ones killed first. This is because HT hardware has finite readout time.
     BusySectorKill       = cms.bool(True),
     BusySectorNumStubs   = cms.uint32(162), # Or 144 if only 320 MHz FW.
     # If BusySectorMbinRanges is not empty, then the BusySectorNumStubs cut is instead applied to the subset of tracks appearing in the following m bin (q/Pt) ranges of the HT array. The sum of the entries in the vector should equal the number of m bins in the HT. (N.B. If EnableMerge2x2 or MiniHTstage = True, then the m bin ranges here correspond to the bins before merging. Also in these cases, the odd m-bin numbers don't correspond to HT outputs, so should be all grouped together on a single imaginary link).
     # If BusySectorMbinOrder is not empty, then the m-bins are grouped in the specified order, instead of sequentially.
     # (Histos NumStubsPerLink, NumStubsVsLink & MeanStubsPerLink useful for optimising this option).
     #
     # Choice for 16x32 coarse HT array followed by 2x2 mini-HT array with 3 GeV Pt threshold.
     BusySectorMbinRanges  = cms.vuint32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 16),   
     BusySectorMbinOrder   = cms.vuint32(0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30, 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31),
     # Choice for 24x32 coarse HT array followed by 2x2 mini-HT array with 2 GeV Pt threshold.
     #BusySectorMbinRanges = cms.vuint32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 24),   
     #BusySectorMbinOrder  = cms.vuint32(0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46, 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47),
     #
     # If BusyInputSectorKill = True, and more than BusyInputSectorNumStubs are input to the HT array from the GP, then
     # the excess stubs are killed. This is because HT hardware has finite readin time.
     # Results unreliable as depend on assumed order of stubs.
     BusyInputSectorKill  = cms.bool(True),
     BusyInputSectorNumStubs  = cms.uint32(162),  #  Or 144 if only 320 MHz FW
     # Multiplex the outputs from several HTs onto a single pair of output optical links?
     # Options: 0 = disable Mux; 1 = Dec. 2016 Mux; 2 = Mar 2018 Mux (for transverse HT readout by m-bin). 
     # (The mux algorithm is hard-wired in class MuxHToutputs, and currently only works if option BusySectorMbinRanges is being used); 3 = Sept 2019 Mux (transerse HT readout by m-bin), with single m bin in entire nonant going to each link.
     MuxOutputsHT = cms.uint32(3),
     # If this is non-empty, then only the specified eta sectors are enabled, to study them individually.
     EtaRegWhitelist = cms.vuint32()
  ),

  #=== Options controlling r-z track filters (or any other track filters run after the Hough transform, as opposed to inside it).
  #=== (Irrelevant for track fitters that don't require any r-z filter run before them).

  RZfilterOpts = cms.PSet(
     # Specify preferred r-z filter (from those available inside TrkRZfilter.cc) - currently only "SeedFilter".
     RZFilterName        = cms.string("SeedFilter"),
     #--- Options relevant for Seed filter, (so only relevant if rzFilterName="SeedFilter").
     # Added resolution beyond that estimated from hit resolution. 
     SeedResolution      = cms.double(0.),
     # Store stubs compatible with all possible good seed.
     KeepAllSeed         = cms.bool(False),
     # Maximum number of seed combinations to bother checking per track candidate.
     #MaxSeedCombinations = cms.uint32(999),
     MaxSeedCombinations = cms.uint32(15),
     # Maximum number of seed combinations consistent with (z0,eta) sector constraints to bother checking per track candidate.
     #MaxGoodSeedCombinations = cms.uint32(13),
     MaxGoodSeedCombinations = cms.uint32(10),
     # Maximum number of seeds that a single stub can be included in.
     MaxSeedsPerStub     = cms.uint32(4),
     # Reject tracks whose estimated rapidity from seed filter is inconsistent range of with eta sector. (Kills some duplicate tracks).
     zTrkSectorCheck     = cms.bool(True),
     # Min. number of layers in rz track that must have stubs for track to be declared found by seed filter.
     MinFilterLayers     = cms.uint32(4)
  ),

  #=== Rules for deciding when the (HT) track finding has found an L1 track candidate

  L1TrackDef = cms.PSet(
     # Min. number of layers the track must have stubs in.
     MinStubLayers        = cms.uint32(5),
     # Change min. number of layers cut to (MinStubLayers - 1) for tracks with Pt exceeding this cut.
     # If this is set to > 10000 , this option is disabled.
     MinPtToReduceLayers  = cms.double(99999.),
     # Change min. number of layers cut to (MinStubLayers - 1) for tracks in these rapidity sectors.
     # (Histogram "AlgEffVsEtaSec" will help you identify which sectors to declare).
     #EtaSecsReduceLayers  = cms.vuint32(),
     EtaSecsReduceLayers  = cms.vuint32(5,12),
     # Define layers using layer ID (true) or by bins in radius of 5 cm width (false).
     UseLayerID           = cms.bool(True),
     # Reduce this layer ID, so that it takes no more than 8 different values in any eta region (simplifies firmware).
     ReducedLayerID       = cms.bool(True)
  ),

  #=== Specification of algorithm to eliminate duplicate tracks.

  DupTrkRemoval = cms.PSet(
    #--- Specify which duplicate removal algorithm(s) to run:  option 0 means disable duplicate track removal, whilst > 0 runs a specific algorithm (options 8 & 25 are only ones implemented).
    # Algorithm used for duplicate removal of 2D tracks produced by r-phi HT. Assumed to run before tracks are output from HT board.
    DupTrkAlgRphi = cms.uint32(0),
    #DupTrkAlgRphi = cms.uint32(25),
    # Algorithm run on all 3D tracks within each sector after r-z track filter.
    # (Ignored if no r-z track filter is run).
    DupTrkAlg3D = cms.uint32(0),
    #DupTrkAlg3D = cms.uint32(8),
    # Algorithm run on tracks after the track helix fit has been done. (Ian's talk in 26/8/2016 meeting)
    # To use it, you must set DupTrkAlgRphi = DupTrkAlg3D = 0.
    DupTrkAlgFit   = cms.uint32(50),
    #--- Options used by individual algorithms.
    # Parameter for "inverse" OSU duplicate-removal algorithm
    # Specifies minimum number of common stubs in same number of layers to keep smaller candidate in comparison.
    DupTrkMinCommonHitsLayers = cms.uint32(5),
  ),

  #=== Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).

  TrackMatchDef = cms.PSet(
     #--- Three different ways to define if a tracking particle matches a reco track candidate. (Usually, set two of them to ultra loose).
     # Min. fraction of matched stubs relative to number of stubs on reco track.
     MinFracMatchStubsOnReco  = cms.double(-99.),
     # Min. fraction of matched stubs relative to number of stubs on tracking particle.
     MinFracMatchStubsOnTP    = cms.double(-99.),
     # Min. number of matched layers.
     MinNumMatchLayers        = cms.uint32(4),
     # Min. number of matched PS layers.
     MinNumMatchPSLayers      = cms.uint32(0),
     # Associate stub to TP only if the TP contributed to both its clusters? (If False, then associate even if only one cluster was made by TP).
     StubMatchStrict          = cms.bool(False)
  ),

  #=== Track Fitting Algorithm Settings.

  TrackFitSettings = cms.PSet(
     #
     #--- Options applicable to all track fitters ---
     #
     # Track Fitting algortihms to use. You can run several in parallel.
     # TrackFitLinearAlgo & ChiSquared* are chi2 fits, KF* is a Kalman filter fit, 
     # & SimpleLR is a linear regression fit that neglects the hit uncertainties. 
     # The number 4 or 5 in the name indicates if 4 or 5 helix parameters are fitted.
     # Options KF4ParamsComb, KF5ParamsComb or SimpleLR are the best ones.
     # KF*ParamsCombHLS is the HLS version of the code, which only works if linked with Vivado libraries.
     TrackFitters = cms.vstring(
                                # "ChiSquared4ParamsApprox",
                                # "SimpleLR",
                                # "KF4ParamsCombHLS",
                                # "KF5ParamsCombHLS",
                                "KF5ParamsComb",
                                "KF4ParamsComb"
                                ),
     # Indicate subset of fitters wanting r-z track filter to be run before them. (Irrelevant for those not specified in "TrackFitters"). 
     # Typically, Chi2 & LR fits work best with r-z filter & KF works best without it.
     UseRZfilter = cms.vstring(
                                "ChiSquared4ParamsApprox",
                                "SimpleLR"
                              ),
     # Print detailed summary of track fit performance at end of job (as opposed to a brief one). 
     DetailedFitOutput = cms.bool(False),
     #
     # Use MC truth to eliminate all fake tracks & all incorrect stubs assigned to tracks before doing fit. 
     TrackFitCheat = cms.bool(False),
     #
     #--- Options for chi2 track fitter ---
     #
     # Number of fitting iterations to undertake. (15 is not realistic in hardware, but is necessary to kill bad hits)
     NumTrackFitIterations = cms.uint32(15),
     # Optionally kill hit with biggest residuals in track fit (occurs after the first fit, so three iterations would have two killings). 
     KillTrackFitWorstHit  = cms.bool(True),
     # Cuts in standard deviations used to kill hits with big residuals during fit. If the residual exceeds the "General" cut, the hit is killed providing it leaves the track with enough hits to survive. If the residual exceeds the "Killing" cut, the hit is killed even if that kills the track.
     GeneralResidualCut = cms.double(3.0),
     KillingResidualCut = cms.double(20.0),
     #
     #--- Additional options for Thomas Schuh's Linear Regression track fitter ---
     #
     # Maximum allowed number of iterations of LR fitter.
     MaxIterationsLR                 = cms.uint32( 8 ),
     # Internal histograms are filled if it is True
     LRFillInternalHists             = cms.bool(False),
     # If False: residual of a stub is the max of its r-phi & r-z residuals. 
     # If True: the residual is the mean of these residuals.
     CombineResiduals                = cms.bool( True ),
     # Correct stub phi coordinate for higher orders in circle expansion, so that a trajectory is straight in r-phi.
     LineariseStubPosition           = cms.bool( True ),
     # Checks if the fitted track is consistent with the sector, if not it will be not accepted.
     CheckSectorConsistency          = cms.bool( False ),
     # Checks if the fitted track r phi parameter  are consistent with the HT candidate parameter within in range of +- 2 cells.
     CheckHTCellConsistency          = cms.bool( False ),
     # Tracks must have stubs in at least this number of PS layers.
     MinPSLayers                     = cms.uint32( 2 ),
     # Digitization 
     DigitizeLR      = cms.bool( False ),
     PhiPrecision    = cms.double( 0.009 / 108. ),
     RPrecision      = cms.double( 0.14 ),
     ZPrecision      = cms.double( 0.28 ),
     ZSlopeWidth     = cms.uint32( 11 ),
     ZInterceptWidth = cms.uint32( 11 ),
     #
     #--- Additional options for Davide Cieri's Simple Linear Regression track fitter ---
     #
     # Digitize Simple Linear Regression variables and calculation. (Disabled if EnableDigitize=False).
     DigitizeSLR         = cms.bool(True),
     # Number of bits to be used in hardware to compute the division needed to calculate the helix  params
     DividerBitsHelix    = cms.uint32(23),
     DividerBitsHelixZ   = cms.uint32(23),
     # Number of bits to reduce the rphi helix parameter calculation weight 
     ShiftingBitsDenRPhi        = cms.uint32(14),
     # Number of bits to reduce the rphi helix parameter calculation weight 
     ShiftingBitsDenRZ        = cms.uint32(14),
     # Number of bits to reduce the phi0 parameter calculation weight 
     ShiftingBitsPhi      = cms.uint32(10),
     # Number of bits to reduce the qOverPt parameter calculation weight 
     ShiftingBitsPt      = cms.uint32(3),
     # Number of bits to reduce the tanLambda parameter calculation weight 
     ShiftingBitsLambda  = cms.uint32(1),
     # Number of bits to reduce the z0 parameter calculation weight
     ShiftingBitsZ0      = cms.uint32(16),
     # Fit ChiSquare Cut (tightening reduces fake track rate at cost of efficiency)
     SLR_chi2cut         = cms.double(100.),
     # Cut on Rphi Residuals (radians) - stubs killed until only 4 left or all have residuals below this cut.
     ResidualCut         = cms.double(0.0),
     #ResidualCut         = cms.double(0.0005), # This allows more than 4 stubs per track.
     #
     #--- Options for Kalman filter track fitters ---
     #
     # Larger number has more debug printout. "1" is useful for understanding why tracks are lost, best combined with TrackFitCheat=True.
     KalmanDebugLevel        = cms.uint32(0),
     # Internal histograms are filled if it is True
     KalmanFillInternalHists  = cms.bool(False),
     # Fit will reject fitted tracks unless it can assign at least this number of stubs to them.
     KalmanMinNumStubs       = cms.uint32(4),
     # Fit will attempt to add up to this nummber of stubs to each fitted tracks, but won't bother adding more.
     KalmanMaxNumStubs       = cms.uint32(4),
     # For 5-param helix fits, calculate also beam-constrained helix params after fit is complete, & use them for duplicate removal if DupTrkAlgFit=50.
     KalmanAddBeamConstr     = cms.bool(True),
     # Remove requirement of at least 2 PS layers per track.
     KalmanRemove2PScut      = cms.bool(False),
     # Allow the KF to skip this many layers in total per track.
     KalmanMaxSkipLayersHard = cms.uint32(1), # For HT tracks with many stubs
     KalmanMaxSkipLayersEasy = cms.uint32(2), # For HT tracks with few stubs
     KalmanMaxStubsEasy      = cms.uint32(10), # Max stubs an HT track can have to be "easy".
     # KF will consider at most this #stubs per layer to save time.
     KalmanMaxStubsPerLayer  = cms.uint32(4),
     # Multiple scattering term - inflate hit phi errors by this divided by Pt
     # (0.00075 gives best helix resolution & 0.00450 gives best chi2 distribution).
     KalmanMultiScattTerm    = cms.double(0.00075), 
     # Scale down chi2 in r-phi plane by this factor to improve electron performance (should be power of 2)
     KalmanChi2RphiScale     = cms.uint32(8),
     # N.B. KF track fit chi2 cut is not cfg param, but instead is hard-wired in KF4ParamsComb::isGoodState(...).
     #--- Enable Higher order corrections
     # Treat z uncertainty in tilted barrel modules correctly.
     KalmanHOtilted          = cms.bool(False),
     # Higher order circle explansion terms for low Pt.
     KalmanHOhelixExp        = cms.bool(False),
     # Alpha correction for non-radial 2S endcap strips. (0=disable correction, 1=correct with offset, 2=correct with non-diagonal stub covariance matrix). -- Option 1 is easier in FPGA, but only works if fit adds PS stubs before 2S ones.
     KalmanHOalpha           = cms.uint32(0),
     # Projection from (r,phi) to (z,phi) for endcap 2S modules. (0=disable correction, 1=correct with offset, 2=correct with non-diagonal stub covariance matrix). -- Option 1 is easier in FPGA, but only works if fit adds PS stubs before 2S ones.
     KalmanHOprojZcorr       = cms.uint32(0),
     # Use dodgy calculation to account for non-radial endcap 2S modules that was used in Dec. 2016 demonstrator & use no special treatment for tilted modules.
     KalmanHOdodgy           = cms.bool(True)
  ),

  #=== Treatment of dead modules.

  DeadModuleOpts = cms.PSet( 
     #--- Either use this private TMTT way of studying dead modules
     # In (eta,phi) sectors containing dead modules, reduce the min. number of layers cut on tracks to (MinStubLayers - 1)?
     # The sectors affected are hard-wired in DeadModuleDB::defineDeadTrackerRegions().
     DeadReduceLayers  = cms.bool( False ),
     # Emulate dead modules by killing fraction of stubs given by DeadSimulateFrac in certain layers & angular regions of 
     # the tracker that are hard-wired in DeadModuleDB::defineDeadSectors(). Disable by setting <= 0. Fully enable by setting to 1.
     # Do not use if KillScenario > 0.
     DeadSimulateFrac = cms.double(-999.),
     #
     #--- Or this use communal way developed with Tracklet of studying dead modules
     # Emulate dead/inefficient modules using the StubKiller code, with stubs killed according to the scenarios of the Stress Test group. 
     # (0=Don't kill any stubs; 1-5 = Scenarios from https://github.com/EmyrClement/StubKiller/blob/master/README.md).
     KillScenario = cms.uint32(0),
     # Modify TMTT tracking to try to recover tracking efficiency in presence of dead modules. (Does nothing if KillScenario = 0).
     KillRecover = cms.bool (True)
  ),

  #=== Fitted track digitisation.

  TrackDigi=cms.PSet(
    # For firmware reasons, can't use common digitisation cfg for all fitters.

    #======= SimpleLR digi parameters ========
    SLR_skipTrackDigi = cms.bool( False ), # Optionally skip track digitisation if done internally inside fitting code.
    SLR_oneOver2rBits = cms.uint32(13),
    SLR_oneOver2rRange = cms.double(0.01354135),
    SLR_d0Bits = cms.uint32(12), # Made up by Ian as never yet discussed.
    SLR_d0Range  = cms.double(10.),
    SLR_phi0Bits = cms.uint32(18),
    SLR_phi0Range = cms.double(1.3962636), # phi0 is actually only digitised relative to centre of sector.
    SLR_z0Bits = cms.uint32(12),
    SLR_z0Range  = cms.double(51.555509),
    SLR_tanlambdaBits = cms.uint32(15),
    SLR_tanlambdaRange = cms.double(32.0),
    SLR_chisquaredBits = cms.uint32(8),
    SLR_chisquaredRange = cms.double(128.),
    
    #====== Kalman Filter digi parameters ========
    KF_skipTrackDigi = cms.bool( False ), # Optionally skip track digitisation if done internally inside fitting code.
    KF_oneOver2rBits = cms.uint32(15),
    KF_oneOver2rRange = cms.double(0.0076171313), # pT > 1.5 GeV
    KF_d0Bits = cms.uint32(12),
    KF_d0Range  = cms.double(31.992876),
    KF_phi0Bits = cms.uint32(12),
    KF_phi0Range = cms.double(0.6981317),  # phi0 digitised relative to centre of sector. (Required range 2pi/18 + 2*overlap; overlap = 0.19206rads*(2GeV/ptCut)*(chosenR/67.24). MUST DOUBLE TO GO TO 2 GEV.
    KF_z0Bits = cms.uint32(12),
    KF_z0Range  = cms.double(45.826419),
    KF_tanlambdaBits = cms.uint32(16),
    KF_tanlambdaRange = cms.double(16.),
    KF_chisquaredBits = cms.uint32(15), # N.B. 17 bits are used internally inside KF.
    KF_chisquaredRange = cms.double(1024.),
    KF_chisquaredBinEdges = cms.vdouble(0, 0.5, 1, 2, 3, 5, 7, 10, 20, 40, 100, 200, 500, 1000, 3000 ), # Additional bin for >3000
    KF_bendchisquaredBinEdges = cms.vdouble(0, 0.5, 1, 2, 3, 5, 10, 50 ), # Additional bin for >50

    #====== Other track fitter Digi params.
    # Currently equal to those for KF, although you can skip track digitisation for them with following.
    Other_skipTrackDigi = cms.bool( True ) 
  ),

  #===== Use HYBRID TRACKING (Tracklet pattern reco + TMTT KF -- requires tracklet C++ too) =====
  
  Hybrid = cms.bool( False),

  #===== Debug printout & plots
  Debug  = cms.uint32(1), #(0=none, 1=print #tracks/event, 2+ print more info)
  # When making helix parameter resolution plots, only use particles from the physics event (True)
  # or also use particles from pileup (False) ?
  ResPlotOpt = cms.bool (True),
  # Specify sector for which debug histos for hexagonal HT will be made.
  iPhiPlot = cms.uint32(0),
  iEtaPlot = cms.uint32(9)
)
