from __future__ import print_function
import sys
import FWCore.ParameterSet.Config as cms

def customlog(s):
    print("# MSG-i trackselectionRefitting:  %s" % s)

def getSequence(process, collection,
                saveCPU = False,
                TTRHBuilder = "WithAngleAndTemplate",
                usePixelQualityFlag = None,
                openMassWindow = False,
                cosmicsDecoMode = False,
                cosmicsZeroTesla = True,
                momentumConstraint = None,
                cosmicTrackSplitting = False,
                isPVValidation = False,
                use_d0cut = True,
                g4Refitting = False):
    """This function returns a cms.Sequence containing as last element the
    module 'FinalTrackRefitter', which can be used as cms.InputTag for
    subsequent processing steps.
    The modules in the sequence are already attached to the given `process`
    object using the given track collection `collection` and the given
    optional arguments.

    Arguments:
    - `process`: 'cms.Process' object to which the modules of the sequence will
                 be attached.
    - `collection`: String indicating the input track collection.
    - `saveCPU`: If set to 'True', some steps are merged to reduce CPU time.
                 Reduces a little the accuracy of the results.
                 This option is currently not recommended.
    - `TTRHBuilder`: Option used for the Track(Re)Fitter modules.
    - `usePixelQualityFlag`: Option used for the TrackHitFilter module.
                             Defaults to 'True' but is automatically set to
                             'False' if a `TTRHBuilder` without templates is
                             used.
                             If this is still wanted for some reason, one can
                             explicitely specify it as 'True'.
    - `openMassWindow`: Used to configure the TwoBodyDecaySelector for ZMuMu.
    - `cosmicsDecoMode`: If set to 'True' a lower Signal/Noise cut is used.
    - `cosmicsZeroTesla`: If set to 'True' a 0T-specific selection is used.
    - `momentumConstraint`: If you want to apply a momentum constraint for the
                            track refitting, e.g. for CRUZET data, you need
                            to provide here the name of the constraint module.
    - `cosmicTrackSplitting`: If set to 'True' cosmic tracks are split before the
                              second track refitter.
    - `isPVValidation`: If set to 'True' most of the selection cuts are overridden
                        to allow unbiased selection of tracks for vertex refitting 
    - `use_d0cut`: If 'True' (default), apply a cut |d0| < 50.
    """

    ###################################################
    # resolve default values incl. consistency checks #
    ###################################################

    customlog("g4Refitting=%s" % g4Refitting)

    if usePixelQualityFlag is None:
        if "Template" not in TTRHBuilder:
            usePixelQualityFlag = False # not defined without templates
            customlog("Using 'TTRHBuilder' without templates %s" % TTRHBuilder)
            customlog(" --> Turning off pixel quality flag in hit filter.")
        else:
            usePixelQualityFlag = True # default for usage with templates


    #############################
    ## setting general options ##
    #############################

    options = {"TrackHitFilter": {},
               "TrackFitter": {},
               "TrackRefitter": {},
               "TrackSelector": {},
               "geopro": {} }

    options["TrackSelector"]["HighPurity"] = {
        "trackQualities": ["highPurity"],
        "filter": True,
        "etaMin": -3.0,
        "etaMax": 3.0,
        "pMin": 8.0
        }
    options["TrackSelector"]["Alignment"] = {
        "filter": True,
        "pMin": 3.0,
        "nHitMin2D": 2,
        "d0Min": -50.0,
        "d0Max": 50.0,
        "etaMin": -3.0,
        "etaMax": 3.0,
        "nHitMin": 8,
        "chi2nMax": 9999.0
        }
    options["TrackRefitter"]["First"] = {
        "NavigationSchool": "",
        "TTRHBuilder": TTRHBuilder,
        }
    options["TrackRefitter"]["Second"] = {
        "NavigationSchool": "",
        "TTRHBuilder": TTRHBuilder,
        }
    options["TrackHitFilter"]["Tracker"] = {
        "useTrajectories": True,
        "minimumHits": 8,
        "commands": cms.vstring("keep PXB", "keep PXE", "keep TIB", "keep TID",
                                "keep TOB", "keep TEC"),
        "replaceWithInactiveHits": True,
        "rejectBadStoNHits": True,
        "rejectLowAngleHits": True,
        "usePixelQualityFlag": usePixelQualityFlag,
        "StoNcommands": cms.vstring("ALL 12.0"),
        "TrackAngleCut": 0.087,
        }
    options["TrackFitter"]["HitFilteredTracks"] = {
        "NavigationSchool": "",
        "TTRHBuilder": TTRHBuilder,
        }
    options["geopro"][""] = {
        }

    if g4Refitting:
        options["TrackRefitter"]["Second"] = {
            "AlgorithmName" : cms.string('undefAlgorithm'),
            "Fitter" : cms.string('G4eFitterSmoother'),
            "GeometricInnerState" : cms.bool(False),
            "MeasurementTracker" : cms.string(''),
            "MeasurementTrackerEvent" : cms.InputTag("MeasurementTrackerEvent"),
            "NavigationSchool" : cms.string('SimpleNavigationSchool'),  # Correct?
            "Propagator" : cms.string('Geant4ePropagator'),
            "TTRHBuilder" : cms.string('WithAngleAndTemplate'),
            "TrajectoryInEvent" : cms.bool(True),
            "beamSpot" : cms.InputTag("offlineBeamSpot"),
            "constraint" : cms.string(''),
            "src" : cms.InputTag("AlignmentTrackSelector"),
            "srcConstr" : cms.InputTag(""),
            "useHitsSplitting" : cms.bool(False),
            "usePropagatorForPCA" : cms.bool(True)   # not sure whether it is needed
        }

    #########################################
    ## setting collection specific options ##
    #########################################
    isCosmics = False

    if collection in ("ALCARECOTkAlMinBias", "generalTracks",
                      "ALCARECOTkAlMinBiasHI", "hiGeneralTracks",
                      "ALCARECOTkAlJetHT", "ALCARECOTkAlDiMuonVertexTracks"):
        options["TrackSelector"]["Alignment"].update({
                "ptMin": 1.0,
                "pMin": 8.,
                })
        options["TrackHitFilter"]["Tracker"].update({
                "minimumHits": 10,
                })
    elif collection in ("ALCARECOTkAlCosmicsCTF0T",
                        "ALCARECOTkAlCosmicsCosmicTF0T",
                        "ALCARECOTkAlCosmicsInCollisions"):
        isCosmics = True
        options["TrackSelector"]["HighPurity"] = {} # drop high purity cut
        if not cosmicsDecoMode:
            options["TrackHitFilter"]["Tracker"].update({
                    "StoNcommands": cms.vstring("ALL 18.0")
                    })
        if cosmicsZeroTesla:
            options["TrackHitFilter"]["Tracker"].update({
                    "TrackAngleCut": 0.1 # Run-I: 0.087 for 0T
                    })
        else:
            options["TrackHitFilter"]["Tracker"].update({
                    "TrackAngleCut": 0.1 # Run-I: 0.35 for 3.8T
                    })
        options["TrackSelector"]["Alignment"].update({
                "pMin": 4.0,
                "etaMin": -99.0,
                "etaMax": 99.0,
                "applyMultiplicityFilter": True,
                "maxMultiplicity": 1
                })
        if cosmicTrackSplitting:
            options["TrackSplitting"] = {}
            options["TrackSplitting"]["TrackSplitting"] = {}
        if not use_d0cut:
            options["TrackSelector"]["Alignment"].update({
                    "d0Min": -99999.0,
                    "d0Max": 99999.0,
                    })
    elif collection in ("ALCARECOTkAlMuonIsolated",
                        "ALCARECOTkAlMuonIsolatedHI",
                        "ALCARECOTkAlMuonIsolatedPA"):
        options["TrackSelector"]["Alignment"].update({
                ("minHitsPerSubDet", "inPIXEL"): 1,
                "ptMin": 5.0,
                "nHitMin": 10,
                "applyMultiplicityFilter": True,
                "maxMultiplicity": 1,
                })
    elif collection in ("ALCARECOTkAlZMuMu",
                        "ALCARECOTkAlZMuMuHI",
                        "ALCARECOTkAlZMuMuPA",
                        "ALCARECOTkAlDiMuon"):
        options["TrackSelector"]["Alignment"].update({
                "ptMin": 15.0,
                "etaMin": -3.0,
                "etaMax": 3.0,
                "nHitMin": 10,
                "applyMultiplicityFilter": True,
                "minMultiplicity": 2,
                "maxMultiplicity": 2,
                ("minHitsPerSubDet", "inPIXEL"): 1,
                ("TwoBodyDecaySelector", "applyChargeFilter"): True,
                ("TwoBodyDecaySelector", "charge"): 0,
                ("TwoBodyDecaySelector",
                 "applyMassrangeFilter"): not openMassWindow,
                ("TwoBodyDecaySelector", "minXMass"): 85.8,
                ("TwoBodyDecaySelector", "maxXMass"): 95.8,
                ("TwoBodyDecaySelector", "daughterMass"): 0.105
                })
        options["TrackHitFilter"]["Tracker"].update({
                "minimumHits": 10,
                })
    elif collection == "ALCARECOTkAlUpsilonMuMu":
        options["TrackSelector"]["Alignment"].update({
                "ptMin": 3.0,
                "etaMin": -2.4,
                "etaMax": 2.4,
                "nHitMin": 10,
                "applyMultiplicityFilter": True,
                "minMultiplicity": 2,
                "maxMultiplicity": 2,
                ("minHitsPerSubDet", "inPIXEL"): 1,
                ("TwoBodyDecaySelector", "applyChargeFilter"): True,
                ("TwoBodyDecaySelector", "charge"): 0,
                ("TwoBodyDecaySelector",
                 "applyMassrangeFilter"): not openMassWindow,
                ("TwoBodyDecaySelector", "minXMass"): 9.2,
                ("TwoBodyDecaySelector", "maxXMass"): 9.7,
                ("TwoBodyDecaySelector", "daughterMass"): 0.105
                })
        options["TrackHitFilter"]["Tracker"].update({
                "minimumHits": 10,
                })
    elif collection == "ALCARECOTkAlJpsiMuMu":
        options["TrackSelector"]["Alignment"].update({
                "ptMin": 1.0,
                "etaMin": -2.4,
                "etaMax": 2.4,
                "nHitMin": 10,
                "applyMultiplicityFilter": True,
                "minMultiplicity": 2,
                "maxMultiplicity": 2,
                ("minHitsPerSubDet", "inPIXEL"): 1,
                ("TwoBodyDecaySelector", "applyChargeFilter"): True,
                ("TwoBodyDecaySelector", "charge"): 0,
                ("TwoBodyDecaySelector",
                 "applyMassrangeFilter"): not openMassWindow,
                ("TwoBodyDecaySelector", "minXMass"): 2.7,
                ("TwoBodyDecaySelector", "maxXMass"): 3.4,
                ("TwoBodyDecaySelector", "daughterMass"): 0.105
                })
        options["TrackHitFilter"]["Tracker"].update({
                "minimumHits": 10,
                })
    else:
        raise ValueError("Unknown input track collection: {}".format(collection))

    if cosmicTrackSplitting and not isCosmics:
        raise ValueError("Can only do cosmic track splitting for cosmics.")



    ####################
    ## save CPU time? ##
    ####################

    if saveCPU:
        if cosmicTrackSplitting:
            raise ValueError("Can't turn on both saveCPU and cosmicTrackSplitting at the same time")
        mods = [("TrackSelector", "Alignment", {"method": "load"}),
                ("TrackRefitter", "First", {"method": "load",
                                            "clone": True}),
                ("TrackHitFilter", "Tracker", {"method": "load"}),
                ("TrackFitter", "HitFilteredTracks", {"method": "import"})]
        options["TrackSelector"]["Alignment"].update(
            options["TrackSelector"]["HighPurity"])
    elif cosmicTrackSplitting:
        mods = [("TrackRefitter", "First", {"method": "load",
                                            "clone": True}),
                ("TrackSelector", "Alignment", {"method": "load"}),
                ("TrackSplitting", "TrackSplitting", {"method": "load"}),
                ("TrackFitter", "HitFilteredTracks", {"method": "import"}),
                ("TrackRefitter", "Second", {"method": "load",
                                             "clone": True})]
    elif g4Refitting:
        mods = [("TrackSelector", "HighPurity", {"method": "import"}),
                ("TrackRefitter", "First", {"method": "load",
                                            "clone": True}),
                ("TrackHitFilter", "Tracker", {"method": "load"}),
                ("TrackFitter", "HitFilteredTracks", {"method": "import"}),
                ("TrackSelector", "Alignment", {"method": "load"}),
                #("geopro","", {"method": "load"}),
                ("TrackRefitter", "Second", {"method": "load",
                                             "clone": True})]
        if isCosmics: mods = mods[1:] # skip high purity selector for cosmics
    else:
        mods = [("TrackSelector", "HighPurity", {"method": "import"}),
                ("TrackRefitter", "First", {"method": "load",
                                            "clone": True}),
                ("TrackHitFilter", "Tracker", {"method": "load"}),
                ("TrackFitter", "HitFilteredTracks", {"method": "import"}),
                ("TrackSelector", "Alignment", {"method": "load"}),
                ("TrackRefitter", "Second", {"method": "load",
                                             "clone": True})]
        if isCosmics: mods = mods[1:] # skip high purity selector for cosmics

    #############################
    ## PV Validation cuts tune ##                 
    #############################

    if isPVValidation:
        options["TrackSelector"]["HighPurity"].update({
                "trackQualities": [],
                "pMin": 0.
                })
        options["TrackSelector"]["Alignment"].update({
                "pMin" :      0.,      
                "ptMin" :     0.,       
                "nHitMin2D" : 0,       
                "nHitMin"   : 0,       
                "d0Min" : -999999.0,
                "d0Max" :  999999.0,
                "dzMin" : -999999.0,
                "dzMax" :  999999.0
                })

    ################################
    ## apply momentum constraint? ##
    ################################

    if momentumConstraint is not None:
        for mod in options["TrackRefitter"]:
            momconstrspecs = momentumConstraint.split(',')
            if len(momconstrspecs)==1:
                options["TrackRefitter"][mod].update({
                    "constraint": "momentum",
                    "srcConstr": momconstrspecs[0]
                    })
            else:
                options["TrackRefitter"][mod].update({
                    "constraint": momconstrspecs[1],
                    "srcConstr": momconstrspecs[0]
                    })



    #######################################################
    # load offline beam spot module required by the refit #
    #######################################################
    process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

    ###############################
    ## put the sequence together ##
    ###############################

    modules = []
    src = collection
    prevsrc = None
    for mod in mods[:-1]:
        src, prevsrc = _getModule(process, src, mod[0], "".join(reversed(mod[:-1])),
                                  options[mod[0]][mod[1]], isCosmics = isCosmics, prevsrc = prevsrc,
                                  **(mod[2])), src
        modules.append(getattr(process, src))
    else:
        if mods[-1][-1]["method"] == "load" and \
                not mods[-1][-1].get("clone", False):
            customlog("Name of the last module needs to be modifiable.")
            sys.exit(1)

        if g4Refitting:
            customlog("Here we must include geopro first")
            process.load('Configuration.StandardSequences.GeometryDB_cff')
            process.load("TrackPropagation.Geant4e.geantRefit_cff")
            modules.append(getattr(process,"geopro"))

        src = _getModule(process, src, mods[-1][0], "FinalTrackRefitter",
                         options[mods[-1][0]][mods[-1][1]],
                         isCosmics = isCosmics, **(mods[-1][2]))
        modules.append(getattr(process, src))

    moduleSum = process.offlineBeamSpot        # first element of the sequence
    if g4Refitting:
        # g4Refitter needs measurements
        moduleSum += getattr(process,"MeasurementTrackerEvent")

    for module in modules:
        # Spply srcConstr fix here
        if hasattr(module,"srcConstr"):
           strSrcConstr = module.srcConstr.getModuleLabel()
           if strSrcConstr:
               procsrcconstr = getattr(process,strSrcConstr)
               if hasattr(procsrcconstr,"src"): # Momentum or track parameter constraints
                  if procsrcconstr.src != module.src:
                     module.srcConstr=''
                     module.constraint=''
                  else:
                     moduleSum += procsrcconstr # Add constraint
               elif hasattr(procsrcconstr,"srcTrk"): # Vertex constraint
                  if procsrcconstr.srcTrk != module.src:
                     module.srcConstr=''
                     module.constraint=''
                  else:
                     procsrcconstrsrcvtx = getattr(process,procsrcconstr.srcVtx.getModuleLabel())
                     if type(procsrcconstrsrcvtx) is cms.EDFilter: # If source of vertices is itself a filter (e.g. good PVs)
                        procsrcconstrsrcvtxprefilter = getattr(process,procsrcconstrsrcvtx.src.getModuleLabel())
                        moduleSum += procsrcconstrsrcvtxprefilter # Add vertex source to constraint before filter
                     moduleSum += procsrcconstrsrcvtx # Add vertex source to constraint
                     moduleSum += procsrcconstr # Add constraint

        moduleSum += module # append the other modules

    return cms.Sequence(moduleSum)

###############################
###############################
###                         ###
###   Auxiliary functions   ###
###                         ###
###############################
###############################


def _getModule(process, src, modType, moduleName, options, **kwargs):
    """General function for attaching the module of type `modType` to the
    cms.Process `process` using `options` for customization and `moduleName` as
    the name of the new attribute of `process`.

    Arguments:
    - `process`: 'cms.Process' object to which the module is attached.
    - `src`: cms.InputTag for this module.
    - `modType`: Type of the requested module.
    - `options`: Dictionary with customized values for the module's options.
    - `**kwargs`: Used to supply options at construction time of the module.
    """

    objTuple = globals()["_"+modType](kwargs)
    method = kwargs.get("method")
    if method == "import":
        __import__(objTuple[0])
        obj = getattr(sys.modules[objTuple[0]], objTuple[1]).clone()
    elif method == "load":
        process.load(objTuple[0])
        if kwargs.get("clone", False):
            obj = getattr(process, objTuple[1]).clone(src=src)
        else:
            obj = getattr(process, objTuple[1])
            moduleName = objTuple[1]
    else:
        customlog("Unknown method: %s" % method)
        sys.exit(1)

    if modType == "TrackSplitting":
        #track splitting takes the TrackSelector as tracks
        # and the first TrackRefitter as tjTkAssociationMapTag
        _customSetattr(obj, "tracks", src)
        _customSetattr(obj, "tjTkAssociationMapTag", kwargs["prevsrc"])
    else:
        obj.src = src

    for option in options:
        _customSetattr(obj, option, options[option])

    if moduleName is not objTuple[1]:
        setattr(process, moduleName, obj)
    return moduleName


def _TrackHitFilter(kwargs):
    """Returns TrackHitFilter module name.

    Arguments:
    - `kwargs`: Not used in this function.
    """

    return ("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff",
            "TrackerTrackHitFilter")


def _TrackSelector(kwargs):
    """Returns TrackSelector module name.

    Arguments:
    - `kwargs`: Not used in this function.
    """

    return ("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi",
            "AlignmentTrackSelector")


def _TrackFitter(kwargs):
    """Returns TrackFitter module name.

    Arguments:
    - `kwargs`: Used to supply options at construction time of the object.
    """

    isCosmics = kwargs.get("isCosmics", False)
    if isCosmics:
        return ("RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff",
                "ctfWithMaterialTracksCosmics")
    else:
        return ("RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff",
                "ctfWithMaterialTracks")


def _TrackRefitter(kwargs):
    """Returns TrackRefitter module name.

    Arguments:
    - `kwargs`: Used to supply options at construction time of the object.
    """

    isCosmics = kwargs.get("isCosmics", False)
    if isCosmics:
        return ("RecoTracker.TrackProducer.TrackRefitters_cff",
                "TrackRefitterP5")
    else:
        return ("RecoTracker.TrackProducer.TrackRefitters_cff",
                "TrackRefitter")

def _TrackSplitting(kwargs):
    return ("RecoTracker.FinalTrackSelectors.cosmicTrackSplitter_cfi",
            "cosmicTrackSplitter")

def _geopro(kwargs):
    return ("TrackPropagation.Geant4e.geantRefit_cff","geopro")


def _customSetattr(obj, attr, val):
    """Sets the attribute `attr` of the object `obj` using the value `val`.
    `attr` can be a string or a tuple of strings, if one wants to set an
    attribute of an attribute, etc.

    Arguments:
    - `obj`: Object, which must have a '__dict__' attribute.
    - `attr`: String or tuple of strings describing the attribute's name.
    - `val`: value of the attribute.
    """

    if isinstance(attr, tuple) and len(attr) > 1:
        _customSetattr(getattr(obj, attr[0]), attr[1:], val)
    else:
        if isinstance(attr, tuple): attr = attr[0]
        setattr(obj, attr, val)

