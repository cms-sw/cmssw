import sys
import FWCore.ParameterSet.Config as cms



def getSequence(process, collection,
                saveCPU = False,
                TTRHBuilder = "WithAngleAndTemplate",
                usePixelQualityFlag = True,
                openMassWindow = False,
                cosmicsDecoMode = False,
                cosmicsZeroTesla = True,
                momentumConstraint = None):
    """This function returns a cms.Sequence containing as last element the
    module 'FinalTrackRefitter', which can be used as cms.InputTag for
    subsequent processing steps.
    The modules in the sequence are already attached to the given `process`
    object using the given track collection `collection` and the given
    optionial arguments.

    Arguments:
    - `process`: 'cms.Process' object to which the modules of the sequence will
                 be attached.
    - `collection`: String indicating the input track collection.
    - `saveCPU`: If set to 'True', some steps are merged to reduce CPU time.
                 Reduces a little the accuracy of the results.
                 This option is currently not recommended.
    - `TTRHBuilder`: Option used for the Track(Re)Fitter modules.
    - `usePixelQualityFlag`: Option used for the TrackHitFilter module.
    - `openMassWindow`: Used to configure the TwoBodyDecaySelector for ZMuMu.
    - `cosmicsDecoMode`: If set to 'True' a lower Signal/Noise cut is used.
    - `cosmicsZeroTesla`: If set to 'True' a 0T-specific selection is used.
    - `momentumConstraint`: If you want to apply a momentum constraint for the
                            track refitting, e.g. for CRUZET data, you need
                            to provide here the name of the constraint module.
    """


    #############################
    ## setting general options ##
    #############################

    options = {"TrackHitFilter": {},
               "TrackFitter": {},
               "TrackRefitter": {},
               "TrackSelector": {}}

    options["TrackSelector"]["HighPurity"] = {
        "trackQualities": ["highPurity"],
        "filter": True,
        "etaMin": -3.0,
        "etaMax": 3.0
        }
    options["TrackSelector"]["Alignment"] = {
        "filter": True,
        "pMin": 3.0,
        "nHitMin2D": 3,
        "d0Min": -50.0,
        "d0Max": 50.0,
        "etaMin": -3.0,
        "etaMax": 3.0,
        "nHitMin": 8,
        "chi2nMax": 9999.0
        }
    options["TrackRefitter"]["First"] = {
        "NavigationSchool": "",
        }
    options["TrackRefitter"]["Second"] = {
        "NavigationSchool": "",
        "TTRHBuilder": TTRHBuilder
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
        "TrackAngleCut": 0.17
        }
    options["TrackFitter"]["HitFilteredTracks"] = {
        "NavigationSchool": "",
        "TTRHBuilder": TTRHBuilder
        }


    #########################################
    ## setting collection specific options ##
    #########################################
    isCosmics = False

    if collection is "ALCARECOTkAlMinBias":
        options["TrackSelector"]["Alignment"].update({
                "ptMin": 1.0,
                })
    elif collection is "ALCARECOTkAlCosmicsCTF0T":
        isCosmics = True
        options["TrackSelector"]["HighPurity"] = {} # drop high purity cut
        if not cosmicsDecoMode:
            options["TrackHitFilter"]["Tracker"].update({
                    "StoNcommands": cms.vstring("ALL 18.0")
                    })
        if cosmicsZeroTesla:
            options["TrackHitFilter"]["Tracker"].update({
                    "TrackAngleCut": 0.087 # Run-I: 0.087 for 0T
                    })
        else:
            options["TrackHitFilter"]["Tracker"].update({
                    "TrackAngleCut": 0.087 # Run-I: 0.35 for 3.8T
                    })
        options["TrackSelector"]["Alignment"].update({
                "pMin": 4.0,
                "etaMin": -99.0,
                "etaMax": 99.0
                })
    elif collection is "ALCARECOTkAlMuonIsolated":
        options["TrackSelector"]["Alignment"].update({
                ("minHitsPerSubDet", "inPIXEL"): 1,
                })
    elif collection is "ALCARECOTkAlZMuMu":
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
                ("TwoBodyDecaySelector",
                 "applyMassrangeFilter"): not openMassWindow,
                ("TwoBodyDecaySelector", "minXMass"): 85.8,
                ("TwoBodyDecaySelector", "maxXMass"): 95.8
                })
        pass
    else:
        print "Unknown input track collection:", collection
        sys.exit(1)



    ####################
    ## save CPU time? ##
    ####################

    if saveCPU:
        mods = [("TrackSelector", "Alignment", {"method": "load"}),
                ("TrackRefitter", "First", {"method": "load",
                                            "clone": True}),
                ("TrackHitFilter", "Tracker", {"method": "load"}),
                ("TrackFitter", "HitFilteredTracks", {"method": "import"})]
        options["TrackSelector"]["Alignment"].update(
            options["TrackSelector"]["HighPurity"])
    else:
        mods = [("TrackSelector", "HighPurity", {"method": "import"}),
                ("TrackRefitter", "First", {"method": "load",
                                            "clone": True}),
                ("TrackHitFilter", "Tracker", {"method": "load"}),
                ("TrackFitter", "HitFilteredTracks", {"method": "import"}),
                ("TrackSelector", "Alignment", {"method": "load"}),
                ("TrackRefitter", "Second", {"method": "load",
                                             "clone": True})]
        if isCosmics: mods = mods[1:]



    ################################
    ## apply momentum constraint? ##
    ################################

    if momentumConstraint is not None:
        for mod in options["TrackRefitter"]:
            options["TrackRefitter"][mod].update({
                "constraint": "momentum",
                "srcConstr": momentumConstraint
                })



    ###############################
    ## put the sequence together ##
    ###############################

    modules = []
    src = collection
    for mod in mods[:-1]:
        src = _getModule(process, src, mod[0], "".join(reversed(mod[:-1])),
                         options[mod[0]][mod[1]], isCosmics = isCosmics,
                         **(mod[2]))
        modules.append(getattr(process, src))
    else:
        if mods[-1][-1]["method"] is "load" and \
                not mods[-1][-1].get("clone", False):
            print "Name of the last module needs to be modifiable."
            sys.exit(1)
        src = _getModule(process, src, mods[-1][0], "FinalTrackRefitter",
                         options[mods[-1][0]][mods[-1][1]],
                         isCosmics = isCosmics, **(mods[-1][2]))
        modules.append(getattr(process, src))

    moduleSum = modules[0]
    for mod in modules[1:]:
        moduleSum += mod
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
    if method is "import":
        __import__(objTuple[0])
        obj = getattr(sys.modules[objTuple[0]], objTuple[1]).clone(src=src)
    elif method is "load":
        process.load(objTuple[0])
        if kwargs.get("clone", False):
            obj = getattr(process, objTuple[1]).clone(src=src)
        else:
            obj = getattr(process, objTuple[1])
            obj.src = src
            moduleName = objTuple[1]
    else:
        print "Unknown method:", method
        sys.exit(1)

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


def _customSetattr(obj, attr, val):
    """Sets the attribute `attr` of the object `obj` using the value `val`.
    `attr` can be a string or a tuple of strings, if one wants to set an
    attribute of an attribute, etc.

    Arguments:
    - `obj`: Object, which must have a '__dict__' attribute.
    - `attr`: String or tuple of strings describing the attribute's name.
    - `val`: value of the attribute.
    """

    if type(attr) is tuple and len(attr) > 1:
        _customSetattr(getattr(obj, attr[0]), attr[1:], val)
    else:
        if type(attr) is tuple: attr = attr[0]
        setattr(obj, attr, val)
