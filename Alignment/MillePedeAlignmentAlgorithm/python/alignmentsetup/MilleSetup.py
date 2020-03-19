from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms


def setup(process, input_files, collection,
          json_file = "",
          cosmics_zero_tesla = False,
          cosmics_deco_mode = True,
          TTRHBuilder = None):
    """Mille-specific setup.

    Arguments:
    - `process`: cms.Process object
    - `input_files`: input file list -> cms.untracked.vstring()
    - `collection`: track collection to be used
    - `cosmics_zero_tesla`: triggers the corresponding track selection
    - `cosmics_deco_mode`: triggers the corresponding track selection
    - `TTRHBuilder`: TransientTrackingRecHitBuilder used in the track selection
                     and refitting sequence;
                     defaults to the default of the above-mentioned sequence
    """

    # no database output in the mille step:
    # --------------------------------------------------------------------------
    process.AlignmentProducer.saveToDB = False
    process.AlignmentProducer.saveApeToDB = False
    process.AlignmentProducer.saveDeformationsToDB = False


    # align calibrations to general settings
    # --------------------------------------------------------------------------
    for calib in process.AlignmentProducer.calibrations:
        calib.saveToDB       = process.AlignmentProducer.saveToDB
        calib.treeFile       = process.AlignmentProducer.algoConfig.treeFile
        calib.mergeTreeFiles = process.AlignmentProducer.algoConfig.mergeTreeFiles


    # Track selection and refitting
    # --------------------------------------------------------------------------
    import Alignment.CommonAlignment.tools.trackselectionRefitting as trackRefitter
    kwargs = {"cosmicsDecoMode": cosmics_deco_mode,
              "cosmicsZeroTesla": cosmics_zero_tesla}
    if TTRHBuilder is not None: kwargs["TTRHBuilder"] = TTRHBuilder
    process.TrackRefittingSequence \
        = trackRefitter.getSequence(process, collection, **kwargs)


    # Ensure the correct APV mode for cosmics
    # --------------------------------------------------------------------------
    if collection in ("ALCARECOTkAlCosmicsCTF0T",
                      "ALCARECOTkAlCosmicsInCollisions"):
        process.load("Alignment.CommonAlignment.apvModeFilter_cfi")
        process.apvModeFilter.apvMode = "deco" if cosmics_deco_mode else "peak"
        from . import helper
        helper.add_filter(process, process.apvModeFilter)


    # Configure the input data
    # --------------------------------------------------------------------------
    process.source = cms.Source("PoolSource", fileNames  = input_files)

    # Set Luminosity-Blockrange from json-file if given
    if (json_file != "") and (json_file != "placeholder_json"):
        import FWCore.PythonUtilities.LumiList as LumiList
        lumi_list = LumiList.LumiList(filename = json_file).getVLuminosityBlockRange()
        process.source.lumisToProcess = lumi_list


    # The executed path
    # --------------------------------------------------------------------------
    process.p = cms.Path(process.TrackRefittingSequence*
                         process.AlignmentProducer)
    if hasattr(process, "mps_filters"): process.p.insert(0, process.mps_filters)
