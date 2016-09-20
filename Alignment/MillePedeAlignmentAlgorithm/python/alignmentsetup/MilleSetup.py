import FWCore.ParameterSet.Config as cms


def setup(process, input_files, collection,
          json_file = "", cosmics_zero_tesla = False, cosmics_deco_mode = True):
    """Mille-specific setup.

    Arguments:
    - `process`: cms.Process object
    - `input_files`: input file list -> cms.untracked.vstring()
    - `collection`: track collection to be used
    - `cosmics_zero_tesla`: triggers the corresponding track selection
    - `cosmics_deco_mode`: triggers the corresponding track selection
    """

    # no database output in the mille step:
    # --------------------------------------------------------------------------
    process.AlignmentProducer.saveToDB = False
    process.AlignmentProducer.saveApeToDB = False
    process.AlignmentProducer.saveDeformationsToDB = False


    # Track selection and refitting
    # --------------------------------------------------------------------------
    process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

    import Alignment.CommonAlignment.tools.trackselectionRefitting as trackRefitter
    process.TrackRefittingSequence = trackRefitter.getSequence(
        process,
        collection,
        cosmicsDecoMode = cosmics_deco_mode,
        cosmicsZeroTesla = cosmics_zero_tesla)

    # Overwrite Track-Selector filter options from unified sequence to 'False'
    process.AlignmentTrackSelector.filter = False
    if collection != "ALCARECOTkAlCosmicsCTF0T":
        # there is no HighPurity selector for cosmics
        process.HighPurityTrackSelector.filter = False


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
    process.p = cms.Path(process.offlineBeamSpot*process.TrackRefittingSequence)
