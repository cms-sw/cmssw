def setup(process, global_tag, zero_tesla = False):
    """General setup of an alignment process.

    Arguments:
    - `process`: cms.Process object
    - `global_tag`: global tag to be used
    - `zero_tesla`: if 'True' the B-field map for 0T is enforced
    """

    # MessageLogger for convenient output
    # --------------------------------------------------------------------------
    process.load('Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.myMessageLogger_cff')

    # Load the conditions
    # --------------------------------------------------------------------------
    if zero_tesla:
        # actually only needed for 0T MC samples, but does not harm for 0T data:
        process.load("Configuration.StandardSequences.MagneticField_0T_cff")
    else:
        process.load('Configuration.StandardSequences.MagneticField_cff')
    process.load('Configuration.Geometry.GeometryRecoDB_cff')
    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

    from Configuration.AlCa.GlobalTag import GlobalTag
    process.GlobalTag = GlobalTag(process.GlobalTag, global_tag)
    print "Using Global Tag:", process.GlobalTag.globaltag._value

    return process # not required because the cms.Process is modified in place
