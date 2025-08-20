import re

def setup(process, global_tag, zero_tesla=False, geometry=""):
    """General setup of an alignment process.

    Arguments:
    - `process`: cms.Process object
    - `global_tag`: global tag to be used
    - `zero_tesla`: if 'True' the B-field map for 0T is enforced
    - `geometry`: geometry to be used (default is an empty string for the standard geometry)
    """

    # MessageLogger for convenient output
    # --------------------------------------------------------------------------
    process.load('Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.myMessageLogger_cff')

    # Load the magnetic field configuration
    # --------------------------------------------------------------------------
    if zero_tesla:
        # For 0T MC samples or data
        process.load("Configuration.StandardSequences.MagneticField_0T_cff")
    else:
        process.load('Configuration.StandardSequences.MagneticField_cff')

    # Load the geometry
    # --------------------------------------------------------------------------
    if geometry == "":
        # Default geometry
        print(f"Using Geometry from DB")
        process.load('Configuration.Geometry.GeometryRecoDB_cff')
    else:
        # Check if the geometry string matches the format "Extended<X>", e.g. ExtendedRun4D110
        if re.match(r"^Extended\w+$", geometry):
            # Dynamically load the specified geometry
            geometry_module = f"Configuration.Geometry.Geometry{geometry}Reco_cff"
            try:
                process.load(geometry_module)
                print(f"Using Geometry: {geometry_module}")
            except Exception as e:
                print(f"Error: Unable to load the geometry module '{geometry_module}'.\n{e}")
                raise
        else:
            raise ValueError(f"Invalid geometry format: '{geometry}'. Expected format is 'Extended<X>'.")

    # Load the conditions (GlobalTag)
    # --------------------------------------------------------------------------
    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

    from Configuration.AlCa.GlobalTag import GlobalTag
    process.GlobalTag = GlobalTag(process.GlobalTag, global_tag)
    print("Using Global Tag:", process.GlobalTag.globaltag._value)

    return process  # Not required since the cms.Process object is modified in place
