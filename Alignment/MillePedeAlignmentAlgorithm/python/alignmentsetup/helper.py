from builtins import range
import os
import FWCore.ParameterSet.Config as cms

def checked_out_MPS():
    """Checks if MPS is checked out locally or taken from the release."""

    checked_out_packages = os.path.join(os.environ["CMSSW_BASE"], "src", ".git",
                                        "info", "sparse-checkout")
    checked_out = False
    git_initialized = False
    try:
        with open(checked_out_packages, "r") as f:
            packages = ("/Alignment/", "/Alignment/MillePedeAlignmentAlgorithm/","/*/")
            for line in f:
                if line.strip() in packages:
                    checked_out = True
                    break
        git_initialized = True  # since the sparse checkout file is there
    except IOError as e:
        if e.args != (2, 'No such file or directory'): raise

    return checked_out, git_initialized


def set_pede_option(process, option, drop = False):
    """Utility function to set or override pede `option` defined in `process`.

    Arguments:
    - `process`: cms.Process object
    - `option`: option string
    - `drop`: if set to 'True' the `option` is dropped completely
    """

    existing_options = process.AlignmentProducer.algoConfig.pedeSteerer.options

    exists = False
    for i in range(len(existing_options)):
        if existing_options[i].split()[0] == option.split()[0]:
           existing_options[i] = option.strip()
           exists = True
           if drop: existing_options.pop(i)
           break

    if not exists and not drop: existing_options.append(option.strip())


def add_filter(process, ed_filter):
    """
    Adds EDFilter to the supplied cms.Process object and returns complete filter
    sequence.

    Arguments:
    - `process`: cms.Process object
    - `ed_filter`: EDFilter
    """

    if not hasattr(process, "mps_filters"): process.mps_filters = cms.Sequence()
    process.mps_filters += ed_filter
    return process.mps_filters
