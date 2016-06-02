import os

def checked_out_MPS():
    """Checks if MPS is checked out locally or taken from the release."""

    checked_out_packages = os.path.join(os.environ["CMSSW_BASE"], "src", ".git",
                                        "info", "sparse-checkout")
    checked_out = False
    git_initialized = False
    try:
        with open(checked_out_packages, "r") as f:
            packages = ("/Alignment/", "/Alignment/MillePedeAlignmentAlgorithm/")
            for line in f:
                if line.strip() in packages:
                    checked_out = True
                    break
        git_initialized = True  # since the sparse checkout file is there
    except IOError as e:
        if e.args != (2, 'No such file or directory'): raise

    return checked_out, git_initialized
