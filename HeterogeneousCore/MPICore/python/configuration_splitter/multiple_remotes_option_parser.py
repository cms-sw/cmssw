import argparse
import pathlib
import sys


def build_global_parser():
    """The parser of the options common to all remotes."""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "config",
        metavar="config.py",
        type=pathlib.Path,
        help="python configuration file to be split"
    )
    parser.add_argument("-c", "--reuse-dumps", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-l",
        "--output-local",
        type=pathlib.Path,
        default=pathlib.Path("local.py"),
    )

    return parser


def build_process_parser():
    """The parser of the options specific to each remote."""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("-m", "--remote-modules", nargs="+", default=None)
    parser.add_argument("-r", "--output-remote", type=pathlib.Path, default=None)
    parser.add_argument("-d", "--duplicate-modules", nargs="+", default=None)
    parser.add_argument("-n", "--remote-process-name", default="")

    return parser


def split_groups(argv):
    """Cut the argument list at each `:`."""
    groups = []
    current = []

    for arg in argv:
        if arg == ":":
            groups.append(current)
            current = []
        else:
            current.append(arg)

    groups.append(current)
    return groups


def print_help():
    """Print the help text."""
    print(
"""
edmMpiSplitConfig script splits a CMSSW configuration \
into local and one or multiple remote processes for MPI execution. 
Selected modules (or sequences) are offloaded to a remote process while \
the remaining modules stay in the local process.
Data dependencies are \
automatically analyzed and the required products are forwarded between processes.


USAGE:
    edmMpiSplitConfig config.py [GLOBAL OPTIONS] [PROCESS OPTIONS] [: PROCESS OPTIONS] ...

DESCRIPTION:
    Arguments before ':' apply to one remote process.
    You can define multiple remote processes by separating groups with ':'.

GLOBAL OPTIONS:
Positional arguments (required):
    config
        Path to the input HLT Python configuration file.

Optional arguments:
    -l, --output-local
        Path to the output configuration for the local process.
        (default: local.py)
   
    -c, --reuse-dumps
        False by default. If this script was run before on the same,
        unmodified input config, pass this argument to reuse the product
        names and dependency graph it dumped then, instead of rerunning
        cmsRun to regenerate them.

    -v, --verbose
        Print debug outputs

PROCESS OPTIONS (can be passed for different remote processes):
    -m, --remote-modules
        Modules to be offloaded to the remote process.
        Sequences can also be passed as a parameter, in which case
        all modules of that sequence will be offloaded
        
    -r, --output-remote
        Path to the output configuration for the remote process.
        (default: remote.py for a single remote, remote0.py, remote1.py, ...
        for several)

    -d, --duplicate-modules
        List of module labels that must run on both local and remote
        processes. Products from these modules are not transferred
        between processes.

    -n, --remote-process-name
        Name of the remote process (default: REMOTE0, REMOTE1, ... in the
        order the ':'-separated groups are given)


SINGLE REMOTE EXAMPLES:

Example 1 (offloading GPU part of ECAL and HCAL):

    edmMpiSplitConfig hlt.py --remote-modules hltEcalDigisSoA hltEcalUncalibRecHitSoA \\
        hltHcalDigisSoA hltHbheRecoSoA hltParticleFlowRecHitHBHESoA hltParticleFlowClusterHBHESoA \\
        --duplicate-modules hltHcalDigis \\
        --output-local local.py \\
        --output-remote remote.py

Example 2 (offloading GPU part of ECAL, HCAL and pixels):

    edmMpiSplitConfig hlt.py --remote-modules hltEcalDigisSoA hltEcalUncalibRecHitSoA \\
        hltHcalDigisSoA hltHbheRecoSoA hltParticleFlowRecHitHBHESoA hltParticleFlowClusterHBHESoA \\
        hltSiPixelClustersSoA hltSiPixelRecHitsSoA hltPixelTracksSoA hltPixelVerticesSoA \\
        --duplicate-modules hltHcalDigis hltOnlineBeamSpot hltOnlineBeamSpotDevice \\
        --output-local local_pixels.py \\
        --output-remote remote_pixels.py

MULTI-REMOTE EXAMPLES:

    Separate remote groups using ':'.
    
    Example (offloading GPU part of ECAL to one process and HCAL to another):
    
    edmMpiSplitConfig hlt.py -l local.py \\
        -m hltEcalDigisSoA hltEcalUncalibRecHitSoA -r remote_ecal.py -n ECAL : \\
        -m hltHcalDigisSoA hltHbheRecoSoA hltParticleFlowRecHitHBHESoA hltParticleFlowClusterHBHESoA \\
        -d hltHcalDigis -r remote_hcal.py -n HCAL

Notes:

• To split a configuration it must be processed with 0 events. That single cmsRun job
    dumps both the C++ product names and the module dependency graph, into the
    '.edmMpiSplitConfig' directory.
• If the splitter was run before on the same input config, --reuse-dumps skips that job and
    reads back what it wrote, which makes the script run faster.
• For some modules it might be better to run on both processes
    instead of sending their products. Use --duplicate-modules option to specify them.
• Dependencies are the ones the framework itself resolved, so they cover consumes(),
    mayConsumes(), EDAliases and products made by the Source.
• Execution order inside dependency groups is preserved.
"""
    )
    

def parse_mpi_style_args(argv):
    """Return a list with one `argparse.Namespace` per remote."""
    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print_help()
        sys.exit(0)

    # splitter options common to all remotes
    global_parser = build_global_parser()
    # splitter options specific to each remote
    process_parser = build_process_parser()

    # parse global options
    global_args, _ = global_parser.parse_known_args(argv)
    # cut the argument list at each `:`
    groups = split_groups(argv)

    # parse per-remote options
    configs = []
    for i, group in enumerate(groups):
        proc_args, _ = process_parser.parse_known_args(group)

        cfg = argparse.Namespace(**vars(global_args))
        cfg.remote_modules = proc_args.remote_modules or []
        cfg.duplicate_modules = proc_args.duplicate_modules or []
        cfg.remote_process_name = proc_args.remote_process_name or f"REMOTE{i}"
        # a single remote writes the plain default file name; several need one each
        cfg.output_remote = proc_args.output_remote or pathlib.Path(
            "remote.py" if len(groups) == 1 else f"remote{i}.py")

        configs.append(cfg)

    return configs
