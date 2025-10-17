import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Configuration for PyTorch Alpaka test"
    )

    parser.add_argument(
        "-nt", "--numberOfThreads",
        type=int,
        default=1,
        help="Number of CMSSW threads"
    )

    parser.add_argument(
        "-ns", "--numberOfStreams",
        type=int,
        default=1,
        help="Number of CMSSW streams"
    )

    parser.add_argument(
        "-ne", "--numberOfEvents",
        type=int,
        default=1,
        help="Number of events to process"
    )

    parser.add_argument(
        "-b", "--backend",
        type=str,
        choices=["serial_sync", "cuda_async", "rocm_async"],
        default="serial_sync",
        help="Accelerator backend"
    )

    parser.add_argument(
        "-bs", "--batchSize",
        type=int,
        default=8,
        help="Batch size"
    )

    parser.add_argument(
        "-e", "--environment",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Environment: 0 - production, 1 - development, 2 - test, 3 - debug"
    )

    parser.add_argument(
        "--simpleNet",
        type=str,
        default="PhysicsTools/PyTorchAlpakaTest/data/SimpleNet.pt",
        help="SimpleNet model (just-in-time compiled)"
    )

    parser.add_argument(
        "--maskedNet",
        type=str,
        default="PhysicsTools/PyTorchAlpakaTest/data/MaskedNet.pt",
        help="MaskedNet model (just-in-time compiled)"
    )

    parser.add_argument(
        "--multiHeadNet",
        type=str,
        default="PhysicsTools/PyTorchAlpakaTest/data/MultiHeadNet.pt",
        help="MultiHeadNet model (just-in-time compiled)"
    )

    parser.add_argument(
        "--tinyResNet",
        type=str,
        default="PhysicsTools/PyTorchAlpakaTest/data/TinyResNet.pt",
        help="TinyResNet model (just-in-time compiled)"
    )

    parser.add_argument(
        "-o", "--only",
        nargs="+",
        default=["SimpleNet", "MultiHeadNet", "MaskedNet", "TinyResNet"],
        choices=["SimpleNet", "MaskedNet", "MultiHeadNet", "TinyResNet"],
        help="Run selected test(s). Default: all modules run in parallel."
    )

    parser.add_argument(
        "-ws", "--wantSummary",
        action="store_true",
        help="Modules execution summary"
    )

    return parser.parse_args()
