import argparse

MODELS = ["SimpleNet", "MultiHeadNet", "MaskedNet", "TinyResNet"]


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for the ONNX Runtime alpaka test")

    parser.add_argument("-nt", "--numberOfThreads", type=int, default=1, help="Number of CMSSW threads")
    parser.add_argument("-ns", "--numberOfStreams", type=int, default=1, help="Number of CMSSW streams")
    parser.add_argument("-ne", "--numberOfEvents", type=int, default=1, help="Number of events to process")
    parser.add_argument("-b", "--backend", type=str, choices=["serial_sync", "cuda_async", "rocm_async"],
                        default="serial_sync", help="Accelerator backend")
    parser.add_argument("-ts", "--totalSize", type=int, default=35, help="Number of elements in each collection")
    parser.add_argument("-bs", "--batchSize", type=int, default=32, help="Size of the mini-batches")
    parser.add_argument("-o", "--only", nargs="+", default=MODELS, choices=MODELS,
                        help="Run selected test(s). Default: all modules run in parallel.")
    parser.add_argument("-ws", "--wantSummary", action="store_true", help="Modules execution summary")

    return parser.parse_args()
