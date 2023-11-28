import FWCore.ParameterSet.Config as cms
import re

algorithms = cms.VPSet()

l1tGTAlgoBlockProducer = cms.EDProducer(
    "L1GTAlgoBlockProducer",
    algorithms = algorithms
)

def collectAlgorithmPaths(process) -> "tuple[cms.Path]":
    str_paths = set()
    for algorithm in algorithms:
        algo_paths = re.sub(r'[()]'," " , algorithm.expression.value()).split()
        for algo in algo_paths:
            if algo in process.pathNames() :
                str_paths.add(algo)
    paths = set()

    for str_path in str_paths:
        paths.add(getattr(process, str_path))

    return tuple(paths)
