import RecoTracker.MkFit.mkFitInputConverter_cfi as mkFitInputConverter_cfi
import RecoTracker.MkFit.mkFitProducer_cfi as mkFitProducer_cfi
import RecoTracker.MkFit.mkFitOutputConverter_cfi as mkFitOutputConverter_cfi

def customizeInitialStepToMkFit(process):
    process.initialStepTrackCandidatesMkFitInput = mkFitInputConverter_cfi.mkFitInputConverter.clone(
        seeds = "initialStepSeeds",
    )
    process.initialStepTrackCandidatesMkFit = mkFitProducer_cfi.mkFitProducer.clone(
        hitsSeeds = "initialStepTrackCandidatesMkFitInput",
    )
    process.initialStepTrackCandidates = mkFitOutputConverter_cfi.mkFitOutputConverter.clone(
        seeds = "initialStepSeeds",
        hitsSeeds = "initialStepTrackCandidatesMkFitInput",
        tracks = "initialStepTrackCandidatesMkFit",
    )
    process.InitialStepTask.add(process.initialStepTrackCandidatesMkFitInput,
                                process.initialStepTrackCandidatesMkFit)
    return process
