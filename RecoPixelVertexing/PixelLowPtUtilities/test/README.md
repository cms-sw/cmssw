### How to produce a calibration file for ClusterShapeHitFilter

1. Create CMSSW release area and add RecoPixelVertexing/PixelLowPtUtilities package:

    ```shell
    git cms-addpkg RecoPixelVertexing/PixelLowPtUtilities
    scram b -j8
    ```
2. Generate at least 10k MinBias events for the geometry in question.
3. Run the first two steps of the standard simulation sequence (up to DIGI2RAW).
4. Generate a configuration file for the step 3 (reconstruction) and patch it by removing "process.schedule" and adding following lines:
    
    ```shell
    process.load('ClusterShapeExtractor_cfi')
    process.clusterShapeExtractor_step = cms.Path(process.clusterShapeExtractor)
    process.schedule = cms.Schedule(
        process.raw2digi_step,
        process.L1Reco_step,
        process.reconstruction_step,
        process.clusterShapeExtractor_step
    )
    ```
  * As the reference example, see *ClusterShapeExtractor_Phase2_cfg.py*.
5. Go to *RecoPixelVertexing/PixelLowPtUtilities/test* directory and run the modified step 3 configuration to produce clusterShape.root
6. Produce a calibration file:

    ```shell
    clusterShapeAnalyzer clusterShape.root pixelShape.par
    ```
