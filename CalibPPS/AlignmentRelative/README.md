# introduction

Alignment of (CT-)PPS detectors (housed in Roman Pots) proceeds in the following conceptual steps
 * relative alignment among sensors - by minimisation of track-hit residuals
 * global alignment of RPs wrt the beam - by exploiting symmetries in observed hit patterns

The alignment is applied separately to each arm of the spectrometer (LHC sectors 45 adn 56). For more details see e.g. this [note](http://cds.cern.ch/record/2256296).

This package implements the first step, "track-based alignment" among RP sensors. The alignment corrections of relevance are transverse shifts (in x and y) a rotations about the beam axis (z). The method is based on minimisation of residuals between hits and reconstructed tracks. Certain misalignment modes do not generate residuals (e.g. global shift or rotation) and are therefore inaccessible to the track-based alignment. Sometimes, these modes are also referred to as "singular". In order to find a solution to the alignment task, user must specify "constraints" which provide information about the inaccessible/singular alignment modes. These constraints may come from the second step (global alignment) as outlined above.

For the following reasons, the track-based alignment is performed in an iterative manner:
 * the treatment of rotations in linearised (sin alpha ~ alpha, etc.)
 * a priory, it is not clear that a hit with large residual is either an outlier (noise or so) or hit from a sensor with large misalignment

Therefore the alignment starts with a large tolerace (don't exclude hits with large misalignments) and gradually (in iterations over the same data) makes the quality cuts stricter (to remove outliers).

The implementation inherits from the code originally developed for TOTEM and described in this [thesis](http://cdsweb.cern.ch/record/1441140). The original code was developed for Si strip detectors, the current implementation can cope also with the Si pixel detectors of PPS. There is some code also for the timing RPs (diamonds), but it is of experimental nature.

# input

For strip RPs, the alignment takes the input from "U-V patterns", i.e. linear rec-hit patterns in U-z and V-z planes.

For pixel RPs, the alignment can take the input
 * either from rec-hits (only from planes with a single hit)
 * or reconstructed tracks (reconstruction provides some suppression of outliers).

One should not use both input methods in the same time (double counting). 

# files

In interface/
 * AlignmentAlgorithm.h: abstract interface of a track-based algorithm
 * AlignmentConstraint.h: information about a constraint (fixing an alignment mode inaccessible to track-based alignment)
 * AlignmentGeometry.h: summary of geometry-related data used in alignment analysis
 * AlignmentResult.h: structure holding the track-based alignment results
 * AlignmentTask.h: information about alignment task
 * HitCollection.h: collection of sensor hits (from strip or pixel sensors)
 * IdealResult.h: predicts the result of track-based alignment under the imposed constraints
 * JanAlignmentAlgorithm.h: an implementation of track-hit minimisation algorithm
 * LocalTrackFit.h: straight-line fit through all RPs in one arm
 * LocalTrackFitter.h: code to make the per-arm track fits
 * SingularMode.h: information about inaccessible/singular modes
 * StraightTrackAlignment.h: common code for all track-based alignment algorithms
 * Utilities.h: common code

In plugins/
 * PPSFastLocalSimulation.cc: a fast per-arm simulation to test the alignment code
 * PPSModifySingularModes.cc: module to modify the singular modes in a track-based alignment result
 * PPSStraightTrackAligner.cc: module to run the track-based alignment

# tests

 * test_with_mc: Monte-Carlo tests of the alignment code
   * simple: a very basic test
   * iterations: script to test alignment convergence in several iterations
   * statistics: script to test statistical properties

 * test_modify_singular_modes: an example how to modify the singular modes

 * test_with_data: a full/real-life example of alignment application to (LHC) data

