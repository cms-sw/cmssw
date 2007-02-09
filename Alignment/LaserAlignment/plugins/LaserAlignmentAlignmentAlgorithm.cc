/*
 * LaserAlignmentAlignmentAlgorithm.icc --- LAS Reconstruction Program - Alignment Algorithm
 */

#include "Alignment/LaserAlignment/plugins/LaserAlignment.h" 

void LaserAlignment::alignmentAlgorithm(edm::ParameterSet const& theAlgorithmConf,
					AlignableTracker * theAlignableTracker)
{
  // we have to submit theAlgorithmConf to the AlignmentAlgorithms!!!!
  // also submit the number of iterations and the vector with
  // fitted beampositions by reference
  // ... and the AlignableTracker


  if (theAlignPosTEC)
    {
      // Alignment of TEC+
      theLASAlignPosTEC->alignment(theAlgorithmConf, theAlignableTracker,
				   theNumberOfIterations,
				   theNumberOfAlignmentIterations, theLaserPhi,
				   theLaserPhiError);
    }

  if (theAlignNegTEC)
    {
      // Alignment of TEC-
      theLASAlignNegTEC->alignment(theAlgorithmConf, theAlignableTracker, 
				   theNumberOfIterations, 
				   theNumberOfAlignmentIterations, 
				   theLaserPhi, theLaserPhiError);
    }

  if (theAlignTEC2TEC)
    {
      // Alignment of TEC-TIB-TOB-TEC
      theLASAlignTEC2TEC->alignment(theAlgorithmConf, theAlignableTracker,
				    theNumberOfIterations, 
				    theNumberOfAlignmentIterations, theLaserPhi,
				    theLaserPhiError);
    }

  // clear the stored values of the positions and errors
  theLaserPhi.clear();
  theLaserPhiError.clear();
  
  // increase the number of Alignment Iterations
  theNumberOfAlignmentIterations++;
}


