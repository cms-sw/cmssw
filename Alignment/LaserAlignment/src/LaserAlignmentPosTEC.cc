/*
 * Alignment of TEC+
 */

#include "Alignment/LaserAlignment/interface/LaserAlignmentPosTEC.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

LaserAlignmentPosTEC::LaserAlignmentPosTEC()
{
}

LaserAlignmentPosTEC::~LaserAlignmentPosTEC()
{
}

void LaserAlignmentPosTEC::alignment(edm::ParameterSet const & theConf, 
				     AlignableTracker * theAlignableTracker,
				     int theNumberOfIterations, 
				     int theAlignmentIteration,
				     std::vector<double>& theLaserPhi, 
				     std::vector<double>& theLaserPhiError)
{
  edm::LogInfo("LaserAlignmentPosTEC") << " ***************************************************** "
				 << "\n *                Alignment of TEC+                  * "
				 << "\n ***************************************************** ";
  
  int theMaxIteration = theNumberOfIterations - 1;

  // create an alignment Object for the positive TEC
  theLaserAlignmentTrackerPosTEC = new LaserAlignmentAlgorithmPosTEC(theConf, theAlignmentIteration);
      
  // do the iterations for the local fits
  for (int theIteration = 0; theIteration < theNumberOfIterations; theIteration++)
    {
      // fill fitted Phi position of the Beams and the Errors on Phi into 
      // a map. Afterwards add the beams to the LaserAlignmentTracker ...
	  
      LogDebug("LaserAlignmentPosTEC") << "  AC1CMS: Total number of Iterations = " << theMaxIteration
				 << "\n  AC1CMS: Current Iteration = " << theIteration
				 << "\n  AC1CMS: Current Alignment Iteration = "  << theAlignmentIteration;
      
      // string is the name (i.e. Beam 0 in TEC+) and std::vector contains
      // 9 values for phi and the error on phi (on each Disc!) for the beam
      std::map<std::string, std::vector<double> > theBeamPosition;
      
      // fill now the map 
      // therefore loop over theLaserPhi and fill
      // Phi + Error in the map for each beam
      
      // Beams in TEC+
      std::vector<double> Beam0R4PosTEC;
      std::vector<double> Beam1R4PosTEC;
      std::vector<double> Beam2R4PosTEC;
      std::vector<double> Beam3R4PosTEC;
      std::vector<double> Beam4R4PosTEC;
      std::vector<double> Beam5R4PosTEC;
      std::vector<double> Beam6R4PosTEC;
      std::vector<double> Beam7R4PosTEC;
      
      std::vector<double> Beam0R6PosTEC;
      std::vector<double> Beam1R6PosTEC;
      std::vector<double> Beam2R6PosTEC;
      std::vector<double> Beam3R6PosTEC;
      std::vector<double> Beam4R6PosTEC;
      std::vector<double> Beam5R6PosTEC;
      std::vector<double> Beam6R6PosTEC;
      std::vector<double> Beam7R6PosTEC;
      
      // Beams in TEC+
      Beam0R4PosTEC.push_back(theLaserPhi.at(0 + theIteration * 434));
      Beam0R4PosTEC.push_back(theLaserPhiError.at(0 + theIteration * 434));
      Beam0R4PosTEC.push_back(theLaserPhi.at(1 + theIteration * 434));
      Beam0R4PosTEC.push_back(theLaserPhiError.at(1 + theIteration * 434));
      Beam0R4PosTEC.push_back(theLaserPhi.at(2 + theIteration * 434));
      Beam0R4PosTEC.push_back(theLaserPhiError.at(2 + theIteration * 434));
      Beam0R4PosTEC.push_back(theLaserPhi.at(3 + theIteration * 434));
      Beam0R4PosTEC.push_back(theLaserPhiError.at(3 + theIteration * 434));
      Beam0R4PosTEC.push_back(theLaserPhi.at(4 + theIteration * 434));
      Beam0R4PosTEC.push_back(theLaserPhiError.at(4 + theIteration * 434));
      Beam0R4PosTEC.push_back(theLaserPhi.at(5 + theIteration * 434));
      Beam0R4PosTEC.push_back(theLaserPhiError.at(5 + theIteration * 434));
      Beam0R4PosTEC.push_back(theLaserPhi.at(6 + theIteration * 434));
      Beam0R4PosTEC.push_back(theLaserPhiError.at(6 + theIteration * 434));
      Beam0R4PosTEC.push_back(theLaserPhi.at(7 + theIteration * 434));
      Beam0R4PosTEC.push_back(theLaserPhiError.at(7 + theIteration * 434));
      Beam0R4PosTEC.push_back(theLaserPhi.at(8 + theIteration * 434));
      Beam0R4PosTEC.push_back(theLaserPhiError.at(8 + theIteration * 434));
      
      Beam1R4PosTEC.push_back(theLaserPhi.at(9 + theIteration * 434));
      Beam1R4PosTEC.push_back(theLaserPhiError.at(9 + theIteration * 434));
      Beam1R4PosTEC.push_back(theLaserPhi.at(10 + theIteration * 434));
      Beam1R4PosTEC.push_back(theLaserPhiError.at(10 + theIteration * 434));
      Beam1R4PosTEC.push_back(theLaserPhi.at(11 + theIteration * 434));
      Beam1R4PosTEC.push_back(theLaserPhiError.at(11 + theIteration * 434));
      Beam1R4PosTEC.push_back(theLaserPhi.at(12 + theIteration * 434));
      Beam1R4PosTEC.push_back(theLaserPhiError.at(12 + theIteration * 434));
      Beam1R4PosTEC.push_back(theLaserPhi.at(13 + theIteration * 434));
      Beam1R4PosTEC.push_back(theLaserPhiError.at(13 + theIteration * 434));
      Beam1R4PosTEC.push_back(theLaserPhi.at(14 + theIteration * 434));
      Beam1R4PosTEC.push_back(theLaserPhiError.at(14 + theIteration * 434));
      Beam1R4PosTEC.push_back(theLaserPhi.at(15 + theIteration * 434));
      Beam1R4PosTEC.push_back(theLaserPhiError.at(15 + theIteration * 434));
      Beam1R4PosTEC.push_back(theLaserPhi.at(16 + theIteration * 434));
      Beam1R4PosTEC.push_back(theLaserPhiError.at(16 + theIteration * 434));
      Beam1R4PosTEC.push_back(theLaserPhi.at(17 + theIteration * 434));
      Beam1R4PosTEC.push_back(theLaserPhiError.at(17 + theIteration * 434));
	  
	  
      Beam2R4PosTEC.push_back(theLaserPhi.at(23 + theIteration * 434));
      Beam2R4PosTEC.push_back(theLaserPhiError.at(23 + theIteration * 434));
      Beam2R4PosTEC.push_back(theLaserPhi.at(24 + theIteration * 434));
      Beam2R4PosTEC.push_back(theLaserPhiError.at(24 + theIteration * 434));
      Beam2R4PosTEC.push_back(theLaserPhi.at(25 + theIteration * 434));
      Beam2R4PosTEC.push_back(theLaserPhiError.at(25 + theIteration * 434));
      Beam2R4PosTEC.push_back(theLaserPhi.at(26 + theIteration * 434));
      Beam2R4PosTEC.push_back(theLaserPhiError.at(26 + theIteration * 434));
      Beam2R4PosTEC.push_back(theLaserPhi.at(27 + theIteration * 434));
      Beam2R4PosTEC.push_back(theLaserPhiError.at(27 + theIteration * 434));
      Beam2R4PosTEC.push_back(theLaserPhi.at(28 + theIteration * 434));
      Beam2R4PosTEC.push_back(theLaserPhiError.at(28 + theIteration * 434));
      Beam2R4PosTEC.push_back(theLaserPhi.at(29 + theIteration * 434));
      Beam2R4PosTEC.push_back(theLaserPhiError.at(29 + theIteration * 434));
      Beam2R4PosTEC.push_back(theLaserPhi.at(30 + theIteration * 434));
      Beam2R4PosTEC.push_back(theLaserPhiError.at(30 + theIteration * 434));
      Beam2R4PosTEC.push_back(theLaserPhi.at(31 + theIteration * 434));
      Beam2R4PosTEC.push_back(theLaserPhiError.at(31 + theIteration * 434));
	  
	  
      Beam3R4PosTEC.push_back(theLaserPhi.at(37 + theIteration * 434));
      Beam3R4PosTEC.push_back(theLaserPhiError.at(37 + theIteration * 434));
      Beam3R4PosTEC.push_back(theLaserPhi.at(38 + theIteration * 434));
      Beam3R4PosTEC.push_back(theLaserPhiError.at(38 + theIteration * 434));
      Beam3R4PosTEC.push_back(theLaserPhi.at(39 + theIteration * 434));
      Beam3R4PosTEC.push_back(theLaserPhiError.at(39 + theIteration * 434));
      Beam3R4PosTEC.push_back(theLaserPhi.at(40 + theIteration * 434));
      Beam3R4PosTEC.push_back(theLaserPhiError.at(40 + theIteration * 434));
      Beam3R4PosTEC.push_back(theLaserPhi.at(41 + theIteration * 434));
      Beam3R4PosTEC.push_back(theLaserPhiError.at(41 + theIteration * 434));
      Beam3R4PosTEC.push_back(theLaserPhi.at(42 + theIteration * 434));
      Beam3R4PosTEC.push_back(theLaserPhiError.at(42 + theIteration * 434));
      Beam3R4PosTEC.push_back(theLaserPhi.at(43 + theIteration * 434));
      Beam3R4PosTEC.push_back(theLaserPhiError.at(43 + theIteration * 434));
      Beam3R4PosTEC.push_back(theLaserPhi.at(44 + theIteration * 434));
      Beam3R4PosTEC.push_back(theLaserPhiError.at(44 + theIteration * 434));
      Beam3R4PosTEC.push_back(theLaserPhi.at(45 + theIteration * 434));
      Beam3R4PosTEC.push_back(theLaserPhiError.at(45 + theIteration * 434));
	  
      Beam4R4PosTEC.push_back(theLaserPhi.at(46 + theIteration * 434));
      Beam4R4PosTEC.push_back(theLaserPhiError.at(46 + theIteration * 434));
      Beam4R4PosTEC.push_back(theLaserPhi.at(47 + theIteration * 434));
      Beam4R4PosTEC.push_back(theLaserPhiError.at(47 + theIteration * 434));
      Beam4R4PosTEC.push_back(theLaserPhi.at(48 + theIteration * 434));
      Beam4R4PosTEC.push_back(theLaserPhiError.at(48 + theIteration * 434));
      Beam4R4PosTEC.push_back(theLaserPhi.at(49 + theIteration * 434));
      Beam4R4PosTEC.push_back(theLaserPhiError.at(49 + theIteration * 434));
      Beam4R4PosTEC.push_back(theLaserPhi.at(50 + theIteration * 434));
      Beam4R4PosTEC.push_back(theLaserPhiError.at(50 + theIteration * 434));
      Beam4R4PosTEC.push_back(theLaserPhi.at(51 + theIteration * 434));
      Beam4R4PosTEC.push_back(theLaserPhiError.at(51 + theIteration * 434));
      Beam4R4PosTEC.push_back(theLaserPhi.at(52 + theIteration * 434));
      Beam4R4PosTEC.push_back(theLaserPhiError.at(52 + theIteration * 434));
      Beam4R4PosTEC.push_back(theLaserPhi.at(53 + theIteration * 434));
      Beam4R4PosTEC.push_back(theLaserPhiError.at(53 + theIteration * 434));
      Beam4R4PosTEC.push_back(theLaserPhi.at(54 + theIteration * 434));
      Beam4R4PosTEC.push_back(theLaserPhiError.at(54 + theIteration * 434));
	  
	  
      Beam5R4PosTEC.push_back(theLaserPhi.at(60 + theIteration * 434));
      Beam5R4PosTEC.push_back(theLaserPhiError.at(60 + theIteration * 434));
      Beam5R4PosTEC.push_back(theLaserPhi.at(61 + theIteration * 434));
      Beam5R4PosTEC.push_back(theLaserPhiError.at(61 + theIteration * 434));
      Beam5R4PosTEC.push_back(theLaserPhi.at(62 + theIteration * 434));
      Beam5R4PosTEC.push_back(theLaserPhiError.at(62 + theIteration * 434));
      Beam5R4PosTEC.push_back(theLaserPhi.at(63 + theIteration * 434));
      Beam5R4PosTEC.push_back(theLaserPhiError.at(63 + theIteration * 434));
      Beam5R4PosTEC.push_back(theLaserPhi.at(64 + theIteration * 434));
      Beam5R4PosTEC.push_back(theLaserPhiError.at(64 + theIteration * 434));
      Beam5R4PosTEC.push_back(theLaserPhi.at(65 + theIteration * 434));
      Beam5R4PosTEC.push_back(theLaserPhiError.at(65 + theIteration * 434));
      Beam5R4PosTEC.push_back(theLaserPhi.at(66 + theIteration * 434));
      Beam5R4PosTEC.push_back(theLaserPhiError.at(66 + theIteration * 434));
      Beam5R4PosTEC.push_back(theLaserPhi.at(67 + theIteration * 434));
      Beam5R4PosTEC.push_back(theLaserPhiError.at(67 + theIteration * 434));
      Beam5R4PosTEC.push_back(theLaserPhi.at(68 + theIteration * 434));
      Beam5R4PosTEC.push_back(theLaserPhiError.at(68 + theIteration * 434));
	  
      Beam6R4PosTEC.push_back(theLaserPhi.at(69 + theIteration * 434));
      Beam6R4PosTEC.push_back(theLaserPhiError.at(69 + theIteration * 434));
      Beam6R4PosTEC.push_back(theLaserPhi.at(70 + theIteration * 434));
      Beam6R4PosTEC.push_back(theLaserPhiError.at(70 + theIteration * 434));
      Beam6R4PosTEC.push_back(theLaserPhi.at(71 + theIteration * 434));
      Beam6R4PosTEC.push_back(theLaserPhiError.at(71 + theIteration * 434));
      Beam6R4PosTEC.push_back(theLaserPhi.at(72 + theIteration * 434));
      Beam6R4PosTEC.push_back(theLaserPhiError.at(72 + theIteration * 434));
      Beam6R4PosTEC.push_back(theLaserPhi.at(73 + theIteration * 434));
      Beam6R4PosTEC.push_back(theLaserPhiError.at(73 + theIteration * 434));
      Beam6R4PosTEC.push_back(theLaserPhi.at(74 + theIteration * 434));
      Beam6R4PosTEC.push_back(theLaserPhiError.at(74 + theIteration * 434));
      Beam6R4PosTEC.push_back(theLaserPhi.at(75 + theIteration * 434));
      Beam6R4PosTEC.push_back(theLaserPhiError.at(75 + theIteration * 434));
      Beam6R4PosTEC.push_back(theLaserPhi.at(76 + theIteration * 434));
      Beam6R4PosTEC.push_back(theLaserPhiError.at(76 + theIteration * 434));
      Beam6R4PosTEC.push_back(theLaserPhi.at(77 + theIteration * 434));
      Beam6R4PosTEC.push_back(theLaserPhiError.at(77 + theIteration * 434));
	  
	  
      Beam7R4PosTEC.push_back(theLaserPhi.at(83 + theIteration * 434));
      Beam7R4PosTEC.push_back(theLaserPhiError.at(83 + theIteration * 434));
      Beam7R4PosTEC.push_back(theLaserPhi.at(84 + theIteration * 434));
      Beam7R4PosTEC.push_back(theLaserPhiError.at(84 + theIteration * 434));
      Beam7R4PosTEC.push_back(theLaserPhi.at(85 + theIteration * 434));
      Beam7R4PosTEC.push_back(theLaserPhiError.at(85 + theIteration * 434));
      Beam7R4PosTEC.push_back(theLaserPhi.at(86 + theIteration * 434));
      Beam7R4PosTEC.push_back(theLaserPhiError.at(86 + theIteration * 434));
      Beam7R4PosTEC.push_back(theLaserPhi.at(87 + theIteration * 434));
      Beam7R4PosTEC.push_back(theLaserPhiError.at(87 + theIteration * 434));
      Beam7R4PosTEC.push_back(theLaserPhi.at(88 + theIteration * 434));
      Beam7R4PosTEC.push_back(theLaserPhiError.at(88 + theIteration * 434));
      Beam7R4PosTEC.push_back(theLaserPhi.at(89 + theIteration * 434));
      Beam7R4PosTEC.push_back(theLaserPhiError.at(89 + theIteration * 434));
      Beam7R4PosTEC.push_back(theLaserPhi.at(90 + theIteration * 434));
      Beam7R4PosTEC.push_back(theLaserPhiError.at(90 + theIteration * 434));
      Beam7R4PosTEC.push_back(theLaserPhi.at(91 + theIteration * 434));
      Beam7R4PosTEC.push_back(theLaserPhiError.at(91 + theIteration * 434));
	  
	  
      Beam0R6PosTEC.push_back(theLaserPhi.at(97 + theIteration * 434));
      Beam0R6PosTEC.push_back(theLaserPhiError.at(97 + theIteration * 434));
      Beam0R6PosTEC.push_back(theLaserPhi.at(98 + theIteration * 434));
      Beam0R6PosTEC.push_back(theLaserPhiError.at(98 + theIteration * 434));
      Beam0R6PosTEC.push_back(theLaserPhi.at(99 + theIteration * 434));
      Beam0R6PosTEC.push_back(theLaserPhiError.at(99 + theIteration * 434));
      Beam0R6PosTEC.push_back(theLaserPhi.at(100 + theIteration * 434));
      Beam0R6PosTEC.push_back(theLaserPhiError.at(100 + theIteration * 434));
      Beam0R6PosTEC.push_back(theLaserPhi.at(101 + theIteration * 434));
      Beam0R6PosTEC.push_back(theLaserPhiError.at(101 + theIteration * 434));
      Beam0R6PosTEC.push_back(theLaserPhi.at(102 + theIteration * 434));
      Beam0R6PosTEC.push_back(theLaserPhiError.at(102 + theIteration * 434));
      Beam0R6PosTEC.push_back(theLaserPhi.at(103 + theIteration * 434));
      Beam0R6PosTEC.push_back(theLaserPhiError.at(103 + theIteration * 434));
      Beam0R6PosTEC.push_back(theLaserPhi.at(104 + theIteration * 434));
      Beam0R6PosTEC.push_back(theLaserPhiError.at(104 + theIteration * 434));
      Beam0R6PosTEC.push_back(theLaserPhi.at(105 + theIteration * 434));
      Beam0R6PosTEC.push_back(theLaserPhiError.at(105 + theIteration * 434));
	  
      Beam1R6PosTEC.push_back(theLaserPhi.at(106 + theIteration * 434));
      Beam1R6PosTEC.push_back(theLaserPhiError.at(106 + theIteration * 434));
      Beam1R6PosTEC.push_back(theLaserPhi.at(107 + theIteration * 434));
      Beam1R6PosTEC.push_back(theLaserPhiError.at(107 + theIteration * 434));
      Beam1R6PosTEC.push_back(theLaserPhi.at(108 + theIteration * 434));
      Beam1R6PosTEC.push_back(theLaserPhiError.at(108 + theIteration * 434));
      Beam1R6PosTEC.push_back(theLaserPhi.at(109 + theIteration * 434));
      Beam1R6PosTEC.push_back(theLaserPhiError.at(109 + theIteration * 434));
      Beam1R6PosTEC.push_back(theLaserPhi.at(110 + theIteration * 434));
      Beam1R6PosTEC.push_back(theLaserPhiError.at(110 + theIteration * 434));
      Beam1R6PosTEC.push_back(theLaserPhi.at(111 + theIteration * 434));
      Beam1R6PosTEC.push_back(theLaserPhiError.at(111 + theIteration * 434));
      Beam1R6PosTEC.push_back(theLaserPhi.at(112 + theIteration * 434));
      Beam1R6PosTEC.push_back(theLaserPhiError.at(112 + theIteration * 434));
      Beam1R6PosTEC.push_back(theLaserPhi.at(113 + theIteration * 434));
      Beam1R6PosTEC.push_back(theLaserPhiError.at(113 + theIteration * 434));
      Beam1R6PosTEC.push_back(theLaserPhi.at(114 + theIteration * 434));
      Beam1R6PosTEC.push_back(theLaserPhiError.at(114 + theIteration * 434));
	  
      Beam2R6PosTEC.push_back(theLaserPhi.at(115 + theIteration * 434));
      Beam2R6PosTEC.push_back(theLaserPhiError.at(115 + theIteration * 434));
      Beam2R6PosTEC.push_back(theLaserPhi.at(116 + theIteration * 434));
      Beam2R6PosTEC.push_back(theLaserPhiError.at(116 + theIteration * 434));
      Beam2R6PosTEC.push_back(theLaserPhi.at(117 + theIteration * 434));
      Beam2R6PosTEC.push_back(theLaserPhiError.at(117 + theIteration * 434));
      Beam2R6PosTEC.push_back(theLaserPhi.at(118 + theIteration * 434));
      Beam2R6PosTEC.push_back(theLaserPhiError.at(118 + theIteration * 434));
      Beam2R6PosTEC.push_back(theLaserPhi.at(119 + theIteration * 434));
      Beam2R6PosTEC.push_back(theLaserPhiError.at(119 + theIteration * 434));
      Beam2R6PosTEC.push_back(theLaserPhi.at(120 + theIteration * 434));
      Beam2R6PosTEC.push_back(theLaserPhiError.at(120 + theIteration * 434));
      Beam2R6PosTEC.push_back(theLaserPhi.at(121 + theIteration * 434));
      Beam2R6PosTEC.push_back(theLaserPhiError.at(121 + theIteration * 434));
      Beam2R6PosTEC.push_back(theLaserPhi.at(122 + theIteration * 434));
      Beam2R6PosTEC.push_back(theLaserPhiError.at(122 + theIteration * 434));
      Beam2R6PosTEC.push_back(theLaserPhi.at(123 + theIteration * 434));
      Beam2R6PosTEC.push_back(theLaserPhiError.at(123 + theIteration * 434));
	  
      Beam3R6PosTEC.push_back(theLaserPhi.at(124 + theIteration * 434));
      Beam3R6PosTEC.push_back(theLaserPhiError.at(124 + theIteration * 434));
      Beam3R6PosTEC.push_back(theLaserPhi.at(125 + theIteration * 434));
      Beam3R6PosTEC.push_back(theLaserPhiError.at(125 + theIteration * 434));
      Beam3R6PosTEC.push_back(theLaserPhi.at(126 + theIteration * 434));
      Beam3R6PosTEC.push_back(theLaserPhiError.at(126 + theIteration * 434));
      Beam3R6PosTEC.push_back(theLaserPhi.at(127 + theIteration * 434));
      Beam3R6PosTEC.push_back(theLaserPhiError.at(127 + theIteration * 434));
      Beam3R6PosTEC.push_back(theLaserPhi.at(128 + theIteration * 434));
      Beam3R6PosTEC.push_back(theLaserPhiError.at(128 + theIteration * 434));
      Beam3R6PosTEC.push_back(theLaserPhi.at(129 + theIteration * 434));
      Beam3R6PosTEC.push_back(theLaserPhiError.at(129 + theIteration * 434));
      Beam3R6PosTEC.push_back(theLaserPhi.at(130 + theIteration * 434));
      Beam3R6PosTEC.push_back(theLaserPhiError.at(130 + theIteration * 434));
      Beam3R6PosTEC.push_back(theLaserPhi.at(131 + theIteration * 434));
      Beam3R6PosTEC.push_back(theLaserPhiError.at(131 + theIteration * 434));
      Beam3R6PosTEC.push_back(theLaserPhi.at(132 + theIteration * 434));
      Beam3R6PosTEC.push_back(theLaserPhiError.at(132 + theIteration * 434));
	  
      Beam4R6PosTEC.push_back(theLaserPhi.at(133 + theIteration * 434));
      Beam4R6PosTEC.push_back(theLaserPhiError.at(133 + theIteration * 434));
      Beam4R6PosTEC.push_back(theLaserPhi.at(134 + theIteration * 434));
      Beam4R6PosTEC.push_back(theLaserPhiError.at(134 + theIteration * 434));
      Beam4R6PosTEC.push_back(theLaserPhi.at(135 + theIteration * 434));
      Beam4R6PosTEC.push_back(theLaserPhiError.at(135 + theIteration * 434));
      Beam4R6PosTEC.push_back(theLaserPhi.at(136 + theIteration * 434));
      Beam4R6PosTEC.push_back(theLaserPhiError.at(136 + theIteration * 434));
      Beam4R6PosTEC.push_back(theLaserPhi.at(137 + theIteration * 434));
      Beam4R6PosTEC.push_back(theLaserPhiError.at(137 + theIteration * 434));
      Beam4R6PosTEC.push_back(theLaserPhi.at(138 + theIteration * 434));
      Beam4R6PosTEC.push_back(theLaserPhiError.at(138 + theIteration * 434));
      Beam4R6PosTEC.push_back(theLaserPhi.at(139 + theIteration * 434));
      Beam4R6PosTEC.push_back(theLaserPhiError.at(139 + theIteration * 434));
      Beam4R6PosTEC.push_back(theLaserPhi.at(140 + theIteration * 434));
      Beam4R6PosTEC.push_back(theLaserPhiError.at(140 + theIteration * 434));
      Beam4R6PosTEC.push_back(theLaserPhi.at(141 + theIteration * 434));
      Beam4R6PosTEC.push_back(theLaserPhiError.at(141 + theIteration * 434));
	  
      Beam5R6PosTEC.push_back(theLaserPhi.at(142 + theIteration * 434));
      Beam5R6PosTEC.push_back(theLaserPhiError.at(142 + theIteration * 434));
      Beam5R6PosTEC.push_back(theLaserPhi.at(143 + theIteration * 434));
      Beam5R6PosTEC.push_back(theLaserPhiError.at(143 + theIteration * 434));
      Beam5R6PosTEC.push_back(theLaserPhi.at(144 + theIteration * 434));
      Beam5R6PosTEC.push_back(theLaserPhiError.at(144 + theIteration * 434));
      Beam5R6PosTEC.push_back(theLaserPhi.at(145 + theIteration * 434));
      Beam5R6PosTEC.push_back(theLaserPhiError.at(145 + theIteration * 434));
      Beam5R6PosTEC.push_back(theLaserPhi.at(146 + theIteration * 434));
      Beam5R6PosTEC.push_back(theLaserPhiError.at(146 + theIteration * 434));
      Beam5R6PosTEC.push_back(theLaserPhi.at(147 + theIteration * 434));
      Beam5R6PosTEC.push_back(theLaserPhiError.at(147 + theIteration * 434));
      Beam5R6PosTEC.push_back(theLaserPhi.at(148 + theIteration * 434));
      Beam5R6PosTEC.push_back(theLaserPhiError.at(148 + theIteration * 434));
      Beam5R6PosTEC.push_back(theLaserPhi.at(149 + theIteration * 434));
      Beam5R6PosTEC.push_back(theLaserPhiError.at(149 + theIteration * 434));
      Beam5R6PosTEC.push_back(theLaserPhi.at(150 + theIteration * 434));
      Beam5R6PosTEC.push_back(theLaserPhiError.at(150 + theIteration * 434));
	  
      Beam6R6PosTEC.push_back(theLaserPhi.at(151 + theIteration * 434));
      Beam6R6PosTEC.push_back(theLaserPhiError.at(151 + theIteration * 434));
      Beam6R6PosTEC.push_back(theLaserPhi.at(152 + theIteration * 434));
      Beam6R6PosTEC.push_back(theLaserPhiError.at(152 + theIteration * 434));
      Beam6R6PosTEC.push_back(theLaserPhi.at(153 + theIteration * 434));
      Beam6R6PosTEC.push_back(theLaserPhiError.at(153 + theIteration * 434));
      Beam6R6PosTEC.push_back(theLaserPhi.at(154 + theIteration * 434));
      Beam6R6PosTEC.push_back(theLaserPhiError.at(154 + theIteration * 434));
      Beam6R6PosTEC.push_back(theLaserPhi.at(155 + theIteration * 434));
      Beam6R6PosTEC.push_back(theLaserPhiError.at(155 + theIteration * 434));
      Beam6R6PosTEC.push_back(theLaserPhi.at(156 + theIteration * 434));
      Beam6R6PosTEC.push_back(theLaserPhiError.at(156 + theIteration * 434));
      Beam6R6PosTEC.push_back(theLaserPhi.at(157 + theIteration * 434));
      Beam6R6PosTEC.push_back(theLaserPhiError.at(157 + theIteration * 434));
      Beam6R6PosTEC.push_back(theLaserPhi.at(158 + theIteration * 434));
      Beam6R6PosTEC.push_back(theLaserPhiError.at(158 + theIteration * 434));
      Beam6R6PosTEC.push_back(theLaserPhi.at(159 + theIteration * 434));
      Beam6R6PosTEC.push_back(theLaserPhiError.at(159 + theIteration * 434));
	  
      Beam7R6PosTEC.push_back(theLaserPhi.at(160 + theIteration * 434));
      Beam7R6PosTEC.push_back(theLaserPhiError.at(160 + theIteration * 434));
      Beam7R6PosTEC.push_back(theLaserPhi.at(161 + theIteration * 434));
      Beam7R6PosTEC.push_back(theLaserPhiError.at(161 + theIteration * 434));
      Beam7R6PosTEC.push_back(theLaserPhi.at(162 + theIteration * 434));
      Beam7R6PosTEC.push_back(theLaserPhiError.at(162 + theIteration * 434));
      Beam7R6PosTEC.push_back(theLaserPhi.at(163 + theIteration * 434));
      Beam7R6PosTEC.push_back(theLaserPhiError.at(163 + theIteration * 434));
      Beam7R6PosTEC.push_back(theLaserPhi.at(164 + theIteration * 434));
      Beam7R6PosTEC.push_back(theLaserPhiError.at(164 + theIteration * 434));
      Beam7R6PosTEC.push_back(theLaserPhi.at(165 + theIteration * 434));
      Beam7R6PosTEC.push_back(theLaserPhiError.at(165 + theIteration * 434));
      Beam7R6PosTEC.push_back(theLaserPhi.at(166 + theIteration * 434));
      Beam7R6PosTEC.push_back(theLaserPhiError.at(166 + theIteration * 434));
      Beam7R6PosTEC.push_back(theLaserPhi.at(167 + theIteration * 434));
      Beam7R6PosTEC.push_back(theLaserPhiError.at(167 + theIteration * 434));
      Beam7R6PosTEC.push_back(theLaserPhi.at(168 + theIteration * 434));
      Beam7R6PosTEC.push_back(theLaserPhiError.at(168 + theIteration * 434));

      /* *************************************** */

      // create entry in the map for Beam 0 in Ring 4 of TEC+
      theBeamPosition["Beam 0 in Ring 4 in TEC+"] = Beam0R4PosTEC;
      // create entry in the map for Beam 1 in Ring 4 of TEC+
      theBeamPosition["Beam 1 in Ring 4 in TEC+"] = Beam1R4PosTEC;
      // create entry in the map for Beam 2 in Ring 4 of TEC+
      theBeamPosition["Beam 2 in Ring 4 in TEC+"] = Beam2R4PosTEC;
      // create entry in the map for Beam 3 in Ring 4 of TEC+
      theBeamPosition["Beam 3 in Ring 4 in TEC+"] = Beam3R4PosTEC;
      // create entry in the map for Beam 4 in Ring 4 of TEC+
      theBeamPosition["Beam 4 in Ring 4 in TEC+"] = Beam4R4PosTEC;
      // create entry in the map for Beam 5 in Ring 4 of TEC+
      theBeamPosition["Beam 5 in Ring 4 in TEC+"] = Beam5R4PosTEC;
      // create entry in the map for Beam 6 in Ring 4 of TEC+
      theBeamPosition["Beam 6 in Ring 4 in TEC+"] = Beam6R4PosTEC;
      // create entry in the map for Beam 7 in Ring 4 of TEC+
      theBeamPosition["Beam 7 in Ring 4 in TEC+"] = Beam7R4PosTEC;
	  
      // create entry in the map for Beam 0 in Ring 6 of TEC+
      theBeamPosition["Beam 0 in Ring 6 in TEC+"] = Beam0R6PosTEC;
      // create entry in the map for Beam 1 in Ring 6 of TEC+
      theBeamPosition["Beam 1 in Ring 6 in TEC+"] = Beam1R6PosTEC;
      // create entry in the map for Beam 2 in Ring 6 of TEC+
      theBeamPosition["Beam 2 in Ring 6 in TEC+"] = Beam2R6PosTEC;
      // create entry in the map for Beam 3 in Ring 6 of TEC+
      theBeamPosition["Beam 3 in Ring 6 in TEC+"] = Beam3R6PosTEC;
      // create entry in the map for Beam 4 in Ring 6 of TEC+
      theBeamPosition["Beam 4 in Ring 6 in TEC+"] = Beam4R6PosTEC;
      // create entry in the map for Beam 5 in Ring 6 of TEC+
      theBeamPosition["Beam 5 in Ring 6 in TEC+"] = Beam5R6PosTEC;
      // create entry in the map for Beam 6 in Ring 6 of TEC+
      theBeamPosition["Beam 6 in Ring 6 in TEC+"] = Beam6R6PosTEC;
      // create entry in the map for Beam 7 in Ring 6 of TEC+
      theBeamPosition["Beam 7 in Ring 6 in TEC+"] = Beam7R6PosTEC;
	  
      // *******************************************************

      // add the beams to millipede
      // Ring 4 of TEC+
      theLaserAlignmentTrackerPosTEC->addLaserBeam(theBeamPosition["Beam 0 in Ring 4 in TEC+"], 0, 4);
      theLaserAlignmentTrackerPosTEC->addLaserBeam(theBeamPosition["Beam 1 in Ring 4 in TEC+"], 1, 4);
      theLaserAlignmentTrackerPosTEC->addLaserBeam(theBeamPosition["Beam 2 in Ring 4 in TEC+"], 2, 4);
      theLaserAlignmentTrackerPosTEC->addLaserBeam(theBeamPosition["Beam 3 in Ring 4 in TEC+"], 3, 4);
      theLaserAlignmentTrackerPosTEC->addLaserBeam(theBeamPosition["Beam 4 in Ring 4 in TEC+"], 4, 4);
      theLaserAlignmentTrackerPosTEC->addLaserBeam(theBeamPosition["Beam 5 in Ring 4 in TEC+"], 5, 4);
      theLaserAlignmentTrackerPosTEC->addLaserBeam(theBeamPosition["Beam 6 in Ring 4 in TEC+"], 6, 4);
      theLaserAlignmentTrackerPosTEC->addLaserBeam(theBeamPosition["Beam 7 in Ring 4 in TEC+"], 7, 4);
      // Ring 6 of TEC+
      theLaserAlignmentTrackerPosTEC->addLaserBeam(theBeamPosition["Beam 0 in Ring 6 in TEC+"], 0, 6);
      theLaserAlignmentTrackerPosTEC->addLaserBeam(theBeamPosition["Beam 1 in Ring 6 in TEC+"], 1, 6);
      theLaserAlignmentTrackerPosTEC->addLaserBeam(theBeamPosition["Beam 2 in Ring 6 in TEC+"], 2, 6);
      theLaserAlignmentTrackerPosTEC->addLaserBeam(theBeamPosition["Beam 3 in Ring 6 in TEC+"], 3, 6);
      theLaserAlignmentTrackerPosTEC->addLaserBeam(theBeamPosition["Beam 4 in Ring 6 in TEC+"], 4, 6);
      theLaserAlignmentTrackerPosTEC->addLaserBeam(theBeamPosition["Beam 5 in Ring 6 in TEC+"], 5, 6);
      theLaserAlignmentTrackerPosTEC->addLaserBeam(theBeamPosition["Beam 6 in Ring 6 in TEC+"], 6, 6);
      theLaserAlignmentTrackerPosTEC->addLaserBeam(theBeamPosition["Beam 7 in Ring 6 in TEC+"], 7, 6);
    }
	  
  // finally do the fit of the global parameters when we have reached the last iteration
  edm::LogInfo("LASAlingPosTEC") << "<LaserAlignmentPosTEC::alignment()>: doing the global fit ... ";
	      
  theLaserAlignmentTrackerPosTEC->doGlobalFit(theAlignableTracker);
  
  // reset Millepede
//   theLaserAlignmentTrackerPosTEC->resetMillepede(theAlignmentIteration);

  // delete the alignment Object for the positive TEC to avoid problems
  // with the alignment of the negative TEC
  if (theLaserAlignmentTrackerPosTEC != 0) { delete theLaserAlignmentTrackerPosTEC; }
}
