/** \file LaserAlignmentTEC2TEC.cc
 *  
 *
 *  $Date: Sun Mar 18 19:36:31 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/interface/LaserAlignmentTEC2TEC.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

LaserAlignmentTEC2TEC::LaserAlignmentTEC2TEC()
{
}

LaserAlignmentTEC2TEC::~LaserAlignmentTEC2TEC()
{
}

void LaserAlignmentTEC2TEC::alignment(edm::ParameterSet const & theConf,
				      AlignableTracker * theAlignableTracker,
				      int theNumberOfIterations, 
				      int theAlignmentIteration,
				      std::vector<double>& theLaserPhi, 
				      std::vector<double>& theLaserPhiError)
{
  edm::LogInfo("LaserAlignmentTEC2TEC") << " ***************************************************** "
				  << "\n *           Alignment of TEC-TOB-TIB-TEC            * "
				  << "\n ***************************************************** ";
      
  int theMaxIteration = theNumberOfIterations - 1;

  // create an alignment Object for TEC-TOB-TIB-TEC
  theLaserAlignmentTrackerTEC2TEC = new LaserAlignmentAlgorithmTEC2TEC(theConf, theAlignmentIteration);

  // do the iterations for the local fits
  for (int theIteration = 0; theIteration < theNumberOfIterations; theIteration++)
    {
      // fill fitted Phi position of the Beams and the Errors on Phi into 
      // a map. Afterwards add the beams to the LaserAlignmentTracker ...

      LogDebug("LaserAlignmentTEC2TEC") << "  AC1CMS: Total number of Iterations = " << theMaxIteration
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
      std::vector<double> Beam1R4PosTEC2TEC;
      std::vector<double> Beam2R4PosTEC2TEC;
      std::vector<double> Beam3R4PosTEC;
      std::vector<double> Beam4R4PosTEC2TEC;
      std::vector<double> Beam5R4PosTEC;
      std::vector<double> Beam6R4PosTEC2TEC;
      std::vector<double> Beam7R4PosTEC2TEC;
	  
      // Beams in TEC-
      std::vector<double> Beam0R4NegTEC;
      std::vector<double> Beam1R4NegTEC2TEC;
      std::vector<double> Beam2R4NegTEC2TEC;
      std::vector<double> Beam3R4NegTEC;
      std::vector<double> Beam4R4NegTEC2TEC;
      std::vector<double> Beam5R4NegTEC;
      std::vector<double> Beam6R4NegTEC2TEC;
      std::vector<double> Beam7R4NegTEC2TEC;
	  
      // Beams in TOB
      std::vector<double> Beam0TOB;
      std::vector<double> Beam1TOB;
      std::vector<double> Beam2TOB;
      std::vector<double> Beam3TOB;
      std::vector<double> Beam4TOB;
      std::vector<double> Beam5TOB;
      std::vector<double> Beam6TOB;
      std::vector<double> Beam7TOB;
	  
      // Beams in TIB
      std::vector<double> Beam0TIB;
      std::vector<double> Beam1TIB;
      std::vector<double> Beam2TIB;
      std::vector<double> Beam3TIB;
      std::vector<double> Beam4TIB;
      std::vector<double> Beam5TIB;
      std::vector<double> Beam6TIB;
      std::vector<double> Beam7TIB;


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
      
      
      Beam1R4PosTEC2TEC.push_back(theLaserPhi.at(18 + theIteration * 434));
      Beam1R4PosTEC2TEC.push_back(theLaserPhiError.at(18 + theIteration * 434));
      Beam1R4PosTEC2TEC.push_back(theLaserPhi.at(19 + theIteration * 434));
      Beam1R4PosTEC2TEC.push_back(theLaserPhiError.at(19 + theIteration * 434));
      Beam1R4PosTEC2TEC.push_back(theLaserPhi.at(20 + theIteration * 434));
      Beam1R4PosTEC2TEC.push_back(theLaserPhiError.at(20 + theIteration * 434));
      Beam1R4PosTEC2TEC.push_back(theLaserPhi.at(21 + theIteration * 434));
      Beam1R4PosTEC2TEC.push_back(theLaserPhiError.at(21 + theIteration * 434));
      Beam1R4PosTEC2TEC.push_back(theLaserPhi.at(22 + theIteration * 434));
      Beam1R4PosTEC2TEC.push_back(theLaserPhiError.at(22 + theIteration * 434));
      
      
      Beam2R4PosTEC2TEC.push_back(theLaserPhi.at(32 + theIteration * 434));
      Beam2R4PosTEC2TEC.push_back(theLaserPhiError.at(32 + theIteration * 434));
      Beam2R4PosTEC2TEC.push_back(theLaserPhi.at(33 + theIteration * 434));
      Beam2R4PosTEC2TEC.push_back(theLaserPhiError.at(33 + theIteration * 434));
      Beam2R4PosTEC2TEC.push_back(theLaserPhi.at(34 + theIteration * 434));
      Beam2R4PosTEC2TEC.push_back(theLaserPhiError.at(34 + theIteration * 434));
      Beam2R4PosTEC2TEC.push_back(theLaserPhi.at(35 + theIteration * 434));
      Beam2R4PosTEC2TEC.push_back(theLaserPhiError.at(35 + theIteration * 434));
      Beam2R4PosTEC2TEC.push_back(theLaserPhi.at(36 + theIteration * 434));
      Beam2R4PosTEC2TEC.push_back(theLaserPhiError.at(36 + theIteration * 434));
  
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
  
 
      Beam4R4PosTEC2TEC.push_back(theLaserPhi.at(55 + theIteration * 434));
      Beam4R4PosTEC2TEC.push_back(theLaserPhiError.at(55 + theIteration * 434));
      Beam4R4PosTEC2TEC.push_back(theLaserPhi.at(56 + theIteration * 434));
      Beam4R4PosTEC2TEC.push_back(theLaserPhiError.at(56 + theIteration * 434));
      Beam4R4PosTEC2TEC.push_back(theLaserPhi.at(57 + theIteration * 434));
      Beam4R4PosTEC2TEC.push_back(theLaserPhiError.at(57 + theIteration * 434));
      Beam4R4PosTEC2TEC.push_back(theLaserPhi.at(58 + theIteration * 434));
      Beam4R4PosTEC2TEC.push_back(theLaserPhiError.at(58 + theIteration * 434));
      Beam4R4PosTEC2TEC.push_back(theLaserPhi.at(59 + theIteration * 434));
      Beam4R4PosTEC2TEC.push_back(theLaserPhiError.at(59 + theIteration * 434));
  
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
  
  
      Beam6R4PosTEC2TEC.push_back(theLaserPhi.at(78 + theIteration * 434));
      Beam6R4PosTEC2TEC.push_back(theLaserPhiError.at(78 + theIteration * 434));
      Beam6R4PosTEC2TEC.push_back(theLaserPhi.at(79 + theIteration * 434));
      Beam6R4PosTEC2TEC.push_back(theLaserPhiError.at(79 + theIteration * 434));
      Beam6R4PosTEC2TEC.push_back(theLaserPhi.at(80 + theIteration * 434));
      Beam6R4PosTEC2TEC.push_back(theLaserPhiError.at(80 + theIteration * 434));
      Beam6R4PosTEC2TEC.push_back(theLaserPhi.at(81 + theIteration * 434));
      Beam6R4PosTEC2TEC.push_back(theLaserPhiError.at(81 + theIteration * 434));
      Beam6R4PosTEC2TEC.push_back(theLaserPhi.at(82 + theIteration * 434));
      Beam6R4PosTEC2TEC.push_back(theLaserPhiError.at(82 + theIteration * 434));
  
  
      Beam7R4PosTEC2TEC.push_back(theLaserPhi.at(92 + theIteration * 434));
      Beam7R4PosTEC2TEC.push_back(theLaserPhiError.at(92 + theIteration * 434));
      Beam7R4PosTEC2TEC.push_back(theLaserPhi.at(93 + theIteration * 434));
      Beam7R4PosTEC2TEC.push_back(theLaserPhiError.at(93 + theIteration * 434));
      Beam7R4PosTEC2TEC.push_back(theLaserPhi.at(94 + theIteration * 434));
      Beam7R4PosTEC2TEC.push_back(theLaserPhiError.at(94 + theIteration * 434));
      Beam7R4PosTEC2TEC.push_back(theLaserPhi.at(95 + theIteration * 434));
      Beam7R4PosTEC2TEC.push_back(theLaserPhiError.at(95 + theIteration * 434));
      Beam7R4PosTEC2TEC.push_back(theLaserPhi.at(96 + theIteration * 434));
      Beam7R4PosTEC2TEC.push_back(theLaserPhiError.at(96 + theIteration * 434));
  

      // Beams in TEC-
      Beam0R4NegTEC.push_back(theLaserPhi.at(169 + theIteration * 434));
      Beam0R4NegTEC.push_back(theLaserPhiError.at(169 + theIteration * 434));
      Beam0R4NegTEC.push_back(theLaserPhi.at(170 + theIteration * 434));
      Beam0R4NegTEC.push_back(theLaserPhiError.at(170 + theIteration * 434));
      Beam0R4NegTEC.push_back(theLaserPhi.at(171 + theIteration * 434));
      Beam0R4NegTEC.push_back(theLaserPhiError.at(171 + theIteration * 434));
      Beam0R4NegTEC.push_back(theLaserPhi.at(172 + theIteration * 434));
      Beam0R4NegTEC.push_back(theLaserPhiError.at(172 + theIteration * 434));
      Beam0R4NegTEC.push_back(theLaserPhi.at(173 + theIteration * 434));
      Beam0R4NegTEC.push_back(theLaserPhiError.at(173 + theIteration * 434));
  
  
      Beam1R4NegTEC2TEC.push_back(theLaserPhi.at(187 + theIteration * 434));
      Beam1R4NegTEC2TEC.push_back(theLaserPhiError.at(187 + theIteration * 434));
      Beam1R4NegTEC2TEC.push_back(theLaserPhi.at(188 + theIteration * 434));
      Beam1R4NegTEC2TEC.push_back(theLaserPhiError.at(188 + theIteration * 434));
      Beam1R4NegTEC2TEC.push_back(theLaserPhi.at(189 + theIteration * 434));
      Beam1R4NegTEC2TEC.push_back(theLaserPhiError.at(189 + theIteration * 434));
      Beam1R4NegTEC2TEC.push_back(theLaserPhi.at(190 + theIteration * 434));
      Beam1R4NegTEC2TEC.push_back(theLaserPhiError.at(190 + theIteration * 434));
      Beam1R4NegTEC2TEC.push_back(theLaserPhi.at(191 + theIteration * 434));
      Beam1R4NegTEC2TEC.push_back(theLaserPhiError.at(191 + theIteration * 434));
  
  
      Beam2R4NegTEC2TEC.push_back(theLaserPhi.at(201 + theIteration * 434));
      Beam2R4NegTEC2TEC.push_back(theLaserPhiError.at(201 + theIteration * 434));
      Beam2R4NegTEC2TEC.push_back(theLaserPhi.at(202 + theIteration * 434));
      Beam2R4NegTEC2TEC.push_back(theLaserPhiError.at(202 + theIteration * 434));
      Beam2R4NegTEC2TEC.push_back(theLaserPhi.at(203 + theIteration * 434));
      Beam2R4NegTEC2TEC.push_back(theLaserPhiError.at(203 + theIteration * 434));
      Beam2R4NegTEC2TEC.push_back(theLaserPhi.at(204 + theIteration * 434));
      Beam2R4NegTEC2TEC.push_back(theLaserPhiError.at(204 + theIteration * 434));
      Beam2R4NegTEC2TEC.push_back(theLaserPhi.at(205 + theIteration * 434));
      Beam2R4NegTEC2TEC.push_back(theLaserPhiError.at(205 + theIteration * 434));
  
      Beam3R4NegTEC.push_back(theLaserPhi.at(206 + theIteration * 434));
      Beam3R4NegTEC.push_back(theLaserPhiError.at(206 + theIteration * 434));
      Beam3R4NegTEC.push_back(theLaserPhi.at(207 + theIteration * 434));
      Beam3R4NegTEC.push_back(theLaserPhiError.at(207 + theIteration * 434));
      Beam3R4NegTEC.push_back(theLaserPhi.at(208 + theIteration * 434));
      Beam3R4NegTEC.push_back(theLaserPhiError.at(208 + theIteration * 434));
      Beam3R4NegTEC.push_back(theLaserPhi.at(209 + theIteration * 434));
      Beam3R4NegTEC.push_back(theLaserPhiError.at(209 + theIteration * 434));
      Beam3R4NegTEC.push_back(theLaserPhi.at(210 + theIteration * 434));
      Beam3R4NegTEC.push_back(theLaserPhiError.at(210 + theIteration * 434));
  
  
      Beam4R4NegTEC2TEC.push_back(theLaserPhi.at(224 + theIteration * 434));
      Beam4R4NegTEC2TEC.push_back(theLaserPhiError.at(224 + theIteration * 434));
      Beam4R4NegTEC2TEC.push_back(theLaserPhi.at(225 + theIteration * 434));
      Beam4R4NegTEC2TEC.push_back(theLaserPhiError.at(225 + theIteration * 434));
      Beam4R4NegTEC2TEC.push_back(theLaserPhi.at(226 + theIteration * 434));
      Beam4R4NegTEC2TEC.push_back(theLaserPhiError.at(226 + theIteration * 434));
      Beam4R4NegTEC2TEC.push_back(theLaserPhi.at(227 + theIteration * 434));
      Beam4R4NegTEC2TEC.push_back(theLaserPhiError.at(227 + theIteration * 434));
      Beam4R4NegTEC2TEC.push_back(theLaserPhi.at(228 + theIteration * 434));
      Beam4R4NegTEC2TEC.push_back(theLaserPhiError.at(228 + theIteration * 434));
  
      Beam5R4NegTEC.push_back(theLaserPhi.at(229 + theIteration * 434));
      Beam5R4NegTEC.push_back(theLaserPhiError.at(229 + theIteration * 434));
      Beam5R4NegTEC.push_back(theLaserPhi.at(230 + theIteration * 434));
      Beam5R4NegTEC.push_back(theLaserPhiError.at(230 + theIteration * 434));
      Beam5R4NegTEC.push_back(theLaserPhi.at(231 + theIteration * 434));
      Beam5R4NegTEC.push_back(theLaserPhiError.at(231 + theIteration * 434));
      Beam5R4NegTEC.push_back(theLaserPhi.at(232 + theIteration * 434));
      Beam5R4NegTEC.push_back(theLaserPhiError.at(232 + theIteration * 434));
      Beam5R4NegTEC.push_back(theLaserPhi.at(233 + theIteration * 434));
      Beam5R4NegTEC.push_back(theLaserPhiError.at(233 + theIteration * 434));
  
  
      Beam6R4NegTEC2TEC.push_back(theLaserPhi.at(247 + theIteration * 434));
      Beam6R4NegTEC2TEC.push_back(theLaserPhiError.at(247 + theIteration * 434));
      Beam6R4NegTEC2TEC.push_back(theLaserPhi.at(248 + theIteration * 434));
      Beam6R4NegTEC2TEC.push_back(theLaserPhiError.at(248 + theIteration * 434));
      Beam6R4NegTEC2TEC.push_back(theLaserPhi.at(249 + theIteration * 434));
      Beam6R4NegTEC2TEC.push_back(theLaserPhiError.at(249 + theIteration * 434));
      Beam6R4NegTEC2TEC.push_back(theLaserPhi.at(250 + theIteration * 434));
      Beam6R4NegTEC2TEC.push_back(theLaserPhiError.at(250 + theIteration * 434));
      Beam6R4NegTEC2TEC.push_back(theLaserPhi.at(251 + theIteration * 434));
      Beam6R4NegTEC2TEC.push_back(theLaserPhiError.at(251 + theIteration * 434));
  
  
      Beam7R4NegTEC2TEC.push_back(theLaserPhi.at(261 + theIteration * 434));
      Beam7R4NegTEC2TEC.push_back(theLaserPhiError.at(261 + theIteration * 434));
      Beam7R4NegTEC2TEC.push_back(theLaserPhi.at(262 + theIteration * 434));
      Beam7R4NegTEC2TEC.push_back(theLaserPhiError.at(262 + theIteration * 434));
      Beam7R4NegTEC2TEC.push_back(theLaserPhi.at(263 + theIteration * 434));
      Beam7R4NegTEC2TEC.push_back(theLaserPhiError.at(263 + theIteration * 434));
      Beam7R4NegTEC2TEC.push_back(theLaserPhi.at(264 + theIteration * 434));
      Beam7R4NegTEC2TEC.push_back(theLaserPhiError.at(264 + theIteration * 434));
      Beam7R4NegTEC2TEC.push_back(theLaserPhi.at(265 + theIteration * 434));
      Beam7R4NegTEC2TEC.push_back(theLaserPhiError.at(265 + theIteration * 434));
  

      // Beams in TOB
      Beam0TOB.push_back(theLaserPhi.at(338 + theIteration * 434));
      Beam0TOB.push_back(theLaserPhiError.at(338 + theIteration * 434));
      Beam0TOB.push_back(theLaserPhi.at(339 + theIteration * 434));
      Beam0TOB.push_back(theLaserPhiError.at(339 + theIteration * 434));
      Beam0TOB.push_back(theLaserPhi.at(340 + theIteration * 434));
      Beam0TOB.push_back(theLaserPhiError.at(340 + theIteration * 434));
      Beam0TOB.push_back(theLaserPhi.at(341 + theIteration * 434));
      Beam0TOB.push_back(theLaserPhiError.at(341 + theIteration * 434));
      Beam0TOB.push_back(theLaserPhi.at(342 + theIteration * 434));
      Beam0TOB.push_back(theLaserPhiError.at(342 + theIteration * 434));
      Beam0TOB.push_back(theLaserPhi.at(343 + theIteration * 434));
      Beam0TOB.push_back(theLaserPhiError.at(343 + theIteration * 434));

      Beam1TOB.push_back(theLaserPhi.at(344 + theIteration * 434));
      Beam1TOB.push_back(theLaserPhiError.at(344 + theIteration * 434));
      Beam1TOB.push_back(theLaserPhi.at(345 + theIteration * 434));
      Beam1TOB.push_back(theLaserPhiError.at(345 + theIteration * 434));
      Beam1TOB.push_back(theLaserPhi.at(346 + theIteration * 434));
      Beam1TOB.push_back(theLaserPhiError.at(346 + theIteration * 434));
      Beam1TOB.push_back(theLaserPhi.at(347 + theIteration * 434));
      Beam1TOB.push_back(theLaserPhiError.at(347 + theIteration * 434));
      Beam1TOB.push_back(theLaserPhi.at(348 + theIteration * 434));
      Beam1TOB.push_back(theLaserPhiError.at(348 + theIteration * 434));
      Beam1TOB.push_back(theLaserPhi.at(349 + theIteration * 434));
      Beam1TOB.push_back(theLaserPhiError.at(349 + theIteration * 434));

      Beam2TOB.push_back(theLaserPhi.at(350 + theIteration * 434));
      Beam2TOB.push_back(theLaserPhiError.at(350 + theIteration * 434));
      Beam2TOB.push_back(theLaserPhi.at(351 + theIteration * 434));
      Beam2TOB.push_back(theLaserPhiError.at(351 + theIteration * 434));
      Beam2TOB.push_back(theLaserPhi.at(352 + theIteration * 434));
      Beam2TOB.push_back(theLaserPhiError.at(352 + theIteration * 434));
      Beam2TOB.push_back(theLaserPhi.at(353 + theIteration * 434));
      Beam2TOB.push_back(theLaserPhiError.at(353 + theIteration * 434));
      Beam2TOB.push_back(theLaserPhi.at(354 + theIteration * 434));
      Beam2TOB.push_back(theLaserPhiError.at(354 + theIteration * 434));
      Beam2TOB.push_back(theLaserPhi.at(355 + theIteration * 434));
      Beam2TOB.push_back(theLaserPhiError.at(355 + theIteration * 434));

      Beam3TOB.push_back(theLaserPhi.at(356 + theIteration * 434));
      Beam3TOB.push_back(theLaserPhiError.at(356 + theIteration * 434));
      Beam3TOB.push_back(theLaserPhi.at(357 + theIteration * 434));
      Beam3TOB.push_back(theLaserPhiError.at(357 + theIteration * 434));
      Beam3TOB.push_back(theLaserPhi.at(358 + theIteration * 434));
      Beam3TOB.push_back(theLaserPhiError.at(358 + theIteration * 434));
      Beam3TOB.push_back(theLaserPhi.at(359 + theIteration * 434));
      Beam3TOB.push_back(theLaserPhiError.at(359 + theIteration * 434));
      Beam3TOB.push_back(theLaserPhi.at(360 + theIteration * 434));
      Beam3TOB.push_back(theLaserPhiError.at(360 + theIteration * 434));
      Beam3TOB.push_back(theLaserPhi.at(361 + theIteration * 434));
      Beam3TOB.push_back(theLaserPhiError.at(361 + theIteration * 434));

      Beam4TOB.push_back(theLaserPhi.at(362 + theIteration * 434));
      Beam4TOB.push_back(theLaserPhiError.at(362 + theIteration * 434));
      Beam4TOB.push_back(theLaserPhi.at(363 + theIteration * 434));
      Beam4TOB.push_back(theLaserPhiError.at(363 + theIteration * 434));
      Beam4TOB.push_back(theLaserPhi.at(364 + theIteration * 434));
      Beam4TOB.push_back(theLaserPhiError.at(364 + theIteration * 434));
      Beam4TOB.push_back(theLaserPhi.at(365 + theIteration * 434));
      Beam4TOB.push_back(theLaserPhiError.at(365 + theIteration * 434));
      Beam4TOB.push_back(theLaserPhi.at(366 + theIteration * 434));
      Beam4TOB.push_back(theLaserPhiError.at(366 + theIteration * 434));
      Beam4TOB.push_back(theLaserPhi.at(367 + theIteration * 434));
      Beam4TOB.push_back(theLaserPhiError.at(367 + theIteration * 434));

      Beam5TOB.push_back(theLaserPhi.at(368 + theIteration * 434));
      Beam5TOB.push_back(theLaserPhiError.at(368 + theIteration * 434));
      Beam5TOB.push_back(theLaserPhi.at(369 + theIteration * 434));
      Beam5TOB.push_back(theLaserPhiError.at(369 + theIteration * 434));
      Beam5TOB.push_back(theLaserPhi.at(370 + theIteration * 434));
      Beam5TOB.push_back(theLaserPhiError.at(370 + theIteration * 434));
      Beam5TOB.push_back(theLaserPhi.at(371 + theIteration * 434));
      Beam5TOB.push_back(theLaserPhiError.at(371 + theIteration * 434));
      Beam5TOB.push_back(theLaserPhi.at(372 + theIteration * 434));
      Beam5TOB.push_back(theLaserPhiError.at(372 + theIteration * 434));
      Beam5TOB.push_back(theLaserPhi.at(373 + theIteration * 434));
      Beam5TOB.push_back(theLaserPhiError.at(373 + theIteration * 434));

      Beam6TOB.push_back(theLaserPhi.at(374 + theIteration * 434));
      Beam6TOB.push_back(theLaserPhiError.at(374 + theIteration * 434));
      Beam6TOB.push_back(theLaserPhi.at(375 + theIteration * 434));
      Beam6TOB.push_back(theLaserPhiError.at(375 + theIteration * 434));
      Beam6TOB.push_back(theLaserPhi.at(376 + theIteration * 434));
      Beam6TOB.push_back(theLaserPhiError.at(376 + theIteration * 434));
      Beam6TOB.push_back(theLaserPhi.at(377 + theIteration * 434));
      Beam6TOB.push_back(theLaserPhiError.at(377 + theIteration * 434));
      Beam6TOB.push_back(theLaserPhi.at(378 + theIteration * 434));
      Beam6TOB.push_back(theLaserPhiError.at(378 + theIteration * 434));
      Beam6TOB.push_back(theLaserPhi.at(379 + theIteration * 434));
      Beam6TOB.push_back(theLaserPhiError.at(379 + theIteration * 434));

      Beam7TOB.push_back(theLaserPhi.at(380 + theIteration * 434));
      Beam7TOB.push_back(theLaserPhiError.at(380 + theIteration * 434));
      Beam7TOB.push_back(theLaserPhi.at(381 + theIteration * 434));
      Beam7TOB.push_back(theLaserPhiError.at(381 + theIteration * 434));
      Beam7TOB.push_back(theLaserPhi.at(382 + theIteration * 434));
      Beam7TOB.push_back(theLaserPhiError.at(382 + theIteration * 434));
      Beam7TOB.push_back(theLaserPhi.at(383 + theIteration * 434));
      Beam7TOB.push_back(theLaserPhiError.at(383 + theIteration * 434));
      Beam7TOB.push_back(theLaserPhi.at(384 + theIteration * 434));
      Beam7TOB.push_back(theLaserPhiError.at(384 + theIteration * 434));
      Beam7TOB.push_back(theLaserPhi.at(385 + theIteration * 434));
      Beam7TOB.push_back(theLaserPhiError.at(385 + theIteration * 434));


      // Beams in TIB
      Beam0TIB.push_back(theLaserPhi.at(386 + theIteration * 434));
      Beam0TIB.push_back(theLaserPhiError.at(386 + theIteration * 434));
      Beam0TIB.push_back(theLaserPhi.at(387 + theIteration * 434));
      Beam0TIB.push_back(theLaserPhiError.at(387 + theIteration * 434));
      Beam0TIB.push_back(theLaserPhi.at(388 + theIteration * 434));
      Beam0TIB.push_back(theLaserPhiError.at(388 + theIteration * 434));
      Beam0TIB.push_back(theLaserPhi.at(389 + theIteration * 434));
      Beam0TIB.push_back(theLaserPhiError.at(389 + theIteration * 434));
      Beam0TIB.push_back(theLaserPhi.at(390 + theIteration * 434));
      Beam0TIB.push_back(theLaserPhiError.at(390 + theIteration * 434));
      Beam0TIB.push_back(theLaserPhi.at(391 + theIteration * 434));
      Beam0TIB.push_back(theLaserPhiError.at(391 + theIteration * 434));

      Beam1TIB.push_back(theLaserPhi.at(392 + theIteration * 434));
      Beam1TIB.push_back(theLaserPhiError.at(392 + theIteration * 434));
      Beam1TIB.push_back(theLaserPhi.at(393 + theIteration * 434));
      Beam1TIB.push_back(theLaserPhiError.at(393 + theIteration * 434));
      Beam1TIB.push_back(theLaserPhi.at(394 + theIteration * 434));
      Beam1TIB.push_back(theLaserPhiError.at(394 + theIteration * 434));
      Beam1TIB.push_back(theLaserPhi.at(395 + theIteration * 434));
      Beam1TIB.push_back(theLaserPhiError.at(395 + theIteration * 434));
      Beam1TIB.push_back(theLaserPhi.at(396 + theIteration * 434));
      Beam1TIB.push_back(theLaserPhiError.at(396 + theIteration * 434));
      Beam1TIB.push_back(theLaserPhi.at(397 + theIteration * 434));
      Beam1TIB.push_back(theLaserPhiError.at(397 + theIteration * 434));

      Beam2TIB.push_back(theLaserPhi.at(398 + theIteration * 434));
      Beam2TIB.push_back(theLaserPhiError.at(398 + theIteration * 434));
      Beam2TIB.push_back(theLaserPhi.at(399 + theIteration * 434));
      Beam2TIB.push_back(theLaserPhiError.at(399 + theIteration * 434));
      Beam2TIB.push_back(theLaserPhi.at(400 + theIteration * 434));
      Beam2TIB.push_back(theLaserPhiError.at(400 + theIteration * 434));
      Beam2TIB.push_back(theLaserPhi.at(401 + theIteration * 434));
      Beam2TIB.push_back(theLaserPhiError.at(401 + theIteration * 434));
      Beam2TIB.push_back(theLaserPhi.at(402 + theIteration * 434));
      Beam2TIB.push_back(theLaserPhiError.at(402 + theIteration * 434));
      Beam2TIB.push_back(theLaserPhi.at(403 + theIteration * 434));
      Beam2TIB.push_back(theLaserPhiError.at(403 + theIteration * 434));

      Beam3TIB.push_back(theLaserPhi.at(404 + theIteration * 434));
      Beam3TIB.push_back(theLaserPhiError.at(404 + theIteration * 434));
      Beam3TIB.push_back(theLaserPhi.at(405 + theIteration * 434));
      Beam3TIB.push_back(theLaserPhiError.at(405 + theIteration * 434));
      Beam3TIB.push_back(theLaserPhi.at(406 + theIteration * 434));
      Beam3TIB.push_back(theLaserPhiError.at(406 + theIteration * 434));
      Beam3TIB.push_back(theLaserPhi.at(407 + theIteration * 434));
      Beam3TIB.push_back(theLaserPhiError.at(407 + theIteration * 434));
      Beam3TIB.push_back(theLaserPhi.at(408 + theIteration * 434));
      Beam3TIB.push_back(theLaserPhiError.at(408 + theIteration * 434));
      Beam3TIB.push_back(theLaserPhi.at(409 + theIteration * 434));
      Beam3TIB.push_back(theLaserPhiError.at(409 + theIteration * 434));

      Beam4TIB.push_back(theLaserPhi.at(410 + theIteration * 434));
      Beam4TIB.push_back(theLaserPhiError.at(410 + theIteration * 434));
      Beam4TIB.push_back(theLaserPhi.at(411 + theIteration * 434));
      Beam4TIB.push_back(theLaserPhiError.at(411 + theIteration * 434));
      Beam4TIB.push_back(theLaserPhi.at(412 + theIteration * 434));
      Beam4TIB.push_back(theLaserPhiError.at(412 + theIteration * 434));
      Beam4TIB.push_back(theLaserPhi.at(413 + theIteration * 434));
      Beam4TIB.push_back(theLaserPhiError.at(413 + theIteration * 434));
      Beam4TIB.push_back(theLaserPhi.at(414 + theIteration * 434));
      Beam4TIB.push_back(theLaserPhiError.at(414 + theIteration * 434));
      Beam4TIB.push_back(theLaserPhi.at(415 + theIteration * 434));
      Beam4TIB.push_back(theLaserPhiError.at(415 + theIteration * 434));

      Beam5TIB.push_back(theLaserPhi.at(416 + theIteration * 434));
      Beam5TIB.push_back(theLaserPhiError.at(416 + theIteration * 434));
      Beam5TIB.push_back(theLaserPhi.at(417 + theIteration * 434));
      Beam5TIB.push_back(theLaserPhiError.at(417 + theIteration * 434));
      Beam5TIB.push_back(theLaserPhi.at(418 + theIteration * 434));
      Beam5TIB.push_back(theLaserPhiError.at(418 + theIteration * 434));
      Beam5TIB.push_back(theLaserPhi.at(419 + theIteration * 434));
      Beam5TIB.push_back(theLaserPhiError.at(419 + theIteration * 434));
      Beam5TIB.push_back(theLaserPhi.at(420 + theIteration * 434));
      Beam5TIB.push_back(theLaserPhiError.at(420 + theIteration * 434));
      Beam5TIB.push_back(theLaserPhi.at(421 + theIteration * 434));
      Beam5TIB.push_back(theLaserPhiError.at(421 + theIteration * 434));

      Beam6TIB.push_back(theLaserPhi.at(422 + theIteration * 434));
      Beam6TIB.push_back(theLaserPhiError.at(422 + theIteration * 434));
      Beam6TIB.push_back(theLaserPhi.at(423 + theIteration * 434));
      Beam6TIB.push_back(theLaserPhiError.at(423 + theIteration * 434));
      Beam6TIB.push_back(theLaserPhi.at(424 + theIteration * 434));
      Beam6TIB.push_back(theLaserPhiError.at(424 + theIteration * 434));
      Beam6TIB.push_back(theLaserPhi.at(425 + theIteration * 434));
      Beam6TIB.push_back(theLaserPhiError.at(425 + theIteration * 434));
      Beam6TIB.push_back(theLaserPhi.at(426 + theIteration * 434));
      Beam6TIB.push_back(theLaserPhiError.at(426 + theIteration * 434));
      Beam6TIB.push_back(theLaserPhi.at(427 + theIteration * 434));
      Beam6TIB.push_back(theLaserPhiError.at(427 + theIteration * 434));

      Beam7TIB.push_back(theLaserPhi.at(428 + theIteration * 434));
      Beam7TIB.push_back(theLaserPhiError.at(428 + theIteration * 434));
      Beam7TIB.push_back(theLaserPhi.at(429 + theIteration * 434));
      Beam7TIB.push_back(theLaserPhiError.at(429 + theIteration * 434));
      Beam7TIB.push_back(theLaserPhi.at(430 + theIteration * 434));
      Beam7TIB.push_back(theLaserPhiError.at(430 + theIteration * 434));
      Beam7TIB.push_back(theLaserPhi.at(431 + theIteration * 434));
      Beam7TIB.push_back(theLaserPhiError.at(431 + theIteration * 434));
      Beam7TIB.push_back(theLaserPhi.at(432 + theIteration * 434));
      Beam7TIB.push_back(theLaserPhiError.at(432 + theIteration * 434));
      Beam7TIB.push_back(theLaserPhi.at(433 + theIteration * 434));
      Beam7TIB.push_back(theLaserPhiError.at(433 + theIteration * 434));


      /* *************************************** */
	  
      // create entry in the map for Beam 0 in Ring 4 of TEC+
      theBeamPosition["Beam 0 in Ring 4 in TEC+"] = Beam0R4PosTEC;
      // create entry in the map for Beam 1 in Ring 4 of TEC2TEC
      theBeamPosition["Beam 1 in TEC2TEC in TEC+"] = Beam1R4PosTEC2TEC;
      // create entry in the map for Beam 2 in Ring 4 of TEC2TEC
      theBeamPosition["Beam 2 in TEC2TEC in TEC+"] = Beam2R4PosTEC2TEC;
      // create entry in the map for Beam 3 in Ring 4 of TEC+
      theBeamPosition["Beam 3 in Ring 4 in TEC+"] = Beam3R4PosTEC;
      // create entry in the map for Beam 4 in Ring 4 of TEC2TEC
      theBeamPosition["Beam 4 in TEC2TEC in TEC+"] = Beam4R4PosTEC2TEC;
      // create entry in the map for Beam 5 in Ring 4 of TEC+
      theBeamPosition["Beam 5 in Ring 4 in TEC+"] = Beam5R4PosTEC;
      // create entry in the map for Beam 6 in Ring 4 of TEC2TEC
      theBeamPosition["Beam 6 in TEC2TEC in TEC+"] = Beam6R4PosTEC2TEC;
      // create entry in the map for Beam 7 in Ring 4 of TEC2TEC
      theBeamPosition["Beam 7 in TEC2TEC in TEC+"] = Beam7R4PosTEC2TEC;
	  
      // create entry in the map for Beam 0 in Ring 4 of TEC-
      theBeamPosition["Beam 0 in Ring 4 in TEC-"] = Beam0R4NegTEC;
      // create entry in the map for Beam 1 in Ring 4 of TEC2TEC
      theBeamPosition["Beam 1 in TEC2TEC in TEC-"] = Beam1R4NegTEC2TEC;
      // create entry in the map for Beam 2 in Ring 4 of TEC2TEC
      theBeamPosition["Beam 2 in TEC2TEC in TEC-"] = Beam2R4NegTEC2TEC;
      // create entry in the map for Beam 3 in Ring 4 of TEC-
      theBeamPosition["Beam 3 in Ring 4 in TEC-"] = Beam3R4NegTEC;
      // create entry in the map for Beam 4 in Ring 4 of TEC2TEC
      theBeamPosition["Beam 4 in TEC2TEC in TEC-"] = Beam4R4NegTEC2TEC;
      // create entry in the map for Beam 5 in Ring 4 of TEC-
      theBeamPosition["Beam 5 in Ring 4 in TEC-"] = Beam5R4NegTEC;
      // create entry in the map for Beam 6 in Ring 4 of TEC2TEC
      theBeamPosition["Beam 6 in TEC2TEC in TEC-"] = Beam6R4NegTEC2TEC;
      // create entry in the map for Beam 7 in Ring 4 of TEC2TEC
      theBeamPosition["Beam 7 in TEC2TEC in TEC-"] = Beam7R4NegTEC2TEC;
	  
      // create entry in the map for Beam 0 of TOB
      theBeamPosition["Beam 0 in TOB"] = Beam0TOB;
      // create entry in the map for Beam 1 of TOB
      theBeamPosition["Beam 1 in TOB"] = Beam1TOB;
      // create entry in the map for Beam 2 of TOB
      theBeamPosition["Beam 2 in TOB"] = Beam2TOB;
      // create entry in the map for Beam 3 of TOB
      theBeamPosition["Beam 3 in TOB"] = Beam3TOB;
      // create entry in the map for Beam 4 of TOB
      theBeamPosition["Beam 4 in TOB"] = Beam4TOB;
      // create entry in the map for Beam 5 of TOB
      theBeamPosition["Beam 5 in TOB"] = Beam5TOB;
      // create entry in the map for Beam 6 of TOB
      theBeamPosition["Beam 6 in TOB"] = Beam6TOB;
      // create entry in the map for Beam 7 of TOB
      theBeamPosition["Beam 7 in TOB"] = Beam7TOB;
	  
      // create entry in the map for Beam 0 of TIB
      theBeamPosition["Beam 0 in TIB"] = Beam0TIB;
      // create entry in the map for Beam 1 of TIB
      theBeamPosition["Beam 1 in TIB"] = Beam1TIB;
      // create entry in the map for Beam 2 of TIB
      theBeamPosition["Beam 2 in TIB"] = Beam2TIB;
      // create entry in the map for Beam 3 of TIB
      theBeamPosition["Beam 3 in TIB"] = Beam3TIB;
      // create entry in the map for Beam 4 of TIB
      theBeamPosition["Beam 4 in TIB"] = Beam4TIB;
      // create entry in the map for Beam 5 of TIB
      theBeamPosition["Beam 5 in TIB"] = Beam5TIB;
      // create entry in the map for Beam 6 of TIB
      theBeamPosition["Beam 6 in TIB"] = Beam6TIB;
      // create entry in the map for Beam 7 of TIB
      theBeamPosition["Beam 7 in TIB"] = Beam7TIB;
	  
      // *******************************************************

      // add the laserbeams
      theLaserAlignmentTrackerTEC2TEC->addLaserBeam(theBeamPosition["Beam 0 in Ring 4 in TEC+"],
					      theBeamPosition["Beam 0 in TOB"],
					      theBeamPosition["Beam 0 in TIB"],
					      theBeamPosition["Beam 0 in Ring 4 in TEC-"], 0, 4);
	  
      theLaserAlignmentTrackerTEC2TEC->addLaserBeam(theBeamPosition["Beam 1 in TEC2TEC in TEC+"],
					      theBeamPosition["Beam 1 in TOB"],
					      theBeamPosition["Beam 1 in TIB"],
					      theBeamPosition["Beam 1 in TEC2TEC in TEC-"], 1, 4);
	  
      theLaserAlignmentTrackerTEC2TEC->addLaserBeam(theBeamPosition["Beam 2 in TEC2TEC in TEC+"],
					      theBeamPosition["Beam 2 in TOB"],
					      theBeamPosition["Beam 2 in TIB"],
					      theBeamPosition["Beam 2 in TEC2TEC in TEC-"], 2, 4);
	  
      theLaserAlignmentTrackerTEC2TEC->addLaserBeam(theBeamPosition["Beam 3 in Ring 4 in TEC+"],
					      theBeamPosition["Beam 3 in TOB"],
					      theBeamPosition["Beam 3 in TIB"],
					      theBeamPosition["Beam 3 in Ring 4 in TEC-"], 3, 4);
	  
      theLaserAlignmentTrackerTEC2TEC->addLaserBeam(theBeamPosition["Beam 4 in TEC2TEC in TEC+"],
					      theBeamPosition["Beam 4 in TOB"],
					      theBeamPosition["Beam 4 in TIB"],
					      theBeamPosition["Beam 4 in TEC2TEC in TEC-"], 4, 4);
	  
      theLaserAlignmentTrackerTEC2TEC->addLaserBeam(theBeamPosition["Beam 5 in Ring 4 in TEC+"],
					      theBeamPosition["Beam 5 in TOB"],
					      theBeamPosition["Beam 5 in TIB"],
					      theBeamPosition["Beam 5 in Ring 4 in TEC-"], 5, 4);
	  
      theLaserAlignmentTrackerTEC2TEC->addLaserBeam(theBeamPosition["Beam 6 in TEC2TEC in TEC+"],
					      theBeamPosition["Beam 6 in TOB"],
					      theBeamPosition["Beam 6 in TIB"],
					      theBeamPosition["Beam 6 in TEC2TEC in TEC-"], 6, 4);
	  
      theLaserAlignmentTrackerTEC2TEC->addLaserBeam(theBeamPosition["Beam 7 in TEC2TEC in TEC+"],
					      theBeamPosition["Beam 7 in TOB"],
					      theBeamPosition["Beam 7 in TIB"],
					      theBeamPosition["Beam 7 in TEC2TEC in TEC-"], 7, 4);
    }
	  
  // finally do the fit of the global parameters when we have reached the last iteration
  edm::LogInfo("LaserAlignmentTEC2TEC") << "<LaserAlignmentTEC2TEC::alignment()>: doing the global fit ... ";
	      
  theLaserAlignmentTrackerTEC2TEC->doGlobalFit(theAlignableTracker);
  
  // delete the alignment Object for TEC-TOB-TIB-TEC
  if (theLaserAlignmentTrackerTEC2TEC != 0) { delete theLaserAlignmentTrackerTEC2TEC; }
}
