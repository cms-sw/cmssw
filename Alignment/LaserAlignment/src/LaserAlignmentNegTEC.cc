/*
 * Alignment of TEC-
 */ 

#include "Alignment/LaserAlignment/interface/LaserAlignmentNegTEC.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

LaserAlignmentNegTEC::LaserAlignmentNegTEC()
{
}

LaserAlignmentNegTEC::~LaserAlignmentNegTEC()
{
}

void LaserAlignmentNegTEC::alignment(edm::ParameterSet const & theConf, 
				     AlignableTracker * theAlignableTracker,
				     int theNumberOfIterations, 
				     int theAlignmentIteration,
				     std::vector<double>& theLaserPhi, 
				     std::vector<double>& theLaserPhiError)
{
  edm::LogInfo("LaserAlignmentNegTEC") << " ***************************************************** "
				 << "\n *                Alignment of TEC-                  * "
				 << "\n ***************************************************** ";
  
  int theMaxIteration = theNumberOfIterations - 1;
  
  // create an alignment Object for the negative TEC
  theLaserAlignmentTrackerNegTEC = new LaserAlignmentAlgorithmNegTEC(theConf, theAlignmentIteration);
      
  // do the iterations for the local fits
  for (int theIteration = 0; theIteration < theNumberOfIterations; theIteration++)
    {
      // fill fitted Phi position of the Beams and the Errors on Phi into 
      // a map. Afterwards add the beams to the LaserAlignmentTracker ...
      
      LogDebug("LaserAlignmentNegTEC") << "  AC1CMS: Total number of Iterations = " << theMaxIteration
				 << "\n  AC1CMS: Current Iteration = " << theIteration
				 << "\n  AC1CMS: Current Alignment Iteration = "  << theAlignmentIteration;

      // string is the name (i.e. Beam 0 in TEC+) and std::vector contains
      // 9 values for phi and the error on phi (on each Disc!) for the beam
      std::map<std::string, std::vector<double> > theBeamPosition;
	  
      // fill now the map 
      // therefore loop over theLaserPhi and fill
      // Phi + Error in the map for each beam
	  
      // Beams in TEC-
      std::vector<double> Beam0R4NegTEC;
      std::vector<double> Beam1R4NegTEC;
      std::vector<double> Beam2R4NegTEC;
      std::vector<double> Beam3R4NegTEC;
      std::vector<double> Beam4R4NegTEC;
      std::vector<double> Beam5R4NegTEC;
      std::vector<double> Beam6R4NegTEC;
      std::vector<double> Beam7R4NegTEC;
	  
      std::vector<double> Beam0R6NegTEC;
      std::vector<double> Beam1R6NegTEC;
      std::vector<double> Beam2R6NegTEC;
      std::vector<double> Beam3R6NegTEC;
      std::vector<double> Beam4R6NegTEC;
      std::vector<double> Beam5R6NegTEC;
      std::vector<double> Beam6R6NegTEC;
      std::vector<double> Beam7R6NegTEC;
      
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
      Beam0R4NegTEC.push_back(theLaserPhi.at(174 + theIteration * 434));
      Beam0R4NegTEC.push_back(theLaserPhiError.at(174 + theIteration * 434));
      Beam0R4NegTEC.push_back(theLaserPhi.at(175 + theIteration * 434));
      Beam0R4NegTEC.push_back(theLaserPhiError.at(175 + theIteration * 434));
      Beam0R4NegTEC.push_back(theLaserPhi.at(176 + theIteration * 434));
      Beam0R4NegTEC.push_back(theLaserPhiError.at(176 + theIteration * 434));
      Beam0R4NegTEC.push_back(theLaserPhi.at(177 + theIteration * 434));
      Beam0R4NegTEC.push_back(theLaserPhiError.at(177 + theIteration * 434));
	  
      Beam1R4NegTEC.push_back(theLaserPhi.at(178 + theIteration * 434));
      Beam1R4NegTEC.push_back(theLaserPhiError.at(178 + theIteration * 434));
      Beam1R4NegTEC.push_back(theLaserPhi.at(179 + theIteration * 434));
      Beam1R4NegTEC.push_back(theLaserPhiError.at(179 + theIteration * 434));
      Beam1R4NegTEC.push_back(theLaserPhi.at(180 + theIteration * 434));
      Beam1R4NegTEC.push_back(theLaserPhiError.at(180 + theIteration * 434));
      Beam1R4NegTEC.push_back(theLaserPhi.at(181 + theIteration * 434));
      Beam1R4NegTEC.push_back(theLaserPhiError.at(181 + theIteration * 434));
      Beam1R4NegTEC.push_back(theLaserPhi.at(182 + theIteration * 434));
      Beam1R4NegTEC.push_back(theLaserPhiError.at(182 + theIteration * 434));
      Beam1R4NegTEC.push_back(theLaserPhi.at(183 + theIteration * 434));
      Beam1R4NegTEC.push_back(theLaserPhiError.at(183 + theIteration * 434));
      Beam1R4NegTEC.push_back(theLaserPhi.at(184 + theIteration * 434));
      Beam1R4NegTEC.push_back(theLaserPhiError.at(184 + theIteration * 434));
      Beam1R4NegTEC.push_back(theLaserPhi.at(185 + theIteration * 434));
      Beam1R4NegTEC.push_back(theLaserPhiError.at(185 + theIteration * 434));
      Beam1R4NegTEC.push_back(theLaserPhi.at(186 + theIteration * 434));
      Beam1R4NegTEC.push_back(theLaserPhiError.at(186 + theIteration * 434));
	  
	  
      Beam2R4NegTEC.push_back(theLaserPhi.at(192 + theIteration * 434));
      Beam2R4NegTEC.push_back(theLaserPhiError.at(192 + theIteration * 434));
      Beam2R4NegTEC.push_back(theLaserPhi.at(193 + theIteration * 434));
      Beam2R4NegTEC.push_back(theLaserPhiError.at(193 + theIteration * 434));
      Beam2R4NegTEC.push_back(theLaserPhi.at(194 + theIteration * 434));
      Beam2R4NegTEC.push_back(theLaserPhiError.at(194 + theIteration * 434));
      Beam2R4NegTEC.push_back(theLaserPhi.at(195 + theIteration * 434));
      Beam2R4NegTEC.push_back(theLaserPhiError.at(195 + theIteration * 434));
      Beam2R4NegTEC.push_back(theLaserPhi.at(196 + theIteration * 434));
      Beam2R4NegTEC.push_back(theLaserPhiError.at(196 + theIteration * 434));
      Beam2R4NegTEC.push_back(theLaserPhi.at(197 + theIteration * 434));
      Beam2R4NegTEC.push_back(theLaserPhiError.at(197 + theIteration * 434));
      Beam2R4NegTEC.push_back(theLaserPhi.at(198 + theIteration * 434));
      Beam2R4NegTEC.push_back(theLaserPhiError.at(198 + theIteration * 434));
      Beam2R4NegTEC.push_back(theLaserPhi.at(199 + theIteration * 434));
      Beam2R4NegTEC.push_back(theLaserPhiError.at(199 + theIteration * 434));
      Beam2R4NegTEC.push_back(theLaserPhi.at(200 + theIteration * 434));
      Beam2R4NegTEC.push_back(theLaserPhiError.at(200 + theIteration * 434));
	  
	  
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
      Beam3R4NegTEC.push_back(theLaserPhi.at(211 + theIteration * 434));
      Beam3R4NegTEC.push_back(theLaserPhiError.at(211 + theIteration * 434));
      Beam3R4NegTEC.push_back(theLaserPhi.at(212 + theIteration * 434));
      Beam3R4NegTEC.push_back(theLaserPhiError.at(212 + theIteration * 434));
      Beam3R4NegTEC.push_back(theLaserPhi.at(213 + theIteration * 434));
      Beam3R4NegTEC.push_back(theLaserPhiError.at(213 + theIteration * 434));
      Beam3R4NegTEC.push_back(theLaserPhi.at(214 + theIteration * 434));
      Beam3R4NegTEC.push_back(theLaserPhiError.at(214 + theIteration * 434));
  
      Beam4R4NegTEC.push_back(theLaserPhi.at(215 + theIteration * 434));
      Beam4R4NegTEC.push_back(theLaserPhiError.at(215 + theIteration * 434));
      Beam4R4NegTEC.push_back(theLaserPhi.at(216 + theIteration * 434));
      Beam4R4NegTEC.push_back(theLaserPhiError.at(216 + theIteration * 434));
      Beam4R4NegTEC.push_back(theLaserPhi.at(217 + theIteration * 434));
      Beam4R4NegTEC.push_back(theLaserPhiError.at(217 + theIteration * 434));
      Beam4R4NegTEC.push_back(theLaserPhi.at(218 + theIteration * 434));
      Beam4R4NegTEC.push_back(theLaserPhiError.at(218 + theIteration * 434));
      Beam4R4NegTEC.push_back(theLaserPhi.at(219 + theIteration * 434));
      Beam4R4NegTEC.push_back(theLaserPhiError.at(219 + theIteration * 434));
      Beam4R4NegTEC.push_back(theLaserPhi.at(220 + theIteration * 434));
      Beam4R4NegTEC.push_back(theLaserPhiError.at(220 + theIteration * 434));
      Beam4R4NegTEC.push_back(theLaserPhi.at(221 + theIteration * 434));
      Beam4R4NegTEC.push_back(theLaserPhiError.at(221 + theIteration * 434));
      Beam4R4NegTEC.push_back(theLaserPhi.at(222 + theIteration * 434));
      Beam4R4NegTEC.push_back(theLaserPhiError.at(222 + theIteration * 434));
      Beam4R4NegTEC.push_back(theLaserPhi.at(223 + theIteration * 434));
      Beam4R4NegTEC.push_back(theLaserPhiError.at(223 + theIteration * 434));
  
  
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
      Beam5R4NegTEC.push_back(theLaserPhi.at(234 + theIteration * 434));
      Beam5R4NegTEC.push_back(theLaserPhiError.at(234 + theIteration * 434));
      Beam5R4NegTEC.push_back(theLaserPhi.at(235 + theIteration * 434));
      Beam5R4NegTEC.push_back(theLaserPhiError.at(235 + theIteration * 434));
      Beam5R4NegTEC.push_back(theLaserPhi.at(236 + theIteration * 434));
      Beam5R4NegTEC.push_back(theLaserPhiError.at(236 + theIteration * 434));
      Beam5R4NegTEC.push_back(theLaserPhi.at(237 + theIteration * 434));
      Beam5R4NegTEC.push_back(theLaserPhiError.at(237 + theIteration * 434));
  
      Beam6R4NegTEC.push_back(theLaserPhi.at(238 + theIteration * 434));
      Beam6R4NegTEC.push_back(theLaserPhiError.at(238 + theIteration * 434));
      Beam6R4NegTEC.push_back(theLaserPhi.at(239 + theIteration * 434));
      Beam6R4NegTEC.push_back(theLaserPhiError.at(239 + theIteration * 434));
      Beam6R4NegTEC.push_back(theLaserPhi.at(240 + theIteration * 434));
      Beam6R4NegTEC.push_back(theLaserPhiError.at(240 + theIteration * 434));
      Beam6R4NegTEC.push_back(theLaserPhi.at(241 + theIteration * 434));
      Beam6R4NegTEC.push_back(theLaserPhiError.at(241 + theIteration * 434));
      Beam6R4NegTEC.push_back(theLaserPhi.at(242 + theIteration * 434));
      Beam6R4NegTEC.push_back(theLaserPhiError.at(242 + theIteration * 434));
      Beam6R4NegTEC.push_back(theLaserPhi.at(243 + theIteration * 434));
      Beam6R4NegTEC.push_back(theLaserPhiError.at(243 + theIteration * 434));
      Beam6R4NegTEC.push_back(theLaserPhi.at(244 + theIteration * 434));
      Beam6R4NegTEC.push_back(theLaserPhiError.at(244 + theIteration * 434));
      Beam6R4NegTEC.push_back(theLaserPhi.at(245 + theIteration * 434));
      Beam6R4NegTEC.push_back(theLaserPhiError.at(245 + theIteration * 434));
      Beam6R4NegTEC.push_back(theLaserPhi.at(246 + theIteration * 434));
      Beam6R4NegTEC.push_back(theLaserPhiError.at(246 + theIteration * 434));
  
  
      Beam7R4NegTEC.push_back(theLaserPhi.at(252 + theIteration * 434));
      Beam7R4NegTEC.push_back(theLaserPhiError.at(252 + theIteration * 434));
      Beam7R4NegTEC.push_back(theLaserPhi.at(253 + theIteration * 434));
      Beam7R4NegTEC.push_back(theLaserPhiError.at(253 + theIteration * 434));
      Beam7R4NegTEC.push_back(theLaserPhi.at(254 + theIteration * 434));
      Beam7R4NegTEC.push_back(theLaserPhiError.at(254 + theIteration * 434));
      Beam7R4NegTEC.push_back(theLaserPhi.at(255 + theIteration * 434));
      Beam7R4NegTEC.push_back(theLaserPhiError.at(255 + theIteration * 434));
      Beam7R4NegTEC.push_back(theLaserPhi.at(256 + theIteration * 434));
      Beam7R4NegTEC.push_back(theLaserPhiError.at(256 + theIteration * 434));
      Beam7R4NegTEC.push_back(theLaserPhi.at(257 + theIteration * 434));
      Beam7R4NegTEC.push_back(theLaserPhiError.at(257 + theIteration * 434));
      Beam7R4NegTEC.push_back(theLaserPhi.at(258 + theIteration * 434));
      Beam7R4NegTEC.push_back(theLaserPhiError.at(258 + theIteration * 434));
      Beam7R4NegTEC.push_back(theLaserPhi.at(259 + theIteration * 434));
      Beam7R4NegTEC.push_back(theLaserPhiError.at(259 + theIteration * 434));
      Beam7R4NegTEC.push_back(theLaserPhi.at(260 + theIteration * 434));
      Beam7R4NegTEC.push_back(theLaserPhiError.at(260 + theIteration * 434));
  
  
      Beam0R6NegTEC.push_back(theLaserPhi.at(266 + theIteration * 434));
      Beam0R6NegTEC.push_back(theLaserPhiError.at(266 + theIteration * 434));
      Beam0R6NegTEC.push_back(theLaserPhi.at(267 + theIteration * 434));
      Beam0R6NegTEC.push_back(theLaserPhiError.at(267 + theIteration * 434));
      Beam0R6NegTEC.push_back(theLaserPhi.at(268 + theIteration * 434));
      Beam0R6NegTEC.push_back(theLaserPhiError.at(268 + theIteration * 434));
      Beam0R6NegTEC.push_back(theLaserPhi.at(269 + theIteration * 434));
      Beam0R6NegTEC.push_back(theLaserPhiError.at(269 + theIteration * 434));
      Beam0R6NegTEC.push_back(theLaserPhi.at(270 + theIteration * 434));
      Beam0R6NegTEC.push_back(theLaserPhiError.at(270 + theIteration * 434));
      Beam0R6NegTEC.push_back(theLaserPhi.at(271 + theIteration * 434));
      Beam0R6NegTEC.push_back(theLaserPhiError.at(271 + theIteration * 434));
      Beam0R6NegTEC.push_back(theLaserPhi.at(272 + theIteration * 434));
      Beam0R6NegTEC.push_back(theLaserPhiError.at(272 + theIteration * 434));
      Beam0R6NegTEC.push_back(theLaserPhi.at(273 + theIteration * 434));
      Beam0R6NegTEC.push_back(theLaserPhiError.at(273 + theIteration * 434));
      Beam0R6NegTEC.push_back(theLaserPhi.at(274 + theIteration * 434));
      Beam0R6NegTEC.push_back(theLaserPhiError.at(274 + theIteration * 434));

      Beam1R6NegTEC.push_back(theLaserPhi.at(275 + theIteration * 434));
      Beam1R6NegTEC.push_back(theLaserPhiError.at(275 + theIteration * 434));
      Beam1R6NegTEC.push_back(theLaserPhi.at(276 + theIteration * 434));
      Beam1R6NegTEC.push_back(theLaserPhiError.at(276 + theIteration * 434));
      Beam1R6NegTEC.push_back(theLaserPhi.at(277 + theIteration * 434));
      Beam1R6NegTEC.push_back(theLaserPhiError.at(277 + theIteration * 434));
      Beam1R6NegTEC.push_back(theLaserPhi.at(278 + theIteration * 434));
      Beam1R6NegTEC.push_back(theLaserPhiError.at(278 + theIteration * 434));
      Beam1R6NegTEC.push_back(theLaserPhi.at(279 + theIteration * 434));
      Beam1R6NegTEC.push_back(theLaserPhiError.at(279 + theIteration * 434));
      Beam1R6NegTEC.push_back(theLaserPhi.at(280 + theIteration * 434));
      Beam1R6NegTEC.push_back(theLaserPhiError.at(280 + theIteration * 434));
      Beam1R6NegTEC.push_back(theLaserPhi.at(281 + theIteration * 434));
      Beam1R6NegTEC.push_back(theLaserPhiError.at(281 + theIteration * 434));
      Beam1R6NegTEC.push_back(theLaserPhi.at(282 + theIteration * 434));
      Beam1R6NegTEC.push_back(theLaserPhiError.at(282 + theIteration * 434));
      Beam1R6NegTEC.push_back(theLaserPhi.at(283 + theIteration * 434));
      Beam1R6NegTEC.push_back(theLaserPhiError.at(283 + theIteration * 434));

      Beam2R6NegTEC.push_back(theLaserPhi.at(284 + theIteration * 434));
      Beam2R6NegTEC.push_back(theLaserPhiError.at(284 + theIteration * 434));
      Beam2R6NegTEC.push_back(theLaserPhi.at(285 + theIteration * 434));
      Beam2R6NegTEC.push_back(theLaserPhiError.at(285 + theIteration * 434));
      Beam2R6NegTEC.push_back(theLaserPhi.at(286 + theIteration * 434));
      Beam2R6NegTEC.push_back(theLaserPhiError.at(286 + theIteration * 434));
      Beam2R6NegTEC.push_back(theLaserPhi.at(287 + theIteration * 434));
      Beam2R6NegTEC.push_back(theLaserPhiError.at(287 + theIteration * 434));
      Beam2R6NegTEC.push_back(theLaserPhi.at(288 + theIteration * 434));
      Beam2R6NegTEC.push_back(theLaserPhiError.at(288 + theIteration * 434));
      Beam2R6NegTEC.push_back(theLaserPhi.at(289 + theIteration * 434));
      Beam2R6NegTEC.push_back(theLaserPhiError.at(289 + theIteration * 434));
      Beam2R6NegTEC.push_back(theLaserPhi.at(290 + theIteration * 434));
      Beam2R6NegTEC.push_back(theLaserPhiError.at(290 + theIteration * 434));
      Beam2R6NegTEC.push_back(theLaserPhi.at(291 + theIteration * 434));
      Beam2R6NegTEC.push_back(theLaserPhiError.at(291 + theIteration * 434));
      Beam2R6NegTEC.push_back(theLaserPhi.at(292 + theIteration * 434));
      Beam2R6NegTEC.push_back(theLaserPhiError.at(292 + theIteration * 434));

      Beam3R6NegTEC.push_back(theLaserPhi.at(293 + theIteration * 434));
      Beam3R6NegTEC.push_back(theLaserPhiError.at(293 + theIteration * 434));
      Beam3R6NegTEC.push_back(theLaserPhi.at(294 + theIteration * 434));
      Beam3R6NegTEC.push_back(theLaserPhiError.at(294 + theIteration * 434));
      Beam3R6NegTEC.push_back(theLaserPhi.at(295 + theIteration * 434));
      Beam3R6NegTEC.push_back(theLaserPhiError.at(295 + theIteration * 434));
      Beam3R6NegTEC.push_back(theLaserPhi.at(296 + theIteration * 434));
      Beam3R6NegTEC.push_back(theLaserPhiError.at(296 + theIteration * 434));
      Beam3R6NegTEC.push_back(theLaserPhi.at(297 + theIteration * 434));
      Beam3R6NegTEC.push_back(theLaserPhiError.at(297 + theIteration * 434));
      Beam3R6NegTEC.push_back(theLaserPhi.at(298 + theIteration * 434));
      Beam3R6NegTEC.push_back(theLaserPhiError.at(298 + theIteration * 434));
      Beam3R6NegTEC.push_back(theLaserPhi.at(299 + theIteration * 434));
      Beam3R6NegTEC.push_back(theLaserPhiError.at(299 + theIteration * 434));
      Beam3R6NegTEC.push_back(theLaserPhi.at(300 + theIteration * 434));
      Beam3R6NegTEC.push_back(theLaserPhiError.at(300 + theIteration * 434));
      Beam3R6NegTEC.push_back(theLaserPhi.at(301 + theIteration * 434));
      Beam3R6NegTEC.push_back(theLaserPhiError.at(301 + theIteration * 434));

      Beam4R6NegTEC.push_back(theLaserPhi.at(302 + theIteration * 434));
      Beam4R6NegTEC.push_back(theLaserPhiError.at(302 + theIteration * 434));
      Beam4R6NegTEC.push_back(theLaserPhi.at(303 + theIteration * 434));
      Beam4R6NegTEC.push_back(theLaserPhiError.at(303 + theIteration * 434));
      Beam4R6NegTEC.push_back(theLaserPhi.at(304 + theIteration * 434));
      Beam4R6NegTEC.push_back(theLaserPhiError.at(304 + theIteration * 434));
      Beam4R6NegTEC.push_back(theLaserPhi.at(305 + theIteration * 434));
      Beam4R6NegTEC.push_back(theLaserPhiError.at(305 + theIteration * 434));
      Beam4R6NegTEC.push_back(theLaserPhi.at(306 + theIteration * 434));
      Beam4R6NegTEC.push_back(theLaserPhiError.at(306 + theIteration * 434));
      Beam4R6NegTEC.push_back(theLaserPhi.at(307 + theIteration * 434));
      Beam4R6NegTEC.push_back(theLaserPhiError.at(307 + theIteration * 434));
      Beam4R6NegTEC.push_back(theLaserPhi.at(308 + theIteration * 434));
      Beam4R6NegTEC.push_back(theLaserPhiError.at(308 + theIteration * 434));
      Beam4R6NegTEC.push_back(theLaserPhi.at(309 + theIteration * 434));
      Beam4R6NegTEC.push_back(theLaserPhiError.at(309 + theIteration * 434));
      Beam4R6NegTEC.push_back(theLaserPhi.at(310 + theIteration * 434));
      Beam4R6NegTEC.push_back(theLaserPhiError.at(310 + theIteration * 434));

      Beam5R6NegTEC.push_back(theLaserPhi.at(311 + theIteration * 434));
      Beam5R6NegTEC.push_back(theLaserPhiError.at(311 + theIteration * 434));
      Beam5R6NegTEC.push_back(theLaserPhi.at(312 + theIteration * 434));
      Beam5R6NegTEC.push_back(theLaserPhiError.at(312 + theIteration * 434));
      Beam5R6NegTEC.push_back(theLaserPhi.at(313 + theIteration * 434));
      Beam5R6NegTEC.push_back(theLaserPhiError.at(313 + theIteration * 434));
      Beam5R6NegTEC.push_back(theLaserPhi.at(314 + theIteration * 434));
      Beam5R6NegTEC.push_back(theLaserPhiError.at(314 + theIteration * 434));
      Beam5R6NegTEC.push_back(theLaserPhi.at(315 + theIteration * 434));
      Beam5R6NegTEC.push_back(theLaserPhiError.at(315 + theIteration * 434));
      Beam5R6NegTEC.push_back(theLaserPhi.at(316 + theIteration * 434));
      Beam5R6NegTEC.push_back(theLaserPhiError.at(316 + theIteration * 434));
      Beam5R6NegTEC.push_back(theLaserPhi.at(317 + theIteration * 434));
      Beam5R6NegTEC.push_back(theLaserPhiError.at(317 + theIteration * 434));
      Beam5R6NegTEC.push_back(theLaserPhi.at(318 + theIteration * 434));
      Beam5R6NegTEC.push_back(theLaserPhiError.at(318 + theIteration * 434));
      Beam5R6NegTEC.push_back(theLaserPhi.at(319 + theIteration * 434));
      Beam5R6NegTEC.push_back(theLaserPhiError.at(319 + theIteration * 434));

      Beam6R6NegTEC.push_back(theLaserPhi.at(320 + theIteration * 434));
      Beam6R6NegTEC.push_back(theLaserPhiError.at(320 + theIteration * 434));
      Beam6R6NegTEC.push_back(theLaserPhi.at(321 + theIteration * 434));
      Beam6R6NegTEC.push_back(theLaserPhiError.at(321 + theIteration * 434));
      Beam6R6NegTEC.push_back(theLaserPhi.at(322 + theIteration * 434));
      Beam6R6NegTEC.push_back(theLaserPhiError.at(322 + theIteration * 434));
      Beam6R6NegTEC.push_back(theLaserPhi.at(323 + theIteration * 434));
      Beam6R6NegTEC.push_back(theLaserPhiError.at(323 + theIteration * 434));
      Beam6R6NegTEC.push_back(theLaserPhi.at(324 + theIteration * 434));
      Beam6R6NegTEC.push_back(theLaserPhiError.at(324 + theIteration * 434));
      Beam6R6NegTEC.push_back(theLaserPhi.at(325 + theIteration * 434));
      Beam6R6NegTEC.push_back(theLaserPhiError.at(325 + theIteration * 434));
      Beam6R6NegTEC.push_back(theLaserPhi.at(326 + theIteration * 434));
      Beam6R6NegTEC.push_back(theLaserPhiError.at(326 + theIteration * 434));
      Beam6R6NegTEC.push_back(theLaserPhi.at(327 + theIteration * 434));
      Beam6R6NegTEC.push_back(theLaserPhiError.at(327 + theIteration * 434));
      Beam6R6NegTEC.push_back(theLaserPhi.at(328 + theIteration * 434));
      Beam6R6NegTEC.push_back(theLaserPhiError.at(328 + theIteration * 434));

      Beam7R6NegTEC.push_back(theLaserPhi.at(329 + theIteration * 434));
      Beam7R6NegTEC.push_back(theLaserPhiError.at(329 + theIteration * 434));
      Beam7R6NegTEC.push_back(theLaserPhi.at(330 + theIteration * 434));
      Beam7R6NegTEC.push_back(theLaserPhiError.at(330 + theIteration * 434));
      Beam7R6NegTEC.push_back(theLaserPhi.at(331 + theIteration * 434));
      Beam7R6NegTEC.push_back(theLaserPhiError.at(331 + theIteration * 434));
      Beam7R6NegTEC.push_back(theLaserPhi.at(332 + theIteration * 434));
      Beam7R6NegTEC.push_back(theLaserPhiError.at(332 + theIteration * 434));
      Beam7R6NegTEC.push_back(theLaserPhi.at(333 + theIteration * 434));
      Beam7R6NegTEC.push_back(theLaserPhiError.at(333 + theIteration * 434));
      Beam7R6NegTEC.push_back(theLaserPhi.at(334 + theIteration * 434));
      Beam7R6NegTEC.push_back(theLaserPhiError.at(334 + theIteration * 434));
      Beam7R6NegTEC.push_back(theLaserPhi.at(335 + theIteration * 434));
      Beam7R6NegTEC.push_back(theLaserPhiError.at(335 + theIteration * 434));
      Beam7R6NegTEC.push_back(theLaserPhi.at(336 + theIteration * 434));
      Beam7R6NegTEC.push_back(theLaserPhiError.at(336 + theIteration * 434));
      Beam7R6NegTEC.push_back(theLaserPhi.at(337 + theIteration * 434));
      Beam7R6NegTEC.push_back(theLaserPhiError.at(337 + theIteration * 434));

      /* *************************************** */

      // create entry in the map for Beam 0 in Ring 4 of TEC-
      theBeamPosition["Beam 0 in Ring 4 in TEC-"] = Beam0R4NegTEC;
      // create entry in the map for Beam 1 in Ring 4 of TEC-
      theBeamPosition["Beam 1 in Ring 4 in TEC-"] = Beam1R4NegTEC;
      // create entry in the map for Beam 2 in Ring 4 of TEC-
      theBeamPosition["Beam 2 in Ring 4 in TEC-"] = Beam2R4NegTEC;
      // create entry in the map for Beam 3 in Ring 4 of TEC-
      theBeamPosition["Beam 3 in Ring 4 in TEC-"] = Beam3R4NegTEC;
      // create entry in the map for Beam 4 in Ring 4 of TEC-
      theBeamPosition["Beam 4 in Ring 4 in TEC-"] = Beam4R4NegTEC;
      // create entry in the map for Beam 5 in Ring 4 of TEC-
      theBeamPosition["Beam 5 in Ring 4 in TEC-"] = Beam5R4NegTEC;
      // create entry in the map for Beam 6 in Ring 4 of TEC-
      theBeamPosition["Beam 6 in Ring 4 in TEC-"] = Beam6R4NegTEC;
      // create entry in the map for Beam 7 in Ring 4 of TEC-
      theBeamPosition["Beam 7 in Ring 4 in TEC-"] = Beam7R4NegTEC;
	  
      // create entry in the map for Beam 0 in Ring 6 of TEC-
      theBeamPosition["Beam 0 in Ring 6 in TEC-"] = Beam0R6NegTEC;
      // create entry in the map for Beam 1 in Ring 6 of TEC-
      theBeamPosition["Beam 1 in Ring 6 in TEC-"] = Beam1R6NegTEC;
      // create entry in the map for Beam 2 in Ring 6 of TEC-
      theBeamPosition["Beam 2 in Ring 6 in TEC-"] = Beam2R6NegTEC;
      // create entry in the map for Beam 3 in Ring 6 of TEC-
      theBeamPosition["Beam 3 in Ring 6 in TEC-"] = Beam3R6NegTEC;
      // create entry in the map for Beam 4 in Ring 6 of TEC-
      theBeamPosition["Beam 4 in Ring 6 in TEC-"] = Beam4R6NegTEC;
      // create entry in the map for Beam 5 in Ring 6 of TEC-
      theBeamPosition["Beam 5 in Ring 6 in TEC-"] = Beam5R6NegTEC;
      // create entry in the map for Beam 6 in Ring 6 of TEC-
      theBeamPosition["Beam 6 in Ring 6 in TEC-"] = Beam6R6NegTEC;
      // create entry in the map for Beam 7 in Ring 6 of TEC-
      theBeamPosition["Beam 7 in Ring 6 in TEC-"] = Beam7R6NegTEC;

      // *******************************************************

      // Ring 4 of TEC-
      theLaserAlignmentTrackerNegTEC->addLaserBeam(theBeamPosition["Beam 0 in Ring 4 in TEC-"], 0, 4);
      theLaserAlignmentTrackerNegTEC->addLaserBeam(theBeamPosition["Beam 1 in Ring 4 in TEC-"], 1, 4);
      theLaserAlignmentTrackerNegTEC->addLaserBeam(theBeamPosition["Beam 2 in Ring 4 in TEC-"], 2, 4);
      theLaserAlignmentTrackerNegTEC->addLaserBeam(theBeamPosition["Beam 3 in Ring 4 in TEC-"], 3, 4);
      theLaserAlignmentTrackerNegTEC->addLaserBeam(theBeamPosition["Beam 4 in Ring 4 in TEC-"], 4, 4);
      theLaserAlignmentTrackerNegTEC->addLaserBeam(theBeamPosition["Beam 5 in Ring 4 in TEC-"], 5, 4);
      theLaserAlignmentTrackerNegTEC->addLaserBeam(theBeamPosition["Beam 6 in Ring 4 in TEC-"], 6, 4);
      theLaserAlignmentTrackerNegTEC->addLaserBeam(theBeamPosition["Beam 7 in Ring 4 in TEC-"], 7, 4);
      // Ring 6 of TEC-
      theLaserAlignmentTrackerNegTEC->addLaserBeam(theBeamPosition["Beam 0 in Ring 6 in TEC-"], 0, 6);
      theLaserAlignmentTrackerNegTEC->addLaserBeam(theBeamPosition["Beam 1 in Ring 6 in TEC-"], 1, 6);
      theLaserAlignmentTrackerNegTEC->addLaserBeam(theBeamPosition["Beam 2 in Ring 6 in TEC-"], 2, 6);
      theLaserAlignmentTrackerNegTEC->addLaserBeam(theBeamPosition["Beam 3 in Ring 6 in TEC-"], 3, 6);
      theLaserAlignmentTrackerNegTEC->addLaserBeam(theBeamPosition["Beam 4 in Ring 6 in TEC-"], 4, 6);
      theLaserAlignmentTrackerNegTEC->addLaserBeam(theBeamPosition["Beam 5 in Ring 6 in TEC-"], 5, 6);
      theLaserAlignmentTrackerNegTEC->addLaserBeam(theBeamPosition["Beam 6 in Ring 6 in TEC-"], 6, 6);
      theLaserAlignmentTrackerNegTEC->addLaserBeam(theBeamPosition["Beam 7 in Ring 6 in TEC-"], 7, 6);
    }
	  
  // finally do the fit of the global parameters when we have reached the last iteration
  edm::LogInfo("LaserAlignmentNegTEC") << "<LaserAlignmentNegTEC::alignment()>:doing the global fit ... ";
	      
  theLaserAlignmentTrackerNegTEC->doGlobalFit(theAlignableTracker);
  
  // delete the alignment Object for the negtive TEC to avoid problems
  // with the alignment of the TIB, TOB and TEC
  if (theLaserAlignmentTrackerNegTEC != 0) { delete theLaserAlignmentTrackerNegTEC; }
}

