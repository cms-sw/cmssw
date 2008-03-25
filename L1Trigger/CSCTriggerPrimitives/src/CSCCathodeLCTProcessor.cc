//-----------------------------------------------------------------------------
//
//   Class: CSCCathodeLCTProcessor
//
//   Description: 
//     This class simulates the functionality of the cathode LCT card.  It is
//     run by the MotherBoard and returns up to two CathodeLCTs. It can be
//     run either in a test mode, where it is passed an array of comparator
//     times and comparator values, or in normal mode where it determines
//     the time and comparator information from the comparator digis.
//
//     The CathodeLCTs come in distrip and halfstrip flavors; they are sorted
//     (from best to worst) as follows: 6/6H, 5/6H, 6/6D, 4/6H, 5/6D, 4/6D.
//
//     Additional comments by Jason Mumford 01/31/01 (mumford@physics.ucla.edu)
//     Removed the card boundaries.  Changed the Pretrigger to emulate
//     the hardware electronic logic.  Also changed the keylayer to be the 4th
//     layer in a chamber instead of the 3rd layer from the interaction region.
//     The code is a more realistic simulation of hardware LCT logic now.
//
//   Author List: Benn Tannenbaum (1999), Jason Mumford (2001-2), Slava Valuev.
//                Porting from ORCA by S. Valuev (Slava.Valuev@cern.ch),
//                May 2006.
//
//   $Date: 2007/04/18 16:08:55 $
//   $Revision: 1.18 $
//
//   Modifications: 
//
//-----------------------------------------------------------------------------

#include <L1Trigger/CSCTriggerPrimitives/src/CSCCathodeLCTProcessor.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Utilities/interface/Exception.h>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>

//-----------------
// Static variables
//-----------------

// This is the strip pattern that we use for pretrigger.
// pre_hit_pattern[0][i] = layer. pre_hit_pattern[1][i] = key_strip offset.
const int CSCCathodeLCTProcessor::pre_hit_pattern[2][NUM_PATTERN_STRIPS] = {
  { 999,  0,  0,  0,  999,
    999,  1,  1,  1,  999,
    999,  2,  2,  2,  999,
              3,                // layer
    999,  4,  4,  4,  999,
    999,  5,  5,  5,  999},
  //-------------------------------------------
  { 999, -1,  0,  1,  999,
    999, -1,  0,  1,  999,
    999, -1,  0,  1,  999,
              0,                // offset
    999, -1,  0,  1,  999,
    999, -1,  0,  1,  999}
};

// The standard set of patterns the CathodeLCTProcessor uses is defined below.
// For the given pattern, set the unused parts of the pattern to 999.
// Pattern[i][16] contains pt bend value. JM
// bend of 0 is left/straight and bend of 1 is right bht 21 June 2001
// note that the left/right-ness of this is exactly opposite of what one would
// expect naively (at least it was for me). The only way to make sure you've
// got the patterns you want is to use the printPatterns() method to dump
// them. BHT 21 June 2001
const int CSCCathodeLCTProcessor::pattern[CSCConstants::NUM_CLCT_PATTERNS][NUM_PATTERN_STRIPS+1] = {
  { 999, 999, 999, 999, 999,
    999, 999, 999, 999, 999,
    999, 999, 999, 999, 999,
              999,            // dummy (reserved)
    999, 999, 999, 999, 999,
    999, 999, 999, 999, 999, 0},
  //-------------------------------------------------------------
  { 999, 999, 999,   0, 999,
    999, 999, 999,   1, 999,
    999, 999,   2,   2, 999,
                3,            // right bending pattern (large)
    999,   4,   4, 999, 999,
    999,   5, 999, 999, 999, 1},
  //-------------------------------------------------------------
  { 999,   0, 999, 999, 999,
    999,   1, 999, 999, 999,
    999,   2,   2, 999, 999,
                3,            // left bending pattern (large)
    999, 999,   4,   4, 999,
    999, 999, 999,   5, 999, 0},
  //-------------------------------------------------------------
  { 999, 999,   0, 999, 999,
    999, 999,   1, 999, 999,
    999, 999,   2, 999, 999,
                3,            // right bending pattern (medium)
    999,   4, 999, 999, 999,
    999,   5, 999, 999, 999, 1},
  //-------------------------------------------------------------
  { 999, 999,   0, 999, 999,
    999, 999,   1, 999, 999,
    999, 999,   2, 999, 999,
                3,            // left bending pattern (medium)
    999, 999, 999,   4, 999,
    999, 999, 999,   5, 999, 0},
  //-------------------------------------------------------------
  { 999, 999, 999,   0, 999,
    999, 999, 999,   1, 999,
    999, 999,   2,   2, 999,
                3,            // right bending pattern (small)
    999, 999,   4, 999, 999,
    999, 999,   5, 999, 999, 1},
  //-------------------------------------------------------------
  { 999,   0, 999, 999, 999,
    999,   1, 999, 999, 999,
    999,   2,   2, 999, 999,
                3,            // left bending pattern (small)
    999, 999,   4, 999, 999,
    999, 999,   5, 999, 999, 0},
  //-------------------------------------------------------------
  { 999, 999,   0, 999, 999,
    999, 999,   1, 999, 999,
    999, 999,   2, 999, 999,
                3,            // straight through pattern
    999, 999,   4, 999, 999,
    999, 999,   5, 999, 999, 1}
};

// Number of di-strips/half-strips per CFEB.
const int CSCCathodeLCTProcessor::cfeb_strips[2] = { 8, 32};

//----------------
// Constructors --
//----------------

CSCCathodeLCTProcessor::CSCCathodeLCTProcessor(unsigned endcap,
					       unsigned station,
					       unsigned sector,
					       unsigned subsector,
					       unsigned chamber,
					       const edm::ParameterSet& conf) :
		     theEndcap(endcap), theStation(station), theSector(sector),
                     theSubsector(subsector), theTrigChamber(chamber) {
  static bool config_dumped = false;

  // CLCT configuration parameters.
  bx_width     = conf.getParameter<unsigned int>("clctBxWidth");    // def = 6
  drift_delay  = conf.getParameter<unsigned int>("clctDriftDelay"); // def = 2
  hs_thresh    = conf.getParameter<unsigned int>("clctHsThresh");   // def = 2
  ds_thresh    = conf.getParameter<unsigned int>("clctDsThresh");   // def = 2
  nph_pattern  = conf.getParameter<unsigned int>("clctNphPattern"); // def = 4

  // Currently used only in the test-beam mode.
  fifo_tbins   = conf.getParameter<unsigned int>("clctFifoTbins");  // def = 12

  // Not used yet.
  fifo_pretrig = conf.getParameter<unsigned int>("clctFifoPretrig");// def = 7

  // Verbosity level, set to 0 (no print) by default.
  infoV        = conf.getUntrackedParameter<int>("verbosity", 0);

  // Check and print configuration parameters.
  checkConfigParameters();
  if (infoV > 0 && !config_dumped) {
    dumpConfigParams();
    config_dumped = true;
  }

  numStrips = 0;
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    if ((i_layer+1)%2 == 0) stagger[i_layer] = 0;
    else                    stagger[i_layer] = 1;
  }

  // engage in various and sundry tests, but only for a single chamber.
  //if (theStation == 2 && theSector == 1 &&
  //    CSCTriggerNumbering::ringFromTriggerLabels(theStation, theTrigChamber) == 1 && 
  //    CSCTriggerNumbering::chamberFromTriggerLabels(theSector, theSubsector,
  //						    theStation, theTrigChamber) == 1) {
    // test all possible patterns in our uber pattern. 
    // testPatterns();
    // this tests to make sure what goes into an LCT is what comes out
    // testLCTs();
    // print out all the patterns to make sure we've got what we think
    // we've got.
    // printPatterns();
  //  }
}

CSCCathodeLCTProcessor::CSCCathodeLCTProcessor() :
  		     theEndcap(1), theStation(1), theSector(1),
                     theSubsector(1), theTrigChamber(1) {
  // constructor for debugging.
  static bool config_dumped = false;

  // CLCT configuration parameters.
  setDefaultConfigParameters();
  infoV =  2;

  // Check and print configuration parameters.
  checkConfigParameters();
  if (!config_dumped) {
    dumpConfigParams();
    config_dumped = true;
  }

  numStrips = CSCConstants::MAX_NUM_STRIPS;
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    if ((i_layer+1)%2 == 0) stagger[i_layer] = 0;
    else                    stagger[i_layer] = 1;
  }
}

void CSCCathodeLCTProcessor::setDefaultConfigParameters() {
  // Set default values for configuration parameters.
  fifo_tbins   = 12;
  fifo_pretrig =  7;
  bx_width     =  6;
  drift_delay  =  2;
  hs_thresh    =  2;
  ds_thresh    =  2;
  nph_pattern  =  4;
}

// Set configuration parameters obtained via EventSetup mechanism.
void CSCCathodeLCTProcessor::setConfigParameters(const L1CSCTPParameters* conf) {
  static bool config_dumped = false;

  fifo_tbins   = conf->clctFifoTbins();
  fifo_pretrig = conf->clctFifoPretrig();
  bx_width     = conf->clctBxWidth();
  drift_delay  = conf->clctDriftDelay();
  nph_pattern  = conf->clctNphPattern();
  hs_thresh    = conf->clctHsThresh();
  ds_thresh    = conf->clctDsThresh();

  // Check and print configuration parameters.
  checkConfigParameters();
  if (!config_dumped) {
    dumpConfigParams();
    config_dumped = true;
  }
}

void CSCCathodeLCTProcessor::checkConfigParameters() const {
  // Make sure that the parameter values are within the allowed range.

  // Max expected values.
  static const unsigned int max_fifo_tbins   = 1 << 5;
  static const unsigned int max_fifo_pretrig = 1 << 5;
  static const unsigned int max_bx_width     = 1 << 4;
  static const unsigned int max_drift_delay  = 1 << 2;
  static const unsigned int max_hs_thresh    = 1 << 3;
  static const unsigned int max_ds_thresh    = 1 << 3;
  static const unsigned int max_nph_pattern  = 1 << 3;

  // Checks.
  if (fifo_tbins >= max_fifo_tbins) {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "+++ Value of fifo_tbins, " << fifo_tbins
      << ", exceeds max allowed, " << max_fifo_tbins-1 << " +++\n";
  }
  if (fifo_pretrig >= max_fifo_pretrig) {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "+++ Value of fifo_pretrig, " << fifo_pretrig
      << ", exceeds max allowed, " << max_fifo_pretrig-1 << " +++\n";
  }
  if (bx_width >= max_bx_width) {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "+++ Value of bx_width, " << bx_width
      << ", exceeds max allowed, " << max_bx_width-1 << " +++\n";
  }
  if (drift_delay >= max_drift_delay) {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "+++ Value of drift_delay, " << drift_delay
      << ", exceeds max allowed, " << max_drift_delay-1 << " +++\n";
  }
  if (hs_thresh >= max_hs_thresh) {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "+++ Value of hs_thresh, " << hs_thresh
      << ", exceeds max allowed, " << max_hs_thresh-1 << " +++\n";
  }
  if (ds_thresh >= max_ds_thresh) {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "+++ Value of ds_thresh, " << ds_thresh
      << ", exceeds max allowed, " << max_ds_thresh-1 << " +++\n";
  }
  if (nph_pattern >= max_nph_pattern) {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "+++ Value of nph_pattern, " << nph_pattern
      << ", exceeds max allowed, " << max_nph_pattern-1 << " +++\n";
  }
}

void CSCCathodeLCTProcessor::clear() {
  bestCLCT.clear();
  secondCLCT.clear();
}

std::vector<CSCCLCTDigi>
CSCCathodeLCTProcessor::run(const CSCComparatorDigiCollection* compdc) {
  // This is the version of the run() function that is called when running
  // over the entire detector.  It gets the comparator & timing info from the
  // comparator digis and then passes them on to another run() function.

  // clear(); // redundant; called by L1MuCSCMotherboard.

  // Get the number of strips and stagger of layers for the given chamber.
  // Do it only once per chamber.
  if (numStrips == 0) {
    CSCTriggerGeomManager* theGeom = CSCTriggerGeometry::get();
    CSCChamber* theChamber = theGeom->chamber(theEndcap, theStation, theSector,
					      theSubsector, theTrigChamber);
    if (theChamber) {
      numStrips = theChamber->layer(1)->geometry()->numberOfStrips();
      // ME1/a is known to the readout hardware as strips 65-80 of ME1/1.
      // Still need to decide whether we do any special adjustments to
      // reconstruct LCTs in this region (3:1 ganged strips); for now, we
      // simply allow for hits in ME1/a and apply standard reconstruction
      // to them.
      if (theStation == 1 &&
	  CSCTriggerNumbering::ringFromTriggerLabels(theStation,
						     theTrigChamber) == 1) {
	numStrips = 80;
      }

      if (numStrips > CSCConstants::MAX_NUM_STRIPS) {
	throw cms::Exception("CSCCathodeLCTProcessor")
	  << "+++ Number of strips, " << numStrips
	  << ", exceeds max expected, " << CSCConstants::MAX_NUM_STRIPS
	  << " +++" << std::endl;
      }
      // The strips for a given layer may be offset from the adjacent layers.
      // This was done in order to improve resolution.  We need to find the
      // 'staggering' for each layer and make necessary conversions in our
      // arrays.  -JM
      for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
	stagger[i_layer] =
	  (theChamber->layer(i_layer+1)->geometry()->stagger() + 1) / 2;
      }
    }
    else {
      edm::LogWarning("CSCCathodeLCTProcessor")
	<< "+++ CSC chamber (endcap = " << theEndcap
	<< " station = " << theStation << " sector = " << theSector
	<< " subsector = " << theSubsector << " trig. id = " << theTrigChamber
	<< ") is not defined in current geometry!"
	<< " Set numStrips to zero +++" << "\n";
      numStrips = 0;
    }
  }

  // Get comparator digis in this chamber.
  bool noDigis = getDigis(compdc);

  if (!noDigis) {
    int time[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_STRIPS];
    int triad[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_STRIPS];
    int digiNum[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_STRIPS];
    for (int i = 0; i < CSCConstants::NUM_LAYERS; i++){
      for (int j = 0; j < CSCConstants::MAX_NUM_STRIPS; j++) {
	time[i][j]    = -999;
	triad[i][j]   =    0;
	digiNum[i][j] = -999;
      }
    }

    for (int i = 0; i < CSCConstants::NUM_LAYERS; i++) {
      std::vector <CSCComparatorDigi> layerDigiV = digiV[i];
      for (unsigned int j = 0; j < layerDigiV.size(); j++) {
	// Get one digi at a time for the layer.  -Jm
	CSCComparatorDigi thisDigi = layerDigiV[j];

	// Dump raw digi info
	if (infoV > 1) {
	  LogDebug("CSCCathodeLCTProcessor")
	    << "Comparator digi: comparator = " << thisDigi.getComparator()
	    << " strip #" << thisDigi.getStrip()
	    << " time bin = " << thisDigi.getTimeBin();
	}

	// Get comparator: 0/1 for left/right halfstrip for each comparator
	// that fired.
	int thisComparator = thisDigi.getComparator();
	if (thisComparator != 0 && thisComparator != 1) {
	  throw cms::Exception("CSCCathodeLCTProcessor")
	    << "+++ Comparator digi with wrong comparator value: digi #" << j
	    << ", comparator = " << thisComparator << " +++" << std::endl;
	}

	// Get strip number.
	int thisStrip = thisDigi.getStrip() - 1; // count from 0
	if (thisStrip < 0 || thisStrip >= numStrips) {
	  throw cms::Exception("CSCCathodeLCTProcessor")
	    << "+++ Comparator digi with wrong strip number: digi #" << j
	    << ", strip = " << thisStrip
	    << ", max strips = " << numStrips << " +++" << std::endl;
	}
	int diStrip = thisStrip/2; // [0-39]

	// Get Bx of this Digi and check that it is within the bounds
	int thisDigiBx = thisDigi.getTimeBin();

	// Total number of time bins in DAQ readout is given by fifo_tbins,
	// which thus determines the maximum length of time interval.
	if (thisDigiBx >= 0 && thisDigiBx < static_cast<int>(fifo_tbins)) {

	  // If there is more than one hit in the same strip, pick one
	  // which occurred earlier.
	  if (time[i][thisStrip] == -999 || time[i][thisStrip] > thisDigiBx) {
	    digiNum[i][thisStrip] = j;
	    time[i][thisStrip]    = thisDigiBx;
	    triad[i][thisStrip]   = thisComparator;
	    if (infoV > 1) {
	      LogDebug("CSCCathodeLCTProcessor")
		<< "Comp digi: layer " << i+1
		<< " digi #"           << j+1
		<< " strip "           << thisStrip
		<< " halfstrip "       << 2*thisStrip + triad[i][thisStrip] + stagger[i]
		<< " distrip "         << diStrip + 
		((thisStrip%2 == 1 && triad[i][thisStrip] == 1 && stagger[i] == 1) ? 1 : 0)
		<< " time "            <<    time[i][thisStrip]
		<< " comparator "      <<   triad[i][thisStrip]
		<< " stagger "         << stagger[i];
	    }
	  }
	}
	else {
	  edm::LogWarning("CSCCathodeLCTProcessor")
	    << "Unexpected BX time of comparator digi: strip = " << thisStrip
	    << ", layer = " << i << ", bx = " << thisDigiBx << " +++\n";
	}
      }
    }

    // Find number of layers containing digis
    int layersHit = 0;
    for (int i = 0; i < CSCConstants::NUM_LAYERS; i++) {
      for (int j = 0; j < CSCConstants::MAX_NUM_STRIPS; j++) {
	if (time[i][j] >= 0) {layersHit++; break;}
      }
    }
    if (layersHit > 3) run(triad, time, digiNum);
  }

  // Return vector of CLCTs.
  std::vector<CSCCLCTDigi> tmpV = getCLCTs();
  return tmpV;
}

void CSCCathodeLCTProcessor::run(int triad[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_STRIPS],
				 int time[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_STRIPS],
				 int digiNum[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_STRIPS]){
  // This version of the run() function can either be called in a standalone
  // test, being passed the comparator output and time, or called by the 
  // run() function above.  It takes the comparator & time info and stuffs
  // it into both half- and di-strip arrays and uses the findLCTs() method
  // to find vectors of LCT candidates. These candidates are sorted and the
  // best two are returned.
  static int test_iteration = 0;
  int halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS];
  int distrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS];
  int j, i_halfstrip, i_distrip;

  for (int i = 0; i < CSCConstants::NUM_LAYERS; i++) {
    // Use the comparator info to setup the halfstrips and distrips.  -BT
    for (j = 0; j < CSCConstants::NUM_HALF_STRIPS; j++){
      halfstrip[i][j] = -999; //halfstrips
      distrip[i][j]   = -999; //and distrips
    }
    // This loop is only for halfstrips.
    for (j = 0; j < CSCConstants::MAX_NUM_STRIPS; j++) {
      if (time[i][j] >= 0) {
	i_halfstrip = 2*j + triad[i][j] + stagger[i];
	// 2*j    : convert strip to 1/2 strip
	// triad  : comparator output
	// stagger: stagger for this layer
	if (i_halfstrip >= 2*numStrips + 1) {
	  throw cms::Exception("CSCCathodeLCTProcessor")
	    << "+++ Found wrong halfstrip number = " << i_halfstrip << " +++"
	    << std::endl;
	}
	halfstrip[i][i_halfstrip] = time[i][j];
      }
    }
    // This loop is only for distrips.  We have to separate the routines
    // because triad and time arrays can be changed by the distripStagger
    // routine which could mess up the halfstrips.
    for (j = 0; j < CSCConstants::MAX_NUM_STRIPS; j++){
      if (time[i][j] >= 0) {
	i_distrip = j/2;
	if (j%2 == 1 && triad[i][j] == 1 && stagger[i] == 1) {
	  // @@ Needs to be checked.
	  bool stagger_debug = (infoV > 2);
	  distripStagger(triad[i], time[i], digiNum[i], j, stagger_debug);
	}
	// triad[i][j] == 1	: hit on right half-strip.
	// stagger[i] == 1	: half-strips are shifted by 1.
	// if these conditions are met add 1; otherwise add 0.
	// So if there is a hit on the far right half-strip, and the
	// half-strips have been staggered to the right, then the di-strip
	// would actually correspond to the next highest di-strip.  -JM
	if (infoV > 2 && test_iteration == 1) {
	  testDistripStagger();
	  test_iteration++;
	}
	if (i_distrip >= numStrips/2 + 1) {
	  throw cms::Exception("CSCCathodeLCTProcessor")
	    << "+++ Found wrong distrip number = " << i_distrip << " +++"
	    << std::endl;
	}
	distrip[i][i_distrip] = time[i][j];
      }
    }
  }
  
  // Save all hits into vectors if needed.
  // saveAllHits(distrip, halfstrip);

  // Now to do the real work of the CathodeProcessor: make a vector of the
  // possible halfstrip LCT candidates and a vector of the distrip LCT 
  // candidates.  Subtract 1 to account for staggering.
#ifdef TB
  std::vector<CSCCLCTDigi> LCTlist = findLCTs(halfstrip,distrip);
#else
  std::vector<CSCCLCTDigi> halfStripLCTs = findLCTs(halfstrip,1,2*numStrips+1);
  std::vector<CSCCLCTDigi> diStripLCTs   = findLCTs(distrip,  0,numStrips/2+1);
  std::vector<CSCCLCTDigi> LCTlist;

  for (unsigned int i = 0; i < halfStripLCTs.size(); i++) // put all
    LCTlist.push_back(halfStripLCTs[i]); // the candidates into a single vector
  for (unsigned int i = 0; i < diStripLCTs.size(); i++)   // and sort them:
    LCTlist.push_back(diStripLCTs[i]);   // 6/6H, 5/6H, 6/6D, 4/6H, 5/6D, 4/6D
#endif
  // LCT sorting
  if (LCTlist.size() > 1)
    sort(LCTlist.begin(), LCTlist.end(), std::greater<CSCCLCTDigi>());

  if (LCTlist.size() > 0) bestCLCT   = LCTlist[0]; // take the best two 
  if (LCTlist.size() > 1) secondCLCT = LCTlist[1]; // candidates
#ifndef TB
  // Irrelevant for test beam implementation.
  if (bestCLCT == secondCLCT) { // if the second one is the same as the first
    secondCLCT.clear();         // (i.e. found the same track both half and
    if (LCTlist.size() > 2) secondCLCT = LCTlist[2]; // distrip), take the 
                                                     // next one.
  }
#endif

  // Get the list of RecDigis included in each CLCT. Also look for their
  // closest SimHits.
  if (bestCLCT.isValid()) {
    bestCLCT.setTrknmb(1);
    if (infoV > 0) {
      LogDebug("CSCCathodeLCTProcessor")
	<< bestCLCT << " found in endcap " << theEndcap
	<< " station " << theStation << " sector " << theSector
	<< " (" << theSubsector
	<< ") ring " << CSCTriggerNumbering::ringFromTriggerLabels(theStation,
						        theTrigChamber)
	<< " chamber " 
	<< CSCTriggerNumbering::chamberFromTriggerLabels(theSector,
                              theSubsector, theStation, theTrigChamber)
	<< " (trig id. " << theTrigChamber << ")" << "\n";
    }
  }
  if (secondCLCT.isValid()) {
    secondCLCT.setTrknmb(2);
    if (infoV > 0) {
      LogDebug("CSCCathodeLCTProcessor")
	<< secondCLCT << " found in endcap " << theEndcap
	<< " station " << theStation << " sector " << theSector
	<< " (" << theSubsector
	<< ") ring " << CSCTriggerNumbering::ringFromTriggerLabels(theStation,
						        theTrigChamber)
	<< " chamber "
	<< CSCTriggerNumbering::chamberFromTriggerLabels(theSector,
                              theSubsector, theStation, theTrigChamber)
	<< " (trig id. " << theTrigChamber << ")" << "\n";
    }
  }
  // Now that we have our 2 best CLCTs, they get correlated with the 2 best
  // ALCTs and then get sent to the MotherBoard.  -JM
}

bool CSCCathodeLCTProcessor::getDigis(const CSCComparatorDigiCollection* compdc) {
  bool noDigis = true;
  int  theRing    = CSCTriggerNumbering::ringFromTriggerLabels(theStation,
							       theTrigChamber);
  int  theChamber = CSCTriggerNumbering::chamberFromTriggerLabels(theSector,
                                     theSubsector, theStation, theTrigChamber);

  // Loop over layers and save comparator digis on each one into digiV[layer].
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    CSCDetId detid(theEndcap, theStation, theRing, theChamber, i_layer+1);

    const CSCComparatorDigiCollection::Range rcompd = compdc->get(detid);

    // Skip if no comparator digis in this layer.
    if (rcompd.second == rcompd.first) continue;

    // If this is the first layer with digis in this chamber, clear digiV
    // array and set the empty flag to false.
    if (noDigis) {
      for (int lay = 0; lay < CSCConstants::NUM_LAYERS; lay++) {
	digiV[lay].clear();
      }
      noDigis = false;
    }

    if (infoV > 1) LogDebug("CSCCathodeLCTProcessor")
      << "found " << rcompd.second - rcompd.first
      << " comparator digi(s) in layer " << i_layer << "; " << theEndcap << " "
      << theStation << " " << theRing << " " << theSector << "("
      << theSubsector << ") " << theChamber << "(" << theTrigChamber << ")";

    for (CSCComparatorDigiCollection::const_iterator digiIt = rcompd.first;
	 digiIt != rcompd.second; ++digiIt) {
      digiV[i_layer].push_back(*digiIt);
    }
  }

  return noDigis;
}

void CSCCathodeLCTProcessor::distripStagger(int stag_triad[CSCConstants::MAX_NUM_STRIPS],
				   int stag_time[CSCConstants::MAX_NUM_STRIPS],
				   int stag_digi[CSCConstants::MAX_NUM_STRIPS],
				   int i_strip, bool debug) {
  // Author: Jason Mumford (mumford@physics.ucla.edu)
  // This routine takes care of the stagger situation where there is a hit
  // on the right half-strip of a di-strip.  If there is a stagger, then
  // we must associate that distrip with the next distrip. The situation
  // gets more complicated if the next distrip also has a hit on its right
  // half-strip.  One could imagine a whole chain of these in which case
  // we need to go into this routine recursively.  The formula is that
  // while this condition is satisfied, we enquire the next distrip,
  // until we have a hit on any other halfstrip (or triad!=3).  Then we
  // must compare the 2 different bx times and take the smallest one.
  // Afterwards, we must cycle out of the routine assigning the bx times
  // to the one strip over.

  if (i_strip >= CSCConstants::MAX_NUM_STRIPS) {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "+++ Found wrong strip number = " << i_strip << " +++" << std::endl;
  }

  if (debug)
    LogDebug("CSCCathodeLCTProcessor")
      << " Enter distripStagger: i_strip = " << i_strip
      << " stag_triad[i_strip] = "   << stag_triad[i_strip]
      << " stag_time[i_strip] =  "   << stag_time[i_strip]
      << " stag_triad[i_strip+2] = " << stag_triad[i_strip+2]
      << " stag_time[i_strip+2] = "  << stag_time[i_strip+2];

  // So if the next distrip has a stagger hit, go into the routine again
  // for the next distrip.
  if (i_strip+2 < CSCConstants::MAX_NUM_STRIPS && stag_triad[i_strip+2] == 1)
    distripStagger(stag_triad, stag_time, stag_digi, i_strip+2);

  // When we have reached a distrip that does not have a staggered hit,
  // if it has a hit, we compare the bx times of the
  // staggered distrip with the non-staggered distrip and we take the
  // smallest of the two and assign it to the shifted distrip time.
  if (stag_time[i_strip+2] >= 0) {
    if (stag_time[i_strip] < stag_time[i_strip+2]) {
      stag_time[i_strip+2] = stag_time[i_strip];
      stag_digi[i_strip+2] = stag_digi[i_strip];
    }
  }
  // If the next distrip did not have a hit, then we merely assign the
  // shifted time to the time associated with the staggered distrip.
  else {
    stag_time[i_strip+2] = stag_time[i_strip];
    stag_digi[i_strip+2] = stag_digi[i_strip];
  }

  // Then to prevent previous staggers from being overwritten, we assign
  // the unshifted time to -999, and then mark the triads that were shifted
  // so that we don't go into the routine ever again (such as when making
  // the next loop over strips).
  stag_time[i_strip]  = -999;
  stag_triad[i_strip] =    4;
  stag_digi[i_strip]  = -999;

  if (debug)
    LogDebug("CSCCathodeLCTProcessor")
      << " Exit  distripStagger: i_strip = " << i_strip
      << " stag_triad[i_strip] = "   << stag_triad[i_strip]
      << " stag_time[i_strip] = "    << stag_time[i_strip]
      << " stag_triad[i_strip+2] = " << stag_triad[i_strip+2]
      << " stag_time[i_strip+2] = "  << stag_time[i_strip+2];
}

// --------- Separate from test beam version of these fcns below --------------

std::vector<CSCCLCTDigi> CSCCathodeLCTProcessor::findLCTs(const int strip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS], int stripType, int nStrips)
{
  int j;
  int best_strip = 0;
  int first_bx = 999;
  const int max_lct_num = 2;
  const int adjacent_strips = 2;
  // Distrip, halfstrip pattern threshold.
  const int ptrn_thrsh[2] = {nph_pattern, nph_pattern};
  int highest_quality = 0;

  int keystrip_data[CSCConstants::NUM_HALF_STRIPS][7];
  int final_lcts[max_lct_num];

  std::vector <CSCCLCTDigi> lctList;

  if (infoV > 1) dumpDigis(strip, stripType, nStrips);

  // Send data to a pretrigger so that we don't excessively look at data
  // that won't give an LCT. If there is a pretrigger, then get all quality
  // and bend for all keystrips.
  if (preTrigger(strip, stripType, nStrips, first_bx)){

    getKeyStripData(strip, keystrip_data, nStrips, first_bx, best_strip, stripType);

    /* Set all final_lcts to impossible key_strip numbers */
    for (j = 0; j < max_lct_num; j++)
      final_lcts[j] = -999;

    // Now take the keystrip with the best quality, and do a search over the
    // rest of the strips for the next highest quality.  Do the search over a 
    // range excluding a certain number of keystrips adjacent to the original
    // best key_strip.
    final_lcts[0] = best_strip;

    for (int key_strip = 0; key_strip < (nStrips-stripType); key_strip++){
      // If indexed strip does not fall within excluded range, then continue
      if (abs(best_strip - key_strip) > adjacent_strips){
	// Match with highest quality
	if (keystrip_data[key_strip][CLCT_QUALITY] > highest_quality){
	  highest_quality = keystrip_data[key_strip][CLCT_QUALITY];
	  final_lcts[1] = key_strip;
	}
      }
    }

    for (j = 0; j < max_lct_num; j++){
      // Only report LCTs if the number of layers hit is greater than or
      // equal to the (variable) valid pattern threshold ptrn_thrsh.
      int keystrip = final_lcts[j];
      if (keystrip >= 0 &&
	  keystrip_data[keystrip][CLCT_QUALITY] >= ptrn_thrsh[stripType]) {
     	// assign the stripType here. 1 = halfstrip, 0 = distrip.
     	keystrip_data[keystrip][CLCT_STRIP_TYPE] = stripType;
	// Now make the LCT words for the 2 highest, and store them in a list
	int theHalfStrip = (keystrip_data[keystrip][CLCT_STRIP_TYPE] ?
			    keystrip_data[keystrip][CLCT_STRIP] :
			    4*keystrip_data[keystrip][CLCT_STRIP]);
	keystrip_data[keystrip][CLCT_CFEB] = theHalfStrip/32;
	int halfstrip_in_cfeb =
	  theHalfStrip - 32*keystrip_data[keystrip][CLCT_CFEB];

	CSCCLCTDigi thisLCT(1, keystrip_data[keystrip][CLCT_QUALITY],
			    keystrip_data[keystrip][CLCT_PATTERN],
			    keystrip_data[keystrip][CLCT_STRIP_TYPE],
			    keystrip_data[keystrip][CLCT_BEND],
			    halfstrip_in_cfeb,
			    keystrip_data[keystrip][CLCT_CFEB],
			    keystrip_data[keystrip][CLCT_BX]);
	if (infoV > 2) {
	  char stripType =
	    (keystrip_data[keystrip][CLCT_STRIP_TYPE] == 0) ? 'D' : 'H';
	  char bend =
	    (keystrip_data[keystrip][CLCT_BEND] == 0) ? 'L' : 'R';
	  LogDebug("CSCCathodeLCTProcessor")
	    << " Raw Find: "
	    << "Key Strip: "  << std::setw(3)
	    << keystrip_data[keystrip][CLCT_STRIP]
	    << " Pattern: "   << std::setw(2)
	    << keystrip_data[keystrip][CLCT_PATTERN]
	    << " Bend: "      << std::setw(1) << bend
	    << " Quality: "   << std::setw(1)
	    << keystrip_data[keystrip][CLCT_QUALITY]
	    << " stripType: " << std::setw(1) << stripType
	    << " BX: "        << std::setw(1)
	    << keystrip_data[keystrip][CLCT_BX];
	}
	lctList.push_back(thisLCT);
      }
    }
  }

  return lctList;
}

bool CSCCathodeLCTProcessor::preTrigger(const int strip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS],
					const int stripType, const int nStrips,
					int& first_bx)
{
  unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS];
  int i_layer, i_strip, this_layer, this_strip;
  int hits, layers_hit;
  bool hit_layer[CSCConstants::NUM_LAYERS];

  const int pre_trigger_layer_min = (stripType == 1) ? hs_thresh : ds_thresh;

  // Clear pulse array.  This array will be used as a bit representation of
  // hit times.  For example: if strip[1][2] has a value of 3, then 1 shifted
  // left 3 will be bit pattern of pulse[1][2].  This would make the pattern
  // look like 0000000000001000.  Later we add on additional bits to signify
  // the duration of a signal (bx_width).  So for the same pulse[1][2] with
  // a bx_width of 3 would look like 0000000000111000.
  for (i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++)
    for (i_strip = 0; i_strip < nStrips; i_strip++)
      pulse[i_layer][i_strip] = 0;

  // The triad decoders output a pulse bx_width wide (150 ns by default)
  // for each half-strip that the CFEB comparators saw a hit on.
  for (i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++){
    // Loop over all strips (depending on half strip or di-strip count)
    for (i_strip = 0; i_strip < nStrips; i_strip++){
      // If no hit time, no need to add a pulse width.
      if (strip[i_layer][i_strip] >= 0){
	// Add on bx_width for a pulse time envelope.  See above...
	for (unsigned int bx = strip[i_layer][i_strip];
	     bx < strip[i_layer][i_strip] + bx_width; bx++) {
	  pulse[i_layer][i_strip] = pulse[i_layer][i_strip] | (1 << bx);
	}
      }
    }
  }

  // Now do a loop over different bunch-crossing times.
  for (unsigned int bx_time = 0; bx_time < fifo_tbins; bx_time++) {
    // For any given bunch-crossing, start at the lowest keystrip and look for
    // the number of separate layers in the pattern for that keystrip that have
    // pulses at that bunch-crossing time.  Do the same for the next keystrip, 
    // etc.  Then do the entire process again for the next bunch-crossing, etc
    // until you find a pre-trigger.
    for (int key_strip = 0; key_strip < nStrips; key_strip++){
      // Clear variables
      hits = 0;
      layers_hit = 0;
      for (i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++)
	hit_layer[i_layer] = false;
      // Loop over pattern strips and look for hits.
      for (i_strip = 0; i_strip < NUM_PATTERN_STRIPS; i_strip++){
	this_layer = pre_hit_pattern[0][i_strip];
	this_strip = pre_hit_pattern[1][i_strip]+key_strip;
	if (this_strip >= 0 && this_strip < nStrips) {
	  // Perform bit operation to see if pulse is 1 at a certain bx_time.
	  if (((pulse[this_layer][this_strip] >> bx_time) & 1) == 1) {
	    hits++;
	    // Store number of layers hit.
	    if (hit_layer[this_layer] == false) {
	      hit_layer[this_layer] = true;
	      layers_hit++;

	      // Look if number of layers hit is greater or equal than some
	      // pre-defined threshold.
	      if (layers_hit >= pre_trigger_layer_min) {
		first_bx = bx_time;
		return true;
	      }
	    }
	  }
	}
      }
    }
  }
  // If the pretrigger was never satisfied, then return false.
  return false;
}

void CSCCathodeLCTProcessor::getKeyStripData(const int strip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS],
        int keystrip_data[CSCConstants::NUM_HALF_STRIPS][7],
        int nStrips, int first_bx, int& best_strip, int stripType) {
  int lct_pattern[NUM_PATTERN_STRIPS];
  int key_strip, this_layer, this_strip;
  int quality, best_quality;
  int bend = 0;
  int highest_quality = 0;
  bool nullPattern;

  for (key_strip = 0; key_strip < nStrips; key_strip++)
    for (int i = 0; i < 7; i++)
      keystrip_data[key_strip][i] = 0;

  // Now we need to look at all the keystrips and take the best pattern
  // for each.  There are multiple patterns available for each keystrip.

  for (key_strip = 0; key_strip < (nStrips-stripType); key_strip++){
    nullPattern = true;
    for (int pattern_strip = 0; pattern_strip < NUM_PATTERN_STRIPS; pattern_strip++){
      this_layer = pre_hit_pattern[0][pattern_strip];
      this_strip = pre_hit_pattern[1][pattern_strip] + key_strip;
      // This conditional statement prevents us from looking at strips
      // that don't exist along the chamber boundaries.
      if ((this_strip >= 0 && this_strip < nStrips) &&
	  strip[this_layer][this_strip] >= 0){
	if (nullPattern) nullPattern = false;
	lct_pattern[pattern_strip] = strip[this_layer][this_strip];
      }
      else
	lct_pattern[pattern_strip] = -999;
      }
    // do the rest only if there is at least one DIGI in the pattern for
    // this keystrip
    if (nullPattern) continue;

    // Initialize best_quality to zero so that we can look for best pattern
    // within a keystrip.
    best_quality = 0;

    // Loop over all possible patterns.
    // Loop in reverse order, in order to give priority to a straighter
    // pattern (larger pattern_num) in case of equal qualities.
    // Exclude pattern 0 since it is not defined.
    for (int pattern_num = CSCConstants::NUM_CLCT_PATTERNS-1; pattern_num > 0; pattern_num--) {
      // Get the pattern quality from lct_pattern.
      // TMB latches LCTs drift_delay clocks after pretrigger.
      int latch_bx = first_bx + drift_delay;
      getPattern(pattern_num, lct_pattern, latch_bx, quality, bend);
      if (infoV > 1) {
	LogDebug("CSCCathodeLCTProcessor")
	  << "Key_strip " << key_strip << " quality of pattern_num "
	  << pattern_num << ": " << quality;
      }
      if (quality > best_quality){
	// Store the best pattern, quality, etc., for each key_strip.
	keystrip_data[key_strip][CLCT_PATTERN] = pattern_num;
	keystrip_data[key_strip][CLCT_BEND]    = bend;
	keystrip_data[key_strip][CLCT_STRIP]   = key_strip;
	keystrip_data[key_strip][CLCT_BX]      = first_bx;
	// keystrip_data[key_strip][CLCT_STRIP_TYPE] = stripType; //assign the stripType elsewhere
	keystrip_data[key_strip][CLCT_QUALITY] = quality;
	if (quality > highest_quality){
	  // Keep track of which strip had the highest quality.
	  // highest_quality refers to the overall highest quality for all
	  // key strips. This is different than best_quality which refers
	  // to the best quality in a keystrip from different patterns.
	  best_strip = key_strip;
	  highest_quality = quality;
	}
	best_quality = quality;
      }
    }
  }
}

void CSCCathodeLCTProcessor::getPattern(int pattern_num,
       int strip_value[NUM_PATTERN_STRIPS], int bx_time,
       int& quality, int& bend){
  // This function takes strip values and bx_time to find out which hits fall
  // within a certain pattern.  Quality, and bend are then calculated based on
  // which strip pattern and how many layers were hit within the pattern.
  int layers_hit = 0;
  bool hit_layer[CSCConstants::NUM_LAYERS];

  // Clear hit_layer array to keep track of number of layers hit.
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++)
    hit_layer[i_layer] = false;

  // Loop over all designated patterns.
  for (int strip_num = 0; strip_num < NUM_PATTERN_STRIPS; strip_num++){
    if (hitIsGood(strip_value[strip_num], bx_time)){
      for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++){
	// Loop over layer and see if corresponding strip is on same layer
	// If so then increment number of hits.
	if (i_layer == pattern[pattern_num][strip_num]){
	  // If layer has no hits, then increment number of layers hit.
	  if (hit_layer[i_layer] == false){
	    layers_hit++;
	    hit_layer[i_layer] = true;
	  }
	}
      }
    }
  }
  // Get bend value from pattern.
  bend = pattern[pattern_num][NUM_PATTERN_STRIPS];
  quality = layers_hit;
}

bool CSCCathodeLCTProcessor::hitIsGood(int hitTime, int BX) {
  // Find out if hit time is good.  Hit should have occurred no more than
  // bx_width clocks before the latching time.
  int dt = BX - hitTime;
  if (dt >= 0 && dt <= static_cast<int>(bx_width)) {return true;}
  else {return false;}
}


// ---------------- Test beam version of fcns ---------------------------------

std::vector <CSCCLCTDigi> CSCCathodeLCTProcessor::findLCTs(const int halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS], const int distrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS]) {
  std::vector <CSCCLCTDigi> lctList;
  int _bx[2] = {999, 999};
  int first_bx = 999;

  if (infoV > 1) {
    dumpDigis(halfstrip, 1, CSCConstants::NUM_HALF_STRIPS);
    dumpDigis(distrip,   0, CSCConstants::NUM_DI_STRIPS);
  }

  // Test beam version of TMB pretrigger and LCT sorting
  int h_keyStrip[MAX_CFEBS];       // one key per CFEB
  unsigned int h_nhits[MAX_CFEBS]; // number of hits in envelope for each key
  int d_keyStrip[MAX_CFEBS];       // one key per CFEB
  unsigned int d_nhits[MAX_CFEBS]; // number of hits in envelope for each key
  int keystrip_data[2][7];    // 2 possible LCTs per CSC x 7 LCT quantities
  unsigned int h_pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS]; // simulate digital one-shot
  unsigned int d_pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS]; // simulate digital one-shot
  bool pre_trig[2] = {false, false};

  // All half-strip and di-strip pattern envelopes are evaluated
  // simultaneously, on every clock cycle.
  pre_trig[0] =
    preTrigger(halfstrip, h_pulse, 1, CSCConstants::NUM_HALF_STRIPS, _bx[0]);
  pre_trig[1] =
    preTrigger(  distrip, d_pulse, 0,   CSCConstants::NUM_DI_STRIPS, _bx[1]);

  // If any of 200 half-strip and di-strip envelopes has enough layers hit in
  // it, TMB will pre-trigger.
  if (pre_trig[0] || pre_trig[1]) {
    first_bx = (_bx[0] < _bx[1]) ? _bx[0] : _bx[1];
    if (infoV > 1) {
      LogDebug("CSCCathodeLCTProcessor")
	<< "half bx " << _bx[0] << " di bx " << _bx[1] << " first " << first_bx
	<< "\n ..... waiting drift delay ..... ";
    }

    // FOR MTCC RUNS ONLY.
#ifdef MTCC
    // Empirically-found trick allowing to dramatically improve agreement
    // with MTCC-II data.  Needs to be better understood.
    // The trick is to ignore hits in a few first time bins when latching
    // hits for priority encode envelopes.  For MTCC-II, we need to ignore
    // hits in time bins 0-3 inclusively.
    int max_bx = 4;
    for (int ilayer = 0; ilayer < CSCConstants::NUM_LAYERS; ilayer++) {
      for (int istrip = 0; istrip < CSCConstants::NUM_HALF_STRIPS; istrip++) {
	for (int bx = 0; bx < max_bx; bx++) {
	  if (((h_pulse[ilayer][istrip] >> bx) & 1) == 1) {
	    h_pulse[ilayer][istrip] = 0;
	  }
	}
      }
      for (int istrip = 0; istrip < CSCConstants::NUM_DI_STRIPS; istrip++) {
	for (int bx = 0; bx < max_bx; bx++) {
	  if (((d_pulse[ilayer][istrip] >> bx) & 1) == 1) {
	    d_pulse[ilayer][istrip] = 0;
	  }
	}
      }
    }
#endif

    // TMB latches LCTs drift_delay clocks after pretrigger.
    int latch_bx = first_bx + drift_delay;
    latchLCTs(h_pulse, h_keyStrip, h_nhits, 1, CSCConstants::NUM_HALF_STRIPS,
	      latch_bx);
    latchLCTs(d_pulse, d_keyStrip, d_nhits, 0,   CSCConstants::NUM_DI_STRIPS,
	      latch_bx);

    if (infoV > 1) {
      LogDebug("CSCCathodeLCTProcessor")
	<< "...............................\n"
	<< "Final halfstrip hits and keys (after drift delay) ...";
      for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
	LogDebug("CSCCathodeLCTProcessor")
	  << "cfeb " << icfeb << " key: " << h_keyStrip[icfeb]
	  << " hits " << h_nhits[icfeb];
      }
      LogDebug("CSCCathodeLCTProcessor")
	<< "Final distrip hits and keys (after drift delay) ...";
      for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
	LogDebug("CSCCathodeLCTProcessor")
	  << "cfeb " << icfeb << " key: " << d_keyStrip[icfeb]
	  << " hits " << d_nhits[icfeb];
      }
    }
    priorityEncode(h_keyStrip, h_nhits, d_keyStrip, d_nhits, keystrip_data);
    getKeyStripData(h_pulse, d_pulse, keystrip_data, first_bx);

    for (int ilct = 0; ilct < 2; ilct++) {
      if (infoV > 1) LogDebug("CSCCathodeLCTProcessor")
	<< "found lcts: ilct " << ilct
	<< "  key strip " << keystrip_data[ilct][CLCT_STRIP];
      if (keystrip_data[ilct][CLCT_STRIP] != -1) {
	int halfstrip_in_cfeb = 0;
	if (keystrip_data[ilct][CLCT_STRIP_TYPE] == 0)
	  halfstrip_in_cfeb = 4*keystrip_data[ilct][CLCT_STRIP] -
                             32*keystrip_data[ilct][CLCT_CFEB];
	else
	  halfstrip_in_cfeb = keystrip_data[ilct][CLCT_STRIP] -
	                     32*keystrip_data[ilct][CLCT_CFEB];

	CSCCLCTDigi thisLCT(1, keystrip_data[ilct][CLCT_QUALITY],
			    keystrip_data[ilct][CLCT_PATTERN],
			    keystrip_data[ilct][CLCT_STRIP_TYPE],
			    keystrip_data[ilct][CLCT_BEND],
			    halfstrip_in_cfeb,
			    keystrip_data[ilct][CLCT_CFEB],
			    keystrip_data[ilct][CLCT_BX]);
	lctList.push_back(thisLCT);
      }
    }
  }

  return lctList;

} //findLCTs -- test beam version


bool CSCCathodeLCTProcessor::preTrigger(const int strip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS],
	   unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS],
	   const int stripType, const int nStrips, int& first_bx) {

  if (infoV > 1)
    LogDebug("CSCCathodeLCTProcessor")
      << "....................PreTrigger...........................";

  // Clear pulse array.  This array will be used as a bit representation of
  // hit times.  For example: if strip[1][2] has a value of 3, then 1 shifted
  // left 3 will be bit pattern of pulse[1][2].  This would make the pattern
  // look like 0000000000001000.  Then add on additional bits to signify
  // the duration of a signal (bx_width) to simulate the TMB's drift delay.
  // So for the same pulse[1][2] with a bx_width of 3 would look like 
  // 0000000000111000. This is similating the digital one-shot in the TMB.
  for (int ilayer = 0; ilayer < CSCConstants::NUM_LAYERS; ilayer++)
    for (int istrip = 0; istrip < nStrips; istrip++)
      pulse[ilayer][istrip] = 0;

  // note: strip[][] is actually the bx TIME of the hit
  for (int istrip = 0; istrip < nStrips; istrip++) { // loop over all (di/half)strips
    for (int ilayer = 0; ilayer < CSCConstants::NUM_LAYERS; ilayer++) {  // loop over layers
      // if there is a hit, simulate digital one shot persistance starting
      // in the bx of the initial hit.  Fill this into pulse[][].
      if (strip[ilayer][istrip] >= 0) {
	for (unsigned int bx = strip[ilayer][istrip];
	     bx < strip[ilayer][istrip] + bx_width; bx++)
	  pulse[ilayer][istrip] = pulse[ilayer][istrip] | (1 << bx);
      }
    }
  }

  // Now do a loop over bx times to see (if/when) track goes over threshold
  for (unsigned int bx_time = 0; bx_time < fifo_tbins; bx_time++) {
    // For any given bunch-crossing, start at the lowest keystrip and look for
    // the number of separate layers in the pattern for that keystrip that have
    // pulses at that bunch-crossing time.  Do the same for the next keystrip, 
    // etc.  Then do the entire process again for the next bunch-crossing, etc
    // until you find a pre-trigger.
    if (preTrigLookUp(pulse, stripType, nStrips, bx_time)) {
      first_bx = bx_time; // bx at time of pretrigger
      return true;
    }
  } // end loop over bx times

  if (infoV > 1) LogDebug("CSCCathodeLCTProcessor")
    << "no pretrigger for strip type " << stripType << ", returning \n";
  return false;
} // preTrigger -- test beam version


bool CSCCathodeLCTProcessor::preTrigLookUp(
	   const unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS],
	   const int stripType, const int nStrips,
	   const unsigned int bx_time) {

  bool hit_layer[CSCConstants::NUM_LAYERS];
  int key_strip, this_layer, this_strip, layers_hit;

  // Layers hit threshold for pretrigger
  const int pre_trigger_layer_min = (stripType == 1) ? hs_thresh : ds_thresh;

  if (stripType != 0 && stripType != 1) {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "+++ preTrigLookUp: stripType = " << stripType
      << " does not correspond to half-strip/di-strip patterns! +++\n";
  }

  for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) { // loop over cfebs
    // Loop over (di-/half-)strips in CFEB.
    for (int istrip = 0; istrip < cfeb_strips[stripType]; istrip++) {
      // Calculate candidate key.
      key_strip = icfeb*cfeb_strips[stripType] + istrip;
      layers_hit = 0;
      for (int ilayer = 0; ilayer < CSCConstants::NUM_LAYERS; ilayer++)
	hit_layer[ilayer] = false;

      // Loop over strips in pretrigger pattern mask and look for hits.
      for (int pstrip = 0; pstrip < NUM_PATTERN_STRIPS; pstrip++) {
	this_layer = pre_hit_pattern[0][pstrip];
	this_strip = pre_hit_pattern[1][pstrip]+key_strip;

	if (this_strip >= 0 && this_strip < nStrips) {
	  // Determine if "one shot" is high at this bx_time
	  if (((pulse[this_layer][this_strip] >> bx_time) & 1) == 1) {
	    if (hit_layer[this_layer] == false) {
	      hit_layer[this_layer] = true;
	      layers_hit++;                  // determines number of layers hit
	      if (layers_hit >= pre_trigger_layer_min) {
		if (infoV > 1) {
		  LogDebug("CSCCathodeLCTProcessor")
		    << "pretrigger at bx: " << bx_time
		    << ", cfeb " << icfeb << ", returning";
		}
		return true;
	      }
	    }
	  }
	}
      } // end loop over strips in pretrigger pattern
    } // end loop over candidate key strips in cfeb
  } // end loop over cfebs, if pretrigger is found, stop looking and return

  return false;

} // preTrigLookUp -- test beam version


void CSCCathodeLCTProcessor::latchLCTs(
	   const unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS],
	   int keyStrip[MAX_CFEBS], unsigned int nhits[MAX_CFEBS],
	   const int stripType, const int nStrips, const int bx_time) {

  bool hit_layer[CSCConstants::NUM_LAYERS];
  int key_strip, this_layer, this_strip;
  int layers_hit, prev_hits;

  if (stripType != 0 && stripType != 1) {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "+++ latchLCTs: stripType = " << stripType
      << " does not correspond to half-strip/di-strip patterns! +++\n";
  }

  for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) { // loop over CFEBs
    prev_hits = 0;
    keyStrip[icfeb] = -1;
    nhits[icfeb]    =  0;
    // Loop over (di-/half-)strips in CFEB.
    for (int istrip = 0; istrip < cfeb_strips[stripType]; istrip++) {
      // Calculate candidate key.
      key_strip = icfeb*cfeb_strips[stripType] + istrip;
      layers_hit = 0;
      for (int ilayer = 0; ilayer < CSCConstants::NUM_LAYERS; ilayer++)
	hit_layer[ilayer] = false;

      // Loop over strips in pretrigger pattern mask and look for hits.
      for (int pstrip = 0; pstrip < NUM_PATTERN_STRIPS; pstrip++) {
	this_layer = pre_hit_pattern[0][pstrip];
	this_strip = pre_hit_pattern[1][pstrip]+key_strip;

	if (this_strip >= 0 && this_strip < nStrips) {
	  // Determine if "one shot" is high at this bx_time
	  if (((pulse[this_layer][this_strip] >> bx_time) & 1) == 1) {
	    if (hit_layer[this_layer] == false) {
	      hit_layer[this_layer] = true;
	      layers_hit++;                  // number of layers hit
	    }
	  }
	}
      } // end loop over strips in pretrigger pattern
      if (infoV > 1) {
	if (layers_hit > 0) LogDebug("CSCCathodeLCTProcessor")
	  << "cfeb: " << icfeb << "  key_strip: " << key_strip
	  << "  nhits: " << layers_hit;
      }
      // If two or more keys have an equal number of hits, the lower number
      // key is taken.  Hence, replace the previous key only if this key has
      // more hits.
      if (layers_hit > prev_hits) {
	prev_hits = layers_hit;
	keyStrip[icfeb] = key_strip;  // key with highest hits is LCT key strip
	nhits[icfeb] = layers_hit;    // corresponding hits in envelope
      }
    }  // end loop over candidate key strips in cfeb
  }  // end loop over cfebs
} // latchLCTs -- test beam version


void CSCCathodeLCTProcessor::priorityEncode(
        const int h_keyStrip[MAX_CFEBS], const unsigned int h_nhits[MAX_CFEBS],
	const int d_keyStrip[MAX_CFEBS], const unsigned int d_nhits[MAX_CFEBS],
	int keystrip_data[2][7]) {

  int ihits[2]; // hold hits for sorting
  int cfebs[2]; // holds CFEB numbers corresponding to highest hits
  const int nlcts = 2;
  int key_strip[MAX_CFEBS], key_phits[MAX_CFEBS], strip_type[MAX_CFEBS];

  // initialize arrays
  for (int ilct = 0; ilct < nlcts; ilct++) {
    for (int j = 0; j < 7; j++) keystrip_data[ilct][j] = -1;
    ihits[ilct] = 0;
    cfebs[ilct] = -1;
  }
  for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
    key_strip[icfeb]  = -1;
    key_phits[icfeb]  = -1;
    strip_type[icfeb] = -1;
  }

  if (infoV > 1) {
    LogDebug("CSCCathodeLCTProcessor")
      << ".....................PriorityEncode.......................";
    std::ostringstream strstrm;
    strstrm << "hkeys:";
    for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
      strstrm << std::setw(4) << h_keyStrip[icfeb];
    }
    strstrm << "\ndkeys:";
    for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
      strstrm << std::setw(4) << d_keyStrip[icfeb];
    }
    LogDebug("CSCCathodeLCTProcessor") << strstrm.str();
  }

  // Loop over CFEBs and determine better of half- or di- strip pattern.
  // If select halfstrip, promote it by adding an extra bit to its hits.
  for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
    if (h_keyStrip[icfeb] != -1 && d_keyStrip[icfeb] != -1) {
      if (h_nhits[icfeb] >= hs_thresh) {
	key_strip[icfeb] = h_keyStrip[icfeb];
	key_phits[icfeb] = h_nhits[icfeb] + 8; // halfstrip promotion
	strip_type[icfeb]= 1;
      }
      // For di-strip envelope there is no requirement that the number of
      // layers hit is >= ds_thresh!!!
      // else if (d_nhits[icfeb] >= ds_thresh) {
      else {
	key_strip[icfeb] = d_keyStrip[icfeb];
	key_phits[icfeb] = d_nhits[icfeb];
	strip_type[icfeb]= 0;
      }
    }
    else if (h_keyStrip[icfeb] != -1) {
      if (h_nhits[icfeb] >= hs_thresh) {
	key_strip[icfeb] = h_keyStrip[icfeb];
	key_phits[icfeb] = h_nhits[icfeb] + 8; // halfstrip promotion
	strip_type[icfeb]= 1;
      }
    }
    else if (d_keyStrip[icfeb] != -1) {
      // if (d_nhits[icfeb] >= ds_thresh) {
	key_strip[icfeb] = d_keyStrip[icfeb];
	key_phits[icfeb] = d_nhits[icfeb];
	strip_type[icfeb]= 0;
      // }
    }
    if (infoV > 1 && strip_type[icfeb] != -1) {
      if (strip_type[icfeb] == 0)
	LogDebug("CSCCathodeLCTProcessor")
	  << "  taking distrip pattern on cfeb " << icfeb;
      else if (strip_type[icfeb] == 1)
	LogDebug("CSCCathodeLCTProcessor")
	  << "  taking halfstrip pattern on cfeb " << icfeb;
      LogDebug("CSCCathodeLCTProcessor")
	<< "     cfeb " << icfeb << " key " << key_strip[icfeb]
	<< " hits " << key_phits[icfeb] << " type " << strip_type[icfeb];
    }
  }

  // Remove duplicate LCTs at boundaries -- it is possilbe to have key[0]
  // be the higher of the two key strips, take this into account, but
  // preserve rank of lcts.
  int key[MAX_CFEBS];
  int loedge, hiedge;

  if (infoV > 1) LogDebug("CSCCathodeLCTProcessor")
    << "...... Remove Duplicates ......";
  for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
    if(strip_type[icfeb] == 0) key[icfeb] = key_strip[icfeb]*4;
    else                       key[icfeb] = key_strip[icfeb];
  }
  for (int icfeb = 0; icfeb < MAX_CFEBS-1; icfeb++) {
    if (key[icfeb] >= 0 && key[icfeb+1] >= 0) {
      loedge = cfeb_strips[1]*(icfeb*8+7)/8;
      hiedge = cfeb_strips[1]*(icfeb*8+9)/8 - 1;
      if (infoV > 1) {
	LogDebug("CSCCathodeLCTProcessor")
	  << "  key 1: " << key[icfeb] << "  key 2: " << key[icfeb+1]
	  << "  low edge:  " << loedge << "  high edge: " << hiedge;
      }
      if (key[icfeb] >= loedge && key[icfeb+1] <= hiedge) {
	if (infoV > 1)
	  LogDebug("CSCCathodeLCTProcessor")
	    << "Duplicate LCTs found at boundary of CFEB " << icfeb << " ...";
	if (key_phits[icfeb+1] > key_phits[icfeb]) {
	  if (infoV > 1)
	    LogDebug("CSCCathodeLCTProcessor") 
	      << "   deleting LCT on CFEB " << icfeb;
	  key_strip[icfeb] = -1;
	  key_phits[icfeb] = -1;
	}
	else {
	  if (infoV > 1)
	    LogDebug("CSCCathodeLCTProcessor")
	      << "   deleting LCT on CFEB " << icfeb+1;
	  key_strip[icfeb+1] = -1;
	  key_phits[icfeb+1] = -1;
	}
      }
    }
  }

  // Now loop over CFEBs and pick best two lcts based on no. hits in envelope.
  // In case of equal quality, select the one on lower-numbered CFEBs.
  if (infoV > 1) LogDebug("CSCCathodeLCTProcessor")
    << "\n...... Select best LCTs  ......";
  for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
    if (key_phits[icfeb] > ihits[0]) {
      ihits[1] = ihits[0];
      cfebs[1] = cfebs[0];
      ihits[0] = key_phits[icfeb];
      cfebs[0] = icfeb;
      if (infoV > 1) {
	std::ostringstream strstrm;
	for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
	  strstrm << std::setw(4) << strip_type[icfeb];
	}
	LogDebug("CSCCathodeLCTProcessor")
	  << "strip_type" << strstrm.str()
	  << "\n best: ihits " << ihits[0] << " cfeb " << cfebs[0]
	  << " strip_type " << ((cfebs[0] >= 0) ? strip_type[cfebs[0]] : -1)
	  << "\n next: ihits " << ihits[1] << " cfeb " << cfebs[1]
	  << " strip_type " << ((cfebs[1] >= 0) ? strip_type[cfebs[1]] : -1);
      }
    }
    else if (key_phits[icfeb] > ihits[1]) {
      ihits[1] = key_phits[icfeb];
      cfebs[1] = icfeb;
      if (infoV > 1)
	LogDebug("CSCCathodeLCTProcessor")
	  << "\n next: ihits " << ihits[1] << " cfeb " << cfebs[1]
	  << " strip_type " << ((cfebs[1] >= 0) ? strip_type[cfebs[1]] : -1);
    }
  }

  // fill lct data array key strip with 2 highest hit lcts (if they exist)
  int jlct = 0;
  for (int ilct = 0; ilct < nlcts; ilct++) {
    if (cfebs[ilct] != -1) {
      keystrip_data[jlct][CLCT_CFEB]       = cfebs[ilct];
      keystrip_data[jlct][CLCT_STRIP]      = key_strip[cfebs[ilct]];
      keystrip_data[jlct][CLCT_STRIP_TYPE] = strip_type[cfebs[ilct]];
      if (infoV > 1)
	LogDebug("CSCCathodeLCTProcessor")
	  << "filling key: " << key_strip[cfebs[ilct]]
	  << " type: " << strip_type[cfebs[ilct]];
      jlct++;
    }
  }
} // priorityEncode -- test beam version


void CSCCathodeLCTProcessor::getKeyStripData(
		const unsigned int h_pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS],
		const unsigned int d_pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS],
		int keystrip_data[2][7], const int first_bx) {

  int lct_pattern[NUM_PATTERN_STRIPS];
  int this_layer, this_strip;
  unsigned int quality = 0, bend = 0;
  unsigned int best_quality, best_pattern;
  bool valid[2] = {false,false};

  // Time at which TMB latches LCTs.
  int latch_bx = first_bx + drift_delay;

  // Look at keystrips determined from priorityEncode and find their best
  // pattern based on number of hits matching that pattern (quality).  Also
  // find bend angle.  There are multiple patterns available for each keystrip.

  if (infoV > 1)
    LogDebug("CSCCathodeLCTProcessor")
      << "...............getKeyStripData....................";

  for (int ilct = 0; ilct < 2; ilct++) {
    if (infoV > 1)
      LogDebug("CSCCathodeLCTProcessor")
	<< "lct " << ilct << " keystrip " << keystrip_data[ilct][CLCT_STRIP]
	<< " type " << keystrip_data[ilct][CLCT_STRIP_TYPE];
    if (keystrip_data[ilct][CLCT_STRIP] == -1) {// flag set in priorityEncode()
      if (infoV > 1)
	LogDebug("CSCCathodeLCTProcessor") << "no lct at ilct " << ilct;
      continue;
    }
    for (int pattern_strip = 0; pattern_strip < NUM_PATTERN_STRIPS;
	 pattern_strip++) {
      lct_pattern[pattern_strip] = -999;
      this_layer = pre_hit_pattern[0][pattern_strip];
      this_strip = pre_hit_pattern[1][pattern_strip] +
	keystrip_data[ilct][CLCT_STRIP];
      // This conditional statement prevents us from looking at strips
      // that don't exist along the chamber boundaries.
      if (keystrip_data[ilct][CLCT_STRIP_TYPE] == 1) {
	if (this_strip >= 0 && this_strip < CSCConstants::NUM_HALF_STRIPS) {
	  // Now look at one-shots in bx where TMB latches the LCTs
	  if (((h_pulse[this_layer][this_strip] >> latch_bx) & 1) == 1)
	    lct_pattern[pattern_strip] = 1;
	}
      }
      else {
	if (this_strip >= 0 && this_strip < CSCConstants::NUM_DI_STRIPS) {
	  // Now look at one-shots in bx where TMB latches the LCTs
	  if (((d_pulse[this_layer][this_strip] >> latch_bx) & 1) == 1)
	    lct_pattern[pattern_strip] = 1;
	}
      }
    }

    // Find best pattern and quality associated with key by looping over all 
    // possible patterns
    best_quality = 0;
    best_pattern = 0;

    for (unsigned int pattern_num = 0;
	 pattern_num < CSCConstants::NUM_CLCT_PATTERNS; pattern_num++) {
      getPattern(pattern_num, lct_pattern, quality, bend);
      if (infoV > 1) LogDebug("CSCCathodeLCTProcessor")
	  << "pattern " << pattern_num << " quality " << quality
	  << " bend " << bend;
      // Number of layers hit matching a pattern template is compared
      // to nph_pattern.  The threshold is the same for both half- and
      // di-strip patterns.
      if (quality >= nph_pattern) {
	// If the number of matches is the same for two pattern templates,
	// the higher pattern-template number is selected.
	if ((quality == best_quality && pattern_num > best_pattern) ||
	    (quality >  best_quality)) {
	  if (infoV > 1) LogDebug("CSCCathodeLCTProcessor")
	    << "valid = true at quality " << quality
	    << "  thresh " << nph_pattern;
	  valid[ilct] = true;
	  keystrip_data[ilct][CLCT_PATTERN]    = pattern_num;
	  keystrip_data[ilct][CLCT_BEND]       = bend;
	  keystrip_data[ilct][CLCT_BX]         = first_bx;
	  //keystrip_data[ilct][CLCT_STRIP_TYPE] = stripType;
	  keystrip_data[ilct][CLCT_QUALITY]    = quality;
	  best_quality = quality;
	  best_pattern = pattern_num;
	}
      }
    }

    if (!valid[ilct]) {
      keystrip_data[ilct][CLCT_STRIP] = -1;  // delete lct
      if (infoV > 1)
	LogDebug("CSCCathodeLCTProcessor")
	  << "lct " << ilct << " not over threshold: deleting";
    }
    else {
      if (infoV > 1) {
	LogDebug("CSCCathodeLCTProcessor")
	  << "\n" << "--------- final LCT: " << ilct << " -------------\n"
	  << " key strip "   << keystrip_data[ilct][CLCT_STRIP]
	  << " pattern_num " << keystrip_data[ilct][CLCT_PATTERN]
	  << " quality "     << keystrip_data[ilct][CLCT_QUALITY]
	  << " bend "        << keystrip_data[ilct][CLCT_BEND]
	  << " bx "          << keystrip_data[ilct][CLCT_BX]
	  << " type "        << keystrip_data[ilct][CLCT_STRIP_TYPE] << "\n";
      }
    }
  } // end loop over lcts
} // getKeyStripData -- test beam version


void CSCCathodeLCTProcessor::getPattern(unsigned int pattern_num,
			 const int strip_value[NUM_PATTERN_STRIPS],
			 unsigned int& quality, unsigned int& bend) {

  // This function takes strip "one-shots" at the correct bx to find out
  // which hits fall within a certain pattern.  Quality and bend are then
  // calculated based on which strip pattern and how many layers were hit
  // within the pattern.

  unsigned int layers_hit = 0;
  bool hit_layer[CSCConstants::NUM_LAYERS];

  // Clear hit_layer array to keep track of number of layers hit.
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++)
    hit_layer[i_layer] = false;

  // Loop over all designated patterns.
  for (int strip_num = 0; strip_num < NUM_PATTERN_STRIPS; strip_num++){
    if (strip_value[strip_num] == 1){
      for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++){
	// Loop over layer and see if corresponding strip is on same layer
	// If so then increment number of hits.
	if (i_layer == pattern[pattern_num][strip_num]){
	  // If layer has no hits, then increment number of layers hit.
	  if (hit_layer[i_layer] == false){
	    layers_hit++;
	    hit_layer[i_layer] = true;
	  }
	}
      }
    }
  }
  // Get bend value from pattern.
  bend = pattern[pattern_num][NUM_PATTERN_STRIPS];
  quality = layers_hit;

} // getPattern -- test beam version

// ---------------- End separation of test beam fcns -------------------------

// Dump of configuration parameters.
void CSCCathodeLCTProcessor::dumpConfigParams() const {
  std::ostringstream strm;
  strm << "\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  strm << "+                  CLCT configuration parameters:                  +\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  strm << " bx_width     [duration of signal pulse, in 25 ns bins] = "
       << bx_width << "\n";
  strm << " drift_delay  [time after pre-trigger before TMB latches LCTs] = "
       << drift_delay << "\n";
  strm << " hs_thresh    [min. number of layers hit for pre-trigger] = "
       << hs_thresh << "\n";
  strm << " ds_thresh    [min. number of layers hit for pre-trigger] = "
       << ds_thresh << "\n";
  strm << " nph_pattern  [min. number of layers hit for trigger] = "
       << nph_pattern << "\n";
  strm << " fifo_tbins   [total number of time bins in DAQ readout] = "
       << fifo_tbins << "\n";
  strm << " fifo_pretrig [start time of cathode raw hits in DAQ readout] = "
       << fifo_pretrig << "\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  LogDebug("CSCCathodeLCTProcessor") << strm.str();
}

// Reasonably nice dump of digis on half-strips and di-strips.
void CSCCathodeLCTProcessor::dumpDigis(const int strip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS], const int stripType, const int nStrips) const
{
  LogDebug("CSCCathodeLCTProcessor")
    << "Endcap " << theEndcap << " station " << theStation << " ring "
    << CSCTriggerNumbering::ringFromTriggerLabels(theStation, theTrigChamber)
    << " chamber "
    << CSCTriggerNumbering::chamberFromTriggerLabels(theSector, theSubsector,
						    theStation, theTrigChamber)
    << " strip type " << stripType << " nStrips " << nStrips;

  std::ostringstream strstrm;
  for (int i_strip = 0; i_strip < nStrips; i_strip++) {
    if (i_strip%10 == 0) {
      if (i_strip < 100) strstrm << i_strip/10;
      else               strstrm << (i_strip-100)/10;
    }
    else                 strstrm << " ";
    if ((i_strip+1)%cfeb_strips[stripType] == 0) strstrm << " ";
  }
  strstrm << "\n";
  for (int i_strip = 0; i_strip < nStrips; i_strip++) {
    strstrm << i_strip%10;
    if ((i_strip+1)%cfeb_strips[stripType] == 0) strstrm << " ";
  }
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    strstrm << "\n";
    for (int i_strip = 0; i_strip < nStrips; i_strip++) {
      if (strip[i_layer][i_strip] >= 0) {
	strstrm << std::hex << strip[i_layer][i_strip] << std::dec;
      }
      else {
	strstrm << "-";
      }
      if ((i_strip+1)%cfeb_strips[stripType] == 0) strstrm << " ";
    }
  }
  LogDebug("CSCCathodeLCTProcessor") << strstrm.str();
}

// Returns vector of found CLCTs, if any.
std::vector<CSCCLCTDigi> CSCCathodeLCTProcessor::getCLCTs() {
  std::vector<CSCCLCTDigi> tmpV;
  if (bestCLCT.isValid())   tmpV.push_back(bestCLCT);
  if (secondCLCT.isValid()) tmpV.push_back(secondCLCT);
  return tmpV;
}

void CSCCathodeLCTProcessor::saveAllHits(
 const int distrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS],
 const int halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS]){

  for (int i = 0; i < CSCConstants::NUM_LAYERS; i++){
    theDiStripHits[i].clear();
    theHalfStripHits[i].clear();
  }

  // Routine which stores time hits on strips so that this information
  // can be accessed by routines outside of this class.
  for (int i = 0; i < CSCConstants::NUM_LAYERS; i++){
    for (int j = 0; j < CSCConstants::NUM_DI_STRIPS; j++){
      theDiStripHits[i].push_back(distrip[i][j]);
    }
    for (int j = 0; j < CSCConstants::NUM_HALF_STRIPS; j++){
      theHalfStripHits[i].push_back(halfstrip[i][j]);
    }
  }
}

void CSCCathodeLCTProcessor::setHalfStripHits(const int layer, 
				          const std::vector<int>& hStripHits) {
  if (layer >= 0 && layer < CSCConstants::NUM_LAYERS &&
      int(hStripHits.size()) <= CSCConstants::NUM_HALF_STRIPS) {
    theHalfStripHits[layer] = hStripHits;
  }
  else {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "setHalfStripHits out of bounds!!! layer = " << layer
      << ", hStripHits.size = " << hStripHits.size() << "!" << std::endl;
  }
}

std::vector<int> CSCCathodeLCTProcessor::halfStripHits(const int layer) const {
  if (layer >= 0 && layer < CSCConstants::NUM_LAYERS) {
    return theHalfStripHits[layer];
  }
  else {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "halfStripHits called with wrong layer = " << layer << "!"
      << std::endl;
  }
}

void CSCCathodeLCTProcessor::setHalfStripHit(const int layer, const int strip,
					     const int hit) {
  if ((layer >= 0 && layer < CSCConstants::NUM_LAYERS) &&
      (strip >= 0 && strip < CSCConstants::NUM_HALF_STRIPS)) {
    theHalfStripHits[layer][strip] = hit;
  }
  else {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "setHalfStripHit called with wrong parameter(s): layer = " << layer
      << ", strip = " << strip << "!" << std::endl;
  }
}

int CSCCathodeLCTProcessor::halfStripHit(const int layer,
					 const int strip) const {
  if ((layer >= 0 && layer < CSCConstants::NUM_LAYERS) &&
      (strip >= 0 && strip < CSCConstants::NUM_HALF_STRIPS)) {
    return theHalfStripHits[layer][strip];
  }
  else {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "halfStripHit called with wrong parameter(s): layer = " << layer
      << ", strip = " << strip << "!" << std::endl;
  }
}

void CSCCathodeLCTProcessor::setDiStripHits(const int layer, 
					  const std::vector<int>& dStripHits) {
  if (layer >= 0 && layer < CSCConstants::NUM_LAYERS &&
      int(dStripHits.size()) <= CSCConstants::NUM_DI_STRIPS) {
    theDiStripHits[layer] = dStripHits;
  }
  else {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "setDiStripHits out of bounds!!! layer = " << layer
      << ", dStripHits.size = " << dStripHits.size() << "!" << std::endl;
  }
}

std::vector<int> CSCCathodeLCTProcessor::diStripHits(const int layer) const {
  if (layer >= 0 && layer < CSCConstants::NUM_LAYERS) {
    return theDiStripHits[layer];
  }
  else {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "diStripHits called with wrong layer = " << layer << "!" << std::endl;
  }
}

void CSCCathodeLCTProcessor::setDiStripHit(const int layer, const int strip,
					   const int hit) {
  if ((layer >= 0 && layer < CSCConstants::NUM_LAYERS) &&
      (strip >= 0 && strip < CSCConstants::NUM_DI_STRIPS)) {
    theDiStripHits[layer][strip] = hit;
  }
  else {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "setDiStripHit called with wrong parameter(s): layer = " << layer
      << ", strip = " << strip << "!" << std::endl;
  }
}

int CSCCathodeLCTProcessor::diStripHit(const int layer,
				       const int strip) const {
  if ((layer >= 0 && layer < CSCConstants::NUM_LAYERS) &&
      (strip >= 0 && strip < CSCConstants::NUM_DI_STRIPS)) {
    return theDiStripHits[layer][strip];
  }
  else {
    throw cms::Exception("CSCCathodeLCTProcessor")
      << "diStripHit called with wrong parameter(s): layer = " << layer
      << ", strip = " << strip << "!" << std::endl;
  }
}


////////////////////////////////////////////////////////////////////////
////////////////////////////Test Routines///////////////////////////////

void CSCCathodeLCTProcessor::testDistripStagger() {
  // Author: Jason Mumford (mumford@physics.ucla.edu)
  // This routine tests the distripStagger routine.
  // @@
  bool debug = true;
  int test_triad[CSCConstants::NUM_DI_STRIPS], test_time[CSCConstants::NUM_DI_STRIPS];
  int test_digi[CSCConstants::NUM_DI_STRIPS];
  int distrip = 0;
  test_triad[distrip] = 3;    //After routine, I expect 4
  test_triad[distrip+1] = 3;  //                        4
  test_triad[distrip+2] = 3;  //                        4 
  test_triad[distrip+3] = 3;  //                        4
  test_triad[distrip+4] = 3;  //                        4
  test_triad[distrip+5] = 3;  //                        4
  test_triad[distrip+6] = 3;  //                        4
  test_triad[distrip+7] = 3;  //                        4
  test_triad[distrip+8] = 3;  //                        4
  test_triad[distrip+9] = 3;  //                        4
  test_triad[distrip+10] = 2;  //                       2

  test_time[distrip] = 4;     //      ""      ""        0
  test_time[distrip+1] = 10;  //                        4
  test_time[distrip+2] = 2;   //                        10
  test_time[distrip+3] = 0;   //                        2
  test_time[distrip+4] = 6;   //                        2
  test_time[distrip+5] = 8;   //                        2
  test_time[distrip+6] = 10;   //                        2
  test_time[distrip+7] = 1;   //                        2
  test_time[distrip+8] = 8;   //                        2
  test_time[distrip+9] = 5;   //                        2
  test_time[distrip+10] = 6;   //                        2

  std::cout << "\n ------------------------------------------------- \n";
  std::cout << "!!!!!!Testing distripStagger routine!!!!!!" << std::endl;
  std::cout << "Values before distripStagger routine:" << std::endl;
  for (int i=distrip; i<distrip+11; i++){
    test_digi[i] = 999;
    std::cout << "test_triad[" << i << "] = " << test_triad[i];
    std::cout << "   test_time[" << i << "] = " << test_time[i] << std::endl;
  }
  distripStagger(test_triad, test_time, test_digi, distrip, debug);
  std::cout << "Values after distripStagger routine:" << std::endl;
  for (int i=distrip; i<distrip+11; i++){
    std::cout << "test_triad[" << i << "] = " << test_triad[i];
    std::cout << "   test_time[" << i << "] = " << test_time[i] << std::endl;
  }
  std::cout << "\n ------------------------------------------------- \n \n";
}

void CSCCathodeLCTProcessor::testLCTs() {
  // test to make sure what goes into an LCT is what comes out.
  for (int ptn = 0; ptn < 8; ptn++) {
    for (int bend = 0; bend < 2; bend++) {
      for (int cfeb = 0; cfeb < MAX_CFEBS; cfeb++) {
	for (int key_strip = 0; key_strip < 32; key_strip++) {
	  for (int bx = 0; bx < 7; bx++) {
	    for (int stripType = 0; stripType < 2; stripType++) {
	      for (int quality = 3; quality < 6; quality++) {
		CSCCLCTDigi thisLCT(1, quality, ptn, stripType, bend,
				    key_strip, cfeb, bx);
		if (ptn != thisLCT.getPattern())
		  edm::LogWarning("CSCCathodeLCTProcessor")
		    << "pattern mismatch: " << ptn << " "
		    << thisLCT.getPattern();
		if (bend != thisLCT.getBend())
		  edm::LogWarning("CSCCathodeLCTProcessor")
		    << "bend mismatch: " << bend << " " << thisLCT.getBend();
		if (cfeb != thisLCT.getCFEB()) 
		  edm::LogWarning("CSCCathodeLCTProcessor")
		    << "cfeb mismatch: " << cfeb << " " << thisLCT.getCFEB();
		if (key_strip != thisLCT.getKeyStrip())
		  edm::LogWarning("CSCCathodeLCTProcessor")
		    << "strip mismatch: " << key_strip << " "
		    << thisLCT.getKeyStrip();
		if (bx != thisLCT.getBX())
		  edm::LogWarning("CSCCathodeLCTProcessor")
		    << "bx mismatch: " << bx << " " << thisLCT.getBX();
		if (stripType != thisLCT.getStripType())
		  edm::LogWarning("CSCCathodeLCTProcessor")
		    << "Strip Type mismatch: " << stripType << " "
		    << thisLCT.getStripType();
		if (quality != thisLCT.getQuality())
		  edm::LogWarning("CSCCathodeLCTProcessor")
		    << "quality mismatch: " << quality << " "
		    << thisLCT.getQuality();
	      }
	    }
	  }
	}
      }
    }
  }
}

void CSCCathodeLCTProcessor::printPatterns() {
  // @@
  std::cout<<" Printing patterns for Cathode LCT"<<std::endl;
  std::cout<<"       ";
  for (int patternNum = 0; patternNum < CSCConstants::NUM_CLCT_PATTERNS; patternNum++) {
    std::cout<<" Pattern "<<patternNum<<" ";
  }
  std::cout<<std::endl;
  std::cout<<" Layer ";
  for (int patternNum = 0; patternNum < CSCConstants::NUM_CLCT_PATTERNS; patternNum++) {
    std::cout<<"   Bend "<<(pattern[patternNum][NUM_PATTERN_STRIPS]==0 ? "L": "R")<<"  ";
  }
  std::cout<<std::endl;
  for (int layer = 0; layer < CSCConstants::NUM_LAYERS; layer++) {
    for (int patternNum = 0; patternNum < CSCConstants::NUM_CLCT_PATTERNS; patternNum++) {
      if (patternNum == 0) std::cout<<"   "<<layer<<"       ";
      if (layer != CSCConstants::KEY_CLCT_LAYER-1) {//that old counting from 1 vs 0 thing.
        int minStrip =0;
	if (layer < CSCConstants::KEY_CLCT_LAYER-1) {
	  minStrip = 3*layer;
	} else {
	  minStrip = 3*layer - 2;// since on the key layer we only have 1 strip
	}
        for (int strip = minStrip; strip < minStrip + 3; strip++) {
	  if (layer == pattern[patternNum][strip]) {
	    std::cout<<"X";
	  } else {
	    std::cout<<"_";
	  }
	}
      } else {// on the key layer we always have a hit, right?
	std::cout<<" X ";
      }
      std::cout<<"        ";
    }
    std::cout<<std::endl;
  }
}
    
void CSCCathodeLCTProcessor::testPatterns() {
//generate all possible combinations of hits in a given area and see what we find.
// Benn Tannenbaum 21 June 2001
  
  //there are 16 strips in our uber-pattern, each of which can be on or off.
  // 2^16 = 65536
  for (int possibleHits = 0; possibleHits < 65536; possibleHits++) {
    int stripsHit[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS];
    for (int layer = 0; layer < CSCConstants::NUM_LAYERS; layer++) {
      for (int strip = 0; strip < CSCConstants::NUM_HALF_STRIPS; strip++) {
	stripsHit[layer][strip] = 0;
      }
    }
//assign one bit to each strip in an array. I'll start centered around strip 10.
    stripsHit[0][ 9] = ( possibleHits &     1 ) != 0;     // 2^0
    stripsHit[0][10] = ( possibleHits &     2 ) != 0;     // 2^1
    stripsHit[0][11] = ( possibleHits &     4 ) != 0;     // 2^2
    stripsHit[1][ 9] = ( possibleHits &     8 ) != 0;     // 2^3
    stripsHit[1][10] = ( possibleHits &    16 ) != 0;     // 2^4
    stripsHit[1][11] = ( possibleHits &    32 ) != 0;     // 2^5
    stripsHit[2][ 9] = ( possibleHits &    64 ) != 0;     // 2^6
    stripsHit[2][10] = ( possibleHits &   128 ) != 0;     // 2^7
    stripsHit[2][11] = ( possibleHits &   256 ) != 0;     // 2^8
    stripsHit[3][10] = ( possibleHits &   512 ) != 0;     // 2^9
    stripsHit[4][ 9] = ( possibleHits &  1024 ) != 0;     // 2^10
    stripsHit[4][10] = ( possibleHits &  2048 ) != 0;     // 2^11
    stripsHit[4][11] = ( possibleHits &  4096 ) != 0;     // 2^12
    stripsHit[5][ 9] = ( possibleHits &  8192 ) != 0;     // 2^13
    stripsHit[5][10] = ( possibleHits & 16384 ) != 0;     // 2^14
    stripsHit[5][11] = ( possibleHits & 32768 ) != 0;     // 2^15
    int numLayersHit = findNumLayersHit(stripsHit);
    std::vector <CSCCLCTDigi> results = findLCTs(stripsHit, 1, 2*numStrips+1);
// print out whatever we find-- but only ones where 4 or more layers are hit
// OR ones where we find something
// key: X    a hit there and was used to find pattern
//      x    a hit not involved in pattern
//      _    empty strip
//      o    a hit was there, but no pattern was found
    if (numLayersHit > 3 || results.size() > 0) {
      std::cout<<"Input "<<possibleHits<<"/"<< 65536 <<" # Found Patterns "<<results.size()<<std::endl<<" ";
      for (int layer = 0; layer < CSCConstants::NUM_LAYERS; layer++) {
	if (layer != CSCConstants::KEY_CLCT_LAYER - 1) {
	  for (int strip = 9; strip < 12; strip++) {
	    if (stripsHit[layer][strip] !=0) {
	      if (results.size() > 0) {
	        int thePatternStrip = strip - (results[0].getKeyStrip() - 2) + 3*layer;
		if (layer>=CSCConstants::KEY_CLCT_LAYER) thePatternStrip -= 2;
	        if (pattern[results[0].getPattern()][thePatternStrip] == layer)
		{
		  std::cout<<"X";
		} else {
		  std::cout<<"x";
		}
              } else {
	        std::cout<<"o";
              }
	    } else {
	      std::cout<<"_";
	    }
	  }
	  std::cout<<"   ";
	  for (unsigned int output = 0; output < results.size(); output++) {
	    int minStrip;
	    if (layer < CSCConstants::KEY_CLCT_LAYER-1) {
	      minStrip = 3*layer;
	    } else {
	      minStrip = 3*layer - 2;// since on the key layer we only have 1 strip
	    }
            for (int strip = minStrip; strip < minStrip + 3; strip++) {
	      if (layer == pattern[results[output].getPattern()][strip]) {
		std::cout<<"X";
	      } else {
		std::cout<<"_";
	      }
	    }
	    std::cout<<"  ";
          }
	} else {
          if (stripsHit[layer][10] != 0) {
	    std::cout<<" X ";
	  } else {
	    std::cout<<" _ ";
	  }
	  for (unsigned int output = 0; output < results.size(); output++)
	    std::cout<<"    X   ";
	}
	if (layer < static_cast<int>(results.size()) ) {
	  std::cout<<results[layer];
	  std::cout<<" ";
	} else {
	  std::cout<<" "<<std::endl<<" ";
	}
      }
    }
  }
}

int CSCCathodeLCTProcessor::findNumLayersHit(int 
          stripsHit[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS]) {
  int number = 0;
  for (int layer = 0; layer < CSCConstants::NUM_LAYERS; layer++) {
    if ((stripsHit[layer][ 9] !=0) || 
        (stripsHit[layer][10] !=0) ||
	(stripsHit[layer][11] !=0) ) number++;
  }
  return number;
}
