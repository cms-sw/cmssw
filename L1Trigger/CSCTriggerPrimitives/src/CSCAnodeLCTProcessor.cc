//-----------------------------------------------------------------------------
//
//   Class: CSCAnodeLCTProcessor
//
//   Description: 
//     This is the simulation for the Anode LCT Processor for the Level-1
//     Trigger.  This processor consists of several stages:
//
//       1. Pulse extension of signals coming from wires.
//       2. Pretrigger for each key-wire.
//       3. Pattern detector if a pretrigger is found for the given key-wire.
//       4. Ghost Cancellation Logic (GCL).
//       5. Best track search and promotion.
//       6. Second best track search and promotion.
//
//     The inputs to the ALCT Processor are wire digis.
//     The output is up to two ALCT digi words.
//
//   Author List: Benn Tannenbaum (1999), Jason Mumford (2002), Slava Valuev.
//                Porting from ORCA by S. Valuev (Slava.Valuev@cern.ch),
//                May 2006.
//
//   $Date: 2006/09/27 10:08:17 $
//   $Revision: 1.7 $
//
//   Modifications: 
//
//-----------------------------------------------------------------------------

#include <L1Trigger/CSCTriggerPrimitives/src/CSCAnodeLCTProcessor.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Utilities/interface/Exception.h>
#include <iostream>

//-----------------
// Static variables
//-----------------

const int CSCAnodeLCTProcessor::bx_min = 0;
const int CSCAnodeLCTProcessor::bx_max = CSCConstants::TOT_BUNCH;

/* This is the pattern envelope, which is used to define the collision
   patterns A and B.
   pattern_envelope[0][i]=layer;
   pattern_envelope[1+MEposition][i]=key_wire offset. */
const int CSCAnodeLCTProcessor::pattern_envelope[CSCConstants::NUM_ALCT_PATTERNS][NUM_PATTERN_WIRES] = {
  //Layer
  { 0,  0,  0,
        1,  1,
            2,
            3,  3,
            4,  4,  4,
            5,  5,  5},

  //Keywire offset for ME1 and ME2
  {-2, -1,  0,
       -1,  0,
            0,
            0,  1,
            0,  1,  2,
            0,  1,  2},

  //Keywire offset for ME3 and ME4
  {2,  1,  0,
       1,  0,
           0,
           0, -1,
           0, -1, -2,
           0, -1, -2}
};

// These mask the pattern envelope to give the desired accelerator pattern
// and collision patterns A and B.
#ifndef TB
const int CSCAnodeLCTProcessor::pattern_mask[CSCConstants::NUM_ALCT_PATTERNS][NUM_PATTERN_WIRES] = {
  // Accelerator pattern
  {0,  0,  1,
       0,  1,
           1,
           1,  0,
           1,  0,  0,
           1,  0,  0},

  // Collision pattern A
  {0,  1,  0,
       1,  1,
           1,
           1,  0,
           0,  1,  0,
           0,  1,  0},

  // Collision pattern B
  {1,  1,  0,
       1,  1,
           1,
           1,  1,
           0,  1,  1,
           0,  0,  1}
};
#else
// Both collision patterns used at the test beam are "completely open"
// by default.
const int CSCAnodeLCTProcessor::pattern_mask[CSCConstants::NUM_ALCT_PATTERNS][NUM_PATTERN_WIRES] = {
  // Accelerator pattern
  {0,  0,  1,
       0,  1,
           1,
           1,  0,
           1,  0,  0,
           1,  0,  0},

  // Collision pattern A
  {1,  1,  1,
       1,  1,
           1,
           1,  1,
           1,  1,  1,
           1,  1,  1},

  // Collision pattern B
  {1,  1,  1,
       1,  1,
           1,
           1,  1,
           1,  1,  1,
           1,  1,  1}
};
#endif

//----------------
// Constructors --
//----------------

CSCAnodeLCTProcessor::CSCAnodeLCTProcessor(unsigned endcap, unsigned station,
					   unsigned sector, unsigned subsector,
					   unsigned chamber,
					   const edm::ParameterSet& conf) : 
		     theEndcap(endcap), theStation(station), theSector(sector),
                     theSubsector(subsector), theTrigChamber(chamber) {
  static bool config_dumped = false;

  // ALCT configuration parameters.
  bx_width     = conf.getParameter<int>("alctBxWidth");     // def = 6
  nph_thresh   = conf.getParameter<int>("alctNphThresh");   // def = 2
  nph_pattern  = conf.getParameter<int>("alctNphPattern");  // def = 4
  drift_delay  = conf.getParameter<int>("alctDriftDelay");  // def = 3
  fifo_pretrig = conf.getParameter<int>("alctFifoPretrig"); // def = 12
  trig_mode    = conf.getParameter<int>("alctTrigMode"); // def = 3 (2 for TB)
  alct_amode   = conf.getParameter<int>("alctMode");        // def = 1

  // Currently used only in the test-beam mode.
  fifo_tbins   = conf.getParameter<int>("alctFifoTbins");   // def = 8
  l1a_window   = conf.getParameter<int>("alctL1aWindow");   // def = 5

  // Verbosity level, set to 0 (no print) by default.
  infoV        = conf.getUntrackedParameter<int>("verbosity", 0);

  // Print configuration parameters.
  if (infoV > 0 && !config_dumped) {
    dumpConfigParams();
    config_dumped = true;
  }

  numWireGroups = 0;
  MESelection   = (theStation < 3) ? 0 : 1;
}

CSCAnodeLCTProcessor::CSCAnodeLCTProcessor() :
  		     theEndcap(1), theStation(1), theSector(1),
		     theSubsector(1), theTrigChamber(1) {
  // Used for debugging. -JM
  static bool config_dumped = false;

  // ALCT configuration parameters.
  bx_width     =  6;
  nph_thresh   =  2;
  nph_pattern  =  4;
  drift_delay  =  3;
  fifo_pretrig = 12;
  trig_mode    =  3;
  alct_amode   =  1;
  fifo_tbins   =  8; // test-beam only.
  l1a_window   =  5; // test-beam only.

  infoV        = 2;

  // Print configuration parameters.
  if (!config_dumped) {
    dumpConfigParams();
    config_dumped = true;
  }

  numWireGroups = CSCConstants::MAX_NUM_WIRES;
  MESelection   = (theStation < 3) ? 0 : 1;
}

void CSCAnodeLCTProcessor::clear() {
  bestALCT.clear();
  secondALCT.clear();
  for (int i = 0; i < CSCConstants::NUM_LAYERS; i++) {
    theWireHits[i].clear();
  }
}

void CSCAnodeLCTProcessor::clear(const int wire, const int pattern) {
  /* Clear the data off of selected pattern */
  if (pattern == 0) quality[wire][0] = -999;
  else {
    quality[wire][1] = -999;
    quality[wire][2] = -999;
  }
}

std::vector<CSCALCTDigi>
CSCAnodeLCTProcessor::run(const CSCWireDigiCollection* wiredc) {
  // This is the main routine for normal running.  It gets wire times
  // from the wire digis and then passes them on to another run() function.

  int wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES];
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++)
    for (int i_wire = 0; i_wire < CSCConstants::MAX_NUM_WIRES; i_wire++)
      wire[i_layer][i_wire] = -999;

  // clear(); // redundant; called by L1MuCSCMotherboard.

  // Get the number of wire groups for the given chamber.
  CSCTriggerGeomManager* theGeom = CSCTriggerGeometry::get();
  CSCChamber* theChamber = theGeom->chamber(theEndcap, theStation, theSector,
					    theSubsector, theTrigChamber);
  if (theChamber) {
    numWireGroups = theChamber->layer(1)->geometry()->numberOfWireGroups();
    if (numWireGroups > CSCConstants::MAX_NUM_WIRES) {
      throw cms::Exception("CSCAnodeLCTProcessor")
	<< "+++ Number of wire groups, " << numWireGroups
	<< ", exceeds max expected, " << CSCConstants::MAX_NUM_WIRES
	<< " +++" << std::endl;
    }
  }
  else {
    edm::LogWarning("CSCAnodeLCTProcessor")
      << "+++ CSC chamber (endcap = " << theEndcap
      << " station = " << theStation << " sector = " << theSector
      << " subsector = " << theSubsector << " trig. id = " << theTrigChamber
      << ") is not defined in current geometry!"
      << " Set numWireGroups to zero +++" << "\n";
    numWireGroups = 0;
  }

  // Get wire digis in this chamber from wire digi collection.
  getDigis(wiredc);

  // First get wire times from the wire digis.
  readWireDigis(wire);

  // Save all hits into vectors if needed.
  // saveAllHits(wire);

  // Then pass an array of wire times on to another run() doing the LCT search.
  run(wire);

  // Return vector of ALCTs.
  std::vector<CSCALCTDigi> tmpV = getALCTs();
  return tmpV;
}

void CSCAnodeLCTProcessor::run(const int wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES]) {
  // This version of the run() function can either be called in a standalone
  // test, being passed the time array, or called by the run() function above.
  // It gets wire times from an input array and then loops over the keywires.
  // All found LCT candidates are sorted and the best two are retained.

  // Check if there are any in-time hits and do the pulse extension.
  bool chamber_empty = pulseExtension(wire);

  // Only do the rest of the processing if chamber is not empty.
  if (!chamber_empty) {
    for (int i_wire = 0; i_wire < numWireGroups; i_wire++) {
      if (preTrigger(i_wire)) {
  	if (infoV > 2) showPatterns(i_wire);
	patternDetection(i_wire);
      }
    }
    ghostCancellationLogic();
    lctSearch();
  }
}

void CSCAnodeLCTProcessor::getDigis(const CSCWireDigiCollection* wiredc) {
  // Routine for getting digis and filling digiV vector.
  int theRing    = CSCTriggerNumbering::ringFromTriggerLabels(theStation,
							      theTrigChamber);
  int theChamber = CSCTriggerNumbering::chamberFromTriggerLabels(theSector,
                                    theSubsector, theStation, theTrigChamber);

  // Loop over layers and save wire digis on each one into digiV[layer].
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    digiV[i_layer].clear();

    CSCDetId detid(theEndcap, theStation, theRing, theChamber, i_layer+1);

    const CSCWireDigiCollection::Range rwired = wiredc->get(detid);

    // Skip if no wire digis in this layer.
    if (rwired.second == rwired.first) continue;

    if (infoV > 1) LogDebug("CSCAnodeLCTProcessor")
      << "found " << rwired.second - rwired.first
      << " wire digi(s) in layer " << i_layer << "; " << theEndcap << " "
      << theStation << " " << theRing << " " << theSector << "("
      << theSubsector << ") " << theChamber << "(" << theTrigChamber << ")";

    for (CSCWireDigiCollection::const_iterator digiIt = rwired.first;
	 digiIt != rwired.second; ++digiIt) {
      digiV[i_layer].push_back(*digiIt);
      if (infoV > 1) LogDebug("CSCAnodeLCTProcessor") << "   " << (*digiIt);
    }
  }
}

void CSCAnodeLCTProcessor::getDigis(const std::vector<std::vector<CSCWireDigi> > digis) {
  // Routine for getting digis and filling digiV vector.
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++){
    digiV[i_layer] = digis[i_layer];
  }
}

void CSCAnodeLCTProcessor::readWireDigis(int wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES]) {
  /* Gets wire times from the wire digis and fills wire[][] array */
  int i_wire, bx_time;

#ifndef TB
  // Keep large [bxwin_min; bxwin_max] in official ORCA for now and let CSC TF
  // handle it.
  int bxwin_min = CSCConstants::MIN_BUNCH;
  int bxwin_max = CSCConstants::MAX_BUNCH + 1;
#else
  // Total number of time bins in DAQ readout is given by fifo_tbins, which
  // thus determines the maximum length of time interval.
  // Anode raw hits in DAQ readout start (fifo_pretrig - 10) clocks before
  // L1Accept.  If times earlier than L1Accept were recorded, we use them
  // since they can modify the ALCTs found later, via ghost-cancellation logic.
  // In addition, we discard hits which come later than ALCT-L1A window,
  // which may shorten this maximum length.
  int bxwin_min   = 0;
  int early_tbins = fifo_pretrig - 10;
  if (early_tbins < 0) {
    edm::LogWarning("CSCAnodeLCTProcessor|OutOfTime")
      << "+++ fifo_pretrig = " << fifo_pretrig
      << "; you are loosing in-time ALCT hits!!! +++" << "\n";
  }
  // +1 is empirical, to get agreement with the test beam data
  int bxwin_max   = l1a_window + early_tbins + 1;
  if (bxwin_max > fifo_tbins) bxwin_max = fifo_tbins;
#endif

  // Loop over all 6 layers.
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    // Loop over all digis in the layer and find the wireGroup and bx
    // time for each.
    for (std::vector<CSCWireDigi>::iterator pld = digiV[i_layer].begin();
	 pld != digiV[i_layer].end(); pld++) {
      i_wire  = pld->getWireGroup()-1;
      bx_time = pld->getBeamCrossingTag();

      // Check that the wires and times are appropriate.
      if (i_wire < 0 || i_wire >= numWireGroups) {
	throw cms::Exception("CSCAnodeLCTProcessor")
	  << "+++ Found wire digi with wrong wire number = " << i_wire
	  << ", max wires = " << numWireGroups << " +++" << std::endl;
      }
      if (bx_time >= bxwin_min && bx_time < bxwin_max) {
	if (infoV > 2) LogDebug("CSCAnodeLCTProcessor")
	  << "Digi on layer " << i_layer << " wire " << i_wire
	  << " at time " << bx_time;

	// Shift all times of interest by TIME_OFFSET, so that they will
	// be positive.
	bx_time += CSCConstants::TIME_OFFSET;
	if (bx_time <= bx_min || bx_time > bx_max) {
	  edm::LogWarning("CSCAnodeLCTProcessor")
	    << "Mismatch in anode BX times " << bx_time << " +++ \n";
	  continue;
	}

	// Finally save times of hit wires.  If there is more than one hit
	// on the same wire, pick the one which occurred earlier.
	if (wire[i_layer][i_wire] <= bx_min ||
	    wire[i_layer][i_wire] > bx_time) {
	  wire[i_layer][i_wire] = bx_time;
	}
      }
      else {
	edm::LogWarning("CSCAnodeLCTProcessor")
	  << "Unexpected BX time of wire digi: wire = " << i_wire
	  << " layer = " << i_layer << ", bx = " << bx_time << " +++ \n";
      }
    }
  }
}

bool CSCAnodeLCTProcessor::pulseExtension(const int wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES]){
  /* A pulse array will be used as a bit representation of hit times.
     For example: if a keywire has a bx_time of 3, then 1 shifted
     left 3 will be bit pattern 0000000000001000.  Bits are then added to
     signify the duration of a signal (bx_width).  So for the pulse with a
     bx_width of 6 will look like 0000000111111000.*/

  bool chamber_empty = true;
  int i_wire, i_layer, digi_num;

  for (i_wire = 0; i_wire < numWireGroups; i_wire++) {
    for (i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
      pulse[i_layer][i_wire] = 0;
    }
    first_bx[i_wire] = -999;
    for (int j = 0; j < 3; j++) quality[i_wire][j] = -999;
  }

  for (i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++){
    digi_num = 0;
    for (i_wire = 0; i_wire < numWireGroups; i_wire++) {
      if (wire[i_layer][i_wire] >= bx_min && wire[i_layer][i_wire] <= bx_max) {
	// Found at least one in-time digi; set chamber_empty to false
	if (chamber_empty) chamber_empty = false;

	// make the pulse
	for (int bx = wire[i_layer][i_wire];
	     bx < (wire[i_layer][i_wire] + bx_width); bx++)
	  pulse[i_layer][i_wire] = pulse[i_layer][i_wire] | (1 << bx);

	// Debug information.
	if (infoV > 1) {
	  LogDebug("CSCAnodeLCTProcessor")
	    << "Wire digi: layer " << i_layer
	    << " digi #" << ++digi_num << " wire group " << i_wire
	    << " time " << wire[i_layer][i_wire];
	  if (infoV > 2) {
	    std::ostringstream strstrm;
	    for (int i = 1; i <= 32; i++) {
	      strstrm << ((pulse[i_layer][i_wire]>>(32-i)) & 1);
	    }
	   LogDebug("CSCAnodeLCTProcessor") << "  Pulse: " << strstrm.str();
	  }
	}
      }
    }
  }

  if (infoV > 1 && !chamber_empty) {
    dumpDigis(wire);
  }

  return chamber_empty;
}

bool CSCAnodeLCTProcessor::preTrigger(const int key_wire) {
  /* Check that there are nph_thresh or more layers hit in collision
     or accelerator patterns for a particular key_wire.  If so, return
     true and the PatternDetection process will start. */

  int layers_hit;
  bool hit_layer[CSCConstants::NUM_LAYERS];
  int this_layer, this_wire;

  // Loop over bx times, accelerator and collision patterns to 
  // look for pretrigger.
  for (int bx_time = bx_min; bx_time <= bx_max; bx_time++) {
    for (int i_pattern = 0; i_pattern < CSCConstants::NUM_ALCT_PATTERNS; i_pattern++) {
      for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++)
	hit_layer[i_layer] = false;
      layers_hit = 0;

      for (int i_wire = 0; i_wire < NUM_PATTERN_WIRES; i_wire++){
	if (pattern_mask[i_pattern][i_wire] != 0){
	  this_layer = pattern_envelope[0][i_wire];
	  this_wire  = pattern_envelope[1+MESelection][i_wire]+key_wire;
	  if ((this_wire >= 0) && (this_wire < numWireGroups)){
	    // Perform bit operation to see if pulse is 1 at a certain bx_time.
	    if (((pulse[this_layer][this_wire] >> bx_time) & 1) == 1){
	      // Store number of layers hit.
	      if (hit_layer[this_layer] == false){
		hit_layer[this_layer] = true;
		layers_hit++;
	      }

	      // See if number of layers hit is greater than or equal to
	      // nph_thresh.
	      if (layers_hit >= nph_thresh) {
		first_bx[key_wire] = bx_time;
		if (infoV > 1) {
		  LogDebug("CSCAnodeLCTProcessor")
		    << "Pretrigger was satisfied for wire: " << key_wire
		    << " pattern: " << i_pattern
		    << " bx_time: " << bx_time;
		}
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

void CSCAnodeLCTProcessor::patternDetection(const int key_wire) {
  /* See if there is a pattern that satisfies nph_pattern number of
     layers hit for either the accelerator or collision patterns.  Use
     the pattern with the best quality. */

  bool hit_layer[CSCConstants::NUM_LAYERS];
  int this_layer, this_wire, temp_quality;
  const std::string ptn_label[] = {"Accelerator", "CollisionA", "CollisionB"};

  for (int i_pattern = 0; i_pattern < CSCConstants::NUM_ALCT_PATTERNS; i_pattern++){
    temp_quality = 0;
    for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++)
      hit_layer[i_layer] = false;

    for (int i_wire = 0; i_wire < NUM_PATTERN_WIRES; i_wire++){
      if (pattern_mask[i_pattern][i_wire] != 0){
	this_layer = pattern_envelope[0][i_wire];
	this_wire  = pattern_envelope[1+MESelection][i_wire]+key_wire;
	if ((this_wire >= 0) && (this_wire < numWireGroups)){

	  // Wait a drift_delay time later and look for layers hit in
	  // the pattern.
	  if ((pulse[this_layer][this_wire] >> 
	       (first_bx[key_wire] + drift_delay)) & 1 == 1) {

	    // If layer has never had a hit before, then increment number
	    // of layer hits.
	    if (hit_layer[this_layer] == false){
	      temp_quality++;
	      // keep track of which layers already had hits.
	      hit_layer[this_layer] = true;
	      if (infoV > 1)
		LogDebug("CSCAnodeLCTProcessor")
		  << "bx_time: " << first_bx[key_wire]
		  << " pattern: " << i_pattern << " keywire: " << key_wire
		  << " layer: "     << this_layer
		  << " quality: "   << temp_quality;
	    }
	  }
	}
      }
    }
    if (temp_quality >= nph_pattern) {
      // Quality reported by the pattern detector is defined as the number
      // of the layers hit in a pattern minus (nph_pattern-1) value
      temp_quality -= (nph_pattern-1);
      if (i_pattern == 0) {
	// Accelerator pattern
	quality[key_wire][0] = temp_quality;
      }
      else {
	// Only one collision pattern (of the best quality) is reported
	if (temp_quality > quality[key_wire][1]) {
	  quality[key_wire][1] = temp_quality;
	  quality[key_wire][2] = i_pattern-1;
	}
      }
      if (infoV > 1) {
	LogDebug("CSCAnodeLCTProcessor")
	  << "Pattern found; keywire: "  << key_wire
	  << " type: " << ptn_label[i_pattern]
	  << " quality: " << temp_quality << "\n";
      }
    }
  }
  if (infoV > 1 && quality[key_wire][1] > 0) {
    if (quality[key_wire][2] == 0)
      LogDebug("CSCAnodeLCTProcessor")
	<< "Collision Pattern A is chosen" << "\n";
    else if (quality[key_wire][2] == 1)
      LogDebug("CSCAnodeLCTProcessor")
	<< "Collision Pattern B is chosen" << "\n";
  }
}

void CSCAnodeLCTProcessor::ghostCancellationLogic() {
  /* This function looks for LCTs on the previous and next wires.  If one
     exists and it has a better quality and a bx_time up to 4 clocks earlier
     than the present, then the present LCT is cancelled.  The present LCT
     also gets cancelled if it has the same quality as the one on the
     previous wire (this has not been done in 2003 test beam).  The
     cancellation is done separately for collision and accelerator patterns. */

  int ghost_cleared[CSCConstants::MAX_NUM_WIRES][2];

  for (int key_wire = 0; key_wire < numWireGroups; key_wire++) {
    for (int i_pattern = 0; i_pattern < 2; i_pattern++) {
      ghost_cleared[key_wire][i_pattern] = 0;

      // Non-empty wire group.
      int qual_this = quality[key_wire][i_pattern];
      if (qual_this > 0) {

	// Previous wire.
	int qual_prev = (key_wire > 0) ? quality[key_wire-1][i_pattern] : 0;
	if (qual_prev > 0) {
	  int dt = first_bx[key_wire] - first_bx[key_wire-1];
	  // Cancel this wire
	  //   1) If the candidate at the previous wire is at the same bx
	  //      clock and has better quality (or equal quality - this has
	  //      been implemented only in 2004).
	  //   2) If the candidate at the previous wire is up to 4 clocks
	  //      earlier, regardless of quality.
	  if (dt == 0) {
	    if (qual_prev >= qual_this) ghost_cleared[key_wire][i_pattern] = 1;
	  }
	  else if (dt > 0 && dt <= 4) {
	    ghost_cleared[key_wire][i_pattern] = 1;
	  }
	}

	// Next wire.
	// Skip this step if this wire is already declared "ghost".
	if (ghost_cleared[key_wire][i_pattern] == 1) continue;
	int qual_next =
	  (key_wire < numWireGroups-1) ? quality[key_wire+1][i_pattern] : 0;
	if (qual_next > 0) {
	  int dt = first_bx[key_wire] - first_bx[key_wire+1];
	  // Same cancellation logic as for the previous wire.
	  if (dt == 0) {
	    if (qual_next > qual_this) ghost_cleared[key_wire][i_pattern] = 1;
	  }
	  else if (dt > 0 && dt <= 4) {
	    ghost_cleared[key_wire][i_pattern] = 1;
	  }
	}
      }
    }
  }

  // All cancellation is done in parallel, so wiregroups do not know what
  // their neighbors are cancelling.
  for (int key_wire = 0; key_wire < numWireGroups; key_wire++) {
    for (int i_pattern = 0; i_pattern < 2; i_pattern++) {
      if (ghost_cleared[key_wire][i_pattern] > 0) {
	clear(key_wire, i_pattern);
	if (infoV > 1) LogDebug("CSCAnodeLCTProcessor")
	  << ((i_pattern == 0) ? "Accelerator" : "Collision")
	  << " pattern ghost cancelled on key_wire: " << key_wire << "\n";
      }
    }
  }
}

void CSCAnodeLCTProcessor::lctSearch() {
  // First modify the quality according alct_amode, then store all
  // of the valid LCTs in an array.
  std::vector<CSCALCTDigi> lct_list;

  for (int i_wire = 0; i_wire < numWireGroups; i_wire++){
    // If there is either accelerator or collision, perform trigMode
    // and alctAmode functions before storing and sorting.
    if ((quality[i_wire][0] > 0) || (quality[i_wire][1] > 0)){
      trigMode(i_wire);
      alctAmode(i_wire);

      // Store any valid accelerator pattern LCTs.
      if (quality[i_wire][0] > 0) {
	int qual = (quality[i_wire][0] & 0x03); // 2 LSBs
	CSCALCTDigi lct_info(1, qual, 1, 0, i_wire, first_bx[i_wire]);
	lct_list.push_back(lct_info);
      }

      // Store any valid collision pattern LCTs.
      if (quality[i_wire][1] > 0) {
	int qual = (quality[i_wire][1] & 0x03); // 2 LSBs
	CSCALCTDigi lct_info(1, qual, 0, quality[i_wire][2], i_wire,
			     first_bx[i_wire]);
	lct_list.push_back(lct_info);
      }
    }
  }

  // Now sort over stored LCTs
  for (std::vector <CSCALCTDigi>::iterator plct = lct_list.begin();
       plct != lct_list.end(); plct++){
#ifdef TB
    // Do not report ALCTs found prior to L1Accept.
    int early_tbins = fifo_pretrig - 10 + CSCConstants::TIME_OFFSET;
    if (plct->getBX() < early_tbins) {
      if (infoV > 1) LogDebug("CSCAnodeLCTProcessor")
	<< " Do not report ALCT on keywire " << plct->getKeyWG()
	<< ": found at bx " << plct->getBX() << ", whereas the earliest "
	<< "allowed bx is " << early_tbins;
      continue;
    }
#endif
    if (!bestALCT.isValid() || *plct > bestALCT) {
      secondALCT = bestALCT;
      bestALCT = *plct;
    }
    else if (!secondALCT.isValid() || *plct > secondALCT) {
      secondALCT = *plct;
    }
  }

#ifdef TB
  // Firmware "feature" in 2003 and 2004 test beam data.
  if (secondALCT.isValid() && secondALCT.getBX() > bestALCT.getBX())
    secondALCT.clear();
#endif

  if (bestALCT.isValid()) {
    bestALCT.setTrknmb(1);
    if (infoV > 0) {
      LogDebug("CSCAnodeLCTProcessor")
	<< bestALCT << " found in endcap " << theEndcap
	<< " station " << theStation << " sector " << theSector
	<< " (" << theSubsector
	<< ") ring " << CSCTriggerNumbering::ringFromTriggerLabels(theStation,
						        theTrigChamber)
	<< " chamber "
	<< CSCTriggerNumbering::chamberFromTriggerLabels(theSector,
                              theSubsector, theStation, theTrigChamber)
	<< " (trig id. " << theTrigChamber << ")" << "\n";
    }
    if (secondALCT.isValid()) {
      secondALCT.setTrknmb(2);
      if (infoV > 0) {
	LogDebug("CSCAnodeLCTProcessor")
	  << secondALCT << " found in endcap " << theEndcap
	  << " station " << theStation << " sector " << theSector
	  << " (" << theSubsector
	  << ") ring "<< CSCTriggerNumbering::ringFromTriggerLabels(theStation,
						       theTrigChamber)
	  << " chamber "
	  << CSCTriggerNumbering::chamberFromTriggerLabels(theSector,
                             theSubsector, theStation, theTrigChamber)
	  << " (trig id. " << theTrigChamber << ")" << "\n";
      }
    }
  }
}

void CSCAnodeLCTProcessor::trigMode(const int key_wire) {
  /* Function which enables/disables either collision or accelerator tracks.
     The function uses the trig_mode parameter to decide. */

  switch(trig_mode) {
  default:
  case 0:
    // Enables both collision and accelerator tracks
    break;
  case 1:
    // Disables collision tracks
    if (quality[key_wire][1] > 0) {
      quality[key_wire][1] = 0;
      if (infoV > 1) LogDebug("CSCAnodeLCTProcessor")
	<< "trigMode(): collision track " << key_wire << " disabled" << "\n";
    }
    break;
  case 2:
    // Disables accelerator tracks
    if (quality[key_wire][0] > 0) {
      quality[key_wire][0] = 0;
      if (infoV > 1) LogDebug("CSCAnodeLCTProcessor")
	<< "trigMode(): accelerator track " << key_wire << " disabled" << "\n";
    }
    break;
  case 3:
    // Disables collision track if there is an accelerator track found
    // in the same wire group at the same time
    if (quality[key_wire][0] > 0 && quality[key_wire][1] > 0) {
      quality[key_wire][1] = 0;
      if (infoV > 1) LogDebug("CSCAnodeLCTProcessor")
	<< "trigMode(): collision track " << key_wire << " disabled" << "\n";
    }
    break;
  }
}

void CSCAnodeLCTProcessor::alctAmode(const int key_wire) {
  /* Function which gives a preference either to the collision patterns
     or accelerator patterns.  The function uses the alct_amode parameter
     to decide. */
  int promotionBit = 1 << 2;

  switch(alct_amode) {
  default:
  case 0:
    // Ignore accelerator muons.
    if (quality[key_wire][0] > 0) {
      quality[key_wire][0] = 0;
      if (infoV > 1) LogDebug("CSCAnodeLCTProcessor")
	<< "alctMode(): accelerator track " << key_wire << " ignored" << "\n";
    }
    break;
  case 1:
    // Prefer collision muons by adding promotion bit.
    if (quality[key_wire][1] > 0) {
      quality[key_wire][1] += promotionBit;
      if (infoV > 1) LogDebug("CSCAnodeLCTProcessor")
	<< "alctMode(): collision track " << key_wire << " promoted" << "\n";
    }
    break;
  case 2:
    // Prefer accelerator muons by adding promotion bit.
    if (quality[key_wire][0] > 0) {
      quality[key_wire][0] += promotionBit;
      if (infoV > 1) LogDebug("CSCAnodeLCTProcessor")
	<< "alctMode(): accelerator track " << key_wire << " promoted"<< "\n";
    }
    break;
  case 3:
    // Ignore collision muons.
    if (quality[key_wire][1] > 0) {
      quality[key_wire][1] = 0;
      if (infoV > 1) LogDebug("CSCAnodeLCTProcessor")
	<< "alctMode(): collision track " << key_wire << " ignored" << "\n";
    }
    break;
  }
}

// Dump of configuration parameters.
void CSCAnodeLCTProcessor::dumpConfigParams() const {
  std::ostringstream strm;
  strm << "\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  strm << "+                  ALCT configuration parameters:                  +\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  strm << " bx_width     [duration of signal pulse, in 25 ns bins] = "
       << bx_width << "\n";
  strm << " nph_thresh   [min. number of layers hit for pre-trigger] = "
       << nph_thresh << "\n";
  strm << " nph_pattern  [min. number of layers hit for trigger] = "
       << nph_pattern << "\n";
  strm << " drift_delay  [drift delay after pre-trigger, in 25 ns bins] = "
       << drift_delay << "\n";
  strm << " fifo_pretrig [start time of anode raw hits in DAQ readout] = "
       << fifo_pretrig << "\n";
  strm << " trig_mode    [enabling/disabling collision/accelerator tracks] = "
       << trig_mode << "\n";
  strm << " alct_amode   [preference to collision/accelerator tracks] = "
       << alct_amode << "\n";
  strm << " fifo_tbins   [total number of time bins in DAQ readout] = "
       << fifo_tbins << "\n";
  strm << " l1a_window   [L1Accept window, in 25 ns bins] = "
       << l1a_window << "\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  LogDebug("CSCAnodeLCTProcessor") << strm.str();
}

// Dump of digis on wire groups.
void CSCAnodeLCTProcessor::dumpDigis(const int wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES]) const {
  LogDebug("CSCAnodeLCTProcessor")
    << "Endcap " << theEndcap << " station " << theStation << " ring "
    << CSCTriggerNumbering::ringFromTriggerLabels(theStation, theTrigChamber)
    << " chamber "
    << CSCTriggerNumbering::chamberFromTriggerLabels(theSector, theSubsector,
						    theStation, theTrigChamber)
    << " nWiregroups " << numWireGroups;

  std::ostringstream strstrm;
  for (int i_wire = 0; i_wire < numWireGroups; i_wire++) {
    if (i_wire%10 == 0) {
      if (i_wire < 100) strstrm << i_wire/10;
      else              strstrm << (i_wire-100)/10;
    }
    else                strstrm << " ";
  }
  strstrm << "\n";
  for (int i_wire = 0; i_wire < numWireGroups; i_wire++) {
    strstrm << i_wire%10;
  }
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    strstrm << "\n";
    for (int i_wire = 0; i_wire < numWireGroups; i_wire++) {
      if (wire[i_layer][i_wire] >= 0) {
	strstrm << std::hex << wire[i_layer][i_wire] << std::dec;
      }
      else {
	strstrm << ".";
      }
    }
  }
  LogDebug("CSCAnodeLCTProcessor") << strstrm.str();
}

// Returns vector of found ALCTs, if any.
std::vector<CSCALCTDigi> CSCAnodeLCTProcessor::getALCTs() {
  std::vector<CSCALCTDigi> tmpV;
  if (bestALCT.isValid())   tmpV.push_back(bestALCT);
  if (secondALCT.isValid()) tmpV.push_back(secondALCT);
  return tmpV;
}

void CSCAnodeLCTProcessor::saveAllHits(const int wires[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES]){
  // Small routine which saves hits on all wires and stores for access to 
  // outside classes.
  for (int i = 0; i < CSCConstants::NUM_LAYERS; i++) {
    for (int j = 0; j < CSCConstants::MAX_NUM_WIRES; j++) {
      theWireHits[i].push_back(wires[i][j]);
    }
  }
}

void CSCAnodeLCTProcessor::setWireHits(const int layer,
				       const std::vector<int>& wireHits) {
  if (layer >= 0 && layer < CSCConstants::NUM_LAYERS) {
    theWireHits[layer] = wireHits;
  }
  else {
    throw cms::Exception("CSCAnodeLCTProcessor")
      << "setWireHits called with wrong layer = " << layer << "!" << std::endl;
  }
}

std::vector<int> CSCAnodeLCTProcessor::wireHits(const int layer) const {
  if (layer >= 0 && layer < CSCConstants::NUM_LAYERS) {
    return theWireHits[layer];
  }
  else {
    throw cms::Exception("CSCAnodeLCTProcessor")
      << "wireHits called with wrong layer = " << layer << "!" << std::endl;
  }
}

void CSCAnodeLCTProcessor::setWireHit(const int layer, const int wire,
				      const int hit) {
  if ((layer >= 0 && layer < CSCConstants::NUM_LAYERS) &&
      (wire  >= 0 && wire  < CSCConstants::MAX_NUM_WIRES)) {
    theWireHits[layer][wire] = hit;
  }
  else {
    throw cms::Exception("CSCAnodeLCTProcessor")
      << "setWireHit called with wrong parameter(s): layer = " << layer
      << ", wire = " << wire << "!" << std::endl;
  }
}
 
int CSCAnodeLCTProcessor::wireHit(const int layer, const int wire) const {
  if ((layer >= 0 && layer < CSCConstants::NUM_LAYERS) &&
      (wire  >= 0 && wire  < CSCConstants::MAX_NUM_WIRES)) {
    return theWireHits[layer][wire];
  }
  else {
    throw cms::Exception("CSCAnodeLCTProcessor")
      << "wireHit called with wrong parameter(s): layer = " << layer
      << ", wire = " << wire << "!" << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////
////////////////////////////Test Routines///////////////////////////////

void CSCAnodeLCTProcessor::showPatterns(const int key_wire) {
  /* Method to test the pretrigger */
  for (int i_pattern = 0; i_pattern < CSCConstants::NUM_ALCT_PATTERNS;
       i_pattern++) {
    std::ostringstream strstrm_header;
    LogDebug("CSCAnodeLCTProcessor")
      << "\n" << "Pattern: " << i_pattern << " Key wire: " << key_wire;
    for (int i = 1; i <= 32; i++) {
      strstrm_header << ((32-i)%10);
    }
    LogDebug("CSCAnodeLCTProcessor") << strstrm_header.str();
    for (int i_wire = 0; i_wire < NUM_PATTERN_WIRES; i_wire++) {
      if (pattern_mask[i_pattern][i_wire] != 0) {
	std::ostringstream strstrm_pulse;
	int this_layer = pattern_envelope[0][i_wire];
	int this_wire  = pattern_envelope[1+MESelection][i_wire]+key_wire;
	for (int i = 1; i <= 32; i++) {
	  strstrm_pulse << ((pulse[this_layer][this_wire]>>(32-i)) & 1);
	}
	LogDebug("CSCAnodeLCTProcessor")
	  << strstrm_pulse.str() << " on layer " << this_layer;
      }
    }
    LogDebug("CSCAnodeLCTProcessor")
      << "-------------------------------------------";
  }
}
