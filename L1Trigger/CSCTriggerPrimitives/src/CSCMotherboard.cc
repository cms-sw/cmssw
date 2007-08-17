//-----------------------------------------------------------------------------
//
//   Class: CSCMotherboard
//
//   Description: 
//    When the Trigger MotherBoard is instantiated it instantiates an ALCT
//    and CLCT board.  The Motherboard takes up to two LCTs from each anode
//    and cathode LCT card and combines them into a single Correlated LCT.
//    The output is up to two Correlated LCTs.
//
//    It can be run in either a test mode, where the arguments are a collection
//    of wire times and arrays of comparator times & comparator results, or
//    for general use, with with wire digi and comparator digi collections as
//    arguments.  In the latter mode, the wire & strip info is passed on the
//    LCTProcessors, where it is decoded and converted into a convenient form.
//    After running the anode and cathode LCTProcessors, TMB correlates the
//    anode and cathode LCTs.  At present, it simply matches the best CLCT
//    with the best ALCT; perhaps a better algorithm will be determined in
//    the future.  The MotherBoard then determines a few more numbers (such as
//    quality and pattern) from the ALCT and CLCT information, and constructs
//    two correlated LCTs.
//
//    correlateLCTs() may need to be modified to take into account a
//    possibility of ALCTs and CLCTs arriving at different bx times.
//
//   Author List: Benn Tannenbaum 28 August 1999 benn@physics.ucla.edu
//                Based on code by Nick Wisniewski (nw@its.caltech.edu)
//                and a framework by Darin Acosta (acosta@phys.ufl.edu).
//
//   $Date: 2007/08/15 12:51:19 $
//   $Revision: 1.9 $
//
//   Modifications: Numerous later improvements by Jason Mumford and
//                  Slava Valuev (see cvs in ORCA).
//   Porting from ORCA by S. Valuev (Slava.Valuev@cern.ch), May 2006.
//
//-----------------------------------------------------------------------------

#include <L1Trigger/CSCTriggerPrimitives/src/CSCMotherboard.h>
//#include <Utilities/Timing/interface/TimingReport.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

CSCMotherboard::CSCMotherboard() :
                   theEndcap(1), theStation(1), theSector(1),
                   theSubsector(1), theTrigChamber(1) {
  // Constructor used only for testing.  -JM
  alct = new CSCAnodeLCTProcessor();
  clct = new CSCCathodeLCTProcessor();
  infoV = 2;
}

CSCMotherboard::CSCMotherboard(unsigned endcap, unsigned station,
			       unsigned sector, unsigned subsector,
			       unsigned chamber,
			       const edm::ParameterSet& conf) :
                   theEndcap(endcap), theStation(station), theSector(sector),
                   theSubsector(subsector), theTrigChamber(chamber) {
  // Normal constructor.  -JM
  // Pass ALCT, CLCT, and common parameters on to ALCT and CLCT processors.

  // Some congiguration parameters and some details of the emulator
  // algorithms depend on whether we want to emulate the trigger logic
  // used in TB/MTCC or its future, hoped-for modification (the latter
  // is used in MC studies).
  edm::ParameterSet commonParams =
    conf.getParameter<edm::ParameterSet>("commonParam");
  isMTCC = commonParams.getParameter<bool>("isMTCC");

  // Choose the appropriate set of configuration parameters depending on
  // isMTCC flag.
  edm::ParameterSet alctParams, clctParams;
  if (!isMTCC) {
    alctParams = conf.getParameter<edm::ParameterSet>("alctParamDef");
    clctParams = conf.getParameter<edm::ParameterSet>("clctParamDef");
  }
  else {
    alctParams = conf.getParameter<edm::ParameterSet>("alctParamMTCC2");
    clctParams = conf.getParameter<edm::ParameterSet>("clctParamMTCC2");
  }
  alct = new CSCAnodeLCTProcessor(endcap, station, sector, subsector,
				  chamber, alctParams, commonParams);
  clct = new CSCCathodeLCTProcessor(endcap, station, sector, subsector,
				    chamber, clctParams, commonParams);

  // Motherboard parameters: common for all configurations.
  edm::ParameterSet tmbParams  =
    conf.getParameter<edm::ParameterSet>("tmbParam");
  infoV = tmbParams.getUntrackedParameter<int>("verbosity", 0);

  // test to make sure that what goes into a correlated LCT is also what
  // comes back out.
  // testLCT();
}

CSCMotherboard::~CSCMotherboard() {
  if (alct) delete alct;
  if (clct) delete clct;
}

void CSCMotherboard::clear() {
  if (alct) alct->clear();
  if (clct) clct->clear();
  firstLCT.clear();
  secondLCT.clear();
}

// Set configuration parameters obtained via EventSetup mechanism.
void CSCMotherboard::setConfigParameters(const L1CSCTPParameters* conf) {
  alct->setConfigParameters(conf);
  clct->setConfigParameters(conf);
  // No config. parameters for the TMB itself yet.
}

void CSCMotherboard::run(
           int time1[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES],
	   int time2[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_STRIPS],
	   int triad[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_STRIPS]) {
  // Debug version.  -JM
  clear();
  alct->run(time1);               // run anode LCT
  clct->run(triad, time2, time2); // run cathodeLCT
  if (alct->bestALCT.isValid() || clct->bestCLCT.isValid() )
    correlateLCTs(alct->bestALCT, alct->secondALCT,
		  clct->bestCLCT, clct->secondCLCT);
}

std::vector<CSCCorrelatedLCTDigi>
CSCMotherboard::run(const CSCWireDigiCollection* wiredc,
		    const CSCComparatorDigiCollection* compdc) {
  clear();
  if (alct && clct) {
    {
      //static TimingReport::Item & alctTimer =
      //(*TimingReport::current())["CSCAnodeLCTProcessor:run"];
      //TimeMe t(alctTimer, false);
      std::vector<CSCALCTDigi> alctV = alct->run(wiredc); // run anodeLCT
    }
    {
      //static TimingReport::Item & clctTimer =
      //(*TimingReport::current())["CSCCathodeLCTProcessor:run"];
      //TimeMe t(clctTimer, false);
      std::vector<CSCCLCTDigi> clctV = clct->run(compdc); // run cathodeLCT
    }
    // It may seem like the next function should be
    // 'if (alct->bestALCT.isValid() && clct->bestCLCT.isValid())'.
    // It is || instead of && because the decision to reject non-valid LCTs
    // is handled further upstream (assuming at least 1 is valid).  -JM
    if (alct->bestALCT.isValid() || clct->bestCLCT.isValid())
      correlateLCTs(alct->bestALCT, alct->secondALCT,
		    clct->bestCLCT, clct->secondCLCT);
    if (infoV > 0) {
      if (firstLCT.isValid()) {
	LogDebug("CSCMotherboard") << firstLCT;
      }
      if (secondLCT.isValid()) {
	LogDebug("CSCMotherboard") << secondLCT;
      }
    }
  }
  else {
    edm::LogWarning("CSCMotherboard")
      << "+++ run() called for non-existing ALCT/CLCT processor! +++ \n";
  }

  std::vector<CSCCorrelatedLCTDigi> tmpV = getLCTs();
  return tmpV;
}

// Returns vector of found correlated LCTs, if any.
std::vector<CSCCorrelatedLCTDigi> CSCMotherboard::getLCTs() {
  std::vector<CSCCorrelatedLCTDigi> tmpV;
  if (firstLCT.isValid())  tmpV.push_back(firstLCT);
  if (secondLCT.isValid()) tmpV.push_back(secondLCT);
  return tmpV;
}

void CSCMotherboard::correlateLCTs(CSCALCTDigi bestALCT,
				   CSCALCTDigi secondALCT,
				   CSCCLCTDigi bestCLCT,
				   CSCCLCTDigi secondCLCT) {

  bool anodeBestValid     = bestALCT.isValid();
  bool anodeSecondValid   = secondALCT.isValid();
  bool cathodeBestValid   = bestCLCT.isValid();
  bool cathodeSecondValid = secondCLCT.isValid();

  // determine STA value; obsolete as of April 2002.
  // int tempSTA = findSTA(anodeBestValid, anodeSecondValid,
  //                       cathodeBestValid, cathodeSecondValid);

  if (anodeBestValid && !anodeSecondValid)     secondALCT = bestALCT;
  if (!anodeBestValid && anodeSecondValid)     bestALCT   = secondALCT;
  if (cathodeBestValid && !cathodeSecondValid) secondCLCT = bestCLCT;
  if (!cathodeBestValid && cathodeSecondValid) bestCLCT   = secondCLCT;

  // TB/MTCC only, or always????
  if (isMTCC && (bestCLCT.isValid() == false && secondCLCT.isValid() == false))
    return;

  firstLCT = constructLCTs(bestALCT, bestCLCT);
  firstLCT.setTrknmb(1);

  if ((secondALCT != bestALCT) || (secondCLCT != bestCLCT)) {
    secondLCT = constructLCTs(secondALCT, secondCLCT);
    secondLCT.setTrknmb(2);
  }
}

// This method calculates all the TMB words and then passes them to the
// constructor of correlated LCTs.
CSCCorrelatedLCTDigi CSCMotherboard::constructLCTs(const CSCALCTDigi& aLCT,
						   const CSCCLCTDigi& cLCT) {
  // CLCT pattern number
  unsigned int pattern = encodePattern(cLCT.getPattern(), cLCT.getStripType());

  // LCT quality number
  unsigned int quality = findQuality(aLCT, cLCT);

  // bunch crossing match; obsolete as of April 2002.
  // int bxnMatch = findBxnMatch(aLCT.getBX(), cLCT.getBX());

  // Bunch crossing: get it from cathode LCT if anode LCT is not there.
  int bx = aLCT.isValid() ? aLCT.getBX() : cLCT.getBX();

  // construct correlated LCT; temporarily assign track number of 0.
  int trknmb = 0;
  CSCCorrelatedLCTDigi thisLCT(trknmb, 1, quality, aLCT.getKeyWG(),
			       cLCT.getKeyStrip(), pattern, cLCT.getBend(),
			       bx);
  return thisLCT;
}

// CLCT pattern number: encodes the pattern number itself and
// whether the pattern consists of half-strips or di-strips.
unsigned int CSCMotherboard::encodePattern(const int ptn,
					   const int stripType) {
  const int kPatternBitWidth = 4;

  // Cathode pattern number is a kPatternBitWidth-1 bit word.
  unsigned int pattern = (abs(ptn) & ((1<<(kPatternBitWidth-1))-1));

  // The pattern has the MSB (4th bit in the default version) set if it
  // consists of half-strips.
  if (stripType) {
    pattern = pattern | (1<<(kPatternBitWidth-1));
  }

  return pattern;
}

// 4-bit LCT quality number.  Definition can be found in
// http://www.phys.ufl.edu/~acosta/tb/tmb_quality.txt.  Made by TMB lookup
// tables and used for MPC sorting.
unsigned int CSCMotherboard::findQuality(const CSCALCTDigi& aLCT,
					 const CSCCLCTDigi& cLCT) {
  unsigned int quality = 0;

  bool isDistrip = (cLCT.getStripType() == 0);

  if (aLCT.isValid() && !(cLCT.isValid())) {    // no CLCT
    if (aLCT.getAccelerator()) {quality =  1;}
    else                       {quality =  3;}
  }
  else if (!(aLCT.isValid()) && cLCT.isValid()) { // no ALCT
    if (isDistrip)             {quality =  4;}
    else                       {quality =  5;}
  }
  else if (aLCT.isValid() && cLCT.isValid()) { // both ALCT and CLCT
    if (aLCT.getAccelerator()) {quality =  2;} // accelerator muon
    else {                                     // collision muon
      // CLCT quality is, in fact, the number of layers hit, so subtract 3
      // to get quality analogous to ALCT one.
      int sumQual = aLCT.getQuality() + (cLCT.getQuality()-3);
      if (sumQual < 1 || sumQual > 6) {
	edm::LogWarning("CSCMotherboard")
	  << "+++ findQuality: sumQual = " << sumQual << "+++ \n";
      }
      if (isDistrip) { // distrip pattern
	if (sumQual == 2)      {quality =  6;}
	else if (sumQual == 3) {quality =  7;}
	else if (sumQual == 4) {quality =  8;}
	else if (sumQual == 5) {quality =  9;}
	else if (sumQual == 6) {quality = 10;}
      }
      else {            // halfstrip pattern
	if (sumQual == 2)      {quality = 11;}
	else if (sumQual == 3) {quality = 12;}
	else if (sumQual == 4) {quality = 13;}
	else if (sumQual == 5) {quality = 14;}
	else if (sumQual == 6) {quality = 15;}
      }
    }
  }
  return quality;
}

// STA is a status word for the ALCTs and CLCTs.  -JM
// In the latest TMB design this word is no longer present /SV, 03-Apr-02/.
int CSCMotherboard::findSTA(const bool a1, const bool a2,
			    const bool c1, const bool c2) {
  int STA = 0; // if no incoming LCTs

  if (a1 && a2 && !c1 && !c2)        // if 2 anode LCTs and 0 cathode LCTs
      STA = 1;
  else if (!a1 && !a2 && c1 && c2)   // if 2 cathode LCTs and 0 anode LCTs
      STA = 1;
  else if (!a1 && a2 && c1 && !c2)   // if ambiguous LCTs
      STA = 1;
  else if (a1 && !a2 && !c1 && c2)   // if ambiguous LCTs
      STA = 1;
  else if (!a1 && a2 && !c1 && c2)   // if ambiguous LCTs
      STA = 1;
  else if (a1 && !a2 && c1 && c2)    // if 1 anode and 2 cathodes exist...
      STA = 2;
  else if (!a1 && a2 && c1 && c2)
      STA = 1;
  else if (a1 && a2 && c1 && !c2)    // if 2 anodes and 1 cathode exist...
      STA = 2;
  else if (a1 && a2 && !c1 && c2)
      STA = 1;
  else if (a1 && !a2 && c1 && !c2)   // if one unambiguous muon
      STA = 3;
  else if (a1 && a2 && c1 && c2)     // if two unambiguous muons
      STA = 3;
  else if (a1 || a2 || c1 || c2 )    // if only 1 LCT
      STA = 1;
  else {
    edm::LogWarning("CSCMotherboard")
      << "+++ findSTA: STA not assigned: \n"
      << " a1 " << a1 << " a2 " << a2 << " c1 " << c1 << " c2 " << c2
      << " +++ \n";
  }

  return STA;
}

// Cathode-Anode bxn match, as defined in Trigger TDR.
// This word is not present in the TMB-02 design /SV, 03-Apr-02/.
int CSCMotherboard::findBxnMatch(const int aBxn, const int cBxn) {
  int bxnMatch = 3; // worst case scenario

  if (aBxn == cBxn) {bxnMatch = 0;} // perfect match
  else if ((aBxn - cBxn) == 1) {bxnMatch = 1;}
  else if ((cBxn - aBxn) == 1) {bxnMatch = 2;}
  return bxnMatch;
}

void CSCMotherboard::testLCT() {
  unsigned int lctPattern, lctQuality;
  for (int pattern = 0; pattern < 8; pattern++) {
    for (int bend = 0; bend < 2; bend++) {
      for (int cfeb = 0; cfeb < 5; cfeb++) {
	for (int strip = 0; strip < 32; strip++) {
	  for (int bx = 0; bx < 7; bx++) {
	    for (int stripType = 0; stripType < 2; stripType++) {
	      for (int quality = 3; quality < 7; quality++) {
		CSCCLCTDigi cLCT(1, quality, pattern, stripType, bend,
				 strip, cfeb, bx);
		lctPattern = encodePattern(cLCT.getPattern(),
					   cLCT.getStripType());
		for (int aQuality = 0; aQuality < 4; aQuality++) {
		  for (int wireGroup = 0; wireGroup < 120; wireGroup++) {
		    for (int abx = 0; abx < 7; abx++) {
		      CSCALCTDigi aLCT(1, aQuality, 0, 1, wireGroup, abx);
		      lctQuality = findQuality(aLCT, cLCT);
		      CSCCorrelatedLCTDigi
			thisLCT(0, 1, lctQuality, aLCT.getKeyWG(),
				cLCT.getKeyStrip(), lctPattern, cLCT.getBend(),
				aLCT.getBX());
		      if (lctPattern != static_cast<unsigned int>(thisLCT.getPattern()) )
			edm::LogWarning("CSCMotherboard")
			  << "pattern mismatch: " << lctPattern
			  << " " << thisLCT.getPattern();
		      if (bend != thisLCT.getBend()) 
			edm::LogWarning("CSCMotherboard")
			  << "bend mismatch: " << bend
			  << " " << thisLCT.getBend();
		      int key_strip = 32*cfeb + strip;
		      if (key_strip != thisLCT.getStrip()) 
			edm::LogWarning("CSCMotherboard")
			  << "strip mismatch: " << key_strip
			  << " " << thisLCT.getStrip();
		      if (wireGroup != thisLCT.getKeyWG()) 
			edm::LogWarning("CSCMotherboard")
			  << "wire group mismatch: " << wireGroup
			  << " " << thisLCT.getKeyWG();
		      if (abx != thisLCT.getBX()) 
			edm::LogWarning("CSCMotherboard")
			  << "bx mismatch: " << abx << " " << thisLCT.getBX();
		      if (lctQuality != static_cast<unsigned int>(thisLCT.getQuality())) 
			edm::LogWarning("CSCMotherboard")
			  << "quality mismatch: " << lctQuality
			  << " " << thisLCT.getQuality();
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
}
