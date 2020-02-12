#include "L1Trigger/TrackFindingTMTT/interface/MiniHTstage.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"

#include <L1Trigger/TrackFindingTMTT/interface/Sector.h>

namespace TMTT {

MiniHTstage::MiniHTstage( const Settings* settings ) :
  settings_( settings ),
  miniHTstage_( settings_->miniHTstage() ),
  muxOutputsHT_( settings_->muxOutputsHT() ),
  houghNbinsPt_( settings_->houghNbinsPt() ),
  houghNbinsPhi_( settings_->houghNbinsPhi() ),
  miniHoughLoadBalance_( settings_->miniHoughLoadBalance() ),
  miniHoughNbinsPt_( settings_->miniHoughNbinsPt() ),
  miniHoughNbinsPhi_( settings_->miniHoughNbinsPhi() ),
  miniHoughMinPt_( settings_->miniHoughMinPt() ),
  miniHoughDontKill_( settings_->miniHoughDontKill() ),
  miniHoughDontKillMinPt_( settings_->miniHoughDontKillMinPt() ),
  numSubSecsEta_( settings_->numSubSecsEta() ),
  numPhiNonants_( settings_->numPhiNonants() ),
  numPhiSecPerNon_( settings_->numPhiSectors() / numPhiNonants_ ),
  numEtaRegions_( settings_->numEtaRegions() ),
  busySectorKill_( settings_->busySectorKill() ),
  busySectorNumStubs_( settings_->busySectorNumStubs()),
  busySectorMbinRanges_( settings_->busySectorMbinRanges() ),
  chosenRofPhi_( settings_->chosenRofPhi() ),
// Get size of 1st stage HT cells.
  binSizeQoverPtAxis_( miniHoughNbinsPt_ * 2. / (float)settings->houghMinPt() / (float)houghNbinsPt_ ),
  binSizePhiTrkAxis_( miniHoughNbinsPhi_ * 2. * M_PI / (float)settings->numPhiSectors() / (float)houghNbinsPhi_ ),
  invPtToDphi_( settings_->invPtToDphi() ),
  nHTlinksPerNonant_(0)
{
  nMiniHTcells_ = miniHoughNbinsPt_*miniHoughNbinsPhi_;

  if (miniHoughLoadBalance_ > 0) {
    if (muxOutputsHT_ == 3) {
      nHTlinksPerNonant_ = busySectorMbinRanges_.size() - 1;
    } else if (muxOutputsHT_ == 2) {
      nHTlinksPerNonant_ = (busySectorMbinRanges_.size() - 1) * numPhiSecPerNon_;
    } else {
      throw cms::Exception("MiniHTState: hard-wired MHT load balancing can't be enabled as assumes HT output MUX scheme >= 2");
    }
  }
}

void MiniHTstage::exec( matrix< HTrphi >& mHtRphis ) {

  for ( unsigned int iPhiNon = 0; iPhiNon < numPhiNonants_; iPhiNon++ ) {
    map< pair<unsigned int, unsigned int>, unsigned int> numStubsPerLinkStage1; // Indices are ([link ID, MHT cell], #stubs).
    map< pair<unsigned int, unsigned int>, unsigned int> numStubsPerLinkStage2; // Indices are ([link ID, MHT cell], #stubs).
    map< unsigned int, unsigned int> numStubsPerLink; // Indices are ([link ID, MHT cell], #stubs).
    for ( unsigned int iSecInNon = 0; iSecInNon < numPhiSecPerNon_; iSecInNon++ ) {
      unsigned int iPhiSec = iPhiNon * numPhiSecPerNon_ + iSecInNon;
      for ( unsigned int iEtaReg = 0; iEtaReg < numEtaRegions_; iEtaReg++ ) {

        Sector sector;
        sector.init( settings_, iPhiSec, iEtaReg );
        const float& phiCentre = sector.phiCentre();
        HTrphi& htRphi = mHtRphis( iPhiSec, iEtaReg );
        const vector< L1track2D >& roughTracks = htRphi.trackCands2D();
        vector< L1track2D > fineTracks;

        for (const L1track2D& roughTrk : roughTracks) {

          float roughTrkPhi = reco::deltaPhi( roughTrk.phi0() - chosenRofPhi_ * invPtToDphi_ * roughTrk.qOverPt() - phiCentre, 0. );
          const pair< unsigned int, unsigned int >& cell = roughTrk.getCellLocationHT();
          const vector< const Stub* >& stubs = roughTrk.getStubs();
          bool fineTrksFound = false;
	  bool storeCoarseTrack = false;
          const unsigned int& link = roughTrk.optoLinkID();

	  if (fabs(roughTrk.qOverPt()) < 1./miniHoughMinPt_) { // Not worth using mini-HT at low Pt due to scattering.

	    for ( unsigned int mBin = 0; mBin < miniHoughNbinsPt_; mBin++ ) {
	      float qOverPtBin = roughTrk.qOverPt() - binSizeQoverPtAxis_ / 2. + ( mBin + .5 ) * binSizeQoverPtAxis_ / settings_->miniHoughNbinsPt();
	      for ( unsigned int cBin = 0; cBin < miniHoughNbinsPhi_; cBin++ ){
		float phiBin = reco::deltaPhi( roughTrkPhi - binSizePhiTrkAxis_ / 2. + ( cBin + .5 ) * binSizePhiTrkAxis_ / settings_->miniHoughNbinsPhi(), 0. );
		HTcell htCell;
		const bool mergedCell = false; // This represents mini cell.
		const bool miniHTcell = true; 
		htCell.init( settings_, iPhiSec, iEtaReg, sector.etaMin(), sector.etaMax(), qOverPtBin, cell.first + mBin, mergedCell, miniHTcell );
		// Firmware doesn't use bend filter in MHT.
		htCell.disableBendFilter();

		for ( auto& stub: stubs ) {
		  // Ensure stubs are digitized with respect to the current phi sector.
		  if ( settings_->enableDigitize() )
		    ( const_cast< Stub* >( stub ) )->digitizeForHTinput( iPhiSec );
		  float phiStub = reco::deltaPhi( stub->phi() + invPtToDphi_ * qOverPtBin * ( stub->r() - chosenRofPhi_ ) - phiCentre, 0. );
		  float dPhi = reco::deltaPhi( phiBin - phiStub, 0. );
		  float dPhiMax = binSizePhiTrkAxis_ / miniHoughNbinsPhi_ / 2. + invPtToDphi_ * binSizeQoverPtAxis_ / (float)miniHoughNbinsPt_ * fabs( stub->r() - chosenRofPhi_ ) / 2.;
		  if ( fabs( dPhi ) <= fabs( reco::deltaPhi( dPhiMax, 0. ) ) ) htCell.store( stub, sector.insideEtaSubSecs( stub ) );
		}
		htCell.end();
		if ( htCell.trackCandFound() ) {

		  // Do load balancing.
		  unsigned int trueLinkID = linkIDLoadBalanced(link, mBin, cBin, htCell.numStubs(), numStubsPerLinkStage1, numStubsPerLinkStage2);

		  pair< unsigned int, unsigned int > cellLocation( cell.first + mBin, cell.second + cBin );
		  pair< float, float > helix2D( qOverPtBin, reco::deltaPhi( phiBin + chosenRofPhi_ * invPtToDphi_ * qOverPtBin + phiCentre, 0. ) );
		  L1track2D fineTrk( settings_, htCell.stubs(), cellLocation, helix2D, iPhiSec, iEtaReg, trueLinkID, mergedCell );
		  // Truncation due to output opto-link bandwidth.
		  bool keep( true );
		  /*
	          pair<unsigned int, unsigned int> encodedLink;
		  if (miniHoughLoadBalance_ > 0) {
		    encodedLink = pair<unsigned int, unsigned int>(newererLink, mhtCell);
		  } else {
		    encodedLink = pair<unsigned int, unsigned int>(newererLink, 0);
		  }
	          numStubsPerLink[ encodedLink ] += htCell.numStubs();
	          if ( busySectorKill_ && numStubsPerLink[ encodedLink ] > busySectorNumStubs_ ) keep = false;
		  */
	          numStubsPerLink[ trueLinkID ] += htCell.numStubs();
	          if ( busySectorKill_ && numStubsPerLink[ trueLinkID ] > busySectorNumStubs_ ) keep = false;
		  if ( keep ) {
		    fineTracks.push_back( fineTrk );
		    fineTrksFound = true;
		  }
		}
	      }
	    }

	  } else { 

	    // Keep rough track if below Pt threshold where mini-HT in use.
	    storeCoarseTrack = true;
	    
	  }

          if ( storeCoarseTrack || ( (not fineTrksFound) && miniHoughDontKill_ && fabs(roughTrk.qOverPt()) < 1./miniHoughDontKillMinPt_ ) ) {
	    // Keeping original track instead of mini-HTtracks.
	    // Invent dummy miniHT cells so as to be able to reuse load balancing, trying all combinations to identify the least used link.
	    pair<unsigned int, unsigned int> bestCell = {0,0};
	    unsigned int bestNumStubsPerLink = 999999;
	    for ( unsigned int mBin = 0; mBin < miniHoughNbinsPt_; mBin++ ) {
	      for ( unsigned int cBin = 0; cBin < miniHoughNbinsPhi_; cBin++ ){
		unsigned int testLinkID = linkIDLoadBalanced(link, mBin, cBin, roughTrk.getNumStubs(), numStubsPerLinkStage1, numStubsPerLinkStage2, true);
		if (numStubsPerLink[ testLinkID ] < bestNumStubsPerLink) {
		  bestCell = {mBin, cBin};
		  bestNumStubsPerLink = numStubsPerLink[ testLinkID ];
		}
	      }
	    }

	    // Repeat for best link, this time incremementing stub counters.
	    unsigned int trueLinkID = linkIDLoadBalanced(link, bestCell.first, bestCell.second, roughTrk.getNumStubs(), numStubsPerLinkStage1, numStubsPerLinkStage2);

            bool keep( true );
	    /*
	    pair<unsigned int, unsigned int> encodedLink;
	    if (miniHoughLoadBalance_ > 0) {
	      encodedLink = pair<unsigned int, unsigned int>(link/nMiniHTcells_, link%nMiniHTcells_);
	    } else {
	      encodedLink = pair<unsigned int, unsigned int>(link, 0);
	    }
            numStubsPerLink[ encodedLink ] += roughTrk.getNumStubs();
            if ( busySectorKill_ && numStubsPerLink[ encodedLink ] > busySectorNumStubs_ ) keep = false;
	    */
	    numStubsPerLink[ trueLinkID ] += roughTrk.getNumStubs();
	    if ( busySectorKill_ && numStubsPerLink[ trueLinkID ] > busySectorNumStubs_ ) keep = false;
            if ( keep ) {
              fineTracks.push_back( roughTrk );
	      fineTracks.back().setOptoLinkID(trueLinkID);
	    }
          }
        }
        // Replace all existing tracks inside HT array with new ones.
        htRphi.replaceTrackCands2D( fineTracks );
      }
    }
  }
}

//=== Do load balancing
//=== (numStubs is stubs on this track, "link" is link ID before load balancing, and return argument is link ID after load balancing).
//=== (numStubsPerLinkStage* are stub counters per link used to determine best balance. If test=true, then these counters are not to be incrememented).

unsigned int MiniHTstage::linkIDLoadBalanced(unsigned int link, unsigned int mBin, unsigned int cBin, unsigned int numStubs, map< pair<unsigned int, unsigned int>, unsigned int>& numStubsPerLinkStage1, map< pair<unsigned int, unsigned int>, unsigned int>& numStubsPerLinkStage2, bool test) const {

  unsigned int mhtCell = miniHoughNbinsPhi_*mBin + cBin; // Send each mini-cell to a different output link 

  // Number of output links after static load balancing roughly same as number of
  // input links with this, with nSep per MHT cell.
  unsigned int nSep = std::ceil(float(nHTlinksPerNonant_)/float(nMiniHTcells_));

  unsigned int newLink, newerLink, newererLink;

  if (miniHoughLoadBalance_ >= 1) {
    // Static load balancing, 4 -> 1, with each MHT cell sent to seperate output link.
    newLink = link%nSep;   // newLink in range 0 to nSep-1.
  } else {
    newLink = link;
  }

  if (miniHoughLoadBalance_ >= 2) {	  	  

    // 2-stage dynamic load balancing amongst links corresponding to same MHT cell.

    // Dynamically mix pairs of neighbouring links.
    unsigned int balancedLinkA = 2*(newLink/2);
    unsigned int balancedLinkB = balancedLinkA + 1;

    pair<unsigned int, unsigned int> encodedLinkA(balancedLinkA, mhtCell); // balancedLink* here in range 0 to nSep-1.
    pair<unsigned int, unsigned int> encodedLinkB(balancedLinkB, mhtCell);
    if (numStubsPerLinkStage1[encodedLinkA] < numStubsPerLinkStage1[encodedLinkB]) {
      newerLink = balancedLinkA;
    } else {
      newerLink = balancedLinkB;
    } 
    pair<unsigned int, unsigned int> encodedLinkAB(newerLink, mhtCell);
    if (not test) numStubsPerLinkStage1[ encodedLinkAB ] += numStubs;

    // Dynamically mix pairs of next-to-neighbouring links.
    unsigned int balancedLinkY = newerLink;
    //unsigned int balancedLinkZ = (newerLink%4<=1) ? balancedLinkY + 2 : balancedLinkY - 2;
    unsigned int balancedLinkZ = (newerLink+2)%nSep;

    pair<unsigned int, unsigned int> encodedLinkY(balancedLinkY, mhtCell);
    pair<unsigned int, unsigned int> encodedLinkZ(balancedLinkZ, mhtCell);
    if (numStubsPerLinkStage2[encodedLinkY] < numStubsPerLinkStage2[encodedLinkZ]) {
      newererLink = balancedLinkY;
    } else {
      newererLink = balancedLinkZ;
    } 
    pair<unsigned int, unsigned int> encodedLinkYZ(newererLink, mhtCell);
    if (not test) numStubsPerLinkStage2[ encodedLinkYZ ] += numStubs;

  } else {
    newerLink = newLink;
    newererLink = newLink;
  }
  unsigned int trueLinkID = (miniHoughLoadBalance_ > 0) ? nMiniHTcells_ * newererLink + mhtCell : newererLink;
  return trueLinkID;
}

}
