#ifndef PhysicsTools_HepMCCandAlgos_interface_FlavorHistorySelectorUtil_h
#define PhysicsTools_HepMCCandAlgos_interface_FlavorHistorySelectorUtil_h


// -*- C++ -*-
//
// Package:    FlavorHistorySelectorUtil
// Class:      FlavorHistorySelectorUtil
// 
/**\class FlavorHistorySelectorUtil FlavorHistorySelectorUtil.cc PhysicsTools/FlavorHistorySelectorUtil/src/FlavorHistorySelectorUtil.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  "Salvatore Rappoccio"
//         Created:  Sat Jun 28 00:41:21 CDT 2008
// $Id: FlavorHistorySelectorUtil.h,v 1.2 2009/07/02 09:59:51 srappocc Exp $
//
//


//---------------------------------------------------------------------------
//   FlavorHistorySelectorUtil
//   This will filter events as follows:
//   - Inputs number of b and c jets (at the generator level), the highest
//     flavor in the event, the flavor source, and the delta R between the
//     two jets (if applicable)
//   - If there are no FlavorHistory's that have flavorSource of "type",
//     then the event is rejected.
//   - If there is at least one FlavorHistory that has flavorSource of "type",
//     then we examine the kinematic criteria:
//        - For delta R method, if there is a sister of the parton
//          that is within "minDR" of "this" parton, the event is rejected,
//          otherwise it is passed.
//        - For the pt method, if the parton itself is less than a pt
//          threshold, it is rejected, and if it is above, it is passed
//---------------------------------------------------------------------------

// system include files
#include <memory>

#include "DataFormats/HepMCCandidate/interface/FlavorHistoryEvent.h"
#include "DataFormats/HepMCCandidate/interface/FlavorHistory.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

//
// class declaration
//

namespace reco { 

class FlavorHistorySelectorUtil {
   public:
     typedef reco::FlavorHistory::FLAVOR_T flavor_type;
     typedef std::vector<int>              flavor_vector;

     FlavorHistorySelectorUtil( unsigned int flavor,
				unsigned int noutput,
				flavor_vector const & flavorSource,
				double minDR,
				double maxDR,
				bool verbose );
     ~FlavorHistorySelectorUtil() {} 

     bool select(unsigned int nb,
		 unsigned int nc,
		 unsigned int highestFlavor,
		 FlavorHistory::FLAVOR_T flavorSource,
		 double dr ) const;

   private:
      // ----------member data ---------------------------
      int             flavor_;         // Flavor to examine
      int             noutput_;        // Required number of output HF jets
      flavor_vector   flavorSource_;   // which type to filter on
      double          minDR_;          // For deltaR scheme
      double          maxDR_;          // For deltaR scheme
      bool            verbose_;        // verbosity

      
};

}

#endif 
