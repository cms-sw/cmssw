#ifndef HLTGlobalSums_h
#define HLTGlobalSums_h

/** \class HLTGlobalSums
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing cuts on
 *  global sums such as the scalar sum of Et (a.k.a. H_T), available
 *  in the MET object.
 *
 *  $Date: 2006/08/23 17:03:01 $
 *  $Revision: 1.13 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<string>

//
// class decleration
//

class HLTGlobalSums : public HLTFilter {

   public:

      explicit HLTGlobalSums(const edm::ParameterSet&);
      ~HLTGlobalSums();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      // configuration
      edm::InputTag inputTag_; // input tag identifying MET product
      std::string observable_; // which observable to cut on
      double Min_,Max_;        // cut: Min<=observable<=Max
      int Min_N_;              // how many needed to pass
};

#endif //HLTGlobalSums_h
