#ifndef HLTGlobalSums_h
#define HLTGlobalSums_h

/** \class HLTGlobalSums
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing cuts on
 *  global sums such as the scalar sum of Et (a.k.a. H_T), available
 *  in the T=CaloMET or T=MET object.
 *
 *  $Date: 2007/04/13 15:57:57 $
 *  $Revision: 1.6 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<string>

//
// class declaration
//

template<typename T>
class HLTGlobalSums : public HLTFilter {

   public:

      explicit HLTGlobalSums(const edm::ParameterSet&);
      ~HLTGlobalSums();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      // configuration
      edm::InputTag inputTag_; // input tag identifying MET product
      std::string observable_; // which observable to cut on
      double min_,max_;        // cut: Min<=observable<=Max
      int min_N_;              // how many needed to pass
};

#endif //HLTGlobalSums_h
