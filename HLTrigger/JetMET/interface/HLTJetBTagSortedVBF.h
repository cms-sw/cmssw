#ifndef HLTJetBTagSortedVBF_h
#define HLTJetBTagSortedVBF_h

/** \class HLTJetBTagSortedVBF
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a
 *  single jet requirement with an Energy threshold (not Et!)
 *  Based on HLTSinglet
 *
 *  $Date: 2012/02/03 10:39:05 $
 *  $Revision: 1.1 $
 *
 *  \author Jacopo Bernardini
 *
 */




#include "HLTrigger/HLTcore/interface/HLTFilter.h"




//
// class declaration
//

class HLTJetBTagSortedVBF : public HLTFilter {

   public:

      explicit HLTJetBTagSortedVBF(const edm::ParameterSet&);
      ~HLTJetBTagSortedVBF();
      
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&,trigger::TriggerFilterObjectWithRefs & filterproduct);
      
   private:
      edm::InputTag inputTag_; 
      double mqq;          
      double detaqq;          
      double ptsqq;           
      double ptsbb;
      double seta; 

};

#endif
