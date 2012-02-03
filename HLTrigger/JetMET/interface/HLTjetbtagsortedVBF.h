#ifndef HLTjetbtagsortedVBF_h
#define HLTjetbtagsortedVBF_h

/** \class HLTjetbtagsortedVBF
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a
 *  single jet requirement with an Energy threshold (not Et!)
 *  Based on HLTSinglet
 *
 *  $Date: 2011/05/01 08:19:55 $
 *  $Revision: 1.6 $
 *
 *  \author Jacopo Bernardini
 *
 */




#include "HLTrigger/HLTcore/interface/HLTFilter.h"




//
// class declaration
//

class HLTjetbtagsortedVBF : public HLTFilter {

   public:

      explicit HLTjetbtagsortedVBF(const edm::ParameterSet&);
      ~HLTjetbtagsortedVBF();
      

      virtual bool hltFilter(edm::Event&, const edm::EventSetup&,trigger::TriggerFilterObjectWithRefs & filterproduct);
      
   private:
      edm::InputTag inputTag_; 
     
      bool saveTags_;           
      double mqq;          
      double detaqq;          
      double ptsqq;           
      double ptsbb;
      double seta; 

     
      

};

#endif
