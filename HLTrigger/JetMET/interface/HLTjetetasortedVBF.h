#ifndef HLTjetetasortedVBF_h
#define HLTjetetasortedVBF_h

/** \class HLTjetetasortedVBF
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



#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"




//
// class declaration
//

class HLTjetetasortedVBF : public HLTFilter {

   public:

      explicit HLTjetetasortedVBF(const edm::ParameterSet&);
      ~HLTjetetasortedVBF();
      

      virtual bool hltFilter(edm::Event&, const edm::EventSetup&,trigger::TriggerFilterObjectWithRefs& filterproduct);
      
   private:
     
      edm::InputTag inputTag1_; 
      bool saveTags_;           
      double mqq;           
      double detaqq; 
      double detabb;        
      double ptsqq;          
      double ptsbb; 
      double seta; 

     
      

};

#endif 
