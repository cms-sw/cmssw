#ifndef HLTJetEtaSortedVBF_h
#define HLTJetEtaSortedVBF_h

/** \class HLTJetEtaSortedVBF
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a
 *  single jet requirement with an Energy threshold (not Et!)
 *  Based on HLTSinglet
 *
 *  $Date: 2012/02/03 10:38:10 $
 *  $Revision: 1.1 $
 *
 *  \author Jacopo Bernardini
 *
 */




#include "HLTrigger/HLTcore/interface/HLTFilter.h"




//
// class declaration
//

class HLTJetEtaSortedVBF : public HLTFilter {

   public:

      explicit HLTJetEtaSortedVBF(const edm::ParameterSet&);
      ~HLTJetEtaSortedVBF();
      
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&,trigger::TriggerFilterObjectWithRefs& filterproduct);
      
   private:
      edm::InputTag inputTag_; 
      double mqq;           
      double detaqq; 
      double detabb;        
      double ptsqq;          
      double ptsbb; 
      double seta; 

};

#endif 
