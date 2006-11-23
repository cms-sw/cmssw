#ifndef BTauReco_EMTauTagIsolation_h
#define BTauReco_EMTauTagIsolation_h
//
// \class EMIsolatedTauTagInfo
//


#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/BTauReco/interface/EMIsolatedTauTagInfoFwd.h"

//
using namespace std;

namespace reco { 

  class EMIsolatedTauTagInfo{

  public:
    //default constructor
    EMIsolatedTauTagInfo() {}


    EMIsolatedTauTagInfo(double discriminator, CaloJet jet) 
      {    
	discriminator_ = discriminator;
	myJet_ = jet;
	
    }
    //destructor
    virtual ~EMIsolatedTauTagInfo() {};

    //get the jet from the jetTag
    const CaloJet & jet() const { return myJet_;  }

    //default discriminator: returns the value of the discriminator computed with the parameters taken from the cfg file in the EDProducer
    double discriminator() const { 
      return discriminator_; 
    }

    //Method to recompute the discriminator
    double discriminator(float rMax, float rMin, float pIsolCut)
      {
	double newDiscriminator_ =0;
	//
	//Put here your code
	//
	return newDiscriminator_;
	
      }

    
    //get the reference to the jet
    virtual EMIsolatedTauTagInfo* clone() const { return new EMIsolatedTauTagInfo( *this ); }
  
  private:
    CaloJet myJet_; //Input jets from the constructor
    double discriminator_; //Default discriminator assigned in the EDProducer 
  };
}

#endif
