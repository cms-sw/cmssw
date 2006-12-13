#ifndef BTauReco_EMTauTagIsolation_h
#define BTauReco_EMTauTagIsolation_h
//
// \class EMIsolatedTauTagInfo
//


#include "DataFormats/BTauReco/interface/JetCrystalsAssociation.h"
#include "DataFormats/BTauReco/interface/EMIsolatedTauTagInfoFwd.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/LorentzVectorFwd.h"
//Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"


namespace reco { 

  class EMIsolatedTauTagInfo{

  public:
    //default constructor
     EMIsolatedTauTagInfo():m_discriminator(0), m_jetCrystalsAssociation() {}


    EMIsolatedTauTagInfo(double discriminator, JetCrystalsAssociationRef jetCrystals) 
      {    
	m_discriminator = discriminator;
	m_jetCrystalsAssociation  = jetCrystals;
	
    }
    //destructor
    virtual ~EMIsolatedTauTagInfo() {};
virtual EMIsolatedTauTagInfo* clone() const { return new EMIsolatedTauTagInfo( * this ); }
    //get the jet from the jetTag
    const Jet & jet() const { *m_jetCrystalsAssociation->key;  }

        const edm::RefVector<LorentzVectorCollection> & lorentzVectorRecHits() const { return m_jetCrystalsAssociation->val; } 

    const JetCrystalsAssociationRef& jcaRef() const { return m_jetCrystalsAssociation; }

    //default discriminator: returns the value of the discriminator computed with the parameters taken from the cfg file in the EDProducer
  double discriminator() const { 
    return m_discriminator; 
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

    

  
  private:

    double m_discriminator; //Default discriminator assigned in the EDProducer 
    JetCrystalsAssociationRef m_jetCrystalsAssociation;
  };
}

#endif
