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
    const Jet & jet() const { return *m_jetCrystalsAssociation->key;  }

        const edm::RefVector<EMLorentzVectorCollection> & lorentzVectorRecHits() const { return m_jetCrystalsAssociation->val; } 

    const JetCrystalsAssociationRef& jcaRef() const { return m_jetCrystalsAssociation; }

    //default discriminator: returns the value of the discriminator computed with the parameters taken from the cfg file in the EDProducer
  double discriminator() const { 
    return m_discriminator; 
    }
  void setDiscriminator(double discriminator) {m_discriminator =discriminator;}  

    //Method to recompute the discriminator
  double pIsol(float rMax, float rMin)
      {
	  const  edm::RefVector<EMLorentzVectorCollection>  myRecHits = lorentzVectorRecHits();
	  const Jet & myJet = jet(); 
	  double energyRMax= 0.;
	  double energyRMin = 0.;
	  
	  edm::RefVector<EMLorentzVectorCollection>::const_iterator mRH =myRecHits.begin();
	  for(;mRH != myRecHits.end();mRH++)
	      {
		  double delta  = ROOT::Math::VectorUtil::DeltaR((myJet).p4().Vect(), (**mRH));
		  if(delta < rMax) {
		    energyRMax = energyRMax +  (**mRH).pt(); 
		  }
		  if(delta < rMin) {
		    energyRMin = energyRMin +  (**mRH).pt();
		  }
	      }
	  //	  std::cout <<"EnergyMax - EnergyMin" << energyRMax <<" - " << energyRMin<<std::endl;
	  double pIsol = energyRMax - energyRMin;
	  return pIsol;

      }
    double discriminator(float rMax, float rMin, float pIsolCut)
      {
	double newDiscriminator_ =0;
	double pIsol_ = pIsol(rMax, rMin);
	if (pIsol_ < pIsolCut) newDiscriminator_ =1.;
				 
	return newDiscriminator_;
	
      }

    

  
  private:

    double m_discriminator; //Default discriminator assigned in the EDProducer 
    JetCrystalsAssociationRef m_jetCrystalsAssociation;
  };
}

#endif
