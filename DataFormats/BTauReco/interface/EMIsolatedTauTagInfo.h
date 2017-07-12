#ifndef DataFormats_BTauReco_EMIsolatedTauTagInfo_h
#define DataFormats_BTauReco_EMIsolatedTauTagInfo_h
//
// \class EMIsolatedTauTagInfo
//

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/LorentzVectorFwd.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/JetCrystalsAssociation.h"

namespace reco {

  class EMIsolatedTauTagInfo : public BaseTagInfo {

  public:
    //default constructor
    EMIsolatedTauTagInfo() : m_discriminator(0), m_jetCrystalsAssociation() { }

    EMIsolatedTauTagInfo(double discriminator, const JetCrystalsAssociationRef & jetCrystals) :
      m_discriminator( discriminator ),
      m_jetCrystalsAssociation( jetCrystals ) { }

    // destructor
    virtual ~EMIsolatedTauTagInfo() { };
    virtual EMIsolatedTauTagInfo* clone() const { return new EMIsolatedTauTagInfo( * this ); }

    // get the jet from the jetTag
    virtual edm::RefToBase<Jet>       jet()                  const { return m_jetCrystalsAssociation->first;  }
    virtual EMLorentzVectorRefVector  lorentzVectorRecHits() const { return m_jetCrystalsAssociation->second; }
    const JetCrystalsAssociationRef & jcaRef()               const { return m_jetCrystalsAssociation; }

    // default discriminator: returns the value of the discriminator computed with the parameters taken from the cfg file in the EDProducer
    float discriminator() const {
        return m_discriminator;
    }

    void setDiscriminator(double discriminator) { m_discriminator = discriminator; }

    //Method to recompute the discriminator
    double pIsol(float rMax, float rMin) const
    {
        const EMLorentzVectorRefVector & myRecHits = m_jetCrystalsAssociation->second;
        const Jet & myJet = * m_jetCrystalsAssociation->first;
        double energyRMax= 0.;
        double energyRMin = 0.;

        for (auto && myRecHit : myRecHits)
        {
            double delta  = ROOT::Math::VectorUtil::DeltaR((myJet).p4().Vect(), (*myRecHit));
            if (delta < rMax) {
                energyRMax += (*myRecHit).pt();
            }
            if (delta < rMin) {
                energyRMin += (*myRecHit).pt();
            }
        }
        double pIsol = energyRMax - energyRMin;
        return pIsol;
    }

    float discriminator(float rMax, float rMin, float pIsolCut) const
    {
        double newDiscriminator_ = 0;
        double pIsol_ = pIsol(rMax, rMin);
        if (pIsol_ < pIsolCut) newDiscriminator_ = 1.;

        return newDiscriminator_;
    }

  private:

    float m_discriminator; //Default discriminator assigned in the EDProducer
    JetCrystalsAssociationRef m_jetCrystalsAssociation;
  };

  DECLARE_EDM_REFS( EMIsolatedTauTagInfo )

}

#endif // DataFormats_BTauReco_EMIsolatedTauTagInfo_h
