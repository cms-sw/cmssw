#ifndef TopObjects_TopObject_h
#define TopObjects_TopObject_h
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"

using namespace std;
using namespace reco;

typedef PixelMatchGsfElectron electronType;
typedef Muon muonType;
typedef CaloJet jetType ;
typedef CaloMET metType ;


template <class ObjectType> class TopObject : public ObjectType
{
   
   public:
      TopObject();
      TopObject(ObjectType);
      virtual ~TopObject();
      
      void    setResET(double);
      void    setResEta(double);
      void    setResPhi(double);
      void    setResD(double);
      void    setResPinv(double);
      void    setResTheta(double); 
      
      double  getResET() const;
      double  getResEta() const;
      double  getResPhi() const;
      double  getResD() const;
      double  getResPinv() const;
      double  getResTheta() const;
      
   protected:
      double resET;
      double resEta;
      double resPhi;
      double resD;
      double resPinv;
      double resTheta;
};


template <class ObjectType> TopObject<ObjectType>::TopObject(){ }
template <class ObjectType> TopObject<ObjectType>::TopObject(ObjectType o): ObjectType(o){ }
template <class ObjectType> TopObject<ObjectType>::~TopObject(){ }

template <class ObjectType> void TopObject<ObjectType>::setResET(double et)       { resET = et; }
template <class ObjectType> void TopObject<ObjectType>::setResEta(double eta)     { resEta = eta; }
template <class ObjectType> void TopObject<ObjectType>::setResPhi(double phi)     { resPhi = phi; }
template <class ObjectType> void TopObject<ObjectType>::setResD(double d)         { resD = d; }
template <class ObjectType> void TopObject<ObjectType>::setResPinv(double pinv)   { resPinv = pinv; }
template <class ObjectType> void TopObject<ObjectType>::setResTheta(double theta) { resTheta = theta; }

template <class ObjectType> double TopObject<ObjectType>::getResET() const 	 { return resET; }
template <class ObjectType> double TopObject<ObjectType>::getResEta() const 	 { return resEta; }
template <class ObjectType> double TopObject<ObjectType>::getResPhi() const 	 { return resPhi; }
template <class ObjectType> double TopObject<ObjectType>::getResD() const 	 { return resD; }
template <class ObjectType> double TopObject<ObjectType>::getResPinv() const 	 { return resPinv; }
template <class ObjectType> double TopObject<ObjectType>::getResTheta() const 	 { return resTheta; }

typedef TopObject<electronType> TopElectron;
typedef TopObject<muonType> TopMuon;
typedef TopObject<jetType> TopJet;
typedef TopObject<metType> TopMET;
typedef TopObject<Particle> TopParticle;

#endif
