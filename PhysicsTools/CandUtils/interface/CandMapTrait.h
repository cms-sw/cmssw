#ifndef PhysicsTools_CandUtils_CandMapTrait_h
#define PhysicsTools_CandUtils_CandMapTrait_h
/* \class reco::helper::CandMapTrait<T>
 *
 * \author Luca Lista, INFN 
 *
 * \version $Id: CandMapTrait.h,v 1.2 2007/10/20 16:00:52 llista Exp $
 *
 */
namespace reco {
  namespace helper {
    template<typename C1, typename C2 = C1>
    struct CandMapTrait {
      typedef edm::AssociationMap<edm::OneToOne<C1, C2> > type;
    };
    
    template<typename C1>
    struct CandMapTrait<C1, CandidateView> {
      typedef edm::AssociationMap<edm::OneToOneGeneric<C1, CandidateView> > type;
    };

    template<typename C2>
    struct CandMapTrait<CandidateView, C2> {
      typedef edm::AssociationMap<edm::OneToOneGeneric<CandidateView, C2> > type;
    };

    template<>
    struct CandMapTrait<CandidateView, CandidateView> {
      typedef edm::AssociationMap<edm::OneToOneGeneric<CandidateView, CandidateView> > type;
    };

    template<typename C>
    struct CandRefTrait{
      typedef edm::Ref<C> ref_type;
      typedef edm::RefProd<C> refProd_type;
    };

    template<typename T>
    struct CandRefTrait<edm::View<T> >{
      typedef edm::RefToBase<T> ref_type;
      typedef edm::RefToBaseProd<T> refProd_type;
    };
  }
}

#endif

