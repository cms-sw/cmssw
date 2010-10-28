#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

/* class TauDiscriminationByStringCut
 * created : Nov 9 2009
 * revised : Tue Nov 10 14:44:40 PDT 2009
 * author : Christian Veelken (veelken@fnal.gov ; UC Davis)
 */

template<class TauType, class TauDiscriminator>
class TauDiscriminationByStringCut :
    public TauDiscriminationProducerBase<TauType, TauDiscriminator> {
   public:
      explicit TauDiscriminationByStringCut(const edm::ParameterSet& iConfig)
          : TauDiscriminationProducerBase<TauType, TauDiscriminator>(iConfig),
          cut_(new StringCutObjectSelector<TauType>(
            iConfig.getParameter<std::string>("cut"))) {
        cutFailValue_ = ( iConfig.exists("cutFailValue") ) ?
            iConfig.getParameter<double>("cutFailValue") : 0.;
        cutPassValue_ = ( iConfig.exists("cutPassValue") ) ?
            iConfig.getParameter<double>("cutPassValue") : 1.;
        this->prediscriminantFailValue_ = cutFailValue_;
      }

      typedef std::vector<TauType> TauCollection;
      typedef edm::Ref<TauCollection> TauRef;

      double discriminate(const TauRef& tau) {
        // StringCutObjectSelector::operator() returns true if tau passes cut
        return ( (*cut_)(*tau) ) ? cutPassValue_ : cutFailValue_;
      }

   private:
      std::auto_ptr<StringCutObjectSelector<TauType> > cut_;
      double cutFailValue_;
      double cutPassValue_;
};

typedef TauDiscriminationByStringCut<reco::PFTau, reco::PFTauDiscriminator>
  PFRecoTauDiscriminationByStringCut;
typedef TauDiscriminationByStringCut<reco::CaloTau, reco::CaloTauDiscriminator>
  CaloRecoTauDiscriminationByStringCut;

DEFINE_FWK_MODULE(PFRecoTauDiscriminationByStringCut);
DEFINE_FWK_MODULE(CaloRecoTauDiscriminationByStringCut);
