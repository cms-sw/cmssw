/* class PFRecoTauDiscriminationAgainstElectronMVA
 * created : Nov 23 2011,
 * revised : ,
 * Authorss : Lorenzo Bianchini (LLR Ecole-Polytechnique)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA.h"

#include <string>

using namespace reco;

class PFRecoTauDiscriminationAgainstElectronMVA : public PFTauDiscriminationProducerBase  {
   public:
      explicit PFRecoTauDiscriminationAgainstElectronMVA(const edm::ParameterSet& iConfig):PFTauDiscriminationProducerBase(iConfig) {

 	 method_                           = iConfig.getParameter<std::string>("method");
	 inputFileName1prongBL_            = iConfig.getParameter<edm::FileInPath>("inputFileName1prongBL");
	 inputFileName1prongStripsWgsfBL_  = iConfig.getParameter<edm::FileInPath>("inputFileName1prongStripsWgsfBL");
	 inputFileName1prongStripsWOgsfBL_ = iConfig.getParameter<edm::FileInPath>("inputFileName1prongStripsWOgsfBL");
	 inputFileName1prongEC_            = iConfig.getParameter<edm::FileInPath>("inputFileName1prongEC");
	 inputFileName1prongStripsWgsfEC_  = iConfig.getParameter<edm::FileInPath>("inputFileName1prongStripsWgsfEC");
	 inputFileName1prongStripsWOgsfEC_ = iConfig.getParameter<edm::FileInPath>("inputFileName1prongStripsWOgsfEC");

	 returnMVA_                 = iConfig.getParameter<bool>("returnMVA");
	 minMVA1prongBL_            = iConfig.getParameter<double>("minMVA1prongBL");
	 minMVA1prongStripsWgsfBL_  = iConfig.getParameter<double>("minMVA1prongStripsWgsfBL");
	 minMVA1prongStripsWOgsfBL_ = iConfig.getParameter<double>("minMVA1prongStripsWOgsfBL");
	 minMVA1prongEC_            = iConfig.getParameter<double>("minMVA1prongEC");
	 minMVA1prongStripsWgsfEC_  = iConfig.getParameter<double>("minMVA1prongStripsWgsfEC");
	 minMVA1prongStripsWOgsfEC_ = iConfig.getParameter<double>("minMVA1prongStripsWOgsfEC");

	 mva_ = new AntiElectronIDMVA();
	 mva_->Initialize(method_,
			  inputFileName1prongBL_.fullPath().data(),
			  inputFileName1prongStripsWgsfBL_.fullPath().data(),
			  inputFileName1prongStripsWOgsfBL_.fullPath().data(),
			  inputFileName1prongEC_.fullPath().data(),
			  inputFileName1prongStripsWgsfEC_.fullPath().data(),
			  inputFileName1prongStripsWOgsfEC_.fullPath().data()
			  );


      }

      double discriminate(const PFTauRef& thePFTauRef);

      ~PFRecoTauDiscriminationAgainstElectronMVA() { delete mva_; }

  private:

      std::string method_ ;
      edm::FileInPath inputFileName1prongBL_;
      edm::FileInPath inputFileName1prongStripsWgsfBL_;
      edm::FileInPath inputFileName1prongStripsWOgsfBL_;
      edm::FileInPath inputFileName1prongEC_;
      edm::FileInPath inputFileName1prongStripsWgsfEC_;
      edm::FileInPath inputFileName1prongStripsWOgsfEC_;
      AntiElectronIDMVA* mva_;
      bool returnMVA_ ;
      double minMVA1prongBL_ ;
      double minMVA1prongStripsWgsfBL_ ;
      double minMVA1prongStripsWOgsfBL_ ;
      double minMVA1prongEC_ ;
      double minMVA1prongStripsWgsfEC_ ;
      double minMVA1prongStripsWOgsfEC_;
};

double PFRecoTauDiscriminationAgainstElectronMVA::discriminate(const PFTauRef& thePFTauRef)
{
  double mva          = -1.0;
  double workingPoint =  0.0;

  if( (*thePFTauRef).leadPFChargedHadrCand().isNonnull() ) {

    mva = mva_->MVAValue( thePFTauRef );

    workingPoint =
      ((*thePFTauRef).signalPFChargedHadrCands().size()==3 ||
       (fabs((*thePFTauRef).eta())<1.5 && ((*thePFTauRef).signalPFGammaCands().size())<=0 && mva > minMVA1prongBL_) ||
       (fabs((*thePFTauRef).eta())<1.5 && ((*thePFTauRef).signalPFGammaCands().size())>0  && (((*thePFTauRef).leadPFChargedHadrCand())->gsfTrackRef()).isNonnull()>0.5 && mva > minMVA1prongStripsWgsfBL_) ||
       (fabs((*thePFTauRef).eta())<1.5 && ((*thePFTauRef).signalPFGammaCands().size())>0  && (((*thePFTauRef).leadPFChargedHadrCand())->gsfTrackRef()).isNonnull()<0.5 && mva > minMVA1prongStripsWOgsfBL_) ||
       (fabs((*thePFTauRef).eta())>1.5 && ((*thePFTauRef).signalPFGammaCands().size())<=0 && mva > minMVA1prongEC_) ||
       (fabs((*thePFTauRef).eta())>1.5 && ((*thePFTauRef).signalPFGammaCands().size())>0  && (((*thePFTauRef).leadPFChargedHadrCand())->gsfTrackRef()).isNonnull()>0.5 && mva > minMVA1prongStripsWgsfEC_) ||
       (fabs((*thePFTauRef).eta())>1.5 && ((*thePFTauRef).signalPFGammaCands().size())>0  && (((*thePFTauRef).leadPFChargedHadrCand())->gsfTrackRef()).isNonnull()<0.5 && mva > minMVA1prongStripsWOgsfEC_)) ? 1.0 : 0.0;

  }

  return ( returnMVA_ ? mva : workingPoint);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstElectronMVA);
