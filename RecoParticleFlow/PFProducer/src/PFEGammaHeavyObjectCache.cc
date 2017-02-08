#include "RecoParticleFlow/PFProducer/interface/PFEGammaHeavyObjectCache.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "TMVA/MethodBDT.h"
#include "TMVA/Reader.h"

namespace pfEGHelpers {
  HeavyObjectCache::HeavyObjectCache(const edm::ParameterSet& conf) {
    {
      const edm::FileInPath& wfile = conf.getParameter<edm::FileInPath>("pf_electronID_mvaWeightFile");
      // Set the tmva reader for electrons
      TMVA::Reader tmvaReaderEle_("!Color:Silent");
      tmvaReaderEle_.AddVariable("lnPt_gsf",&lnPt_gsf);
      tmvaReaderEle_.AddVariable("Eta_gsf",&Eta_gsf);
      tmvaReaderEle_.AddVariable("dPtOverPt_gsf",&dPtOverPt_gsf);
      tmvaReaderEle_.AddVariable("DPtOverPt_gsf",&DPtOverPt_gsf);
      //tmvaReaderEle_.AddVariable("nhit_gsf",&nhit_gsf);
      tmvaReaderEle_.AddVariable("chi2_gsf",&chi2_gsf);
      //tmvaReaderEle_.AddVariable("DPtOverPt_kf",&DPtOverPt_kf);
      tmvaReaderEle_.AddVariable("nhit_kf",&nhit_kf);
      tmvaReaderEle_.AddVariable("chi2_kf",&chi2_kf);
      tmvaReaderEle_.AddVariable("EtotPinMode",&EtotPinMode);
      tmvaReaderEle_.AddVariable("EGsfPoutMode",&EGsfPoutMode);
      tmvaReaderEle_.AddVariable("EtotBremPinPoutMode",&EtotBremPinPoutMode);
      tmvaReaderEle_.AddVariable("DEtaGsfEcalClust",&DEtaGsfEcalClust);
      tmvaReaderEle_.AddVariable("SigmaEtaEta",&SigmaEtaEta);
      tmvaReaderEle_.AddVariable("HOverHE",&HOverHE);
      //   tmvaReaderEle_.AddVariable("HOverPin",&HOverPin);
      tmvaReaderEle_.AddVariable("lateBrem",&lateBrem);
      tmvaReaderEle_.AddVariable("firstBrem",&firstBrem);
      std::unique_ptr<TMVA::IMethod> temp( tmvaReaderEle_.BookMVA("BDT", wfile.fullPath().c_str()) );
      gbrEle_.reset( new GBRForest( dynamic_cast<TMVA::MethodBDT*>( tmvaReaderEle_.FindMVA("BDT") ) ) );
    }
    {
      const edm::FileInPath& wfile = conf.getParameter<edm::FileInPath>("pf_convID_mvaWeightFile");
      //Book MVA (single leg)
      TMVA::Reader tmvaReader_("!Color:Silent");  
      tmvaReader_.AddVariable("del_phi",&del_phi);  
      tmvaReader_.AddVariable("nlayers", &nlayers);  
      tmvaReader_.AddVariable("chi2",&chi2);  
      tmvaReader_.AddVariable("EoverPt",&EoverPt);  
      tmvaReader_.AddVariable("HoverPt",&HoverPt);  
      tmvaReader_.AddVariable("track_pt", &track_pt);  
      tmvaReader_.AddVariable("STIP",&STIP);  
      tmvaReader_.AddVariable("nlost", &nlost);  
      std::unique_ptr<TMVA::IMethod> temp( tmvaReader_.BookMVA("BDT", wfile.fullPath().c_str()) );
      gbrSingleLeg_.reset( new GBRForest( dynamic_cast<TMVA::MethodBDT*>( tmvaReader_.FindMVA("BDT") ) ) );
    }
  }
}
