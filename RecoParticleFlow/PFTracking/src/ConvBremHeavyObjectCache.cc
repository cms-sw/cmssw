#include "RecoParticleFlow/PFTracking/interface/ConvBremHeavyObjectCache.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodBDT.h"


namespace convbremhelpers {
  HeavyObjectCache::HeavyObjectCache(const edm::ParameterSet& conf) {

    pfcalib_.reset( new PFEnergyCalibration() );

    const bool useConvBremFinder_ = conf.getParameter<bool>("useConvBremFinder");

    if(useConvBremFinder_) {
      const std::string& mvaWeightFileConvBremBarrelLowPt  = 
        conf.getParameter<std::string>("pf_convBremFinderID_mvaWeightFileBarrelLowPt");
      const std::string mvaWeightFileConvBremBarrelHighPt  = 
        conf.getParameter<std::string>("pf_convBremFinderID_mvaWeightFileBarrelHighPt");
      const std::string mvaWeightFileConvBremEndcapsLowPt  = 
        conf.getParameter<std::string>("pf_convBremFinderID_mvaWeightFileEndcapsLowPt");
      const std::string mvaWeightFileConvBremEndcapsHighPt = 
        conf.getParameter<std::string>("pf_convBremFinderID_mvaWeightFileEndcapsHighPt");
  
      const std::string path_mvaWeightFileConvBremBarrelLowPt  = 
        edm::FileInPath( mvaWeightFileConvBremBarrelLowPt.c_str() ).fullPath();
      const std::string path_mvaWeightFileConvBremBarrelHighPt  = 
        edm::FileInPath( mvaWeightFileConvBremBarrelHighPt.c_str() ).fullPath();
      const std::string path_mvaWeightFileConvBremEndcapsLowPt  = 
        edm::FileInPath( mvaWeightFileConvBremEndcapsLowPt.c_str() ).fullPath();
      const std::string path_mvaWeightFileConvBremEndcapsHighPt = 
        edm::FileInPath( mvaWeightFileConvBremEndcapsHighPt.c_str() ).fullPath();
      
      gbrBarrelLowPt_   = setupMVA(path_mvaWeightFileConvBremBarrelLowPt);
      gbrBarrelHighPt_  = setupMVA(path_mvaWeightFileConvBremBarrelHighPt);
      gbrEndcapsLowPt_  = setupMVA(path_mvaWeightFileConvBremEndcapsLowPt);
      gbrEndcapsHighPt_ = setupMVA(path_mvaWeightFileConvBremEndcapsHighPt);

    }
  }

  std::unique_ptr<const GBRForest> HeavyObjectCache::setupMVA(const std::string& weights) {
    TMVA::Reader reader("!Color:Silent");
    reader.AddVariable("kftrack_secR",&secR);
    reader.AddVariable("kftrack_sTIP",&sTIP);
    reader.AddVariable("kftrack_nHITS1",&nHITS1);
    reader.AddVariable("kftrack_Epout",&Epout);
    reader.AddVariable("kftrack_detaBremKF",&detaBremKF);
    reader.AddVariable("kftrack_ptRatioGsfKF",&ptRatioGsfKF);
    std::unique_ptr<TMVA::IMethod> temp( reader.BookMVA("BDT", weights.c_str()) );
    return std::unique_ptr<const GBRForest>( new GBRForest( dynamic_cast<TMVA::MethodBDT*>( reader.FindMVA("BDT") ) ) );
  }  
}
