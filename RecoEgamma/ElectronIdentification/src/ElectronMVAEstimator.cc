#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimator.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "TMVA/Reader.h"
#include "TMVA/MethodBDT.h"
#include "TMVA/MethodCategory.h"

namespace {
  constexpr char ele_mva_name[] = "BDTSimpleCat";

  class ExtraDeliciousHack : public TMVA::MethodCategory {
  public:
    ExtraDeliciousHack( TMVA::DataSetInfo& dsi,
                        const TString& theWeightFile,
                        TDirectory* theTargetDir = NULL ) : 
      TMVA::MethodCategory(dsi,theWeightFile,theTargetDir) {      
    }
    TMVA::IMethod* GetMethod(const std::string& title) const {
      for( unsigned i = 0; i < fMethods.size(); ++i ) {
        std::cout << fMethods[i]->GetName() << std::endl;
      }
      return TMVA::MethodCategory::GetMethod(TString(title.c_str()));
    }  
  };
}

ElectronMVAEstimator::ElectronMVAEstimator():
  cfg_{}
{}

ElectronMVAEstimator::ElectronMVAEstimator(std::string fileName):
  cfg_{} 
{
  TMVA::Reader tmvaReader("!Color:Silent");
  tmvaReader.AddVariable("fbrem",&fbrem);
  tmvaReader.AddVariable("detain", &detain);
  tmvaReader.AddVariable("dphiin", &dphiin);
  tmvaReader.AddVariable("sieie", &sieie);
  tmvaReader.AddVariable("hoe", &hoe);
  tmvaReader.AddVariable("eop", &eop);
  tmvaReader.AddVariable("e1x5e5x5", &e1x5e5x5);
  tmvaReader.AddVariable("eleopout", &eleopout);
  tmvaReader.AddVariable("detaeleout", &detaeleout);
  tmvaReader.AddVariable("kfchi2", &kfchi2);
  tmvaReader.AddVariable("kfhits", &mykfhits);
  tmvaReader.AddVariable("mishits",&mymishits);
  tmvaReader.AddVariable("dist", &absdist);
  tmvaReader.AddVariable("dcot", &absdcot);
  tmvaReader.AddVariable("nvtx", &myNvtx);

  tmvaReader.AddSpectator("eta",&eta);
  tmvaReader.AddSpectator("pt",&pt);
  tmvaReader.AddSpectator("ecalseed",&ecalseed);
  
  // Taken from Daniele (his mail from the 30/11)
  //  tmvaReader.BookMVA("BDTSimpleCat","../Training/weights_Root527b_3Depth_DanVarConvRej_2PtBins_10Pt_800TPrune5_Min100Events_NoBjets_half/TMVA_BDTSimpleCat.weights.xm");
  // training of the 7/12 with Nvtx added
  std::unique_ptr<TMVA::IMethod> temp( tmvaReader.BookMVA(ele_mva_name,fileName.c_str()) );
  gbr.reset(new GBRForest( dynamic_cast<TMVA::MethodBDT*>( tmvaReader.FindMVA(ele_mva_name) ) ) );
}

ElectronMVAEstimator::ElectronMVAEstimator(const Configuration & cfg):cfg_(cfg){
  std::vector<std::string> weightsfiles;
  std::string path_mvaWeightFileEleID;
  for(unsigned ifile=0 ; ifile < cfg_.vweightsfiles.size() ; ++ifile) {
    path_mvaWeightFileEleID = edm::FileInPath ( cfg_.vweightsfiles[ifile].c_str() ).fullPath();
    weightsfiles.push_back(path_mvaWeightFileEleID);
  }
  TMVA::Reader tmvaReader("!Color:Silent");
  tmvaReader.AddVariable("fbrem",&fbrem);
  tmvaReader.AddVariable("detain", &detain);
  tmvaReader.AddVariable("dphiin", &dphiin);
  tmvaReader.AddVariable("sieie", &sieie);
  tmvaReader.AddVariable("hoe", &hoe);
  tmvaReader.AddVariable("eop", &eop);
  tmvaReader.AddVariable("e1x5e5x5", &e1x5e5x5);
  tmvaReader.AddVariable("eleopout", &eleopout);
  tmvaReader.AddVariable("detaeleout", &detaeleout);
  tmvaReader.AddVariable("kfchi2", &kfchi2);
  tmvaReader.AddVariable("kfhits", &mykfhits);
  tmvaReader.AddVariable("mishits",&mymishits);
  tmvaReader.AddVariable("dist", &absdist);
  tmvaReader.AddVariable("dcot", &absdcot);
  tmvaReader.AddVariable("nvtx", &myNvtx);

  tmvaReader.AddSpectator("eta",&eta);
  tmvaReader.AddSpectator("pt",&pt);
  tmvaReader.AddSpectator("ecalseed",&ecalseed);
  
  // Taken from Daniele (his mail from the 30/11)
  //  tmvaReader.BookMVA("BDTSimpleCat","../Training/weights_Root527b_3Depth_DanVarConvRej_2PtBins_10Pt_800TPrune5_Min100Events_NoBjets_half/TMVA_BDTSimpleCat.weights.xm");
  // training of the 7/12 with Nvtx added
  std::cout << "parsing tmva xml" << std::endl;
  std::unique_ptr<TMVA::IMethod> temp( tmvaReader.BookMVA(ele_mva_name,weightsfiles[0]) );
  tmvaReader.Print("V");
  std::cout << "parsed tmva xml" << std::endl;
  std::cout << "converting to GBR" << std::endl;
  std::cout << "mva ptr = " << tmvaReader.FindMVA(ele_mva_name) << std::endl;
  TMVA::MethodCategory* cate = dynamic_cast<TMVA::MethodCategory*>(tmvaReader.FindMVA(ele_mva_name));
  std::cout << "casted to MethodCategory = " << cate << std::endl;
  cate->Print();
  std::cout << cate->DataInfo().GetName() << std::endl;
  std::cout << cate->GetWeightFileName() << std::endl;
  ExtraDeliciousHack hack(cate->DataInfo(),cate->GetWeightFileName(),NULL);
  std::ifstream the_xml_file(cate->GetWeightFileName().Data());
  std::string xml_str((std::istreambuf_iterator<char>(the_xml_file)),
                      std::istreambuf_iterator<char>());
  the_xml_file.close();
  std::cout << xml_str << std::endl;
  std::cout << "made hack!" << std::endl;   
  hack.SetupMethod();
  std::cout << "Setup" << std::endl;
  hack.DeclareCompatibilityOptions();
  std::cout << "options" << std::endl;
  hack.ReadStateFromXMLString(xml_str.c_str());
  std::cout << "state from file" << std::endl;
  hack.CheckSetup();
  std::cout << "checksetup" << std::endl;
  std::cout << "sub ptr = " << hack.GetMethod(std::string("BDT::Category_BDTSimpleCat_10")) << std::endl;
  gbr.reset(new GBRForest( dynamic_cast<TMVA::MethodBDT*>( tmvaReader.FindMVA(ele_mva_name) ) ) );
  std::cout << "converted to GBR" << std::endl;
}

double ElectronMVAEstimator::mva(const reco::GsfElectron& myElectron, int nvertices ) const {
  float vars[18];

  vars[0] = myElectron.fbrem();
  vars[1] = std::abs(myElectron.deltaEtaSuperClusterTrackAtVtx());
  vars[2] = std::abs(myElectron.deltaPhiSuperClusterTrackAtVtx());
  vars[3] = myElectron.sigmaIetaIeta();
  vars[4] = myElectron.hcalOverEcal();
  vars[5] = myElectron.eSuperClusterOverP();
  vars[6] = (myElectron.e5x5()) !=0. ? 1.-(myElectron.e1x5()/myElectron.e5x5()) : -1. ;
  vars[7] = myElectron.eEleClusterOverPout();
  vars[8] = std::abs(myElectron.deltaEtaEleClusterTrackAtCalo());
  
  bool validKF= false;

  reco::TrackRef myTrackRef = myElectron.closestCtfTrackRef();
  validKF = (myTrackRef.isAvailable());
  validKF = (myTrackRef.isNonnull());  

  vars[9] = (validKF) ? myTrackRef->normalizedChi2() : 0 ;
  vars[10] = (validKF) ? myTrackRef->hitPattern().trackerLayersWithMeasurement() : -1.; 
  vars[11] = myElectron.gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
  vars[12] = std::abs(myElectron.convDist());
  vars[13] = std::abs(myElectron.convDcot());
  vars[14] = nvertices;
  vars[15] = myElectron.eta();
  vars[16] = myElectron.pt();
  vars[17] = myElectron.ecalDrivenSeed();
  
  bindVariables(vars);
  double result =  gbr->GetAdaBoostClassifier(vars);
//  
//  std::cout << "fbrem" << vars[0] << std::endl;
//  std::cout << "detain"<< vars[1] << std::endl;
//  std::cout << "dphiin"<< vars[2] << std::endl;
//  std::cout << "sieie"<< vars[3] << std::endl;
//  std::cout << "hoe"<< vars[4] << std::endl;
//  std::cout << "eop"<< vars[5] << std::endl;
//  std::cout << "e1x5e5x5"<< vars[6] << std::endl;
//  std::cout << "eleopout"<< vars[7] << std::endl;
//  std::cout << "detaeleout"<< vars[8] << std::endl;
//  std::cout << "kfchi2"<< vars[9] << std::endl;
//  std::cout << "kfhits"<< vars[10] << std::endl;
//  std::cout << "mishits"<<vars[11] << std::endl;
//  std::cout << "dist"<< vars[12] << std::endl;
//  std::cout << "dcot"<< vars[13] << std::endl;
//  std::cout << "nvtx"<< vars[14] << std::endl;
//  std::cout << "eta"<< vars[15] << std::endl;
//  std::cout << "pt"<< vars[16] << std::endl;
//  std::cout << "ecalseed"<< vars[17] << std::endl;
//
//  std::cout << " MVA " << result << std::endl;
  return result;
}


void ElectronMVAEstimator::bindVariables(float vars[18]) const {
  if(vars[0] < -1.)
    vars[1] = -1.;  
  
  if(vars[1] > 0.06)
    vars[1] = 0.06;
    
  if(vars[2] > 0.6)
    vars[2] = 0.6;
  
  if(vars[5] > 20.)
    vars[5] = 20.;
    
  if(vars[7] > 20.)
    vars[7] = 20;
  
  if(vars[8] > 0.2)
    vars[8] = 0.2;
  
  if(vars[9] < 0.)
    vars[9] = 0.;
  
  if(vars[9] > 15.)
    vars[9] = 15.;
    
  if(vars[6] < -1.)
    vars[6] = -1;

  if(vars[6] > 2.)
    vars[6] = 2.; 
    
  if(vars[12] > 15.)
    vars[12] = 15.;
    
  if(vars[13] > 3.)
    vars[13] = 3.;
}
