#include "RecoEgamma/ElectronIdentification/interface/SoftElectronMVAEstimator.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

SoftElectronMVAEstimator::SoftElectronMVAEstimator(const Configuration & cfg):cfg_(cfg){
  std::vector<std::string> weightsfiles;
  std::string path_mvaWeightFileEleID;
  for(unsigned ifile=0 ; ifile < cfg_.vweightsfiles.size() ; ++ifile) {
    path_mvaWeightFileEleID = edm::FileInPath ( cfg_.vweightsfiles[ifile].c_str() ).fullPath();
    weightsfiles.push_back(path_mvaWeightFileEleID);
  }

  for (unsigned int i=0;i<fmvaReader.size(); ++i) {
    if (fmvaReader[i]) delete fmvaReader[i];
  }
  fmvaReader.clear();

  //initialize
  //Define expected number of bins
  UInt_t ExpectedNBins = 1;

  //Check number of weight files given
  if (ExpectedNBins != cfg_.vweightsfiles.size() ) {
    std::cout << "Error: Expected Number of bins = " << ExpectedNBins << " does not equal to weightsfiles.size() = "
              << cfg_.vweightsfiles.size() << std::endl;

    assert(ExpectedNBins == cfg_.vweightsfiles.size());
  }


  for (unsigned int i=0;i<ExpectedNBins; ++i) {
    tmvaReader_ = new TMVA::Reader("!Color:Silent");
    tmvaReader_->AddVariable("fbrem",                   &fbrem);
    tmvaReader_->AddVariable("EtotOvePin",                      &EtotOvePin);
    tmvaReader_->AddVariable("EClusOverPout",                   &eleEoPout);
    tmvaReader_->AddVariable("EBremOverDeltaP",                 &EBremOverDeltaP);
    tmvaReader_->AddVariable("logSigmaEtaEta",                  &logSigmaEtaEta);
    tmvaReader_->AddVariable("DeltaEtaTrackEcalSeed",           &DeltaEtaTrackEcalSeed);
    tmvaReader_->AddVariable("HoE",                             &HoE);
    tmvaReader_->AddVariable("gsfchi2",                         &gsfchi2);
    tmvaReader_->AddVariable("kfchi2",                          &kfchi2);
    tmvaReader_->AddVariable("kfhits",                          &kfhits);
    tmvaReader_->AddVariable("SigmaPtOverPt",                   &SigmaPtOverPt);
    tmvaReader_->AddVariable("deta",                            &deta);
    tmvaReader_->AddVariable("dphi",                            &dphi);
    tmvaReader_->AddVariable("detacalo",                        &detacalo);
    tmvaReader_->AddVariable("see",                             &see);
    tmvaReader_->AddVariable("spp",                             &spp); 	
    tmvaReader_->AddVariable("R9",                             	&R9);
    tmvaReader_->AddVariable("etawidth",                        &etawidth);
    tmvaReader_->AddVariable("phiwidth",                        &phiwidth);
    tmvaReader_->AddVariable("e1x5e5x5",                        &OneMinusE1x5E5x5);
    tmvaReader_->AddVariable("IoEmIoP",				&IoEmIoP);
    tmvaReader_->AddVariable("PreShowerOverRaw",		&PreShowerOverRaw);
    tmvaReader_->AddVariable("nPV",                		&nPV);

    tmvaReader_->AddVariable( "pt",                            &pt);
    tmvaReader_->AddVariable( "eta",                           &eta);

    tmvaReader_->AddSpectator( "pt",                            &pt);
    tmvaReader_->AddSpectator( "eta",                           &eta);

//    tmvaReader_->AddSpectator( "nPV",                           &nPV);

    // Taken from Daniele (his mail from the 30/11)
    //  tmvaReader_->BookMVA("BDTSimpleCat","../Training/weights_Root527b_3Depth_DanVarConvRej_2PtBins_10Pt_800TPrune5_Min100Events_NoBjets_half/TMVA_BDTSimpleCat.weights.xm");
    // training of the 7/12 with Nvtx added
    tmvaReader_->BookMVA("BDT",weightsfiles[i]);
    fmvaReader.push_back(tmvaReader_);
//    delete tmvaReader_;
  }

}


SoftElectronMVAEstimator::~SoftElectronMVAEstimator()
{
  for (unsigned int i=0;i<fmvaReader.size(); ++i) {
    if (fmvaReader[i]) delete fmvaReader[i];
  }
}


UInt_t SoftElectronMVAEstimator::GetMVABin(int pu, double eta, double pt) const {

    //Default is to return the first bin
    unsigned int bin = 0;

 bool ptrange[3],etarange[3],purange[2];
        ptrange[0]=pt > 2 && pt < 5;
        ptrange[1]=pt > 5 && pt < 10;
        ptrange[2]=pt > 10;
        etarange[0]=fabs(eta) < 0.8;
        etarange[1]=fabs(eta) > 0.8 && fabs(eta) <1.4;
        etarange[2]=fabs(eta) > 1.4;
        purange[0]=nPV<=20;
        purange[1]=nPV>20;

        int index=0;
        for(int kPU=0;kPU<2;kPU++)
        for(int kETA=0;kETA<3;kETA++)
        for(int kPT=0;kPT<3;kPT++){
                if (purange[kPU] && ptrange[kPT] && etarange[kETA]) bin=index;
                index++;
        } 
  return bin;
}



double SoftElectronMVAEstimator::mva(const reco::GsfElectron& myElectron,const edm::Event & evt)  {

 edm::Handle<reco::VertexCollection> FullprimaryVertexCollection;
 evt.getByLabel("offlinePrimaryVertices", FullprimaryVertexCollection);
 const reco::VertexCollection pvc = *(FullprimaryVertexCollection.product());
 
  fbrem                 	=myElectron.fbrem();
  EtotOvePin            	=myElectron.eSuperClusterOverP();
  eleEoPout             	=myElectron.eEleClusterOverPout();
  float etot                    =myElectron.eSuperClusterOverP()*myElectron.trackMomentumAtVtx().R();
  float eEcal                   =myElectron.eEleClusterOverPout()*myElectron.trackMomentumAtEleClus().R();
  float dP                      =myElectron.trackMomentumAtVtx().R()-myElectron.trackMomentumAtEleClus().R();
  EBremOverDeltaP       	=(etot-eEcal)/dP;
  logSigmaEtaEta        	=log(myElectron.sigmaEtaEta());
  DeltaEtaTrackEcalSeed 	=myElectron.deltaEtaEleClusterTrackAtCalo();
  HoE                   	=myElectron.hcalOverEcalBc();

  bool validKF= false;
  reco::TrackRef myTrackRef     = myElectron.closestCtfTrackRef();
  validKF                       = (myTrackRef.isAvailable() && myTrackRef.isNonnull());
  kfchi2                	=(validKF) ? myTrackRef->normalizedChi2() : 0 ;
  kfhits                	=(validKF) ? myTrackRef->hitPattern().trackerLayersWithMeasurement() : -1. ;
  gsfchi2               	=myElectron.gsfTrack()->normalizedChi2();
  SigmaPtOverPt         	=myElectron.gsfTrack().get()->ptModeError()/myElectron.gsfTrack().get()->ptMode() ;
  deta                  	=myElectron.deltaEtaSuperClusterTrackAtVtx();
  dphi                  	=myElectron.deltaPhiSuperClusterTrackAtVtx();
  detacalo              	=myElectron.deltaEtaSeedClusterTrackAtCalo();
  see                   	=myElectron.sigmaIetaIeta();
  spp				=myElectron.sigmaIphiIphi();
  R9                    =myElectron.r9();
  IoEmIoP               =  (1.0/myElectron.ecalEnergy()) - (1.0 / myElectron.p());
  etawidth              	=myElectron.superCluster()->etaWidth();
  phiwidth              	=myElectron.superCluster()->phiWidth();
  OneMinusE1x5E5x5      	=(myElectron.e5x5()) !=0. ? 1.-(myElectron.e1x5()/myElectron.e5x5()) : -1. ;
  pt                    	=myElectron.pt();
  eta                   	=myElectron.eta();
  nPV=pvc.size();
  PreShowerOverRaw=myElectron.superCluster()->preshowerEnergy() / myElectron.superCluster()->rawEnergy();

/*
  std::cout<<"fbrem "<<fbrem<<std::endl;
  std::cout<<"EtotOvePin "<<EtotOvePin<<std::endl;
  std::cout<<"eleEoPout "<<eleEoPout<<std::endl;
  std::cout<<"EBremOverDeltaP "<<EBremOverDeltaP<<std::endl;
  std::cout<<"logSigmaEtaEta "<<logSigmaEtaEta<<std::endl;
  std::cout<<"DeltaEtaTrackEcalSeed "<<DeltaEtaTrackEcalSeed<<std::endl;
  std::cout<<"HoE "<<HoE<<std::endl;
  std::cout<<"kfchi2 "<<kfchi2<<std::endl;
  std::cout<<"kfhits "<<kfhits<<std::endl;
  std::cout<<"gsfchi2 "<<gsfchi2<<std::endl;
  std::cout<<"SigmaPtOverPt "<<SigmaPtOverPt<<std::endl;
  std::cout<<"deta "<<deta<<std::endl;
  std::cout<<"dphi "<<dphi<<std::endl;
  std::cout<<"detacalo "<<detacalo<<std::endl;
  std::cout<<"see "<<see<<std::endl;
  std::cout<< "spp "             <<          spp<< std::endl;
  std::cout<< "R9 "             <<         R9<< std::endl;
  std::cout<< "IoEmIoP "        <<         IoEmIoP<< std::endl;
  std::cout<<"etawidth "<<etawidth<<std::endl;
  std::cout<<"phiwidth "<<phiwidth<<std::endl;
  std::cout<<"OneMinusE1x5E5x5 "<<OneMinusE1x5E5x5<<std::endl;
  std::cout<<"PreShowerOverRaw "<<PreShowerOverRaw<<std::endl;
*/
  bindVariables();
//  double result= fmvaReader[GetMVABin(nPV,eta,pt)]->EvaluateMVA("BDT");
  double result= fmvaReader[0]->EvaluateMVA("BDT");
//  double result =  tmvaReader_->EvaluateMVA("BDT");
  return result;
}


void SoftElectronMVAEstimator::bindVariables() {
  if(fbrem < -1.)
    fbrem = -1.;

  deta = fabs(deta);
  if(deta > 0.06)
    deta = 0.06;


  dphi = fabs(dphi);
  if(dphi > 0.6)
    dphi = 0.6;


  if(EoP > 20.)
    EoP = 20.;

  if(eleEoPout > 20.)
    eleEoPout = 20.;


  detacalo = fabs(detacalo);
  if(detacalo > 0.2)
    detacalo = 0.2;

  if(OneMinusE1x5E5x5 < -1.)
    OneMinusE1x5E5x5 = -1;

  if(OneMinusE1x5E5x5 > 2.)
    OneMinusE1x5E5x5 = 2.;



  if(gsfchi2 > 200.)
    gsfchi2 = 200;


  if(kfchi2 > 10.)
    kfchi2 = 10.;

}
