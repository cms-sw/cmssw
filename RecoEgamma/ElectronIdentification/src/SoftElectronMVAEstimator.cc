#include "RecoEgamma/ElectronIdentification/interface/SoftElectronMVAEstimator.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodBDT.h"

SoftElectronMVAEstimator::SoftElectronMVAEstimator(const Configuration & cfg):cfg_(cfg){
  std::vector<std::string> weightsfiles;
  std::string path_mvaWeightFileEleID;
  for(unsigned ifile=0 ; ifile < cfg_.vweightsfiles.size() ; ++ifile) {
    path_mvaWeightFileEleID = edm::FileInPath ( cfg_.vweightsfiles[ifile].c_str() ).fullPath();
    weightsfiles.push_back(path_mvaWeightFileEleID);
  }
  
  //initialize
  //Define expected number of bins

  //Check number of weight files given
  if (ExpectedNBins != cfg_.vweightsfiles.size() ) {
    std::cout << "Error: Expected Number of bins = " << ExpectedNBins << " does not equal to weightsfiles.size() = "
              << cfg_.vweightsfiles.size() << std::endl;

    assert(ExpectedNBins == cfg_.vweightsfiles.size());
  }


  for (unsigned int i=0;i<ExpectedNBins; ++i) {
    TMVA::Reader tmvaReader("!Color:Silent");
    tmvaReader.AddVariable("fbrem",                   &fbrem);
    tmvaReader.AddVariable("EtotOvePin",                      &EtotOvePin);
    tmvaReader.AddVariable("EClusOverPout",                   &eleEoPout);
    tmvaReader.AddVariable("EBremOverDeltaP",                 &EBremOverDeltaP);
    tmvaReader.AddVariable("logSigmaEtaEta",                  &logSigmaEtaEta);
    tmvaReader.AddVariable("DeltaEtaTrackEcalSeed",           &DeltaEtaTrackEcalSeed);
    tmvaReader.AddVariable("HoE",                             &HoE);
    tmvaReader.AddVariable("gsfchi2",                         &gsfchi2);
    tmvaReader.AddVariable("kfchi2",                          &kfchi2);
    tmvaReader.AddVariable("kfhits",                          &kfhits);
    tmvaReader.AddVariable("SigmaPtOverPt",                   &SigmaPtOverPt);
    tmvaReader.AddVariable("deta",                            &deta);
    tmvaReader.AddVariable("dphi",                            &dphi);
    tmvaReader.AddVariable("detacalo",                        &detacalo);
    tmvaReader.AddVariable("see",                             &see);
    tmvaReader.AddVariable("spp",                             &spp); 	
    tmvaReader.AddVariable("R9",                             	&R9);
    tmvaReader.AddVariable("etawidth",                        &etawidth);
    tmvaReader.AddVariable("phiwidth",                        &phiwidth);
    tmvaReader.AddVariable("e1x5e5x5",                        &OneMinusE1x5E5x5);
    tmvaReader.AddVariable("IoEmIoP",				&IoEmIoP);
    tmvaReader.AddVariable("PreShowerOverRaw",		&PreShowerOverRaw);
    tmvaReader.AddVariable("nPV",                		&nPV);

    tmvaReader.AddVariable( "pt",                            &pt);
    tmvaReader.AddVariable( "eta",                           &eta);

    tmvaReader.AddSpectator( "pt",                            &pt);
    tmvaReader.AddSpectator( "eta",                           &eta);
    
    // Taken from Daniele (his mail from the 30/11)    
    // training of the 7/12 with Nvtx added
    std::unique_ptr<TMVA::IMethod> temp( tmvaReader.BookMVA("BDT",weightsfiles[i]) );
    gbr[i].reset( new GBRForest( dynamic_cast<TMVA::MethodBDT*>( tmvaReader.FindMVA("BDT") ) ) );
  }
}


SoftElectronMVAEstimator::~SoftElectronMVAEstimator()
{ }


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



double SoftElectronMVAEstimator::mva(const reco::GsfElectron& myElectron,
                                     const reco::VertexCollection& pvc) const {
  float vars[25];

  vars[0]  = myElectron.fbrem();// fbrem
  vars[1]  = myElectron.eSuperClusterOverP(); //EtotOvePin
  vars[2]   = myElectron.eEleClusterOverPout(); //eleEoPout 
  
  float etot  = myElectron.eSuperClusterOverP()*myElectron.trackMomentumAtVtx().R();
  float eEcal = myElectron.eEleClusterOverPout()*myElectron.trackMomentumAtEleClus().R();
  float dP    = myElectron.trackMomentumAtVtx().R()-myElectron.trackMomentumAtEleClus().R();
  vars[3]  = (etot-eEcal)/dP; //EBremOverDeltaP
  vars[4]  = std::log(myElectron.sigmaEtaEta()); //logSigmaEtaEta
  vars[5]  = myElectron.deltaEtaEleClusterTrackAtCalo(); //DeltaEtaTrackEcalSeed
  vars[6]  = myElectron.hcalOverEcalBc(); //HoE  

  bool validKF= false;
  reco::TrackRef myTrackRef     = myElectron.closestCtfTrackRef();
  validKF                       = (myTrackRef.isAvailable() && myTrackRef.isNonnull());
  vars[7]  = myElectron.gsfTrack()->normalizedChi2(); //gsfchi2
  vars[8]  = (validKF) ? myTrackRef->normalizedChi2() : 0 ; //kfchi2
  vars[9]  = (validKF) ? myTrackRef->hitPattern().trackerLayersWithMeasurement() : -1. ; //kfhits
  
  vars[10] = myElectron.gsfTrack().get()->ptModeError()/myElectron.gsfTrack().get()->ptMode() ; //SigmaPtOverPt 
  vars[11]  = myElectron.deltaEtaSuperClusterTrackAtVtx(); //deta
  vars[12]  = myElectron.deltaPhiSuperClusterTrackAtVtx(); //dphi
  vars[13]  = myElectron.deltaEtaSeedClusterTrackAtCalo(); //detacalo 
  vars[14]  = myElectron.sigmaIetaIeta(); //see
  vars[15]  = myElectron.sigmaIphiIphi(); //spp
  vars[16]  = myElectron.r9(); //R9
  vars[17]  = myElectron.superCluster()->etaWidth(); //etawidth
  vars[18]  = myElectron.superCluster()->phiWidth(); //phiwidth
  vars[19]  = (myElectron.e5x5()) !=0. ? 1.-(myElectron.e1x5()/myElectron.e5x5()) : -1. ; //OneMinusE1x5E5x5
  vars[20]  = (1.0/myElectron.ecalEnergy()) - (1.0 / myElectron.p()); // IoEmIoP
  vars[21]  = myElectron.superCluster()->preshowerEnergy() / myElectron.superCluster()->rawEnergy(); //PreShowerOverRaw
  vars[22]  = pvc.size(); // nPV
  vars[23]  = myElectron.pt(); //pt
  vars[24]  = myElectron.eta(); //eta
  
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
  bindVariables(vars);

  double result= gbr[0]->GetClassifier(vars);

  return result;
}


void SoftElectronMVAEstimator::bindVariables(float vars[25]) const {
  if( vars[0] < -1.) //fbrem
    vars[0] = -1.;

  vars[11] = std::abs(vars[11]); // deta
  if(vars[11] > 0.06)
    vars[11] = 0.06;

  vars[12] = std::abs(vars[12]);
  if(vars[12] > 0.6)
    vars[12] = 0.6;

  //if(EoP > 20.)
  //  EoP = 20.;

  if(vars[2] > 20.) //eleEoPout
    vars[2] = 20.;
  
  vars[13] = std::abs(vars[13]); //detacalo
  if(vars[13] > 0.2)
    vars[13] = 0.2;

  if( vars[19] < -1.) //OneMinusE1x5E5x5
    vars[19] = -1;

  if( vars[19] > 2.) //OneMinusE1x5E5x5
    vars[19] = 2.;

  if( vars[7] > 200.) //gsfchi2
    vars[7] = 200;
  
  if( vars[8] > 10.) //kfchi2
    vars[8]  = 10.;

}
