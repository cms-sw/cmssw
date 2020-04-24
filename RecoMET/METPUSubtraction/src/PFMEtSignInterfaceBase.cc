#include "RecoMET/METPUSubtraction/interface/PFMEtSignInterfaceBase.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMET/METAlgorithms/interface/significanceAlgo.h"

#include <TVectorD.h>

const double defaultPFMEtResolutionX = 10.;
const double defaultPFMEtResolutionY = 10.;

const double epsilon = 1.e-9;

PFMEtSignInterfaceBase::PFMEtSignInterfaceBase(const edm::ParameterSet& cfg)
  : pfMEtResolution_(nullptr),
    inputFile_(nullptr),
    lut_(nullptr)
{
  pfMEtResolution_ = new metsig::SignAlgoResolutions(cfg);

  if ( cfg.exists("addJERcorr") ) {
    edm::ParameterSet cfgJERcorr = cfg.getParameter<edm::ParameterSet>("addJERcorr");
    edm::FileInPath inputFileName = cfgJERcorr.getParameter<edm::FileInPath>("inputFileName");
    std::string lutName = cfgJERcorr.getParameter<std::string>("lutName");
    if ( inputFileName.location()!=edm::FileInPath::Local ) 
      throw cms::Exception("PFMEtSignInterfaceBase") 
        << " Failed to find File = " << inputFileName << " !!\n";
    
    inputFile_ = new TFile(inputFileName.fullPath().data());
    lut_ = dynamic_cast<TH2*>(inputFile_->Get(lutName.data()));
    if ( !lut_ ) 
      throw cms::Exception("PFMEtSignInterfaceBase") 
        << " Failed to load LUT = " << lutName.data() << " from file = " << inputFileName.fullPath().data() << " !!\n";
  }
  
  verbosity_ = cfg.exists("verbosity") ?
    cfg.getParameter<int>("verbosity") : 0;
}

PFMEtSignInterfaceBase::~PFMEtSignInterfaceBase()
{
  delete pfMEtResolution_;
  delete inputFile_;
  delete lut_;
}

reco::METCovMatrix PFMEtSignInterfaceBase::operator()(const std::vector<metsig::SigInputObj>& pfMEtSignObjects) const
{
  // if ( this->verbosity_ ) {
  //   std::cout << "<PFMEtSignInterfaceBase::operator()>:" << std::endl;
  //   std::cout << " pfMEtSignObjects: entries = " << pfMEtSignObjects.size() << std::endl;
  //   double dpt2Sum = 0.;
  //   for ( std::vector<metsig::SigInputObj>::const_iterator pfMEtSignObject = pfMEtSignObjects.begin();
  // 	  pfMEtSignObject != pfMEtSignObjects.end(); ++pfMEtSignObject ) {
  //     std::cout << pfMEtSignObject->get_type() << ": pt = " << pfMEtSignObject->get_energy() << "," 
  // 		<< " phi = " << pfMEtSignObject->get_phi() << " --> dpt = " << pfMEtSignObject->get_sigma_e() << std::endl;
  //     dpt2Sum += pfMEtSignObject->get_sigma_e();
  //   }
  //   std::cout << "--> sqrt(sum(dpt^2)) = " << sqrt(dpt2Sum) << std::endl;
  // }

  reco::METCovMatrix pfMEtCov;
  if ( pfMEtSignObjects.size() >= 2 ) {
    metsig::significanceAlgo pfMEtSignAlgorithm;
    pfMEtSignAlgorithm.addObjects(pfMEtSignObjects);
    pfMEtCov = pfMEtSignAlgorithm.getSignifMatrix();
 
    double det=0;
    pfMEtCov.Det(det);

    // if ( this->verbosity_ && std::abs(det) > epsilon ) {
    //   //keep TMatrixD as it is much easier to find 
    //   //eigenvectors and values than with SMatrix;
    //   //not used anyway, except for debugging
    //   TMatrixD tmpMatrix(2,2);
    //   tmpMatrix(0,0) = pfMEtCov(0,0);
    //   tmpMatrix(0,1) = pfMEtCov(0,1);
    //   tmpMatrix(1,0) = pfMEtCov(1,0);
    //   tmpMatrix(1,1) = pfMEtCov(1,1);

    //   TVectorD eigenValues(2);
    //   TMatrixD eigenVectors = tmpMatrix.EigenVectors(eigenValues);
    //   // CV: eigenvectors are stored in columns 
    //   //     and are sorted such that the one corresponding to the highest eigenvalue is in the **first** column
    //   for ( unsigned iEigenVector = 0; iEigenVector < 2; ++iEigenVector ) {
    // 	std::cout << "eigenVector #" << iEigenVector << " (eigenValue = " << eigenValues(iEigenVector) << "):" 
    // 		  << " x = " << eigenVectors(0, iEigenVector) << ", y = " << eigenVectors(1, iEigenVector) << std::endl;
    //   }
    // }
    
    //--- substitute (PF)MEt resolution matrix by default values 
    //    in case resolution matrix cannot be inverted
    if (std::abs(det) < epsilon ) {
      edm::LogWarning("PFMEtSignInterfaceBase::operator()") 
	<< "Inversion of PFMEt covariance matrix failed, det = " << det
	<< " --> replacing covariance matrix by resolution defaults !!";
      pfMEtCov(0,0) = defaultPFMEtResolutionX*defaultPFMEtResolutionX;
      pfMEtCov(0,1) = 0.;
      pfMEtCov(1,0) = 0.;
      pfMEtCov(1,1) = defaultPFMEtResolutionY*defaultPFMEtResolutionY;
    }
  } else {
    pfMEtCov(0,0) = 0.;
    pfMEtCov(0,1) = 0.;
    pfMEtCov(1,0) = 0.;
    pfMEtCov(1,1) = 0.;
  }

  return pfMEtCov;
}
