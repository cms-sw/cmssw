#include "RecoEgamma/PhotonIdentification/plugins/PhotonMVAEstimatorRun2Spring16NonTrig.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "TMVA/MethodBDT.h"
#include <vector>

PhotonMVAEstimatorRun2Spring16NonTrig::PhotonMVAEstimatorRun2Spring16NonTrig(const edm::ParameterSet& conf):
  AnyMVAEstimatorRun2Base(conf),
  _MethodName("BDTG method"),
  _phoChargedIsolationLabel(conf.getParameter<edm::InputTag>("phoChargedIsolation")), 
  _phoPhotonIsolationLabel(conf.getParameter<edm::InputTag>("phoPhotonIsolation")), 
  _phoWorstChargedIsolationLabel(conf.getParameter<edm::InputTag>("phoWorstChargedIsolation")), 
  _rhoLabel(conf.getParameter<edm::InputTag>("rho")),
  _effectiveAreas( (conf.getParameter<edm::FileInPath>("effAreasConfigFile")).fullPath()),
  _phoIsoPtScalingCoeff(conf.getParameter<std::vector<double >>("phoIsoPtScalingCoeff")),
  _phoIsoCutoff(conf.getParameter<double>("phoIsoCutoff"))
{

  //
  // Construct the MVA estimators
  //
  _tag = conf.getParameter<std::string>("mvaTag");

  const std::vector <std::string> weightFileNames
    = conf.getParameter<std::vector<std::string> >("weightFileNames");

  if( (int)(weightFileNames.size()) != nCategories )
    throw cms::Exception("MVA config failure: ")
      << "wrong number of weightfiles" << std::endl;

  _gbrForests.clear();
  // The method name is just a key to retrieve this method later, it is not
  // a control parameter for a reader (the full definition of the MVA type and
  // everything else comes from the xml weight files).
   
  // Create a TMVA reader object for each category
  for(int i=0; i<nCategories; i++){

    // Use unique_ptr so that all readers are properly cleaned up
    // when the vector clear() is called in the destructor

    edm::FileInPath weightFile( weightFileNames[i] );
    _gbrForests.push_back( createSingleReader(i, weightFile ) );

  }

}

PhotonMVAEstimatorRun2Spring16NonTrig::
~PhotonMVAEstimatorRun2Spring16NonTrig(){
}

float PhotonMVAEstimatorRun2Spring16NonTrig::
mvaValue(const edm::Ptr<reco::Candidate>& particle, const edm::Event& iEvent) const {  

  const int iCategory = findCategory( particle );
  const std::vector<float> vars = std::move( fillMVAVariables( particle, iEvent ) );  
  
  const float result = _gbrForests.at(iCategory)->GetClassifier(vars.data());

  // DEBUG
  const bool debug = false;
  // The list of variables here must match EXACTLY the list and order in the
  // packMVAVariables() call for barrel and endcap in this file.
  if( debug ){
    printf("Printout of the photon variable inputs for MVA:\n");
    printf("  varSCPhi_            %f\n", vars[0]   );
    printf("  varR9_            %f\n", vars[1]   ); 
    printf("  varSieie_         %f\n", vars[2]   );
    printf("  varSieip_         %f\n", vars[3]   ); 
    printf("  varE2x2overE5x5_  %f\n", vars[4]   ); 
    printf("  varSCEta_         %f\n", vars[5]   ); 
    printf("  varRawE_          %f\n", vars[6]   ); 
    printf("  varSCEtaWidth_    %f\n", vars[7]   ); 
    printf("  varSCPhiWidth_    %f\n", vars[8]  ); 
    printf("  varRho_           %f\n", vars[9]  );
    if( !isEndcapCategory( iCategory ) ) {
      printf("  varPhoIsoRaw_     %f\n", vars[10]  );
    } else {
      printf("  varPhoIsoRawCorr_  %f\n", vars[10]  ); // for endcap MVA only in 2016
    }
    printf("  varChIsoRaw_      %f\n", vars[11]  ); 
    printf("  varWorstChRaw_    %f\n", vars[12]  );
    if( isEndcapCategory( iCategory ) ) {
      printf("  varESEnOverRawE_  %f\n", vars[13]  ); // for endcap MVA only
      printf("  varESEffSigmaRR_  %f\n", vars[14]  ); // for endcap MVA only
    } 
  }
  
  return result;
}

int PhotonMVAEstimatorRun2Spring16NonTrig::findCategory( const edm::Ptr<reco::Candidate>& particle) const {
  
  // Try to cast the particle into a reco particle.
  // This should work for both reco and pat.
  const edm::Ptr<reco::Photon> phoRecoPtr = ( edm::Ptr<reco::Photon> )particle;
  if( phoRecoPtr.isNull() )
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::Photon or pat::Photon," << std::endl
      << " but appears to be neither" << std::endl;

  float eta = phoRecoPtr->superCluster()->eta();

  //
  // Determine the category
  //
  int  iCategory = UNDEFINED;
  const float ebeeSplit = 1.479; // division between barrel and endcap

  if ( std::abs(eta) < ebeeSplit)  
    iCategory = CAT_EB;

  if (std::abs(eta) >= ebeeSplit) 
    iCategory = CAT_EE;

  return iCategory;
}

bool PhotonMVAEstimatorRun2Spring16NonTrig::
isEndcapCategory(int category) const {

  // For this specific MVA the function is trivial, but kept for possible
  // future evolution to an MVA with more categories in eta
  bool isEndcap = false;
  if( category == CAT_EE )
    isEndcap = true;

  return isEndcap;
}


std::unique_ptr<const GBRForest> PhotonMVAEstimatorRun2Spring16NonTrig::
createSingleReader(const int iCategory, const edm::FileInPath &weightFile){

  //
  // Create the reader  
  //
  TMVA::Reader tmpTMVAReader( "!Color:Silent:Error" );

  //
  // Configure all variables and spectators. Note: the order and names
  // must match what is found in the xml weights file!
  //

  tmpTMVAReader.AddVariable("scPhi", &_allMVAVars.scPhi);
  tmpTMVAReader.AddVariable("r9"        , &_allMVAVars.varR9);
  tmpTMVAReader.AddVariable("sigmaIetaIeta", &_allMVAVars.varSieie);
  tmpTMVAReader.AddVariable("covIEtaIPhi", &_allMVAVars.varSieip);
  tmpTMVAReader.AddVariable("s4", &_allMVAVars.varE2x2overE5x5);
  tmpTMVAReader.AddVariable("scEta" , &_allMVAVars.varSCEta);
  tmpTMVAReader.AddVariable("SCRawE"      , &_allMVAVars.varRawE);
  tmpTMVAReader.AddVariable("etaWidth", &_allMVAVars.varSCEtaWidth);
  tmpTMVAReader.AddVariable("phiWidth", &_allMVAVars.varSCPhiWidth);
  //Pileup
  tmpTMVAReader.AddVariable("rho"       , &_allMVAVars.varRho);
  //EB only, because of loss of transparency in EE
  if(!isEndcapCategory(iCategory)){
    tmpTMVAReader.AddVariable("CITK_isoPhotons" , &_allMVAVars.varPhoIsoRaw);
  }
  else{
    tmpTMVAReader.AddVariable("CITK_isoPhoCorrMax2p5", &_allMVAVars.varPhoIsoCorr);
  }
  //isolations in both EB and EE
  tmpTMVAReader.AddVariable("CITK_isoChargedHad"  , &_allMVAVars.varChIsoRaw);
  tmpTMVAReader.AddVariable("chgIsoWrtWorstVtx", &_allMVAVars.varWorstChRaw);

  // Endcap only variables
  if( isEndcapCategory(iCategory) ){
    tmpTMVAReader.AddVariable("esEffSigmaRR"      , &_allMVAVars.varESEffSigmaRR);
    tmpTMVAReader.AddVariable("esEnergy/SCRawE" , &_allMVAVars.varESEnOverRawE);
  }

  //
  // Book the method and set up the weights file
  //
  std::unique_ptr<TMVA::IMethod> temp( tmpTMVAReader.BookMVA(_MethodName , weightFile.fullPath() ) );

  return std::unique_ptr<const GBRForest>( new GBRForest( dynamic_cast<TMVA::MethodBDT*>( tmpTMVAReader.FindMVA(_MethodName) ) ) );

}

// A function that should work on both pat and reco objects
std::vector<float> PhotonMVAEstimatorRun2Spring16NonTrig::fillMVAVariables(const edm::Ptr<reco::Candidate>& particle, const edm::Event& iEvent) const {
  
  // 
  // Declare all value maps corresponding to the above tokens
  //
  edm::Handle<edm::ValueMap<float> > full5x5SigmaIEtaIEtaMap;
  edm::Handle<edm::ValueMap<float> > full5x5SigmaIEtaIPhiMap;
  edm::Handle<edm::ValueMap<float> > full5x5E2x2Map;
  edm::Handle<edm::ValueMap<float> > full5x5E5x5Map;
  edm::Handle<edm::ValueMap<float> > esEffSigmaRRMap;
  //
  edm::Handle<edm::ValueMap<float> > phoChargedIsolationMap;
  edm::Handle<edm::ValueMap<float> > phoPhotonIsolationMap;
  edm::Handle<edm::ValueMap<float> > phoWorstChargedIsolationMap;

  // Rho will be pulled from the event content
  edm::Handle<double> rho;

  // Get the isolation maps
  iEvent.getByLabel(_phoChargedIsolationLabel, phoChargedIsolationMap);
  iEvent.getByLabel(_phoPhotonIsolationLabel, phoPhotonIsolationMap);
  iEvent.getByLabel(_phoWorstChargedIsolationLabel, phoWorstChargedIsolationMap);

  // Get rho
  iEvent.getByLabel(_rhoLabel,rho);

  // Make sure everything is retrieved successfully
  if(! ( phoChargedIsolationMap.isValid()
         && phoPhotonIsolationMap.isValid()
         && phoWorstChargedIsolationMap.isValid()
         && rho.isValid() ) )
    throw cms::Exception("MVA failure: ")
      << "Failed to retrieve event content needed for this MVA" 
      << std::endl
      << "Check python MVA configuration file and make sure all needed"
      << std::endl
      << "producers are running upstream" << std::endl;

  // Try to cast the particle into a reco particle.
  // This should work for both reco and pat.
  const edm::Ptr<reco::Photon> phoRecoPtr(particle);
  if( phoRecoPtr.isNull() )
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::Photon or pat::Photon," << std::endl
      << " but appears to be neither" << std::endl;

  // Both pat and reco particles have exactly the same accessors.
  auto superCluster = phoRecoPtr->superCluster();
  // Full 5x5 cluster shapes. We could take some of this directly from
  // the photon object, but some of these are not available.
  float e2x2 = std::numeric_limits<float>::max();
  float e5x5 = std::numeric_limits<float>::max();
  float full5x5_sigmaIetaIeta = std::numeric_limits<float>::max();
  float full5x5_sigmaIetaIphi = std::numeric_limits<float>::max();
  float effSigmaRR = std::numeric_limits<float>::max();

  AllVariables allMVAVars;

  const auto& full5x5_pss = phoRecoPtr->full5x5_showerShapeVariables();
  e2x2 = full5x5_pss.e2x2;
  e5x5 = full5x5_pss.e5x5;
  full5x5_sigmaIetaIeta = full5x5_pss.sigmaIetaIeta;
  full5x5_sigmaIetaIphi = full5x5_pss.sigmaIetaIphi;
  effSigmaRR = full5x5_pss.effSigmaRR;

  allMVAVars.scPhi           = superCluster->phi();
  allMVAVars.varR9           = phoRecoPtr->r9() ;
  allMVAVars.varSieie        = full5x5_sigmaIetaIeta; 
  allMVAVars.varSieip        = full5x5_sigmaIetaIphi;
  allMVAVars.varE2x2overE5x5 = e2x2/e5x5;
  allMVAVars.varSCEta        = superCluster->eta(); 
  allMVAVars.varRawE         = superCluster->rawEnergy(); 
  allMVAVars.varSCEtaWidth   = superCluster->etaWidth(); 
  allMVAVars.varSCPhiWidth   = superCluster->phiWidth(); 
  allMVAVars.varESEnOverRawE = superCluster->preshowerEnergy() / superCluster->rawEnergy();
  allMVAVars.varESEffSigmaRR = effSigmaRR;
  allMVAVars.varRho          = *rho; 
  allMVAVars.varPhoIsoRaw    = (*phoPhotonIsolationMap)[phoRecoPtr];  
  allMVAVars.varChIsoRaw     = (*phoChargedIsolationMap)[phoRecoPtr];
  allMVAVars.varWorstChRaw   = (*phoWorstChargedIsolationMap)[phoRecoPtr];

  //photon iso corrected:

  double eA = _effectiveAreas.getEffectiveArea( abs(superCluster->eta()) );
  double phoIsoPtScalingCoeffVal = 0;
  if( !isEndcapCategory( findCategory ( particle ) ) ) 
    phoIsoPtScalingCoeffVal = _phoIsoPtScalingCoeff.at(0); // barrel case
  else
    phoIsoPtScalingCoeffVal =  _phoIsoPtScalingCoeff.at(1); //endcap case
  double phoIsoCorr = (*phoPhotonIsolationMap)[phoRecoPtr] - eA*(*rho) - phoIsoPtScalingCoeffVal*phoRecoPtr->pt();

  allMVAVars.varPhoIsoCorr = TMath::Max(phoIsoCorr, _phoIsoCutoff);

  constrainMVAVariables(allMVAVars);
  //
  // Important: the order of variables in the "vars" vector has to be EXACTLY
  // the same as in the .xml file defining the MVA.
  //
  std::vector<float> vars;
  if( isEndcapCategory( findCategory( particle ) ) ) {
    vars = std::move( packMVAVariables(
				       allMVAVars.scPhi,
                                       allMVAVars.varR9,
                                       allMVAVars.varSieie,
                                       allMVAVars.varSieip,
                                       allMVAVars.varE2x2overE5x5,
                                       allMVAVars.varSCEta,
                                       allMVAVars.varRawE,
                                       allMVAVars.varSCEtaWidth,
                                       allMVAVars.varSCPhiWidth,
				       allMVAVars.varRho,
				       allMVAVars.varPhoIsoCorr,
				       allMVAVars.varChIsoRaw,
				       allMVAVars.varWorstChRaw,
				       allMVAVars.varESEffSigmaRR,
				       allMVAVars.varESEnOverRawE
				       ) 
                      ); 
  } else {
    vars = std::move( packMVAVariables(
				       allMVAVars.scPhi,
                                       allMVAVars.varR9,
                                       allMVAVars.varSieie,
                                       allMVAVars.varSieip,
                                       allMVAVars.varE2x2overE5x5,
                                       allMVAVars.varSCEta,
                                       allMVAVars.varRawE,
                                       allMVAVars.varSCEtaWidth,
                                       allMVAVars.varSCPhiWidth,
                                       allMVAVars.varRho,
                                       allMVAVars.varPhoIsoRaw,
                                       allMVAVars.varChIsoRaw,
                                       allMVAVars.varWorstChRaw
				       ) 
                      );
  }
  return vars;
}

void PhotonMVAEstimatorRun2Spring16NonTrig::constrainMVAVariables(AllVariables&)const {

  // Check that variables do not have crazy values
  
  // This function is currently empty as this specific MVA was not
  // developed with restricting variables to specific physical ranges.
  return;

}

void PhotonMVAEstimatorRun2Spring16NonTrig::setConsumes(edm::ConsumesCollector&& cc) const {
  cc.consumes<edm::ValueMap<float> >(_phoChargedIsolationLabel);
  cc.consumes<edm::ValueMap<float> >(_phoPhotonIsolationLabel);
  cc.consumes<edm::ValueMap<float> >( _phoWorstChargedIsolationLabel);
  cc.consumes<double>(_rhoLabel);
}

DEFINE_EDM_PLUGIN(AnyMVAEstimatorRun2Factory,
                  PhotonMVAEstimatorRun2Spring16NonTrig,
                  "PhotonMVAEstimatorRun2Spring16NonTrig");

