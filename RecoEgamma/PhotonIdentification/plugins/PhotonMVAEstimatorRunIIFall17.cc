#include "RecoEgamma/PhotonIdentification/plugins/PhotonMVAEstimatorRunIIFall17.h"
#include "RecoEgamma/EgammaTools/interface/GBRForestTools.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "TMVA/MethodBDT.h"
#include <vector>

PhotonMVAEstimatorRunIIFall17::PhotonMVAEstimatorRunIIFall17(const edm::ParameterSet& conf):
  AnyMVAEstimatorRun2Base(conf),
  methodName_("BDTG method"),
  phoChargedIsolationLabel_(conf.getParameter<edm::InputTag>("phoChargedIsolation")), 
  phoPhotonIsolationLabel_(conf.getParameter<edm::InputTag>("phoPhotonIsolation")), 
  phoWorstChargedIsolationLabel_(conf.getParameter<edm::InputTag>("phoWorstChargedIsolation")), 
  rhoLabel_(conf.getParameter<edm::InputTag>("rho")),
  effectiveAreas_( (conf.getParameter<edm::FileInPath>("effAreasConfigFile")).fullPath()),
  phoIsoPtScalingCoeff_(conf.getParameter<std::vector<double >>("phoIsoPtScalingCoeff"))

{
  
  //
  // Construct the MVA estimators
  //
  tag_ = conf.getParameter<std::string>("mvaTag");
  
  const std::vector <std::string> weightFileNames
    = conf.getParameter<std::vector<std::string> >("weightFileNames");
  
  if( weightFileNames.size() != nCategories )
    throw cms::Exception("MVA config failure: ")
      << "wrong number of weightfiles" << std::endl;
  
  gbrForests_.clear();
  // The method name is just a key to retrieve this method later, it is not
  // a control parameter for a reader (the full definition of the MVA type and
  // everything else comes from the xml weight files).
  
  // Create a TMVA reader object for each category
  for(uint i=0; i<nCategories; i++){
    
    // Use unique_ptr so that all readers are properly cleaned up
    // when the vector clear() is called in the destructor
    
    edm::FileInPath weightFile( weightFileNames[i] );
    gbrForests_.push_back( GBRForestTools::createGBRForest( weightFile ) );
    
  }
  
}

PhotonMVAEstimatorRunIIFall17::
~PhotonMVAEstimatorRunIIFall17(){
}

float PhotonMVAEstimatorRunIIFall17::
mvaValue(const edm::Ptr<reco::Candidate>& particle, const edm::Event& iEvent) const {  
  
  const int iCategory = findCategory( particle );

  const std::vector<float> vars = fillMVAVariables( particle, iEvent ) ; 
  const float result = gbrForests_.at(iCategory)->GetResponse(vars.data()); // The BDT score
  
  // DEBUG
  const bool debug = false;
  // The list of variables here must match EXACTLY the list and order in the
  // packMVAVariables() call for barrel and endcap in this file.
  if( debug ){
    printf("Printout of the photon variable inputs for MVA:\n");
    printf("  varRawE_          %f\n", vars[0]   ); 
    printf("  varR9_            %f\n", vars[1]   ); 
    printf("  varSieie_         %f\n", vars[2]   );
    printf("  varSCEtaWidth_    %f\n", vars[3]   ); 
    printf("  varSCPhiWidth_    %f\n", vars[4]   ); 
    printf("  varSieip_         %f\n", vars[5]   ); 
    printf("  varE2x2overE5x5_  %f\n", vars[6]   ); 
    printf("  varPhoIsoRaw_     %f\n", vars[7]   );
    printf("  varChIsoRaw_      %f\n", vars[8]   ); 
    printf("  varWorstChRaw_    %f\n", vars[9]   );
    printf("  varSCEta_         %f\n", vars[10]  ); 
    printf("  varRho_           %f\n", vars[11]  );
    if( isEndcapCategory( iCategory ) ) {
      printf("  varESEffSigmaRR_  %f\n", vars[12]  ); // for endcap MVA only
      printf("  varESEnOverRawE_  %f\n", vars[13]  ); // for endcap MVA only
    } 
  }
  
  return result;
}

int PhotonMVAEstimatorRunIIFall17::findCategory( const edm::Ptr<reco::Candidate>& particle) const {
  
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
  else
    iCategory = CAT_EE;
  
  return iCategory;
}

bool PhotonMVAEstimatorRunIIFall17::
isEndcapCategory(int category) const {
  
  // For this specific MVA the function is trivial, but kept for possible
  // future evolution to an MVA with more categories in eta
  bool isEndcap = false;
  if( category == CAT_EE )
    isEndcap = true;
  
  return isEndcap;
}

// A function that should work on both pat and reco objects
std::vector<float> PhotonMVAEstimatorRunIIFall17::fillMVAVariables(const edm::Ptr<reco::Candidate>& particle, const edm::Event& iEvent) const {
  
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
  iEvent.getByLabel(phoChargedIsolationLabel_, phoChargedIsolationMap);
  iEvent.getByLabel(phoPhotonIsolationLabel_, phoPhotonIsolationMap);
  iEvent.getByLabel(phoWorstChargedIsolationLabel_, phoWorstChargedIsolationMap);

  // Get rho
  iEvent.getByLabel(rhoLabel_,rho);

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
  
  allMVAVars.varRawE         = superCluster->rawEnergy(); 
  allMVAVars.varR9           = phoRecoPtr->r9() ;
  allMVAVars.varSieie        = full5x5_sigmaIetaIeta; 
  allMVAVars.varSCEtaWidth   = superCluster->etaWidth(); 
  allMVAVars.varSCPhiWidth   = superCluster->phiWidth(); 
  allMVAVars.varSieip        = full5x5_sigmaIetaIphi;
  allMVAVars.varE2x2overE5x5 = e2x2/e5x5;
  allMVAVars.varPhoIsoRaw    = (*phoPhotonIsolationMap)[phoRecoPtr];  
  allMVAVars.varChIsoRaw     = (*phoChargedIsolationMap)[phoRecoPtr];
  allMVAVars.varWorstChRaw   = (*phoWorstChargedIsolationMap)[phoRecoPtr];
  allMVAVars.varSCEta        = superCluster->eta(); 
  allMVAVars.varRho          = *rho; 
  allMVAVars.varESEffSigmaRR = effSigmaRR;
  allMVAVars.varESEnOverRawE = superCluster->preshowerEnergy() / superCluster->rawEnergy();
  
  constrainMVAVariables(allMVAVars);
  //
  // Important: the order of variables in the "vars" vector has to be EXACTLY
  // the same as in the .xml file defining the MVA.
  //
  std::vector<float> vars;
  if( isEndcapCategory( findCategory( particle ) ) ) {
    vars = packMVAVariables(
			    allMVAVars.varRawE,
			    allMVAVars.varR9,
			    allMVAVars.varSieie,
			    allMVAVars.varSCEtaWidth,
			    allMVAVars.varSCPhiWidth,
			    allMVAVars.varSieip,
			    allMVAVars.varE2x2overE5x5,
			    allMVAVars.varPhoIsoRaw,
			    allMVAVars.varChIsoRaw,
			    allMVAVars.varWorstChRaw,
			    allMVAVars.varSCEta,
			    allMVAVars.varRho,
			    allMVAVars.varESEffSigmaRR,
			    allMVAVars.varESEnOverRawE
			    ) 
      ; 
  } else {
    vars = packMVAVariables(
			    allMVAVars.varRawE,
			    allMVAVars.varR9,
			    allMVAVars.varSieie,
			    allMVAVars.varSCEtaWidth,
			    allMVAVars.varSCPhiWidth,
			    allMVAVars.varSieip,
			    allMVAVars.varE2x2overE5x5,
			    allMVAVars.varPhoIsoRaw,
			    allMVAVars.varChIsoRaw,
			    allMVAVars.varWorstChRaw,
			    allMVAVars.varSCEta,
			    allMVAVars.varRho
			    ) 
      ;
  }
  return vars;
}

void PhotonMVAEstimatorRunIIFall17::constrainMVAVariables(AllVariables&)const {
  
  // Check that variables do not have crazy values
  
  // This function is currently empty as this specific MVA was not
  // developed with restricting variables to specific physical ranges.
    
}

void PhotonMVAEstimatorRunIIFall17::setConsumes(edm::ConsumesCollector&& cc) const {
  cc.consumes<edm::ValueMap<float> >(phoChargedIsolationLabel_);
  cc.consumes<edm::ValueMap<float> >(phoPhotonIsolationLabel_);
  cc.consumes<edm::ValueMap<float> >( phoWorstChargedIsolationLabel_);
  cc.consumes<double>(rhoLabel_);
}

DEFINE_EDM_PLUGIN(AnyMVAEstimatorRun2Factory,
                  PhotonMVAEstimatorRunIIFall17,
                  "PhotonMVAEstimatorRunIIFall17");

