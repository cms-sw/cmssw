#include "RecoEgamma/PhotonIdentification/interface/PhotonMVAEstimatorRun2Spring15NonTrig.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

PhotonMVAEstimatorRun2Spring15NonTrig::PhotonMVAEstimatorRun2Spring15NonTrig(const edm::ParameterSet& conf):
  AnyMVAEstimatorRun2Base(conf)
{

  //
  // Construct the MVA estimators
  //
  const std::vector <std::string> weightFileNames
    = conf.getParameter<std::vector<std::string> >("weightFileNames");

  if( (int)(weightFileNames.size()) != nCategories )
    throw cms::Exception("MVA config failure: ")
      << "wrong number of weightfiles" << std::endl;

  _tmvaReaders.clear();
  // The method name is just a key to retrieve this method later, it is not
  // a control parameter for a reader (the full definition of the MVA type and
  // everything else comes from the xml weight files).
  _MethodName = "BDTG method"; 
  // Create a TMVA reader object for each category
  for(int i=0; i<nCategories; i++){

    // Use unique_ptr so that all readers are properly cleaned up
    // when the vector clear() is called in the destructor

    edm::FileInPath weightFile( weightFileNames[i] );
    _tmvaReaders.push_back( std::unique_ptr<TMVA::Reader> 
			    ( createSingleReader(i, weightFile ) ) );

  }

}

PhotonMVAEstimatorRun2Spring15NonTrig::
~PhotonMVAEstimatorRun2Spring15NonTrig(){
  
  _tmvaReaders.clear();
}

void PhotonMVAEstimatorRun2Spring15NonTrig::setConsumes(edm::ConsumesCollector&& cc){

  // All tokens for event content needed by this MVA
  // Cluster shapes
  _full5x5SigmaIEtaIEtaMapToken = cc.consumes <edm::ValueMap<float> >
    (_conf.getParameter<edm::InputTag>("full5x5SigmaIEtaIEtaMap"));

  _full5x5SigmaIEtaIPhiMapToken = cc.consumes <edm::ValueMap<float> >
    (_conf.getParameter<edm::InputTag>("full5x5SigmaIEtaIPhiMap"));

  _full5x5E1x3MapToken = cc.consumes <edm::ValueMap<float> >
    (_conf.getParameter<edm::InputTag>("full5x5E1x3Map"));

  _full5x5E2x2MapToken = cc.consumes <edm::ValueMap<float> >
    (_conf.getParameter<edm::InputTag>("full5x5E2x2Map"));

  _full5x5E2x5MaxMapToken = cc.consumes <edm::ValueMap<float> >
    (_conf.getParameter<edm::InputTag>("full5x5E2x5MaxMap"));

  _full5x5E5x5MapToken = cc.consumes <edm::ValueMap<float> >
    (_conf.getParameter<edm::InputTag>("full5x5E5x5Map"));

  _esEffSigmaRRMapToken = cc.consumes <edm::ValueMap<float> >
    (_conf.getParameter<edm::InputTag>("esEffSigmaRRMap"));

  // Isolations
  _phoChargedIsolationToken = cc.consumes <edm::ValueMap<float> >
    (_conf.getParameter<edm::InputTag>("phoChargedIsolation"));

  _phoPhotonIsolationToken = cc.consumes <edm::ValueMap<float> >
    (_conf.getParameter<edm::InputTag>("phoPhotonIsolation"));

  _phoWorstChargedIsolationToken = cc.consumes <edm::ValueMap<float> >
    (_conf.getParameter<edm::InputTag>("phoWorstChargedIsolation"));

  // Pileup
  _rhoToken = cc.consumes<double> (_conf.getParameter<edm::InputTag>("rho"));
  
}

void PhotonMVAEstimatorRun2Spring15NonTrig::getEventContent(const edm::Event& iEvent){

  // Get the full5x5 and ES maps
  iEvent.getByToken(_full5x5SigmaIEtaIEtaMapToken, _full5x5SigmaIEtaIEtaMap);
  iEvent.getByToken(_full5x5SigmaIEtaIPhiMapToken, _full5x5SigmaIEtaIPhiMap);
  iEvent.getByToken(_full5x5E1x3MapToken, _full5x5E1x3Map);
  iEvent.getByToken(_full5x5E2x2MapToken, _full5x5E2x2Map);
  iEvent.getByToken(_full5x5E2x5MaxMapToken, _full5x5E2x5MaxMap);
  iEvent.getByToken(_full5x5E5x5MapToken, _full5x5E5x5Map);
  iEvent.getByToken(_esEffSigmaRRMapToken, _esEffSigmaRRMap);

  // Get the isolation maps
  iEvent.getByToken(_phoChargedIsolationToken, _phoChargedIsolationMap);
  iEvent.getByToken(_phoPhotonIsolationToken, _phoPhotonIsolationMap);
  iEvent.getByToken(_phoWorstChargedIsolationToken, _phoWorstChargedIsolationMap);

  // Get rho
  iEvent.getByToken(_rhoToken,_rho);

  // Make sure everything is retrieved successfully
  if(! (_full5x5SigmaIEtaIEtaMap.isValid()
	&& _full5x5SigmaIEtaIPhiMap.isValid()
	&& _full5x5E1x3Map.isValid()
	&& _full5x5E2x2Map.isValid()
	&& _full5x5E2x5MaxMap.isValid()
	&& _full5x5E5x5Map.isValid()
	&& _esEffSigmaRRMap.isValid()
	&& _phoChargedIsolationMap.isValid()
	&& _phoPhotonIsolationMap.isValid()
	&& _phoWorstChargedIsolationMap.isValid()
	&& _rho.isValid() ) )
    throw cms::Exception("MVA failure: ")
      << "Failed to retrieve event content needed for this MVA" 
      << std::endl
      << "Check python MVA configuration file and make sure all needed"
      << std::endl
      << "producers are running upstream" << std::endl;
}


float PhotonMVAEstimatorRun2Spring15NonTrig::
mvaValue( const edm::Ptr<reco::Candidate>& particle){
  
  int iCategory = findCategory( particle );
  fillMVAVariables( particle );  
  constrainMVAVariables();
  float result = _tmvaReaders.at(iCategory)->EvaluateMVA(_MethodName);

  // DEBUG
  const bool debug = false;
  if( debug ){
    printf("Printout of the photon variable inputs for MVA:\n");
    printf("  varPhi_           %f\n", _allMVAVars.varPhi         );
    printf("  varR9_            %f\n", _allMVAVars.varR9          ); 
    printf("  varSieie_         %f\n", _allMVAVars.varSieie       );
    printf("  varSieip_         %f\n", _allMVAVars.varSieip       ); 
    printf("  varE1x3overE5x5_  %f\n", _allMVAVars.varE1x3overE5x5); 
    printf("  varE2x2overE5x5_  %f\n", _allMVAVars.varE2x2overE5x5); 
    printf("  varE2x5overE5x5_  %f\n", _allMVAVars.varE2x5overE5x5); 
    printf("  varSCEta_         %f\n", _allMVAVars.varSCEta       ); 
    printf("  varRawE_          %f\n", _allMVAVars.varRawE        ); 
    printf("  varSCEtaWidth_    %f\n", _allMVAVars.varSCEtaWidth  ); 
    printf("  varSCPhiWidth_    %f\n", _allMVAVars.varSCPhiWidth  ); 
    printf("  varRho_           %f\n", _allMVAVars.varRho         );
    printf("  varPhoIsoRaw_     %f\n", _allMVAVars.varPhoIsoRaw   );
    printf("  varChIsoRaw_      %f\n", _allMVAVars.varChIsoRaw    ); 
    printf("  varWorstChRaw_    %f\n", _allMVAVars.varWorstChRaw  );
    printf("  varESEnOverRawE_  %f\n", _allMVAVars.varESEnOverRawE); // for endcap MVA only
    printf("  varESEffSigmaRR_  %f\n", _allMVAVars.varESEffSigmaRR); // for endcap MVA only
    // The spectators
    printf("  varPt_    %f\n", _allMVAVars.varPt          ); 
    printf("  varEta_  %f\n", _allMVAVars.varEta         );
  }

  return result;
}

int PhotonMVAEstimatorRun2Spring15NonTrig::findCategory( const edm::Ptr<reco::Candidate>& particle){
  
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

bool PhotonMVAEstimatorRun2Spring15NonTrig::
isEndcapCategory(int category ){

  // For this specific MVA the function is trivial, but kept for possible
  // future evolution to an MVA with more categories in eta
  bool isEndcap = false;
  if( category == CAT_EE )
    isEndcap = true;

  return isEndcap;
}


TMVA::Reader *PhotonMVAEstimatorRun2Spring15NonTrig::
createSingleReader(const int iCategory, const edm::FileInPath &weightFile){

  //
  // Create the reader  
  //
  TMVA::Reader *tmpTMVAReader = new TMVA::Reader( "!Color:Silent:Error" );

  //
  // Configure all variables and spectators. Note: the order and names
  // must match what is found in the xml weights file!
  //

  tmpTMVAReader->AddVariable("recoPhi"   , &_allMVAVars.varPhi);
  tmpTMVAReader->AddVariable("r9"        , &_allMVAVars.varR9);
  tmpTMVAReader->AddVariable("sieieFull5x5", &_allMVAVars.varSieie);
  tmpTMVAReader->AddVariable("sieipFull5x5", &_allMVAVars.varSieip);
  tmpTMVAReader->AddVariable("e1x3Full5x5/e5x5Full5x5"        , &_allMVAVars.varE1x3overE5x5);
  tmpTMVAReader->AddVariable("e2x2Full5x5/e5x5Full5x5"        , &_allMVAVars.varE2x2overE5x5);
  tmpTMVAReader->AddVariable("e2x5Full5x5/e5x5Full5x5"        , &_allMVAVars.varE2x5overE5x5);
  tmpTMVAReader->AddVariable("recoSCEta" , &_allMVAVars.varSCEta);
  tmpTMVAReader->AddVariable("rawE"      , &_allMVAVars.varRawE);
  tmpTMVAReader->AddVariable("scEtaWidth", &_allMVAVars.varSCEtaWidth);
  tmpTMVAReader->AddVariable("scPhiWidth", &_allMVAVars.varSCPhiWidth);

  // Endcap only variables
  if( isEndcapCategory(iCategory) ){
    tmpTMVAReader->AddVariable("esEn/rawE" , &_allMVAVars.varESEnOverRawE);
    tmpTMVAReader->AddVariable("esRR"      , &_allMVAVars.varESEffSigmaRR);
  }

  // Pileup
  tmpTMVAReader->AddVariable("rho"       , &_allMVAVars.varRho);

  // Isolations
  tmpTMVAReader->AddVariable("phoIsoRaw" , &_allMVAVars.varPhoIsoRaw);
  tmpTMVAReader->AddVariable("chIsoRaw"  , &_allMVAVars.varChIsoRaw);
  tmpTMVAReader->AddVariable("chWorstRaw", &_allMVAVars.varWorstChRaw);

  // Spectators
  tmpTMVAReader->AddSpectator("recoPt" , &_allMVAVars.varPt);
  tmpTMVAReader->AddSpectator("recoEta", &_allMVAVars.varEta);

  //
  // Book the method and set up the weights file
  //
  tmpTMVAReader->BookMVA(_MethodName , weightFile.fullPath() );

  return tmpTMVAReader;
}

// A function that should work on both pat and reco objects
void PhotonMVAEstimatorRun2Spring15NonTrig::fillMVAVariables(const edm::Ptr<reco::Candidate>& particle){

  // Try to cast the particle into a reco particle.
  // This should work for both reco and pat.
  const edm::Ptr<reco::Photon> phoRecoPtr = ( edm::Ptr<reco::Photon> )particle;
  if( phoRecoPtr.isNull() )
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::Photon or pat::Photon," << std::endl
      << " but appears to be neither" << std::endl;

  // Both pat and reco particles have exactly the same accessors.
  auto superCluster = phoRecoPtr->superCluster();
  // Full 5x5 cluster shapes. We could take some of this directly from
  // the photon object, but some of these are not available.
  float e1x3 = (*_full5x5E1x3Map   )[ phoRecoPtr ];
  float e2x2 = (*_full5x5E2x2Map   )[ phoRecoPtr ];
  float e2x5 = (*_full5x5E2x5MaxMap)[ phoRecoPtr ];
  float e5x5 = (*_full5x5E5x5Map   )[ phoRecoPtr ];

  _allMVAVars.varPhi          = phoRecoPtr->phi();
  _allMVAVars.varR9           = phoRecoPtr->r9() ;
  _allMVAVars.varSieie        = (*_full5x5SigmaIEtaIEtaMap)[ phoRecoPtr ]; // in principle, in the photon object as well
  _allMVAVars.varSieip        = (*_full5x5SigmaIEtaIPhiMap)[ phoRecoPtr ]; // not in the photon object
  _allMVAVars.varE1x3overE5x5 = e1x3/e5x5;
  _allMVAVars.varE2x2overE5x5 = e2x2/e5x5;
  _allMVAVars.varE2x5overE5x5 = e2x5/e5x5;
  _allMVAVars.varSCEta        = superCluster->eta(); 
  _allMVAVars.varRawE         = superCluster->rawEnergy(); 
  _allMVAVars.varSCEtaWidth   = superCluster->etaWidth(); 
  _allMVAVars.varSCPhiWidth   = superCluster->phiWidth(); 
  _allMVAVars.varESEnOverRawE = superCluster->preshowerEnergy() / superCluster->rawEnergy();
  _allMVAVars.varESEffSigmaRR = (*_esEffSigmaRRMap)[ phoRecoPtr ];
  _allMVAVars.varRho          = *_rho; 
  _allMVAVars.varPhoIsoRaw    = (*_phoPhotonIsolationMap)[phoRecoPtr];  
  _allMVAVars.varChIsoRaw     = (*_phoChargedIsolationMap)[phoRecoPtr];
  _allMVAVars.varWorstChRaw   = (*_phoWorstChargedIsolationMap)[phoRecoPtr];
  // Declare spectator vars
  _allMVAVars.varPt = phoRecoPtr->pt(); 
  _allMVAVars.varEta = phoRecoPtr->eta();

}

void PhotonMVAEstimatorRun2Spring15NonTrig::constrainMVAVariables(){

  // Check that variables do not have crazy values
  
  // This function is currently empty as this specific MVA was not
  // developed with restricting variables to specific physical ranges.
  return;

}

