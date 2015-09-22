#include "RecoEgamma/PhotonIdentification/interface/PhotonMVAEstimatorRun2Phys14NonTrig.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "TMVA/MethodBDT.h"

PhotonMVAEstimatorRun2Phys14NonTrig::PhotonMVAEstimatorRun2Phys14NonTrig(const edm::ParameterSet& conf) :
  AnyMVAEstimatorRun2Base(conf),
  // The method name is just a key to retrieve this method later, it is not
  // a control parameter for a reader (the full definition of the MVA type and
  // everything else comes from the xml weight files).
  _MethodName("BDTG method"),
  _full5x5SigmaIEtaIEtaMapLabel(conf.getParameter<edm::InputTag>("full5x5SigmaIEtaIEtaMap")), 
  _full5x5SigmaIEtaIPhiMapLabel(conf.getParameter<edm::InputTag>("full5x5SigmaIEtaIPhiMap")), 
  _full5x5E1x3MapLabel(conf.getParameter<edm::InputTag>("full5x5E1x3Map")), 
  _full5x5E2x2MapLabel(conf.getParameter<edm::InputTag>("full5x5E2x2Map")), 
  _full5x5E2x5MaxMapLabel(conf.getParameter<edm::InputTag>("full5x5E2x5MaxMap")), 
  _full5x5E5x5MapLabel(conf.getParameter<edm::InputTag>("full5x5E5x5Map")), 
  _esEffSigmaRRMapLabel(conf.getParameter<edm::InputTag>("esEffSigmaRRMap")), 
  _phoChargedIsolationLabel(conf.getParameter<edm::InputTag>("phoChargedIsolation")), 
  _phoPhotonIsolationLabel(conf.getParameter<edm::InputTag>("phoPhotonIsolation")), 
  _phoWorstChargedIsolationLabel(conf.getParameter<edm::InputTag>("phoWorstChargedIsolation")), 
  _rhoLabel(conf.getParameter<edm::InputTag>("rho"))
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

  // Create a TMVA reader object for each category
  for(int i=0; i<nCategories; i++){
    // Use unique_ptr so that all MVAs are properly cleaned up
    // in the destructor
    edm::FileInPath weightFile( weightFileNames[i] );
    _gbrForests.push_back( createSingleReader(i, weightFile ) );
  }
}

PhotonMVAEstimatorRun2Phys14NonTrig::
~PhotonMVAEstimatorRun2Phys14NonTrig() {
}

float PhotonMVAEstimatorRun2Phys14NonTrig::
mvaValue( const edm::Ptr<reco::Candidate>& particle, const edm::Event& iEvent) const {
  
  const int iCategory = findCategory( particle );
  const std::vector<float> vars = std::move( fillMVAVariables( particle, iEvent ) );  
  const float result = _gbrForests.at(iCategory)->GetClassifier(vars.data());

  // DEBUG
  constexpr bool debug = false;
  if( debug ){
    printf("Printout of the photon variable inputs for MVA:\n");
    printf("  varPhi_           %f\n", vars[0]   );
    printf("  varR9_            %f\n", vars[1]   ); 
    printf("  varSieie_         %f\n", vars[2]   );
    printf("  varSieip_         %f\n", vars[3]   ); 
    printf("  varE1x3overE5x5_  %f\n", vars[4]   ); 
    printf("  varE2x2overE5x5_  %f\n", vars[5]   ); 
    printf("  varE2x5overE5x5_  %f\n", vars[6]   ); 
    printf("  varSCEta_         %f\n", vars[7]   ); 
    printf("  varRawE_          %f\n", vars[8]   ); 
    printf("  varSCEtaWidth_    %f\n", vars[9]   ); 
    printf("  varSCPhiWidth_    %f\n", vars[10]  ); 
    printf("  varRho_           %f\n", vars[11]  );
    printf("  varPhoIsoRaw_     %f\n", vars[12]  );
    printf("  varChIsoRaw_      %f\n", vars[13]  ); 
    printf("  varWorstChRaw_    %f\n", vars[14]  );
    if( isEndcapCategory( iCategory ) ) {
      printf("  varESEnOverRawE_  %f\n", vars[15]  ); // for endcap MVA only
      printf("  varESEffSigmaRR_  %f\n", vars[16]  ); // for endcap MVA only
      // The spectators
      printf("  varPt_    %f\n", vars[17]          ); 
      printf("  varEta_  %f\n",  vars[18]          );
    } else {
      // The spectators
      printf("  varPt_    %f\n", vars[15]          ); 
      printf("  varEta_  %f\n",  vars[16]          );
    }    
  }

  return result;
}

int PhotonMVAEstimatorRun2Phys14NonTrig::findCategory( const edm::Ptr<reco::Candidate>& particle) const {
  
  // Try to cast the particle into a reco particle.
  // This should work for both reco and pat.
  const edm::Ptr<reco::Photon> phoRecoPtr(particle);
  if( phoRecoPtr.isNull() )
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::Photon or pat::Photon," << std::endl
      << " but appears to be neither" << std::endl;

  const float eta = phoRecoPtr->superCluster()->eta();

  //
  // Determine the category
  //
  int  iCategory = UNDEFINED;
  constexpr float ebeeSplit = 1.479; // division between barrel and endcap

  if( std::abs(eta) < ebeeSplit )  
    iCategory = CAT_EB;

  if( std::abs(eta) >= ebeeSplit ) 
    iCategory = CAT_EE;

  return iCategory;
}

bool PhotonMVAEstimatorRun2Phys14NonTrig::
isEndcapCategory(int category ) const {

  // For this specific MVA the function is trivial, but kept for possible
  // future evolution to an MVA with more categories in eta
  bool isEndcap = false;
  if( category == CAT_EE )
    isEndcap = true;

  return isEndcap;
}


std::unique_ptr<const GBRForest> PhotonMVAEstimatorRun2Phys14NonTrig::
createSingleReader(const int iCategory, const edm::FileInPath &weightFile) {

  //
  // Create the reader  
  //
  TMVA::Reader tmpTMVAReader( "!Color:Silent:Error" );

  //
  // Configure all variables and spectators. Note: the order and names
  // must match what is found in the xml weights file!
  //
  tmpTMVAReader.AddVariable("recoPhi"   , &_allMVAVars.varPhi);
  tmpTMVAReader.AddVariable("r9"        , &_allMVAVars.varR9);
  tmpTMVAReader.AddVariable("sieie_2012", &_allMVAVars.varSieie);
  tmpTMVAReader.AddVariable("sieip_2012", &_allMVAVars.varSieip);
  tmpTMVAReader.AddVariable("e1x3_2012/e5x5_2012"        , &_allMVAVars.varE1x3overE5x5);
  tmpTMVAReader.AddVariable("e2x2_2012/e5x5_2012"        , &_allMVAVars.varE2x2overE5x5);
  tmpTMVAReader.AddVariable("e2x5_2012/e5x5_2012"        , &_allMVAVars.varE2x5overE5x5);
  tmpTMVAReader.AddVariable("recoSCEta" , &_allMVAVars.varSCEta);
  tmpTMVAReader.AddVariable("rawE"      , &_allMVAVars.varRawE);
  tmpTMVAReader.AddVariable("scEtaWidth", &_allMVAVars.varSCEtaWidth);
  tmpTMVAReader.AddVariable("scPhiWidth", &_allMVAVars.varSCPhiWidth);

  // Endcap only variables
  if( isEndcapCategory(iCategory) ){
    tmpTMVAReader.AddVariable("esEn/rawE" , &_allMVAVars.varESEnOverRawE);
    tmpTMVAReader.AddVariable("esRR"      , &_allMVAVars.varESEffSigmaRR);
  }

  // Pileup
  tmpTMVAReader.AddVariable("rho"       , &_allMVAVars.varRho);

  // Isolations
  tmpTMVAReader.AddVariable("phoIsoRaw" , &_allMVAVars.varPhoIsoRaw);
  tmpTMVAReader.AddVariable("chIsoRaw"  , &_allMVAVars.varChIsoRaw);
  tmpTMVAReader.AddVariable("chWorstRaw", &_allMVAVars.varWorstChRaw);

  // Spectators
  tmpTMVAReader.AddSpectator("recoPt" , &_allMVAVars.varPt);
  tmpTMVAReader.AddSpectator("recoEta", &_allMVAVars.varEta);

  //
  // Book the method and set up the weights file
  //
  std::unique_ptr<TMVA::IMethod> temp( tmpTMVAReader.BookMVA(_MethodName , weightFile.fullPath() ) );

  return std::unique_ptr<const GBRForest>( new GBRForest( dynamic_cast<TMVA::MethodBDT*>( tmpTMVAReader.FindMVA(_MethodName) ) ) );
}

// A function that should work on both pat and reco objects
std::vector<float> PhotonMVAEstimatorRun2Phys14NonTrig::fillMVAVariables(const edm::Ptr<reco::Candidate>& particle, const edm::Event& iEvent) const {
  // 
  // Declare all value maps corresponding to the above tokens
  //
  edm::Handle<edm::ValueMap<float> > full5x5SigmaIEtaIEtaMap;
  edm::Handle<edm::ValueMap<float> > full5x5SigmaIEtaIPhiMap;
  edm::Handle<edm::ValueMap<float> > full5x5E1x3Map;
  edm::Handle<edm::ValueMap<float> > full5x5E2x2Map;
  edm::Handle<edm::ValueMap<float> > full5x5E2x5MaxMap;
  edm::Handle<edm::ValueMap<float> > full5x5E5x5Map;
  edm::Handle<edm::ValueMap<float> > esEffSigmaRRMap;
  //
  edm::Handle<edm::ValueMap<float> > phoChargedIsolationMap;
  edm::Handle<edm::ValueMap<float> > phoPhotonIsolationMap;
  edm::Handle<edm::ValueMap<float> > phoWorstChargedIsolationMap;

  // Rho will be pulled from the event content  
  edm::Handle<double> rho;
  
  // Get the full5x5 and ES maps
  iEvent.getByLabel(_full5x5SigmaIEtaIEtaMapLabel, full5x5SigmaIEtaIEtaMap);
  iEvent.getByLabel(_full5x5SigmaIEtaIPhiMapLabel, full5x5SigmaIEtaIPhiMap);
  iEvent.getByLabel(_full5x5E1x3MapLabel, full5x5E1x3Map);
  iEvent.getByLabel(_full5x5E2x2MapLabel, full5x5E2x2Map);
  iEvent.getByLabel(_full5x5E2x5MaxMapLabel, full5x5E2x5MaxMap);
  iEvent.getByLabel(_full5x5E5x5MapLabel, full5x5E5x5Map);
  iEvent.getByLabel(_esEffSigmaRRMapLabel, esEffSigmaRRMap);

  // Get the isolation maps
  iEvent.getByLabel(_phoChargedIsolationLabel, phoChargedIsolationMap);
  iEvent.getByLabel(_phoPhotonIsolationLabel, phoPhotonIsolationMap);
  iEvent.getByLabel(_phoWorstChargedIsolationLabel, phoWorstChargedIsolationMap);

  // Get rho
  iEvent.getByLabel(_rhoLabel,rho);

  // Make sure everything is retrieved successfully
  if(! (full5x5SigmaIEtaIEtaMap.isValid()
	&& full5x5SigmaIEtaIPhiMap.isValid()
	&& full5x5E1x3Map.isValid()
	&& full5x5E2x2Map.isValid()
	&& full5x5E2x5MaxMap.isValid()
	&& full5x5E5x5Map.isValid()
	&& esEffSigmaRRMap.isValid()
	&& phoChargedIsolationMap.isValid()
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
  const edm::Ptr<reco::Photon> phoRecoPtr = ( edm::Ptr<reco::Photon> )particle;
  AllVariables allMVAVars;
  if( phoRecoPtr.isNull() )
    throw cms::Exception("MVA failure: ")
      << " given particle is expected to be reco::Photon or pat::Photon," << std::endl
      << " but appears to be neither" << std::endl;

  // Both pat and reco particles have exactly the same accessors.
  auto superCluster = phoRecoPtr->superCluster();
  // Full 5x5 cluster shapes. We could take some of this directly from
  // the photon object, but some of these are not available.
  float e1x3 = (*full5x5E1x3Map   )[ phoRecoPtr ];
  float e2x2 = (*full5x5E2x2Map   )[ phoRecoPtr ];
  float e2x5 = (*full5x5E2x5MaxMap)[ phoRecoPtr ];
  float e5x5 = (*full5x5E5x5Map   )[ phoRecoPtr ];

  allMVAVars.varPhi          = phoRecoPtr->phi();
  allMVAVars.varR9           = phoRecoPtr->r9() ;
  allMVAVars.varSieie        = (*full5x5SigmaIEtaIEtaMap)[ phoRecoPtr ]; // in principle, in the photon object as well
  allMVAVars.varSieip        = (*full5x5SigmaIEtaIPhiMap)[ phoRecoPtr ]; // not in the photon object
  allMVAVars.varE1x3overE5x5 = e1x3/e5x5;
  allMVAVars.varE2x2overE5x5 = e2x2/e5x5;
  allMVAVars.varE2x5overE5x5 = e2x5/e5x5;
  allMVAVars.varSCEta        = superCluster->eta(); 
  allMVAVars.varRawE         = superCluster->rawEnergy(); 
  allMVAVars.varSCEtaWidth   = superCluster->etaWidth(); 
  allMVAVars.varSCPhiWidth   = superCluster->phiWidth(); 
  allMVAVars.varESEnOverRawE = superCluster->preshowerEnergy() / superCluster->rawEnergy();
  allMVAVars.varESEffSigmaRR = (*esEffSigmaRRMap)[ phoRecoPtr ];
  allMVAVars.varRho          = *rho; 
  allMVAVars.varPhoIsoRaw    = (*phoPhotonIsolationMap)[phoRecoPtr];  
  allMVAVars.varChIsoRaw     = (*phoChargedIsolationMap)[phoRecoPtr];
  allMVAVars.varWorstChRaw   = (*phoWorstChargedIsolationMap)[phoRecoPtr];
  // Declare spectator vars
  allMVAVars.varPt = phoRecoPtr->pt(); 
  allMVAVars.varEta = phoRecoPtr->eta();

  constrainMVAVariables(allMVAVars);

  std::vector<float> vars;
  if( isEndcapCategory( findCategory( particle ) ) ) {
    vars = std::move( packMVAVariables(allMVAVars.varPhi,
                                       allMVAVars.varR9,
                                       allMVAVars.varSieie,
                                       allMVAVars.varSieip,
                                       allMVAVars.varE1x3overE5x5,
                                       allMVAVars.varE2x2overE5x5,
                                       allMVAVars.varE2x5overE5x5,
                                       allMVAVars.varSCEta,
                                       allMVAVars.varRawE,
                                       allMVAVars.varSCEtaWidth,
                                       allMVAVars.varSCPhiWidth,
                                       allMVAVars.varESEnOverRawE,
                                       allMVAVars.varESEffSigmaRR,
                                       allMVAVars.varRho,
                                       allMVAVars.varPhoIsoRaw,
                                       allMVAVars.varChIsoRaw,
                                       allMVAVars.varWorstChRaw,
                                       // Declare spectator vars
                                       allMVAVars.varPt,
                                       allMVAVars.varEta) 
                      ); 
  } else {
    vars = std::move( packMVAVariables(allMVAVars.varPhi,
                                       allMVAVars.varR9,
                                       allMVAVars.varSieie,
                                       allMVAVars.varSieip,
                                       allMVAVars.varE1x3overE5x5,
                                       allMVAVars.varE2x2overE5x5,
                                       allMVAVars.varE2x5overE5x5,
                                       allMVAVars.varSCEta,
                                       allMVAVars.varRawE,
                                       allMVAVars.varSCEtaWidth,
                                       allMVAVars.varSCPhiWidth,
                                       allMVAVars.varRho,
                                       allMVAVars.varPhoIsoRaw,
                                       allMVAVars.varChIsoRaw,
                                       allMVAVars.varWorstChRaw,
                                       // Declare spectator vars
                                       allMVAVars.varPt,
                                       allMVAVars.varEta) 
                      );
  }

  return vars;
}

void PhotonMVAEstimatorRun2Phys14NonTrig::constrainMVAVariables(AllVariables&) const {

  // Check that variables do not have crazy values
  
  // This function is currently empty as this specific MVA was not
  // developed with restricting variables to specific physical ranges.
  return;

}

void PhotonMVAEstimatorRun2Phys14NonTrig::setConsumes(edm::ConsumesCollector&& cc) const {
  cc.consumes<edm::ValueMap<float> >(_full5x5SigmaIEtaIEtaMapLabel);
  cc.consumes<edm::ValueMap<float> >(_full5x5SigmaIEtaIPhiMapLabel); 
  cc.consumes<edm::ValueMap<float> >(_full5x5E1x3MapLabel); 
  cc.consumes<edm::ValueMap<float> >(_full5x5E2x2MapLabel);
  cc.consumes<edm::ValueMap<float> >(_full5x5E2x5MaxMapLabel);
  cc.consumes<edm::ValueMap<float> >(_full5x5E5x5MapLabel);
  cc.consumes<edm::ValueMap<float> >(_esEffSigmaRRMapLabel);
  cc.consumes<edm::ValueMap<float> >(_phoChargedIsolationLabel);
  cc.consumes<edm::ValueMap<float> >(_phoPhotonIsolationLabel);
  cc.consumes<edm::ValueMap<float> >( _phoWorstChargedIsolationLabel);
  cc.consumes<double>(_rhoLabel);
}
