#include "RecoEgamma/PhotonIdentification/plugins/PhotonMVAEstimatorRun2Spring16NonTrig.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "TMVA/MethodBDT.h"

PhotonMVAEstimatorRun2Spring16NonTrig::PhotonMVAEstimatorRun2Spring16NonTrig(const edm::ParameterSet& conf):
  AnyMVAEstimatorRun2Base(conf),
  _MethodName("BDTG method"),
  _useValueMaps(conf.getParameter<bool>("useValueMaps")),
  _full5x5SigmaIEtaIEtaMapLabel(conf.getParameter<edm::InputTag>("full5x5SigmaIEtaIEtaMap")), 
  _full5x5SigmaIEtaIPhiMapLabel(conf.getParameter<edm::InputTag>("full5x5SigmaIEtaIPhiMap")), 
  _full5x5E2x2MapLabel(conf.getParameter<edm::InputTag>("full5x5E2x2Map")), 
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
  if( debug ){
    printf("Printout of the photon variable inputs for MVA:\n");
    printf("  varSCPhi_            %f\n", vars[1]   );
    printf("  varR9_            %f\n", vars[2]   ); 
    printf("  varSieie_         %f\n", vars[3]   );
    printf("  varSieip_         %f\n", vars[4]   ); 
    printf("  varE2x2overE5x5_  %f\n", vars[5]   ); 
    printf("  varSCEta_         %f\n", vars[6]   ); 
    printf("  varRawE_          %f\n", vars[7]   ); 
    printf("  varSCEtaWidth_    %f\n", vars[8]   ); 
    printf("  varSCPhiWidth_    %f\n", vars[9]  ); 
    printf("  varRho_           %f\n", vars[10]  );
    if( !isEndcapCategory( iCategory ) ) {
      printf("  varPhoIsoRaw_     %f\n", vars[11]  );
    }
    printf("  varChIsoRaw_      %f\n", vars[12]  ); 
    printf("  varWorstChRaw_    %f\n", vars[13]  );
    if( isEndcapCategory( iCategory ) ) {
      printf("  varESEnOverRawE_  %f\n", vars[14]  ); // for endcap MVA only
      printf("  varESEffSigmaRR_  %f\n", vars[15]  ); // for endcap MVA only
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

  //EB:

  /*
  <Variables NVar="13">
    <Variable VarIndex="0" Expression="scPhi" Label="scPhi" Title="scPhi" Unit="" Internal="scPhi" Type="F" Min="-3.14157844e+00" Max="3.14158368e+00"/>
    <Variable VarIndex="1" Expression="r9" Label="r9" Title="r9" Unit="" Internal="r9" Type="F" Min="1.01813994e-01" Max="1.00000012e+00"/>
    <Variable VarIndex="2" Expression="sigmaIetaIeta" Label="sigmaIetaIeta" Title="sigmaIetaIeta" Unit="" Internal="sigmaIetaIeta" Type="F" Min="1.01688434e-04" Max="3.03453747e-02"/>
    <Variable VarIndex="3" Expression="covIEtaIPhi" Label="covIEtaIPhi" Title="covIEtaIPhi" Unit="" Internal="covIEtaIPhi" Type="F" Min="-5.92368655e-04" Max="9.99900024e+02"/>
    <Variable VarIndex="4" Expression="s4" Label="s4" Title="s4" Unit="" Internal="s4" Type="F" Min="-3.24029237e-01" Max="1.00077236e+00"/>
    <Variable VarIndex="5" Expression="scEta" Label="scEta" Title="scEta" Unit="" Internal="scEta" Type="F" Min="-1.44419956e+00" Max="1.44419491e+00"/>
    <Variable VarIndex="6" Expression="SCRawE" Label="SCRawE" Title="SCRawE" Unit="" Internal="SCRawE" Type="F" Min="7.66105461e+00" Max="1.38820911e+03"/>
    <Variable VarIndex="7" Expression="etaWidth" Label="etaWidth" Title="etaWidth" Unit="" Internal="etaWidth" Type="F" Min="8.18247732e-04" Max="1.63682073e-01"/>
    <Variable VarIndex="8" Expression="phiWidth" Label="phiWidth" Title="phiWidth" Unit="" Internal="phiWidth" Type="F" Min="3.52814413e-06" Max="5.56705654e-01"/>
    <Variable VarIndex="9" Expression="rho" Label="rho" Title="rho" Unit="" Internal="rho" Type="F" Min="0.00000000e+00" Max="4.32884483e+01"/>
    <Variable VarIndex="10" Expression="CITK_isoPhotons" Label="CITK_isoPhotons" Title="CITK_isoPhotons" Unit="" Internal="CITK_isoPhotons" Type="F" Min="0.00000000e+00" Max="2.02342285e+02"/>
    <Variable VarIndex="11" Expression="CITK_isoChargedHad" Label="CITK_isoChargedHad" Title="CITK_isoChargedHad" Unit="" Internal="CITK_isoChargedHad" Type="F" Min="0.00000000e+00" Max="2.80166992e+02"/>
    <Variable VarIndex="12" Expression="chgIsoWrtWorstVtx" Label="chgIsoWrtWorstVtx" Title="chgIsoWrtWorstVtx" Unit="" Internal="chgIsoWrtWorstVtx" Type="F" Min="-1.00000000e+00" Max="2.56465820e+02"/>
  </Variables>
  */

  //EE:

  /*
  <Variables NVar="14">
    <Variable VarIndex="0" Expression="scPhi" Label="scPhi" Title="scPhi" Unit="" Internal="scPhi" Type="F" Min="-3.14157391e+00" Max="3.14155674e+00"/>
    <Variable VarIndex="1" Expression="r9" Label="r9" Title="r9" Unit="" Internal="r9" Type="F" Min="1.07955128e-01" Max="1.00000024e+00"/>
    <Variable VarIndex="2" Expression="sigmaIetaIeta" Label="sigmaIetaIeta" Title="sigmaIetaIeta" Unit="" Internal="sigmaIetaIeta" Type="F" Min="1.75891572e-03" Max="8.14865530e-02"/>
    <Variable VarIndex="3" Expression="covIEtaIPhi" Label="covIEtaIPhi" Title="covIEtaIPhi" Unit="" Internal="covIEtaIPhi" Type="F" Min="-2.39617634e-03" Max="2.66308710e-03"/>
    <Variable VarIndex="4" Expression="s4" Label="s4" Title="s4" Unit="" Internal="s4" Type="F" Min="2.86898892e-02" Max="9.89912212e-01"/>
    <Variable VarIndex="5" Expression="scEta" Label="scEta" Title="scEta" Unit="" Internal="scEta" Type="F" Min="-2.49999595e+00" Max="2.49999166e+00"/>
    <Variable VarIndex="6" Expression="SCRawE" Label="SCRawE" Title="SCRawE" Unit="" Internal="SCRawE" Type="F" Min="6.33558369e+00" Max="2.86951831e+03"/>
    <Variable VarIndex="7" Expression="etaWidth" Label="etaWidth" Title="etaWidth" Unit="" Internal="etaWidth" Type="F" Min="1.48537615e-03" Max="3.33718777e-01"/>
    <Variable VarIndex="8" Expression="phiWidth" Label="phiWidth" Title="phiWidth" Unit="" Internal="phiWidth" Type="F" Min="5.76065679e-04" Max="9.32549536e-01"/>
    <Variable VarIndex="9" Expression="rho" Label="rho" Title="rho" Unit="" Internal="rho" Type="F" Min="0.00000000e+00" Max="4.19737549e+01"/>
    <Variable VarIndex="10" Expression="CITK_isoChargedHad" Label="CITK_isoChargedHad" Title="CITK_isoChargedHad" Unit="" Internal="CITK_isoChargedHad" Type="F" Min="0.00000000e+00" Max="2.03726562e+02"/>
    <Variable VarIndex="11" Expression="chgIsoWrtWorstVtx" Label="chgIsoWrtWorstVtx" Title="chgIsoWrtWorstVtx" Unit="" Internal="chgIsoWrtWorstVtx" Type="F" Min="0.00000000e+00" Max="1.91058594e+02"/>
    <Variable VarIndex="12" Expression="esEffSigmaRR" Label="esEffSigmaRR" Title="esEffSigmaRR" Unit="" Internal="esEffSigmaRR" Type="F" Min="0.00000000e+00" Max="1.41421356e+01"/>
    <Variable VarIndex="13" Expression="esEnergy/SCRawE" Label="esEnergy/SCRawE" Title="esEnergy/SCRawE" Unit="" Internal="esEnergy_D_SCRawE" Type="F" Min="0.00000000e+00" Max="4.66867542e+00"/>
  </Variables>
  */

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

  // Get the full5x5 and ES maps
  if( _useValueMaps ) {
    iEvent.getByLabel(_full5x5SigmaIEtaIEtaMapLabel, full5x5SigmaIEtaIEtaMap);
    iEvent.getByLabel(_full5x5SigmaIEtaIPhiMapLabel, full5x5SigmaIEtaIPhiMap);
    iEvent.getByLabel(_full5x5E2x2MapLabel, full5x5E2x2Map);
    iEvent.getByLabel(_full5x5E5x5MapLabel, full5x5E5x5Map);
    iEvent.getByLabel(_esEffSigmaRRMapLabel, esEffSigmaRRMap);
  }

  // Get the isolation maps
  iEvent.getByLabel(_phoChargedIsolationLabel, phoChargedIsolationMap);
  iEvent.getByLabel(_phoPhotonIsolationLabel, phoPhotonIsolationMap);
  iEvent.getByLabel(_phoWorstChargedIsolationLabel, phoWorstChargedIsolationMap);

  // Get rho
  iEvent.getByLabel(_rhoLabel,rho);

  // Make sure everything is retrieved successfully
  if(! ( ( !_useValueMaps || ( full5x5SigmaIEtaIEtaMap.isValid()
                               && full5x5SigmaIEtaIPhiMap.isValid()
                               && full5x5E2x2Map.isValid()
                               && full5x5E5x5Map.isValid()
                               && esEffSigmaRRMap.isValid() ) )
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

  if( _useValueMaps ) { //(before 752)
    // in principle, in the photon object as well
    // not in the photon object 
    e2x2 = (*full5x5E2x2Map   )[ phoRecoPtr ];
    e5x5 = (*full5x5E5x5Map   )[ phoRecoPtr ];
    full5x5_sigmaIetaIeta = (*full5x5SigmaIEtaIEtaMap)[ phoRecoPtr ];
    full5x5_sigmaIetaIphi = (*full5x5SigmaIEtaIPhiMap)[ phoRecoPtr ];
    effSigmaRR = (*esEffSigmaRRMap)[ phoRecoPtr ];
  } else {
    // from 753
    const auto& full5x5_pss = phoRecoPtr->full5x5_showerShapeVariables();
    e2x2 = full5x5_pss.e2x2;
    e5x5 = full5x5_pss.e5x5;
    full5x5_sigmaIetaIeta = full5x5_pss.sigmaIetaIeta;
    full5x5_sigmaIetaIphi = full5x5_pss.sigmaIetaIphi;
    effSigmaRR = full5x5_pss.effSigmaRR;
  }

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

  constrainMVAVariables(allMVAVars);

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
                                       allMVAVars.varESEnOverRawE,
                                       allMVAVars.varESEffSigmaRR,
                                       allMVAVars.varRho,
                                       allMVAVars.varChIsoRaw,
                                       allMVAVars.varWorstChRaw
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
  if( _useValueMaps ) {
    cc.consumes<edm::ValueMap<float> >(_full5x5SigmaIEtaIEtaMapLabel);
    cc.consumes<edm::ValueMap<float> >(_full5x5SigmaIEtaIPhiMapLabel); 
    cc.consumes<edm::ValueMap<float> >(_full5x5E2x2MapLabel);
    cc.consumes<edm::ValueMap<float> >(_full5x5E5x5MapLabel);
    cc.consumes<edm::ValueMap<float> >(_esEffSigmaRRMapLabel);
  }
  cc.consumes<edm::ValueMap<float> >(_phoChargedIsolationLabel);
  cc.consumes<edm::ValueMap<float> >(_phoPhotonIsolationLabel);
  cc.consumes<edm::ValueMap<float> >( _phoWorstChargedIsolationLabel);
  cc.consumes<double>(_rhoLabel);
}

DEFINE_EDM_PLUGIN(AnyMVAEstimatorRun2Factory,
                  PhotonMVAEstimatorRun2Spring16NonTrig,
                  "PhotonMVAEstimatorRun2Spring16NonTrig");

