#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"
#include <iomanip> 

class L1TCaloParamsViewer: public edm::EDAnalyzer {
private:
    bool printPUSParams;
    bool printTauCalibLUT;
    bool printJetCalibLUT;
    bool printJetCalibPar;
    bool printEgCalibLUT;
    bool printEgIsoLUT;

    std::string hash(void *buf, size_t len) const ;

public:
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    explicit L1TCaloParamsViewer(const edm::ParameterSet& pset) : edm::EDAnalyzer(){
       printPUSParams   = pset.getUntrackedParameter<bool>("printPUSParams",  false);
       printTauCalibLUT = pset.getUntrackedParameter<bool>("printTauCalibLUT",false);
       printJetCalibLUT = pset.getUntrackedParameter<bool>("printJetCalibLUT",false);
       printJetCalibPar = pset.getUntrackedParameter<bool>("printJetCalibParams",false);
       printEgCalibLUT  = pset.getUntrackedParameter<bool>("printEgCalibLUT", false);
       printEgIsoLUT    = pset.getUntrackedParameter<bool>("printEgIsoLUT",   false);
    }

    virtual ~L1TCaloParamsViewer(void){}
};

#include <openssl/sha.h>
#include <math.h>
#include <iostream>
using namespace std;

std::string L1TCaloParamsViewer::hash(void *buf, size_t len) const {
    char tmp[SHA_DIGEST_LENGTH*2+1];
    bzero(tmp,sizeof(tmp));
    SHA_CTX ctx;
    if( !SHA1_Init( &ctx ) )
        throw cms::Exception("L1TCaloParamsViewer::hash")<<"SHA1 initialization error";

    if( !SHA1_Update( &ctx, buf, len ) )
        throw cms::Exception("L1TCaloParamsViewer::hash")<<"SHA1 processing error";

    unsigned char hash[SHA_DIGEST_LENGTH];
    if( !SHA1_Final(hash, &ctx) )
        throw cms::Exception("L1TCaloParamsViewer::hash")<<"SHA1 finalization error";

    // re-write bytes in hex
    for(unsigned int i=0; i<20; i++)
        ::sprintf(&tmp[i*2], "%02x", hash[i]);

    tmp[20*2] = 0;
    return std::string(tmp);
}

void L1TCaloParamsViewer::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup){

    edm::ESHandle<l1t::CaloParams> handle1;
    evSetup.get<L1TCaloParamsRcd>().get( handle1 ) ;
    boost::shared_ptr<l1t::CaloParams> ptr(new l1t::CaloParams(*(handle1.product ())));

    l1t::CaloParamsHelper *ptr1 = 0;
    ptr1 = (l1t::CaloParamsHelper*) (&(*ptr));

    edm::LogInfo("")<<"L1TCaloParamsViewer:";

    cout<<endl<<" Towers: "<<endl;
    cout<<"  towerLsbH=       "<<ptr1->towerLsbH()<<endl;
    cout<<"  towerLsbE=       "<<ptr1->towerLsbE()<<endl;
    cout<<"  towerLsbSum=     "<<ptr1->towerLsbSum()<<endl;
    cout<<"  towerNBitsH=     "<<ptr1->towerNBitsH()<<endl;
    cout<<"  towerNBitsE=     "<<ptr1->towerNBitsE()<<endl;
    cout<<"  towerNBitsSum=   "<<ptr1->towerNBitsSum()<<endl;
    cout<<"  towerNBitsRatio= "<<ptr1->towerNBitsRatio()<<endl;
    cout<<"  towerMaskE=      "<<ptr1->towerMaskE()<<endl;
    cout<<"  towerMaskH=      "<<ptr1->towerMaskH()<<endl;
    cout<<"  towerMaskSum=    "<<ptr1->towerMaskSum()<<endl;
    cout<<"  towerEncoding=    "<<ptr1->doTowerEncoding()<<endl;

    cout<<endl<<" Regions: "<<endl;
    cout<<"  regionLsb=       "<<ptr1->regionLsb()<<endl;
    cout<<"  regionPUSType=   "<<ptr1->regionPUSType()<<endl;
    cout<<"  regionPUSParams= ["<<ptr1->regionPUSParams().size()<<"] ";
    float pusParams[ptr1->regionPUSParams().size()];

    for(unsigned int i=0; i<ptr1->regionPUSParams().size(); i++){
        pusParams[i] = ceil(2*ptr1->regionPUSParams()[i]);
        if( printPUSParams ) cout<<"   "<<ceil(2*pusParams[i])<<endl;
    }

    if( ptr1->regionPUSParams().size() )
        cout << hash(pusParams, sizeof(float)*ptr1->regionPUSParams().size()) << endl;
    else cout<<endl;


    cout<<endl<<" EG: "<<endl;
    cout<<"  egLsb=                  "<<ptr1->egLsb()<<endl;
    cout<<"  egSeedThreshold=        "<<ptr1->egSeedThreshold()<<endl;
    cout<<"  egNeighbourThreshold=   "<<ptr1->egNeighbourThreshold()<<endl;
    cout<<"  egHcalThreshold=        "<<ptr1->egHcalThreshold()<<endl;

    if( !ptr1->egTrimmingLUT()->empty() ){
        cout<<"  egTrimmingLUT=          ["<<ptr1->egTrimmingLUT()->maxSize()<<"] "<<flush;
        int egTrimming[ptr1->egTrimmingLUT()->maxSize()];
        for(unsigned int i=0; i<ptr1->egTrimmingLUT()->maxSize(); i++) egTrimming[i] = ptr1->egTrimmingLUT()->data(i);
        cout << hash( egTrimming, sizeof(int)*ptr1->egTrimmingLUT()->maxSize() ) << endl;
    } else {
        cout<<"  egTrimmingLUT=          [0] "<<endl;
    }

    cout<<"  egMaxHcalEt=            "<<ptr1->egMaxHcalEt()<<endl;
    cout<<"  egMaxPtHOverE=          "<<ptr1->egMaxPtHOverE()<<endl;
    cout<<"  egMaxHcalEt=            "<<ptr1->egMaxHcalEt()<<endl;
    cout<<"  egMinPtJetIsolation=    "<<ptr1->egMinPtJetIsolation()<<endl;
    cout<<"  egMaxPtJetIsolation=    "<<ptr1->egMaxPtJetIsolation()<<endl;
    cout<<"  egMinPtHOverEIsolation= "<<ptr1->egMinPtHOverEIsolation()<<endl;
    cout<<"  egMaxPtHOverEIsolation= "<<ptr1->egMaxPtHOverEIsolation()<<endl;

    if( !ptr1->egMaxHOverELUT()->empty() ){
        cout<<"  egMaxHOverELUT=         ["<<ptr1->egMaxHOverELUT()->maxSize()<<"] ";
        int egMaxHOverE[ptr1->egMaxHOverELUT()->maxSize()];
        for(unsigned int i=0; i<ptr1->egMaxHOverELUT()->maxSize(); i++) egMaxHOverE[i] = ptr1->egMaxHOverELUT()->data(i);
        cout << hash( egMaxHOverE, sizeof(int)*ptr1->egMaxHOverELUT()->maxSize() ) << endl;
    } else {
        cout<<"  egMaxHOverELUT=         [0]"<<endl;
    }

    if( !ptr1->egCompressShapesLUT()->empty() ){
        cout<<"  egCompressShapesLUT=    ["<<ptr1->egCompressShapesLUT()->maxSize()<<"] ";
        int egCompressShapes[ptr1->egCompressShapesLUT()->maxSize()];
        for(unsigned int i=0; i<ptr1->egCompressShapesLUT()->maxSize(); i++) egCompressShapes[i] = ptr1->egCompressShapesLUT()->data(i);
        cout << hash( egCompressShapes, sizeof(int)*ptr1->egCompressShapesLUT()->maxSize() ) << endl;
    } else {
        cout<<"  egCompressShapesLUT=    [0]"<<endl;
    }

    //    cout<<"  egShapeIdType=          "<<ptr1->egShapeIdType()<<endl;
    //    cout<<"  egShapeIdVersion=       "<<ptr1->egShapeIdVersion()<<endl;
    if( !ptr1->egShapeIdLUT()->empty() ){
        cout<<"  egShapeIdLUT=           ["<<ptr1->egShapeIdLUT()->maxSize()<<"] "<<flush;
        int egShapeId[ptr1->egShapeIdLUT()->maxSize()];
        for(unsigned int i=0; i<ptr1->egShapeIdLUT()->maxSize(); i++) egShapeId[i] = ptr1->egShapeIdLUT()->data(i);
        cout << hash( egShapeId, sizeof(int)*ptr1->egShapeIdLUT()->maxSize() )<<endl;
    } else {
        cout<<"  egShapeIdLUT=           [0]"<<endl;
    }

    cout<<"  egPUSType=              "<<ptr1->egPUSType()<<endl;

    if( !ptr1->egIsolationLUT()->empty() ){
        cout<<"  egIsoLUT=               ["<<ptr1->egIsolationLUT()->maxSize()<<"] "<<flush;
        int egIsolation[ptr1->egIsolationLUT()->maxSize()];
        for(unsigned int i=0; i<ptr1->egIsolationLUT()->maxSize(); i++) egIsolation[i] = ptr1->egIsolationLUT()->data(i);
        cout << hash( egIsolation, sizeof(int)*ptr1->egIsolationLUT()->maxSize() ) << endl;
        if( printEgIsoLUT )
            for(unsigned int i=0; i<ptr1->egIsolationLUT()->maxSize(); i++)
            cout<<i<<" " << egIsolation[i]<<endl;
    } else {
        cout<<"  egIsoLUT=               [0]"<<endl;
    }

    cout<<"  egIsoAreaNrTowersEta=   "<<ptr1->egIsoAreaNrTowersEta()<<endl;
    cout<<"  egIsoAreaNrTowersPhi=   "<<ptr1->egIsoAreaNrTowersPhi()<<endl;
    cout<<"  egIsoVetoNrTowersPhi=   "<<ptr1->egIsoVetoNrTowersPhi()<<endl;
    cout<<"  egPUSParams=            ["<<ptr1->egPUSParams().size()<<"] "<<flush;
    float egPUSParams[ptr1->egPUSParams().size()];
    for(unsigned int i=0; i<ptr1->egPUSParams().size(); i++) egPUSParams[i] = ptr1->egPUSParams()[i];

    if( ptr1->egPUSParams().size() )
       cout << hash( egPUSParams, sizeof(float)*ptr1->egPUSParams().size() ) << endl;
    else cout<<endl;

    cout<<"  egCalibrationType=      "<<ptr1->egCalibrationType()<<endl;
    //    cout<<"  egCalibrationVersion=   "<<ptr1->egCalibrationVersion()<<endl;
    if( !ptr1->egCalibrationLUT()->empty() ){
        cout<<"  egCalibrationLUT=       ["<<ptr1->egCalibrationLUT()->maxSize()<<"] "<<flush;
        int egCalibration[ptr1->egCalibrationLUT()->maxSize()];
        for(unsigned int i=0; i<ptr1->egCalibrationLUT()->maxSize(); i++) egCalibration[i] = ptr1->egCalibrationLUT()->data(i);
        cout << hash( egCalibration, sizeof(int)*ptr1->egCalibrationLUT()->maxSize() ) << endl;
        if( printEgCalibLUT )
            for(unsigned int i=0; i<ptr1->egCalibrationLUT()->maxSize(); i++)
            cout<<i<<" " << egCalibration[i]<<endl;
    } else {
        cout<<"  egCalibrationLUT=       [0]"<<endl;
    }

    cout<<endl<<" Tau: "<<endl;
    cout<<"  tauLsb=                 "<<ptr1->tauLsb()<<endl;
    cout<<"  tauSeedThreshold=       "<<ptr1->tauSeedThreshold()<<endl;
    cout<<"  tauNeighbourThreshold=  "<<ptr1->tauNeighbourThreshold()<<endl;
    cout<<"  tauMaxPtTauVeto=        "<<ptr1->tauMaxPtTauVeto()<<endl;
    cout<<"  tauMinPtJetIsolationB=  "<<ptr1->tauMinPtJetIsolationB()<<endl;
    cout<<"  tauPUSType=             "<<ptr1->tauPUSType()<<endl;
    cout<<"  tauMaxJetIsolationB=    "<<ptr1->tauMaxJetIsolationB()<<endl;
    cout<<"  tauMaxJetIsolationA=    "<<ptr1->tauMaxJetIsolationA()<<endl;
    cout<<"  tauIsoAreaNrTowersEta=  "<<ptr1->tauIsoAreaNrTowersEta()<<endl;
    cout<<"  tauIsoAreaNrTowersPhi=  "<<ptr1->tauIsoAreaNrTowersPhi()<<endl;
    cout<<"  tauIsoVetoNrTowersPhi=  "<<ptr1->tauIsoVetoNrTowersPhi()<<endl;
    if( !ptr1->tauIsolationLUT()->empty() ){
        cout<<"  tauIsoLUT=              ["<<ptr1->tauIsolationLUT()->maxSize()<<"] "<<flush;
        int tauIsolation[ptr1->tauIsolationLUT()->maxSize()];
        for(unsigned int i=0; i<ptr1->tauIsolationLUT()->maxSize(); i++) tauIsolation[i] = ptr1->tauIsolationLUT()->data(i);
        cout << hash( tauIsolation, sizeof(int)*ptr1->tauIsolationLUT()->maxSize() ) << endl;
    } else {
        cout<<"  tauIsoLUT=              [0]"<<endl;
    }

    if( !ptr1->tauCalibrationLUT()->empty() ){
        cout<<"  tauCalibrationLUT=      ["<<ptr1->tauCalibrationLUT()->maxSize()<<"] "<<flush;
        int tauCalibration[ptr1->tauCalibrationLUT()->maxSize()];
        for(unsigned int i=0; i<ptr1->tauCalibrationLUT()->maxSize(); i++)
            tauCalibration[i] = ptr1->tauCalibrationLUT()->data(i);
        cout << hash( tauCalibration, sizeof(int)*ptr1->tauCalibrationLUT()->maxSize() ) << endl;

        if( printTauCalibLUT )
            for(unsigned int i=0; i<ptr1->tauCalibrationLUT()->maxSize(); i++)
            cout<<i<<" "<<tauCalibration[i]<<endl;

    } else {
        cout<<"  tauCalibrationLUT=      [0]"<<endl;
    }

    if( !ptr1->tauEtToHFRingEtLUT()->empty() ){
        cout<<"  tauEtToHFRingEtLUT=     ["<<ptr1->tauEtToHFRingEtLUT()->maxSize()<<"] "<<flush;
        int tauEtToHFRingEt[ptr1->tauEtToHFRingEtLUT()->maxSize()];
        for(unsigned int i=0; i<ptr1->tauEtToHFRingEtLUT()->maxSize(); i++) tauEtToHFRingEt[i] = ptr1->tauEtToHFRingEtLUT()->data(i);

         cout << hash(  tauEtToHFRingEt, sizeof(int)*ptr1->tauEtToHFRingEtLUT()->maxSize() ) << endl;
    } else {
        cout<<"  tauEtToHFRingEtLUT=     [0]"<<endl;
    }

    cout<<"  isoTauEtaMin=           "<<ptr1->isoTauEtaMin()<<endl;
    cout<<"  isoTauEtaMax=           "<<ptr1->isoTauEtaMax()<<endl;
    cout<<"  tauPUSParams=           ["<<ptr1->tauPUSParams().size()<<"] "<<flush;
    float tauPUSParams[ptr1->tauPUSParams().size()];
    for(unsigned int i=0; i<ptr1->tauPUSParams().size(); i++) tauPUSParams[i] = ptr1->tauPUSParams()[i];

    if( ptr1->tauPUSParams().size() )
        cout << hash( tauPUSParams, sizeof(float)*ptr1->tauPUSParams().size()  ) << endl;
    else cout<<endl;

    cout<<endl<<" Jets: "<<endl;
    cout<<"  jetLsb=                 "<<ptr1->jetLsb()<<endl;
    cout<<"  jetSeedThreshold=       "<<ptr1->jetSeedThreshold()<<endl;
    cout<<"  jetNeighbourThreshold=  "<<ptr1->jetNeighbourThreshold()<<endl;
    cout<<"  jetPUSType=             "<<ptr1->jetPUSType()<<endl;
    cout<<"  jetCalibrationType=     "<<ptr1->jetCalibrationType()<<endl;
    cout<<"  jetCalibrationParams=   ["<<ptr1->jetCalibrationParams().size()<<"] "<<flush;
    float jetCalibrationParams[ptr1->jetCalibrationParams().size()]; // deliberately drop double precision
    for(unsigned int i=0; i<ptr1->jetCalibrationParams().size(); i++) jetCalibrationParams[i] = ptr1->jetCalibrationParams()[i];

    if( ptr1->jetCalibrationParams().size() ){
        cout << hash( jetCalibrationParams, sizeof(float)*ptr1->jetCalibrationParams().size() ) << endl;
        if( printJetCalibPar )
            for(unsigned int i=0; i<ptr1->jetCalibrationParams().size(); i++)
                cout<<i<<" " << std::setprecision(14) << jetCalibrationParams[i]<<endl;

    } else cout<<endl;

    if( !ptr1->jetCalibrationLUT()->empty() ){
        cout<<"  jetCalibrationLUT=      ["<<ptr1->jetCalibrationLUT()->maxSize()<<"] "<<flush;
        int jetCalibration[ptr1->jetCalibrationLUT()->maxSize()];
        for(unsigned int i=0; i<ptr1->jetCalibrationLUT()->maxSize(); i++)
            jetCalibration[i] = ptr1->jetCalibrationLUT()->data(i);

        cout << hash( jetCalibration, sizeof(int)*ptr1->jetCalibrationLUT()->maxSize()  ) << endl;

        if( printJetCalibLUT )
            for(unsigned int i=0; i<ptr1->jetCalibrationLUT()->maxSize(); i++)
            cout<<i<<" "<<jetCalibration[i]<<endl;

    } else {
        cout<<"  jetCalibrationLUT=      [0]"<<endl;
    }

    cout<<endl<<" Sums: "<<endl;
    cout<<"  etSumLsb=               "<<ptr1->etSumLsb()<<endl;
///    cout<<"  etSumEtaMin=            ["; for(unsigned int i=0; ptr1->etSumEtaMin(i)<0.001; i++) cout<<(i==0?"":",")<<ptr1->etSumEtaMin(i); cout<<"]"<<endl;
///    cout<<"  etSumEtaMax=            ["; for(unsigned int i=0; ptr1->etSumEtaMax(i)<0.001; i++) cout<<(i==0?"":",")<<ptr1->etSumEtaMax(i); cout<<"]"<<endl;
///    cout<<"  etSumEtThreshold=       ["; for(unsigned int i=0; ptr1->etSumEtThreshold(i)<0.001; i++) cout<<(i==0?"":",")<<ptr1->etSumEtThreshold(i); cout<<"]"<<endl;

    cout<<endl<<" HI centrality trigger: "<<endl;
    cout<<"  centralityLUT=          ["; for(unsigned int i=0; i<ptr1->centralityLUT()->maxSize(); i++) cout<<(i==0?"":",")<<ptr1->centralityLUT()->data(i); cout<<"]"<<endl;

    std::vector<int> mbt = ptr1->minimumBiasThresholds();
    cout<<"  minimumBiasThresholds=  ["; for(unsigned int i=0; i<mbt.size(); i++) cout<<mbt[i]; cout<<"]"<<endl; 

    cout<<endl<<"centralityRegionMask() = "<<ptr1->centralityRegionMask()<<endl;
    cout<<endl<<"jetRegionMask() = "<<ptr1->jetRegionMask()<<endl;
    cout<<endl<<"tauRegionMask() = "<<ptr1->tauRegionMask()<<endl;

    cout<<endl<<" HI Q2 trigger: "<<endl;
    cout<<"  q2LUT=                  ["; for(unsigned int i=0; i<ptr1->q2LUT()->maxSize(); i++) cout<<(i==0?"":",")<<ptr1->q2LUT()->data(i); cout<<"]"<<endl;

    cout<<endl<<" Layer1: "<<endl;
    std::vector<double> ecalSF = ptr1->layer1ECalScaleFactors();
    cout<<"  layer1ECalScaleFactors= ["<< ecalSF.size()<<"] "<<flush;
    float _ecalSF[ ecalSF.size() ];
    for(unsigned int i=0; i<ecalSF.size(); i++) _ecalSF[i] =  ecalSF[i];
    cout << hash( _ecalSF , sizeof(float)*ecalSF.size() ) << endl;
    //for(unsigned int i=0; i<ecalSF.size(); i++) cout<<(i==0?"":",")<<ecalSF[i]; cout<<"]"<<endl;
    std::vector<double> hcalSF = ptr1->layer1HCalScaleFactors();
    cout<<"  layer1HCalScaleFactors= ["<< hcalSF.size()<<"] "<<flush;
    float _hcalSF[ hcalSF.size() ];
    for(unsigned int i=0; i<hcalSF.size(); i++){
        // round false precision
//        double significand;
//        int    exponent;
//        significand = frexp( hcalSF[i],  &exponent );
//         _hcalSF[i] = ldexp( int(significand*10000)/10000., exponent );
         _hcalSF[i] = int(hcalSF[i]*1000.)/1000.;
    }
    cout << hash( _hcalSF, sizeof(float)*hcalSF.size() ) << endl;
    //for(unsigned int i=0; i<hcalSF.size(); i++) cout<<(i==0?"":",")<<hcalSF[i]; cout<<"]"<<endl;
    std::vector<double> hfSF   = ptr1->layer1HFScaleFactors();
    cout<<"  layer1HFScaleFactors=   ["<< hfSF.size()<<"] "<<flush;
    float _hfSF[ hfSF.size() ];
    for(unsigned int i=0; i<hfSF.size(); i++) _hfSF[i] =  hfSF[i];
    cout << hash(  _hfSF, sizeof(float)*hfSF.size() ) << endl;
    //for(unsigned int i=0; i<hfSF.size(); i++) cout<<(i==0?"":",")<<hfSF[i]; cout<<"]"<<endl;

    std::vector<int>    ecalScaleET = ptr1->layer1ECalScaleETBins();
    cout<<"  layer1ECalScaleETBins=  ["; for(unsigned int i=0; i<ecalScaleET.size(); i++) cout<<(i==0?"":",")<<ecalScaleET[i]; cout<<"]"<<endl;
    std::vector<int>    hcalScaleET = ptr1->layer1HCalScaleETBins();
    cout<<"  layer1HCalScaleETBins=  ["; for(unsigned int i=0; i<hcalScaleET.size(); i++) cout<<(i==0?"":",")<<hcalScaleET[i]; cout<<"]"<<endl;
    std::vector<int>    hfScaleET   = ptr1->layer1HFScaleETBins();
    cout<<"  layer1HFScaleETBins=    ["; for(unsigned int i=0; i<hfScaleET.size(); i++) cout<<(i==0?"":",")<<hfScaleET[i]; cout<<"]"<<endl;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TCaloParamsViewer);

