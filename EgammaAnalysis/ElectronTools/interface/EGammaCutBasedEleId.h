
//
// to use this code outside of CMSSW
// set this definition
//

//#define STANDALONEID
#ifndef STANDALONEID
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#endif
#include "EgammaAnalysis/ElectronTools/interface/ElectronEffectiveArea.h"

#include <vector>

namespace EgammaCutBasedEleId {

//
// typedefs
//

typedef std::vector< edm::Handle< edm::ValueMap<reco::IsoDeposit> > >   IsoDepositMaps;
typedef std::vector< edm::Handle< edm::ValueMap<double> > >             IsoDepositVals;

//
// defined ID working points
//

enum WorkingPoint {
    VETO,
    LOOSE,
    MEDIUM,
    TIGHT
};

enum TriggerWorkingPoint {
    TRIGGERTIGHT,
    TRIGGERWP70
};

//
// cuts used within working points
//

enum CutType {
    DETAIN          = (1<<0),
    DPHIIN          = (1<<1),
    SIGMAIETAIETA   = (1<<2),
    HOE             = (1<<3),
    OOEMOOP         = (1<<4),
    D0VTX           = (1<<5),
    DZVTX           = (1<<6),
    ISO             = (1<<7),
    VTXFIT          = (1<<8),
    MHITS           = (1<<9)
};

//
// all possible cuts pass
//

static const unsigned int PassAll         = DETAIN | DPHIIN | SIGMAIETAIETA | HOE | OOEMOOP | D0VTX | DZVTX | ISO | VTXFIT | MHITS;

//
// CMSSW interface
//

#ifndef STANDALONEID

bool PassWP(WorkingPoint workingPoint,
    const reco::GsfElectronRef &ele,
    const edm::Handle<reco::ConversionCollection> &conversions,
    const reco::BeamSpot &beamspot,
    const edm::Handle<reco::VertexCollection> &vtxs,
    const double &iso_ch,
    const double &iso_em,
    const double &iso_nh,
    const double &rho,
    ElectronEffectiveArea::ElectronEffectiveAreaTarget EAtarget);


bool PassWP(WorkingPoint workingPoint,
    const reco::GsfElectron &ele,
    const edm::Handle<reco::ConversionCollection> &conversions,
    const reco::BeamSpot &beamspot,
    const edm::Handle<reco::VertexCollection> &vtxs,
    const double &iso_ch,
    const double &iso_em,
    const double &iso_nh,
    const double &rho,
    ElectronEffectiveArea::ElectronEffectiveAreaTarget EAtarget);


bool PassTriggerCuts(TriggerWorkingPoint triggerWorkingPoint, const reco::GsfElectronRef &ele);

bool PassTriggerCuts(TriggerWorkingPoint triggerWorkingPoint, const reco::GsfElectron &ele);

bool PassEoverPCuts(const reco::GsfElectronRef &ele);

bool PassEoverPCuts(const reco::GsfElectron &ele);

unsigned int TestWP(WorkingPoint workingPoint,
    const reco::GsfElectronRef &ele,
    const edm::Handle<reco::ConversionCollection> &conversions,
    const reco::BeamSpot &beamspot,
    const edm::Handle<reco::VertexCollection> &vtxs,
    const double &iso_ch,
    const double &iso_em,
    const double &iso_nh,
    const double &rho,
    ElectronEffectiveArea::ElectronEffectiveAreaTarget EAtarget);

unsigned int TestWP(WorkingPoint workingPoint,
    const reco::GsfElectron &ele,
    const edm::Handle<reco::ConversionCollection> &conversions,
    const reco::BeamSpot &beamspot,
    const edm::Handle<reco::VertexCollection> &vtxs,
    const double &iso_ch,
    const double &iso_em,
    const double &iso_nh,
    const double &rho,
    ElectronEffectiveArea::ElectronEffectiveAreaTarget EAtarget);

#endif

//
// implementation of working points
// used by CMSSW interface, does not 
// itself depend on CMSSW code
//

bool PassWP(WorkingPoint workingPoint, bool isEB, float pt, float eta,
    float dEtaIn, float dPhiIn, float sigmaIEtaIEta, float hoe,
    float ooemoop, float d0vtx, float dzvtx, float iso_ch, float iso_em, float iso_nh, 
    bool vtxFitConversion, unsigned int mHits, double rho, ElectronEffectiveArea::ElectronEffectiveAreaTarget EAtarget);

bool PassTriggerCuts(TriggerWorkingPoint triggerWorkingPoint, bool isEB, float pt, 
    float dEtaIn, float dPhiIn, float sigmaIEtaIEta, float hoe,
    float trackIso, float ecalIso, float hcalIso);

bool PassEoverPCuts(float eta, float eopin, float fbrem);

unsigned int TestWP(WorkingPoint workingPoint, bool isEB, float pt, float eta,
    float dEtaIn, float dPhiIn, float sigmaIEtaIEta, float hoe,
    float ooemoop, float d0vtx, float dzvtx, float iso_ch, float iso_em, float iso_nh, 
    bool vtxFitConversion, unsigned int mHits, double rho, ElectronEffectiveArea::ElectronEffectiveAreaTarget EAtarget);

// print the bit mask
void PrintDebug(unsigned int mask);

}

