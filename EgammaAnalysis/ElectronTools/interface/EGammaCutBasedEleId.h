
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

bool PassWP(const WorkingPoint workingPoint,
    const reco::GsfElectronRef &ele,
    const edm::Handle<reco::ConversionCollection> &conversions,
    const reco::BeamSpot &beamspot,
    const edm::Handle<reco::VertexCollection> &vtxs,
    const double &iso_ch,
    const double &iso_em,
    const double &iso_nh,
    const double &rho);


bool PassWP(const WorkingPoint workingPoint,
    const reco::GsfElectron &ele,
    const edm::Handle<reco::ConversionCollection> &conversions,
    const reco::BeamSpot &beamspot,
    const edm::Handle<reco::VertexCollection> &vtxs,
    const double &iso_ch,
    const double &iso_em,
    const double &iso_nh,
    const double &rho);


bool PassTriggerCuts(const TriggerWorkingPoint triggerWorkingPoint, const reco::GsfElectronRef &ele);

bool PassTriggerCuts(const TriggerWorkingPoint triggerWorkingPoint, const reco::GsfElectron &ele);

bool PassEoverPCuts(const reco::GsfElectronRef &ele);

bool PassEoverPCuts(const reco::GsfElectron &ele);

unsigned int TestWP(const WorkingPoint workingPoint,
    const reco::GsfElectronRef &ele,
    const edm::Handle<reco::ConversionCollection> &conversions,
    const reco::BeamSpot &beamspot,
    const edm::Handle<reco::VertexCollection> &vtxs,
    const double &iso_ch,
    const double &iso_em,
    const double &iso_nh,
    const double &rho);

unsigned int TestWP(const WorkingPoint workingPoint,
    const reco::GsfElectron &ele,
    const edm::Handle<reco::ConversionCollection> &conversions,
    const reco::BeamSpot &beamspot,
    const edm::Handle<reco::VertexCollection> &vtxs,
    const double &iso_ch,
    const double &iso_em,
    const double &iso_nh,
    const double &rho);

#endif

//
// implementation of working points
// used by CMSSW interface, does not 
// itself depend on CMSSW code
//

bool PassWP(WorkingPoint workingPoint, const bool isEB, const float pt, const float eta,
    const float dEtaIn, const float dPhiIn, const float sigmaIEtaIEta, const float hoe,
    const float ooemoop, const float d0vtx, const float dzvtx, const float iso_ch, const float iso_em, const float iso_nh, 
    const bool vtxFitConversion, const unsigned int mHits, const double rho);

bool PassTriggerCuts(const TriggerWorkingPoint triggerWorkingPoint, const bool isEB, const float pt, 
    const float dEtaIn, const float dPhiIn, const float sigmaIEtaIEta, const float hoe,
    const float trackIso, const float ecalIso, const float hcalIso);

bool PassEoverPCuts(const float eta, const float eopin, const float fbrem);

unsigned int TestWP(WorkingPoint workingPoint, const bool isEB, const float pt, const float eta,
    const float dEtaIn, const float dPhiIn, const float sigmaIEtaIEta, const float hoe,
    const float ooemoop, const float d0vtx, const float dzvtx, const float iso_ch, const float iso_em, const float iso_nh, 
    const bool vtxFitConversion, const unsigned int mHits, const double rho);

// print the bit mask
void PrintDebug(unsigned int mask);

}

