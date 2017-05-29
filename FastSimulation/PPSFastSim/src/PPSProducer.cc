// -*- C++ -*-
//
// Package:    PPSProducer
// Class:      PPSProducer
// 
/**\class PPSProducer PPSProducer.cc FastSimulation/PPSFastSim/src/PPSProducer.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author:  Luiz Martins Mundim Filho,22 1-020,+41227677686,
//         Created:  Thu Apr 25 17:37:14 CEST 2013
// $Id$
// 
//


// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"
#include "HepMC/GenParticle.h"
//
#include "FastSimulation/PPSFastObjects/interface/PPSSpectrometer.h"
#include "FastSimulation/PPSFastSim/interface/PPSSim.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
// class declaration
//

class PPSProducer : public edm::EDProducer {
    public:
        explicit PPSProducer(const edm::ParameterSet&);
        ~PPSProducer();

        static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    private:
        virtual void beginJob() ;
        virtual void produce(edm::Event&, const edm::EventSetup&);
        virtual void endJob() ;

        virtual void beginRun(edm::Run&, edm::EventSetup const&);
        virtual void endRun(edm::Run&, edm::EventSetup const&);
        virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
        virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

        // ----------member data ---------------------------
        bool    fVerbose;
        bool   fUseHepMCProducer;
        PPSSim* pps;
        //Including GetToken 
        edm::InputTag gensrc;
        edm::EDGetTokenT<edm::HepMCProduct> gensrcTokenHepMC;

        edm::InputTag gensrcPart;
        edm::EDGetTokenT<std::vector<reco::GenParticle>> gensrcPartToken;		
};
//
PPSProducer::PPSProducer(const edm::ParameterSet& iConfig):fVerbose(false)
{
    //register your products
    produces<PPSSpectrometer<Sim> >("PPSSim");
    produces<PPSSpectrometer<Gen> >("PPSGen");
    produces<PPSSpectrometer<Reco> >("PPSReco");
    //now do what ever other initialization is needed
    pps = NULL;
    edm::FileInPath beam1filename_  = iConfig.getParameter<edm::FileInPath> ("Beam1File");
    edm::FileInPath beam2filename_  = iConfig.getParameter<edm::FileInPath> ("Beam2File");
    int    beam1dir            = iConfig.getParameter<int>("Beam1Direction");
    int    beam2dir            = iConfig.getParameter<int>("Beam2Direction");
    bool   showbeam            = iConfig.getUntrackedParameter<bool>("ShowBeamLine",false);
    bool   simbeam             = iConfig.getUntrackedParameter<bool>("SimBeamProfile",false);
    double fVtxMeanX           = iConfig.getUntrackedParameter<double>("VtxMeanX",0.);
    double fVtxMeanY           = iConfig.getUntrackedParameter<double>("VtxMeanY",0.);
    double fVtxMeanZ           = iConfig.getUntrackedParameter<double>("VtxMeanZ",0.);
    string ip                  = iConfig.getParameter<string>("CollisionPoint");
    fVerbose            = iConfig.getUntrackedParameter<int>("Verbosity",0);
    double fTrackerLength      = iConfig.getParameter<double>("TrackerLength");
    double fTrackerZPosition   = iConfig.getParameter<double>("TrackerZPosition");
    double fTrackerWidth       = iConfig.getParameter<double>("TrackerWidth"); // tracker width in mm
    double fTrackerHeight      = iConfig.getParameter<double>("TrackerHeight"); // tracker height in mm
    double fTrackerInsertion   = iConfig.getParameter<double>("TrackerInsertion"); // Number of sigms (X) from the beam for the tracker
    double fToFInsertion       = iConfig.getParameter<double>("ToFInsertion");     // Number of sigms (X) from the beam for the tof
    double fToFCellW           = iConfig.getParameter<double>("ToFCellWidth");      // tof  width in mm
    double fToFCellH           = iConfig.getParameter<double>("ToFCellHeight");     // tof height in mm
    double fToFPitchX          = iConfig.getParameter<double>("ToFPitchX");         // cell pitch in X (in microns)
    double fToFPitchY          = iConfig.getParameter<double>("ToFPitchY");         // cell pitch in Y (in microns)
    int    fToFNCellX          = iConfig.getParameter<int>("ToFNCellX");            // number of cells in X
    int    fToFNCellY          = iConfig.getParameter<int>("ToFNCellY");            // number of cells in Y
    double fTrk1XOffset        = iConfig.getParameter<double>("TrkDet1XOffset");
    double fTrk2XOffset        = iConfig.getParameter<double>("TrkDet2XOffset");
    double fToFZPosition        = iConfig.getParameter<double>("ToFZPosition");
    double fTCL4Position       = iConfig.getUntrackedParameter<double>("TCL4Position",0.);
    double fTCL5Position       = iConfig.getUntrackedParameter<double>("TCL5Position",0.);
    bool   fSmearHit           = iConfig.getParameter<bool>("SmearHit");
    double fHitSigmaX          = iConfig.getParameter<double>("HitSigmaX");
    double fHitSigmaY          = iConfig.getParameter<double>("HitSigmaY");
    double fTimeSigma          = iConfig.getParameter<double>("TimeSigma");
    bool   fSmearAngle         = iConfig.getParameter<bool>("SmearAngle");
    double fBeamAngleRMS       = iConfig.getParameter<double>("BeamAngleRMS"); // beam angular dispersion in urad
    bool   fSmearEnergy        = iConfig.getParameter<bool>("SmearEnergy");
    double fBeamEnergy         = iConfig.getParameter<double>("BeamEnergy");    // beam energy in GeV
    double fBeamEnergyRMS      = iConfig.getParameter<double>("BeamEnergyRMS");    // beam energy dispersion in GeV
    double fBeamSizeAtTrk1    = iConfig.getParameter<double>("BeamSizeAtTrk1"); // beam sigma(X) at first tracker station in mm
    double fBeamSizeAtTrk2    = iConfig.getParameter<double>("BeamSizeAtTrk2"); // beam sigma(X) at second tracker station in mm
    double fBeamSizeAtToF     = iConfig.getParameter<double>("BeamSizeAtToF"); // beam sigma(X) at timing station in mm
    double fPhiMin             = iConfig.getParameter<double>("PhiMin");
    double fPhiMax             = iConfig.getParameter<double>("PhiMax");
    double fCentralMass        = iConfig.getParameter<double>("CentralMass");
    double fCentralMassErr     = iConfig.getParameter<double>("CentralMassErr");
    bool   fKickersOFF         = iConfig.getUntrackedParameter<bool>("KickersOFF",true);
    bool   fCrossAngleCorr     = iConfig.getParameter<bool>("CrossAngleCorr");
    double fCrossingAngle      = iConfig.getParameter<double>("CrossingAngle");
    double fEtaMin             = iConfig.getParameter<double>("EtaMin");
    double fMomentumMin        = iConfig.getParameter<double>("MomentumMin");
    double fImpParcut          = iConfig.getParameter<double>("TrackImpactParameterCut"); // exclude hit combination that lead to high imp. par. reco tracks (in cm)
    double fMinthx             = iConfig.getParameter<double>("MinThetaXatDet1"); // minimum thetaX at first tracker detector (in urad)
    double fMaxthx             = iConfig.getParameter<double>("MaxThetaXatDet1"); // maximum thetaX at first tracker detector (in urad)
    double fMinthy             = iConfig.getParameter<double>("MinThetaYatDet1"); // minimum thetaY at first tracker detector (in urad)
    double fMaxthy             = iConfig.getParameter<double>("MaxThetaYatDet1"); // maximum thetaY at first tracker detector (in urad)
    double fMaxXfromBeam       = iConfig.getParameter<double>("MaxXfromBeam");    // maximum distance (X) from beam a hit is accepted (in mm, negative)
    double fMaxYfromBeam       = iConfig.getParameter<double>("MaxYfromBeam");    // maximum distance (Y) from beam a hit is accepted (in mm, positive, simetric)
    double fDetectorClosestX   = iConfig.getParameter<double>("DetectorClosestX");// minimum distance (X) from beam a hit is accepted (in mm, negative)
    bool   fFilterHitMap       = iConfig.getParameter<bool>("FilterHitMap");       // apply geometrical cuts in the hit position (RP window+distance from beam)
    bool   fApplyFiducialCuts  = iConfig.getParameter<bool>("ApplyFiducialCuts");  // apply geometrical cuts in the hit position (Detector size)

    gensrc              = iConfig.getParameter<edm::InputTag>("genSource");
    gensrcTokenHepMC = consumes<edm::HepMCProduct>(gensrc);

    gensrcPart           = iConfig.getParameter<edm::InputTag>("genSource");
    gensrcPartToken = consumes<std::vector<reco::GenParticle>>(gensrcPart);

    // New Flag to use the HepMCProducer (True)  or GenParticle + PU Particles (false)
    fUseHepMCProducer         = iConfig.getUntrackedParameter<bool>("UseHepMCProducer",false);

    // To get Path for Optics files
    std::string beam1filename = beam1filename_.fullPath();
    std::string beam2filename = beam2filename_.fullPath();
    if (beam1filename.find(".") == 0 && beam2filename.find(".") ==0  ){
        beam1filename.erase(0, 2);
        beam2filename.erase(0, 2);		}

    pps = new PPSSim(true); // instanciate PPSSim with External Generator
    pps->set_KickersOFF(fKickersOFF);
    pps->set_BeamLineFile(beam1filename,beam2filename);
    pps->set_BeamDirection(beam1dir,beam2dir);
    pps->set_BeamEnergySmearing(fSmearEnergy);
    pps->set_BeamEnergy(fBeamEnergy);
    pps->set_BeamEnergyRMS(fBeamEnergyRMS);
    pps->set_BeamAngleSmearing(fSmearAngle);
    pps->set_BeamAngleRMS(fBeamAngleRMS);
    pps->set_BeamXSizes(fBeamSizeAtTrk1,fBeamSizeAtTrk2,fBeamSizeAtToF);
    pps->set_TCLPosition("TCL4",fTCL4Position,fTCL4Position);
    pps->set_TCLPosition("TCL5",fTCL5Position,fTCL5Position);
    if (showbeam) pps->set_ShowBeamLine();
    if (simbeam)  pps->set_GenBeamProfile();
    pps->set_VertexPosition(fVtxMeanX,fVtxMeanY,fVtxMeanZ);
    pps->set_CollisionPoint(ip);
    pps->set_TrackerZPosition(fTrackerZPosition);
    pps->set_TrackerInsertion(fTrackerInsertion);
    pps->set_ToFInsertion(fToFInsertion);
    pps->set_TrackerLength(fTrackerLength);
    pps->set_TrackerSize(fTrackerWidth,fTrackerHeight);
    pps->set_ToFCellSize(fToFCellW,fToFCellH);
    pps->set_ToFNCells(fToFNCellX,fToFNCellY);
    pps->set_ToFPitch(fToFPitchX*um_to_mm,fToFPitchY*um_to_mm);
    pps->set_ToFZPosition(fToFZPosition);
    pps->set_ToFResolution(fTimeSigma);
    pps->set_TrackerMisAlignment(fTrk1XOffset,fTrk2XOffset,fTrk1XOffset,fTrk2XOffset); // use the same offset for the forward and backward arm
    pps->set_HitSmearing(fSmearHit);
    pps->set_VertexSmearing(false); // when using cmssw, vertex smearing is done somewhere else
    pps->set_phiMin(fPhiMin);
    pps->set_phiMax(fPhiMax);
    pps->set_etaMin(fEtaMin);
    pps->set_momentumMin(fMomentumMin);
    pps->set_CentralMass(fCentralMass,fCentralMassErr);
    pps->set_HitSmearing(fSmearHit);
    pps->set_TrackerResolution((fHitSigmaX+fHitSigmaY)/2.*um_to_mm);
    pps->set_Verbose(fVerbose);
    pps->set_CrossingAngleCorrection(fCrossAngleCorr);
    pps->set_CrossingAngle(fCrossingAngle);
    pps->set_TrackImpactParameterCut(fImpParcut);
    pps->set_ThetaXRangeatDet1(fMinthx,fMaxthx);
    pps->set_ThetaYRangeatDet1(fMinthy,fMaxthy);
    pps->set_WindowForTrack(fMaxXfromBeam,fMaxYfromBeam,fDetectorClosestX);
    pps->set_FilterHitMap(fFilterHitMap);
    pps->set_ApplyFiducialCuts(fApplyFiducialCuts);
}
//
// member functions
//

// ------------ method called to produce the data  ------------
void PPSProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

    using namespace edm;
    using namespace reco;
    using namespace HepMC;
    using namespace CLHEP;

    pps->BeginEvent();
    if(fUseHepMCProducer){
        try{
            Handle<HepMCProduct> genHandle;
            iEvent.getByToken(gensrcTokenHepMC,genHandle);
            const HepMC::GenEvent* evt = genHandle->GetEvent();
            pps->ReadGenEvent(evt);
        }
        catch(const Exception&){
            edm::LogWarning("debug") <<"PPSProducer::produce: No MC information";
            return;
        }
    } else {
        try{
            // Try first genParticles with PileUp events
            Handle<std::vector<reco::GenParticle> > genParticles;
            iEvent.getByToken(gensrcPartToken, genParticles);
            pps->ReadGenEvent(&(*genParticles));
        }
        catch(const Exception&){
            edm::LogWarning("debug") <<"PPSProducer::produce: No GenParticle information. Trying HepMC::Event.";
            return;
        }
    }
    pps->Run();
    pps->EndEvent();
    std::auto_ptr<PPSSpectrometer<Gen> > fGen(new PPSSpectrometer<Gen>(*(pps->get_GenDataHolder())));
    std::auto_ptr<PPSSpectrometer<Sim> > fSim(new PPSSpectrometer<Sim>(*(pps->get_SimDataHolder())));
    std::auto_ptr<PPSSpectrometer<Reco> > fReco(new PPSSpectrometer<Reco>(*(pps->get_RecoDataHolder())));
    iEvent.put(fGen,"PPSGen");
    iEvent.put(fSim,"PPSSim");
    iEvent.put(fReco,"PPSReco");
}

PPSProducer::~PPSProducer()
{

    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)

}
// ------------ method called once each job just before starting event loop  ------------
void PPSProducer::beginJob()
{
    if (pps) pps->BeginRun();
}

// ------------ method called once each job just after ending the event loop  ------------
void PPSProducer::endJob() {
    if(pps) pps->EndRun();
}

// ------------ method called when starting to processes a run  ------------
void PPSProducer::beginRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void PPSProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void PPSProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void PPSProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void PPSProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    //The following says we do not know what parameters are allowed so do no validation
    // Please change this to state exactly what you do use, even if it is no parameters
    edm::ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PPSProducer);
