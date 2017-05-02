#ifndef FastSimulation_CTPPSFastTrackingProducer_h
#define FastSimulation_CTPPSFastTrackingProducer_h

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FastSimDataFormats/CTPPSFastSim/interface/CTPPSFastRecHit.h"
#include "FastSimDataFormats/CTPPSFastSim/interface/CTPPSFastRecHitContainer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FastSimDataFormats/CTPPSFastSim/interface/CTPPSFastTrack.h"
#include "FastSimDataFormats/CTPPSFastSim/interface/CTPPSFastTrackContainer.h"

#include "FastSimulation/CTPPSFastGeometry/interface/CTPPSTrkDetector.h"
#include "FastSimulation/CTPPSFastGeometry/interface/CTPPSToFDetector.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "SimTransport/HectorProducer/interface/CTPPSHectorParameters.h"
//#include "FastSimulation/CTPPSSimHitProducer/plugins/FastCTPPSParameters.h"

//CLHEP
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include <CLHEP/Vector/LorentzVector.h>

//C++ library
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <math.h>
#include <map>
#include <vector>
#include <utility>
#include <math.h>

#include <TMatrixD.h>

// hector includes
#include "H_Parameters.h"
#include "H_BeamLine.h"
#include "H_RecRPObject.h"
#include "H_BeamParticle.h"

#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"


//
// class declaration
//

class CTPPSFastTrackingProducer : public edm::stream::EDProducer<> {
    public:
        explicit CTPPSFastTrackingProducer(const edm::ParameterSet&);
        ~CTPPSFastTrackingProducer();
        typedef CLHEP::HepLorentzVector LorentzVector;

    private:
        virtual void beginStream(edm::StreamID) override;
        virtual void produce(edm::Event&, const edm::EventSetup&) override;
        virtual void endStream() override;
        //this function will only be called once per event
        virtual void beginEvent(edm::Event& event, const edm::EventSetup& eventSetup);
        virtual void endEvent(edm::Event& event, const edm::EventSetup& eventSetup);

        // ----------member data ---------------------------

        typedef std::vector<CTPPSFastRecHit> CTPPSFastRecHitContainer;
        edm::EDGetTokenT< CTPPSFastRecHitContainer > _recHitToken;
        void ReadRecHits(edm::Handle<CTPPSFastRecHitContainer> &);
        void FastReco(int Direction,H_RecRPObject* station);
        void Reconstruction();	
        void ReconstructArm(H_RecRPObject* pps_station, double x1,double y1,double x2,double y2, double& tx, double& ty,double& eloss);
        void MatchCellId(int cellId, vector<int> vrecCellId, vector<double> vrecTof, bool& match, double& recTof);
        bool SearchTrack(int ,int ,int Direction,double& xi,double& t,double& partP,double& pt,double& thx,double& thy,double& x0,double& y0, double& xt, double& yt, double& X1d, double& Y1d, double& X2d, double& Y2d);
        void LorentzBoost(LorentzVector& p_out, const string& frame);
        void Get_t_and_xi(const LorentzVector* p,double& t, double& xi);
        void TrackerStationClear();
        void TrackerStationStarting();
        void ProjectToToF(const double x1, const double y1, const double x2, const double y2, double& xt, double& yt) {    
            xt = ((fz_timing-fz_tracker2)*(x2-x1)/(fz_tracker2-fz_tracker1)) + x2;
            yt = ((fz_timing-fz_tracker2)*(y2-y1)/(fz_tracker2-fz_tracker1)) + y2;
        };
        // Hector objects

        std::map<unsigned int, H_BeamParticle*> m_beamPart;
	std::unique_ptr<H_BeamLine> m_beamlineCTPPS1;
        std::unique_ptr<H_BeamLine> m_beamlineCTPPS2;

        std::unique_ptr<H_RecRPObject> pps_stationF;
        std::unique_ptr<H_RecRPObject> pps_stationB;

        string beam1filename;
        string beam2filename;

        // Defaults
        double lengthctpps ;
        bool   m_verbosity;
        double fBeamEnergy;
        double fBeamMomentum;
        bool   fCrossAngleCorr;
        double fCrossingAngle;
        ////////////////////////////////////////////////
        std::unique_ptr<CTPPSTrkStation> TrkStation_F; // auxiliary object with the tracker geometry
        std::unique_ptr<CTPPSTrkStation> TrkStation_B;
        std::unique_ptr<CTPPSTrkDetector> det1F; 
        std::unique_ptr<CTPPSTrkDetector> det1B;
        std::unique_ptr<CTPPSTrkDetector> det2F; 
        std::unique_ptr<CTPPSTrkDetector> det2B;
        std::unique_ptr<CTPPSToFDetector> detToF_F;
        std::unique_ptr<CTPPSToFDetector> detToF_B;

        std::vector<CTPPSFastTrack> theCTPPSFastTrack;

        CTPPSFastTrack track;

        std::vector<int> recCellId_F, recCellId_B ; 
        std::vector<double> recTof_F, recTof_B ; 
	
	double fz_tracker1, fz_tracker2, fz_timing;
	double fTrackerWidth,fTrackerHeight,fTrackerInsertion,fBeamXRMS_Trk1,fBeamXRMS_Trk2,fTrk1XOffset,fTrk2XOffset;
        std::vector<double> fToFCellWidth;
        double fToFCellHeight,fToFPitchX,fToFPitchY;
        int fToFNCellX,fToFNCellY;
        double fToFInsertion,fBeamXRMS_ToF,fToFXOffset,fTimeSigma,fImpParcut;		


};
#endif

