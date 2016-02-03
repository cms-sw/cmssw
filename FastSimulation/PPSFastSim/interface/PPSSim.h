#ifndef PPSSim_h
#define PPSSim_h
// ROOT #includes
#include "TROOT.h"
#include "Rtypes.h"
#include "TH2F.h"
#include "TDirectory.h"
#include "TRandom3.h"
#include "TMath.h"
#include "TLorentzVector.h"
#include "TString.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <math.h> 
// Framework includes
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "FWCore/Framework/interface/Event.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"
#include "HepMC/GenParticle.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "FastSimulation/PPSFastObjects/interface/PPSSpectrometer.h"
#include "FastSimulation/PPSFastSim/interface/PPSTrkDetector.h"
#include "FastSimulation/PPSFastSim/interface/PPSToFDetector.h"
// Hector #includes
#include "H_BeamLine.h"
#include "H_BeamParticle.h"
#include "H_RecRPObject.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
/*
   The virtuality coming from the reconstruction in Hector (H_RecRPObject) must be summed with
   twice the proton mass squared to recover the quadrimomentum lost of the scattered proton
   */
#include "FastSimulation/PPSFastSim/interface/PPSConstants.h"
//=====================================================================================================
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class PPSSim {
    public:
        PPSSim(bool =false);
        ~PPSSim() {};

        void BeginRun();
        void BeginEvent();
        void EndRun();
        void EndEvent();

        //void set_Strengths();
        void set_Verbose(bool f)   {fVerbose= f;};
        void set_KickersOFF(bool k){fKickersOFF=k;};
        void set_etaMin(double em) {fEtaMin = em;};
        void set_momentumMin(double pm){fMomentumMin=pm;};
        void set_phiMin(double pm) {fPhiMin = pm;};
        void set_phiMax(double pm) {fPhiMax = pm;};
        void set_xiMin(double xi)  {xi_min  = xi;};
        void set_xiMax(double xi)  {xi_max  = xi;};
        void set_tMin(double t)    {t_min   = t;};
        void set_tMax(double t)    {t_max   = t;};
        void set_CentralMass(double m,double me) {fCentralMass=m;fCentralMassErr=me;};
        void set_TrackerZPosition(double p)   {fTrackerZPosition=p;};
        void set_TrackerMisAlignment(double x1F, double x2F, double x1B, double x2B) {
            fDet1XOffsetF=x1F; fDet2XOffsetF=x2F; fDet1XOffsetB=x1B; fDet2XOffsetB=x2B;}

        void set_FilterHitMap(bool f)        {fFilterHitMap=f;};
        void set_ApplyFiducialCuts(bool f)   {fApplyFiducialCuts=f;};
        void set_WindowForTrack(double x, double y,double c)
        {fMaxXfromBeam=x;fMaxYfromBeam=y;fDetectorClosestX=c;};
        void set_TrackerLength(double p)     {fTrackerLength=p;};
        void set_TrackerSize(double w,double h) {fTrackerWidth=w;fTrackerHeight=h;};
        void set_ToFCellSize(double w,double h)     {fToFCellW=w;fToFCellH=h;};
        void set_ToFPitch(double x, double y)       {fToFPitchX=x;fToFPitchY=y;};
        void set_ToFNCells(int x,int y)             {fToFNCellX=x;fToFNCellY=y;};
        void set_ToFZPosition(double p)       {fToFZPosition=p;};
        void set_TCLPosition(const string& tcl,double z1,double z2) {
            if (tcl=="TCL4")      {fTCL4Position1=z1;fTCL4Position2=z2;}
            else if (tcl=="TCL5") {fTCL5Position1=z1;fTCL5Position2=z2;}
            else edm::LogWarning("debug")  <<"WARNING: Unknown Collimator " << tcl ;
        }
        void set_ToFResolution(double p)     {fTimeSigma=p;};
        void set_VertexSmearing(bool f=true) {fSmearVertex = f;};
        void set_HitSmearing(bool f=true)    {fSmearHit = f;};
        void set_TrackerResolution(double r) {fHitSigmaX=r;fHitSigmaY=r;};
        void set_BeamLineFile(std::string b1, std::string b2) {fBeamLine1File=b1;fBeamLine2File=b2;};
        void set_BeamDirection(int b1dir,int b2dir) {fBeam1Direction=b1dir;fBeam2Direction=b2dir;};
        void set_ShowBeamLine()                     {fShowBeamLine=true;};
        void set_GenBeamProfile()                   {fSimBeam=true;};
        void set_CollisionPoint(std::string ip)        {fCollisionPoint=ip;};
        void set_VertexPosition(const double x, const double y, const double z)
        {fVtxMeanX=x;fVtxMeanY=y;fVtxMeanZ=z;};

        int  add_Vertex(const double x, const double y, const double z)
        {fVertex[NVertex++]=TVector3(x,y,z);return NVertex;};
        void add_OutgoingParticle(int ivtx, const TLorentzVector* pF,const TLorentzVector* pB)
        {if (ivtx>=NVertex) {edm::LogWarning("debug")  <<"Error: Adding particle to a non existing vertex.";}
            protonsOut[ivtx]=std::pair<const TLorentzVector*,const TLorentzVector*>(pF,pB);
        };

        void set_CrossingAngleCorrection(bool f=false) {fCrossAngleCorr=f;};
        void set_CrossingAngle(double cr)         {fCrossingAngle=cr;};
        void set_BeamEnergy(double e)             {fBeamEnergy=e;fBeamMomentum = sqrt(fBeamEnergy*fBeamEnergy - ProtonMassSQ);};
        void set_BeamEnergySmearing(bool f=false) {fSmearEnergy=f;};
        void set_BeamEnergyRMS(double rms)        {fBeamEnergyRMS=rms;};
        void set_BeamAngleSmearing(bool f=false)  {fSmearAngle=f;};
        void set_BeamAngleRMS(double rms)         {fBeamAngleRMS=rms;};
        void set_BeamXSizes(double bsig_det1,double bsig_det2,double bsig_tof) {fBeamXRMS_Trk1=bsig_det1;
            fBeamXRMS_Trk2=bsig_det2;
            fBeamXRMS_ToF=bsig_tof;
        };
        void set_TrackerInsertion(double xpos) {fTrackerInsertion=xpos;};
        void set_ToFInsertion(double xpos)     {fToFInsertion=xpos;};
        void set_TrackImpactParameterCut(double rip){fTrackImpactParameterCut=rip;}
        void set_ThetaXRangeatDet1(double thx_min,double thx_max){fMinThetaXatDet1=thx_min;fMaxThetaXatDet1=thx_max;}
        void set_ThetaYRangeatDet1(double thy_min,double thy_max){fMinThetaYatDet1=thy_min;fMaxThetaYatDet1=thy_max;}

        void ReadGenEvent(const std::vector<TrackingParticle>*);

        void ReadGenEvent(const HepMC::GenEvent* ); // read data from HepMC::GenEvent
        void ReadGenEvent(const std::vector<reco::GenParticle>* ); // read data from reco::GenParticleCollection
        void set_GenData();   // to be used when the generation is done by external generator
        TH2F*  GenBeamProfile(const double& z);
        void Generation();
        void Simulation();
        void Reconstruction();
        bool SearchTrack(int ,int ,int Direction,double& xi,double& t,double& partP,double& pt,double& thx,double& thy,double& x0,double& y0);
        void TrackerReco(int Direction,H_RecRPObject* station,PPSBaseData* arm);
        void ToFReco();
        void Digitization();
        void TrackerDigi(const PPSBaseData*,PPSTrkStation*);
        void ToFDigi(const PPSBaseData*,PPSToFDetector*);
        void ReconstructArm(H_RecRPObject* pps_station, double x1,double y1,double x2,double y2, double& tx, double& ty,double& eloss);
        void Get_t_and_xi(const TLorentzVector* p,double& t, double& xi);

        void Propagate(H_BeamParticle* p1,int Direction) ; // returns true if particle has stopped before detector position
        void GenSingleParticle(double& , double& ,double&);
        void GenCentralMass(double& , double& ,double&,double&,double&,double&);

        void Run();
        void SmearVertexPosition(double& ,double& ,double&);
        void CrossingAngleCorrection(H_BeamParticle& p_out, const int Direction);
        void CrossingAngleCorrection(TLorentzVector& );
        void ApplyBeamSmearing(TLorentzVector&);
        void LorentzBoost(TLorentzVector& p_out, const string& frame);
        TLorentzVector shoot(const double& t, const double& xi,const double& phi,const int);
        void LoadParameters();
        void PrintParameters();
        void HitSmearing(double& x, double& y, double& z);
        void ToFSmearing(double& t) {if (fSmearHit) t = gRandom3->Gaus(t,fTimeSigma);};
        double Minimum_t(const double& xi);

        bool isPhysical(const double& xi)    { return (Minimum_t(xi) < t_max)&&xi<=1.&&xi>=0; }
        PPSSpectrometer<Gen> * get_GenDataHolder() {return fGen;};
        PPSSpectrometer<Sim> * get_SimDataHolder() {return fSim;};
        PPSSpectrometer<Reco> * get_RecoDataHolder(){return fReco;};
        double get_BeamMomentum()            {return fBeamMomentum;};
        double get_BeamEnergy()              {return fBeamEnergy;};
        TH2F* get_Beam1Profile()             {return beam1profile;};
        TH2F* get_Beam2Profile()             {return beam2profile;};
        void ProjectToToF(const double x1, const double y1, const double x2, const double y2, double& xt, double& yt) {    
            xt = (fToFZPosition - (fTrackerZPosition+fTrackerLength))*(x2-x1)/(fTrackerLength)+x2;
            yt = (fToFZPosition - (fTrackerZPosition+fTrackerLength))*(y2-y1)/(fTrackerLength)+y2;
        };

    private:
        bool         fExternalGenerator;
        bool         fVerbose;
        int          NEvent;
        std::string  fGenMode;
        TRandom3 *gRandom3;
        TH2F*            beam1profile;
        TH2F*            beam2profile;
        PPSSpectrometer<Gen>* fGen;
        PPSSpectrometer<Sim>* fSim;
        PPSSpectrometer<Reco>* fReco;
        // Hector objects
        H_BeamLine*    beamlineF;
        H_BeamLine*    beamlineB;
        H_RecRPObject* pps_stationF;
        H_RecRPObject* pps_stationB;

        // LHC and det parameters
        std::string         fBeamLine1File;
        std::string         fBeamLine2File;
        int                 fBeam1Direction;
        int                 fBeam2Direction;
        bool                fShowBeamLine;
        std::string         fCollisionPoint;
        float          fBeamLineLength;
        double         fBeamEnergy;
        double         fBeamMomentum;
        double         fBeamXRMS_Trk1; // beam X size at tracker station 1
        double         fBeamXRMS_Trk2; // beam X size at tracker station 2
        double         fBeamXRMS_ToF; // beam X size at tof station
        double         fCrossingAngle; // in micro radians
        bool           fCrossAngleCorr;
        bool           fKickersOFF;
        double         fDetectorClosestX;
        double         fMaxXfromBeam;
        double         fMaxYfromBeam;
        double         fTrackerZPosition;
        double         fTrackerLength;
        double         fTrackerWidth;
        double         fTrackerHeight;
        double         fToFWidth;
        double         fToFHeight;
        double         fToFCellW;
        double         fToFCellH;
        double         fToFPitchX;
        double         fToFPitchY;
        int            fToFNCellX;
        int            fToFNCellY;
        double         fToFZPosition;
        double         fTrackerInsertion; // position of tracker during data taking (in number of sigmas)
        double         fToFInsertion;     // position of tof during data taking (in number of sigmas)
        double         fTCL4Position1;
        double         fTCL4Position2;
        double         fTCL5Position1;
        double         fTCL5Position2;
        std::pair<double,double>   fBeam1PosAtTCL4;
        std::pair<double,double>   fBeam1RMSAtTCL4;
        std::pair<double,double>   fBeam2PosAtTCL4;
        std::pair<double,double>   fBeam2RMSAtTCL4;
        std::pair<double,double>   fBeam1PosAtTCL5;
        std::pair<double,double>   fBeam1RMSAtTCL5;
        std::pair<double,double>   fBeam2PosAtTCL5;
        std::pair<double,double>   fBeam2RMSAtTCL5;

        PPSTrkStation* TrkStation_F; // auxiliary object with the tracker geometry
        PPSTrkStation* TrkStation_B; // auxiliary object with the tracker geometry
        PPSToFDetector* ToFDet_F;  // idem for the ToF detector
        PPSToFDetector* ToFDet_B;  // idem for the ToF detector
        // Parameters for vertex smearing
        bool   fSmearVertex;
        double fVtxMeanX;
        double fVtxMeanY;
        double fVtxMeanZ;
        double fVtxSigmaX;
        double fVtxSigmaY;
        double fVtxSigmaZ;

        // Parameters for hit smearing
        double fSmearHit;
        double fHitSigmaX;
        double fHitSigmaY;
        double fHitSigmaZ;
        double fTimeSigma;

        // Parameters for the detector missalignment
        double fDet1XOffsetF;
        double fDet2XOffsetF;
        double fDet1XOffsetB;
        double fDet2XOffsetB;

        // Parameter for time smearing

        // Parameter for angular smearing
        bool   fSmearAngle;
        double fBeamAngleRMS; // in micro radians

        // Parameter for energy smearing
        bool   fSmearEnergy;
        double fBeamEnergyRMS;

        // Parameter for the Reconstruction
        bool   fFilterHitMap;
        bool   fApplyFiducialCuts;
        double fTrackImpactParameterCut; // maximum impact parameter at IP in cm
        double fMinThetaXatDet1;         // minimum thetaX at first tracker detector in urad
        double fMaxThetaXatDet1;         // maximum thetaX at first tracker detector in urad
        double fMinThetaYatDet1;         // minimum thetaY at first tracker detector in urad
        double fMaxThetaYatDet1;         // maximum thetaY at first tracker detector in urad

        // Parameters for the simulation
        double xi_min;
        double xi_max;
        double t_min;
        double t_max;
        double fPhiMin;
        double fPhiMax;
        double fEtaMin;
        double fMomentumMin;
        double fCentralMass;
        double fCentralMassErr;
        std::vector<double> CheckPoints;
        // Generated proton
        int NVertex;
        map<int,TVector3> fVertex;
        map<int,pair<const TLorentzVector*,const TLorentzVector*> > protonsOut;
        map<int,pair<bool,bool> >                       fHasStopped;
        bool fHasStoppedF;
        bool fHasStoppedB;
        const TLorentzVector* protonF;
        const TLorentzVector* protonB;
        //
        TLorentzVector Beam1;
        TLorentzVector Beam2;
        // Simulated hits
        bool fSimBeam;
        // private methods
};
#endif
