// -*- C++ -*-
//
// Package:    FastSimulation/CTPPSFastTrackingProducer
// Class:      CTPPSFastTrackingProducer
// 
/**\class CTPPSFastTrackingProducer CTPPSFastTrackingProducer.cc FastSimulation/CTPPSFastTrackingProducer/plugins/CTPPSFastTrackingProducer.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author:  Sandro Fonseca De Souza
//         Created:  Thu, 29 Sep 2016 16:13:41 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FastSimulation/CTPPSFastTrackingProducer/interface/CTPPSFastTrackingProducer.h"
//using namespace edm;

//////////////////////
// constructors and destructor
//
CTPPSFastTrackingProducer::CTPPSFastTrackingProducer(const edm::ParameterSet& iConfig):
    m_verbosity(false), fBeamMomentum(0.), fCrossAngleCorr(false), fCrossingAngle(0.)
{
    //register your products
    produces<edm::CTPPSFastTrackContainer>("CTPPSFastTrack");
    using namespace edm;
    _recHitToken = consumes<CTPPSFastRecHitContainer>(iConfig.getParameter<edm::InputTag>("recHitTag"));
    m_verbosity = iConfig.getParameter<bool>("Verbosity");
    // User definitons

    // Read beam parameters needed for Hector reconstruction
    lengthctpps     = iConfig.getParameter<double>("BeamLineLengthCTPPS" );
    beam1filename   = iConfig.getParameter<string>("Beam1");
    beam2filename = iConfig.getParameter<string>("Beam2"); 
    fBeamEnergy = iConfig.getParameter<double>("BeamEnergy"); // beam energy in GeV
    fBeamMomentum = sqrt(fBeamEnergy*fBeamEnergy - ProtonMassSQ); 
    fCrossAngleCorr = iConfig.getParameter<bool>("CrossAngleCorr");
    fCrossingAngle = iConfig.getParameter<double>("CrossingAngle");
  
    //Read detectors positions and parameters

    fz_tracker1   = iConfig.getParameter<double>("Z_Tracker1");
    fz_tracker2   = iConfig.getParameter<double>("Z_Tracker2");
    fz_timing =  iConfig.getParameter<double>("Z_Timing");	 
  //
    fTrackerWidth       = iConfig.getParameter<double>("TrackerWidth");
    fTrackerHeight      = iConfig.getParameter<double>("TrackerHeight");
    fTrackerInsertion   = iConfig.getParameter<double>("TrackerInsertion");
    fBeamXRMS_Trk1 = iConfig.getParameter<double>("BeamXRMS_Trk1");
    fBeamXRMS_Trk2 = iConfig.getParameter<double>("BeamXRMS_Trk2");
    fTrk1XOffset = iConfig.getParameter<double>("Trk1XOffset");
    fTrk2XOffset = iConfig.getParameter<double>("Trk2XOffset");
    fToFCellWidth = iConfig.getUntrackedParameter<std::vector<double> >("ToFCellWidth");
    fToFCellHeight = iConfig.getParameter<double>("ToFCellHeight");
    fToFPitchX = iConfig.getParameter<double>("ToFPitchX");
    fToFPitchY = iConfig.getParameter<double>("ToFPitchY");
    fToFNCellX          = iConfig.getParameter<int>("ToFNCellX");
    fToFNCellY          = iConfig.getParameter<int>("ToFNCellY");
    fToFInsertion       = iConfig.getParameter<double>("ToFInsertion");
    fBeamXRMS_ToF =  iConfig.getParameter<double>("BeamXRMS_ToF");
    fToFXOffset =  iConfig.getParameter<double>("ToFXOffset");
    fTimeSigma =  iConfig.getParameter<double>("TimeSigma");
    fImpParcut =  iConfig.getParameter<double>("ImpParcut");	



    // reading beamlines
    FileInPath b1(beam1filename.c_str());
    FileInPath b2(beam2filename.c_str());
    //
    if(lengthctpps>0. ) {
        m_beamlineCTPPS1 = std::unique_ptr<H_BeamLine>(new H_BeamLine( -1, lengthctpps + 0.1 )); // (direction, length)
        m_beamlineCTPPS2 = std::unique_ptr<H_BeamLine>(new H_BeamLine( 1, lengthctpps + 0.1 )); //
        m_beamlineCTPPS1->fill( b2.fullPath(), 1, "IP5" );
        m_beamlineCTPPS2->fill( b1.fullPath(), 1, "IP5" );
        m_beamlineCTPPS1->offsetElements( 120, 0.097 );
        m_beamlineCTPPS2->offsetElements( 120, 0.097 );
        m_beamlineCTPPS1->calcMatrix();
        m_beamlineCTPPS2->calcMatrix();
    } else {
        if ( m_verbosity ) LogDebug("CTPPSFastTrackingProducer") << "CTPPSFastTrackingProducer: WARNING: lengthctpps=  " << lengthctpps;
    } 

    // Create a particle to get the beam energy from the beam file
    // Take care: the z inside the station is in meters
    pps_stationF = std::unique_ptr<H_RecRPObject>(new H_RecRPObject(fz_tracker1,fz_tracker2,*m_beamlineCTPPS1));
    pps_stationB = std::unique_ptr<H_RecRPObject>(new H_RecRPObject(fz_tracker1,fz_tracker2,*m_beamlineCTPPS2));
    //
    //Tracker Detector Description
    det1F =std::unique_ptr<CTPPSTrkDetector>(new CTPPSTrkDetector(fTrackerWidth,fTrackerHeight,fTrackerInsertion*fBeamXRMS_Trk1+fTrk1XOffset));
    det2F =std::unique_ptr<CTPPSTrkDetector>(new CTPPSTrkDetector(fTrackerWidth,fTrackerHeight,fTrackerInsertion*fBeamXRMS_Trk2+fTrk2XOffset));		
    det1B =std::unique_ptr<CTPPSTrkDetector>(new CTPPSTrkDetector(fTrackerWidth,fTrackerHeight,fTrackerInsertion*fBeamXRMS_Trk1+fTrk1XOffset));
    det2B =std::unique_ptr<CTPPSTrkDetector>(new CTPPSTrkDetector(fTrackerWidth,fTrackerHeight,fTrackerInsertion*fBeamXRMS_Trk2+fTrk2XOffset));	

    //Timing Detector Description
    std::vector<double> vToFCellWidth;
    for (int i = 0 ; i < 8 ; i++){
        vToFCellWidth.push_back(fToFCellWidth[i]);
    }
    double pos_tof = fToFInsertion*fBeamXRMS_ToF+fToFXOffset;
    detToF_F =std::unique_ptr<CTPPSToFDetector>(new CTPPSToFDetector(fToFNCellX,fToFNCellY,vToFCellWidth,fToFCellHeight,fToFPitchX,fToFPitchY,pos_tof,fTimeSigma)); 
    detToF_B =std::unique_ptr<CTPPSToFDetector>(new CTPPSToFDetector(fToFNCellX,fToFNCellY,vToFCellWidth,fToFCellHeight,fToFPitchX,fToFPitchY,pos_tof,fTimeSigma)); 

}
CTPPSFastTrackingProducer::~CTPPSFastTrackingProducer()
{
    for (std::map<unsigned int,H_BeamParticle*>::iterator it = m_beamPart.begin(); it != m_beamPart.end(); ++it ) {
        delete (*it).second;
    }
}
// ------------ method called to produce the data  ------------
    void
CTPPSFastTrackingProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;
    //using namespace std;	
    TrackerStationStarting();
    Handle<CTPPSFastRecHitContainer> recHits;
    iEvent.getByToken( _recHitToken,recHits);
    recCellId_F.clear();     recCellId_B.clear();  
    recTof_F.clear();     recTof_B.clear();  
    ReadRecHits(recHits);	
    Reconstruction();
    TrackerStationClear();

    std::unique_ptr<CTPPSFastTrackContainer> output_tracks(new CTPPSFastTrackContainer);
    int  n = 0;
    for ( std::vector<CTPPSFastTrack>::const_iterator i = theCTPPSFastTrack.begin();
            i != theCTPPSFastTrack.end(); i++ ) {
        output_tracks->push_back(*i);
        n += 1;
    }

    iEvent.put(std::move(output_tracks),"CTPPSFastTrack");

}//end
void CTPPSFastTrackingProducer::beginEvent(edm::Event& event, const edm::EventSetup& eventSetup)
{

    TrackerStationStarting();

}
////////////////////
void CTPPSFastTrackingProducer::endEvent(edm::Event& event, const edm::EventSetup& eventSetup)
{
    TrackerStationClear();

}

///////////////////////// 
void CTPPSFastTrackingProducer::TrackerStationClear(){

    TrkStation_F->first.clear(); TrkStation_F->second.clear(); 
    TrkStation_B->first.clear(); TrkStation_B->second.clear();


}
/////////////////////////
void CTPPSFastTrackingProducer::TrackerStationStarting(){
    det1F->clear();
    det1B->clear();
    det2F->clear();
    det2B->clear();
    detToF_F->clear();
    detToF_B->clear();

}


////////////////////////////
    void 
CTPPSFastTrackingProducer::ReadRecHits(edm::Handle<CTPPSFastRecHitContainer> &recHits)
{
    // DetId codification for PSimHit taken from CTPPSPixel- It will be replaced by CTPPSDetId
    // 2014314496 -> Tracker1 zPositive
    // 2014838784 -> Tracker2 zPositive
    // 2046820352 -> Timing   zPositive
    // 2031091712 -> Tracker1 zNegative
    // 2031616000 -> Tracker2 zNegative
    // 2063597568 -> Timing   zNegative

    for (unsigned int irecHits = 0; irecHits < recHits->size(); ++irecHits)
    {
        const CTPPSFastRecHit* recHitDet = &(*recHits)[irecHits];
        unsigned int detlayerId = recHitDet->detUnitId();
        double x = recHitDet->entryPoint().x();
        double y = recHitDet->entryPoint().y();
        double z = recHitDet->entryPoint().z();
        float tof = recHitDet->tof();  
        if(detlayerId == 2014314496) det1F->AddHit(detlayerId,x,y,z);
        else if(detlayerId == 2014838784) det2F->AddHit(detlayerId,x,y,z);
        else if(detlayerId == 2031091712) det1B->AddHit(detlayerId,x,y,z);
        else if(detlayerId == 2031616000) det2B->AddHit(detlayerId,x,y,z);
        else if(detlayerId == 2046820352) { detToF_F->AddHit(x,y,tof); recCellId_F.push_back(detToF_F->findCellId(x,y)); recTof_F.push_back(tof); } 
        else if(detlayerId == 2063597568) { detToF_B->AddHit(x,y,tof); recCellId_B.push_back(detToF_B->findCellId(x,y)); recTof_B.push_back(tof); }

    }//LOOP TRK
    //creating Stations
    TrkStation_F = std::unique_ptr<CTPPSTrkStation>(new std::pair<CTPPSTrkDetector,CTPPSTrkDetector>(*det1F,*det2F));
    TrkStation_B = std::unique_ptr<CTPPSTrkStation>(new std::pair<CTPPSTrkDetector,CTPPSTrkDetector>(*det1B,*det2B));
} // end function

void CTPPSFastTrackingProducer::Reconstruction()
{
    theCTPPSFastTrack.clear(); 
    int Direction;
    Direction=1; //cms positive Z / forward 
    FastReco(Direction,&*pps_stationF);
    Direction=-1; //cms negative Z / backward
    FastReco(Direction,&*pps_stationB);
}//end Reconstruction

bool CTPPSFastTrackingProducer::SearchTrack(int i,int j,int Direction,double& xi,double& t,double& partP,double& pt,double& thx,double& thy,double& x0, double& y0, double& xt, double& yt, double& X1d, double& Y1d, double& X2d, double& Y2d)
{
    // Given 1 hit in Tracker1 and 1 hit in Tracker2 try to make a track with Hector
    double theta=0.;
    xi = 0; t=0; partP=0; pt=0; x0=0.;y0=0.;xt =0.;yt =0.;X1d=0.;Y1d=0.;X2d=0.;Y2d=0.; 
    CTPPSTrkDetector* det1 = NULL;
    CTPPSTrkDetector* det2 = NULL;
    H_RecRPObject*  station = NULL;
    // Separate in forward and backward stations according to direction
    if (Direction>0) {
        det1=&(TrkStation_F->first);det2=&(TrkStation_F->second);
        station = &*pps_stationF;
    } else {
        det1=&(TrkStation_B->first);det2=&(TrkStation_B->second);
        station = &*pps_stationB;
    }
    if (det1->ppsNHits_<=i||det2->ppsNHits_<=j) return false;
    double x1 = det1->ppsX_.at(i); double y1 = det1->ppsY_.at(i);
    double x2 = det2->ppsX_.at(j); double y2 = det2->ppsY_.at(j);
    double eloss;
    //thx and thy are returned in microrad
    ReconstructArm(station,x1,y1,x2,y2,thx,thy,eloss);
    // Protect for unphysical results
    if (std::isnan(eloss)||std::isinf(eloss)||
            std::isnan(thx)  || std::isinf(thx) ||
            std::isnan(thy)  || std::isinf(thy)) return false;
    //
    if (-thx<-100||-thx>300) return false;
    if (thy<-200||thy>200) return false;
    //
    if ( m_verbosity ) std::cout << "thx " << thx << " thy " << thy << " eloss " << eloss << std::endl;

    // Get the start point of the reconstructed track near the origin made by Hector
    x0 = -station->getX0()*um_to_cm;
    y0 = station->getY0()*um_to_cm;
    double ImpPar=sqrt(x0*x0+y0*y0);
    if (ImpPar>fImpParcut) return false;
    if (eloss<0.||eloss>fBeamEnergy) return false;
    // Calculate the reconstructed track parameters 
    theta = sqrt(thx*thx+thy*thy)*urad;
    xi    = eloss/fBeamEnergy;
    double energy= fBeamEnergy*(1.-xi);
    partP = sqrt(energy*energy-ProtonMassSQ);
    t     = -2.*(ProtonMassSQ - fBeamEnergy*energy + fBeamMomentum*partP*cos(theta));
    pt    = sqrt(pow(partP*thx*urad,2)+pow(partP*thy*urad,2));
    if (xi<0.||xi>1.||t<0.||t>10.||pt<=0.) {
        xi = 0.; t=0.; partP=0.; pt=0.; theta=0.; x0=0.;y0=0.;
        return false; // unphysical values 
    }
    //Try to include the timing detector in the track
    ProjectToToF(x1, y1, x2, y2, xt, yt);
    X1d = x1;
    Y1d = y1;
    X2d = x2;
    Y2d = y2;
    return true;
}//end  SearchTrack

void CTPPSFastTrackingProducer::ReconstructArm(H_RecRPObject* pps_station, double x1, double y1, double x2, double y2, double& tx, double& ty, double& eloss)
{
    tx=0.;
    ty=0.;
    eloss=0.;
    if (!pps_station) return;
    // Change the orientation and units according to Hector and LHC coordinates
    x1*=-mm_to_um;
    x2*=-mm_to_um;
    y1*= mm_to_um;
    y2*= mm_to_um;
    pps_station->setPositions(x1,y1,x2,y2);
    double energy = pps_station->getE(AM); // dummy call needed to calculate some Hector internal parameter
    if (std::isnan(energy)||std::isinf(energy)) return;
    tx =  -pps_station->getTXIP();  // change orientation to CMS
    ty =  pps_station->getTYIP();
    eloss = pps_station->getE();
}

void CTPPSFastTrackingProducer::LorentzBoost(LorentzVector& p_out, const string& frame)
{
    // Use a matrix
    double microrad = 1.e-6;
    TMatrixD tmpboost(4,4);
    double alpha_ = 0.;
    double phi_  = fCrossingAngle*microrad;
    if (p_out.pz()<0) phi_*=-1;
    tmpboost(0,0) = 1./cos(phi_);
    tmpboost(0,1) = - cos(alpha_)*sin(phi_);
    tmpboost(0,2) = - tan(phi_)*sin(phi_);
    tmpboost(0,3) = - sin(alpha_)*sin(phi_);
    tmpboost(1,0) = - cos(alpha_)*tan(phi_);
    tmpboost(1,1) = 1.;
    tmpboost(1,2) = cos(alpha_)*tan(phi_);
    tmpboost(1,3) = 0.;
    tmpboost(2,0) = 0.;
    tmpboost(2,1) = - cos(alpha_)*sin(phi_);
    tmpboost(2,2) = cos(phi_);
    tmpboost(2,3) = - sin(alpha_)*sin(phi_);
    tmpboost(3,0) = - sin(alpha_)*tan(phi_);
    tmpboost(3,1) = 0.;
    tmpboost(3,2) = sin(alpha_)*tan(phi_);
    tmpboost(3,3) = 1.;

    if(frame=="LAB") tmpboost.Invert();

    TMatrixD p4(4,1);
    p4(0,0) = p_out.e();
    p4(1,0) = p_out.px();
    p4(2,0) = p_out.py();
    p4(3,0) = p_out.pz();
    TMatrixD p4lab(4,1);
    p4lab = tmpboost * p4;
    p_out.setPx(p4lab(1,0));
    p_out.setPy(p4lab(2,0));
    p_out.setPz(p4lab(3,0));
    p_out.setE(p4lab(0,0));
}
void CTPPSFastTrackingProducer::MatchCellId(int cellId, vector<int> vrecCellId, vector<double> vrecTof, bool& match, double& recTof){
    for (unsigned int i = 0 ; i < vrecCellId.size(); i++){
        if(cellId == vrecCellId.at(i)) {
            match = true;
            recTof = vrecTof.at(i); 
            continue; 
        } 
    } 
}

void CTPPSFastTrackingProducer::FastReco(int Direction,H_RecRPObject* station)
{
    double theta = 0.;
    double xi,t,partP,pt,phi,x0,y0,thx,thy,xt,yt,X1d,Y1d,X2d,Y2d;
    CTPPSTrkDetector* Trk1 = NULL;
    CTPPSTrkDetector* Trk2 = NULL;
    double pos_tof = fToFInsertion*fBeamXRMS_ToF+fToFXOffset;
    int cellId = 0;
    std::vector<double> vToFCellWidth;
    for (int i = 0 ; i < 8 ; i++){
        vToFCellWidth.push_back(fToFCellWidth[i]);
    }
    CTPPSToFDetector* ToF = new CTPPSToFDetector(fToFNCellX,fToFNCellY,vToFCellWidth,fToFCellHeight,fToFPitchX,fToFPitchY,pos_tof,fTimeSigma);
    if (Direction>0) {
        Trk1=&(TrkStation_F->first);Trk2=&(TrkStation_F->second);
    } else {
        Trk1=&(TrkStation_B->first);Trk2=&(TrkStation_B->second);
    }
    // Make a track from EVERY pair of hits combining Tracker1 and Tracker2.
    // The tracks may not be independent as 1 hit may belong to more than 1 track.
    for(int i=0;i<(int)Trk1->ppsNHits_;i++) {
        for(int j=0;j<(int)Trk2->ppsNHits_;j++){
            if (SearchTrack(i,j,Direction,xi,t,partP,pt,thx,thy,x0,y0,xt,yt,X1d,Y1d,X2d,Y2d)) {
                // Check if the hitted timing cell matches the reconstructed track
                cellId = ToF->findCellId(xt,yt);
                double recTof = 0.; 
                bool matchCellId = false; 
                if (Direction > 0 ) {
                    theta = sqrt(thx*thx+thy*thy)*urad;
                    MatchCellId(cellId, recCellId_F, recTof_F, matchCellId, recTof); 
                } 
                else if (Direction<0) { 
                    theta = CLHEP::pi - sqrt(thx*thx+thy*thy)*urad;
                    MatchCellId(cellId, recCellId_B, recTof_B, matchCellId, recTof); 
                }
                phi   = (Direction>0)?-atan2(thy,-thx):atan2(thy,thx); // defined according to the positive direction


                double px = partP*sin(theta)*cos(phi);
                double py = partP*sin(theta)*sin(phi);
                double pz = partP*cos(theta);
                double  e = sqrt(partP*partP+ProtonMassSQ);
                LorentzVector p(px,py,pz,e);
                // Invert the Lorentz boost made to take into account the crossing angle during simulation
                if (fCrossAngleCorr) LorentzBoost(p,"MC");
                //Getting the Xi and t (squared four momentum transferred) of the reconstructed track
                Get_t_and_xi(const_cast<LorentzVector*>(&p),t,xi);
                double pxx = p.px(); double pyy = p.py(); double pzz = p.pz(); //double ee = p.E();	
                math::XYZVector momentum (pxx,pyy,pzz);
                math::XYZPoint vertex (x0,y0,0);

                track.setp(momentum);
                track.setvertex(vertex);
                track.sett(t);
                track.setxi(xi);
                track.setx1(X1d);
                track.sety1(Y1d);
                track.setx2(X2d);
                track.sety2(Y2d);
                if (matchCellId) {
                    track.setcellid(cellId);
                    track.settof(recTof);
                } 
                else {
                    track.setcellid(0);
                    track.settof(0.);
                } 
                theCTPPSFastTrack.push_back(track);
            }
        } 
    }
}//end FastReco

void CTPPSFastTrackingProducer::Get_t_and_xi(const LorentzVector* proton,double& t,double& xi) {
    t = 0.;
    xi = -1.;
    if (!proton) return;
    double mom = sqrt((proton->px())*(proton->px())+(proton->py())*(proton->py())+(proton->pz())*(proton->pz()));
    if (mom>fBeamMomentum) mom=fBeamMomentum;
    double energy = proton->e();
    double theta  = (proton->pz()>0)?proton->theta():CLHEP::pi-proton->theta();
    t      = -2.*(ProtonMassSQ-fBeamEnergy*energy+fBeamMomentum*mom*cos(theta));
    xi     = (1.0-energy/fBeamEnergy);
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------

void CTPPSFastTrackingProducer::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------

void CTPPSFastTrackingProducer::endStream() 
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(CTPPSFastTrackingProducer);
