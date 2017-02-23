// ROOT #includes
#include "FastSimulation/PPSFastSim/interface/PPSSim.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <TMatrixD.h>
//=====================================================================================================
PPSSim::PPSSim(bool ext_gen): fExternalGenerator(ext_gen),
    fVerbose(false),NEvent(0),fGenMode(""),
    fBeamLine1File(""),fBeamLine2File(""),fBeam1Direction(1),fBeam2Direction(1),fShowBeamLine(false),
    fCollisionPoint(""),fBeamLineLength(500),fBeamEnergy(0),fBeamMomentum(0),
    fBeamXRMS_Trk1(0.),fBeamXRMS_Trk2(0.),fBeamXRMS_ToF(0.),
    fCrossingAngle(0.),fCrossAngleCorr(false),fKickersOFF(false),
    fDetectorClosestX(-2.),fMaxXfromBeam(-25),fMaxYfromBeam(10),
    fTrackerZPosition(0.),fTrackerLength(0.),fTrackerWidth(0.),fTrackerHeight(0.),
    fToFWidth(0.),fToFHeight(0.),fToFCellW(0.),fToFCellH(0.),fToFPitchX(0.),fToFPitchY(0.),fToFNCellX(0),fToFNCellY(0),
    fToFZPosition(0.),fTrackerInsertion(0.),fToFInsertion(0.),
    fTCL4Position1(0.),fTCL4Position2(0.),fTCL5Position1(0.),fTCL5Position2(0.),
    fSmearVertex(false),fVtxMeanX(0.),fVtxMeanY(0.),fVtxMeanZ(0.),fVtxSigmaX(0.),fVtxSigmaY(0.),fVtxSigmaZ(0.),
    fSmearHit(1.),fHitSigmaX(0.),fHitSigmaY(0.),fHitSigmaZ(0.),fTimeSigma(0.),
    fDet1XOffsetF(0.),fDet2XOffsetF(0.),fDet1XOffsetB(0.),fDet2XOffsetB(0.),
    fSmearAngle(false),fBeamAngleRMS(0.),fSmearEnergy(false),fBeamEnergyRMS(0.),
    fFilterHitMap(true),fApplyFiducialCuts(true),
    fTrackImpactParameterCut(0.),fMinThetaXatDet1(-200),fMaxThetaXatDet1(200),
    fMinThetaYatDet1(-200),fMaxThetaYatDet1(200),
    xi_min(0.),xi_max(0.),t_min(0.),t_max(0.),fPhiMin(-TMath::Pi()),fPhiMax(TMath::Pi()),
    fEtaMin(7.),fMomentumMin(3000.),fCentralMass(0.),fCentralMassErr(0.),
    CheckPoints(),fSimBeam(false)
{
    beam1profile=NULL;
    beam2profile=NULL;
    gRandom3  = new TRandom3(0);
}
void PPSSim::BeginRun()
{    
    if (fVerbose) edm::LogWarning("debug") << "fBeamLine1File: " << fBeamLine1File ;
    if (fVerbose) edm::LogWarning("debug") << "fBeamLine2File: " << fBeamLine2File ;	
    extern int kickers_on;
    kickers_on = (fKickersOFF)?0:1;
    beamlineF = new H_BeamLine(-1,fBeamLineLength);
    beamlineB = new H_BeamLine( 1,fBeamLineLength);
    beamlineF->fill(fBeamLine2File,fBeam2Direction,fCollisionPoint);
    beamlineB->fill(fBeamLine1File,fBeam1Direction,fCollisionPoint);
    //
    //set_Strengths();
    beamlineF->offsetElements( 120, 0.097);
    beamlineB->offsetElements( 120, 0.097);
    beamlineF->calcMatrix(); beamlineB->calcMatrix();
    if (fShowBeamLine) {
        beamlineF->showElements();
        beamlineB->showElements();
    }
    // Create a particle to get the beam energy from the beam file
    //
    pps_stationF = new H_RecRPObject(fTrackerZPosition,fTrackerZPosition+fTrackerLength,*beamlineF);
    pps_stationB = new H_RecRPObject(fTrackerZPosition,fTrackerZPosition+fTrackerLength,*beamlineB);
    //
    // check the kinematic limits in case it is requested to generate the higgs mass in the central system
    if (fGenMode=="M_X") { // check the kinematic limits
        if (xi_min*(2.*fBeamEnergy)>fCentralMass+fCentralMassErr||xi_max*(2.*fBeamEnergy)<fCentralMass-fCentralMassErr) {
            edm::LogWarning("debug") << "xi limits outside kinematic limits for the given Central mass. Stopping..." ;
            exit(1);
        }
    }
    if (fSimBeam) {
        TH2F* h1 = new TH2F(*GenBeamProfile(-fTrackerZPosition));
        TH2F* h2 = new TH2F(*GenBeamProfile(fTrackerZPosition));
        if (h1) {h1->SetName("BeamProfileB_1"); h1->Write(); delete h1;}
        if (h2) {h2->SetName("BeamProfileF_1"); h2->Write(); delete h2;}
        h1 = GenBeamProfile(-(fTrackerZPosition+fTrackerLength));
        h2 = GenBeamProfile(  fTrackerZPosition+fTrackerLength);
        if (h1) {h1->SetName("BeamProfileB_2"); h1->Write(); delete h1;}
        if (h2) {h2->SetName("BeamProfileF_2"); h2->Write(); delete h2;}
        TH2F* ht = GenBeamProfile(fToFZPosition);
        if (ht) {ht->SetName("BeamProfileB_ToF"); ht->Write(); delete ht;}
        ht = GenBeamProfile(-fToFZPosition);
        if (ht) {ht->SetName("BeamProfileF_ToF"); ht->Write(); delete ht;}
        // if colimator position not defined, try to get it from the beamline
        if (fTCL4Position1*fTCL4Position2==0) {
            H_OpticalElement* tcl = beamlineB->getElement("TCL.4R5.B1"); if (tcl) fTCL4Position1=tcl->getS();
            tcl = beamlineF->getElement("TCL.4L5.B2");if (tcl) fTCL4Position2=tcl->getS();
        }
        if (fTCL5Position1*fTCL5Position2==0) {
            H_OpticalElement* tcl = beamlineB->getElement("TCL.5R5.B1"); if (tcl) fTCL5Position1=tcl->getS();
            tcl = beamlineF->getElement("TCL.5L5.B2");if (tcl) fTCL5Position2=tcl->getS();
            tcl=beamlineF->getElement("TCL.5R5.B1");
        }
        if (fTCL4Position1*fTCL4Position2>0){
            TH2F* h1 = GenBeamProfile(-fTCL4Position1); // use negative position to tell gen. which beamline to chose
            fBeam1PosAtTCL4=make_pair<double,double>(h1->GetMean(1),h1->GetMean(2));
            fBeam1RMSAtTCL4=make_pair<double,double>(h1->GetRMS(1),h1->GetRMS(2));
            TH2F* h2 = GenBeamProfile( fTCL4Position2);
            fBeam2PosAtTCL4=make_pair<double,double>(h2->GetMean(1),h2->GetMean(2));
            fBeam2RMSAtTCL4=make_pair<double,double>(h2->GetRMS(1),h2->GetRMS(2));
        }
        if (fTCL5Position1*fTCL5Position2>0){
            TH2F* h1 = GenBeamProfile(-fTCL5Position1); // use negative position to tell gen. which beamline to chose
            fBeam1PosAtTCL5=make_pair<double,double>(h1->GetMean(1),h1->GetMean(2));
            fBeam1RMSAtTCL5=make_pair<double,double>(h1->GetRMS(1),h1->GetRMS(2));
            TH2F* h2 = GenBeamProfile( fTCL5Position2);
            fBeam2PosAtTCL5=make_pair<double,double>(h2->GetMean(1),h2->GetMean(2));
            fBeam2RMSAtTCL5=make_pair<double,double>(h2->GetRMS(1),h2->GetRMS(2));
        }
    }
    // 
    fGen = new PPSSpectrometer<Gen>();
    fSim = new PPSSpectrometer<Sim>();
    fReco= new PPSSpectrometer<Reco>();
    //
    PPSTrkDetector* det1 = new PPSTrkDetector(fTrackerWidth,fTrackerHeight,fTrackerInsertion*fBeamXRMS_Trk1);
    PPSTrkDetector* det2 = new PPSTrkDetector(fTrackerWidth,fTrackerHeight,fTrackerInsertion*fBeamXRMS_Trk1);
    TrkStation_F = new std::pair<PPSTrkDetector,PPSTrkDetector>(*det1,*det2);
    ToFDet_F  = new PPSToFDetector(fToFNCellX,fToFNCellY,fToFCellW,fToFCellH,fToFPitchX,fToFPitchY,fToFInsertion*fBeamXRMS_ToF,fTimeSigma);

    det1 = new PPSTrkDetector(fTrackerWidth,fTrackerHeight,fTrackerInsertion*fBeamXRMS_Trk2);
    det2 = new PPSTrkDetector(fTrackerWidth,fTrackerHeight,fTrackerInsertion*fBeamXRMS_Trk2);
    TrkStation_B = new std::pair<PPSTrkDetector,PPSTrkDetector>(*det1,*det2);
    ToFDet_B  = new PPSToFDetector(fToFNCellX,fToFNCellY,fToFCellW,fToFCellH,fToFPitchX,fToFPitchY,fToFInsertion*fBeamXRMS_ToF,fTimeSigma);
    fToFHeight = ToFDet_F->GetHeight();
    fToFWidth  = ToFDet_F->GetWidth();
    //   Check the overall kinematic limits
    if (fGenMode=="M_X") {
        if (xi_min*(2.*fBeamEnergy)>fCentralMass+fCentralMassErr||xi_max*(2.*fBeamEnergy)<fCentralMass-fCentralMassErr) {
            edm::LogWarning("debug") << "xi limits outside kinematic limits for the given Central mass. Stopping..." ;
            exit(1);
        }
    }
}

//
void PPSSim::EndRun()
{
}
void PPSSim::BeginEvent()
{
    fGen->clear();
    fSim->clear();
    fReco->clear();
    protonF = NULL;
    protonB = NULL;
    fHasStoppedF = false;
    fHasStoppedB = false;
    NVertex=0;
    fVertex.clear();
    protonsOut.clear();
    fHasStopped.clear();
    TrkStation_F->first.clear(); TrkStation_F->second.clear(); 
    TrkStation_B->first.clear(); TrkStation_B->second.clear();
    ToFDet_F->clear();
    ToFDet_B->clear();

}
void PPSSim::EndEvent()
{
    for(int i=0;i<NVertex;i++) {
        protonF = (protonsOut[i].first); protonB = (protonsOut[i].second);
        if (protonF) delete protonF;
        if (protonB) delete protonB;
    }
    protonsOut.clear();
    protonF=NULL;
    protonB=NULL;
}
void PPSSim::Run() {
    if (!fExternalGenerator) Generation();
    if (fVerbose) edm::LogWarning("debug") << "PPSSim: Starting Simulation step."; 
    Simulation();
    if (fVerbose) edm::LogWarning("debug") << "PPSSim: Starting Digitization step.";
    Digitization(); // a fake digitization procedure
    if (fVerbose) edm::LogWarning("debug") << "PPSSim: Starting Reconstruction step.";
    Reconstruction();
}
void PPSSim::Generation()
{
    // Uses the CMS units in the vertex and mm for the PPS parameters
    // sorts a proton in the forward direction
    double t1,xi1,phi1;
    double t2,xi2,phi2;
    if (fGenMode=="M_X") {
        if (fCentralMass==0) {
            edm::LogWarning("debug") << "PPSSim::Generation: Central mass not defined. Exiting...";
            exit(1);
        }
        GenCentralMass(t1,t2,xi1,xi2,phi1,phi2);
    }
    else {
        GenSingleParticle(t1,xi1,phi1);
        GenSingleParticle(t2,xi2,phi2);
    }
    int nvtx=add_Vertex(fVtxMeanX,fVtxMeanY,fVtxMeanZ);
    protonF=new TLorentzVector(shoot(t1,xi1,phi1,1)); //  1 = positive/forward direction
    protonB=new TLorentzVector(shoot(t2,xi2,phi2,-1));
    add_OutgoingParticle(nvtx-1,protonF,protonB);
    fHasStoppedF=false;fHasStoppedB=false;
    set_GenData();
}
//
// fill the tree structures 
void PPSSim::ReadGenEvent(const std::vector<TrackingParticle>* gentrackingP)
{
}
//Using reco gen particle
void PPSSim::ReadGenEvent(const std::vector<reco::GenParticle>* genP)
{
    if (!genP) return;
    int ivtx = -1;
    int colId = -1;
    double momB = 0.;
    double momF = 0.;
    TLorentzVector* pF = NULL;
    TLorentzVector* pB = NULL;
    double vtxX=0.;
    double vtxY=0.;
    double vtxZ=0.;
    for(size_t i=0;i<genP->size();++i) {
        const reco::GenParticle& p = (*genP)[i];
        if (p.pdgId()!=2212) continue;
        if (p.status()  !=1) continue;
        //double pz = p.pt()*sinh (p.eta());
        double px = p.px();double py=p.py();double pz = p.pz();
        if (ivtx<0) {
            ivtx=0;
            vtxX=p.vx();vtxY=p.vy();vtxZ=p.vz();// Contrary to HepMC, reco::genParticle already uses cm, so no convertion is needed
            colId = p.collisionId();
        }

        if (colId!=p.collisionId()) {
            int nvtx=add_Vertex(vtxX,vtxY,vtxZ);
            if (ivtx!=nvtx-1) {edm::LogWarning("debug")<< "WARNING: unexpected vertex number.";}
            add_OutgoingParticle(nvtx-1,pF,pB);
            colId = p.collisionId();
            vtxX=p.vx();vtxY=p.vy();vtxZ=p.vz();
            ivtx++;
            momF=0.; momB=0.;
            pF=NULL;pB=NULL;
        } else {
            // verify the vertex consistency
            if (vtxX!=p.vx()||vtxY!=p.vy()||vtxZ!=p.vz()) {
                edm::LogWarning("debug") << "WARNING: unexpected new vertex position";
            }
        }
        if (p.eta()>0&&momF<pz) {
            momF=pz; pF = new TLorentzVector(px,py,pz,sqrt(px*px+py*py+pz*pz+ProtonMassSQ));
        } else if (p.eta()<0&&momB<fabs(pz)) {
            momB=fabs(pz);pB = new TLorentzVector(px,py,pz,sqrt(px*px+py*py+pz*pz+ProtonMassSQ));
        }
        // this is the  last particle, add it anyway..
        if (i==genP->size()-1) {
            int nvtx=add_Vertex(vtxX,vtxY,vtxZ);
            if (ivtx!=nvtx-1) {edm::LogWarning("debug") << "WARNING: unexpected vertex number.";}
            if (fVerbose) {if(pF) pF->Print();if (pB) pB->Print();}
            add_OutgoingParticle(nvtx-1,pF,pB);
        }
    }
    set_GenData();
}
void PPSSim::ReadGenEvent(const HepMC::GenEvent* evt)
{
    using namespace CLHEP;
    TLorentzVector* pF = NULL;
    TLorentzVector* pB = NULL;
    if (!evt) return;
    int nvtx =0;
    double vtxX = 0.;
    double vtxY = 0.;
    double vtxZ = 0.;
    for(HepMC::GenEvent::vertex_const_iterator ivtx = evt->vertices_begin();ivtx!=evt->vertices_end();ivtx++) {
        if (fVerbose) (*ivtx)->print();
        if ((*ivtx)->id()!=0) continue;
        vtxX = (*ivtx)->position().x()/cm; // CMS uses cm but HepMC uses mm
        vtxY = (*ivtx)->position().y()/cm;
        vtxZ = (*ivtx)->position().z()/cm;

        // choose the highest momentum particle on each side to be propagated
        double momF = 0.; double momB = 0.;

        for(HepMC::GenVertex::particles_out_const_iterator pItr = (*ivtx)->particles_out_const_begin();
                pItr!= (*ivtx)->particles_out_const_end();pItr++) {
            if (fVerbose) (*pItr)->print();
            if ((*pItr)->status() != 1) continue; // this is not a final state particle
            if ((*pItr)->pdg_id()!=2212) continue; // only protons to be processed
            if (fabs((*pItr)->momentum().eta()) < fEtaMin) continue; 
            if ((*pItr)->momentum().e()<fMomentumMin) continue; 
            if ((*pItr)->momentum().pz()>0&&(*pItr)->momentum().pz()>momF) {
                momF= (*pItr)->momentum().pz();
                pF  = new TLorentzVector((*pItr)->momentum().px(),(*pItr)->momentum().py(),(*pItr)->momentum().pz(),
                        sqrt(ProtonMassSQ+pow((*pItr)->momentum().px(),2)+pow((*pItr)->momentum().py(),2)+pow((*pItr)->momentum().pz(),2)));
            } else if ((*pItr)->momentum().pz()<0&&fabs((*pItr)->momentum().pz())>momB){
                momB = fabs((*pItr)->momentum().pz());
                pB   = new TLorentzVector((*pItr)->momentum().px(),(*pItr)->momentum().py(),(*pItr)->momentum().pz(),
                        sqrt(ProtonMassSQ+pow((*pItr)->momentum().px(),2)+pow((*pItr)->momentum().py(),2)+pow((*pItr)->momentum().pz(),2)));
            }
        }
    }
    if (!pF&&!pB) return;
    nvtx=add_Vertex(vtxX,vtxY,vtxZ);
    add_OutgoingParticle(nvtx-1,pF,pB);
    if (fVerbose) {
        if (pF) {pF->Print();}
        if (pB) {pB->Print();}
    }
    set_GenData();
}
void PPSSim::set_GenData()
{
    for(int i=0;i<NVertex;i++) {
        protonF = (protonsOut[i].first); protonB = (protonsOut[i].second);
        int tF=-1;if (protonF) tF=i;
        int tB=-1;if (protonB) tB=i;
        fGen->Vertices->Add(fVertex[i].x(),fVertex[i].y(),fVertex[i].z(),tF,tB);
        double t,xi;
        if (protonF) {Get_t_and_xi(protonF,t,xi); fGen->ArmF.addParticle(*protonF,t,xi);}
        if (protonB) {Get_t_and_xi(protonB,t,xi); fGen->ArmB.addParticle(*protonB,t,xi);}
        if (protonF){
            ApplyBeamSmearing(const_cast<TLorentzVector&>(*protonF));
        }
        if (protonB) {
            ApplyBeamSmearing(const_cast<TLorentzVector&>(*protonB));
        }
    }
}
void PPSSim::Get_t_and_xi(const TLorentzVector* proton,double& t,double& xi) {
    t = 0.;
    xi = -1.;
    if (!proton) return;
    double mom    = proton->P();
    if (mom>fBeamMomentum) mom=fBeamMomentum;
    double energy = proton->E();
    double theta  = (proton->Pz()>0)?proton->Theta():TMath::Pi()-proton->Theta();
    t      = -2.*(ProtonMassSQ-fBeamEnergy*energy+fBeamMomentum*mom*cos(theta));
    xi     = (1.0-energy/fBeamEnergy);
}
void PPSSim::Simulation()
{
    for(int i=0;i<NVertex;i++) {
        double vtxX=fVertex[i].x();
        double vtxY=fVertex[i].y();
        double vtxZ=fVertex[i].z();
        protonF = (protonsOut[i].first); protonB = (protonsOut[i].second);
        fHasStoppedF=false;fHasStoppedB=false;
        // At this point, one should be using the CMS units (cm)
        int tF=-1; if (protonF) tF=i;
        int tB=-1; if (protonB) tB=i;

        fSim->Vertices->Add(vtxX,vtxY,vtxZ,tF,tB);
        // FIRST, propagate to the positive(forward) direction, then to the other side
        //
        // Propagate until PPS, filling pps_station accordingly for the reconstruction if needed
        // Remember: 
        //         HECTOR uses um for X and Y coordinates and m for Z
        //         is backward for LHC, which means when propagating to the CMS forward direction, one needs to rotate it
        //         by means of doing x_LHC = -x_CMS and z = -z 

        H_BeamParticle *part = NULL;
        if (protonF) {
            int Direction = 1;
            double t,xi;
            Get_t_and_xi(protonF,t,xi);
            fSim->ArmF.AddTrack(*protonF,t,xi);
            //
            if (fCrossAngleCorr) LorentzBoost(const_cast<TLorentzVector&>(*protonF),"LAB");
            //
            part = new H_BeamParticle(ProtonMass,1);
            part->setPosition(-(vtxX-fVtxMeanX)*cm_to_um,(vtxY-fVtxMeanY)*cm_to_um,0.,0.,-(vtxZ)*cm_to_m);
            part->set4Momentum(-protonF->Px(),protonF->Py(),-protonF->Pz(),protonF->E());
            part->computePath(beamlineF);
            Propagate(part,Direction);
            if (part) {delete part;part = NULL;}
        } 
        //
        //  Propagate to the negative/backward direction
        //
        if (protonB) {
            int Direction = -1;
            double t,xi;
            Get_t_and_xi(protonB,t,xi);
            fSim->ArmB.AddTrack(*protonB,t,xi);
            //
            if (fCrossAngleCorr) LorentzBoost(const_cast<TLorentzVector&>(*protonB),"LAB");
            //
            part = new H_BeamParticle(ProtonMass,1);
            part->setPosition(-(vtxX-fVtxMeanX)*cm_to_um,(vtxY-fVtxMeanY)*cm_to_um,0.,0.,-(vtxZ)*cm_to_m);
            part->set4Momentum(-protonB->Px(),protonB->Py(),-protonB->Pz(),protonB->E()); // HECTOR uses always positive z momentum
            part->computePath(beamlineB);
            Propagate(part,Direction);
            if (part) {delete part;part = NULL;}
        } 
    }
}
void PPSSim::Reconstruction()
{
    int Direction;
    Direction=1;
    TrackerReco(Direction,pps_stationF,&(fReco->ArmF));
    Direction=-1;
    TrackerReco(Direction,pps_stationB,&(fReco->ArmB));
    ToFReco();
}
bool PPSSim::SearchTrack(int i,int j,int Direction,double& xi,double& t,double& partP,double& pt,double& thx,double& thy,double& x0, double& y0)
{
    double theta=0.;
    xi = 0; t=0; partP=0; pt=0; x0=0.;y0=0.;
    PPSTrkDetector* det1 = NULL;
    PPSTrkDetector* det2 = NULL;
    H_RecRPObject*  station = NULL;
    if (Direction>0) {
        det1=&(TrkStation_F->first);det2=&(TrkStation_F->second);
        station = pps_stationF;
    } else {
        det1=&(TrkStation_B->first);det2=&(TrkStation_B->second);
        station = pps_stationB;
    }

    if (det1->NHits<=i||det2->NHits<=j) return false;
    //
    double x1 = det1->X.at(i); double y1 = det1->Y.at(i);
    double x2 = det2->X.at(j); double y2 = det2->Y.at(j);
    double eloss;
    ReconstructArm(station, x1,y1,x2,y2,thx,thy,eloss);
    // Protect for unphysical results
    if (std::isnan(eloss)||std::isinf(eloss)||
            std::isnan(thx)  || std::isinf(thx) ||
            std::isnan(thy)  || std::isinf(thy)) return false;
    //
    if (-thx<-100||-thx>300) return false;
    if (thy<-200||thy>200) return false;
    //
    x0 = -station->getX0()*um_to_cm;
    y0 = station->getY0()*um_to_cm;
    double ImpPar=sqrt(x0*x0+y0*y0);
    if (fTrackImpactParameterCut>0.) {
        if (ImpPar>fTrackImpactParameterCut) return false;
    }
    if (eloss<0||eloss>fBeamEnergy) return false;
    theta = sqrt(thx*thx+thy*thy)*urad;
    xi    = eloss/fBeamEnergy;
    double energy= fBeamEnergy*(1.-xi);
    partP = sqrt(energy*energy-ProtonMassSQ);
    t     = -2.*(ProtonMassSQ - fBeamEnergy*energy+fBeamMomentum*partP*cos(theta));
    pt    = sqrt(pow(partP*thx*urad,2)+pow(partP*thy*urad,2));
    if (xi<0.||xi>1.||t<0.||t>10.||pt<=0.) {
        xi = 0; t=0; partP=0; pt=0; theta=0; x0=0.;y0=0.;
        return false; // unphysical values 
    }
    return true;
}
void PPSSim::TrackerReco(int Direction,H_RecRPObject* station,PPSBaseData* arm_base)
{
    //
    PPSRecoData* arm = dynamic_cast<PPSRecoData*>(arm_base);
    double xi,t,partP,pt,phi,theta,x0,y0,thx,thy;
    PPSTrkDetector* Trk1 = NULL;
    PPSTrkDetector* Trk2 = NULL;
    if (Direction>0) {
        Trk1=&(TrkStation_F->first);Trk2=&(TrkStation_F->second);
    } else {
        Trk1=&(TrkStation_B->first);Trk2=&(TrkStation_B->second);
    }
    for(int i=0;i<Trk1->NHits;i++) arm->AddHitTrk1(Trk1->X.at(i),Trk1->Y.at(i));
    for(int i=0;i<Trk2->NHits;i++) arm->AddHitTrk2(Trk2->X.at(i),Trk2->Y.at(i));

    for(int i=0;i<(int)Trk1->NHits;i++) {
        for(int j=0;j<(int)Trk2->NHits;j++){
            if (SearchTrack(i,j,Direction,xi,t,partP,pt,thx,thy,x0,y0)) {
                theta = sqrt(thx*thx+thy*thy)*urad;
                phi   = (Direction>0)?-atan2(thy,-thx):atan2(thy,thx); // defined according to the positive direction
                if (Direction<0) { theta=TMath::Pi()-theta; }
                double px = partP*sin(theta)*cos(phi);
                double py = partP*sin(theta)*sin(phi);
                double pz = partP*cos(theta);
                double  e = sqrt(partP*partP+ProtonMassSQ);
                TLorentzVector p(px,py,pz,e);
                if (fCrossAngleCorr) LorentzBoost(p,"MC");
                Get_t_and_xi(const_cast<TLorentzVector*>(&p),t,xi);
                arm->AddTrack(p,t,xi);
                arm->get_Track().set_HitDet1(Trk1->X.at(i),Trk1->Y.at(i));
                arm->get_Track().set_HitDet2(Trk2->X.at(j),Trk2->Y.at(j));
                arm->get_Track().set_X0(x0);
                arm->get_Track().set_Y0(y0);
                arm->get_Track().set_Phi(phi);
                arm->get_Track().set_ThetaAtIP(thx,thy); // thx is given in CMS coordinates
            }
        } 
    }
}
void PPSSim::ToFReco()
{
    PPSRecoTracks* tracksF=&(fReco->ArmF.Tracks);
    if (tracksF->size()==0) return;
    PPSRecoTracks* tracksB=&(fReco->ArmB.Tracks);
    if (tracksB->size()==0) return;
    PPSRecoVertex* vtxs = (PPSRecoVertex*)fReco->Vertices;
    vtxs->clear();

    double vtxX,vtxY,vtxZ;
    double tofF,tofB,ToFtot,d_ToFtot;
    double xt,yt;
    int cellidF=0,cellidB=0;

    d_ToFtot = sqrt(2.)*fTimeSigma; // uncertainty on ToFtot due to detector resolution
    int Nsigma = 3.0;              // # of sigmas (CL for vertex reconstruction)

    for(int i=0;i<(int)tracksF->size();i++){
        ProjectToToF(tracksF->at(i).Det1.X,tracksF->at(i).Det1.Y,tracksF->at(i).Det2.X,tracksF->at(i).Det2.Y,xt,yt);
        cellidF = ToFDet_F->findCellId(xt,yt);

        if (cellidF==0) continue;
        for(int j=0;j<(int)tracksB->size();j++) {
            ProjectToToF(tracksB->at(j).Det1.X,tracksB->at(j).Det1.Y,tracksB->at(j).Det2.X,tracksB->at(j).Det2.Y,xt,yt);
            cellidB = ToFDet_B->findCellId(xt,yt);

            if (cellidB==0) continue;
            for(int k=0;k<ToFDet_F->GetMultiplicityByCell(cellidF);k++) {
                tofF=ToFDet_F->get_ToF(cellidF).at(k);
                if (ToFDet_F->GetADC(cellidF,k)==0) edm::LogWarning("debug") << "WARNING: no ADC found";
                for(int l=0;l<ToFDet_B->GetMultiplicityByCell(cellidB);l++) {
                    tofB=ToFDet_B->get_ToF(cellidB).at(l);
                    if (ToFDet_B->GetADC(cellidB,l)==0) edm::LogWarning("debug") << "WARNING: no ADC found";
                    ToFtot = tofF+tofB;
                    if (fabs(ToFtot-2*fToFZPosition/c_light_ns)>Nsigma*d_ToFtot) continue;
                    vtxZ=-c_light_ns*(tofF-tofB)/2.0*m_to_cm;
                    vtxX=(tracksF->at(i).get_X0()+tracksB->at(j).get_X0())/2.; // this is not very meaningful, there is not enough precision
                    vtxY=(tracksF->at(i).get_Y0()+tracksB->at(j).get_Y0())/2.; // idem
                    if (ToFDet_F->GetMultiplicityByCell(cellidF)==1&&ToFDet_B->GetMultiplicityByCell(cellidB)==1&&
                            ToFDet_F->GetADC(cellidF,k)==1&&ToFDet_B->GetADC(cellidB,l)==1) {
                        double xc=0.,yc=0.;
                        ToFDet_F->get_CellCenter(cellidF,xc,yc);
                        tracksF->at(i).set_HitToF(cellidF,tofF,xc,yc);// Add this information only for vertices
                        ToFDet_B->get_CellCenter(cellidB,xc,yc);
                        tracksB->at(j).set_HitToF(cellidB,tofB,xc,yc);// without ambiguities, using the tof cell center
                        vtxs->AddGolden(vtxX,vtxY,vtxZ,i,j);
                    } else {
                        vtxs->Add(vtxX,vtxY,vtxZ,i,j);
                    }
                }
            }
        } 
    } 
}
void PPSSim::ReconstructArm(H_RecRPObject* pps_station, double x1, double y1, double x2, double y2, double& tx, double& ty, double& eloss)
{
    tx=0.;
    ty=0.;
    eloss=0.;
    if (!pps_station) return;
    // Change the orientation and units according to Hector
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
void PPSSim::Digitization()
{
    //    Fake method to mimic a digitization procedure
    //    Just copy the information from the fSim branch and smear the hit according to a given
    //    detector resolution;

    TrackerDigi(&(fSim->ArmF),TrkStation_F);
    TrackerDigi(&(fSim->ArmB),TrkStation_B);
    ToFDigi(&(fSim->ArmF),ToFDet_F);
    ToFDigi(&(fSim->ArmB),ToFDet_B);
}
void PPSSim::TrackerDigi(const PPSBaseData* arm_sim,PPSTrkStation* TrkDet)
{
    if(!arm_sim||!TrkDet) return;
    //
    PPSTrkDetector* det1 = &(TrkDet->first);
    PPSTrkDetector* det2 = &(TrkDet->second);
    det1->clear();
    det2->clear();
    for(int i=0;i<const_cast<PPSBaseData*>(arm_sim)->TrkDet1.NHits();i++){
        //if (arm_sim->TrkDet1.HasStopped.at(i)) {arm_reco->TrkDet1.AddHit(0,0,0,0,0,1);continue;}
        double x = arm_sim->TrkDet1.at(i).X-fDet1XOffsetF*um_to_mm;
        double y = arm_sim->TrkDet1.at(i).Y;
        double z = 0.;//arm_sim->TrkDet1.Z.at(i);
        HitSmearing(x,y,z);
        if (fFilterHitMap&&(x>fDetectorClosestX||x<fMaxXfromBeam||fabs(y)>fabs(fMaxYfromBeam))) {
            continue;
        } 
        if (fApplyFiducialCuts) {
            double xmin = fTrackerInsertion*fBeamXRMS_Trk1;
            double xmax = xmin+fTrackerWidth;
            if (fabs(x)<xmin||fabs(x)>xmax||fabs(y)>fabs(fTrackerHeight/2)) { // use ABS because the detector are on the negative X side
                continue;
            }
        }
        det1->AddHit(x,y,z);
    }
    for(int i=0;i<const_cast<PPSBaseData*>(arm_sim)->TrkDet2.NHits();i++){
        double x = arm_sim->TrkDet2.at(i).X-fDet2XOffsetF*um_to_mm;
        double y = arm_sim->TrkDet2.at(i).Y;
        double z = 0.;
        HitSmearing(x,y,z);
        if (fFilterHitMap&&(x>fDetectorClosestX||x<fMaxXfromBeam||fabs(y)>fabs(fMaxYfromBeam))) {
            continue;
        } 
        if (fApplyFiducialCuts) {
            double xmin = fTrackerInsertion*fBeamXRMS_Trk2;
            double xmax = xmin+fTrackerWidth;
            if (fabs(x)<xmin||fabs(x)>xmax||fabs(y)>fabs(fTrackerHeight/2)) { // use ABS because the detector are on the negative X side
                continue;
            }
        }
        det2->AddHit(x,y,z);
    }
}
void PPSSim::ToFDigi(const PPSBaseData* arm_sim,PPSToFDetector* ToFDet)
{
    if(!arm_sim||!ToFDet) return;
    // what direction?
    PPSRecoData* arm_reco=NULL;
    ToFDet->clear();
    if (ToFDet==ToFDet_F) arm_reco = &(fReco->ArmF);
    if (ToFDet==ToFDet_B) arm_reco = &(fReco->ArmB);
    //
    for(int i=0;i<const_cast<PPSBaseData*>(arm_sim)->ToFDet.NHits();i++){
        double x = arm_sim->ToFDet.at(i).X;
        double y = arm_sim->ToFDet.at(i).Y;
        if (fFilterHitMap&&(x>fDetectorClosestX||x<fMaxXfromBeam||fabs(y)>fabs(fMaxYfromBeam))) {
            continue;
        }
        if (fApplyFiducialCuts) {
            double xmin = fToFInsertion*fBeamXRMS_ToF;
            double xmax = xmin+fToFWidth;
            if (fabs(x)<xmin||fabs(x)>xmax||fabs(y)>fabs(fToFHeight/2)) { // use ABS because the detector are on the negative X side
                continue;
            }
        }
        double t = arm_sim->ToFDet.at(i).ToF;
        if (t>0) ToFSmearing(t);
        ToFDet->AddHit(x,y,t);
        if (!arm_reco) continue;
        int cellid = ToFDet->findCellId(x,y);
        if (cellid==0) continue;
        // find x,y of the center of the cell
        double xc=0;
        double yc=0;
        if (ToFDet->get_CellCenter(cellid,xc,yc)) arm_reco->AddHitToF(cellid,t,xc,yc);
        else arm_reco->AddHitToF(cellid,t,0.,0.);
    }
}
void PPSSim::GenSingleParticle(double& t, double& xi, double& phi)
{
    phi = gRandom3->Uniform(fPhiMin,fPhiMax);
    if (fGenMode=="linear"||fGenMode=="uniform") {
        xi = gRandom3->Uniform(xi_min,xi_max);
        t  = gRandom3->Uniform(t_min,t_max);
    }
    else if (fGenMode=="log"){
        if (t_min==0) t_min = 1e-6; // limit t to 1 MeV
        xi = pow(10,gRandom3->Uniform(log10(xi_min),log10(xi_max)));
        t  = pow(10,gRandom3->Uniform(log10(t_min),log10(t_max)));
    }
    double min_t = Minimum_t(xi);
    if (t<min_t) t = min_t;
}
void PPSSim::GenCentralMass(double& t1, double& t2, double& xi1, double& xi2, double& phi1, double& phi2)
{
    if (fCentralMassErr>0) {
        double m_h = gRandom3->Gaus(fCentralMass,fCentralMassErr);
        while(1) {
            xi1 = gRandom3->Uniform(xi_min,xi_max);
            xi2 = gRandom3->Uniform(xi_min,xi_max);
            double mh_2 = sqrt(xi1*xi2)*2.*fBeamEnergy;
            if ((fabs(m_h-mh_2)<fCentralMassErr) &&
                    (isPhysical(xi1)&&isPhysical(xi2))) break;// check validity of kinematic region
        }
    } else {
        while(1) {
            xi1 = gRandom3->Uniform(xi_min,xi_max);
            xi2 = pow(0.5*fCentralMass/fBeamEnergy,2)/xi1;
            if (isPhysical(xi1)&&isPhysical(xi2)) break;
        }
    }

    phi1 = gRandom3->Uniform(fPhiMin,fPhiMax);
    phi2 = gRandom3->Uniform(fPhiMin,fPhiMax);
    t1   = gRandom3->Uniform(Minimum_t(xi1),t_max);
    t2   = gRandom3->Uniform(Minimum_t(xi2),t_max);
}
void PPSSim::LorentzBoost(TLorentzVector& p_out, const string& frame)
{
    // Use a matrix
    double microrad = 1.e-6;
    TMatrixD tmpboost(4,4);
    double alpha_ = 0.;
    double phi_  = fCrossingAngle*microrad;
    if (p_out.Pz()<0) phi_*=-1;
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
    p4(0,0) = p_out.E();
    p4(1,0) = p_out.Px();
    p4(2,0) = p_out.Py();
    p4(3,0) = p_out.Pz();
    TMatrixD p4lab(4,1);
    p4lab = tmpboost * p4;
    p_out.SetPxPyPzE(p4lab(1,0),p4lab(2,0),p4lab(3,0),p4lab(0,0));
}
void PPSSim::ApplyBeamSmearing(TLorentzVector& p_out)
{
    double microrad = 1.e-6;
    double theta = p_out.Theta(); if (p_out.Pz()<0) theta=TMath::Pi()-theta;
    double dtheta_x = (double)(fSmearAngle)?gRandom3->Gaus(0.,fBeamAngleRMS):0;
    double dtheta_y = (double)(fSmearAngle)?gRandom3->Gaus(0.,fBeamAngleRMS):0;
    double denergy  = (double)(fSmearEnergy)?gRandom3->Gaus(0.,fBeamEnergyRMS):0.;

    double px = p_out.P()*sin(theta+dtheta_x*microrad)*cos(p_out.Phi());
    double py = p_out.P()*sin(theta+dtheta_y*microrad)*sin(p_out.Phi());
    double pz = p_out.P()*(cos(theta)+denergy);

    if (p_out.Pz()<0) pz*=-1;

    double e  = sqrt(px*px+py*py+pz*pz+ProtonMassSQ);
    p_out.SetPxPyPzE(px,py,pz,e);
}
void PPSSim::CrossingAngleCorrection(TLorentzVector& p_out)
{
    double microrad = 1.e-6;
    double theta = p_out.Theta(); if (p_out.Pz()<0) theta=TMath::Pi()-theta;
    double dtheta_x = (double)((fSmearAngle)?gRandom3->Gaus(0.,fBeamAngleRMS):0+
            (p_out.Pz()>0)?fCrossingAngle:-fCrossingAngle);
    double dtheta_y = (double)(fSmearAngle)?gRandom3->Gaus(0.,fBeamAngleRMS):0;
    double denergy  = (double)(fSmearEnergy)?gRandom3->Gaus(0.,fBeamEnergyRMS):0.;

    double px = p_out.P()*(theta*cos(p_out.Phi())+dtheta_x*microrad);
    double py = p_out.P()*(theta*sin(p_out.Phi())+dtheta_y*microrad);
    double pz = p_out.P()*(cos(theta)+denergy);

    if (p_out.Pz()<0) pz*=-1;

    double e  = sqrt(px*px+py*py+pz*pz+ProtonMassSQ);
    p_out.SetPxPyPzE(px,py,pz,e);
}
void PPSSim::CrossingAngleCorrection(H_BeamParticle& p_out, const int Direction)
{
    // 
    // Remember: Hector  used X,Z inverted in ref. to CMS, but pz is always positive
    double partP = sqrt(pow(p_out.getE(),2)-ProtonMassSQ);
    double px = -Direction*partP*p_out.getTX()*urad;
    double py = partP*p_out.getTY()*urad;
    double pz = Direction*partP*cos(sqrt(pow(p_out.getTX(),2)+pow(p_out.getTY(),2))*urad);
    TLorentzVector p(px,py,pz,p_out.getE());
    CrossingAngleCorrection(p);
    p_out.set4Momentum(-Direction*p.Px(),p.Py(),Direction*p.Pz(),p.E());
    return;
}
TLorentzVector PPSSim::shoot(const double& t,const double& xi, const double& phi,const int Direction)
{
    long double energy=fBeamEnergy*(1.-xi);
    long double partP = sqrt((long double)(energy*energy-ProtonMassSQ));
    long double theta = acos((-t/2. - ProtonMassSQ + fBeamEnergy*energy)/(fBeamMomentum*partP)); // this angle is the scattering one
    long double px = partP*sin(theta)*cos((long double)phi)*Direction;
    long double py = partP*sin(theta)*sin((long double)phi);
    long double pz = partP*cos(theta)*Direction;
    return TLorentzVector((double)px,(double)py,(double)pz,(double)energy);
}
void PPSSim::Propagate(H_BeamParticle* pbeam,int Direction) {
    PPSSimData* arm = NULL;
    H_BeamLine* beamline = NULL;
    double startZ = -pbeam->getS(); // in the CMS ref. frame, in meters
    double tcl4pos = 0;
    double tcl5pos = 0;

    if (Direction>0) {arm = &(fSim->ArmF);beamline=beamlineF;tcl4pos=fTCL4Position2;tcl5pos=fTCL5Position2;}
    if (Direction<0) {arm = &(fSim->ArmB);beamline=beamlineB;tcl4pos=fTCL4Position1;tcl5pos=fTCL5Position1;}
    // Propagate until TCL4 and 5
    if (tcl4pos>0) {
        double beampos = (Direction<0)?fBeam1PosAtTCL4.first:fBeam2PosAtTCL4.first;
        double beamrms = (Direction<0)?fBeam1RMSAtTCL4.first:fBeam2RMSAtTCL4.first;
        pbeam->propagate(tcl4pos);
        double xpos = -pbeam->getX()*um_to_mm;
        arm->get_Track().set_XatTCL4(fabs(xpos-beampos)/beamrms);
    }
    if (tcl5pos>0) {
        double beampos = (Direction<0)?fBeam1PosAtTCL5.first:fBeam2PosAtTCL5.first;
        double beamrms = (Direction<0)?fBeam1RMSAtTCL5.first:fBeam2RMSAtTCL5.first;
        pbeam->propagate(tcl5pos);double xpos=-pbeam->getX()*um_to_mm;
        arm->get_Track().set_XatTCL5(fabs(xpos-beampos)/beamrms);
    }
    //
    pbeam->propagate(fTrackerZPosition);

    int stopped = (pbeam->stopped(beamline) && pbeam->getStoppingElement()->getS()<fTrackerZPosition)?1:0;
    if (stopped) return;

    // uses mm for X,Y and m for Z in the PPS station
    double x1 = -pbeam->getX()*um_to_mm;
    double y1 = pbeam->getY()*um_to_mm;
    //
    arm->get_Track().set_HitDet1(x1,y1);
    arm->AddHitTrk1(x1,y1);

    pbeam->propagate(fTrackerZPosition+fTrackerLength);

    stopped=(pbeam->stopped(beamline) && pbeam->getStoppingElement()->getS()<fTrackerZPosition+fTrackerLength)?1:0;
    if (stopped) return;

    double x2 = -pbeam->getX()*um_to_mm;
    double y2 = pbeam->getY()*um_to_mm;
    arm->get_Track().set_HitDet2(x2,y2);
    arm->AddHitTrk2(x2,y2);

    // Propagate until Time detector
    pbeam->propagate(fToFZPosition);
    double xt = -pbeam->getX()*um_to_mm;
    double yt = pbeam->getY()*um_to_mm;
    stopped=(pbeam->stopped(beamline) && pbeam->getStoppingElement()->getS()<fToFZPosition)?1:0;
    if (stopped) return;
    //
    double tof = (fToFZPosition-Direction*startZ)/c_light_ns;
    arm->get_Track().set_HitToF(0,tof,xt,yt);
    arm->AddHitToF(0,tof,xt,yt);
}
void PPSSim::SmearVertexPosition(double& vtxX,double& vtxY, double& vtxZ)
{
    vtxX = fVtxMeanX;
    vtxY = fVtxMeanY;
    vtxZ = fVtxMeanZ;
    if (fSmearVertex) {
        vtxX=gRandom3->Gaus(fVtxMeanX,fVtxSigmaX); // in cm
        vtxY=gRandom3->Gaus(fVtxMeanY,fVtxSigmaY); // in cm
        vtxZ=gRandom3->Gaus(fVtxMeanZ,fVtxSigmaZ); // in cm
    }
}
void PPSSim::HitSmearing(double& x, double& y, double& z)
{
    //
    // X,Y in PPS is in mm, Z in m, but the hit resolution is given in mm. Then, to avoid smearing
    // into a too narow distribution, converts to mm and then, converts back the z coordinats to m
    //
    if (fSmearHit) {
        x = gRandom3->Gaus(x,fHitSigmaX);
        y = gRandom3->Gaus(y,fHitSigmaY);
        z = gRandom3->Gaus(z*m_to_mm,fHitSigmaZ)*mm_to_m;
    }
    return;
}
double PPSSim::Minimum_t(const double& xi)
{
    double partE = fBeamEnergy*(1.- xi);
    double partP = sqrt(partE*partE-ProtonMassSQ);
    return -2.*(fBeamMomentum*partP-fBeamEnergy*partE+ProtonMassSQ);
}

void PPSSim::PrintParameters()
{
    edm::LogWarning("debug") << "Running with:\n"
        << "TrackerPosition    = " <<  fTrackerZPosition << "\n"
        << "TrackerLength      = " <<  fTrackerLength << "\n"
        << "TrackerZPosition   = " <<  fTrackerZPosition << "\n"
        << "TrackerLength      = " <<  fTrackerLength << "\n"
        << "ToFZPosition       = " <<  fToFZPosition << "\n"
        << "BeamLineLength     = " <<  fBeamLineLength << "\n"
        << "SmearVertex        = " <<  fSmearVertex << "\n"
        << "VtxMeanX           = " <<  fVtxMeanX << "\n"
        << "VtxMeanY           = " <<  fVtxMeanY << "\n"
        << "VtxMeanZ           = " <<  fVtxMeanZ << "\n"
        << "VtxSigmaX          = " <<  fVtxSigmaX << "\n"
        << "VtxSigmaY          = " <<  fVtxSigmaY << "\n"
        << "VtxSigmaZ          = " <<  fVtxSigmaZ << "\n"
        << "VtxMeanZ           = " <<  fVtxMeanZ << "\n"
        << "VtxSigmaX          = " <<  fVtxSigmaX << "\n"
        << "VtxSigmaY          = " <<  fVtxSigmaY << "\n"
        << "VtxSigmaZ          = " <<  fVtxSigmaZ << "\n"
        << "SmearHit           = " <<  fSmearHit << "\n"
        << "HitSigmaX          = " <<  fHitSigmaX << "\n"
        << "HitSigmaY          = " <<  fHitSigmaY << "\n"
        << "HitSigmaZ          = " <<  fHitSigmaZ << "\n"
        << "TimeSigma          = " <<  fTimeSigma << "\n"
        << "SimBeam            = " <<  fSimBeam   << "\n"
        << "PhiMin             = " <<  fPhiMin    << "\n"
        << "PhiMax             = " <<  fPhiMax    << "\n"
        << "EtaMin             = " <<  fEtaMin    << "\n"
        << "MomentumMin        = " <<  fMomentumMin << "\n"
        << "CrossAngleCorr     = " <<  fCrossAngleCorr << "\n"
        << "KickersOFF         = " <<  fKickersOFF << "\n"
        << "Central Mass       = " <<  fCentralMass << " +- " << fCentralMassErr << "\n"
        << "TrackImpactParameterCut = " << fTrackImpactParameterCut << "\n"
        << "MinThetaXatDet1    = " <<fMinThetaXatDet1 << "\n"
        << "MaxThetaXatDet1    = " <<fMaxThetaXatDet1 << "\n"
        << "MinThetaYatDet1    = " <<fMinThetaYatDet1 << "\n"
        << "MaxThetaYatDet1    = " <<fMaxThetaYatDet1 << "\n";
}

TH2F* PPSSim::GenBeamProfile(const double& z)
{
    float beamp_w = 20.0;//beam pipe width
    int  direction=int(z/fabs(z));
    int   nbins = 500;
    TH2F* beamprofile = (TH2F*)gDirectory->FindObject("BeamProfile");
    if (beamprofile) delete beamprofile;
    beamprofile = new TH2F("BeamProfile",Form("Beam Profile at z=%3.2f; X (mm); Y (mm)",z),nbins,-beamp_w,beamp_w,nbins,-beamp_w,beamp_w);
    for(int n=0;n<100000;n++) {
        H_BeamParticle p1; // beam particle generated in the ref. frame of Hector/LHC
        p1.smearPos();
        if (fCrossAngleCorr) CrossingAngleCorrection(p1,direction); // apply the crossing angle correction (boost)
        else { p1.smearAng();p1.smearE(); }   // if no correnction for crossing angle, apply just the smearing
        //
        // set the vertex, given in the CMS ref. frame (x-> -x; z-> -z)
        p1.setPosition(
                p1.getX(),p1.getY(),
                p1.getTX(),p1.getTY(),
                p1.getS());
        //
        if (z<0) p1.computePath(beamlineB);
        else     p1.computePath(beamlineF);
        p1.propagate(fabs(z));
        beamprofile->Fill(-p1.getX()*um_to_mm,p1.getY()*um_to_mm);
    }
    return beamprofile;
}
/*
   void PPSSim::set_Strengths()
   {
// The offset is always positive, since both outgoind beamline is in the outside of the reference orbit
//
std::map<std::string,double> strengths;
std::map<std::string,double> betaX;
std::map<std::string,double> betaY;
std::map<std::string,double> DX;
std::map<std::string,double> DY;
ifstream in("beta90m.par");
if (!in) exit(2);
std::string opt_elm;
double str;
double betax, betay, dx, dy;
while(!in.eof()) {
in >> opt_elm >> betax >> betay >> dx >> dy >> str;
strengths[opt_elm] = str;
betaX[opt_elm] = betax;
betaY[opt_elm] = betay;
DX[opt_elm] = dx;
DY[opt_elm] = dy;
}
for(int i=0;i<beamlineF->getNumberOfElements();i++){
H_OpticalElement* optE = beamlineF->getElement(i);
std::string type = optE->getTypeString();
if (type.find("Dipole")<type.length()||type.find("Quadrupole")<type.length()) {
std::string name = optE->getName();
optE->setK(strengths[name]);
optE->setBetaX(betaX[name]);
optE->setBetaY(betaY[name]);
optE->setDX(DX[name]);
optE->setDY(DY[name]);
}
}
for(int i=0;i<beamlineB->getNumberOfElements();i++){
H_OpticalElement* optE = beamlineB->getElement(i);
std::string type = optE->getTypeString();
if (type.find("Dipole")<type.length()||type.find("Quadrupole")<type.length()) {
std::string name = optE->getName();
optE->setK(strengths[name]);
optE->setBetaX(betaX[name]);
optE->setBetaY(betaY[name]);
optE->setDX(DX[name]);
optE->setDY(DY[name]);
}
}
}
*/
