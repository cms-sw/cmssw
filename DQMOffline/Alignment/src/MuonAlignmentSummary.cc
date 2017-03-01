/*
 *  DQM client for muon alignment summary
 *
 *  \author J. Fernandez - Univ. Oviedo <Javier.Fernandez@cern.ch>
 */

#include "DQMOffline/Alignment/interface/MuonAlignmentSummary.h"




MuonAlignmentSummary::MuonAlignmentSummary(const edm::ParameterSet& pSet) {

    parameters = pSet;

    meanPositionRange = parameters.getUntrackedParameter<double>("meanPositionRange");
    rmsPositionRange = parameters.getUntrackedParameter<double>("rmsPositionRange");
    meanAngleRange = parameters.getUntrackedParameter<double>("meanAngleRange");
    rmsAngleRange = parameters.getUntrackedParameter<double>("rmsAngleRange");

    doDT = parameters.getUntrackedParameter<bool>("doDT");
    doCSC = parameters.getUntrackedParameter<bool>("doCSC");

    MEFolderName = parameters.getParameter<std::string>("FolderName");
    topFolder << MEFolderName+"Alignment/Muon";

    if (!(doDT || doCSC) ) {
        edm::LogError("MuonAlignmentSummary") <<" Error!! At least one Muon subsystem (DT or CSC) must be monitorized!!" << std::endl;
        edm::LogError("MuonAlignmentSummary") <<" Please enable doDT or doCSC to True in your python cfg file!!!" << std::endl;
        exit(1);
    }

}

MuonAlignmentSummary::~MuonAlignmentSummary() {
}


void MuonAlignmentSummary::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {

    metname = "MuonAlignmentSummary";

    LogTrace(metname)<<"[MuonAlignmentSummary] Parameters initialization";


    if (doDT) {
        ibooker.setCurrentFolder(topFolder.str() + "/DT");
        hLocalPositionDT = ibooker.book2D("hLocalPositionDT",
                "Local DT position (cm) absolute MEAN residuals;Sector;;cm", 14, 1, 15, 40, 0, 40);

        hLocalAngleDT = ibooker.book2D("hLocalAngleDT",
                "Local DT angle (rad) absolute MEAN residuals;Sector;;rad", 14, 1, 15, 40, 0, 40);

        hLocalPositionRmsDT = ibooker.book2D("hLocalPositionRmsDT",
                "Local DT position (cm) RMS residuals;Sector;;cm", 14, 1, 15, 40, 0, 40);

        hLocalAngleRmsDT = ibooker.book2D("hLocalAngleRmsDT",
                "Local DT angle (rad) RMS residuals;Sector;;rad", 14, 1, 15, 40, 0, 40);

        hLocalXMeanDT = ibooker.book1D("hLocalXMeanDT",
                "Distribution of absolute MEAN Local X (cm) residuals for DT;<X> (cm);number of chambers", 100, 0, meanPositionRange);

        hLocalXRmsDT = ibooker.book1D("hLocalXRmsDT",
                "Distribution of RMS Local X (cm) residuals for DT;X RMS (cm);number of chambers", 100, 0, rmsPositionRange);

        hLocalYMeanDT = ibooker.book1D("hLocalYMeanDT",
                "Distribution of absolute MEAN Local Y (cm) residuals for DT;<Y> (cm);number of chambers", 100, 0, meanPositionRange);

        hLocalYRmsDT = ibooker.book1D("hLocalYRmsDT",
                "Distribution of RMS Local Y (cm) residuals for DT;Y RMS (cm);number of chambers", 100, 0, rmsPositionRange);

        hLocalPhiMeanDT = ibooker.book1D("hLocalPhiMeanDT",
                "Distribution of absolute MEAN #phi (rad) residuals for DT;<#phi>(rad);number of chambers", 100, 0, meanAngleRange);

        hLocalPhiRmsDT = ibooker.book1D("hLocalPhiRmsDT",
                "Distribution of RMS #phi (rad) residuals for DT;#phi RMS (rad);number of chambers", 100, 0, rmsAngleRange);

        hLocalThetaMeanDT = ibooker.book1D("hLocalThetaMeanDT",
                "Distribution of absolute MEAN #theta (rad) residuals for DT;<#theta>(rad);number of chambers", 100, 0, meanAngleRange);

        hLocalThetaRmsDT = ibooker.book1D("hLocalThetaRmsDT",
                "Distribution of RMS #theta (rad) residuals for DT;#theta RMS (rad);number of chambers", 100, 0, rmsAngleRange);

        hLocalPositionDT->Reset();
        hLocalAngleDT->Reset();
        hLocalPositionRmsDT->Reset();
        hLocalAngleRmsDT->Reset();
        hLocalXMeanDT->Reset();
        hLocalXRmsDT->Reset();
        hLocalYMeanDT->Reset();
        hLocalYRmsDT->Reset();
        hLocalPhiMeanDT->Reset();
        hLocalPhiRmsDT->Reset();
        hLocalThetaMeanDT->Reset();
        hLocalThetaRmsDT->Reset();

    }

    if (doCSC) {

        ibooker.setCurrentFolder(topFolder.str()+"/CSC");
        hLocalPositionCSC = ibooker.book2D("hLocalPositionCSC",
                "Local CSC position (cm) absolute MEAN residuals;Sector;;cm", 36, 1, 37, 40, 0, 40);

        hLocalAngleCSC = ibooker.book2D("hLocalAngleCSC",
                "Local CSC angle (rad) absolute MEAN residuals;Sector;;rad", 36, 1, 37, 40, 0, 40);

        hLocalPositionRmsCSC = ibooker.book2D("hLocalPositionRmsCSC",
                "Local CSC position (cm) RMS residuals;Sector;;cm", 36, 1, 37, 40, 0, 40);

        hLocalAngleRmsCSC = ibooker.book2D("hLocalAngleRmsCSC",
                "Local CSC angle (rad) RMS residuals;Sector;;rad", 36, 1, 37, 40, 0, 40);

        hLocalXMeanCSC = ibooker.book1D("hLocalXMeanCSC",
                "Distribution of absolute MEAN Local X (cm) residuals for CSC;<X> (cm);number of chambers", 100, 0, meanPositionRange);

        hLocalXRmsCSC = ibooker.book1D("hLocalXRmsCSC",
                "Distribution of RMS Local X (cm) residuals for CSC;X RMS (cm);number of chambers", 100, 0, rmsPositionRange);

        hLocalYMeanCSC = ibooker.book1D("hLocalYMeanCSC",
                "Distribution of absolute MEAN Local Y (cm) residuals for CSC;<Y> (cm);number of chambers", 100, 0, meanPositionRange);

        hLocalYRmsCSC = ibooker.book1D("hLocalYRmsCSC",
                "Distribution of RMS Local Y (cm) residuals for CSC;Y RMS (cm);number of chambers", 100, 0, rmsPositionRange);

        hLocalPhiMeanCSC = ibooker.book1D("hLocalPhiMeanCSC",
                "Distribution of absolute MEAN #phi (rad) residuals for CSC;<#phi>(rad);number of chambers", 100, 0, meanAngleRange);

        hLocalPhiRmsCSC = ibooker.book1D("hLocalPhiRmsCSC",
                "Distribution of RMS #phi (rad) residuals for CSC;#phi RMS (rad);number of chambers", 100, 0, rmsAngleRange);

        hLocalThetaMeanCSC = ibooker.book1D("hLocalThetaMeanCSC",
                "Distribution of absolute MEAN #theta (rad) residuals for CSC;<#theta>(rad);number of chambers", 100, 0, meanAngleRange);

        hLocalThetaRmsCSC = ibooker.book1D("hLocalThetaRmsCSC",
                "Distribution of RMS #theta (rad) residuals for CSC;#theta RMS (rad);number of chambers", 100, 0, rmsAngleRange);

        hLocalPositionCSC->Reset();
        hLocalAngleCSC->Reset();
        hLocalPositionRmsCSC->Reset();
        hLocalAngleRmsCSC->Reset();
        hLocalXMeanCSC->Reset();
        hLocalXRmsCSC->Reset();
        hLocalYMeanCSC->Reset();
        hLocalYRmsCSC->Reset();
        hLocalPhiMeanCSC->Reset();
        hLocalPhiRmsCSC->Reset();
        hLocalThetaMeanCSC->Reset();
        hLocalThetaRmsCSC->Reset();

    }

    LogTrace(metname)<<"[MuonAlignmentSummary] Saving the histos";

    char binLabel[15];

    for (int station = -4; station < 5; station++) {
        if (doDT) {
            if (station > 0) {

                for (int wheel = -2;wheel < 3; wheel++) {

                    for (int sector = 1; sector < 15; sector++) {

                        if (!((sector == 13 || sector == 14) && station != 4)) {

                            std::stringstream Wheel; Wheel<<wheel;
                            std::stringstream Station; Station<<station;
                            std::stringstream Sector; Sector<<sector;

                            std::string nameOfHistoLocalX="ResidualLocalX_W"+Wheel.str()+"MB"+Station.str()+"S"+Sector.str();
                            std::string nameOfHistoLocalPhi= "ResidualLocalPhi_W"+Wheel.str()+"MB"+Station.str()+"S"+Sector.str();
                            std::string nameOfHistoLocalTheta= "ResidualLocalTheta_W"+Wheel.str()+"MB"+Station.str()+"S"+Sector.str();
                            std::string nameOfHistoLocalY= "ResidualLocalY_W"+Wheel.str()+"MB"+Station.str()+"S"+Sector.str();

                            std::string path= topFolder.str()+
                                "/DT/Wheel"+Wheel.str()+
                                "/Station"+Station.str()+
                                "/Sector"+Sector.str()+"/";

                            std::string histo = path + nameOfHistoLocalX;

                            Int_t nstation=station - 1;
                            Int_t nwheel=wheel+2;
                            MonitorElement * localX = igetter.get(histo);
                            if (localX) {

                                Double_t Mean = localX->getMean();
                                Double_t Error = localX->getMeanError();

                                Int_t ybin = 1 + nwheel * 8 + nstation * 2;
                                hLocalPositionDT->setBinContent(sector, ybin, fabs(Mean));
                                snprintf(binLabel, sizeof(binLabel), "MB%d/%d_X", wheel, station);
                                hLocalPositionDT->setBinLabel(ybin, binLabel, 2);
                                hLocalPositionRmsDT->setBinContent(sector, ybin, Error);
                                hLocalPositionRmsDT->setBinLabel(ybin, binLabel, 2);

                                if (localX->getEntries() != 0){
                                    hLocalXMeanDT->Fill(fabs(Mean));
                                    hLocalXRmsDT->Fill(Error);}
                            }

                            histo = path+nameOfHistoLocalPhi;
                            MonitorElement * localPhi = igetter.get(histo);
                            if (localPhi) {

                                Double_t Mean = localPhi->getMean();
                                Double_t Error = localPhi->getMeanError();

                                Int_t ybin = 1 + nwheel * 8 + nstation * 2;
                                hLocalAngleDT->setBinContent(sector, ybin, fabs(Mean));
                                snprintf(binLabel, sizeof(binLabel), "MB%d/%d_#phi", wheel, station);
                                hLocalAngleDT->setBinLabel(ybin ,binLabel, 2);
                                hLocalAngleRmsDT->setBinContent(sector, ybin, Error);
                                hLocalAngleRmsDT->setBinLabel(ybin, binLabel, 2);

                                if (localPhi->getEntries() != 0) {
                                    hLocalPhiMeanDT->Fill(fabs(Mean));
                                    hLocalPhiRmsDT->Fill(Error);}
                            }

                            if (station != 4) {

                                histo=path+nameOfHistoLocalY;
                                MonitorElement * localY = igetter.get(histo);
                                if (localY) {

                                    Double_t Mean = localY->getMean();
                                    Double_t Error = localY->getMeanError();

                                    Int_t ybin = 2 + nwheel * 8 + nstation * 2;
                                    hLocalPositionDT->setBinContent(sector,ybin,fabs(Mean));
                                    snprintf(binLabel, sizeof(binLabel), "MB%d/%d_Y", wheel, station);
                                    hLocalPositionDT->setBinLabel(ybin, binLabel, 2);
                                    hLocalPositionRmsDT->setBinContent(sector, ybin, Error);
                                    hLocalPositionRmsDT->setBinLabel(ybin, binLabel, 2);
                                    if (localY->getEntries() != 0) {
                                        hLocalYMeanDT->Fill(fabs(Mean));
                                        hLocalYRmsDT->Fill(Error);}
                                }
                                histo = path+nameOfHistoLocalTheta;
                                MonitorElement * localTheta = igetter.get(histo);
                                if (localTheta) {
                                    Double_t Mean = localTheta->getMean();
                                    Double_t Error = localTheta->getMeanError();

                                    Int_t ybin = 2 + nwheel * 8 + nstation * 2;
                                    hLocalAngleDT->setBinContent(sector, ybin, fabs(Mean));
                                    snprintf(binLabel, sizeof(binLabel), "MB%d/%d_#theta", wheel, station);
                                    hLocalAngleDT->setBinLabel(ybin, binLabel, 2);
                                    hLocalAngleRmsDT->setBinContent(sector, ybin, Error);
                                    hLocalAngleRmsDT->setBinLabel(ybin, binLabel, 2);
                                    if (localTheta->getEntries() != 0) {
                                        hLocalThetaMeanDT->Fill(fabs(Mean));
                                        hLocalThetaRmsDT->Fill(Error);}
                                }
                            }// station != 4
                        } //avoid non existing sectors
                    } //sector 
                } //wheel
            } //station>0
        }// doDT

        if (doCSC){
            if (station != 0) {

                for (int ring = 1; ring < 5; ring++) {

                    for (int chamber = 1; chamber < 37; chamber++){

                        if ( !( ((abs(station)==2 || abs(station)==3 || abs(station)==4) && ring==1 && chamber>18) || 
                                ((abs(station)==2 || abs(station)==3 || abs(station)==4) && ring>2)) ) {
                            std::stringstream Ring; Ring<<ring;
                            std::stringstream Station; Station<<station;
                            std::stringstream Chamber; Chamber<<chamber;

                            std::string nameOfHistoLocalX="ResidualLocalX_ME"+Station.str()+"R"+Ring.str()+"C"+Chamber.str();
                            std::string nameOfHistoLocalPhi= "ResidualLocalPhi_ME"+Station.str()+"R"+Ring.str()+"C"+Chamber.str();
                            std::string nameOfHistoLocalTheta= "ResidualLocalTheta_ME"+Station.str()+"R"+Ring.str()+"C"+Chamber.str();
                            std::string nameOfHistoLocalY= "ResidualLocalY_ME"+Station.str()+"R"+Ring.str()+"C"+Chamber.str();

                            std::string path = topFolder.str()+
                                "/CSC/Station"+Station.str()+
                                "/Ring"+Ring.str()+
                                "/Chamber"+Chamber.str()+"/";

                            Int_t ybin = abs(station) * 2 + ring;
                            if (abs(station) == 1) ybin = ring;
                            if (station > 0) ybin = ybin + 10;
                            else ybin = 11 - ybin;
                            std::string histo = path + nameOfHistoLocalX;
                            MonitorElement * localX = igetter.get(histo);
                            if (localX) {

                                Double_t Mean=localX->getMean();
                                Double_t Error=localX->getMeanError();

                                Int_t ybin2= 2 * ybin - 1;
                                hLocalPositionCSC->setBinContent(chamber,ybin2,fabs(Mean));
                                snprintf(binLabel, sizeof(binLabel), "ME%d/%d_X", station, ring);
                                hLocalPositionCSC->setBinLabel(ybin2, binLabel, 2);
                                hLocalPositionRmsCSC->setBinContent(chamber, ybin2, Error);
                                hLocalPositionRmsCSC->setBinLabel(ybin2, binLabel, 2);
                                if (localX->getEntries() != 0) {
                                    hLocalXMeanCSC->Fill(fabs(Mean));
                                    hLocalXRmsCSC->Fill(Error);}
                            }
                            histo = path +	nameOfHistoLocalPhi;

                            MonitorElement * localPhi = igetter.get(histo);
                            if (localPhi) {

                                Double_t Mean=localPhi->getMean();
                                Double_t Error=localPhi->getMeanError();

                                Int_t ybin2 = 2 * ybin - 1;
                                hLocalAngleCSC->setBinContent(chamber, ybin2, fabs(Mean));
                                snprintf(binLabel, sizeof(binLabel), "ME%d/%d_#phi", station, ring);
                                hLocalAngleCSC->setBinLabel(ybin2, binLabel, 2);
                                hLocalAngleRmsCSC->setBinContent(chamber, ybin2, Error);
                                hLocalAngleRmsCSC->setBinLabel(ybin2, binLabel, 2);
                                if (localPhi->getEntries() != 0){
                                    hLocalPhiMeanCSC->Fill(fabs(Mean));
                                    hLocalPhiRmsCSC->Fill(Error);}
                            }
                            histo = path +	nameOfHistoLocalTheta;
                            MonitorElement * localTheta = igetter.get(histo);
                            if (localTheta) {

                                Double_t Mean = localTheta->getMean();
                                Double_t Error = localTheta->getMeanError();

                                Int_t ybin2 = 2 * ybin;
                                hLocalAngleCSC->setBinContent(chamber, ybin2, fabs(Mean));
                                snprintf(binLabel, sizeof(binLabel), "ME%d/%d_#theta", station, ring);
                                hLocalAngleCSC->setBinLabel(ybin2, binLabel, 2);
                                hLocalAngleRmsCSC->setBinContent(chamber, ybin2, Error);
                                hLocalAngleRmsCSC->setBinLabel(ybin2, binLabel, 2);
                                if (localTheta->getEntries() != 0) {
                                    hLocalThetaMeanCSC->Fill(fabs(Mean));
                                    hLocalThetaRmsCSC->Fill(Error);}

                            }
                            histo = path +	nameOfHistoLocalY;

                            MonitorElement * localY = igetter.get(histo);
                            if (localY) {

                                Double_t Mean=localY->getMean();
                                Double_t Error=localY->getMeanError();

                                Int_t ybin2 = 2 * ybin;
                                hLocalPositionCSC->setBinContent(chamber, ybin2, fabs(Mean));
                                snprintf(binLabel, sizeof(binLabel), "ME%d/%d_Y", station, ring);
                                hLocalPositionCSC->setBinLabel(ybin2, binLabel, 2);
                                hLocalPositionRmsCSC->setBinContent(chamber, ybin2, Error);
                                hLocalPositionRmsCSC->setBinLabel(ybin2, binLabel, 2);
                                if (localY->getEntries() != 0) {
                                    hLocalYMeanCSC->Fill(fabs(Mean));
                                    hLocalYRmsCSC->Fill(Error);}
                            }
                        } //avoid non existing rings
                    } //chamber
                } //ring
            } // station!=0
        }// doCSC
    } // loop on stations
}
