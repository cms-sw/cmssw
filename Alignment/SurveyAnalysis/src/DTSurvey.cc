#include <fstream>

#include "Alignment/SurveyAnalysis/interface/DTSurveyChamber.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h" 
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTChamber.h" 

#include "Alignment/SurveyAnalysis/interface/DTSurvey.h"
#include <iostream>

DTSurvey::DTSurvey(const std::string& Wheel, const std::string& Chambers, int n)
  : chambers(nullptr) {
  
  nameOfWheelInfoFile = Wheel;
  nameOfChamberInfoFile = Chambers;
  id = n;   
  
  FillWheelInfo();
}


DTSurvey::~DTSurvey() {
  delete [] chambers;
}


void DTSurvey::CalculateChambers() {
  for(int stationCounter = 0; stationCounter < 4; stationCounter++) {
    for(int sectorCounter = 0; sectorCounter < 14; sectorCounter++) {
      if(chambers[stationCounter][sectorCounter]->getNumberPoints() > 2) {
        chambers[stationCounter][sectorCounter]->compute();
      }
    }
  }
}


const DTSurveyChamber * DTSurvey::getChamber(int station, int sector) const {return chambers[station][sector];}

void DTSurvey::ReadChambers(edm::ESHandle<DTGeometry> pDD) {
  
  //Create the chambers
  chambers = new DTSurveyChamber ** [4];
  for (int cont_stat = 0; cont_stat < 4; cont_stat++) {
    chambers[cont_stat] = new DTSurveyChamber * [14];
    for(int cont_sect = 0; cont_sect < 14; cont_sect++) {
      DTChamberId mId(id, cont_stat+1, cont_sect+1);
      chambers[cont_stat][cont_sect] = new DTSurveyChamber(id, cont_stat+1, cont_sect+1, mId.rawId());
    }
  }

  std::cout << nameOfChamberInfoFile << std::endl;
  std::ifstream file(nameOfChamberInfoFile.c_str());
  while(!file.eof()) {
    int code, station, sector;
    double x, y, z, rms, dx, dy, dz;
    file >> code >> x >> y >> z >> rms >> dx >> dy >> dz;
    if(file.eof()) break;
    x = x/10.0; y=y/10.0; z=z/10.0; dx=dx/10.0; dy=dy/10.0; dz=dz/10.0;rms=rms/10.0; 
    station = code/10000 - 1;
    sector = (code-(station+1)*10000)/100 - 1;
    //De momento vamos a actuar como si no hubiera otra forma de resolver esto
    TMatrixD r(3,1);
    r(0,0) = x; r(1,0) = y;r(2,0) = z+OffsetZ;
    TMatrixD disp(3,1);
    disp(0,0) = dx; disp(1,0) = dy; disp(2,0) = dz;
    TMatrixD rp = Rot*r-delta;
    disp = disp-r+rp;
    
    GlobalPoint rg(r(0,0), r(1,0), r(2,0));
    GlobalPoint rt(r(0,0)-disp(0,0), r(1,0)-disp(1,0), r(2,0)-disp(2,0));
    DTChamberId mId(id, station+1, sector+1);
    const DTChamber *mChamber = static_cast<const DTChamber *>(pDD->idToDet(mId));
    LocalPoint rl = mChamber->toLocal(rg);
    LocalPoint rtl = mChamber->toLocal(rt);
    TMatrixD rLocal(3,1);
    rLocal(0,0) = rl.x(); rLocal(1,0) = rl.y(); rLocal(2,0) = rl.z();
    TMatrixD rTeo(3,1);
    rTeo(0,0) = rtl.x(); rTeo(1,0) = rtl.y(); rTeo(2,0) = rtl.z();
    TMatrixD diff = rLocal-rTeo;
    TMatrixD errors(3,1);
    errors(0,0) = rms; errors(1,0) = rms; errors(2,0) = rms;
    chambers[station][sector]->addPoint(code, rLocal, diff, errors);
  }
  file.close();
}

/*
void DTSurvey::ToDB(MuonAlignment *myMuonAlignment) {

  for(int station = 0; station < 4; station++) {
    for(int sector = 0; sector < 14; sector++) {
      if(chambers[station][sector]->getNumberPoints() > 2) {
        std::vector<float> displacements;
        std::vector<float> rotations;
        displacements.push_back(chambers[station][sector]->getDeltaX());
        displacements.push_back(chambers[station][sector]->getDeltaY());
        displacements.push_back(chambers[station][sector]->getDeltaZ());
        rotations.push_back(chambers[station][sector]->getAlpha());
        rotations.push_back(chambers[station][sector]->getBeta());
        rotations.push_back(chambers[station][sector]->getGamma());
        DTChamberId mId(id, station+1, sector+1);
        myMuonAlignment->moveAlignableLocalCoord(mId, displacements, rotations);
      }
    }
  }
}
*/

void DTSurvey::FillWheelInfo() {

  std::ifstream wheeltowheel(nameOfWheelInfoFile.c_str());
  float zOffset, deltax, deltay, deltaz, alpha, beta, gamma;
  wheeltowheel >> zOffset >> deltax >> deltay >> deltaz >> alpha >> beta >> gamma;
  wheeltowheel.close();
  
  OffsetZ = zOffset;

  //Build displacement vector
  delta.ResizeTo(3,1);
  delta(0,0) = deltax/10.0;
  delta(1,0) = deltay/10.0;
  delta(2,0) = deltaz/10.0;
  
  //Build rotation matrix
  Rot.ResizeTo(3,3);
  TMatrixD alpha_m(3,3);
  TMatrixD beta_m(3,3);
  TMatrixD gamma_m(3,3);
  alpha_m.Zero();
  beta_m.Zero();
  gamma_m.Zero();
  for(int k = 0; k < 3; k++) {
    alpha_m(k,k) = 1.0;
    beta_m(k,k) = 1.0;
    gamma_m(k,k) = 1.0;
  }
  alpha /= 1000.0; //New scale: angles in radians
  beta /= 1000.0;
  gamma /= 1000.0;
  alpha_m(1,1) = cos(alpha);
  alpha_m(1,2) = sin(alpha);
  alpha_m(2,1) = -sin(alpha);
  alpha_m(2,2) = cos(alpha);
  beta_m(0,0) = cos(beta);
  beta_m(0,2) = -sin(beta);
  beta_m(2,0) = sin(beta);
  beta_m(2,2) = cos(beta);
  gamma_m(0,0) = cos(gamma);
  gamma_m(0,1) = sin(gamma);
  gamma_m(1,0) = -sin(gamma);
  gamma_m(1,1) = cos(gamma);
  Rot = alpha_m*beta_m*gamma_m;
}


std::ostream &operator<<(std::ostream & flux, const DTSurvey& obj) {

  for(int stationCounter = 0; stationCounter < 4; stationCounter++) {
    for(int sectorCounter = 0; sectorCounter < 14; sectorCounter++) {
      if(obj.getChamber(stationCounter,sectorCounter)->getNumberPoints() > 2) {
        const DTSurveyChamber *m_chamber = obj.getChamber(stationCounter, sectorCounter);
        flux << *m_chamber;
      }
    }
  }
  return flux;
}
