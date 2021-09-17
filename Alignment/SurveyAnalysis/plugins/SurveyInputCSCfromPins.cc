#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Alignment/MuonAlignment/interface/AlignableCSCChamber.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCStation.h"

#include <iostream>
#include <fstream>
#include "TFile.h"
#include "TTree.h"
#include "TRandom3.h"

#include "SurveyInputCSCfromPins.h"

#define SQR(x) ((x) * (x))

SurveyInputCSCfromPins::SurveyInputCSCfromPins(const edm::ParameterSet &cfg)
    : DTGeoToken_(esConsumes()),
      CSCGeoToken_(esConsumes()),
      GEMGeoToken_(esConsumes()),
      m_pinPositions(cfg.getParameter<std::string>("pinPositions")),
      m_rootFile(cfg.getParameter<std::string>("rootFile")),
      m_verbose(cfg.getParameter<bool>("verbose")),
      m_errorX(cfg.getParameter<double>("errorX")),
      m_errorY(cfg.getParameter<double>("errorY")),
      m_errorZ(cfg.getParameter<double>("errorZ")),
      m_missingErrorTranslation(cfg.getParameter<double>("missingErrorTranslation")),
      m_missingErrorAngle(cfg.getParameter<double>("missingErrorAngle")),
      m_stationErrorX(cfg.getParameter<double>("stationErrorX")),
      m_stationErrorY(cfg.getParameter<double>("stationErrorY")),
      m_stationErrorZ(cfg.getParameter<double>("stationErrorZ")),
      m_stationErrorPhiX(cfg.getParameter<double>("stationErrorPhiX")),
      m_stationErrorPhiY(cfg.getParameter<double>("stationErrorPhiY")),
      m_stationErrorPhiZ(cfg.getParameter<double>("stationErrorPhiZ")) {}

void SurveyInputCSCfromPins::orient(align::LocalVector LC1,
                                    align::LocalVector LC2,
                                    double a,
                                    double b,
                                    double &T,
                                    double &dx,
                                    double &dy,
                                    double &dz,
                                    double &PhX,
                                    double &PhZ) {
  double cosPhX, sinPhX, cosPhZ, sinPhZ;

  LocalPoint LP1(LC1.x(), LC1.y() + a, LC1.z() + b);
  LocalPoint LP2(LC2.x(), LC2.y() - a, LC2.z() + b);

  LocalPoint P((LP1.x() - LP2.x()) / (2. * a), (LP1.y() - LP2.y()) / (2. * a), (LP1.z() - LP2.z()) / (2. * a));
  LocalPoint Pp((LP1.x() + LP2.x()) / (2.), (LP1.y() + LP2.y()) / (2.), (LP1.z() + LP2.z()) / (2.));

  T = P.mag();

  sinPhX = P.z() / T;
  cosPhX = sqrt(1 - SQR(sinPhX));
  cosPhZ = P.y() / (T * cosPhX);
  sinPhZ = -P.x() / (T * cosPhZ);

  PhX = atan2(sinPhX, cosPhX);

  PhZ = atan2(sinPhZ, cosPhZ);

  dx = Pp.x() - sinPhZ * sinPhX * b;
  dy = Pp.y() + cosPhZ * sinPhX * b;
  dz = Pp.z() - cosPhX * b;
}

void SurveyInputCSCfromPins::errors(double a,
                                    double b,
                                    bool missing1,
                                    bool missing2,
                                    double &dx_dx,
                                    double &dy_dy,
                                    double &dz_dz,
                                    double &phix_phix,
                                    double &phiz_phiz,
                                    double &dy_phix) {
  dx_dx = 0.;
  dy_dy = 0.;
  dz_dz = 0.;
  phix_phix = 0.;
  phiz_phiz = 0.;
  dy_phix = 0.;

  const double trials = 10000.;  // two significant digits

  for (int i = 0; i < trials; i++) {
    LocalVector LC1, LC2;
    if (missing1) {
      LC1 = LocalVector(gRandom->Gaus(0., m_missingErrorTranslation),
                        gRandom->Gaus(0., m_missingErrorTranslation),
                        gRandom->Gaus(0., m_missingErrorTranslation));
    } else {
      LC1 = LocalVector(gRandom->Gaus(0., m_errorX), gRandom->Gaus(0., m_errorY), gRandom->Gaus(0., m_errorZ));
    }

    if (missing2) {
      LC2 = LocalVector(gRandom->Gaus(0., m_missingErrorTranslation),
                        gRandom->Gaus(0., m_missingErrorTranslation),
                        gRandom->Gaus(0., m_missingErrorTranslation));
    } else {
      LC2 = LocalVector(gRandom->Gaus(0., m_errorX), gRandom->Gaus(0., m_errorY), gRandom->Gaus(0., m_errorZ));
    }

    double dx, dy, dz, PhX, PhZ, T;
    orient(LC1, LC2, a, b, T, dx, dy, dz, PhX, PhZ);

    dx_dx += dx * dx;
    dy_dy += dy * dy;
    dz_dz += dz * dz;
    phix_phix += PhX * PhX;
    phiz_phiz += PhZ * PhZ;
    dy_phix += dy * PhX;  // the only non-zero off-diagonal element
  }

  dx_dx /= trials;
  dy_dy /= trials;
  dz_dz /= trials;
  phix_phix /= trials;
  phiz_phiz /= trials;
  dy_phix /= trials;
}

void SurveyInputCSCfromPins::analyze(const edm::Event &, const edm::EventSetup &iSetup) {
  if (theFirstEvent) {
    edm::LogInfo("SurveyInputCSCfromPins") << "***************ENTERING INITIALIZATION******************"
                                           << "  \n";

    std::ifstream in;
    in.open(m_pinPositions.c_str());

    Double_t x1, y1, z1, x2, y2, z2, a, b, tot = 0.0, maxErr = 0.0, h, s1, dx, dy, dz, PhX, PhZ, T;

    int ID1, ID2, ID3, ID4, ID5, i = 1, ii = 0;

    TFile *file1 = new TFile(m_rootFile.c_str(), "recreate");
    TTree *tree1 = new TTree("tree1", "alignment pins");

    if (m_verbose) {
      tree1->Branch("displacement_x_pin1_cm", &x1, "x1/D");
      tree1->Branch("displacement_y_pin1_cm", &y1, "y1/D");
      tree1->Branch("displacement_z_pin1_cm", &z1, "z1/D");
      tree1->Branch("displacement_x_pin2_cm", &x2, "x2/D");
      tree1->Branch("displacement_y_pin2_cm", &y2, "y2/D");
      tree1->Branch("displacement_z_pin2_cm", &z2, "z2/D");
      tree1->Branch("error_vector_length_cm", &h, "h/D");
      tree1->Branch("stretch_diff_cm", &s1, "s1/D");
      tree1->Branch("stretch_factor", &T, "T/D");
      tree1->Branch("chamber_displacement_x_cm", &dx, "dx/D");
      tree1->Branch("chamber_displacement_y_cm", &dy, "dy/D");
      tree1->Branch("chamber_displacement_z_cm", &dz, "dz/D");
      tree1->Branch("chamber_rotation_x_rad", &PhX, "PhX/D");
      tree1->Branch("chamber_rotation_z_rad", &PhZ, "PhZ/D");
    }

    const DTGeometry *dtGeometry = &iSetup.getData(DTGeoToken_);
    const CSCGeometry *cscGeometry = &iSetup.getData(CSCGeoToken_);
    const GEMGeometry *gemGeometry = &iSetup.getData(GEMGeoToken_);

    AlignableMuon *theAlignableMuon = new AlignableMuon(dtGeometry, cscGeometry, gemGeometry);
    AlignableNavigator *theAlignableNavigator = new AlignableNavigator(theAlignableMuon);

    const auto &theEndcaps = theAlignableMuon->CSCEndcaps();

    for (const auto &aliiter : theEndcaps) {
      addComponent(aliiter);
    }

    while (in.good()) {
      bool missing1 = false;
      bool missing2 = false;

      in >> ID1 >> ID2 >> ID3 >> ID4 >> ID5 >> x1 >> y1 >> z1 >> x2 >> y2 >> z2 >> a >> b;

      if (fabs(x1 - 1000.) < 1e5 && fabs(y1 - 1000.) < 1e5 && fabs(z1 - 1000.) < 1e5) {
        missing1 = true;
        x1 = x2;
        y1 = y2;
        z1 = z2;
      }

      if (fabs(x2 - 1000.) < 1e5 && fabs(y2 - 1000.) < 1e5 && fabs(z2 - 1000.) < 1e5) {
        missing2 = true;
        x2 = x1;
        y2 = y1;
        z2 = z1;
      }

      x1 = x1 / 10.0;
      y1 = y1 / 10.0;
      z1 = z1 / 10.0;
      x2 = x2 / 10.0;
      y2 = y2 / 10.0;
      z2 = z2 / 10.0;

      CSCDetId layerID(ID1, ID2, ID3, ID4, 1);

      // We cannot use chamber ID (when ID5=0), because AlignableNavigator gives the error (aliDet and aliDetUnit are undefined for chambers)

      Alignable *theAlignable1 = theAlignableNavigator->alignableFromDetId(layerID);
      Alignable *chamberAli = theAlignable1->mother();

      LocalVector LC1 = chamberAli->surface().toLocal(GlobalVector(x1, y1, z1));
      LocalVector LC2 = chamberAli->surface().toLocal(GlobalVector(x2, y2, z2));

      orient(LC1, LC2, a, b, T, dx, dy, dz, PhX, PhZ);

      GlobalPoint PG1 = chamberAli->surface().toGlobal(LocalPoint(LC1.x(), LC1.y() + a, LC1.z() + b));
      chamberAli->surface().toGlobal(LocalPoint(LC2.x(), LC2.y() - a, LC2.z() + b));

      LocalVector lvector(dx, dy, dz);
      GlobalVector gvector = (chamberAli->surface()).toGlobal(lvector);
      chamberAli->move(gvector);

      chamberAli->rotateAroundLocalX(PhX);
      chamberAli->rotateAroundLocalZ(PhZ);

      double dx_dx, dy_dy, dz_dz, phix_phix, phiz_phiz, dy_phix;
      errors(a, b, missing1, missing2, dx_dx, dy_dy, dz_dz, phix_phix, phiz_phiz, dy_phix);
      align::ErrorMatrix error = ROOT::Math::SMatrixIdentity();
      error(0, 0) = dx_dx;
      error(1, 1) = dy_dy;
      error(2, 2) = dz_dz;
      error(3, 3) = phix_phix;
      error(4, 4) = m_missingErrorAngle;
      error(5, 5) = phiz_phiz;
      error(1, 3) = dy_phix;
      error(3, 1) = dy_phix;  // just in case

      chamberAli->setSurvey(new SurveyDet(chamberAli->surface(), error));

      if (m_verbose) {
        edm::LogInfo("SurveyInputCSCfromPins") << " survey information = " << chamberAli->survey() << "  \n";

        LocalPoint LP1n = chamberAli->surface().toLocal(PG1);

        LocalPoint hiP(LP1n.x(), LP1n.y() - a * T, LP1n.z() - b);

        h = hiP.mag();
        s1 = LP1n.y() - a;

        if (h > maxErr) {
          maxErr = h;

          ii = i;
        }

        edm::LogInfo("SurveyInputCSCfromPins")
            << "  \n"
            << "i " << i++ << " " << ID1 << " " << ID2 << " " << ID3 << " " << ID4 << " " << ID5 << " error  " << h
            << " \n"
            << " x1 " << x1 << " y1 " << y1 << " z1 " << z1 << " x2 " << x2 << " y2 " << y2 << " z2 " << z2 << " \n"
            << " error " << h << " S1 " << s1 << " \n"
            << " dx " << dx << " dy " << dy << " dz " << dz << " PhX " << PhX << " PhZ " << PhZ << " \n";

        tot += h;

        tree1->Fill();
      }
    }

    in.close();

    if (m_verbose) {
      file1->Write();
      edm::LogInfo("SurveyInputCSCfromPins")
          << " Total error  " << tot << "  Max Error " << maxErr << " N " << ii << "  \n";
    }

    file1->Close();

    for (const auto &aliiter : theEndcaps) {
      fillAllRecords(aliiter);
    }

    delete theAlignableMuon;
    delete theAlignableNavigator;

    edm::LogInfo("SurveyInputCSCfromPins") << "*************END INITIALIZATION***************"
                                           << "  \n";

    theFirstEvent = false;
  }
}

void SurveyInputCSCfromPins::fillAllRecords(Alignable *ali) {
  if (ali->survey() == nullptr) {
    AlignableCSCChamber *ali_AlignableCSCChamber = dynamic_cast<AlignableCSCChamber *>(ali);
    AlignableCSCStation *ali_AlignableCSCStation = dynamic_cast<AlignableCSCStation *>(ali);

    if (ali_AlignableCSCChamber != nullptr) {
      CSCDetId detid(ali->geomDetId());
      if (abs(detid.station()) == 1 && (detid.ring() == 1 || detid.ring() == 4)) {
        align::ErrorMatrix error = ROOT::Math::SMatrixIdentity();
        error(0, 0) = m_missingErrorTranslation;
        error(1, 1) = m_missingErrorTranslation;
        error(2, 2) = m_missingErrorTranslation;
        error(3, 3) = m_missingErrorAngle;
        error(4, 4) = m_missingErrorAngle;
        error(5, 5) = m_missingErrorAngle;

        ali->setSurvey(new SurveyDet(ali->surface(), error));
      } else {
        double a = 100.;
        double b = -9.4034;
        if (abs(detid.station()) == 1 && detid.ring() == 2)
          a = -90.260;
        else if (abs(detid.station()) == 1 && detid.ring() == 3)
          a = -85.205;
        else if (abs(detid.station()) == 2 && detid.ring() == 1)
          a = -97.855;
        else if (abs(detid.station()) == 2 && detid.ring() == 2)
          a = -164.555;
        else if (abs(detid.station()) == 3 && detid.ring() == 1)
          a = -87.870;
        else if (abs(detid.station()) == 3 && detid.ring() == 2)
          a = -164.555;
        else if (abs(detid.station()) == 4 && detid.ring() == 1)
          a = -77.890;

        double dx_dx, dy_dy, dz_dz, phix_phix, phiz_phiz, dy_phix;
        errors(a, b, true, true, dx_dx, dy_dy, dz_dz, phix_phix, phiz_phiz, dy_phix);
        align::ErrorMatrix error = ROOT::Math::SMatrixIdentity();
        error(0, 0) = dx_dx;
        error(1, 1) = dy_dy;
        error(2, 2) = dz_dz;
        error(3, 3) = phix_phix;
        error(4, 4) = m_missingErrorAngle;
        error(5, 5) = phiz_phiz;
        error(1, 3) = dy_phix;
        error(3, 1) = dy_phix;  // just in case

        ali->setSurvey(new SurveyDet(ali->surface(), error));
      }
    }

    else if (ali_AlignableCSCStation != nullptr) {
      align::ErrorMatrix error = ROOT::Math::SMatrixIdentity();
      error(0, 0) = m_stationErrorX;
      error(1, 1) = m_stationErrorY;
      error(2, 2) = m_stationErrorZ;
      error(3, 3) = m_stationErrorPhiX;
      error(4, 4) = m_stationErrorPhiY;
      error(5, 5) = m_stationErrorPhiZ;

      ali->setSurvey(new SurveyDet(ali->surface(), error));
    }

    else {
      align::ErrorMatrix error = ROOT::Math::SMatrixIdentity();
      ali->setSurvey(new SurveyDet(ali->surface(), error * (1e-10)));
    }
  }

  for (const auto &iter : ali->components()) {
    fillAllRecords(iter);
  }
}

// Plug in to framework
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SurveyInputCSCfromPins);
