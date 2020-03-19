#include "Calibration/EcalTBTools/interface/TB06RecoH2.h"
#include "Calibration/EcalTBTools/interface/TB06TreeH2.h"
#include "TFile.h"
#include "TTree.h"

#include <iostream>

TB06TreeH2::TB06TreeH2(const std::string &fileName, const std::string &treeName)
    : m_file(nullptr), m_tree(nullptr), m_data(nullptr), m_dataSize(0) {
  TDirectory *dir = gDirectory;
  m_file = new TFile(fileName.c_str(), "RECREATE");
  m_file->cd();
  m_tree = new TTree(treeName.c_str(), "Analysis tree");
  m_tree->SetAutoSave(10000000);
  dir->cd();

  //  m_tree->cd () ;
  m_data = new TClonesArray(TB06RecoH2::Class(), 1);
  m_data->ExpandCreateFast(1);

  //  m_tree->Branch ("EGCO", &m_data, 64000, 2) ;
  m_tree->Branch("TB06O", &m_data, 64000, 2);
  m_tree->Print();
}

// -------------------------------------------------------------------

TB06TreeH2::~TB06TreeH2() {
  std::cout << "[TB06TreeH2][dtor] saving TTree " << m_tree->GetName() << " with " << m_tree->GetEntries() << " entries"
            << " on file: " << m_file->GetName() << std::endl;

  m_file->Write();
  delete m_tree;
  m_file->Close();
  delete m_file;
  delete m_data;
}

// -------------------------------------------------------------------

//! to be called at each loop
void TB06TreeH2::store(const int &tableIsMoving,
                       const int &run,
                       const int &event,
                       const int &S6adc,
                       const double &xhodo,
                       const double &yhodo,
                       const double &xslope,
                       const double &yslope,
                       const double &xquality,
                       const double &yquality,
                       const int &icMax,
                       const int &ietaMax,
                       const int &iphiMax,
                       const double &beamEnergy,
                       const double ampl[49],
                       const int &wcAXo,
                       const int &wcAYo,
                       const int &wcBXo,
                       const int &wcBYo,
                       const int &wcCXo,
                       const int &wcCYo,
                       const double &xwA,
                       const double &ywA,
                       const double &xwB,
                       const double &ywB,
                       const double &xwC,
                       const double &ywC,
                       const float &S1adc,
                       const float &S2adc,
                       const float &S3adc,
                       const float &S4adc,
                       const float &VM1,
                       const float &VM2,
                       const float &VM3,
                       const float &VM4,
                       const float &VM5,
                       const float &VM6,
                       const float &VM7,
                       const float &VM8,
                       const float &VMF,
                       const float &VMB,
                       const float &CK1,
                       const float &CK2,
                       const float &CK3,
                       const float &BH1,
                       const float &BH2,
                       const float &BH3,
                       const float &BH4,
                       const float &TOF1S,
                       const float &TOF2S,
                       const float &TOF1J,
                       const float &TOF2J) {
  m_data->Clear();
  TB06RecoH2 *entry = static_cast<TB06RecoH2 *>(m_data->AddrAt(0));

  entry->reset();
  //  reset (entry->myCalibrationMap) ;

  entry->tableIsMoving = tableIsMoving;
  entry->run = run;
  entry->event = event;
  entry->S6ADC = S6adc;

  entry->MEXTLindex = icMax;
  entry->MEXTLeta = ietaMax;
  entry->MEXTLphi = iphiMax;
  entry->MEXTLenergy = ampl[24];
  entry->beamEnergy = beamEnergy;

  for (int eta = 0; eta < 7; ++eta)
    for (int phi = 0; phi < 7; ++phi) {
      // FIXME capire l'orientamento di phi!
      // FIXME capire se eta, phi iniziano da 1 o da 0
      entry->localMap[eta][phi] = ampl[eta * 7 + phi];
    }

  //[Edgar] S1 uncleaned, uncalibrated energy
  entry->S1uncalib_ = ampl[24];

  //[Edgar] S25 uncleaned, uncalibrated energy
  for (int eta = 1; eta < 6; ++eta)
    for (int phi = 1; phi < 6; ++phi) {
      entry->S25uncalib_ += entry->localMap[eta][phi];
    }

  //[Edgar] S49 uncleaned, uncalibrated energy
  for (int eta = 0; eta < 7; ++eta)
    for (int phi = 0; phi < 7; ++phi) {
      entry->S49uncalib_ += entry->localMap[eta][phi];
    }

  //[Edgar] S9 uncleaned, uncalibrated energy
  for (int eta = 2; eta < 5; ++eta)
    for (int phi = 2; phi < 5; ++phi) {
      entry->S9uncalib_ += entry->localMap[eta][phi];
    }

  entry->xHodo = xhodo;
  entry->yHodo = yhodo;
  entry->xSlopeHodo = xslope;
  entry->ySlopeHodo = yslope;
  entry->xQualityHodo = xquality;
  entry->yQualityHodo = yquality;
  entry->wcAXo_ = wcAXo;
  entry->wcAYo_ = wcAYo;
  entry->wcBXo_ = wcBXo;
  entry->wcBYo_ = wcBYo;
  entry->wcCXo_ = wcCXo;
  entry->wcCYo_ = wcCYo;
  entry->xwA_ = xwA;
  entry->ywA_ = ywA;
  entry->xwB_ = xwB;
  entry->ywB_ = ywB;
  entry->xwC_ = xwC;
  entry->ywC_ = ywC;
  entry->S1adc_ = S1adc;
  entry->S2adc_ = S2adc;
  entry->S3adc_ = S3adc;
  entry->S4adc_ = S4adc;
  entry->VM1_ = VM1;
  entry->VM2_ = VM2;
  entry->VM3_ = VM3;
  entry->VM4_ = VM4;
  entry->VM5_ = VM5;
  entry->VM6_ = VM6;
  entry->VM7_ = VM7;
  entry->VM8_ = VM8;
  entry->VMF_ = VMF;
  entry->VMB_ = VMB;
  entry->CK1_ = CK1;
  entry->CK2_ = CK2;
  entry->CK3_ = CK3;
  entry->BH1_ = BH1;
  entry->BH2_ = BH2;
  entry->BH3_ = BH3;
  entry->BH4_ = BH4;
  entry->TOF1S_ = TOF1S;
  entry->TOF2S_ = TOF2S;
  entry->TOF1J_ = TOF1J;
  entry->TOF2J_ = TOF2J;

  entry->convFactor = 0.;

  /*
  // loop over the 5x5 see (1)
  for (int xtal=0 ; xtal<25 ; ++xtal)
    {
      int ieta = xtal/5 + 3 ;
      int iphi = xtal%5 + 8 ;
      entry->myCalibrationMap[ieta][iphi] = ampl[xtal] ;
    } // loop over the 5x5

  entry->electron_Tr_Pmag_ = beamEnergy ;

        entry->centralCrystalEta_ = ietaMax ;
        entry->centralCrystalPhi_ = iphiMax ;
        entry->centralCrystalEnergy_ = ampl[12] ;

  // this is a trick
  entry->electron_Tr_Peta_ = xhodo ;
  entry->electron_Tr_Pphi_ = yhodo ;
  */
  m_tree->Fill();
}

// -------------------------------------------------------------------

void TB06TreeH2::reset(float crystal[11][21]) {
  for (int eta = 0; eta < 11; ++eta) {
    for (int phi = 0; phi < 21; ++phi) {
      crystal[eta][phi] = -999.;
    }
  }
}

// -------------------------------------------------------------------

void TB06TreeH2::check() {
  TB06RecoH2 *entry = static_cast<TB06RecoH2 *>(m_data->AddrAt(0));

  std::cout << "[TB06TreeH2][check]reading . . . \n";
  std::cout << "[TB06TreeH2][check] entry->run: " << entry->run << "\n";
  std::cout << "[TB06TreeH2][check] entry->event: " << entry->event << "\n";
  std::cout << "[TB06TreeH2][check] entry->tableIsMoving: " << entry->tableIsMoving << "\n";
  std::cout << "[TB06TreeH2][check] entry->MEXTLeta: " << entry->MEXTLeta << "\n";
  std::cout << "[TB06TreeH2][check] entry->MEXTLphi: " << entry->MEXTLphi << "\n";
  std::cout << "[TB06TreeH2][check] entry->MEXTLenergy: " << entry->MEXTLenergy << "\n";

  for (int eta = 0; eta < 7; ++eta)
    for (int phi = 0; phi < 7; ++phi)
      std::cout << "[TB06TreeH2][check]   entry->localMap[" << eta << "][" << phi << "]: " << entry->localMap[eta][phi]
                << "\n";

  std::cout << "[TB06TreeH2][check] entry->xHodo: " << entry->xHodo << "\n";
  std::cout << "[TB06TreeH2][check] entry->yHodo: " << entry->yHodo << "\n";
  std::cout << "[TB06TreeH2][check] entry->xSlopeHodo: " << entry->xSlopeHodo << "\n";
  std::cout << "[TB06TreeH2][check] entry->ySlopeHodo: " << entry->ySlopeHodo << "\n";
  std::cout << "[TB06TreeH2][check] entry->xQualityHodo: " << entry->xQualityHodo << "\n";
  std::cout << "[TB06TreeH2][check] entry->yQualityHodo: " << entry->yQualityHodo << "\n";
  std::cout << "[TB06TreeH2][check] entry->convFactor: " << entry->convFactor << "\n";

  /* to be implemented with the right variables
  std::cout << "[TB06TreeH2][check] ------------------------" << std::endl ;
  std::cout << "[TB06TreeH2][check] " << entry->variable_name << std::endl ;
  */
}

/* (1) to fill the 25 crystals vector

   for (UInt_t icry=0 ; icry<25 ; ++icry)
     {
       UInt_t row = icry / 5 ;
       Int_t column = icry % 5 ;
       try
           {
             EBDetId tempo (maxHitId.ieta()+column-2,
                            maxHitId.iphi()+row-2,
                            EBDetId::ETAPHIMODE) ;

             Xtals5x5.push_back (tempo) ;
             amplitude [icry] = hits->find (Xtals5x5[icry])->energy () ;

           }
       catch ( std::runtime_error &e )
           {
             std::cout << "Cannot construct 5x5 matrix around EBDetId "
                     << maxHitId << std::endl ;
             return ;
           }
     } // loop over the 5x5 matrix


*/
