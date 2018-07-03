#include "Calibration/EcalTBTools/interface/TB06Tree.h"
#include "TFile.h" 
#include "TTree.h" 
#include "Calibration/EcalTBTools/interface/TB06Reco.h"

#include <iostream>


TB06Tree::TB06Tree (const std::string & fileName, 
		    const std::string & treeName):
  m_file (nullptr), m_tree (nullptr), m_data (nullptr), m_dataSize (0)  
{
  TDirectory *dir = gDirectory ;
  m_file = new TFile (fileName.c_str (),"RECREATE") ;
  m_file->cd () ;
  m_tree = new TTree (treeName.c_str(),"Analysis tree") ;
  m_tree->SetAutoSave (10000000) ;
  dir->cd () ;

  //  m_tree->cd () ;
  m_data = new TClonesArray (TB06Reco::Class (), 1) ;
  m_data->ExpandCreateFast (1) ;

  //  m_tree->Branch ("EGCO", &m_data, 64000, 2) ;
  m_tree->Branch ("TB06O", &m_data, 64000, 2) ;
  m_tree->Print () ;
 
}


// -------------------------------------------------------------------      

TB06Tree::~TB06Tree () 
{
  std::cout << "[TB06Tree][dtor] saving TTree " << m_tree->GetName ()
            << " with " << m_tree->GetEntries () << " entries"
            << " on file: " << m_file->GetName () << std::endl ;

  m_file->Write () ;
  delete m_tree ;
  m_file->Close () ;
  delete m_file ;
  delete m_data ;
}


// -------------------------------------------------------------------   

//! to be called at each loop
  void TB06Tree::store (const int & tableIsMoving,
                      const int & run, const int & event,
                      const int & S6adc ,
                      const double & xhodo, const double & yhodo, 
                      const double & xslope, const double & yslope, 
                      const double & xquality, const double & yquality,
                      const int & icMax,
                      const int & ietaMax, const int & iphiMax,
                      const double & beamEnergy, 
                      const double ampl[49]) 
{

  m_data->Clear () ;
  TB06Reco * entry = static_cast<TB06Reco*> (m_data->AddrAt (0)) ;

  entry->reset () ;
  //  reset (entry->myCalibrationMap) ;

  entry->tableIsMoving = tableIsMoving ;
  entry->run = run ;
  entry->event = event ;
  entry->S6ADC = S6adc ;

  entry->MEXTLindex = icMax ;
  entry->MEXTLeta = ietaMax ;
  entry->MEXTLphi = iphiMax ;
  entry->MEXTLenergy = ampl[24] ;
  entry->beamEnergy = beamEnergy ;

  for (int eta = 0 ; eta<7 ; ++eta)
    for (int phi = 0 ; phi<7 ; ++phi)
      {
	// FIXME capire l'orientamento di phi!
	// FIXME capire se eta, phi iniziano da 1 o da 0
        entry->localMap[eta][phi] = ampl[eta*7+phi] ;
      }

  entry->xHodo = xhodo ;
  entry->yHodo = yhodo ;
  entry->xSlopeHodo = xslope ;
  entry->ySlopeHodo = yslope ;
  entry->xQualityHodo = xquality ;
  entry->yQualityHodo = yquality ;

  entry->convFactor = 0. ;

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
  m_tree->Fill () ;
}


// -------------------------------------------------------------------


void TB06Tree::reset (float crystal[11][21])
{
  for (int eta =0 ; eta<11 ; ++eta)    
    {
      for (int phi =0 ; phi<21 ; ++phi) 
        {
          crystal[eta][phi] = -999. ;  
        }   
    }
}


// -------------------------------------------------------------------


void TB06Tree::check ()
{
  TB06Reco * entry = static_cast<TB06Reco*> (m_data->AddrAt (0)) ;

  std::cout << "[TB06Tree][check]reading . . . \n" ;
  std::cout << "[TB06Tree][check] entry->run: " << entry->run << "\n" ;
  std::cout << "[TB06Tree][check] entry->event: " << entry->event << "\n" ;
  std::cout << "[TB06Tree][check] entry->tableIsMoving: " << entry->tableIsMoving << "\n" ;
  std::cout << "[TB06Tree][check] entry->MEXTLeta: " << entry->MEXTLeta << "\n" ;
  std::cout << "[TB06Tree][check] entry->MEXTLphi: " << entry->MEXTLphi << "\n" ;
  std::cout << "[TB06Tree][check] entry->MEXTLenergy: " << entry->MEXTLenergy << "\n" ;

  for (int eta = 0 ; eta<7 ; ++eta)
      for (int phi = 0 ; phi<7 ; ++phi)
        std::cout << "[TB06Tree][check]   entry->localMap[" << eta
                  << "][" << phi << "]: "
                  << entry->localMap[eta][phi] << "\n" ;

  std::cout << "[TB06Tree][check] entry->xHodo: " << entry->xHodo << "\n" ;
  std::cout << "[TB06Tree][check] entry->yHodo: " << entry->yHodo << "\n" ;
  std::cout << "[TB06Tree][check] entry->xSlopeHodo: " << entry->xSlopeHodo << "\n" ;
  std::cout << "[TB06Tree][check] entry->ySlopeHodo: " << entry->ySlopeHodo << "\n" ;
  std::cout << "[TB06Tree][check] entry->xQualityHodo: " << entry->xQualityHodo << "\n" ;
  std::cout << "[TB06Tree][check] entry->yQualityHodo: " << entry->yQualityHodo << "\n" ;
  std::cout << "[TB06Tree][check] entry->convFactor: " << entry->convFactor << "\n" ;

  /* to be implemented with the right variables
  std::cout << "[TB06Tree][check] ------------------------" << std::endl ;
  std::cout << "[TB06Tree][check] " << entry->variable_name << std::endl ;
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
