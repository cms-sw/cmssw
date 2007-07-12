// -*- C++ -*-
//
// Package:    EcalCorrelatedNoiseAnalysisModules
// Class:      EcalCorrelatedNoisePedestalRunAnalyzer
// 
// class EcalCorrelatedNoisePedestalRunAnalyzer
// EcalCorrelatedNoisePedestalRunAnalyzer.cc
// CalibCalorimetry/EcalCorrelatedNoiseAnalysisModules/src/EcalCorrelatedNoisePedestalRunAnalyzer.cc

// Description: <one line class summary>

// Implementation:
//     <Notes on implementation>

//
// Original Author:  Bernard Fabbro
//         Created:  Fri Jun  2 10:27:01 CEST 2006
// $Id$
//
//          Update:  08/06/2007  

// CMSSW include files

#include <signal.h>
#include <time.h>

#include "Riostream.h"

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisModules/interface/EcalCorrelatedNoisePedestalRunAnalyzer.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEBParameters.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EcalCorrelatedNoisePedestalRunAnalyzer::EcalCorrelatedNoisePedestalRunAnalyzer(const edm::ParameterSet& pSet) : 
  verbosity_(pSet.getUntrackedParameter("verbosity", 1U)),
  nChannels_(0), iEvent_(0)
{
  //now do what ever initialization is needed

  using namespace edm;

  fgMaxCar = (Int_t)512;
  
  fTTBELL = '\007';

  //.................................. Init event number and counters for errors in BuildEventDistributions
  fEvtNumber = 0;

  fFalseBurst1 = 0;
  fFalseBurst2 = 0;
  fFalseBurst3 = 0;

  //.................................. Init parameters

  //......... USER DEPENDANT PARAMETERS
  fAnalysisName    = "CnP";              // Correlated noise Pedestal (DEFAULT ANALYSIS NAME)
  //......... DATA DEPENDANT PARAMETERS
  fRunNumber       = 0;
  fSuperModule     = 0;
  fNentries        = 0;
  //......... CONDITION DEPENDANT PARAMETERS
  TEBParameters* MyEcal = new TEBParameters();
  fNbOfTakenEvents = MyEcal->fMaxEvtsInBurstPedRun;
                // = nb of events in burst for a given gain (150 for TB2006)
  delete MyEcal;

  //......... ANCILLARY PARAMETERS

  fFirstTakenEvt = 0;   // evt number of the first burst

  fTimeBurst1_1 = 0;
  Int_t MaxCar = fgMaxCar; 
  fDateBurst1_1.Resize(MaxCar);
  fDateBurst1_1 = "?";

  fTimeBurst1_2 = 0;
  MaxCar = fgMaxCar; 
  fDateBurst1_2.Resize(MaxCar);
  fDateBurst1_2 = "?";

  fTimeBurst2_1 = 0;
  MaxCar = fgMaxCar; 
  fDateBurst2_1.Resize(MaxCar);
  fDateBurst2_1 = "?";

  fTimeBurst2_2 = 0;
  MaxCar = fgMaxCar; 
  fDateBurst2_2.Resize(MaxCar);
  fDateBurst2_2 = "?";


  fTimeBurst3_1 = 0;
  MaxCar = fgMaxCar; 
  fDateBurst3_1.Resize(MaxCar);
  fDateBurst3_1 = "?";

  fTimeBurst3_2 = 0;
  MaxCar = fgMaxCar; 
  fDateBurst3_2.Resize(MaxCar);
  fDateBurst3_2 = "?";

  //..................................

  eventHeaderProducer_ = pSet.getParameter<std::string>("eventHeaderProducer");
}
// end of constructor

EcalCorrelatedNoisePedestalRunAnalyzer::~EcalCorrelatedNoisePedestalRunAnalyzer()
{
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)

  //------------------------------------------------- BURST 1
  if ( fTimeBurst1_1 > fTimeBurst1_2 )
    {
      Int_t MaxCar = fgMaxCar; fDateBurst1_2.Resize(MaxCar);
      fDateBurst1_2 = "                                            ";
    }

  fMyCnaRunEBBurst1->StartStopDate(fDateBurst1_1, fDateBurst1_2);
  fMyCnaRunEBBurst1->StartStopTime(fTimeBurst1_1, fTimeBurst1_2);

  fMyCnaRunEBBurst1->GetReadyToCompute();

  fMyCnaRunEBBurst1->ComputeExpectationValuesOfSamples();
  fMyCnaRunEBBurst1->ComputeVariancesOfSamples();
  //fMyCnaRunEBBurst1->MakeHistosOfSampleDistributions(); 
  //fMyCnaRunEBBurst1->MakeHistosOfSamplesAsFunctionOfEvent();
  fMyCnaRunEBBurst1->ComputeCorrelationsBetweenSamples();
  //fMyCnaRunEBBurst1->ComputeCorrelationsBetweenChannelsMeanOverSamples();
  fMyCnaRunEBBurst1->ComputeCorrelationsBetweenTowersMeanOverSamplesAndChannels();
  fMyCnaRunEBBurst1->ComputeExpectationValuesOfExpectationValuesOfSamples();
  fMyCnaRunEBBurst1->ComputeExpectationValuesOfSigmasOfSamples();
  fMyCnaRunEBBurst1->ComputeExpectationValuesOfCorrelationsBetweenSamples();
  fMyCnaRunEBBurst1->ComputeSigmasOfExpectationValuesOfSamples();
  fMyCnaRunEBBurst1->ComputeSigmasOfSigmasOfSamples();
  fMyCnaRunEBBurst1->ComputeSigmasOfCorrelationsBetweenSamples();

  //fMyCnaRunEBBurst1->GetPathForResultsAsciiFiles();
  //fMyCnaRunEBBurst1->WriteAsciiCorrelationsBetweenSamples(1268);

  fMyCnaRunEBBurst1->GetPathForResultsRootFiles();
  if(fMyCnaRunEBBurst1->WriteRootFile() == kTRUE )
    {
      cout << "*EcalCorrelatedNoisePedestalRunAnalyzer> Write ROOT file OK" << endl;
    }
  else 
    {
      cout << "!EcalCorrelatedNoisePedestalRunAnalyzer> PROBLEM with write ROOT file." << fTTBELL << endl;
    }

  delete fMyCnaRunEBBurst1;
  
  //------------------------------------------------- BURST 2
  if ( fTimeBurst2_1 > fTimeBurst2_2 )
    {
      Int_t MaxCar = fgMaxCar; fDateBurst2_2.Resize(MaxCar);
      fDateBurst2_2 = "                                            ";
    }
  
  fMyCnaRunEBBurst2->StartStopDate(fDateBurst2_1, fDateBurst2_2);
  fMyCnaRunEBBurst2->StartStopTime(fTimeBurst2_1, fTimeBurst2_2);
  
  fMyCnaRunEBBurst2->GetReadyToCompute();
  
  fMyCnaRunEBBurst2->ComputeExpectationValuesOfSamples();
  fMyCnaRunEBBurst2->ComputeVariancesOfSamples();
  //fMyCnaRunEBBurst2->MakeHistosOfSampleDistributions(); 
  //fMyCnaRunEBBurst2->MakeHistosOfSamplesAsFunctionOfEvent();
  fMyCnaRunEBBurst2->ComputeCorrelationsBetweenSamples();
  //fMyCnaRunEBBurst2->ComputeCorrelationsBetweenChannelsMeanOverSamples();
  fMyCnaRunEBBurst2->ComputeCorrelationsBetweenTowersMeanOverSamplesAndChannels();
  fMyCnaRunEBBurst2->ComputeExpectationValuesOfExpectationValuesOfSamples();
  fMyCnaRunEBBurst2->ComputeExpectationValuesOfSigmasOfSamples();
  fMyCnaRunEBBurst2->ComputeExpectationValuesOfCorrelationsBetweenSamples();
  fMyCnaRunEBBurst2->ComputeSigmasOfExpectationValuesOfSamples();
  fMyCnaRunEBBurst2->ComputeSigmasOfSigmasOfSamples();
  fMyCnaRunEBBurst2->ComputeSigmasOfCorrelationsBetweenSamples();
  
  fMyCnaRunEBBurst2->GetPathForResultsRootFiles();
  if(fMyCnaRunEBBurst2->WriteRootFile() == kTRUE )
    {
      cout << "*EcalCorrelatedNoisePedestalRunAnalyzer> Write ROOT file OK" << endl;
    }
  else 
    {
      cout << "!EcalCorrelatedNoisePedestalRunAnalyzer> PROBLEM with write ROOT file." << fTTBELL << endl;
    }
  
  fMyCnaRunEBBurst2->GetPathForResultsAsciiFiles();         
  fMyCnaRunEBBurst2->WriteAsciiExpectationValuesOfSamples();
  fMyCnaRunEBBurst2->WriteAsciiVariancesOfSamples();
  fMyCnaRunEBBurst2->WriteAsciiCnaChannelTable();

  delete fMyCnaRunEBBurst2;
  
  //------------------------------------------------- BURST 3
  if ( fTimeBurst3_1 > fTimeBurst3_2 )
    {
      Int_t MaxCar = fgMaxCar; fDateBurst3_2.Resize(MaxCar);
      fDateBurst3_2 = "                                            ";
    }

  fMyCnaRunEBBurst3->StartStopDate(fDateBurst3_1, fDateBurst3_2);
  fMyCnaRunEBBurst3->StartStopTime(fTimeBurst3_1, fTimeBurst3_2);

  fMyCnaRunEBBurst3->GetReadyToCompute();

  fMyCnaRunEBBurst3->ComputeExpectationValuesOfSamples();
  fMyCnaRunEBBurst3->ComputeVariancesOfSamples();
  //fMyCnaRunEBBurst3->MakeHistosOfSampleDistributions(); 
  //fMyCnaRunEBBurst3->MakeHistosOfSamplesAsFunctionOfEvent();
  fMyCnaRunEBBurst3->ComputeCorrelationsBetweenSamples();
  //fMyCnaRunEBBurst3->ComputeCorrelationsBetweenChannelsMeanOverSamples();
  fMyCnaRunEBBurst3->ComputeCorrelationsBetweenTowersMeanOverSamplesAndChannels();
  fMyCnaRunEBBurst3->ComputeExpectationValuesOfExpectationValuesOfSamples();
  fMyCnaRunEBBurst3->ComputeExpectationValuesOfSigmasOfSamples();
  fMyCnaRunEBBurst3->ComputeExpectationValuesOfCorrelationsBetweenSamples();
  fMyCnaRunEBBurst3->ComputeSigmasOfExpectationValuesOfSamples();
  fMyCnaRunEBBurst3->ComputeSigmasOfSigmasOfSamples();
  fMyCnaRunEBBurst3->ComputeSigmasOfCorrelationsBetweenSamples();

  fMyCnaRunEBBurst3->GetPathForResultsRootFiles();
  if(fMyCnaRunEBBurst3->WriteRootFile() == kTRUE )
    {
      cout << "*EcalCorrelatedNoisePedestalRunAnalyzer> Write ROOT file OK" << endl;
    }
  else 
    {
      cout << "!EcalCorrelatedNoisePedestalRunAnalyzer> PROBLEM with write ROOT file." << fTTBELL << endl;
    }

  delete fMyCnaRunEBBurst3;

  //---------------------------------------- end of treatment ---------------------------------------------------

  cout << "*EcalCorrelatedNoisePedestalRunAnalyzer> Numbers of events with ERROR(S) returned by BuildEventDistributions(): "
       << endl;
  cout << "                                     Burst 1 : " << fFalseBurst1 <<  endl;
  cout << "                                     Burst 2 : " << fFalseBurst2 <<  endl;
  cout << "                                     Burst 3 : " << fFalseBurst3 <<  endl;

  cout << endl<< "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - " <<  endl;

  cout << "*EcalCorrelatedNoisePedestalRunAnalyzer-destructor> fDateBurst1_1 = " << fDateBurst1_1 
       << "                    fDateBurst1_2 = " << fDateBurst1_2 << endl << endl;
  cout << "*EcalCorrelatedNoisePedestalRunAnalyzer-destructor> fDateBurst2_1 = " << fDateBurst2_1 
       << "                    fDateBurst2_2 = " << fDateBurst2_2 << endl << endl;
  cout << "*EcalCorrelatedNoisePedestalRunAnalyzer-destructor> fDateBurst3_1 = " << fDateBurst3_1 
       << "                    fDateBurst3_2 = " << fDateBurst3_2 << endl << endl;

  delete fMyEBNumbering;
}
// end of destructor


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EcalCorrelatedNoisePedestalRunAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  
  //  cout << endl
  //	<< "---------------------------------------------------------------"
  //	<< endl;
  //  cout << "EcalCorrelatedNoisePedestalRunAnalyzer::analyze(...)> BEGINNING for event number = " << iEvent_ << endl;

  Handle<EcalTBEventHeader> pEventHeader;
  const EcalTBEventHeader* myEventHeader = 0;
  
  iEvent.getByLabel(eventHeaderProducer_ , pEventHeader);
  myEventHeader = pEventHeader.product();
  fRunNumber    = myEventHeader->runNumber();
  fSuperModule  = myEventHeader->smInBeam(); 

  Int_t nFirstTakenEventBurst1 = fFirstTakenEvt;
  Int_t nFirstTakenEventBurst2 = nFirstTakenEventBurst1 + fNbOfTakenEvents;
  Int_t nFirstTakenEventBurst3 = nFirstTakenEventBurst2 + fNbOfTakenEvents;

  if( fEvtNumber == 0 )
    {
      //------------------------------------- CNA init  
      fMyCnaRunEBBurst1 = new TCnaRunEB();
      fMyCnaRunEBBurst2 = new TCnaRunEB();
      fMyCnaRunEBBurst3 = new TCnaRunEB();

      fMyCnaRunEBBurst1->PrintAllComments();
      fMyCnaRunEBBurst2->PrintAllComments();
      fMyCnaRunEBBurst3->PrintAllComments();
 
      cout << endl;
      
      cout << "*EcalCorrelatedNoisePedestalRunAnalyzer> RunNumber          = " << fRunNumber   << endl;
      cout << "*EcalCorrelatedNoisePedestalRunAnalyzer> SuperModule number = " << fSuperModule << endl; 

      // string cintoto; cout << endl << "*EcalCorrelatedNoisePedestalRunAnalyzer> enter 0 to continue ==> ";   cin >> cintoto;
      
      
      //-----------------------------------------------------------
      fNentries    = 3*fNbOfTakenEvents;       //  3 bursts de 150 chacun. Devrait se recuperer par les data
      
      
      fMyCnaRunEBBurst1->GetReadyToReadData(fAnalysisName,          fRunNumber,
					    nFirstTakenEventBurst1, fNbOfTakenEvents,
					    fSuperModule,           fNentries);
      
      fMyCnaRunEBBurst2->GetReadyToReadData(fAnalysisName,          fRunNumber,
					    nFirstTakenEventBurst2, fNbOfTakenEvents,
					    fSuperModule,           fNentries);
      
      fMyCnaRunEBBurst3->GetReadyToReadData(fAnalysisName,          fRunNumber,
					    nFirstTakenEventBurst3, fNbOfTakenEvents,
					    fSuperModule,           fNentries);
      
      fMyEBNumbering = new TEBNumbering();
    }
  
  //..........................................
  
  edm::Handle<EBDigiCollection> digis;
  //  iEvent.getByLabel("ecalEBunpacker", digis);
  iEvent.getByLabel("ecalTBunpack", digis);
  
  // Initialize vectors if not already done
  if ( int(digis->size()) > nChannels_ ) {
    nChannels_ = digis->size();
  }
  
  Int_t iNbOfCh = Int_t(digis->size());

  Int_t TakenEventNumberInBurst1 = fEvtNumber - nFirstTakenEventBurst1;
  Int_t TakenEventNumberInBurst2 = fEvtNumber - nFirstTakenEventBurst2;
  Int_t TakenEventNumberInBurst3 = fEvtNumber - nFirstTakenEventBurst3;

  Int_t nLastEventNumberInBurst1 = nFirstTakenEventBurst2;  
  Int_t nLastEventNumberInBurst2 = nFirstTakenEventBurst3;  
  Int_t nLastEventNumberInBurst3 = nFirstTakenEventBurst3 + fNbOfTakenEvents;
 
  Int_t iCurrentBurstNumber = 0;
  
  if( fEvtNumber >= nFirstTakenEventBurst1 && fEvtNumber < nLastEventNumberInBurst1 ){iCurrentBurstNumber = 1;}
  if( fEvtNumber >= nFirstTakenEventBurst2 && fEvtNumber < nLastEventNumberInBurst2 ){iCurrentBurstNumber = 2;}
  if( fEvtNumber >= nFirstTakenEventBurst3 && fEvtNumber < nLastEventNumberInBurst3 ){iCurrentBurstNumber = 3;}

  //....................................................... date of first and last event
  //---------------------------------------------BURST 1  
  if( iCurrentBurstNumber == 1 )
    {
      if( fEvtNumber ==  nFirstTakenEventBurst1 )
	{
          time_t         i_current_ev_time = (time_t)myEventHeader->begBurstTimeSec();
	  const time_t*  p_current_ev_time = &i_current_ev_time;
	  char*          astime            = ctime(p_current_ev_time);
	  fTimeBurst1_1 = i_current_ev_time;   
	  fDateBurst1_1 = astime;
	}
      if( fEvtNumber == nLastEventNumberInBurst1 - 1 )
	{
          time_t         i_current_ev_time = (time_t)myEventHeader->endBurstTimeSec();
	  const time_t*  p_current_ev_time = &i_current_ev_time;
	  char*          astime            = ctime(p_current_ev_time);
	  fTimeBurst1_2 = i_current_ev_time;
	  fDateBurst1_2 = astime;
	}
    }
  else  // not in burst 1
    {
      //---------------------------------------------BURST 2
      if( iCurrentBurstNumber == 2 )
	{
	  if( fEvtNumber ==  nFirstTakenEventBurst2 )
	    {
	      time_t         i_current_ev_time = (time_t)myEventHeader->begBurstTimeSec();
	      const time_t*  p_current_ev_time = &i_current_ev_time;
	      char*          astime            = ctime(p_current_ev_time);
	      fTimeBurst2_1 = i_current_ev_time;   
	      fDateBurst2_1 = astime;
	    }
	  if( fEvtNumber == nLastEventNumberInBurst2 - 1 )
	    {
	      time_t         i_current_ev_time = (time_t)myEventHeader->endBurstTimeSec();
	      const time_t*  p_current_ev_time = &i_current_ev_time;
	      char*          astime            = ctime(p_current_ev_time);

	      fTimeBurst2_2 = i_current_ev_time;
	      fDateBurst2_2 = astime;
	    }
	}
      else  // not in burst 1 and not in burst 2
	{
	  //---------------------------------------------BURST 3
	  if( iCurrentBurstNumber == 3 )
	    {
	      if( fEvtNumber ==  nFirstTakenEventBurst3 )
		{
		  time_t         i_current_ev_time = (time_t)myEventHeader->begBurstTimeSec();
		  const time_t*  p_current_ev_time = &i_current_ev_time;
		  char*          astime            = ctime(p_current_ev_time);
		  fTimeBurst3_1 = i_current_ev_time;   
		  fDateBurst3_1 = astime;
		}
	      if( fEvtNumber == nLastEventNumberInBurst3 - 1 )
		{
		  time_t         i_current_ev_time = (time_t)myEventHeader->endBurstTimeSec();
		  const time_t*  p_current_ev_time = &i_current_ev_time;
		  char*          astime            = ctime(p_current_ev_time);
		  fTimeBurst3_2 = i_current_ev_time;
		  fDateBurst3_2 = astime;
		}
	    }
	  else  // not in burst 1 and not in burst 2 and not in burst 3
	    {
	      cout << "!EcalCorrelatedNoisePedestalRunAnalyzer::analyze(...) *** ERROR *** ===> Event number "
		   << iEvent_ << ": QUANTITIES OUT OF RANGE!" << fTTBELL << endl;
	      cout << "!EcalCorrelatedNoisePedestalRunAnalyzer::analyze(...)>  digis->size()  = " << iNbOfCh << endl;
	      
	      cout << "*EcalCorrelatedNoisePedestalRunAnalyzer::analyze(...)> End of analysis forced after ERROR." << endl;
	      kill(getpid(),SIGUSR2);  
	    }
	}
    }
  
  //................................................. Calls to BuildEventDistributions
  if( Int_t(digis->end()-digis->begin()) >= 0 ||
      Int_t(digis->end()-digis->begin()) <  Int_t(digis->size()) )
    {
      TEBParameters* MyEcal = new TEBParameters();

      // Loop over Ecal barrel digis
      for (EBDigiCollection::const_iterator digiItr = digis->begin(); 
	   digiItr != digis->end(); ++digiItr) 
	{
	  int iEta = digiItr->id().ieta();
	  int iPhi = digiItr->id().iphi();
	  
	  Int_t iSMCrys  = (iEta-1)*(MyEcal->fMaxTowPhiInSM*MyEcal->fMaxCrysPhiInTow) + iPhi;
	  Int_t smTower  = fMyEBNumbering->GetSMTowFromSMCrys(iSMCrys);
	  Int_t iTowEcha = fMyEBNumbering->GetTowEchaFromSMCrys(iSMCrys);
	  
	  Int_t nSample  = digiItr->size();

	  if( nSample >= 0 || nSample < MyEcal->fMaxSampADC )
	    {
	      // Loop over the samples
	      for (Int_t iSample = 0; iSample < nSample; ++iSample)
		{
		  Double_t adc = (Double_t)((*digiItr).sample(iSample).adc());
		  
		  if( iCurrentBurstNumber == 1 )
		    {
		      if( fMyCnaRunEBBurst1->BuildEventDistributions
			  (TakenEventNumberInBurst1,smTower,iTowEcha,iSample,adc) == kFALSE ){fFalseBurst1++;}
		    }
		  if( iCurrentBurstNumber == 2 )
		    {
		      if( fMyCnaRunEBBurst2->BuildEventDistributions
			  (TakenEventNumberInBurst2,smTower,iTowEcha,iSample,adc) == kFALSE ){fFalseBurst2++;}
		    }
		  if( iCurrentBurstNumber == 3 )
		    {
		      if( fMyCnaRunEBBurst3->BuildEventDistributions
			  (TakenEventNumberInBurst3,smTower,iTowEcha,iSample,adc) == kFALSE ){fFalseBurst3++;}
		    }
		}
	    }
	  else
	    {
	      cout << "EcalCorrelatedNoisePedestalRunAnalyzer::analyze(...)> nSample out of bounds = " << nSample << endl;
	    }
	}
      delete MyEcal;
    }

  // cout << "EcalCorrelatedNoisePedestalRunAnalyzer::analyze(...)> END for event number = " << iEvent_ << endl;
  // cout << "---------------------------------------------------------------"
  //     << endl;
  
  fEvtNumber++;
  iEvent_++;
}
// end of analyzer
