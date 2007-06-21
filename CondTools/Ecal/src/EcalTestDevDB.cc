#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

#include "DataFormats/Provenance/interface/Timestamp.h"

#include "CondTools/Ecal/interface/EcalTestDevDB.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <map>
#include <vector>

using namespace std;

EcalTestDevDB::EcalTestDevDB(const edm::ParameterSet& iConfig) :
  m_timetype(iConfig.getParameter<std::string>("timetype")),
  m_cacheIDs(),
  m_records()
{

  std::string container;
  std::string tag;
  std::string record;

  m_firstRun=(unsigned long)atoi( iConfig.getParameter<std::string>("firstRun").c_str());
  m_lastRun=(unsigned long)atoi( iConfig.getParameter<std::string>("lastRun").c_str());
  m_interval=(unsigned int)atoi( iConfig.getParameter<std::string>("interval").c_str());

  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters toCopy = iConfig.getParameter<Parameters>("toCopy");
  for(Parameters::iterator i = toCopy.begin(); i != toCopy.end(); ++i) {
    container = i->getParameter<std::string>("container");
    record = i->getParameter<std::string>("record");
    m_cacheIDs.insert( std::make_pair(container, 0) );
    m_records.insert( std::make_pair(container, record) );
  }
  
}


EcalTestDevDB::~EcalTestDevDB()
{
  
}

void EcalTestDevDB::analyze( const edm::Event& evt, const edm::EventSetup& evtSetup)
{

  edm::Service<cond::service::PoolDBOutputService> dbOutput;
  if ( !dbOutput.isAvailable() ) {
    throw cms::Exception("PoolDBOutputService is not available");
  }

 

  std::string container;
  std::string record;
  typedef std::map<std::string, std::string>::const_iterator recordIter;
  for (recordIter i = m_records.begin(); i != m_records.end(); ++i) {
    container = (*i).first;
    record = (*i).second;

    std::string recordName = m_records[container];

    
    // Loop through each of the runs

    unsigned long nrec=(m_lastRun-m_firstRun)/m_interval+1;
    unsigned long nstart=0;
    if (m_firstRun == 0 && m_lastRun == 0) {
       // it should do at least once the loop
      nstart=0;
      nrec=1;
    }

    for(unsigned long i=nstart; i<nrec; i++) {
      unsigned long irun=m_firstRun+i*m_interval;
 
 
      // Arguments 0 0 mean infinite IOV
      if (m_firstRun == 0 && m_lastRun == 0) {
	cout << "Infinite IOV mode" << endl;
	irun = edm::IOVSyncValue::endOfTime().eventID().run();
      }

      cout << "Starting Transaction for run " << irun << "..." << flush;
    
 

      if (container == "EcalPedestals") {
	EcalPedestals* condObject= generateEcalPedestals();

	// cambiare dappertutto cosi` !!!!!!!!!!!!!!!!!!!!!!!!!!
	if(irun==m_firstRun && dbOutput->isNewTagRequest(recordName)) {
	  // create new
	  std::cout<<"First One "<<std::endl;
	  dbOutput->createNewIOV<const EcalPedestals>( condObject, dbOutput->endOfTime() ,recordName);
	} else {
	  // append
	  std::cout<<"Old One "<<std::endl;
	  dbOutput->appendSinceTime<const EcalPedestals>( condObject, irun , recordName);
	}
	
      } else if (container == "EcalADCToGeVConstant") {
	
	EcalADCToGeVConstant* condObject= generateEcalADCToGeVConstant();
	if(irun==m_firstRun && dbOutput->isNewTagRequest(recordName)) {
	  // create new
	  std::cout<<"First One "<<std::endl;
	  dbOutput->createNewIOV<const EcalADCToGeVConstant>( condObject, dbOutput->endOfTime() ,recordName);
	} else {
	  // append
	  std::cout<<"Old One "<<std::endl;
	  dbOutput->appendSinceTime<const EcalADCToGeVConstant>( condObject, irun , recordName);
	}
	
	
      } else if (container == "EcalIntercalibConstants") {
	EcalIntercalibConstants* condObject= generateEcalIntercalibConstants();
	if(irun==m_firstRun && dbOutput->isNewTagRequest(recordName)) {
	  // create new
	  std::cout<<"First One "<<std::endl;
	  dbOutput->createNewIOV<const EcalIntercalibConstants>( condObject, dbOutput->endOfTime() ,recordName);
	} else {
	  // append
	  std::cout<<"Old One "<<std::endl;
	  dbOutput->appendSinceTime<const EcalIntercalibConstants>( condObject, irun , recordName);
	}
	
      } else if (container == "EcalGainRatios") {
	EcalGainRatios* condObject= generateEcalGainRatios();
	if(irun==m_firstRun && dbOutput->isNewTagRequest(recordName)) {
	  // create new
	  std::cout<<"First One "<<std::endl;
	  dbOutput->createNewIOV<const EcalGainRatios>( condObject, dbOutput->endOfTime() ,recordName);
	} else {
	  // append
	  std::cout<<"Old One "<<std::endl;
	  dbOutput->appendSinceTime<const EcalGainRatios>( condObject, irun , recordName);
	}
	
      } else if (container == "EcalWeightXtalGroups") {
	EcalWeightXtalGroups* condObject= generateEcalWeightXtalGroups();
	if(irun==m_firstRun && dbOutput->isNewTagRequest(recordName)) {
	  // create new
	  std::cout<<"First One "<<std::endl;
	  dbOutput->createNewIOV<const EcalWeightXtalGroups>( condObject,  dbOutput->endOfTime() ,recordName);
	} else {
	  // append
	  std::cout<<"Old One "<<std::endl;
	  dbOutput->appendSinceTime<const EcalWeightXtalGroups>( condObject, irun , recordName);
	}
	
      } else if (container == "EcalTBWeights") {
	EcalTBWeights* condObject= generateEcalTBWeights();
	if(irun==m_firstRun && dbOutput->isNewTagRequest(recordName)) {
	  // create new
	  std::cout<<"First One "<<std::endl;
	  dbOutput->createNewIOV<const EcalTBWeights>( condObject,  dbOutput->endOfTime() ,recordName);
	} else {
	  // append
	  std::cout<<"Old One "<<std::endl;
	  dbOutput->appendSinceTime<const EcalTBWeights>( condObject, irun , recordName);
	}
	
      } else if (container == "EcalLaserAPDPNRatios") {
	EcalLaserAPDPNRatios* condObject= generateEcalLaserAPDPNRatios();
	if(irun==m_firstRun && dbOutput->isNewTagRequest(recordName)) {
	  // create new
	  std::cout<<"First One "<<std::endl;
	  dbOutput->createNewIOV<const EcalLaserAPDPNRatios>( condObject,  dbOutput->endOfTime() ,recordName);
	} else {
	  // append
	  std::cout<<"Old One "<<std::endl;
	  dbOutput->appendSinceTime<const EcalLaserAPDPNRatios>( condObject, irun , recordName);
	}
      } else if (container == "EcalLaserAPDPNRatiosRef") {
	EcalLaserAPDPNRatiosRef* condObject= generateEcalLaserAPDPNRatiosRef();
	if(irun==m_firstRun && dbOutput->isNewTagRequest(recordName)) {
	  // create new
	  std::cout<<"First One "<<std::endl;
	  dbOutput->createNewIOV<const EcalLaserAPDPNRatiosRef>( condObject,  dbOutput->endOfTime() ,recordName);
	} else {
	  // append
	  std::cout<<"Old One "<<std::endl;
	  dbOutput->appendSinceTime<const EcalLaserAPDPNRatiosRef>( condObject, irun , recordName);
	}
      } else if (container == "EcalLaserAlphas") {
	EcalLaserAlphas* condObject= generateEcalLaserAlphas();
	if(irun==m_firstRun && dbOutput->isNewTagRequest(recordName)) {
	  // create new
	  std::cout<<"First One "<<std::endl;
	  dbOutput->createNewIOV<const EcalLaserAlphas>( condObject,  dbOutput->endOfTime() ,recordName);
	} else {
	  // append
	  std::cout<<"Old One "<<std::endl;
	  dbOutput->appendSinceTime<const EcalLaserAlphas>( condObject, irun , recordName);
	}
      } else {
	cout << "it does not work yet for " << container << "..." << flush;
	
      }
      
      
    }
    
    
  }
  
  
}


//-------------------------------------------------------------
EcalPedestals*
EcalTestDevDB::generateEcalPedestals() {
//-------------------------------------------------------------

  EcalPedestals* peds = new EcalPedestals();
  EcalPedestals::Item item;
  for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
    if(iEta==0) continue;
    for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
      item.mean_x1  = 200.*( (double)std::rand()/(double(RAND_MAX)+double(1)) );
      item.rms_x1   = (double)std::rand()/(double(RAND_MAX)+double(1));
      item.mean_x6  = 1200.*( (double)std::rand()/(double(RAND_MAX)+double(1)) );
      item.rms_x6   = 6.*( (double)std::rand()/(double(RAND_MAX)+double(1)) );
      item.mean_x12 = 2400.*( (double)std::rand()/(double(RAND_MAX)+double(1)) );
      item.rms_x12  = 12.*( (double)std::rand()/(double(RAND_MAX)+double(1)) );

      EBDetId ebdetid(iEta,iPhi);
      peds->m_pedestals.insert(std::make_pair(ebdetid.rawId(),item));
    }
  }
  return peds;
}

//-------------------------------------------------------------
EcalADCToGeVConstant*
EcalTestDevDB::generateEcalADCToGeVConstant() {
//-------------------------------------------------------------
  
  double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
  EcalADCToGeVConstant* agc = new EcalADCToGeVConstant(36.+r*4., 60.+r*4);
  return agc;
}

//-------------------------------------------------------------
EcalIntercalibConstants*
EcalTestDevDB::generateEcalIntercalibConstants() {
//-------------------------------------------------------------

  EcalIntercalibConstants* ical = new EcalIntercalibConstants();

  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId:: MAX_IPHI; ++iphi) {

      EBDetId ebid(ieta,iphi);

      double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
      ical->setValue( ebid.rawId(), 0.85 + r*0.3 );
    } // loop over phi
  } // loop over eta
  return ical;
}

//-------------------------------------------------------------
EcalGainRatios*
EcalTestDevDB::generateEcalGainRatios() {
//-------------------------------------------------------------

  // create gain ratios
  EcalGainRatios* gratio = new EcalGainRatios;

  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId:: MAX_IPHI; ++iphi) {

      EBDetId ebid(ieta,iphi);

      double r = (double)std::rand()/( double(RAND_MAX)+double(1) );

      EcalMGPAGainRatio gr;
      gr.setGain12Over6( 1.9 + r*0.2 );
      gr.setGain6Over1( 5.9 + r*0.2 );

      gratio->setValue( ebid.rawId(), gr );

    } // loop over phi
  } // loop over eta
  return gratio;
}

//-------------------------------------------------------------
EcalWeightXtalGroups*
EcalTestDevDB::generateEcalWeightXtalGroups() {
//-------------------------------------------------------------

  EcalWeightXtalGroups* xtalGroups = new EcalWeightXtalGroups();
  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId:: MAX_IPHI; ++iphi) {
      EBDetId ebid(ieta,iphi);
      xtalGroups->setValue(ebid.rawId(), EcalXtalGroupId(ieta) ); // define rings in eta
    }
  }
  return xtalGroups;
}

//-------------------------------------------------------------
EcalTBWeights*
EcalTestDevDB::generateEcalTBWeights() {
//-------------------------------------------------------------

  EcalTBWeights* tbwgt = new EcalTBWeights();

  // create weights for each distinct group ID
  int nMaxTDC = 10;
  for(int igrp=-EBDetId::MAX_IETA; igrp<=EBDetId::MAX_IETA; ++igrp) {
    if(igrp==0) continue;
    for(int itdc=1; itdc<=nMaxTDC; ++itdc) {
      // generate random number
      double r = (double)std::rand()/( double(RAND_MAX)+double(1) );

      // make a new set of weights
      EcalWeightSet wgt;
      EcalWeightSet::EcalWeightMatrix& mat1 = wgt.getWeightsBeforeGainSwitch();
      EcalWeightSet::EcalWeightMatrix& mat2 = wgt.getWeightsAfterGainSwitch();

      for(size_t i=0; i<3; ++i) {
        for(size_t j=0; j<10; ++j) {
          double ww = igrp*itdc*r + i*10. + j;
          //std::cout << "row: " << i << " col: " << j << " -  val: " << ww  << std::endl;
	  mat1(i,j)=ww;
	  mat2(i,j)=100+ww;
        }
      }

      // fill the chi2 matrcies
      r = (double)std::rand()/( double(RAND_MAX)+double(1) );
      EcalWeightSet::EcalChi2WeightMatrix& mat3 = wgt.getChi2WeightsBeforeGainSwitch();
      EcalWeightSet::EcalChi2WeightMatrix& mat4 = wgt.getChi2WeightsAfterGainSwitch();
      for(size_t i=0; i<10; ++i) {
        for(size_t j=0; j<10; ++j) {
          double ww = igrp*itdc*r + i*10. + j;
	  mat3(i,j)=1000+ww;
	  mat4(i,j)=1000+100+ww;
        }
      }

      // put the weight in the container
      tbwgt->setValue(std::make_pair(igrp,itdc), wgt);
    }
  }
  return tbwgt;
}

//-------------------------------------------------------------
EcalLaserAPDPNRatios*
EcalTestDevDB::generateEcalLaserAPDPNRatios() {
  EcalLaserAPDPNRatios* laser = new EcalLaserAPDPNRatios();

  EcalLaserAPDPNRatios::EcalLaserAPDPNpair APDPNpair;
  EcalLaserAPDPNRatios::EcalLaserTimeStamp TimeStamp;
 
  for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
    if(iEta==0) continue;
    for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
      APDPNpair.p1=std::rand()/((RAND_MAX)+1);
      APDPNpair.p2=std::rand()/((RAND_MAX)+1);
   
      EBDetId ebid(iEta,iPhi);
      laser->setValue( ebid.rawId(), APDPNpair);
    }
  }
  for(int i=0; i<88; i++){
    TimeStamp.t1 = (uint32_t) 1;
    TimeStamp.t2 = (uint32_t) 1.5;
   
    laser->setTime(i, TimeStamp);
  }
  return laser;
}

//-------------------------------------------------------------
EcalLaserAPDPNRatiosRef*
EcalTestDevDB::generateEcalLaserAPDPNRatiosRef() {
  EcalLaserAPDPNRatiosRef* laser = new EcalLaserAPDPNRatiosRef();
  EcalLaserAPDPNRatiosRef::EcalLaserAPDPNref val;

 
  for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
    if(iEta==0) continue;
    for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
      val=std::rand()/((RAND_MAX)+1);
      EBDetId ebid(iEta,iPhi);
      laser->setValue( ebid.rawId(), val);
    }
  }
  return laser;
}

//-------------------------------------------------------------
EcalLaserAlphas*
EcalTestDevDB::generateEcalLaserAlphas() {
  EcalLaserAlphas* laser = new EcalLaserAlphas();

  EcalLaserAlphas::EcalLaserAlpha alpha;
 
  for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
    if(iEta==0) continue;
    for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
      alpha=1.56*(1+(std::rand()/((RAND_MAX)+1))*0.15);   
      EBDetId ebid(iEta,iPhi);
      laser->setValue( ebid.rawId(), alpha);
    }
  }
  return laser;
}
