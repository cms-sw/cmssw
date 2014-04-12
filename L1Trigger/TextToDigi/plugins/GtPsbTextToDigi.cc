
#include "GtPsbTextToDigi.h"
// general
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>
#include <iostream>
// gct
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
//#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

GtPsbTextToDigi::GtPsbTextToDigi(const edm::ParameterSet& iConfig):
  m_fileEventOffset(iConfig.getUntrackedParameter<int>("FileEventOffset",0)),
  m_textFileName(iConfig.getParameter<std::string>("TextFileName")),
  m_nevt(0) {
  // Produces collections
  produces<L1GctEmCandCollection>("isoEm");
  produces<L1GctEmCandCollection>("nonIsoEm");
  produces<L1GctJetCandCollection>("cenJets");
  produces<L1GctJetCandCollection>("forJets");
  produces<L1GctJetCandCollection>("tauJets");
  //produces<L1GctEtTotal>();

  // Open the input files
  for (int ifile=0; ifile<4; ifile++){
    // gct em cand coll: (noiso) 0<->0, 1<->1, (iso) 6<->2, 7<->3
    int ii = (ifile<2)?ifile:ifile+4;
    std::stringstream fileStream;
    fileStream << m_textFileName << ii << ".txt";
    std::string fileName(fileStream.str());
    m_file[ifile].open(fileName.c_str(),std::ios::in);
    if(!m_file[ifile].good()) {
      throw cms::Exception("GtPsbTextToDigiTextFileOpenError")
	//LogDebug("GtPsbTextToDigi")
	<< "GtPsbTextToDigi::GtPsbTextToDigi : "
	<< " couldn't open the file " << fileName 
	//<< "...skipping!" 
	<< std::endl;
    }
    std::hex(m_file[ifile]); 
  }

  // Initialize bc0 position holder
  for(int i=0; i<4; i++)
    m_bc0[i]=-1;  
}

GtPsbTextToDigi::~GtPsbTextToDigi() {
  // Close the input files
  for (unsigned i=0; i<4; i++){  
    m_file[i].close();
  }
}

/// Append empty digi collection
void GtPsbTextToDigi::putEmptyDigi(edm::Event& iEvent) {
  std::auto_ptr<L1GctEmCandCollection>  gctIsolaEm( new L1GctEmCandCollection () );
  std::auto_ptr<L1GctEmCandCollection>  gctNoIsoEm( new L1GctEmCandCollection () );
  std::auto_ptr<L1GctJetCandCollection> gctCenJets( new L1GctJetCandCollection() );
  std::auto_ptr<L1GctJetCandCollection> gctForJets( new L1GctJetCandCollection() );
  std::auto_ptr<L1GctJetCandCollection> gctTauJets( new L1GctJetCandCollection() );
  //std::auto_ptr<L1GctEtTotal>           gctEtTotal( new L1GctEtTotal          () );
  for (int i=0; i<4; i++){  
    gctIsolaEm->push_back(L1GctEmCand (0,1));
    gctNoIsoEm->push_back(L1GctEmCand (0,0));
    gctCenJets->push_back(L1GctJetCand(0,0,0));
    gctForJets->push_back(L1GctJetCand(0,0,1));
    gctTauJets->push_back(L1GctJetCand(0,1,0));
    //gctEtTotal->push_back(());
  }
  iEvent.put(gctIsolaEm, "isoEm");
  iEvent.put(gctNoIsoEm, "nonIsoEm");
  iEvent.put(gctCenJets, "cenJets");
  iEvent.put(gctForJets, "forJets");
  iEvent.put(gctTauJets, "tauJets");
  //iEvent.put(gctEtTotal);

  LogDebug("GtPsbTextToDigi") << "putting empty digi (evt:" << m_nevt << ")\n"; 
}

void GtPsbTextToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // specify clock cycle bit sequence 1 0 1 0... or 0 1 0 1...
  unsigned short cbs[2] = {1,0};

  // Skip event if required
  if (m_nevt < m_fileEventOffset){ 
    putEmptyDigi(iEvent);
    LogDebug("GtPsbTextToDigi")
      << "[GtPsbTextToDigi::produce()] skipping event " << m_nevt 
      << std::endl;
    m_nevt++;
    return;
  } else if (m_nevt==0 && m_fileEventOffset<0) {
    // skip first fileEventOffset input events
    unsigned long int buff;
    for (int ievt=0; ievt<abs(m_fileEventOffset); ievt++){  
      for (int ifile=0; ifile<4; ifile++){  
  	for (int cycle=0; cycle<2; cycle++){  
	  std::hex(m_file[ifile]);
  	  m_file[ifile] >> buff;
	  unsigned tmp = (buff>>15)&0x1;
	  if(tmp!=cbs[cycle]) {
	    if(m_bc0[ifile]==-1 && cycle==1 && tmp==1)
	      m_bc0[ifile] = ievt;
	    else 
	      throw cms::Exception("GtPsbTextToDigiTextFileFormatError") 
		//std::cout << "GtPsbTextToDigiTextFileFormatError" 
		<< "GtPsbTextToDigi::produce : "
		<< " found format inconsistency in file #" << ifile 
		<< "\n in skipped line:" << ievt*2+1
		<< " cycle:" << tmp << " is different from " << cbs[cycle]
		<< std::endl;
	  }
  	} 
      }
      LogDebug("GtPsbTextToDigi")
	<< "[GtPsbTextToDigi::produce()] skipping input " << ievt 
	<< std::endl;
    }
  }
  m_nevt++;
  
  // New collections
  std::auto_ptr<L1GctEmCandCollection>  gctIsolaEm( new L1GctEmCandCollection () );
  std::auto_ptr<L1GctEmCandCollection>  gctNoIsoEm( new L1GctEmCandCollection () );
  std::auto_ptr<L1GctJetCandCollection> gctCenJets( new L1GctJetCandCollection() ); 
  std::auto_ptr<L1GctJetCandCollection> gctForJets( new L1GctJetCandCollection() ); 
  std::auto_ptr<L1GctJetCandCollection> gctTauJets( new L1GctJetCandCollection() );
  //std::auto_ptr<L1GctEtTotal>           gctEtTotal( new L1GctEtTotal          () );

  /// buffer
  uint16_t data[4][2]= {{0}};
  for (int i=0; i<4; i++)
    for (int j=0; j<2; j++)
      data[i][j]=0;

  // Loop over files
  for (int ifile=0; ifile<4; ifile++){  
    int ii = (ifile<2)?ifile:ifile+4;
    
    // Check we're not at the end of the file
    if(m_file[ifile].eof()) {
      LogDebug("GtPsbTextToDigi")
	<< "GtPsbTextToDigi::produce : "
	<< " unexpected end of file " << m_textFileName << ii << ".txt"
	<< std::endl;
      putEmptyDigi(iEvent);
      continue;
    }      
    
    if(!m_file[ifile].good()) {
      LogDebug("GtPsbTextToDigi")
	<< "GtPsbTextToDigi::produce : "
	<< " problem reading file " << m_textFileName << ii << ".txt"
	<< std::endl;
      putEmptyDigi(iEvent);
      continue;
    }
    
    /// Read in file
    unsigned long int uLongBuffer;
    
    for(unsigned cycle=0; cycle<2; cycle++) {
      m_file[ifile] >> uLongBuffer;
      unsigned tmp = (uLongBuffer>>15)&0x1;
      
      /// cycle debuging (temporary)
      if(false && tmp!=cbs[cycle]) 
      	std::cout << "[GtPsbTextToDigi::produce()] asserting " 
      		  << " evt:"   << m_nevt
      		  << " ifile:" << ifile
      		  << " cycle:" << cbs[cycle]
      		  << std::hex
      		  << " buffer:"<< uLongBuffer
      		  << " tmp: "  << tmp
      		  << std::dec 
      		  << "\n\n" << std::flush;

      if(tmp!=cbs[cycle]){
	if(m_bc0[ifile]==-1 && cycle==1 && tmp==1){
	  m_bc0[ifile] = (m_nevt-m_fileEventOffset);
	}else{
	  throw cms::Exception("GtPsbTextToDigiTextFileFormatError") 
	    //std::cout << "GtPsbTextToDigiTextFileFormatError " 
	    << "GtPsbTextToDigi::produce : "
	    << " found format inconsistency in file #" << ifile 
	    << "\n in line:" << (m_nevt-m_fileEventOffset)*2-1  
	    << " cycle:" << tmp << " is different from " << cbs[cycle]
	    << std::endl;
        }
      }
      data[ifile][cycle] = (uLongBuffer&0x7fff);
    } //cycle loop
  } //ifile loop
  
  /// Fill in digi collections
  unsigned iIsola, iNoIso;
  for (unsigned cycle=0; cycle<2; cycle++){  
    for (unsigned i=0; i<2; i++){  
      iIsola = i+2;
      iNoIso = i;
      gctIsolaEm->push_back(L1GctEmCand(data[iIsola][cycle]&0x7fff,1));
      gctNoIsoEm->push_back(L1GctEmCand(data[iNoIso][cycle]&0x7fff,0));
      L1GctEmCand candI(data[iIsola][cycle],1);
      L1GctEmCand candN(data[iNoIso][cycle],0);
    }
  }

  /// Put collections
  iEvent.put(gctIsolaEm, "isoEm");
  iEvent.put(gctNoIsoEm, "nonIsoEm");
  iEvent.put(gctCenJets, "cenJets");
  iEvent.put(gctForJets, "forJets");
  iEvent.put(gctTauJets, "tauJets");
  //iEvent.put(gctEtTotal);

}

void GtPsbTextToDigi::endJob() {
  /// Check BC0 signals consistency
  int nmem = 4;
  bool match = true;
  for(int i=0; i<nmem-1; i++)
    match &= (m_bc0[i]==m_bc0[i+1]);  
  LogDebug("GtPsbTextToDigi") << "[GtPsbTextToDigi::endJob()] ";
  if(!match) 
    LogDebug("GtPsbTextToDigi") << "did not find matching BC0 in all input files: ";
  else
    LogDebug("GtPsbTextToDigi") << "detected common BC0 in all input files: ";
  for(int i=0; i<nmem; i++)
    LogDebug("GtPsbTextToDigi") << " " << m_bc0[i];  
  LogDebug("GtPsbTextToDigi") << std::flush << std::endl;

}
