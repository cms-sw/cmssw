/**
 * module dumping TGraph with 50 data frames from Pn Diodes
 *   
 * 
 * $Date: 2011/10/10 09:05:21 $
 * $Revision: 1.4 $
 * \author K. Kaadze
 * \author G. Franzoni 
 *
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>

#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h>
#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h>

#include "CaloOnlineTools/EcalTools/plugins/EcalPnGraphs.h"
#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"

#include <iostream>
#include <vector>
#include <map>

#include "TFile.h"
#include "TGraph.h"


//=============================================================================
EcalPnGraphs::EcalPnGraphs(const edm::ParameterSet& ps){  
//=============================================================================

  digiProducer_     = ps.getParameter<std::string>("digiProducer");
  fileName = ps.getUntrackedParameter<std::string >("fileName", std::string("toto") );

  first_Pn = 0;

  listPns = ps.getUntrackedParameter<std::vector<int> >("listPns", std::vector<int>());
  numPn   = ps.getUntrackedParameter< int >("numPn");

  std::vector<int> listDefaults;
  listDefaults.push_back(-1);
  feds_ = ps.getUntrackedParameter<std::vector<int> > ("requestedFeds",listDefaults);
  bool fedIsGiven = false;

  std::vector<std::string> ebDefaults;
  ebDefaults.push_back("none");
  ebs_  = ps.getUntrackedParameter<std::vector<std::string> >("requestedEbs",ebDefaults);

  //FEDs and EBs
  if ( feds_[0] != -1 ) {
    edm::LogInfo("EcalPnGraphs") << "FED id is given! Goining to beginJob! ";
    fedIsGiven = true;
  }else {
    feds_.clear();
    if ( ebs_[0] !="none" ) {
      //EB id is given and convert to FED id
      fedMap = new EcalFedMap(); 
      for (std::vector<std::string>::const_iterator ebItr = ebs_.begin(); 
	   ebItr!= ebs_.end();  ++ebItr) {
	feds_.push_back(fedMap->getFedFromSlice(*ebItr));
      }
      delete fedMap;
    } else {
      //Select all FEDs in the Event
      for ( int i=601; i<655; ++i){
	feds_.push_back(i);
      }
    }
  }
  
  // consistency checks checks
  inputIsOk       = true;
  //check with FEDs
  if ( fedIsGiven ) {
    std::vector<int>::iterator fedIter;
    for (fedIter = feds_.begin(); fedIter!=feds_.end(); ++fedIter) {
      if ( (*fedIter) < 601 || (*fedIter) > 654) {                      
	std::cout << "[EcalPnGraphs] pn number : " << (*fedIter) << " found in listFeds. "
		  << " Valid range is [601-654]. Returning." << std::endl;
	inputIsOk = false;
	return;
      }
    }
  }

  //Check with Pns
  if ( listPns[0] != -1 ) { 
    std::vector<int>::iterator intIter;
    for (intIter = listPns.begin(); intIter != listPns.end(); intIter++)  {  
      if ( ((*intIter) < 1) ||  (10 < (*intIter)) )       {  
	std::cout << "[EcalPnGraphs] pn number : " << (*intIter) << " found in listPns. "
		  << " Valid range is 1-10. Returning." << std::endl;
	inputIsOk = false;
	return;
      }
      if (!first_Pn )   first_Pn = (*intIter);	
    }
  } else {
    listPns.clear();
    listPns.push_back(5); 
    listPns.push_back(6); 
  }
  
  // setting the abcissa array once for all
  for (int i=0; i<50; i++)        abscissa[i] = i;
  
  // local event counter (in general different from LV1)
  eventCounter =0;
}


//=============================================================================
EcalPnGraphs::~EcalPnGraphs(){  
//=============================================================================
  //delete *;
}

//=============================================================================
void EcalPnGraphs::beginJob() {
//=============================================================================
  edm::LogInfo("EcalPhGraphs") << "entering beginJob! " ;
}

//=============================================================================
void EcalPnGraphs::analyze( const edm::Event & e, const  edm::EventSetup& c){
//=============================================================================

  eventCounter++;
  if (!inputIsOk) return;

  // retrieving crystal PN diodes from Event
  edm::Handle<EcalPnDiodeDigiCollection>  pn_digis;
  try {
    e.getByLabel(digiProducer_, pn_digis);
  } catch (cms::Exception& ex) {
    edm::LogError("EcalPnGraphs") << "PNs were not found!";
  }

  // getting the list of all the Pns which will be dumped on TGraph
  // - listPns is the list as given by the user
  // -numPn is the number of Pns (centered at Pn from listPns) for which graphs are required
  std::vector<int>::iterator pn_it;
  for ( pn_it = listPns.begin();  pn_it != listPns.end() ; pn_it++  )
    {
      int ipn    = (*pn_it);
      int hpn = numPn;
      
      for (int u = (-hpn) ; u<=hpn; u++){	  
	  int ipn_c = ipn + u;
	  if (ipn_c < 1 || ipn_c > 10) continue;
	  std::vector<int>::iterator notInList = find(listAllPns.begin(), listAllPns.end(), ipn_c);
	  if ( notInList == listAllPns.end() ) {
	    listAllPns.push_back ( ipn_c );
	  }
      }
    }
  
  //Loop over PN digis
  for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pn_digis->begin(); pnItr != pn_digis->end(); ++pnItr )  {
    //Get PNid of a digi
    int ipn = (*pnItr).id().iPnId();
    //Get DCC id where the digi is from
    int ieb    = EcalPnDiodeDetId((*pnItr).id()).iDCCId();
    
    //Make sure that these are PnDigis from the requested FEDid
    int FEDid = ieb + 600;

    std::vector<int>::iterator fedIter = find(feds_.begin(), feds_.end(), FEDid);

    if ( fedIter == feds_.end() ) {
      edm::LogWarning("EcalPnGraphs")<< "For Event " << eventCounter << " PnDigis are not found from requested SM!. returning...";
      return;
    }
    // selecting desired Pns only
    std::vector<int>::iterator iPnIter;
    iPnIter     = find( listAllPns.begin() , listAllPns.end() , ipn);
    if (iPnIter == listAllPns.end()) continue; 
    
    for ( int i=0; i< (*pnItr).size() && i<50; ++i ) {
      ordinate[i] = (*pnItr).sample(i).adc();
    }
    //make grapn of ph digis
    TGraph oneGraph(50, abscissa,ordinate);
    std::string title;
    title = "Graph_ev" + intToString( eventCounter )
          + "_FED" + intToString( FEDid )
          + "_ipn" + intToString( ipn );
    oneGraph.SetTitle(title.c_str());
    oneGraph.SetName(title.c_str());
    graphs.push_back(oneGraph);
    
  }// loop over Pn digis
}

std::string EcalPnGraphs::intToString(int num)
{
  //
  // outputs the number into the string stream and then flushes
  // the buffer (makes sure the output is put into the stream)
  //
  std::ostringstream myStream; //creates an ostringstream object
  myStream << num << std::flush;
  
  return(myStream.str()); //returns the string form of the stringstream object
} 

//=============================================================================
void EcalPnGraphs::endJob() {
//=============================================================================
  fileName +=  ( std::string("_Pn")    + intToString(first_Pn) ); 
  fileName += ".graph.root";

  root_file = new TFile( fileName.c_str() , "RECREATE" );
  std::vector<TGraph>::iterator gr_it;
  for ( gr_it = graphs.begin(); gr_it !=  graphs.end(); gr_it++ )      (*gr_it).Write();
  root_file->Close();

  edm::LogInfo("EcalPnGraphs") << "DONE!.... " ;
}

