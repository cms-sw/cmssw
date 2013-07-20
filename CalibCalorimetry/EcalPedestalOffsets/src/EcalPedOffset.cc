//TODO: fix header here?
/**
 * \file EcalPedOffset.cc
 *
 * $Date: 2013/05/30 22:33:07 $
 * $Revision: 1.18 $
 * \author P. Govoni (pietro.govoni@cernNOSPAM.ch) - originally
 * \author S. Cooper (seth.cooper@cernNOSPAM.ch)
 * Last updated: @DATE@ @AUTHOR@
 *
*/

#include <memory>
#include <iostream>
#include <fstream>

#include "CalibCalorimetry/EcalPedestalOffsets/interface/EcalPedOffset.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/RunTag.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonRunIOV.h"
#include "OnlineDB/EcalCondDB/interface/MonPedestalOffsetsDat.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

using namespace cms;
using namespace edm;

//! ctor
EcalPedOffset::EcalPedOffset (const ParameterSet& paramSet) :
  m_barrelDigiCollection (paramSet.getParameter<edm::InputTag> ("EBdigiCollection")),
  m_endcapDigiCollection (paramSet.getParameter<edm::InputTag> ("EEdigiCollection")),
  m_headerCollection (paramSet.getParameter<edm::InputTag> ("headerCollection")),
  m_xmlFile (paramSet.getParameter<std::string> ("xmlFile")),
  m_DACmin (paramSet.getUntrackedParameter<int> ("DACmin",0)),
  m_DACmax (paramSet.getUntrackedParameter<int> ("DACmax",256)),
  m_RMSmax (paramSet.getUntrackedParameter<double> ("RMSmax",2)),
  m_bestPed (paramSet.getUntrackedParameter<int> ("bestPed",200)), 
  m_dbHostName (paramSet.getUntrackedParameter<std::string> ("dbHostName","0")),
  m_dbName (paramSet.getUntrackedParameter<std::string> ("dbName","0")),
  m_dbUserName (paramSet.getUntrackedParameter<std::string> ("dbUserName")),
  m_dbPassword (paramSet.getUntrackedParameter<std::string> ("dbPassword")),
  m_dbHostPort (paramSet.getUntrackedParameter<int> ("dbHostPort",1521)),
  m_create_moniov (paramSet.getUntrackedParameter<bool>("createMonIOV", false)),
  m_location (paramSet.getUntrackedParameter<std::string>("location", "H4")),
  m_run(-1),
  m_plotting (paramSet.getParameter<std::string> ("plotting")),
  m_maxSlopeAllowed_ (paramSet.getUntrackedParameter<double> ("maxSlopeAllowed",-29)),
  m_minSlopeAllowed_ (paramSet.getUntrackedParameter<double> ("minSlopeAllowed",-18)),
  m_maxChi2OverNDFAllowed_ (paramSet.getUntrackedParameter<double> ("maxChi2OverNDF",5))
{
  edm::LogInfo ("EcalPedOffset") << " reading "
    << " m_DACmin: " << m_DACmin
    << " m_DACmax: " << m_DACmax
    << " m_RMSmax: " << m_RMSmax
    << " m_bestPed: " << m_bestPed ;
}


// -----------------------------------------------------------------------------


//! dtor
EcalPedOffset::~EcalPedOffset ()
{
  for (std::map<int,TPedValues*>::iterator mapIt = m_pedValues.begin ();
      mapIt != m_pedValues.end ();
      ++mapIt)
    delete mapIt->second ; 
  for (std::map<int,TPedResult*>::iterator mapIt = m_pedResult.begin ();
      mapIt != m_pedResult.end ();
      ++mapIt)
    delete mapIt->second ; 
}


// -----------------------------------------------------------------------------


//! begin the run
void EcalPedOffset::beginRun (Run const &, EventSetup const& eventSetup)
{
  LogDebug ("EcalPedOffset") << "entering beginRun..." ;

  edm::ESHandle< EcalElectronicsMapping > handle;
  eventSetup.get< EcalMappingRcd >().get(handle);
  ecalElectronicsMap_ = handle.product();

}


// -----------------------------------------------------------------------------


//! perform the analysis
void EcalPedOffset::analyze (Event const& event, 
    EventSetup const& eventSetup) 
{
  LogDebug ("EcalPedOffset") << "entering analyze ...";

  // get the headers
  // (one header for each supermodule)
  edm::Handle<EcalRawDataCollection> DCCHeaders;
  event.getByLabel(m_headerCollection, DCCHeaders);

  std::map <int,int> DACvalues;

  if(m_run==-1)
    m_run = event.id().run();

  // loop over the headers
  for (EcalRawDataCollection::const_iterator headerItr= DCCHeaders->begin();
      headerItr != DCCHeaders->end (); 
      ++headerItr) 
  {
    EcalDCCHeaderBlock::EcalDCCEventSettings settings = headerItr->getEventSettings();
    int FEDid = 600+headerItr->id();
    DACvalues[FEDid] = settings.ped_offset;
    LogDebug("EcalPedOffset") << "Found FED: " << FEDid << " in DCC header";
  }

  bool barrelDigisFound = true;
  bool endcapDigisFound = true;
  // get the barrel digis
  // (one digi for each crystal)
  Handle<EBDigiCollection> barrelDigis;
  event.getByLabel(m_barrelDigiCollection, barrelDigis);
  if(!barrelDigis.isValid())
  {
    edm::LogError ("EcalPedOffset") << "Error! can't get the product " 
      << m_barrelDigiCollection << "; not reading barrel digis";
    barrelDigisFound = false;
  }

  if(barrelDigis->size()==0) 
  {
    edm::LogInfo("EcalPedOffset") << "Size of EBDigiCollection is zero;"
      << " not reading barrel digis";
    barrelDigisFound = false;
  }

  // get the endcap digis
  // (one digi for each crystal)
  Handle<EEDigiCollection> endcapDigis;
  event.getByLabel(m_endcapDigiCollection, endcapDigis);
  if(!endcapDigis.isValid())
  {
    edm::LogError ("EcalPedOffset") << "Error! can't get the product " 
      << m_endcapDigiCollection << "; not reading endcap digis";
    endcapDigisFound = false;
  }

  if(endcapDigis->size()==0) 
  {
    edm::LogInfo("EcalPedOffset") << "Size of EEDigiCollection is zero;"
      << " not reading endcap digis";
    endcapDigisFound = false;
  }

  
  if(barrelDigisFound)
    readDACs(barrelDigis, DACvalues);
  if(endcapDigisFound)
    readDACs(endcapDigis, DACvalues);
  if(!barrelDigisFound && !endcapDigisFound)
    edm::LogError ("EcalPedOffset") << "No digis found in the event!";
  
}


// -----------------------------------------------------------------------------


void EcalPedOffset::readDACs(const edm::Handle<EBDigiCollection>& pDigis,
                             const std::map<int,int>& _DACvalues)
{
  std::map<int,int> DACvalues = _DACvalues;
  // loop over the digis
  for (EBDigiCollection::const_iterator itdigi = pDigis->begin(); 
      itdigi != pDigis->end(); 
      ++itdigi)
  {
    int gainId = ((EBDataFrame)(*itdigi)).sample(0).gainId();
    EBDetId detId = EBDetId(itdigi->id());
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(detId);
    int FEDid = 600+elecId.dccId();
    int crystalId = detId.ic();

    //TODO: Behavior here
    if(DACvalues.find(FEDid)==DACvalues.end())
    {
      edm::LogError("EcalPedOffset")
        << "Error! EB DCCid of digi does not match any DCCid found in DCC headers" << FEDid;
    }

    if (!m_pedValues.count(FEDid))
    {
      LogDebug("EcalPedOffset") << "Inserting new TPedValues object for FED:" << FEDid;
      m_pedValues[FEDid] = new TPedValues(m_RMSmax,m_bestPed);
    }

    // loop over the samples
    for (int iSample = 0; iSample < EcalDataFrame::MAXSAMPLES; ++iSample) 
    {
      m_pedValues[FEDid]->insert(gainId,
          crystalId,
          DACvalues[FEDid],
          ((EBDataFrame)(*itdigi)).sample(iSample).adc(),
          crystalId);
    }
    
  } //end loop over digis
}

// -----------------------------------------------------------------------------


void EcalPedOffset::readDACs(const edm::Handle<EEDigiCollection>& pDigis,
                             const std::map<int,int>& _DACvalues)
{
  std::map<int,int> DACvalues = _DACvalues;
  // loop over the digis
  for (EEDigiCollection::const_iterator itdigi = pDigis->begin(); 
      itdigi != pDigis->end(); 
      ++itdigi)
  {
    int gainId = ((EEDataFrame)(*itdigi)).sample(0).gainId();
    //int gainId = itdigi->sample(0).gainId();
    EEDetId detId = EEDetId(itdigi->id());
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(detId);
    int FEDid = 600+elecId.dccId();
    int crystalId = 25*elecId.towerId()+5*elecId.stripId()+elecId.xtalId();
    int endcapCrystalId = 100*elecId.towerId()+5*(elecId.stripId()-1)+elecId.xtalId();
    
    //TODO: Behavior here
    if(DACvalues.find(FEDid)==DACvalues.end())
    {
      edm::LogError("EcalPedOffset")
        << "Error! EE DCCid of digi does not match any DCCid found in DCC headers: " << FEDid;
    }

    if (!m_pedValues.count(FEDid))
    {
      LogDebug("EcalPedOffset") << "Inserting new TPedValues object for FED:" << FEDid;
      m_pedValues[FEDid] = new TPedValues(m_RMSmax,m_bestPed);
    }
    
    // loop over the samples
    for (int iSample = 0; iSample < EcalDataFrame::MAXSAMPLES; ++iSample) 
    {
      m_pedValues[FEDid]->insert(gainId,
          crystalId,
          DACvalues[FEDid],
          ((EEDataFrame)(*itdigi)).sample(iSample).adc(),
          endcapCrystalId);
    }
    
  } //end loop over digis
}

// -----------------------------------------------------------------------------


//! perform the minimization and write results
void EcalPedOffset::endJob () 
{
  for (std::map<int,TPedValues*>::const_iterator smPeds = m_pedValues.begin ();
      smPeds != m_pedValues.end (); 
      ++smPeds)
  {
    m_pedResult[smPeds->first] = 
      new TPedResult ((smPeds->second)->terminate (m_DACmin, m_DACmax));
  } 
  edm::LogInfo ("EcalPedOffset") << " results map size " 
    << m_pedResult.size ();
  writeXMLFiles(m_xmlFile);

  if (m_plotting != '0') makePlots ();
  if (m_dbHostName != '0') writeDb ();       
}


// -----------------------------------------------------------------------------


//! write the m_pedResult in the DB
//!FIXME divide into sub-tasks
void EcalPedOffset::writeDb () 
{
  LogDebug ("EcalPedOffset") << " entering writeDb ..." ;

  // connect to the database
  EcalCondDBInterface* DBconnection ;
  try
  {
    LogInfo("EcalPedOffset") << "Opening DB connection with TNS_ADMIN ...";
    DBconnection = new EcalCondDBInterface(m_dbName, m_dbUserName, m_dbPassword);
  } catch (std::runtime_error &e) {
    LogError("EcalPedOffset") << e.what();
    if ( m_dbHostName.size() != 0 )
    {
      try
      {
        LogInfo("EcalPedOffset") << "Opening DB connection without TNS_ADMIN ...";
        DBconnection = new EcalCondDBInterface(m_dbHostName, m_dbName, 
            m_dbUserName, m_dbPassword, m_dbHostPort);
      } catch (std::runtime_error &e) {
        LogError("EcalPedOffset") << e.what();
        return;
      }
    }
    else
      return;
  }

  // define the query for RunIOV to get the right place in the database
  RunTag runtag ;  
  LocationDef locdef ;
  RunTypeDef rundef ;
  locdef.setLocation (m_location);

  runtag.setGeneralTag ("PEDESTAL-OFFSET");
  rundef.setRunType ("PEDESTAL-OFFSET");
  //rundef.setRunType ("TEST");
  //runtag.setGeneralTag ("TEST");

  runtag.setLocationDef (locdef);
  runtag.setRunTypeDef (rundef);


  run_t run = m_run ; //FIXME dal config file
  //RunIOV runiov = DBconnection->fetchRunIOV (&runtag, run);
  RunIOV runiov = DBconnection->fetchRunIOV(m_location, run);
  
  // MonRunIOV
  MonVersionDef monverdef ;  
  monverdef.setMonitoringVersion ("test01");
  MonRunTag montag ;
  montag.setMonVersionDef (monverdef);
  montag.setGeneralTag ("CMSSW");

  subrun_t subrun = 1 ; //hardcoded!

  MonRunIOV moniov ;

  try{
    runtag = runiov.getRunTag();
    moniov = DBconnection->fetchMonRunIOV(&runtag, &montag, run, subrun);
  } 
  catch (std::runtime_error &e) {
    if(m_create_moniov){
      //if not already in the DB create a new MonRunIOV
      Tm startSubRun;
      startSubRun.setToCurrentGMTime();
      // setup the MonIOV
      moniov.setRunIOV(runiov);
      moniov.setSubRunNumber(subrun);
      moniov.setSubRunStart(startSubRun);
      moniov.setMonRunTag(montag);
      LogDebug ("EcalPedOffset") <<" creating a new MonRunIOV" ;
    }
    else{
      edm::LogError ("EcalPedOffset") << " no MonRunIOV existing in the DB" ;
      edm::LogError ("EcalPedOffset") << " the result will not be stored into the DB" ;
      if ( DBconnection ) {delete DBconnection;}
      return;
    }
  }

  // create the table to be filled and the map to be inserted
  EcalLogicID ecid ;
  std::map<EcalLogicID, MonPedestalOffsetsDat> DBdataset ;
  MonPedestalOffsetsDat DBtable ;

  // fill the table

  // loop over the super-modules
  for (std::map<int,TPedResult*>::const_iterator result = m_pedResult.begin ();
      result != m_pedResult.end ();
      ++result)
  {
    // loop over the crystals
    for (int xtal = 0 ; xtal<1700 ; ++xtal)
    {
      DBtable.setDACG1 (result->second->m_DACvalue[2][xtal]);
      DBtable.setDACG6 (result->second->m_DACvalue[1][xtal]);
      DBtable.setDACG12 (result->second->m_DACvalue[0][xtal]);
      DBtable.setTaskStatus (1); //FIXME to be set correctly

      // fill the table
      if ( DBconnection ) 
      {
        try 
        {
          int fedid = result->first;
          int eid = m_pedValues[fedid]->getCrystalNumber(xtal);
          // If eid is zero, that crystal was not present in digis
          if(eid==0)
            continue;

          if (fedid >= 601 && fedid <= 609)
          {
            // Add the FEDid part in for DB
            eid = eid+10000*(fedid-600);
            ecid = DBconnection->getEcalLogicID("EE_elec_crystal_number", eid);
          }
          else if (fedid >= 610 && fedid <= 627)
          {
            ecid = DBconnection->getEcalLogicID("EB_crystal_number", fedid-610+19,
                eid);
          } 
          else if (fedid >= 628 && fedid <= 645)
          {
            ecid = DBconnection->getEcalLogicID("EB_crystal_number", fedid-628+1, 
                eid);
          }
          else if (fedid >= 646 && fedid <= 654)
          {
            // Add the FEDid part in for DB
            eid = eid+10000*(fedid-600);
            ecid = DBconnection->getEcalLogicID("EE_elec_crystal_number", eid);
          }
          else
            LogError("EcalPedOffset") << "FEDid is out of range 601-654";

          DBdataset[ecid] = DBtable ;
        } catch (std::runtime_error &e) {
          edm::LogError ("EcalPedOffset") << e.what();
        }
      }
    } // loop over the crystals
  } // loop over the super-modules

  // insert the map of tables in the database
  if ( DBconnection ) {
    try {
      LogDebug ("EcalPedOffset") << "Inserting dataset ... " << std::flush;
      if ( DBdataset.size() != 0 ) DBconnection->insertDataSet (&DBdataset, &moniov);
      LogDebug ("EcalPedOffset") << "done." ;
    } catch (std::runtime_error &e) {
      edm::LogError ("EcalPedOffset") << e.what ();
    }
  }

  if ( DBconnection ) {delete DBconnection;}
}


// -----------------------------------------------------------------------------


//! write the m_pedResults to XML files
void EcalPedOffset::writeXMLFiles(std::string fileName)
{
  // loop over the super-modules
  for (std::map<int,TPedResult*>::const_iterator smRes = m_pedResult.begin();
      smRes != m_pedResult.end(); 
      ++smRes)
  {
    std::string thisSMFileName = fileName;
    // open the output stream
    thisSMFileName+="_";
    thisSMFileName+=intToString(smRes->first);
    thisSMFileName+=".xml";
    std::ofstream xml_outfile;
    xml_outfile.open(thisSMFileName.c_str());

    // write the header file
    xml_outfile<<"<offsets>"<<std::endl;
    xml_outfile << "<PEDESTAL_OFFSET_RELEASE VERSION_ID = \"SM1_VER1\"> \n";
    xml_outfile << "  <RELEASE_ID>RELEASE_1</RELEASE_ID>\n";
    xml_outfile << "  <SUPERMODULE>";
    xml_outfile << smRes->first;
    xml_outfile << "</SUPERMODULE>\n";
    xml_outfile << "  <TIME_STAMP> 070705 </TIME_STAMP>" << std::endl;

    // loop over the crystals
    for (int xtal = 0 ; xtal < 1700 ; ++xtal) 
    {
      int crystalNumber = m_pedValues[smRes->first]->getCrystalNumber(xtal);
      if(crystalNumber==0)
        continue;
      xml_outfile << "  <PEDESTAL_OFFSET>\n";
      xml_outfile << "    <HIGH>" << ((smRes->second)->m_DACvalue)[0][xtal] << "</HIGH>\n";
      xml_outfile << "    <MED>" << ((smRes->second)->m_DACvalue)[1][xtal] << "</MED>\n";
      xml_outfile << "    <LOW>" << ((smRes->second)->m_DACvalue)[2][xtal] << "</LOW>\n";
      xml_outfile << "    <CRYSTAL> "<< crystalNumber << " </CRYSTAL>\n";
      xml_outfile << "  </PEDESTAL_OFFSET>" << std::endl;            
    } 

    // close the open tags  
    xml_outfile << " </PEDESTAL_OFFSET_RELEASE>" << std::endl;
    xml_outfile << "</offsets>" << std::endl;
    xml_outfile.close ();
  } // loop over the super-modules


}


// -----------------------------------------------------------------------------


//! create the plots of the DAC pedestal trend
void EcalPedOffset::makePlots () 
{
  LogDebug ("EcalPedOffset") << " entering makePlots ..." ;

  edm::LogInfo ("EcalPedOffset") << " map size: " 
    << m_pedValues.size();

  // create the ROOT file
  m_plotting+=".root";

  TFile * rootFile = new TFile(m_plotting.c_str(),"RECREATE");

  // loop over the supermodules
  for (std::map<int,TPedValues*>::const_iterator smPeds = m_pedValues.begin();
      smPeds != m_pedValues.end(); 
      ++smPeds)
  {
    // make a folder in the ROOT file
    char folderName[120] ;
    sprintf (folderName,"FED%02d",smPeds->first);
    rootFile->mkdir(folderName);
    smPeds->second->makePlots(rootFile,folderName,m_maxSlopeAllowed_,
        m_minSlopeAllowed_,m_maxChi2OverNDFAllowed_);
  }

  rootFile->Close();
  delete rootFile;

  LogDebug ("EcalPedOffset") << " DONE"; 
}

// -----------------------------------------------------------------------------

// convert an int to a string
std::string EcalPedOffset::intToString(int num)
{

  // outputs the number into the string stream and then flushes
  // the buffer (makes sure the output is put into the stream)
  std::ostringstream myStream;
  myStream << num << std::flush;
  return(myStream.str()); //returns the string form of the stringstream object
}
