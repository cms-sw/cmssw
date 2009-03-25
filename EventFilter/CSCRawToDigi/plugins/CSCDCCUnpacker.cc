#include "EventFilter/CSCRawToDigi/interface/CSCDCCUnpacker.h"

//Framework stuff
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
//FEDRawData
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"


//Digi stuff
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCFEBStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"

#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCFEBStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDMBStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCDMBStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCTMBStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCTMBStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDDUStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCDDUStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDCCStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCDCCStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigiCollection.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCLCTData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCRPCData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"
#include "CondFormats/CSCObjects/interface/CSCCrateMap.h"
#include "CondFormats/DataRecord/interface/CSCCrateMapRcd.h"
#include <EventFilter/CSCRawToDigi/interface/CSCMonitorInterface.h>
#include "FWCore/ServiceRegistry/interface/Service.h"



#include <iostream>
#include <sstream>
#include <string>

CSCDCCUnpacker::CSCDCCUnpacker(const edm::ParameterSet & pset) :
  numOfEvents(0) {
  printEventNumber = pset.getUntrackedParameter<bool>("PrintEventNumber", true);
  debug = pset.getUntrackedParameter<bool>("Debug", false);
  useExaminer = pset.getUntrackedParameter<bool>("UseExaminer", true);
  examinerMask = pset.getUntrackedParameter<unsigned int>("ExaminerMask",0x7FB7BF6);
  instatiateDQM = pset.getUntrackedParameter<bool>("runDQM", false);
  errorMask = pset.getUntrackedParameter<unsigned int>("ErrorMask",0xDFCFEFFF);
  unpackStatusDigis = pset.getUntrackedParameter<bool>("UnpackStatusDigis", false);
  inputObjectsTag = pset.getParameter<edm::InputTag>("InputObjects");
  unpackMTCCData = pset.getUntrackedParameter<bool>("isMTCCData", false);

  // Enable Format Status Digis
  useFormatStatus = pset.getUntrackedParameter<bool>("UseFormatStatus", false);

  // Selective unpacking mode will skip only troublesome CSC blocks and not whole DCC/DDU block
  useSelectiveUnpacking = pset.getUntrackedParameter<bool>("UseSelectiveUnpacking", false);

  if(instatiateDQM)  {
    monitor = edm::Service<CSCMonitorInterface>().operator->();
  }
  
  produces<CSCWireDigiCollection>("MuonCSCWireDigi");
  produces<CSCStripDigiCollection>("MuonCSCStripDigi");
  produces<CSCComparatorDigiCollection>("MuonCSCComparatorDigi");
  produces<CSCALCTDigiCollection>("MuonCSCALCTDigi");
  produces<CSCCLCTDigiCollection>("MuonCSCCLCTDigi");
  produces<CSCRPCDigiCollection>("MuonCSCRPCDigi");
  produces<CSCCorrelatedLCTDigiCollection>("MuonCSCCorrelatedLCTDigi");
  
  if (unpackStatusDigis)  {
    produces<CSCCFEBStatusDigiCollection>("MuonCSCCFEBStatusDigi");
    produces<CSCTMBStatusDigiCollection>("MuonCSCTMBStatusDigi");
    produces<CSCDMBStatusDigiCollection>("MuonCSCDMBStatusDigi");
    produces<CSCALCTStatusDigiCollection>("MuonCSCALCTStatusDigi");
    produces<CSCDDUStatusDigiCollection>("MuonCSCDDUStatusDigi");
    produces<CSCDCCStatusDigiCollection>("MuonCSCDCCStatusDigi");
    produces<CSCDCCFormatStatusDigiCollection>("MuonCSCDCCFormatStatusDigi");
  }
  //CSCAnodeData::setDebug(debug);
  CSCALCTHeader::setDebug(debug);
  CSCCLCTData::setDebug(debug);
  CSCEventData::setDebug(debug);
  CSCTMBData::setDebug(debug);
  CSCDCCEventData::setDebug(debug);
  CSCDDUEventData::setDebug(debug);
  CSCTMBHeader::setDebug(debug);
  CSCRPCData::setDebug(debug);
  CSCDDUEventData::setErrorMask(errorMask);
  
}

CSCDCCUnpacker::~CSCDCCUnpacker(){ 
  //fill destructor here
}


void CSCDCCUnpacker::produce(edm::Event & e, const edm::EventSetup& c){

  ///access database for mapping
  // Do we really have to do this every event???
  // ... Yes, because framework is more efficient than you are at caching :)
  // (But if you want to actually DO something specific WHEN the mapping changes, check out ESWatcher)
  edm::ESHandle<CSCCrateMap> hcrate;
  c.get<CSCCrateMapRcd>().get(hcrate); 
  const CSCCrateMap* pcrate = hcrate.product();


  if (printEventNumber) ++numOfEvents;

  /// Get a handle to the FED data collection
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByLabel(inputObjectsTag, rawdata);
    
  /// create the collections of CSC digis
  std::auto_ptr<CSCWireDigiCollection> wireProduct(new CSCWireDigiCollection);
  std::auto_ptr<CSCStripDigiCollection> stripProduct(new CSCStripDigiCollection);
  std::auto_ptr<CSCALCTDigiCollection> alctProduct(new CSCALCTDigiCollection);
  std::auto_ptr<CSCCLCTDigiCollection> clctProduct(new CSCCLCTDigiCollection);
  std::auto_ptr<CSCComparatorDigiCollection> comparatorProduct(new CSCComparatorDigiCollection);
  std::auto_ptr<CSCRPCDigiCollection> rpcProduct(new CSCRPCDigiCollection);
  std::auto_ptr<CSCCorrelatedLCTDigiCollection> corrlctProduct(new CSCCorrelatedLCTDigiCollection);
  std::auto_ptr<CSCCFEBStatusDigiCollection> cfebStatusProduct(new CSCCFEBStatusDigiCollection);
  std::auto_ptr<CSCDMBStatusDigiCollection> dmbStatusProduct(new CSCDMBStatusDigiCollection);
  std::auto_ptr<CSCTMBStatusDigiCollection> tmbStatusProduct(new CSCTMBStatusDigiCollection);
  std::auto_ptr<CSCDDUStatusDigiCollection> dduStatusProduct(new CSCDDUStatusDigiCollection);
  std::auto_ptr<CSCDCCStatusDigiCollection> dccStatusProduct(new CSCDCCStatusDigiCollection);
  std::auto_ptr<CSCALCTStatusDigiCollection> alctStatusProduct(new CSCALCTStatusDigiCollection);

  std::auto_ptr<CSCDCCFormatStatusDigiCollection> formatStatusProduct(new CSCDCCFormatStatusDigiCollection);
 

  // If set selective unpacking mode 
  // hardcoded examiner mask below to check for DCC and DDU level errors will be used first
  // then examinerMask for CSC level errors will be used during unpacking of each CSC block
  unsigned long dccBinCheckMask = 0x06080016;

  for (int id=FEDNumbering::MINCSCFEDID;
       id<=FEDNumbering::MAXCSCFEDID; ++id) { // loop over DCCs
    /// uncomment this for regional unpacking
    /// if (id!=SOME_ID) continue;
    
    /// Take a reference to this FED's data
    const FEDRawData& fedData = rawdata->FEDData(id);
    unsigned long length =  fedData.size();
    
    if (length>=32){ ///if fed has data then unpack it
      CSCDCCExaminer* examiner = NULL;
      std::stringstream examiner_out, examiner_err;
      goodEvent = true;
      if (useExaminer) {///examine event for integrity
	// CSCDCCExaminer examiner;
	examiner = new CSCDCCExaminer();
	examiner->output1().redirect(examiner_out);
	examiner->output2().redirect(examiner_err);
	if( examinerMask&0x40000 ) examiner->crcCFEB(1);
	if( examinerMask&0x8000  ) examiner->crcTMB (1);
	if( examinerMask&0x0400  ) examiner->crcALCT(1);
	examiner->output1().show();
	examiner->output2().show();
	examiner->setMask(examinerMask);
	const short unsigned int *data = (short unsigned int *)fedData.data();

	/*short unsigned * buf = (short unsigned int *)fedData.data();
	  std::cout <<std::endl<<length/2<<" words of data:"<<std::endl;
	  for (int i=0;i<length/2;i++) {
	  printf("%04x %04x %04x %04x\n",buf[i+3],buf[i+2],buf[i+1],buf[i]);
	  i+=3;
	  }*/



	if( examiner->check(data,long(fedData.size()/2)) < 0 )	{
	  goodEvent=false;
	} 
	else {	  
	  if (useSelectiveUnpacking)  goodEvent=!(examiner->errors()&dccBinCheckMask);
	  else goodEvent=!(examiner->errors()&examinerMask);
	}

	// Fill Format status digis per FED
	// Remove examiner->errors() != 0 check if we need to put status digis for every event
	if (useFormatStatus && (examiner->errors() !=0))
	  // formatStatusProduct->insertDigi(CSCDetId(1,1,1,1,1), CSCDCCFormatStatusDigi(id,examiner,dccBinCheckMask));
	  formatStatusProduct->insertDigi(CSCDetId(1,1,1,1,1), 
		CSCDCCFormatStatusDigi(id,dccBinCheckMask,
			examiner->getMask(), 
			examiner->errors(),
			examiner->errorsDetailedDDU(),
			examiner->errorsDetailed(),
			examiner->payloadDetailed(),
			examiner->statusDetailed()));
      }
	  
      
      if (goodEvent) {
	///get a pointer to data and pass it to constructor for unpacking

	CSCDCCExaminer * ptrExaminer = examiner;
	if (!useSelectiveUnpacking) ptrExaminer = NULL;
		
  	CSCDCCEventData dccData((short unsigned int *) fedData.data(),ptrExaminer);
	
	if(instatiateDQM) monitor->process(examiner, &dccData);
	
	///get a reference to dduData
	const std::vector<CSCDDUEventData> & dduData = dccData.dduData();
	
	/// set default detid to that for E=+z, S=1, R=1, C=1, L=1
	CSCDetId layer(1, 1, 1, 1, 1);
	
	if (unpackStatusDigis) 
	  dccStatusProduct->insertDigi(layer, CSCDCCStatusDigi(dccData.dccHeader().data(),
							       dccData.dccTrailer().data(),
							       examiner->errors()));

	for (unsigned int iDDU=0; iDDU<dduData.size(); ++iDDU) {  // loop over DDUs
	  /// skip the DDU if its data has serious errors
	  /// define a mask for serious errors 
	  if (dduData[iDDU].trailer().errorstat()&errorMask) {
	    LogTrace("CSCDCCUnpacker|CSCRawToDigi") << "DDU# " << iDDU << " has serious error - no digis unpacked! " <<
	      std::hex << dduData[iDDU].trailer().errorstat();
	    continue; // to next iteration of DDU loop
	  }
		  
	  if (unpackStatusDigis) dduStatusProduct->
				   insertDigi(layer, CSCDDUStatusDigi(dduData[iDDU].header().data(),
								      dduData[iDDU].trailer().data()));
	  
	  ///get a reference to chamber data
	  const std::vector<CSCEventData> & cscData = dduData[iDDU].cscData();


	  for (unsigned int iCSC=0; iCSC<cscData.size(); ++iCSC) { // loop over CSCs

	    ///first process chamber-wide digis such as LCT
	    int vmecrate = cscData[iCSC].dmbHeader()->crateID();
	    int dmb = cscData[iCSC].dmbHeader()->dmbID();

	    ///adjust crate numbers for MTCC data
	    if (unpackMTCCData)
	      switch (vmecrate) {
	      case 0:
		vmecrate=23;
		break;
	      case 1:
		vmecrate=17;
		break;
	      case 2:
		vmecrate=11;
		break;
	      case 3:
		vmecrate=10;
		break;
	      default:
		break;
	      }
		      
	    int icfeb = 0;  /// default value for all digis not related to cfebs
	    int ilayer = 0; /// layer=0 flags entire chamber

	    if (debug)
              LogTrace ("CSCDCCUnpacker|CSCRawToDigi") << "crate = " << vmecrate << "; dmb = " << dmb;
	    
	    if ((vmecrate>=1)&&(vmecrate<=60) && (dmb>=1)&&(dmb<=10)&&(dmb!=6)) {
	      layer = pcrate->detId(vmecrate, dmb,icfeb,ilayer );
	    } 
	    else{
	      LogTrace ("CSCDCCUnpacker|CSCRawToDigi") << " detID input out of range!!! ";
              LogTrace ("CSCDCCUnpacker|CSCRawToDigi")
                << " skipping chamber vme= " << vmecrate << " dmb= " << dmb;
	      continue; // to next iteration of iCSC loop
	    }


	    /// check alct data integrity 
	    int nalct = cscData[iCSC].dmbHeader()->nalct();
	    bool goodALCT=false;
	    //if (nalct&&(cscData[iCSC].dataPresent>>6&0x1)==1) {
	    if (nalct&&cscData[iCSC].alctHeader()) {  
	      if (cscData[iCSC].alctHeader()->check()){
		goodALCT=true;
	      }
	      else {
		LogTrace ("CSCDCCUnpacker|CSCRawToDigi") <<
		  "not storing ALCT digis; alct is bad or not present";
	      }
	    } else {
	      if (debug) LogTrace ("CSCDCCUnpacker|CSCRawToDigi") << "nALCT==0 !!!";
	    }

	    /// fill alct digi
	    if (goodALCT){
	      std::vector <CSCALCTDigi>  alctDigis =
		cscData[iCSC].alctHeader()->ALCTDigis();
	      
	      ///ugly kludge to fix wiregroup numbering in MTCC data
	      if ( unpackMTCCData && (((layer.ring()==3)&&(layer.station()==1))||
				      ((layer.ring()==1)&&(layer.station()==3))||
				      ((layer.ring()==1)&&(layer.station()==4)))){
		for (int unsigned i=0; i<alctDigis.size(); ++i) {
		  if (alctDigis[i].isValid()) {
		    int wiregroup = alctDigis[i].getKeyWG();
		    if (wiregroup < 16) LogTrace("CSCDCCUnpacker|CSCRawToDigi")
		      << "ALCT digi: wire group " << wiregroup
		      << " is out of range!" << "vme ="
		      << vmecrate <<"  dmb=" <<dmb <<" "<<layer;
		    else {
		      wiregroup -= 16; /// adjust by 16
		      alctDigis[i].setWireGroup(wiregroup);
		    }
		  }
		}
	      }
	      alctProduct->put(std::make_pair(alctDigis.begin(), alctDigis.end()),layer);
	    }
		    
		  
	    ///check tmb data integrity
	    int nclct = cscData[iCSC].dmbHeader()->nclct();
	    bool goodTMB=false;
	    //	    if (nclct&&(cscData[iCSC].dataPresent>>5&0x1)==1) {
	    if (nclct&&cscData[iCSC].tmbData()) {
	      if (cscData[iCSC].tmbHeader()->check()){
		if (cscData[iCSC].clctData()->check()) goodTMB=true; 
	      }
	      else {
		LogTrace ("CSCDCCUnpacker|CSCRawToDigi") <<
		  "one of TMB checks failed! not storing TMB digis ";
	      }
	    }
	    else {
	      if (debug) LogTrace ("CSCDCCUnpacker|CSCRawToDigi") << "nCLCT==0 !!!";
	    }
		      
	    /// fill correlatedlct and clct digis
	    if (goodTMB) { 
	      std::vector <CSCCorrelatedLCTDigi>  correlatedlctDigis =
		cscData[iCSC].tmbHeader()->CorrelatedLCTDigis(layer.rawId());
	      
	      ///ugly kludge to fix wiregroup numbering in MTCC data
	      if ( unpackMTCCData && (((layer.ring()==3)&&(layer.station()==1))||
				      ((layer.ring()==1)&&(layer.station()==3))||
				      ((layer.ring()==1)&&(layer.station()==4)))){
		for (int unsigned i=0; i<correlatedlctDigis.size(); ++i) {
		  if (correlatedlctDigis[i].isValid()) {
		    int wiregroup = correlatedlctDigis[i].getKeyWG();
		    if (wiregroup < 16) LogTrace("CSCDCCUnpacker|CSCRawToDigi")
		      << "CorrelatedLCT digi: wire group " << wiregroup
		      << " is out of range!  vme ="<<vmecrate <<"  dmb=" <<dmb
		      <<"  "<<layer;
		    else {
		      wiregroup -= 16; /// adjust by 16
		      correlatedlctDigis[i].setWireGroup(wiregroup);
		    }
		  }
		}
	      }
	      corrlctProduct->put(std::make_pair(correlatedlctDigis.begin(),
						 correlatedlctDigis.end()),layer);

		      
	      std::vector <CSCCLCTDigi>  clctDigis =
		cscData[iCSC].tmbHeader()->CLCTDigis(layer.rawId());
	      clctProduct->put(std::make_pair(clctDigis.begin(), clctDigis.end()),layer);
	      
	      /// fill cscrpc digi
	      if (cscData[iCSC].tmbData()->checkSize()) {
		if (cscData[iCSC].tmbData()->hasRPC()) {
		  std::vector <CSCRPCDigi>  rpcDigis =
		    cscData[iCSC].tmbData()->rpcData()->digis();
		  rpcProduct->put(std::make_pair(rpcDigis.begin(), rpcDigis.end()),layer);
		}
	      } 
	      else LogTrace("CSCDCCUnpacker|CSCRawToDigi") <<" TMBData check size failed!";
	    }
		    
		  
	    /// fill cfeb status digi
	    if (unpackStatusDigis) {
	      for ( icfeb = 0; icfeb < 5; ++icfeb ) {///loop over status digis
		if ( cscData[iCSC].cfebData(icfeb) != NULL )
		  cfebStatusProduct->
		    insertDigi(layer, cscData[iCSC].cfebData(icfeb)->statusDigi());
	      }
	      /// fill dmb status digi
	      dmbStatusProduct->insertDigi(layer, CSCDMBStatusDigi(cscData[iCSC].dmbHeader()->data(),
								   cscData[iCSC].dmbTrailer()->data()));
	      if (goodTMB)  tmbStatusProduct->
			      insertDigi(layer, CSCTMBStatusDigi(cscData[iCSC].tmbHeader()->data(),
								 cscData[iCSC].tmbData()->tmbTrailer()->data()));
	      if (goodALCT) alctStatusProduct->
			      insertDigi(layer, CSCALCTStatusDigi(cscData[iCSC].alctHeader()->data(),
								  cscData[iCSC].alctTrailer()->data()));
	    }
		

	    /// fill wire, strip and comparator digis...
	    for (int ilayer = 1; ilayer <= 6; ++ilayer) {
	      /// set layer, dmb and vme are valid because already checked in line 240
	      // (You have to be kidding. Line 240 in whose universe?)

	      // Allocate all ME1/1 wire digis to ring 1
	      layer = pcrate->detId( vmecrate, dmb, 0, ilayer );

	      std::vector <CSCWireDigi> wireDigis =  cscData[iCSC].wireDigis(ilayer);
	      
	      ///ugly kludge to fix wire group numbers for ME3/1, ME4/1 and ME1/3 chambers in MTCC data
	      if ( unpackMTCCData && (((layer.ring()==3)&&(layer.station()==1))||
				      ((layer.ring()==1)&&(layer.station()==3))||
				      ((layer.ring()==1)&&(layer.station()==4)))){
		for (int unsigned i=0; i<wireDigis.size(); ++i) {
		  int wiregroup = wireDigis[i].getWireGroup();
		  if (wiregroup <= 16) LogTrace("CSCDCCUnpacker|CSCRawToDigi")
		    << "Wire digi: wire group " << wiregroup
		    << " is out of range!  vme ="<<vmecrate 
		    <<"  dmb=" <<dmb <<"  "<<layer;
		  else {
		    wiregroup -= 16; /// adjust by 16
		    wireDigis[i].setWireGroup(wiregroup);
		  }
		}
	      }
	      wireProduct->put(std::make_pair(wireDigis.begin(), wireDigis.end()),layer);
	      
	      for ( icfeb = 0; icfeb < 5; ++icfeb ) {
		layer = pcrate->detId( vmecrate, dmb, icfeb,ilayer );
		if (cscData[iCSC].cfebData(icfeb)) {
		  std::vector<CSCStripDigi> stripDigis;
		  cscData[iCSC].cfebData(icfeb)->digis(layer.rawId(),stripDigis);
		  stripProduct->put(std::make_pair(stripDigis.begin(), 
						   stripDigis.end()),layer);
		}
			      
		if (goodTMB){
		  std::vector <CSCComparatorDigi>  comparatorDigis =
		    cscData[iCSC].clctData()->comparatorDigis(layer.rawId(), icfeb);
		  // Set cfeb=0, so that ME1/a and ME1/b comparators go to
		  // ring 1.
		  layer = pcrate->detId( vmecrate, dmb, 0, ilayer );
		  comparatorProduct->put(std::make_pair(comparatorDigis.begin(), 
							comparatorDigis.end()),layer);
		}
	      } // end of loop over cfebs
	    } // end of loop over layers
	  } // end of loop over chambers
	} // endof loop over DDUs
      } // end of good event
      else   {
        LogTrace("CSCDCCUnpacker|CSCRawToDigi") <<
          "ERROR! Examiner rejected FED #" << id;
        if (examiner) {
          for (int i=0; i<examiner->nERRORS; ++i) {
            if (((examinerMask&examiner->errors())>>i)&0x1)
              LogTrace("CSCDCCUnpacker|CSCRawToDigi")<<examiner->errName(i);
          }
          if (debug) {
            LogTrace("CSCDCCUnpacker|CSCRawToDigi")
              << " Examiner errors:0x" << std::hex << examiner->errors()
              << " & 0x" << examinerMask
              << " = " << (examiner->errors()&examinerMask);
            if (examinerMask&examiner->errors()) {
              LogTrace("CSCDCCUnpacker|CSCRawToDigi")
                << "Examiner output: " << examiner_out.str();
              LogTrace("CSCDCCUnpacker|CSCRawToDigi")
                << "Examiner errors: " << examiner_err.str();
            }
          }
        }

	// dccStatusProduct->insertDigi(CSCDetId(1,1,1,1,1), CSCDCCStatusDigi(examiner->errors()));
	// if(instatiateDQM)  monitor->process(examiner, NULL);
      }
      if (examiner!=NULL) delete examiner;
    } // end of if fed has data
  } // end of loop over DCCs
  // put into the event
  e.put(wireProduct,          "MuonCSCWireDigi");
  e.put(stripProduct,         "MuonCSCStripDigi");
  e.put(alctProduct,          "MuonCSCALCTDigi");
  e.put(clctProduct,          "MuonCSCCLCTDigi");
  e.put(comparatorProduct,    "MuonCSCComparatorDigi");
  e.put(rpcProduct,           "MuonCSCRPCDigi");
  e.put(corrlctProduct,       "MuonCSCCorrelatedLCTDigi");

  if (useFormatStatus)  e.put(formatStatusProduct,    "MuonCSCDCCFormatStatusDigi");

  if (unpackStatusDigis)
    {
      e.put(cfebStatusProduct,    "MuonCSCCFEBStatusDigi");
      e.put(dmbStatusProduct,     "MuonCSCDMBStatusDigi");
      e.put(tmbStatusProduct,     "MuonCSCTMBStatusDigi");
      e.put(dduStatusProduct,     "MuonCSCDDUStatusDigi");
      e.put(dccStatusProduct,     "MuonCSCDCCStatusDigi");
      e.put(alctStatusProduct,    "MuonCSCALCTStatusDigi");
    }
  if (printEventNumber) LogTrace("CSCDCCUnpacker|CSCRawToDigi") 
   <<"[CSCDCCUnpacker]: " << numOfEvents << " events processed ";
}




