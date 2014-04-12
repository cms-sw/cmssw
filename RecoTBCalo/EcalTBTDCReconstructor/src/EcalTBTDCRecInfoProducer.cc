#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBTDCRecInfoProducer.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRawInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
    
EcalTBTDCRecInfoProducer::EcalTBTDCRecInfoProducer(edm::ParameterSet const& ps)
{
  rawInfoCollection_ = ps.getParameter<std::string>("rawInfoCollection");
  rawInfoProducer_   = ps.getParameter<std::string>("rawInfoProducer");
  eventHeaderCollection_ = ps.getParameter<std::string>("eventHeaderCollection");
  eventHeaderProducer_   = ps.getParameter<std::string>("eventHeaderProducer");
  recInfoCollection_        = ps.getParameter<std::string>("recInfoCollection");

//   std::vector<double> planeShift_def;
//   planeShift_def.push_back( -0.333 );
//   planeShift_def.push_back( -0.333 );
//   planeShift_def.push_back( -0.333 );
//   planeShift_def.push_back( -0.333 );


//   std::vector<double> zPosition_def;
//   zPosition_def.push_back( -0.333 );
//   zPosition_def.push_back( -0.333 );
//   zPosition_def.push_back( -0.333 );
//   zPosition_def.push_back( -0.333 );
  std::vector<EcalTBTDCRecInfoAlgo::EcalTBTDCRanges> tdcRanges;

  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters ranges=ps.getParameter<Parameters>("tdcRanges");
  for(Parameters::iterator itRanges = ranges.begin(); itRanges != ranges.end(); ++itRanges) 
    {
      EcalTBTDCRecInfoAlgo::EcalTBTDCRanges aRange;
      aRange.runRanges.first = itRanges->getParameter<int>("startRun");
      aRange.runRanges.second = itRanges->getParameter<int>("endRun");
      aRange.tdcMin = itRanges->getParameter< std::vector<double> >("tdcMin");
      aRange.tdcMax = itRanges->getParameter< std::vector<double> >("tdcMax");
      tdcRanges.push_back(aRange);
    }
  
  use2004OffsetConvention_ = ps.getUntrackedParameter< bool >("use2004OffsetConvention",false);

  produces<EcalTBTDCRecInfo>(recInfoCollection_);

  algo_ = new EcalTBTDCRecInfoAlgo(tdcRanges);
}

EcalTBTDCRecInfoProducer::~EcalTBTDCRecInfoProducer() {
  if (algo_)
    delete algo_;
}

void EcalTBTDCRecInfoProducer::produce(edm::Event& e, const edm::EventSetup& es)
{
  // Get input
   edm::Handle<EcalTBTDCRawInfo> ecalRawTDC;  
   const EcalTBTDCRawInfo* ecalTDCRawInfo = 0;

   //evt.getByLabel( digiProducer_, digiCollection_, pDigis);
   e.getByLabel( rawInfoProducer_, ecalRawTDC);
   if (ecalRawTDC.isValid()) {
     ecalTDCRawInfo = ecalRawTDC.product();
   }

   if (! ecalTDCRawInfo )
     {
       edm::LogError("EcalTBTDCRecInfoError") << "Error! can't get the product " << rawInfoCollection_.c_str() ;
       return;
     }

   if ( (*ecalTDCRawInfo).size() < 1 )
     { 
       edm::LogError("EcalTBTDcRecInfoError") << "Less than one TDC good channel found. Aborting" << rawInfoCollection_.c_str() ;
       return;
     }
   // Get input
   edm::Handle<EcalTBEventHeader> tbEventHeader;  
   const EcalTBEventHeader* ecalEventHeader = 0;
   //evt.getByLabel( digiProducer_, digiCollection_, pDigis);
   e.getByLabel( eventHeaderProducer_, tbEventHeader);
   if (tbEventHeader.isValid()) {
     ecalEventHeader = tbEventHeader.product();
   }
   
   if (! ecalEventHeader )
     {
       edm::LogError("EcalTBTDCRecInfoError") << "Error! can't get the product " << eventHeaderCollection_.c_str();
       return;
     }

  // Create empty output
  std::auto_ptr<EcalTBTDCRecInfo> recInfo(new EcalTBTDCRecInfo(algo_->reconstruct(*ecalRawTDC,*tbEventHeader,use2004OffsetConvention_)));
  
  e.put(recInfo,recInfoCollection_);
} 


