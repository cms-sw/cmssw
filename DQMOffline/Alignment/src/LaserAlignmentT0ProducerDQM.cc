
#include "DQMOffline/Alignment/interface/LaserAlignmentT0ProducerDQM.h"





///
///
///
LaserAlignmentT0ProducerDQM::LaserAlignmentT0ProducerDQM( const edm::ParameterSet& aConfiguration ) {
  
  theDqmStore = edm::Service<DQMStore>().operator->();
  theConfiguration = aConfiguration;
  FillDetectorId();

}





///
///
///
LaserAlignmentT0ProducerDQM::~LaserAlignmentT0ProducerDQM() {
}





///
///
///
void LaserAlignmentT0ProducerDQM::beginJob() {

  // upper and lower treshold for a profile considered showing a signal
  theLowerAdcThreshold = theConfiguration.getParameter<unsigned int>( "LowerAdcThreshold" );
  theUpperAdcThreshold = theConfiguration.getParameter<unsigned int>( "UpperAdcThreshold" );

  // the list of input digi products from the cfg
  theDigiProducerList = theConfiguration.getParameter<std::vector<edm::ParameterSet> >( "DigiProducerList" );

  std::string folderName = theConfiguration.getParameter<std::string>( "FolderName" ); 
  theDqmStore->setCurrentFolder( folderName );

  std::string nameAndTitle;
  std::stringstream labelBuilder;

  const short nBeams = 8;
  const short nDisks = 9;

  // for the alignment tubes modules:
  // x: 16 modules (5*TEC-, 6*TIB, 6*TOB, 5*TEC+), all from -z to z
  // y: 8 beams
  nameAndTitle = "NumberOfSignals_AlignmentTubes";
  nSignalsAT   = theDqmStore->book2D( nameAndTitle, nameAndTitle, 22, 0, 22, nBeams, 0, nBeams );
  //  nSignalsAT->setAxisTitle( "z-pos", 1 );
  //  nSignalsAT->setAxisTitle( "beam", 2 );

  // create bin labels for the AT histograms (subdets mixed here)
  for( unsigned int i = 1; i <= 5; ++i ) {
    labelBuilder.clear(); labelBuilder.str( "" );
    labelBuilder << "TEC- D" << 5-i; // TEC-
    nSignalsAT->setBinLabel( i, labelBuilder.str(), 1 );
    labelBuilder.clear(); labelBuilder.str( "" );
    labelBuilder << "TEC+ D" << i-1; // TEC+
    nSignalsAT->setBinLabel( 17+i, labelBuilder.str(), 1 );
  }
  for( unsigned int i = 0; i < 6; ++i ) {
    labelBuilder.clear(); labelBuilder.str( "" );
    labelBuilder << "TIB" << i; // TIB
    nSignalsAT->setBinLabel( 6+i, labelBuilder.str(), 1 );
    labelBuilder.clear(); labelBuilder.str( "" );
    labelBuilder << "TOB" << i; // TOB
    nSignalsAT->setBinLabel( 12+i, labelBuilder.str(), 1 );
  }


  // for the tec internal modules:
  // x: disk1...disk9 (from inner to outer, so z changes direction!)
  // y: 8 beams
  nameAndTitle       = "NumberOfSignals_TEC+R4";
  nSignalsTECPlusR4  = theDqmStore->book2D( nameAndTitle, nameAndTitle, nDisks, 0, nDisks, nBeams, 0, nBeams );
  //  nSignalsTECPlusR4->setAxisTitle( "disk", 1 );
  //  nSignalsTECPlusR4->setAxisTitle( "beam", 2 );

  nameAndTitle       = "NumberOfSignals_TEC+R6";
  nSignalsTECPlusR6  = theDqmStore->book2D( nameAndTitle, nameAndTitle, nDisks, 0, nDisks, nBeams, 0, nBeams );
  //  nSignalsTECPlusR6->setAxisTitle( "disk", 1 );
  //  nSignalsTECPlusR6->setAxisTitle( "beam", 2 );

  nameAndTitle       = "NumberOfSignals_TEC-R4";
  nSignalsTECMinusR4 = theDqmStore->book2D( nameAndTitle, nameAndTitle, nDisks, 0, nDisks, nBeams, 0, nBeams );
  //  nSignalsTECMinusR4->setAxisTitle( "disk", 1 );
  //  nSignalsTECMinusR4->setAxisTitle( "beam", 2 );

  nameAndTitle       = "NumberOfSignals_TEC-R6";
  nSignalsTECMinusR6 = theDqmStore->book2D( nameAndTitle, nameAndTitle, nDisks, 0, nDisks, nBeams, 0, nBeams );
  //  nSignalsTECMinusR6->setAxisTitle( "disk", 1 );
  //  nSignalsTECMinusR6->setAxisTitle( "beam", 2 );



  // disk labels common for all TEC internal histograms
  for( unsigned int disk = 0; disk < 9; ++disk ) {
    labelBuilder.clear(); labelBuilder.str( "" );
    labelBuilder << "DISK" << disk;
    nSignalsTECPlusR4->setBinLabel( disk+1, labelBuilder.str(), 1 );
    nSignalsTECPlusR6->setBinLabel( disk+1, labelBuilder.str(), 1 );
    nSignalsTECMinusR4->setBinLabel( disk+1, labelBuilder.str(), 1 );
    nSignalsTECMinusR6->setBinLabel( disk+1, labelBuilder.str(), 1 );
  }


  // beam labels common for all histograms
  for( unsigned int beam = 0; beam < 8; ++beam ) {
    labelBuilder.clear(); labelBuilder.str( "" );
    labelBuilder << "BEAM" << beam;
    nSignalsAT->setBinLabel( beam+1, labelBuilder.str(), 2 );
    nSignalsTECPlusR4->setBinLabel( beam+1, labelBuilder.str(), 2 );
    nSignalsTECPlusR6->setBinLabel( beam+1, labelBuilder.str(), 2 );
    nSignalsTECMinusR4->setBinLabel( beam+1, labelBuilder.str(), 2 );
    nSignalsTECMinusR6->setBinLabel( beam+1, labelBuilder.str(), 2 );
  }

}





///
///
///
void LaserAlignmentT0ProducerDQM::analyze( const edm::Event& aEvent, const edm::EventSetup& aSetup ) {


  // loop all input products
  for ( std::vector<edm::ParameterSet>::iterator aDigiProducer = theDigiProducerList.begin(); aDigiProducer != theDigiProducerList.end(); ++aDigiProducer ) {
    const std::string digiProducer = aDigiProducer->getParameter<std::string>( "DigiProducer" );
    const std::string digiLabel = aDigiProducer->getParameter<std::string>( "DigiLabel" );
    const std::string digiType = aDigiProducer->getParameter<std::string>( "DigiType" );

    // now a distinction of cases: raw or processed digis?

    // first we go for raw digis => SiStripRawDigi
    if( digiType == "Raw" ) {

      // retrieve the SiStripRawDigis collection
      edm::Handle< edm::DetSetVector<SiStripRawDigi> > rawDigis;
      aEvent.getByLabel( digiProducer, digiLabel, rawDigis );
      
      // eval & fill histos from raw digis
      FillFromRawDigis( *rawDigis );

    }

    
    // next we assume "ZeroSuppressed" (non-raw) => SiStripDigi
    else if( digiType == "Processed" ) {
      
      edm::Handle< edm::DetSetVector<SiStripDigi> > processedDigis;
      aEvent.getByLabel( digiProducer, digiLabel, processedDigis );
      
      // eval & fill histos from processed digis
      FillFromProcessedDigis( *processedDigis );

    }



    // otherwise we have a problem
    else {
      throw cms::Exception( "LaserAlignmentT0ProducerDQM" ) << " ERROR ** Unknown DigiType: " << digiType << " specified in config." << std::endl;
    }
    

  } // loop all input products



}





///
///
///
void LaserAlignmentT0ProducerDQM::endJob() {

  bool writeToPlainROOTFile = theConfiguration.getParameter<bool>( "OutputInPlainROOT" );
  
  if( writeToPlainROOTFile ) {
    std::string outputFileName = theConfiguration.getParameter<std::string>( "PlainOutputFileName" );
    theDqmStore->showDirStructure();
    theDqmStore->save( outputFileName );
  }

}





///
///
///
void LaserAlignmentT0ProducerDQM::FillFromRawDigis( const edm::DetSetVector<SiStripRawDigi>& aDetSetVector ) {

  LASGlobalLoop moduleLoop;
  int det, ring, beam, disk, pos;




  // tec internal modules
  det = 0; ring = 0; beam = 0; disk = 0;
  do {

    bool isAboveThreshold = false;
    bool isExceedThreshold = false;

    // retrieve the raw id of that module
    const int detRawId = detectorId.GetTECEntry( det, ring, beam, disk );

    // search the digis for this raw id
    edm::DetSetVector<SiStripRawDigi>::const_iterator detSetIter = aDetSetVector.find( detRawId );

    // raw DetSets may not be missing
    if( detSetIter == aDetSetVector.end() ) {
      throw cms::Exception( "[LaserAlignmentT0ProducerDQM::FillFromRawDigis]" ) << " ** ERROR: No raw DetSet found for det: " << detRawId << "." << std::endl;
    }

    // access single modules' digis
    edm::DetSet<SiStripRawDigi>::const_iterator digiRangeIterator = detSetIter->data.begin();

    for( ; digiRangeIterator != detSetIter->data.end(); ++digiRangeIterator ) {
      const SiStripRawDigi& digi = *digiRangeIterator;

      // loop all digis and 
      // look for at least one strip above the threshold (-> assume a signal)
      // look if no strip is above threshold (-> assume overdrive)
      if( digi.adc() > theLowerAdcThreshold  ) isAboveThreshold = true;
      if( digi.adc() > theUpperAdcThreshold  ) isExceedThreshold = true;

    }

    // if we have signal, fill the histos
    if( isAboveThreshold && !isExceedThreshold ) {
      
      // determine the appropriate histogram & bin from the position variables
      if( det == 0 ) { // TEC+
	if( ring == 0 ) nSignalsTECPlusR4->Fill( disk, beam ); // R4
	else nSignalsTECPlusR6->Fill( disk, beam ); // R6
      }
      else { // TEC-
	if( ring == 0 ) nSignalsTECMinusR4->Fill( disk, beam ); // R4
	else nSignalsTECMinusR6->Fill( disk, beam ); // R6
      }
    
    }


  } while( moduleLoop.TECLoop( det, ring, beam, disk ) );




  // endcap modules (AT beams)
  det = 0; beam = 0; disk = 0;
  do {

    bool isAboveThreshold = false;
    bool isExceedThreshold = false;

    // retrieve the raw id of that module
    const int detRawId = detectorId.GetTEC2TECEntry( det, beam, disk );

    // search the digis for this raw id
    edm::DetSetVector<SiStripRawDigi>::const_iterator detSetIter = aDetSetVector.find( detRawId );

    // raw DetSets may not be missing
    if( detSetIter == aDetSetVector.end() ) {
      throw cms::Exception( "[LaserAlignmentT0ProducerDQM::FillFromRawDigis]" ) << " ** ERROR: No raw DetSet found for det: " << detRawId << "." << std::endl;
    }

    // access single modules' digis
    edm::DetSet<SiStripRawDigi>::const_iterator digiRangeIterator = detSetIter->data.begin();

    for( ; digiRangeIterator != detSetIter->data.end(); ++digiRangeIterator ) {
      const SiStripRawDigi& digi = *digiRangeIterator;

      // loop all digis and 
      // look for at least one strip above the threshold (-> assume a signal)
      // look if no strip is above threshold (-> assume overdrive)
      if( digi.adc() > theLowerAdcThreshold  ) isAboveThreshold = true;
      if( digi.adc() > theUpperAdcThreshold  ) isExceedThreshold = true;

    }

    // if we have signal, fill the histos
    if( isAboveThreshold && !isExceedThreshold ) {

      // there is only one histogram for all AT hits
      // but the bin scheme is a little complicated:
      // the TEC(AT) go in the first 5(-) and last 5(+) of 22 bins along x

      if( det == 1 ) nSignalsAT->Fill( 4 - disk, beam ); // TEC-
      else nSignalsAT->Fill( 17 + disk, beam ); // TEC+
    
    }


  } while( moduleLoop.TEC2TECLoop( det, beam, disk ) );



  // barrel modules (AT beams)
  det = 2; beam = 0; pos = 0;
  do {

    bool isAboveThreshold = false;
    bool isExceedThreshold = false;

    // retrieve the raw id of that module
    const int detRawId = detectorId.GetTIBTOBEntry( det, beam, pos );

    // search the digis for this raw id
    edm::DetSetVector<SiStripRawDigi>::const_iterator detSetIter = aDetSetVector.find( detRawId );

    // raw DetSets may not be missing
    if( detSetIter == aDetSetVector.end() ) {
      throw cms::Exception( "[LaserAlignmentT0ProducerDQM::FillFromRawDigis]" ) << " ** ERROR: No raw DetSet found for det: " << detRawId << "." << std::endl;
    }

    // access single modules' digis
    edm::DetSet<SiStripRawDigi>::const_iterator digiRangeIterator = detSetIter->data.begin();

    for( ; digiRangeIterator != detSetIter->data.end(); ++digiRangeIterator ) {
      const SiStripRawDigi& digi = *digiRangeIterator;

      // loop all digis and 
      // look for at least one strip above the threshold (-> assume a signal)
      // look if no strip is above threshold (-> assume overdrive)
      if( digi.adc() > theLowerAdcThreshold  ) isAboveThreshold = true;
      if( digi.adc() > theUpperAdcThreshold  ) isExceedThreshold = true;

    }

    // if we have signal, fill the histos
    if( isAboveThreshold && !isExceedThreshold ) {

      // there is only one histogram for all AT hits
      // but the bin scheme is a little complicated:
      // the TIB go into bins 6-11, TOB in 12-17

      if( det == 2 ) nSignalsAT->Fill( 5 + (5 - pos), beam ); // TIB
      else nSignalsAT->Fill( 11 + (5 - pos), beam ); // TOB
    
    }


  } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );



}





///
///
///
void LaserAlignmentT0ProducerDQM::FillFromProcessedDigis( const edm::DetSetVector<SiStripDigi>& aDetSetVector ) {

  LASGlobalLoop moduleLoop;
  int det, ring, beam, disk, pos;

  // tec internal modules
  det = 0; ring = 0; beam = 0; disk = 0;
  do {

    bool isAboveThreshold = false;
    bool isExceedThreshold = false;

    // retrieve the raw id of that module
    const int detRawId = detectorId.GetTECEntry( det, ring, beam, disk );

    // search the digis for this raw id
    edm::DetSetVector<SiStripDigi>::const_iterator detSetIter = aDetSetVector.find( detRawId );

    // processed DetSets may be missing (=empty), just skip
    if( detSetIter == aDetSetVector.end() ) continue;

    // access single modules' digis
    edm::DetSet<SiStripDigi>::const_iterator digiRangeIterator = detSetIter->data.begin();

    for( ; digiRangeIterator != detSetIter->data.end(); ++digiRangeIterator ) {
      const SiStripDigi& digi = *digiRangeIterator;

      // loop all digis and 
      // look for at least one strip above the threshold (-> assume a signal)
      // look if no strip is above threshold (->assume overdrive)
      if( digi.adc() > theLowerAdcThreshold  ) isAboveThreshold = true;
      if( digi.adc() > theUpperAdcThreshold  ) isExceedThreshold = true;

    }

    // if we have signal, fill the histos
    if( isAboveThreshold && !isExceedThreshold ) {
      
      // determine the appropriate histogram & bin from the position variables
      if( det == 0 ) { // TEC+
	if( ring == 0 ) nSignalsTECPlusR4->Fill( disk, beam ); // R4
	else nSignalsTECPlusR6->Fill( disk, beam ); // R6
      }
      else { // TEC-
	if( ring == 0 ) nSignalsTECMinusR4->Fill( disk, beam ); // R4
	else nSignalsTECMinusR6->Fill( disk, beam ); // R6
      }
    
    }


  } while( moduleLoop.TECLoop( det, ring, beam, disk ) );



  // endcap modules (AT beams)
  det = 0; beam = 0; disk = 0;
  do {

    bool isAboveThreshold = false;
    bool isExceedThreshold = false;

    // retrieve the raw id of that module
    const int detRawId = detectorId.GetTEC2TECEntry( det, beam, disk );

    // search the digis for this raw id
    edm::DetSetVector<SiStripDigi>::const_iterator detSetIter = aDetSetVector.find( detRawId );

    // processed DetSets may be missing (=empty), just skip
    if( detSetIter == aDetSetVector.end() ) continue;

    // access single modules' digis
    edm::DetSet<SiStripDigi>::const_iterator digiRangeIterator = detSetIter->data.begin();

    for( ; digiRangeIterator != detSetIter->data.end(); ++digiRangeIterator ) {
      const SiStripDigi& digi = *digiRangeIterator;

      // loop all digis and 
      // look for at least one strip above the threshold (-> assume a signal)
      // look if no strip is above threshold (-> assume overdrive)
      if( digi.adc() > theLowerAdcThreshold  ) isAboveThreshold = true;
      if( digi.adc() > theUpperAdcThreshold  ) isExceedThreshold = true;

    }

    // if we have signal, fill the histos
    if( isAboveThreshold && !isExceedThreshold ) {

      // there is only one histogram for all AT hits
      // but the bin scheme is a little complicated:
      // the TEC(AT) go in the first 5(-) and last 5(+) of 22 bins along x

      if( det == 1 ) nSignalsAT->Fill( 4 - disk, beam ); // TEC-
      else nSignalsAT->Fill( 17 + disk, beam ); // TEC+
    
    }


  } while( moduleLoop.TEC2TECLoop( det, beam, disk ) );



  // barrel modules (AT beams)
  det = 2; beam = 0; pos = 0;
  do {

    bool isAboveThreshold = false;
    bool isExceedThreshold = false;

    // retrieve the raw id of that module
    const int detRawId = detectorId.GetTIBTOBEntry( det, beam, pos );

    // search the digis for this raw id
    edm::DetSetVector<SiStripDigi>::const_iterator detSetIter = aDetSetVector.find( detRawId );

    // processed DetSets may be missing (=empty), just skip
    if( detSetIter == aDetSetVector.end() ) continue;

    // access single modules' digis
    edm::DetSet<SiStripDigi>::const_iterator digiRangeIterator = detSetIter->data.begin();

    for( ; digiRangeIterator != detSetIter->data.end(); ++digiRangeIterator ) {
      const SiStripDigi& digi = *digiRangeIterator;

      // loop all digis and 
      // look for at least one strip above the threshold (-> assume a signal)
      // look if no strip is above threshold (-> assume overdrive)
      if( digi.adc() > theLowerAdcThreshold  ) isAboveThreshold = true;
      if( digi.adc() > theUpperAdcThreshold  ) isExceedThreshold = true;

    }

    // if we have signal, fill the histos
    if( isAboveThreshold && !isExceedThreshold ) {

      // there is only one histogram for all AT hits
      // but the bin scheme is a little complicated:
      // the TIB go into bins 6-11, TOB in 12-17

      if( det == 2 ) nSignalsAT->Fill( 5 + (5 - pos), beam ); // TIB
      else nSignalsAT->Fill( 11 + (5 - pos), beam ); // TOB
    
    }


  } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );


}





///
/// all the detector ids for the LAS modules hard-coded
/// 
/// ugly code duplication but the deadline approaches..
/// LATER must make this code somehow common to both LaserAlignment and LaserAlignmentT0ProducerDQM
///
void LaserAlignmentT0ProducerDQM::FillDetectorId( void ) {
  
  // these are the detids of the TEC modules hit
  // by the AT as well as the TEC beams
  tecDoubleHitDetId.push_back( 470307208 );
  tecDoubleHitDetId.push_back( 470323592 );
  tecDoubleHitDetId.push_back( 470339976 );
  tecDoubleHitDetId.push_back( 470356360 );
  tecDoubleHitDetId.push_back( 470372744 );
  tecDoubleHitDetId.push_back( 470307976 );
  tecDoubleHitDetId.push_back( 470324360 );
  tecDoubleHitDetId.push_back( 470340744 );
  tecDoubleHitDetId.push_back( 470357128 );
  tecDoubleHitDetId.push_back( 470373512 );
  tecDoubleHitDetId.push_back( 470308488 );
  tecDoubleHitDetId.push_back( 470324872 );
  tecDoubleHitDetId.push_back( 470341256 );
  tecDoubleHitDetId.push_back( 470357640 );
  tecDoubleHitDetId.push_back( 470374024 );
  tecDoubleHitDetId.push_back( 470045064 );
  tecDoubleHitDetId.push_back( 470061448 );
  tecDoubleHitDetId.push_back( 470077832 );
  tecDoubleHitDetId.push_back( 470094216 );
  tecDoubleHitDetId.push_back( 470110600 );
  tecDoubleHitDetId.push_back( 470045832 );
  tecDoubleHitDetId.push_back( 470062216 );
  tecDoubleHitDetId.push_back( 470078600 );
  tecDoubleHitDetId.push_back( 470094984 );
  tecDoubleHitDetId.push_back( 470111368 );
  tecDoubleHitDetId.push_back( 470046344 );
  tecDoubleHitDetId.push_back( 470062728 );
  tecDoubleHitDetId.push_back( 470079112 );
  tecDoubleHitDetId.push_back( 470095496 );
  tecDoubleHitDetId.push_back( 470111880 );

  // now all the modules (above included)

  // TEC+
  detectorId.SetTECEntry( 0, 0, 0, 0, 470307208 );
  detectorId.SetTECEntry( 0, 0, 0, 1, 470323592 );
  detectorId.SetTECEntry( 0, 0, 0, 2, 470339976 );
  detectorId.SetTECEntry( 0, 0, 0, 3, 470356360 );
  detectorId.SetTECEntry( 0, 0, 0, 4, 470372744 );
  detectorId.SetTECEntry( 0, 0, 0, 5, 470389128 );
  detectorId.SetTECEntry( 0, 0, 0, 6, 470405512 );
  detectorId.SetTECEntry( 0, 0, 0, 7, 470421896 );
  detectorId.SetTECEntry( 0, 0, 0, 8, 470438280 );
  detectorId.SetTECEntry( 0, 0, 1, 0, 470307464 );
  detectorId.SetTECEntry( 0, 0, 1, 1, 470323848 );
  detectorId.SetTECEntry( 0, 0, 1, 2, 470340232 );
  detectorId.SetTECEntry( 0, 0, 1, 3, 470356616 );
  detectorId.SetTECEntry( 0, 0, 1, 4, 470373000 );
  detectorId.SetTECEntry( 0, 0, 1, 5, 470389384 );
  detectorId.SetTECEntry( 0, 0, 1, 6, 470405768 );
  detectorId.SetTECEntry( 0, 0, 1, 7, 470422152 );
  detectorId.SetTECEntry( 0, 0, 1, 8, 470438536 );
  detectorId.SetTECEntry( 0, 0, 2, 0, 470307720 );
  detectorId.SetTECEntry( 0, 0, 2, 1, 470324104 );
  detectorId.SetTECEntry( 0, 0, 2, 2, 470340488 );
  detectorId.SetTECEntry( 0, 0, 2, 3, 470356872 );
  detectorId.SetTECEntry( 0, 0, 2, 4, 470373256 );
  detectorId.SetTECEntry( 0, 0, 2, 5, 470389640 );
  detectorId.SetTECEntry( 0, 0, 2, 6, 470406024 );
  detectorId.SetTECEntry( 0, 0, 2, 7, 470422408 );
  detectorId.SetTECEntry( 0, 0, 2, 8, 470438792 );
  detectorId.SetTECEntry( 0, 0, 3, 0, 470307976 );
  detectorId.SetTECEntry( 0, 0, 3, 1, 470324360 );
  detectorId.SetTECEntry( 0, 0, 3, 2, 470340744 );
  detectorId.SetTECEntry( 0, 0, 3, 3, 470357128 );
  detectorId.SetTECEntry( 0, 0, 3, 4, 470373512 );
  detectorId.SetTECEntry( 0, 0, 3, 5, 470389896 );
  detectorId.SetTECEntry( 0, 0, 3, 6, 470406280 );
  detectorId.SetTECEntry( 0, 0, 3, 7, 470422664 );
  detectorId.SetTECEntry( 0, 0, 3, 8, 470439048 );
  detectorId.SetTECEntry( 0, 0, 4, 0, 470308232 );
  detectorId.SetTECEntry( 0, 0, 4, 1, 470324616 );
  detectorId.SetTECEntry( 0, 0, 4, 2, 470341000 );
  detectorId.SetTECEntry( 0, 0, 4, 3, 470357384 );
  detectorId.SetTECEntry( 0, 0, 4, 4, 470373768 );
  detectorId.SetTECEntry( 0, 0, 4, 5, 470390152 );
  detectorId.SetTECEntry( 0, 0, 4, 6, 470406536 );
  detectorId.SetTECEntry( 0, 0, 4, 7, 470422920 );
  detectorId.SetTECEntry( 0, 0, 4, 8, 470439304 );
  detectorId.SetTECEntry( 0, 0, 5, 0, 470308488 );
  detectorId.SetTECEntry( 0, 0, 5, 1, 470324872 );
  detectorId.SetTECEntry( 0, 0, 5, 2, 470341256 );
  detectorId.SetTECEntry( 0, 0, 5, 3, 470357640 );
  detectorId.SetTECEntry( 0, 0, 5, 4, 470374024 );
  detectorId.SetTECEntry( 0, 0, 5, 5, 470390408 );
  detectorId.SetTECEntry( 0, 0, 5, 6, 470406792 );
  detectorId.SetTECEntry( 0, 0, 5, 7, 470423176 );
  detectorId.SetTECEntry( 0, 0, 5, 8, 470439560 );
  detectorId.SetTECEntry( 0, 0, 6, 0, 470308744 );
  detectorId.SetTECEntry( 0, 0, 6, 1, 470325128 );
  detectorId.SetTECEntry( 0, 0, 6, 2, 470341512 );
  detectorId.SetTECEntry( 0, 0, 6, 3, 470357896 );
  detectorId.SetTECEntry( 0, 0, 6, 4, 470374280 );
  detectorId.SetTECEntry( 0, 0, 6, 5, 470390664 );
  detectorId.SetTECEntry( 0, 0, 6, 6, 470407048 );
  detectorId.SetTECEntry( 0, 0, 6, 7, 470423432 );
  detectorId.SetTECEntry( 0, 0, 6, 8, 470439816 );
  detectorId.SetTECEntry( 0, 0, 7, 0, 470309000 );
  detectorId.SetTECEntry( 0, 0, 7, 1, 470325384 );
  detectorId.SetTECEntry( 0, 0, 7, 2, 470341768 );
  detectorId.SetTECEntry( 0, 0, 7, 3, 470358152 );
  detectorId.SetTECEntry( 0, 0, 7, 4, 470374536 );
  detectorId.SetTECEntry( 0, 0, 7, 5, 470390920 );
  detectorId.SetTECEntry( 0, 0, 7, 6, 470407304 );
  detectorId.SetTECEntry( 0, 0, 7, 7, 470423688 );
  detectorId.SetTECEntry( 0, 0, 7, 8, 470440072 );
  detectorId.SetTECEntry( 0, 1, 0, 0, 470307272 );
  detectorId.SetTECEntry( 0, 1, 0, 1, 470323656 );
  detectorId.SetTECEntry( 0, 1, 0, 2, 470340040 );
  detectorId.SetTECEntry( 0, 1, 0, 3, 470356424 );
  detectorId.SetTECEntry( 0, 1, 0, 4, 470372808 );
  detectorId.SetTECEntry( 0, 1, 0, 5, 470389192 );
  detectorId.SetTECEntry( 0, 1, 0, 6, 470405576 );
  detectorId.SetTECEntry( 0, 1, 0, 7, 470421960 );
  detectorId.SetTECEntry( 0, 1, 0, 8, 470438344 );
  detectorId.SetTECEntry( 0, 1, 1, 0, 470307528 );
  detectorId.SetTECEntry( 0, 1, 1, 1, 470323912 );
  detectorId.SetTECEntry( 0, 1, 1, 2, 470340296 );
  detectorId.SetTECEntry( 0, 1, 1, 3, 470356680 );
  detectorId.SetTECEntry( 0, 1, 1, 4, 470373064 );
  detectorId.SetTECEntry( 0, 1, 1, 5, 470389448 );
  detectorId.SetTECEntry( 0, 1, 1, 6, 470405832 );
  detectorId.SetTECEntry( 0, 1, 1, 7, 470422216 );
  detectorId.SetTECEntry( 0, 1, 1, 8, 470438600 );
  detectorId.SetTECEntry( 0, 1, 2, 0, 470307784 );
  detectorId.SetTECEntry( 0, 1, 2, 1, 470324168 );
  detectorId.SetTECEntry( 0, 1, 2, 2, 470340552 );
  detectorId.SetTECEntry( 0, 1, 2, 3, 470356936 );
  detectorId.SetTECEntry( 0, 1, 2, 4, 470373320 );
  detectorId.SetTECEntry( 0, 1, 2, 5, 470389704 );
  detectorId.SetTECEntry( 0, 1, 2, 6, 470406088 );
  detectorId.SetTECEntry( 0, 1, 2, 7, 470422472 );
  detectorId.SetTECEntry( 0, 1, 2, 8, 470438856 );
  detectorId.SetTECEntry( 0, 1, 3, 0, 470308040 );
  detectorId.SetTECEntry( 0, 1, 3, 1, 470324424 );
  detectorId.SetTECEntry( 0, 1, 3, 2, 470340808 );
  detectorId.SetTECEntry( 0, 1, 3, 3, 470357192 );
  detectorId.SetTECEntry( 0, 1, 3, 4, 470373576 );
  detectorId.SetTECEntry( 0, 1, 3, 5, 470389960 );
  detectorId.SetTECEntry( 0, 1, 3, 6, 470406344 );
  detectorId.SetTECEntry( 0, 1, 3, 7, 470422728 );
  detectorId.SetTECEntry( 0, 1, 3, 8, 470439112 );
  detectorId.SetTECEntry( 0, 1, 4, 0, 470308296 );
  detectorId.SetTECEntry( 0, 1, 4, 1, 470324680 );
  detectorId.SetTECEntry( 0, 1, 4, 2, 470341064 );
  detectorId.SetTECEntry( 0, 1, 4, 3, 470357448 );
  detectorId.SetTECEntry( 0, 1, 4, 4, 470373832 );
  detectorId.SetTECEntry( 0, 1, 4, 5, 470390216 );
  detectorId.SetTECEntry( 0, 1, 4, 6, 470406600 );
  detectorId.SetTECEntry( 0, 1, 4, 7, 470422984 );
  detectorId.SetTECEntry( 0, 1, 4, 8, 470439368 );
  detectorId.SetTECEntry( 0, 1, 5, 0, 470308552 );
  detectorId.SetTECEntry( 0, 1, 5, 1, 470324936 );
  detectorId.SetTECEntry( 0, 1, 5, 2, 470341320 );
  detectorId.SetTECEntry( 0, 1, 5, 3, 470357704 );
  detectorId.SetTECEntry( 0, 1, 5, 4, 470374088 );
  detectorId.SetTECEntry( 0, 1, 5, 5, 470390472 );
  detectorId.SetTECEntry( 0, 1, 5, 6, 470406856 );
  detectorId.SetTECEntry( 0, 1, 5, 7, 470423240 );
  detectorId.SetTECEntry( 0, 1, 5, 8, 470439624 );
  detectorId.SetTECEntry( 0, 1, 6, 0, 470308808 );
  detectorId.SetTECEntry( 0, 1, 6, 1, 470325192 );
  detectorId.SetTECEntry( 0, 1, 6, 2, 470341576 );
  detectorId.SetTECEntry( 0, 1, 6, 3, 470357960 );
  detectorId.SetTECEntry( 0, 1, 6, 4, 470374344 );
  detectorId.SetTECEntry( 0, 1, 6, 5, 470390728 );
  detectorId.SetTECEntry( 0, 1, 6, 6, 470407112 );
  detectorId.SetTECEntry( 0, 1, 6, 7, 470423496 );
  detectorId.SetTECEntry( 0, 1, 6, 8, 470439880 );
  detectorId.SetTECEntry( 0, 1, 7, 0, 470309064 );
  detectorId.SetTECEntry( 0, 1, 7, 1, 470325448 );
  detectorId.SetTECEntry( 0, 1, 7, 2, 470341832 );
  detectorId.SetTECEntry( 0, 1, 7, 3, 470358216 );
  detectorId.SetTECEntry( 0, 1, 7, 4, 470374600 );
  detectorId.SetTECEntry( 0, 1, 7, 5, 470390984 );
  detectorId.SetTECEntry( 0, 1, 7, 6, 470407368 );
  detectorId.SetTECEntry( 0, 1, 7, 7, 470423752 );
  detectorId.SetTECEntry( 0, 1, 7, 8, 470440136 );

  // TEC-
  detectorId.SetTECEntry( 1, 0, 0, 0, 470045064 );
  detectorId.SetTECEntry( 1, 0, 0, 1, 470061448 );
  detectorId.SetTECEntry( 1, 0, 0, 2, 470077832 );
  detectorId.SetTECEntry( 1, 0, 0, 3, 470094216 );
  detectorId.SetTECEntry( 1, 0, 0, 4, 470110600 );
  detectorId.SetTECEntry( 1, 0, 0, 5, 470126984 );
  detectorId.SetTECEntry( 1, 0, 0, 6, 470143368 );
  detectorId.SetTECEntry( 1, 0, 0, 7, 470159752 );
  detectorId.SetTECEntry( 1, 0, 0, 8, 470176136 );
  detectorId.SetTECEntry( 1, 0, 1, 0, 470045320 );
  detectorId.SetTECEntry( 1, 0, 1, 1, 470061704 );
  detectorId.SetTECEntry( 1, 0, 1, 2, 470078088 );
  detectorId.SetTECEntry( 1, 0, 1, 3, 470094472 );
  detectorId.SetTECEntry( 1, 0, 1, 4, 470110856 );
  detectorId.SetTECEntry( 1, 0, 1, 5, 470127240 );
  detectorId.SetTECEntry( 1, 0, 1, 6, 470143624 );
  detectorId.SetTECEntry( 1, 0, 1, 7, 470160008 );
  detectorId.SetTECEntry( 1, 0, 1, 8, 470176392 );
  detectorId.SetTECEntry( 1, 0, 2, 0, 470045576 );
  detectorId.SetTECEntry( 1, 0, 2, 1, 470061960 );
  detectorId.SetTECEntry( 1, 0, 2, 2, 470078344 );
  detectorId.SetTECEntry( 1, 0, 2, 3, 470094728 );
  detectorId.SetTECEntry( 1, 0, 2, 4, 470111112 );
  detectorId.SetTECEntry( 1, 0, 2, 5, 470127496 );
  detectorId.SetTECEntry( 1, 0, 2, 6, 470143880 );
  detectorId.SetTECEntry( 1, 0, 2, 7, 470160264 );
  detectorId.SetTECEntry( 1, 0, 2, 8, 470176648 );
  detectorId.SetTECEntry( 1, 0, 3, 0, 470045832 );
  detectorId.SetTECEntry( 1, 0, 3, 1, 470062216 );
  detectorId.SetTECEntry( 1, 0, 3, 2, 470078600 );
  detectorId.SetTECEntry( 1, 0, 3, 3, 470094984 );
  detectorId.SetTECEntry( 1, 0, 3, 4, 470111368 );
  detectorId.SetTECEntry( 1, 0, 3, 5, 470127752 );
  detectorId.SetTECEntry( 1, 0, 3, 6, 470144136 );
  detectorId.SetTECEntry( 1, 0, 3, 7, 470160520 );
  detectorId.SetTECEntry( 1, 0, 3, 8, 470176904 );
  detectorId.SetTECEntry( 1, 0, 4, 0, 470046088 );
  detectorId.SetTECEntry( 1, 0, 4, 1, 470062472 );
  detectorId.SetTECEntry( 1, 0, 4, 2, 470078856 );
  detectorId.SetTECEntry( 1, 0, 4, 3, 470095240 );
  detectorId.SetTECEntry( 1, 0, 4, 4, 470111624 );
  detectorId.SetTECEntry( 1, 0, 4, 5, 470128008 );
  detectorId.SetTECEntry( 1, 0, 4, 6, 470144392 );
  detectorId.SetTECEntry( 1, 0, 4, 7, 470160776 );
  detectorId.SetTECEntry( 1, 0, 4, 8, 470177160 );
  detectorId.SetTECEntry( 1, 0, 5, 0, 470046344 );
  detectorId.SetTECEntry( 1, 0, 5, 1, 470062728 );
  detectorId.SetTECEntry( 1, 0, 5, 2, 470079112 );
  detectorId.SetTECEntry( 1, 0, 5, 3, 470095496 );
  detectorId.SetTECEntry( 1, 0, 5, 4, 470111880 );
  detectorId.SetTECEntry( 1, 0, 5, 5, 470128264 );
  detectorId.SetTECEntry( 1, 0, 5, 6, 470144648 );
  detectorId.SetTECEntry( 1, 0, 5, 7, 470161032 );
  detectorId.SetTECEntry( 1, 0, 5, 8, 470177416 );
  detectorId.SetTECEntry( 1, 0, 6, 0, 470046600 );
  detectorId.SetTECEntry( 1, 0, 6, 1, 470062984 );
  detectorId.SetTECEntry( 1, 0, 6, 2, 470079368 );
  detectorId.SetTECEntry( 1, 0, 6, 3, 470095752 );
  detectorId.SetTECEntry( 1, 0, 6, 4, 470112136 );
  detectorId.SetTECEntry( 1, 0, 6, 5, 470128520 );
  detectorId.SetTECEntry( 1, 0, 6, 6, 470144904 );
  detectorId.SetTECEntry( 1, 0, 6, 7, 470161288 );
  detectorId.SetTECEntry( 1, 0, 6, 8, 470177672 );
  detectorId.SetTECEntry( 1, 0, 7, 0, 470046856 );
  detectorId.SetTECEntry( 1, 0, 7, 1, 470063240 );
  detectorId.SetTECEntry( 1, 0, 7, 2, 470079624 );
  detectorId.SetTECEntry( 1, 0, 7, 3, 470096008 );
  detectorId.SetTECEntry( 1, 0, 7, 4, 470112392 );
  detectorId.SetTECEntry( 1, 0, 7, 5, 470128776 );
  detectorId.SetTECEntry( 1, 0, 7, 6, 470145160 );
  detectorId.SetTECEntry( 1, 0, 7, 7, 470161544 );
  detectorId.SetTECEntry( 1, 0, 7, 8, 470177928 );
  detectorId.SetTECEntry( 1, 1, 0, 0, 470045128 );
  detectorId.SetTECEntry( 1, 1, 0, 1, 470061512 );
  detectorId.SetTECEntry( 1, 1, 0, 2, 470077896 );
  detectorId.SetTECEntry( 1, 1, 0, 3, 470094280 );
  detectorId.SetTECEntry( 1, 1, 0, 4, 470110664 );
  detectorId.SetTECEntry( 1, 1, 0, 5, 470127048 );
  detectorId.SetTECEntry( 1, 1, 0, 6, 470143432 );
  detectorId.SetTECEntry( 1, 1, 0, 7, 470159816 );
  detectorId.SetTECEntry( 1, 1, 0, 8, 470176200 );
  detectorId.SetTECEntry( 1, 1, 1, 0, 470045384 );
  detectorId.SetTECEntry( 1, 1, 1, 1, 470061768 );
  detectorId.SetTECEntry( 1, 1, 1, 2, 470078152 );
  detectorId.SetTECEntry( 1, 1, 1, 3, 470094536 );
  detectorId.SetTECEntry( 1, 1, 1, 4, 470110920 );
  detectorId.SetTECEntry( 1, 1, 1, 5, 470127304 );
  detectorId.SetTECEntry( 1, 1, 1, 6, 470143688 );
  detectorId.SetTECEntry( 1, 1, 1, 7, 470160072 );
  detectorId.SetTECEntry( 1, 1, 1, 8, 470176456 );
  detectorId.SetTECEntry( 1, 1, 2, 0, 470045640 );
  detectorId.SetTECEntry( 1, 1, 2, 1, 470062024 );
  detectorId.SetTECEntry( 1, 1, 2, 2, 470078408 );
  detectorId.SetTECEntry( 1, 1, 2, 3, 470094792 );
  detectorId.SetTECEntry( 1, 1, 2, 4, 470111176 );
  detectorId.SetTECEntry( 1, 1, 2, 5, 470127560 );
  detectorId.SetTECEntry( 1, 1, 2, 6, 470143944 );
  detectorId.SetTECEntry( 1, 1, 2, 7, 470160328 );
  detectorId.SetTECEntry( 1, 1, 2, 8, 470176712 );
  detectorId.SetTECEntry( 1, 1, 3, 0, 470045896 );
  detectorId.SetTECEntry( 1, 1, 3, 1, 470062280 );
  detectorId.SetTECEntry( 1, 1, 3, 2, 470078664 );
  detectorId.SetTECEntry( 1, 1, 3, 3, 470095048 );
  detectorId.SetTECEntry( 1, 1, 3, 4, 470111432 );
  detectorId.SetTECEntry( 1, 1, 3, 5, 470127816 );
  detectorId.SetTECEntry( 1, 1, 3, 6, 470144200 );
  detectorId.SetTECEntry( 1, 1, 3, 7, 470160584 );
  detectorId.SetTECEntry( 1, 1, 3, 8, 470176968 );
  detectorId.SetTECEntry( 1, 1, 4, 0, 470046152 );
  detectorId.SetTECEntry( 1, 1, 4, 1, 470062536 );
  detectorId.SetTECEntry( 1, 1, 4, 2, 470078920 );
  detectorId.SetTECEntry( 1, 1, 4, 3, 470095304 );
  detectorId.SetTECEntry( 1, 1, 4, 4, 470111688 );
  detectorId.SetTECEntry( 1, 1, 4, 5, 470128072 );
  detectorId.SetTECEntry( 1, 1, 4, 6, 470144456 );
  detectorId.SetTECEntry( 1, 1, 4, 7, 470160840 );
  detectorId.SetTECEntry( 1, 1, 4, 8, 470177224 );
  detectorId.SetTECEntry( 1, 1, 5, 0, 470046408 );
  detectorId.SetTECEntry( 1, 1, 5, 1, 470062792 );
  detectorId.SetTECEntry( 1, 1, 5, 2, 470079176 );
  detectorId.SetTECEntry( 1, 1, 5, 3, 470095560 );
  detectorId.SetTECEntry( 1, 1, 5, 4, 470111944 );
  detectorId.SetTECEntry( 1, 1, 5, 5, 470128328 );
  detectorId.SetTECEntry( 1, 1, 5, 6, 470144712 );
  detectorId.SetTECEntry( 1, 1, 5, 7, 470161096 );
  detectorId.SetTECEntry( 1, 1, 5, 8, 470177480 );
  detectorId.SetTECEntry( 1, 1, 6, 0, 470046664 );
  detectorId.SetTECEntry( 1, 1, 6, 1, 470063048 );
  detectorId.SetTECEntry( 1, 1, 6, 2, 470079432 );
  detectorId.SetTECEntry( 1, 1, 6, 3, 470095816 );
  detectorId.SetTECEntry( 1, 1, 6, 4, 470112200 );
  detectorId.SetTECEntry( 1, 1, 6, 5, 470128584 );
  detectorId.SetTECEntry( 1, 1, 6, 6, 470144968 );
  detectorId.SetTECEntry( 1, 1, 6, 7, 470161352 );
  detectorId.SetTECEntry( 1, 1, 6, 8, 470177736 );
  detectorId.SetTECEntry( 1, 1, 7, 0, 470046920 );
  detectorId.SetTECEntry( 1, 1, 7, 1, 470063304 );
  detectorId.SetTECEntry( 1, 1, 7, 2, 470079688 );
  detectorId.SetTECEntry( 1, 1, 7, 3, 470096072 );
  detectorId.SetTECEntry( 1, 1, 7, 4, 470112456 );
  detectorId.SetTECEntry( 1, 1, 7, 5, 470128840 );
  detectorId.SetTECEntry( 1, 1, 7, 6, 470145224 );
  detectorId.SetTECEntry( 1, 1, 7, 7, 470161608 );
  detectorId.SetTECEntry( 1, 1, 7, 8, 470177992 );

  // TIB
  detectorId.SetTIBTOBEntry( 2, 0, 0, 369174604 );
  detectorId.SetTIBTOBEntry( 2, 0, 1, 369174600 );
  detectorId.SetTIBTOBEntry( 2, 0, 2, 369174596 );
  detectorId.SetTIBTOBEntry( 2, 0, 3, 369170500 );
  detectorId.SetTIBTOBEntry( 2, 0, 4, 369170504 );
  detectorId.SetTIBTOBEntry( 2, 0, 5, 369170508 );
  detectorId.SetTIBTOBEntry( 2, 1, 0, 369174732 );
  detectorId.SetTIBTOBEntry( 2, 1, 1, 369174728 );
  detectorId.SetTIBTOBEntry( 2, 1, 2, 369174724 );
  detectorId.SetTIBTOBEntry( 2, 1, 3, 369170628 );
  detectorId.SetTIBTOBEntry( 2, 1, 4, 369170632 );
  detectorId.SetTIBTOBEntry( 2, 1, 5, 369170636 );
  detectorId.SetTIBTOBEntry( 2, 2, 0, 369174812 );
  detectorId.SetTIBTOBEntry( 2, 2, 1, 369174808 );
  detectorId.SetTIBTOBEntry( 2, 2, 2, 369174804 );
  detectorId.SetTIBTOBEntry( 2, 2, 3, 369170708 );
  detectorId.SetTIBTOBEntry( 2, 2, 4, 369170712 );
  detectorId.SetTIBTOBEntry( 2, 2, 5, 369170716 );
  detectorId.SetTIBTOBEntry( 2, 3, 0, 369174940 );
  detectorId.SetTIBTOBEntry( 2, 3, 1, 369174936 );
  detectorId.SetTIBTOBEntry( 2, 3, 2, 369174932 );
  detectorId.SetTIBTOBEntry( 2, 3, 3, 369170836 );
  detectorId.SetTIBTOBEntry( 2, 3, 4, 369170840 );
  detectorId.SetTIBTOBEntry( 2, 3, 5, 369170844 );
  detectorId.SetTIBTOBEntry( 2, 4, 0, 369175068 );
  detectorId.SetTIBTOBEntry( 2, 4, 1, 369175064 );
  detectorId.SetTIBTOBEntry( 2, 4, 2, 369175060 );
  detectorId.SetTIBTOBEntry( 2, 4, 3, 369170964 );
  detectorId.SetTIBTOBEntry( 2, 4, 4, 369170968 );
  detectorId.SetTIBTOBEntry( 2, 4, 5, 369170972 );
  detectorId.SetTIBTOBEntry( 2, 5, 0, 369175164 );
  detectorId.SetTIBTOBEntry( 2, 5, 1, 369175160 );
  detectorId.SetTIBTOBEntry( 2, 5, 2, 369175156 );
  detectorId.SetTIBTOBEntry( 2, 5, 3, 369171060 );
  detectorId.SetTIBTOBEntry( 2, 5, 4, 369171064 );
  detectorId.SetTIBTOBEntry( 2, 5, 5, 369171068 );
  detectorId.SetTIBTOBEntry( 2, 6, 0, 369175292 );
  detectorId.SetTIBTOBEntry( 2, 6, 1, 369175288 );
  detectorId.SetTIBTOBEntry( 2, 6, 2, 369175284 );
  detectorId.SetTIBTOBEntry( 2, 6, 3, 369171188 );
  detectorId.SetTIBTOBEntry( 2, 6, 4, 369171192 );
  detectorId.SetTIBTOBEntry( 2, 6, 5, 369171196 );
  detectorId.SetTIBTOBEntry( 2, 7, 0, 369175372 );
  detectorId.SetTIBTOBEntry( 2, 7, 1, 369175368 );
  detectorId.SetTIBTOBEntry( 2, 7, 2, 369175364 );
  detectorId.SetTIBTOBEntry( 2, 7, 3, 369171268 );
  detectorId.SetTIBTOBEntry( 2, 7, 4, 369171272 );
  detectorId.SetTIBTOBEntry( 2, 7, 5, 369171276 );

  // TOB
  detectorId.SetTIBTOBEntry( 3, 0, 0, 436232314 );
  detectorId.SetTIBTOBEntry( 3, 0, 1, 436232306 );
  detectorId.SetTIBTOBEntry( 3, 0, 2, 436232298 );
  detectorId.SetTIBTOBEntry( 3, 0, 3, 436228198 );
  detectorId.SetTIBTOBEntry( 3, 0, 4, 436228206 );
  detectorId.SetTIBTOBEntry( 3, 0, 5, 436228214 );
  detectorId.SetTIBTOBEntry( 3, 1, 0, 436232506 );
  detectorId.SetTIBTOBEntry( 3, 1, 1, 436232498 );
  detectorId.SetTIBTOBEntry( 3, 1, 2, 436232490 );
  detectorId.SetTIBTOBEntry( 3, 1, 3, 436228390 );
  detectorId.SetTIBTOBEntry( 3, 1, 4, 436228398 );
  detectorId.SetTIBTOBEntry( 3, 1, 5, 436228406 );
  detectorId.SetTIBTOBEntry( 3, 2, 0, 436232634 );
  detectorId.SetTIBTOBEntry( 3, 2, 1, 436232626 );
  detectorId.SetTIBTOBEntry( 3, 2, 2, 436232618 );
  detectorId.SetTIBTOBEntry( 3, 2, 3, 436228518 );
  detectorId.SetTIBTOBEntry( 3, 2, 4, 436228526 );
  detectorId.SetTIBTOBEntry( 3, 2, 5, 436228534 );
  detectorId.SetTIBTOBEntry( 3, 3, 0, 436232826 );
  detectorId.SetTIBTOBEntry( 3, 3, 1, 436232818 );
  detectorId.SetTIBTOBEntry( 3, 3, 2, 436232810 );
  detectorId.SetTIBTOBEntry( 3, 3, 3, 436228710 );
  detectorId.SetTIBTOBEntry( 3, 3, 4, 436228718 );
  detectorId.SetTIBTOBEntry( 3, 3, 5, 436228726 );
  detectorId.SetTIBTOBEntry( 3, 4, 0, 436233018 );
  detectorId.SetTIBTOBEntry( 3, 4, 1, 436233010 );
  detectorId.SetTIBTOBEntry( 3, 4, 2, 436233002 );
  detectorId.SetTIBTOBEntry( 3, 4, 3, 436228902 );
  detectorId.SetTIBTOBEntry( 3, 4, 4, 436228910 );
  detectorId.SetTIBTOBEntry( 3, 4, 5, 436228918 );
  detectorId.SetTIBTOBEntry( 3, 5, 0, 436233146 );
  detectorId.SetTIBTOBEntry( 3, 5, 1, 436233138 );
  detectorId.SetTIBTOBEntry( 3, 5, 2, 436233130 );
  detectorId.SetTIBTOBEntry( 3, 5, 3, 436229030 );
  detectorId.SetTIBTOBEntry( 3, 5, 4, 436229038 );
  detectorId.SetTIBTOBEntry( 3, 5, 5, 436229046 );
  detectorId.SetTIBTOBEntry( 3, 6, 0, 436233338 );
  detectorId.SetTIBTOBEntry( 3, 6, 1, 436233330 );
  detectorId.SetTIBTOBEntry( 3, 6, 2, 436233322 );
  detectorId.SetTIBTOBEntry( 3, 6, 3, 436229222 );
  detectorId.SetTIBTOBEntry( 3, 6, 4, 436229230 );
  detectorId.SetTIBTOBEntry( 3, 6, 5, 436229238 );
  detectorId.SetTIBTOBEntry( 3, 7, 0, 436233466 );
  detectorId.SetTIBTOBEntry( 3, 7, 1, 436233458 );
  detectorId.SetTIBTOBEntry( 3, 7, 2, 436233450 );
  detectorId.SetTIBTOBEntry( 3, 7, 3, 436229350 );
  detectorId.SetTIBTOBEntry( 3, 7, 4, 436229358 );
  detectorId.SetTIBTOBEntry( 3, 7, 5, 436229366 );

  // TEC+ AT
  detectorId.SetTEC2TECEntry( 0, 0, 0, 470307208 );
  detectorId.SetTEC2TECEntry( 0, 0, 1, 470323592 );
  detectorId.SetTEC2TECEntry( 0, 0, 2, 470339976 );
  detectorId.SetTEC2TECEntry( 0, 0, 3, 470356360 );
  detectorId.SetTEC2TECEntry( 0, 0, 4, 470372744 );
  detectorId.SetTEC2TECEntry( 0, 1, 0, 470307468 );
  detectorId.SetTEC2TECEntry( 0, 1, 1, 470323852 );
  detectorId.SetTEC2TECEntry( 0, 1, 2, 470340236 );
  detectorId.SetTEC2TECEntry( 0, 1, 3, 470356620 );
  detectorId.SetTEC2TECEntry( 0, 1, 4, 470373004 );
  detectorId.SetTEC2TECEntry( 0, 2, 0, 470307716 );
  detectorId.SetTEC2TECEntry( 0, 2, 1, 470324100 );
  detectorId.SetTEC2TECEntry( 0, 2, 2, 470340484 );
  detectorId.SetTEC2TECEntry( 0, 2, 3, 470356868 );
  detectorId.SetTEC2TECEntry( 0, 2, 4, 470373252 );
  detectorId.SetTEC2TECEntry( 0, 3, 0, 470307976 );
  detectorId.SetTEC2TECEntry( 0, 3, 1, 470324360 );
  detectorId.SetTEC2TECEntry( 0, 3, 2, 470340744 );
  detectorId.SetTEC2TECEntry( 0, 3, 3, 470357128 );
  detectorId.SetTEC2TECEntry( 0, 3, 4, 470373512 );
  detectorId.SetTEC2TECEntry( 0, 4, 0, 470308236 );
  detectorId.SetTEC2TECEntry( 0, 4, 1, 470324620 );
  detectorId.SetTEC2TECEntry( 0, 4, 2, 470341004 );
  detectorId.SetTEC2TECEntry( 0, 4, 3, 470357388 );
  detectorId.SetTEC2TECEntry( 0, 4, 4, 470373772 );
  detectorId.SetTEC2TECEntry( 0, 5, 0, 470308488 );
  detectorId.SetTEC2TECEntry( 0, 5, 1, 470324872 );
  detectorId.SetTEC2TECEntry( 0, 5, 2, 470341256 );
  detectorId.SetTEC2TECEntry( 0, 5, 3, 470357640 );
  detectorId.SetTEC2TECEntry( 0, 5, 4, 470374024 );
  detectorId.SetTEC2TECEntry( 0, 6, 0, 470308748 );
  detectorId.SetTEC2TECEntry( 0, 6, 1, 470325132 );
  detectorId.SetTEC2TECEntry( 0, 6, 2, 470341516 );
  detectorId.SetTEC2TECEntry( 0, 6, 3, 470357900 );
  detectorId.SetTEC2TECEntry( 0, 6, 4, 470374284 );
  detectorId.SetTEC2TECEntry( 0, 7, 0, 470308996 );
  detectorId.SetTEC2TECEntry( 0, 7, 1, 470325380 );
  detectorId.SetTEC2TECEntry( 0, 7, 2, 470341764 );
  detectorId.SetTEC2TECEntry( 0, 7, 3, 470358148 );
  detectorId.SetTEC2TECEntry( 0, 7, 4, 470374532 );

  // TEC- AT
  detectorId.SetTEC2TECEntry( 1, 0, 0, 470045064 );
  detectorId.SetTEC2TECEntry( 1, 0, 1, 470061448 );
  detectorId.SetTEC2TECEntry( 1, 0, 2, 470077832 );
  detectorId.SetTEC2TECEntry( 1, 0, 3, 470094216 );
  detectorId.SetTEC2TECEntry( 1, 0, 4, 470110600 );
  detectorId.SetTEC2TECEntry( 1, 1, 0, 470045316 );
  detectorId.SetTEC2TECEntry( 1, 1, 1, 470061700 );
  detectorId.SetTEC2TECEntry( 1, 1, 2, 470078084 );
  detectorId.SetTEC2TECEntry( 1, 1, 3, 470094468 );
  detectorId.SetTEC2TECEntry( 1, 1, 4, 470110852 );
  detectorId.SetTEC2TECEntry( 1, 2, 0, 470045580 );
  detectorId.SetTEC2TECEntry( 1, 2, 1, 470061964 );
  detectorId.SetTEC2TECEntry( 1, 2, 2, 470078348 );
  detectorId.SetTEC2TECEntry( 1, 2, 3, 470094732 );
  detectorId.SetTEC2TECEntry( 1, 2, 4, 470111116 );
  detectorId.SetTEC2TECEntry( 1, 3, 0, 470045832 );
  detectorId.SetTEC2TECEntry( 1, 3, 1, 470062216 );
  detectorId.SetTEC2TECEntry( 1, 3, 2, 470078600 );
  detectorId.SetTEC2TECEntry( 1, 3, 3, 470094984 );
  detectorId.SetTEC2TECEntry( 1, 3, 4, 470111368 );
  detectorId.SetTEC2TECEntry( 1, 4, 0, 470046084 );
  detectorId.SetTEC2TECEntry( 1, 4, 1, 470062468 );
  detectorId.SetTEC2TECEntry( 1, 4, 2, 470078852 );
  detectorId.SetTEC2TECEntry( 1, 4, 3, 470095236 );
  detectorId.SetTEC2TECEntry( 1, 4, 4, 470111620 );
  detectorId.SetTEC2TECEntry( 1, 5, 0, 470046344 );
  detectorId.SetTEC2TECEntry( 1, 5, 1, 470062728 );
  detectorId.SetTEC2TECEntry( 1, 5, 2, 470079112 );
  detectorId.SetTEC2TECEntry( 1, 5, 3, 470095496 );
  detectorId.SetTEC2TECEntry( 1, 5, 4, 470111880 );
  detectorId.SetTEC2TECEntry( 1, 6, 0, 470046596 );
  detectorId.SetTEC2TECEntry( 1, 6, 1, 470062980 );
  detectorId.SetTEC2TECEntry( 1, 6, 2, 470079364 );
  detectorId.SetTEC2TECEntry( 1, 6, 3, 470095748 );
  detectorId.SetTEC2TECEntry( 1, 6, 4, 470112132 );
  detectorId.SetTEC2TECEntry( 1, 7, 0, 470046860 );
  detectorId.SetTEC2TECEntry( 1, 7, 1, 470063244 );
  detectorId.SetTEC2TECEntry( 1, 7, 2, 470079628 );
  detectorId.SetTEC2TECEntry( 1, 7, 3, 470096012 );
  detectorId.SetTEC2TECEntry( 1, 7, 4, 470112396 );

}

//define this as a plug-in
DEFINE_FWK_MODULE(LaserAlignmentT0ProducerDQM);

