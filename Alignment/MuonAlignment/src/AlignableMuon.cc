// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


// Geometry interface
#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerStringToEnum.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include "Alignment/MuonAlignment/interface/AlignableMuon.h"

// Muon  components
#include "Alignment/MuonAlignment/interface/AlignableDTChamber.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCChamber.h"
#include "Alignment/MuonAlignment/interface/AlignableDTStation.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCStation.h"
#include "Alignment/MuonAlignment/interface/AlignableDTWheel.h"
#include "Alignment/MuonAlignment/interface/AlignableDTBarrel.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCEndcap.h"


//--------------------------------------------------------------------------------------------------
AlignableMuon::AlignableMuon( const edm::EventSetup& iSetup  )
{

  // The XML geometry is accessed as in DTGeometry and CSC Geometry classes.
  // Since this geometry does not contain exactly the same structures we
  // need in alignment, we implement the alignable hierarchy here.
  //

  edm::LogInfo("AlignableMuon") << "Constructing alignable muon objects"; 
  

  // Get DTGeometry pointer
  iSetup.get<MuonGeometryRecord>().get( pDT );     

  // Build the muon barrel
  buildDTBarrel( pDT );

  // Get CSCGeometry pointer
  iSetup.get<MuonGeometryRecord>().get( pCSC ); 

  // Build the muon end caps
  buildCSCEndcap( pCSC );

  edm::LogInfo("AlignableMuon") << "Constructing alignable muon objects DONE"; 


}

AlignableMuon::AlignableMuon( DTGeometry& theDTGeometry , CSCGeometry& theCSCGeometry )
{

  // Build the muon barrel
  buildDTBarrel( &theDTGeometry );

  // Build the muon end caps
  buildCSCEndcap( &theCSCGeometry );


  edm::LogInfo("AlignableMuon") << "Constructing alignable muon objects DONE";


}

//--------------------------------------------------------------------------------------------------
AlignableMuon::~AlignableMuon() 
{

  for ( std::vector<Alignable*>::iterator iter=theMuonComponents.begin();
	iter != theMuonComponents.end(); iter++){
    delete *iter;
  }
      

}


//--------------------------------------------------------------------------------------------------
void AlignableMuon::buildDTBarrel( edm::ESHandle<DTGeometry> pDT  )
{
  
 LogDebug("Position") << "Constructing AlignableDTBarrel"; 

  // Temporary container for chambers in a given station and stations in a given wheel
  std::vector<AlignableDTChamber*>   tmpDTChambersInStation;
  std::vector<AlignableDTStation*>   tmpDTStationsInWheel;


  // Loop over wheels ( -2..2 )
  for( int iwh = -2 ; iwh < 3 ; iwh++ ){

    // Loop over stations ( 1..4 )
    for( int ist = 1 ; ist < 5 ; ist++ ){
  
      // Loop over geom DT Chambers
      std::vector<GeomDet*> theSLs;
      for( std::vector<DTChamber*>::const_iterator det = pDT->chambers().begin(); 
                                             det != pDT->chambers().end(); ++det ){
        // Get the chamber ID
        DTChamberId chamberId = (*det)->id(); 

        // Get wheel,station and sector of the chamber
        int wh = chamberId.wheel();
        int st = chamberId.station();
//        int se = chamberId.sector();

        // Select the chambers in a given wheel in a given station
        if ( iwh == wh && ist == st ){


          // Get the vector of super layers in this chamber. Here one needs const_cast
          std::vector<const GeomDet*> tmpSLs = (*det)->components();
          for ( std::vector<const GeomDet*>::const_iterator it = tmpSLs.begin(); 
                                                           it != tmpSLs.end() ; it++ ){    
            GeomDet* tmpSL = const_cast< GeomDet* > ( *it ) ;
            theSLs.push_back( tmpSL );
          }

          // Create the alignable DT chamber 
          AlignableDTChamber* tmpDTChamber  = new AlignableDTChamber( theSLs );

 
          // Store the DT chambers in a given DT Station and Wheel
	  tmpDTChambersInStation.push_back( tmpDTChamber );
	  
	// End chamber selection
	} 

      // End loop over chambers
      }  
       
      // Store the DT chambers 
      theDTChambers.insert( theDTChambers.end(), tmpDTChambersInStation.begin(),
                            tmpDTChambersInStation.end() );

      // Create the alignable DT station with chambers in a given station and wheel 
      AlignableDTStation* tmpDTStation  = new AlignableDTStation( tmpDTChambersInStation );
     
      // Store the DT stations in a given wheel  
      tmpDTStationsInWheel.push_back( tmpDTStation );

      // Clear the temporary vector of chambers in a station
      tmpDTChambersInStation.clear();

    // End loop over stations
    }

    // Store The DT stations
      theDTStations.insert( theDTStations.end(), tmpDTStationsInWheel.begin(),
                            tmpDTStationsInWheel.end() );

    // Create the alignable DT wheel
    AlignableDTWheel* tmpWheel  = new AlignableDTWheel( tmpDTStationsInWheel );
     

    // Store the DT wheels  
    theDTWheels.push_back( tmpWheel );

    // Clear temporary vector of stations in a wheel
    tmpDTStationsInWheel.clear();


  // End loop over wheels   
  }    
          
  // Create the alignable Muon Barrel
  AlignableDTBarrel* tmpDTBarrel  = new AlignableDTBarrel( theDTWheels );  
  
  // Store the barrel
  theDTBarrel.push_back( tmpDTBarrel );

  // Store the barrel in the muon 
  theMuonComponents.push_back( tmpDTBarrel );


}



//--------------------------------------------------------------------------------------------------


void AlignableMuon::buildCSCEndcap( edm::ESHandle<CSCGeometry> pCSC  )
{
  
 LogDebug("Position") << "Constructing AlignableDTBarrel"; 


  // Temporary container for chambers in a given station
  std::vector<AlignableCSCChamber*>   tmpCSCChambersInStation;

  // Loop over endcaps ( 1..2 )
  for( int iec = 1 ; iec < 3 ; iec++ ){

    // Loop over stations ( 1..4 )
    for( int ist = 1 ; ist < 5 ; ist++ ){
  
      // Loop over geom CSC Chambers
      std::vector<GeomDet*> theLayers;
      for( std::vector<CSCChamber*>::const_iterator det = pCSC->chambers().begin();  
                                              det != pCSC->chambers().end(); ++det ){

        // Get the CSCDet ID
        CSCDetId cscId = (*det)->id();

        // Get chamber, station, ring, layer and endcap labels of the CSC chamber
        int ch = cscId.chamber();
        int st = cscId.station();
//        int ri = cscId.ring();
//        int la = cscId.layer();
        int ec = cscId.endcap();


        // Select the chambers in a given endcap in a given station
        if ( iec == ec && ist == st ){


          std::cout << "        " << std::endl ;
          std::cout << "        " << std::endl ;
          std::cout << "Chamber=" << ch << std::endl ;
          std::cout << "Station=" << st << std::endl ;
          std::cout << "Endcap =" << ec << std::endl ;
          std::cout << "        " << std::endl ;
          std::cout << "        " << std::endl ;
    

          // Get the vector of layers in this chamber. Here one needs const_cast
          std::vector<const GeomDet*> tmpLs = (*det)->components();
          for ( std::vector<const GeomDet*>::const_iterator it = tmpLs.begin();
                                                           it != tmpLs.end() ; it++ ){
            GeomDet* tmpL = const_cast< GeomDet* > ( *it ) ;
            theLayers.push_back( tmpL );
          }

        // End If chamber selection
	}

        // Create the alignable CSC chamber 
        AlignableCSCChamber* tmpCSCChamber  = new AlignableCSCChamber( theLayers );
  
        // Store the alignable CSC chambers
        tmpCSCChambersInStation.push_back( tmpCSCChamber );    

      // End loop over geom CSC chambers
      }

      // Store the alignable CSC chambers
      theCSCChambers.insert(  theCSCChambers.end(), tmpCSCChambersInStation.begin(),
                              tmpCSCChambersInStation.end() );    

      // Create the alignable CSC station with chambers in a given station 
      AlignableCSCStation* tmpCSCStation  = new AlignableCSCStation( tmpCSCChambersInStation );
     
      // Store the CSC stations in a given endcap  
      theCSCStations.push_back( tmpCSCStation );

      // Clear the temporary vector of chambers in station
      tmpCSCChambersInStation.clear();

    // End loop over stations
    }

    // Create the alignable CSC endcap 
    AlignableCSCEndcap* tmpEndcap  = new AlignableCSCEndcap( theCSCStations );
     
    // Store the CSC stations in a given endcap  
    theCSCEndcaps.push_back( tmpEndcap );

  // End loop over endcaps  
  }

  // Store the encaps in the muon components  
  theMuonComponents.insert(  theMuonComponents.end(), theCSCEndcaps.begin(), theCSCEndcaps.end() );    

    
}


//--------------------------------------------------------------------------------------------------
std::vector<Alignable*> AlignableMuon::DTChambers()
{
  std::vector<Alignable*> result;
  copy( theDTChambers.begin(), theDTChambers.end(), back_inserter(result) );
  return result;
}

//--------------------------------------------------------------------------------------------------
std::vector<Alignable*> AlignableMuon::DTStations()
{
  std::vector<Alignable*> result;
  copy( theDTStations.begin(), theDTStations.end(), back_inserter(result) );
  return result;
}


//--------------------------------------------------------------------------------------------------
std::vector<Alignable*> AlignableMuon::DTWheels()
{
  std::vector<Alignable*> result;
  copy( theDTWheels.begin(), theDTWheels.end(), back_inserter(result) );
  return result;
}

//--------------------------------------------------------------------------------------------------
std::vector<Alignable*> AlignableMuon::DTBarrel()
{
  std::vector<Alignable*> result ;
  copy( theDTBarrel.begin(), theDTBarrel.end(), back_inserter(result) );
  return result;
}

//--------------------------------------------------------------------------------------------------
std::vector<Alignable*> AlignableMuon::CSCChambers()
{
  std::vector<Alignable*> result;
  copy( theCSCChambers.begin(), theCSCChambers.end(), back_inserter(result) );
  return result;
}

//--------------------------------------------------------------------------------------------------
std::vector<Alignable*> AlignableMuon::CSCStations()
{
  std::vector<Alignable*> result;
  copy( theCSCStations.begin(), theCSCStations.end(), back_inserter(result) );
  return result;
}

//--------------------------------------------------------------------------------------------------
std::vector<Alignable*> AlignableMuon::CSCEndcaps()
{
  std::vector<Alignable*> result;
  copy( theCSCEndcaps.begin(), theCSCEndcaps.end(), back_inserter(result) );
  return result;
}


//--------------------------------------------------------------------------------------------------



