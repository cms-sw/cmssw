// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


// Geometry interface
#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerStringToEnum.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"

// Tracker components
#include "Alignment/MuonAlignment/interface/AlignableDTChamber.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCChamber.h"
#include "Alignment/MuonAlignment/interface/AlignableDTStation.h"
#include "Alignment/MuonAlignment/interface/AlignableCSCStation.h"
#include "Alignment/MuonAlignment/interface/AlignableDTWheel.h"
#include "Alignment/MuonAlignment/interface/AlignableMuBarrel.h"
#include "Alignment/MuonAlignment/interface/AlignableMuEndCap.h"

#include "Alignment/MuonAlignment/interface/AlignableMuon.h"


//--------------------------------------------------------------------------------------------------
AlignableMuon::AlignableMuon( const edm::Event& iEvent, const edm::EventSetup& iSetup  )
{

  // The XML geometry is accessed as in DTGeometry and CSC Geometry classes.
  // Since this geometry does not contain exactly the same structures we
  // need in alignment, we implement the alignable hierarchy here.
  //

  edm::LogInfo("AlignableMuon") << "Constructing alignable muon objects"; 
  

  // Get DTGeometry pointer
  iSetup.get<MuonGeometryRecord>().get( pDT );     

  // Build the muon barrel
  buildMuBarrel( pDT );

  // Get CSCGeometry pointer
  iSetup.get<MuonGeometryRecord>().get( pCSC ); 

  // Get the vector of CSC chambers;
  const std::vector<CSCChamber*> theCSCChambers = pCSC->chambers()

  // Build the muon end caps
  buildMuEndCap( pCSC );

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
void AlignableMuon::buildMuBarrel( edm::ESHandle<DTGeometry> pDT  )
{
  
 LogDebug("Position") << "Constructing AlignableMuBarrel"; 

  // Get the vector of DT chambers;
  const std::vector<DTChamber*> theDTChambers = pDT->chambers()

  // Temporary container for chambers in a given station
  std::vector<AlignableDTChamber*>   tmpDTChambersInStation;

  // Loop over wheels
  for( int iwh = -2 ; iwh =< 2 ; iwh++ ){

    // Loop over stations
    for( int ist = 1 ; ist =< 4 ; ist++ ){
  
      // Loop over geom DT Chambers
      for(vector<DTChamber*>::const_iterator det = pDT->chambers().begin(); 
                                             det != pDT->chambers().end(); ++det){
        // Get the chamber ID
        DTChamberId chamberId = (*det)->id(); 

        // Get wheel,station and sector of the chamber
        int wh = chamberId.wheel();
        int st = chamberId.station();
        int se = chamberId.sector();

        // Select the chambers in a given wheel in a given station
        if ( iwh == wh && ist == st ){

          // Get the GeomDet chamber corresponding to this ID
          const GeomDet* gdetc = det->idToDet(id);

          // Get the vector of super layers in this chamber
          const std::vector<DTSuperLayer*> &theSL = det->superLayers();
      
          // Create the alignable DT chamber 
          AlignableDTChamber* tmpDTChamber  = new AlignableDTChamber(theSL);

          // Store the DT chambers in a given DT Station
	  tmpDTChambersInStation.push_back( tmpDTChamber );
	  
	// End chamber selection
	} 

      // End loop over chambers
      }  
       
      // Store the DT chambers 
      theDTChambers.insert( theDTChambers.end(), tmpDTChambersInStation.begin(),
                            tmpDTChambersInStation.end() );

      // Create the alignable DT station with chambers in a given station 
      AlignableDTStation* tmpDTStation  = new AlignableDTStation( tmpDTChambersInStation );
     
      // Store the DT stations in a given wheel  
      theDTStations.push_back( tmpDTStation );

      // Clear the temporary vector of chambers
      tmpDTChambersInStation.clear();

    // End loop over stations
    }

    // Create the alignable DT station 
    AlignableDTWheel* tmpWheel  = new AlignableDTWheel( theDTStations );
     
    // Store the DT stations in a given wheel  
    theDTWheels.push_back( tmpWheel );

  // End loop over wheels   
  }    
          
  // Create the alignable Muon Barrel
  theMuBarrel  = new AlignableMuBarrel( theDTWheels );  
  
  // Store the barrel 
  theMuonComponents.push_back( theMuBarrel );


}



//--------------------------------------------------------------------------------------------------


void AlignableMuon::buildMuEndCap( edm::ESHandle<CSCGeometry> pCSC  )
{
  
 LogDebug("Position") << "Constructing AlignableMuBarrel"; 

  // Get the vector of CSC chambers;
  const std::vector<CSCChamber*> theCSCChambers = pCSC->chambers()

  // Temporary container for chambers in a given station
  std::vector<AlignableCSCChamber*>   tmpCSCChambersInStation;

  // Loop over endcaps
  for( int iec = 1 ; iec =< 2 ; iec++ ){

    // Loop over stations
    for( int ist = 1 ; ist =< 4 ; ist++ ){
  
      // Loop over geom CSC Chambers
      for(vector<CSCChamber*>::const_iterator det = pCSC->chambers().begin(); 
                                              det != pCSC->chambers().end(); ++det){

        // Select the chambers in a given endcap in a given station
        if ( iec == ec && ist == st ){

          // Get the CSCDet ID
          CSCDetId cscId = (*det)->id(); 

          // Get chamber, station, ring, layer and endcap labels of the CSC chamber
          int ch = cscId.chamber();
          int st = cscId.station();
          int ri = cscId.ring();
          int la = cscId.layer();
          int ec = cscId.endcap();


          std::cout << "        " << std::endl ;
          std::cout << "        " << std::endl ;
          std::cout << "Chamber=" << ch << std::endl ;
          std::cout << "Station=" << st << std::endl ;
          std::cout << "Ring   =" << ri << std::endl ;
          std::cout << "Layer  =" << la << std::endl ;
          std::cout << "Endcap =" << es << std::endl ;
          std::cout << "        " << std::endl ;
          std::cout << "        " << std::endl ;
    

          // Get the GeomDet chamber corresponding to this ID
          const GeomDet* gdetc = det->idToDet(id);

          // Get the vector of layers(1..6) in this chamber
          const std::vector<CSCLayer*> theLayers;
          for( int ii = 1 ; ii=<6 ; ii++ ){
            theLayers.push_back( (det*)->layer(ii) );
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

      // Clear the temporary vector of chambers
      tmpCSCChambersInStation.clear();

    // End loop over stations
    }

    // Create the alignable CSC endcap 
    AlignableCSCEndCap* tmpEndCap  = new AlignableCSCEndCap( theCSCStations );
     
    // Store the CSC stations in a given endcap  
    theMuEndCaps.push_back( tmpEndCap );

  // End loop over wheels   
  }

  // Store the encaps in the muon components  
  theMuonComponents.insert(  theMuonComponents.end(), theMuEndCaps.begin(), theMuEndCaps.end() );    

    
}


//--------------------------------------------------------------------------------------------------
