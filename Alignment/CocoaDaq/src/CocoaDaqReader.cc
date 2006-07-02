#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurements.h"
#include "Alignment/CocoaDaq/interface/CocoaDaqReader.h"

CocoaDaqReader* CocoaDaqReader::theDaqReader = 0;

//----------------------------------------------------------------------
void CocoaDaqReader::SetDaqReader( CocoaDaqReader* reader ) 
{
  if( theDaqReader != 0 ) {
    std::cerr << "!!FATAL ERROR CocoaDaqReader:: trying to instantiate two CocoaDaqReader " << std::endl;
    std::exception();
  }

  theDaqReader = reader; 
}

//----------------------------------------------------------------------
void CocoaDaqReader::BuildMeasurementsFromOptAlign( std::vector<OpticalAlignMeasurementInfo>& measList )
{

}
