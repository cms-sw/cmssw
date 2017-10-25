#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurements.h"
#include "Alignment/CocoaDaq/interface/CocoaDaqReader.h"

CocoaDaqReader* CocoaDaqReader::theDaqReader = nullptr;

//----------------------------------------------------------------------
void CocoaDaqReader::SetDaqReader( CocoaDaqReader* reader ) 
{
  if( theDaqReader != nullptr ) {
    std::cerr << "!!FATAL ERROR CocoaDaqReader:: trying to instantiate two CocoaDaqReader " << std::endl;
    std::exception();
  }

  theDaqReader = reader; 
}

//----------------------------------------------------------------------
void CocoaDaqReader::BuildMeasurementsFromOptAlign( std::vector<OpticalAlignMeasurementInfo>& measList )
{

}
