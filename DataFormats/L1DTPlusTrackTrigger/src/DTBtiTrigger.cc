/*! \class DTBtiTrigger
 *  \author Ignazio Lazzizzera
 *  \author Sara Vanini
 *  \brief used to store BTI information within DT TP seed creation
 *  \date 2009, Feb 2
 */

#include "DataFormats/L1DTPlusTrackTrigger/interface/DTBtiTrigger.h"
#include <sstream>

/// Constructors
DTBtiTrigger::DTBtiTrigger() :
  DTBtiTrigData(){}

DTBtiTrigger::DTBtiTrigger( const DTBtiTrigData& bti ) :
  DTBtiTrigData(bti),
  _position(Global3DPoint()),
  _direction(Global3DVector())
{
  _wheel      = this->wheel();
  _station    = this->station();
  _sector     = this->sector();
  _superLayer = this->btiSL();
}

DTBtiTrigger::DTBtiTrigger( const DTBtiTrigData& bti, 
                            Global3DPoint position,
                            Global3DVector direction ) : 
  DTBtiTrigData(bti),
  _position(position),
  _direction(direction)
{
  _wheel      = this->wheel();
  _station    = this->station();
  _sector     = this->sector();
  _superLayer = this->btiSL();
}

/// Debug function
std::string DTBtiTrigger::sprint() const
{
  std::ostringstream outString;
  outString << "  wheel "    << this->wheel() 
            << " station "   << this->station() 
            << " sector "    << this->sector() 
            << " SL "        << this->btiSL() 
            << " Nr "        << this->btiNumber() << std::endl;
  outString << "  step "     << this->step() 
            << " code "      << this->code() 
            << " K "         << this->K() 
            << " X "         << this->X() << std::endl;
  outString << "  position " << this->cmsPosition() << std::endl;
  outString << "  direction" << this->cmsDirection() << std::endl;
  return outString.str();
}

