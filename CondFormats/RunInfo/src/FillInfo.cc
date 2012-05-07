#include "CondFormats/RunInfo/interface/FillInfo.h"
#include <algorithm>
#include <iterator>
#include <stdexcept>

FillInfo::FillInfo(): m_isData( false )
		    , m_lhcFill( 0 )
		    , m_bunches1( 0 )
		    , m_bunches2( 0 )
		    , m_collidingBunches( 0 )
		    , m_targetBunches( 0 )
		    , m_fillType( FillTypeId::UNKNOWN )
		    , m_particles1( ParticleTypeId::NONE )
		    , m_particles2( ParticleTypeId::NONE )
		    , m_crossingAngle( 0. )
		    , m_energy( 0. )
		    , m_createTime( 0 )
		    , m_beginTime( 0 )
		    , m_endTime( 0 )
		    , m_injectionScheme( "None" )
{}

FillInfo::FillInfo( unsigned short const & lhcFill, bool const & fromData ): m_isData( fromData )
									   , m_lhcFill( lhcFill )
									   , m_bunches1( 0 )
									   , m_bunches2( 0 )
									   , m_collidingBunches( 0 )
									   , m_targetBunches( 0 )
									   , m_fillType( FillTypeId::UNKNOWN )
									   , m_particles1( ParticleTypeId::NONE )
									   , m_particles2( ParticleTypeId::NONE )
									   , m_crossingAngle( 0. )
									   , m_energy( 0. )
									   , m_createTime( 0 )
									   , m_beginTime( 0 )
									   , m_endTime( 0 )
									   , m_injectionScheme( "None" )
{}

FillInfo::~FillInfo() {}

//reset instance
void FillInfo::setFill( unsigned short const & lhcFill, bool const & fromData ) {
  m_isData = fromData;
  m_lhcFill = lhcFill;
  m_bunches1 = 0;
  m_bunches2 = 0;
  m_collidingBunches = 0;
  m_targetBunches = 0;
  m_fillType = FillTypeId::UNKNOWN;
  m_particles1 = ParticleTypeId::NONE;
  m_particles2 = ParticleTypeId::NONE;
  m_crossingAngle = 0.;
  m_energy = 0.;
  m_createTime = 0;
  m_beginTime = 0;
  m_endTime = 0;
  m_injectionScheme = "None";
  m_bunchConfiguration1.reset();
  m_bunchConfiguration2.reset();
}

//getters
unsigned short const & FillInfo::fillNumber() const {
  return m_lhcFill;
}

bool const & FillInfo::isData() const {
  return m_isData;
}

unsigned short const & FillInfo::bunchesInBeam1() const {
  return m_bunches1;
}

unsigned short const & FillInfo::bunchesInBeam2() const {
  return m_bunches2;
}

unsigned short const & FillInfo::collidingBunches() const {
  return m_collidingBunches;
}

unsigned short const & FillInfo::targetBunches() const {
  return m_targetBunches;
}

FillInfo::FillTypeId const & FillInfo::fillType() const {
  return m_fillType;
}

FillInfo::ParticleTypeId const & FillInfo::particleTypeForBeam1() const {
  return m_particles1;
}

FillInfo::ParticleTypeId const & FillInfo::particleTypeForBeam2() const {
  return m_particles2;
}

float const & FillInfo::crossingAngle() const {
  return m_crossingAngle;
}

float const & FillInfo::intensityForBeam1() const {
  return m_intensity1;
}

float const & FillInfo::intensityForBeam2() const {
  return m_intensity2;
}

float const & FillInfo::energy() const {
  return m_energy;
}

cond::Time_t const & FillInfo::createTime() const {
  return m_createTime;
}

cond::Time_t const & FillInfo::beginTime() const {
  return m_beginTime;
}

cond::Time_t const & FillInfo::endTime() const {
  return m_endTime;
}

std::string const & FillInfo::injectionScheme() const {
  return m_injectionScheme;
}

std::bitset<FillInfo::bunchSlots+1> const & FillInfo::bunchBitsetForBeam1() const {
  return m_bunchConfiguration1;  
}

std::bitset<FillInfo::bunchSlots+1> const & FillInfo::bunchBitsetForBeam2() const {
  return m_bunchConfiguration2;  
}

//returns a boolean, true if the bunch slot number is in the circulating bunch configuration
bool FillInfo::isBunchInBeam1( size_t const & bunch ) const {
  if( bunch == 0 )
    throw std::out_of_range( "0 not allowed" ); //CMS starts counting bunch crossing from 1!
  return m_bunchConfiguration1.test( bunch );
}

bool FillInfo::isBunchInBeam2( size_t const & bunch ) const {
  if( bunch == 0 )
    throw std::out_of_range( "0 not allowed" ); //CMS starts counting bunch crossing from 1!
  return m_bunchConfiguration2.test( bunch );
}

/*
//member functions returning *by value* a vector with all filled bunch slots
std::vector<unsigned short> FillInfo::bunchConfiguration1ForBeam1() const {

}

std::vector<unsigned short> FillInfo::bunchConfigurationForBeam2() const {
  
}
*/

//setters
unsigned short & FillInfo::bunchesInBeam1() {
  return m_bunches1;
}

unsigned short & FillInfo::bunchesInBeam2() {
  return m_bunches2;
}

unsigned short & FillInfo::collidingBunches() {
  return m_collidingBunches;
}

unsigned short & FillInfo::targetBunches() {
  return m_targetBunches;
}

FillInfo::FillTypeId & FillInfo::fillType() {
  return m_fillType;
}

FillInfo::ParticleTypeId & FillInfo::particleTypeForBeam1() {
  return m_particles1;
}

FillInfo::ParticleTypeId & FillInfo::particleTypeForBeam2() {
  return m_particles2;
}

float & FillInfo::crossingAngle() {
  return m_crossingAngle;
}

float & FillInfo::intensityForBeam1() {
  return m_intensity1;
}

float & FillInfo::intensityForBeam2() {
  return m_intensity2;
}

float & FillInfo::energy() {
  return m_energy;
}

cond::Time_t & FillInfo::createTime() {
  return m_createTime;
}

cond::Time_t & FillInfo::beginTime() {
  return m_beginTime;
}

cond::Time_t & FillInfo::endTime() {
  return m_endTime;
}

std::string & FillInfo::injectionScheme() {
  return m_injectionScheme;
}

std::bitset<FillInfo::bunchSlots+1> FillInfo::bunchBitsetForBeam1() {
  return m_bunchConfiguration1;
}

std::bitset<FillInfo::bunchSlots+1> FillInfo::bunchBitsetForBeam2() {
  return m_bunchConfiguration2;
}

//sets all values in one go
void FillInfo::setBeamInfo( unsigned short const & bunches1
			    ,unsigned short const & bunches2
			    ,unsigned short const & collidingBunches
			    ,unsigned short const & targetBunches
			    ,FillTypeId const & fillType
			    ,ParticleTypeId const & particleType1
			    ,ParticleTypeId const & particleType2
			    ,float const & angle
			    ,float const & intensity1
			    ,float const & intensity2
			    ,float const & energy
			    ,cond::Time_t const & createTime
			    ,cond::Time_t const & beginTime
			    ,cond::Time_t const & endTime
			    ,std::string const & scheme
			    ,std::bitset<bunchSlots+1> const & bunchConf1
			    ,std::bitset<bunchSlots+1> const & bunchConf2 ) {
  this->bunchesInBeam1() = bunches1;
  this->bunchesInBeam2() = bunches2;
  this->collidingBunches() = collidingBunches;
  this->targetBunches() = targetBunches;
  this->fillType() = fillType;
  this->particleTypeForBeam1() = particleType1;
  this->particleTypeForBeam2() = particleType2;
  this->crossingAngle() = angle;
  this->intensityForBeam1() = intensity1;
  this->intensityForBeam2() = intensity2;
  this->energy() = energy;
  this->createTime() = createTime;
  this->beginTime() = beginTime;
  this->endTime() = endTime;
  this->injectionScheme() = scheme;
  this->bunchBitsetForBeam1() = bunchConf1;
  this->bunchBitsetForBeam2() = bunchConf2;
}

void FillInfo::print(std::stringstream & ss) const {
  ss << "LHC fill: " << m_lhcFill << std::endl
     << "Bunches in Beam 1: " << m_bunches1 << std::endl
     << "Bunches in Beam 2: " << m_bunches2 << std::endl
     << "Colliding bunches at IP5: " << m_collidingBunches << std::endl
     << "Target bunches at IP5: " << m_targetBunches << std::endl
     << "Fill type: " << m_fillType << std::endl
     << "Particle type for Beam 1: " << m_particles1 << std::endl
     << "Particle type for Beam 2: " << m_particles2 << std::endl
     << "Crossing angle (urad): " << m_crossingAngle << std::endl
     << "Average Intensity for Beam 1 (number of charges): " << m_intensity1 << std::endl
     << "Average Intensity for Beam 2 (number of charges): " << m_intensity2 << std::endl
     << "Energy (GeV): " << m_energy << std::endl
     << "Creation time of the fill: " << m_createTime << std::endl
     << "Begin time of Stable Beam flag: " << m_beginTime << std::endl
     << "End time of the fill: " << m_endTime << std::endl
     << "Injection scheme as given by LPC: " << m_injectionScheme << std::endl;
  /*
     << "Bunches filled for Beam 1: ";
  std::copy(m_bunchConfiguration1.begin(), m_bunchConfiguration1.end(), std::ostream_iterator<unsigned short>(ss, ", "));
  ss << std::endl;
  ss << "Bunches filled for Beam 2: ";
  std::copy(m_bunchConfiguration2.begin(), m_bunchConfiguration2.end(), std::ostream_iterator<unsigned short>(ss, ", "));
  ss << std::endl;
    */
}

std::ostream & operator<< (std::ostream & os, FillInfo fillInfo) {
	std::stringstream ss;
	fillInfo.print(ss);
	os << ss.str();
	return os;
}
