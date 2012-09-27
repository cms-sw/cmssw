#include "CondFormats/RunInfo/interface/FillInfo.h"
#include "CondFormats/Common/interface/TimeConversions.h"
#include <algorithm>
#include <iterator>
#include <stdexcept>

//helper function: returns the positions of the bits in the bitset that are set (i.e., have a value of 1).
std::vector<unsigned short> bitsetToVector( std::bitset<FillInfo::bunchSlots+1> const & bs ) {
  std::vector<unsigned short> vec;
  //reserve space only for the bits in the bitset that are set
  vec.reserve( bs.count() );
  for( size_t i = 0; i < bs.size(); ++i ) {
    if( bs.test( i ) )
      vec.push_back( (unsigned short)i );
  }
  return vec;
}

//helper function: returns the enum for fill types in string type
std::string fillTypeToString( FillInfo::FillTypeId const & fillType ) {
  std::string s_fillType( "UNKNOWN" );
  switch( fillType ) {
  case FillInfo::UNKNOWN :
    s_fillType = std::string( "UNKNOWN" );
    break;
  case FillInfo::PROTONS :
    s_fillType = std::string( "PROTONS" );
    break;
  case FillInfo::IONS :
    s_fillType = std::string( "IONS" );
    break;
  case FillInfo::COSMICS :
    s_fillType = std::string( "COSMICS" );
    break;
  case FillInfo::GAP :
    s_fillType = std::string( "GAP" );
    break;
  default :
    s_fillType = std::string( "UNKNOWN" );
  }
  return s_fillType;
}

//helper function: returns the enum for particle types in string type
std::string particleTypeToString( FillInfo::ParticleTypeId const & particleType ) {
  std::string s_particleType( "NONE" );
  switch( particleType ) {
  case FillInfo::NONE :
    s_particleType = std::string( "NONE" );
    break;
  case FillInfo::PROTON :
    s_particleType = std::string( "PROTON" );
    break;
  case FillInfo::PB82 :
    s_particleType = std::string( "PB82" );
    break;
  case FillInfo::AR18 :
    s_particleType = std::string( "AR18" );
    break;
  case FillInfo::D :
    s_particleType = std::string( "D" );
    break;
  case FillInfo::XE54 :
    s_particleType = std::string( "XE54" );
    break;
  default :
    s_particleType = std::string( "NONE" );
  }
  return s_particleType;
}

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
		    , m_betastar( 0. )
		    , m_intensity1( 0. )
		    , m_intensity2( 0. )
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
									   , m_betastar( 0. )
									   , m_intensity1( 0. )
									   , m_intensity2( 0. )
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
  m_betastar = 0.;
  m_intensity1 = 0;
  m_intensity2 = 0;
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

float const & FillInfo::betaStar() const {
  return m_betastar;
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

//member functions returning *by value* a vector with all filled bunch slots
std::vector<unsigned short> FillInfo::bunchConfigurationForBeam1() const {
  return bitsetToVector( m_bunchConfiguration1 );
}

std::vector<unsigned short> FillInfo::bunchConfigurationForBeam2() const {
  return bitsetToVector( m_bunchConfiguration2 );
}

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

float & FillInfo::betaStar() {
  return m_betastar;
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

std::bitset<FillInfo::bunchSlots+1> & FillInfo::bunchBitsetForBeam1() {
  return m_bunchConfiguration1;
}

std::bitset<FillInfo::bunchSlots+1> & FillInfo::bunchBitsetForBeam2() {
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
			    ,float const & beta
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
  this->betaStar() = beta;
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

void FillInfo::print( std::stringstream & ss ) const {
  ss << "LHC fill: " << m_lhcFill << std::endl
     << "Bunches in Beam 1: " << m_bunches1 << std::endl
     << "Bunches in Beam 2: " << m_bunches2 << std::endl
     << "Colliding bunches at IP5: " << m_collidingBunches << std::endl
     << "Target bunches at IP5: " << m_targetBunches << std::endl
     << "Fill type: " << fillTypeToString( m_fillType ) << std::endl
     << "Particle type for Beam 1: " << particleTypeToString( m_particles1 ) << std::endl
     << "Particle type for Beam 2: " << particleTypeToString( m_particles2 ) << std::endl
     << "Crossing angle (urad): " << m_crossingAngle << std::endl
     << "Beta star (cm): " << m_betastar << std::endl
     << "Average Intensity for Beam 1 (number of charges): " << m_intensity1 << std::endl
     << "Average Intensity for Beam 2 (number of charges): " << m_intensity2 << std::endl
     << "Energy (GeV): " << m_energy << std::endl
     << "Creation time of the fill: " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( m_createTime ) ) << std::endl
     << "Begin time of Stable Beam flag: " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( m_beginTime ) ) << std::endl
     << "End time of the fill: " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( m_endTime ) ) << std::endl
     << "Injection scheme as given by LPC: " << m_injectionScheme << std::endl;
  std::vector<unsigned short> bunchVector1 = this->bunchConfigurationForBeam1();
  std::vector<unsigned short> bunchVector2 = this->bunchConfigurationForBeam2();
  ss << "Bunches filled for Beam 1 (total " << bunchVector1.size() << "): ";
  std::copy( bunchVector1.begin(), bunchVector1.end(), std::ostream_iterator<unsigned short>( ss, ", " ) );
  ss << std::endl;
  ss << "Bunches filled for Beam 2 (total " << bunchVector2.size() << "): ";
  std::copy( bunchVector2.begin(), bunchVector2.end(), std::ostream_iterator<unsigned short>( ss, ", " ) );
  ss << std::endl;
}

std::ostream & operator<<( std::ostream & os, FillInfo fillInfo ) {
  std::stringstream ss;
  fillInfo.print( ss );
  os << ss.str();
  return os;
}
