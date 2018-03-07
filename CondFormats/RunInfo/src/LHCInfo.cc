#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/Common/interface/TimeConversions.h"
#include <algorithm>
#include <iterator>
#include <vector>
#include <stdexcept>

//helper function: returns the positions of the bits in the bitset that are set (i.e., have a value of 1).
static std::vector<unsigned short> bitsetToVector( std::bitset<LHCInfo::bunchSlots+1> const & bs ) {
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
static std::string fillTypeToString( LHCInfo::FillTypeId const & fillType ) {
  std::string s_fillType( "UNKNOWN" );
  switch( fillType ) {
  case LHCInfo::UNKNOWN :
    s_fillType = std::string( "UNKNOWN" );
    break;
  case LHCInfo::PROTONS :
    s_fillType = std::string( "PROTONS" );
    break;
  case LHCInfo::IONS :
    s_fillType = std::string( "IONS" );
    break;
  case LHCInfo::COSMICS :
    s_fillType = std::string( "COSMICS" );
    break;
  case LHCInfo::GAP :
    s_fillType = std::string( "GAP" );
    break;
  default :
    s_fillType = std::string( "UNKNOWN" );
  }
  return s_fillType;
}

//helper function: returns the enum for particle types in string type
static std::string particleTypeToString( LHCInfo::ParticleTypeId const & particleType ) {
  std::string s_particleType( "NONE" );
  switch( particleType ) {
  case LHCInfo::NONE :
    s_particleType = std::string( "NONE" );
    break;
  case LHCInfo::PROTON :
    s_particleType = std::string( "PROTON" );
    break;
  case LHCInfo::PB82 :
    s_particleType = std::string( "PB82" );
    break;
  case LHCInfo::AR18 :
    s_particleType = std::string( "AR18" );
    break;
  case LHCInfo::D :
    s_particleType = std::string( "D" );
    break;
  case LHCInfo::XE54 :
    s_particleType = std::string( "XE54" );
    break;
  default :
    s_particleType = std::string( "NONE" );
  }
  return s_particleType;
}

LHCInfo::LHCInfo(): m_isData( false )
		  , m_intParams( ISIZE )
		  , m_floatParams( FSIZE )
		  , m_timeParams( TSIZE )
		  , m_stringParams( SSIZE )
{}

LHCInfo::LHCInfo( unsigned short const & lhcFill, bool const & fromData ): m_isData( fromData )
									 , m_intParams( ISIZE )
									 , m_floatParams( FSIZE )
									 , m_timeParams( TSIZE )
									 , m_stringParams( SSIZE )
{}

LHCInfo::~LHCInfo() {}

//reset instance
void LHCInfo::setFill( unsigned short const & lhcFill, bool const & fromData ) {
  m_isData = fromData;
  m_intParams.resize( ISIZE, std::vector<unsigned int>(1,0) );
  m_intParams[ LHC_FILL ][0] = lhcFill;
  m_floatParams.resize( FSIZE, std::vector<float>(1,0.));
  m_floatParams[ LUMI_PER_B ] = std::vector<float>();
  m_timeParams.resize( TSIZE, std::vector<unsigned long long>(1,0) );
  m_stringParams.resize( SSIZE, std::vector<std::string>() );
  m_stringParams[ INJECTION_SCHEME ] = std::vector<std::string>(1,"None");
  m_bunchConfiguration1.reset();
  m_bunchConfiguration2.reset();
}

namespace LHCInfoImpl {
  template <typename T> const T& getParams( const std::vector<T>& params, size_t index ){
    if( index >= params.size() ) throw std::out_of_range("Parameter with index "+std::to_string(index)+" is out of range.");
    return params[index];
  }  

  template <typename T> const T& getOneParam( const std::vector< std::vector<T> >& params, size_t index ){
    if( index >= params.size() ) throw std::out_of_range("Parameter with index "+std::to_string(index)+" is out of range.");
    const std::vector<T>& inner = params[index];
    if( inner.empty() ) throw std::out_of_range("Parameter with index "+std::to_string(index)+" has not value stored.");
    return inner[ 0 ];
  }

  template <typename T> void setOneParam( std::vector< std::vector<T> >& params, size_t index, const T& value ){
    if( index >= params.size() ) throw std::out_of_range("Parameter with index "+std::to_string(index)+" is out of range.");
    params[ index ] = std::vector<T>(1,value);
  }

  template <typename T> void setParams( std::vector<T>& params, size_t index, const T& value ){
    if( index >= params.size() ) throw std::out_of_range("Parameter with index "+std::to_string(index)+" is out of range.");
    params[ index ] = value;
  }

}

//getters
unsigned short const LHCInfo::fillNumber() const {
 return LHCInfoImpl::getOneParam( m_intParams, LHC_FILL );
}

bool const LHCInfo::isData() const {
  return m_isData;
}

unsigned short const LHCInfo::bunchesInBeam1() const {
  return LHCInfoImpl::getOneParam( m_intParams, BUNCHES_1 );
}

unsigned short const LHCInfo::bunchesInBeam2() const {
  return LHCInfoImpl::getOneParam( m_intParams, BUNCHES_2 );
}

unsigned short const LHCInfo::collidingBunches() const {
  return LHCInfoImpl::getOneParam( m_intParams, COLLIDING_BUNCHES );
}

unsigned short const LHCInfo::targetBunches() const {
  return LHCInfoImpl::getOneParam( m_intParams, TARGET_BUNCHES );
}

LHCInfo::FillTypeId const LHCInfo::fillType() const {
  return static_cast<FillTypeId>(LHCInfoImpl::getOneParam( m_intParams, FILL_TYPE ));
}

LHCInfo::ParticleTypeId const LHCInfo::particleTypeForBeam1() const {
  return static_cast<ParticleTypeId>(LHCInfoImpl::getOneParam( m_intParams, PARTICLES_1 ));
}

LHCInfo::ParticleTypeId const LHCInfo::particleTypeForBeam2() const {
  return static_cast<ParticleTypeId>(LHCInfoImpl::getOneParam( m_intParams, PARTICLES_2 ));
}

float const LHCInfo::crossingAngle() const {
  return LHCInfoImpl::getOneParam( m_floatParams, CROSSING_ANGLE );
}

float const LHCInfo::betaStar() const {
  return LHCInfoImpl::getOneParam( m_floatParams, BETA_STAR );
}

float const LHCInfo::intensityForBeam1() const {
  return LHCInfoImpl::getOneParam( m_floatParams, INTENSITY_1 );
}

float const LHCInfo::intensityForBeam2() const {
  return LHCInfoImpl::getOneParam( m_floatParams, INTENSITY_2 );
}

float const LHCInfo::energy() const {
  return LHCInfoImpl::getOneParam( m_floatParams, ENERGY );
}

float const LHCInfo::delivLumi() const {
  return LHCInfoImpl::getOneParam( m_floatParams, DELIV_LUMI );
}

float const LHCInfo::recLumi() const {
  return LHCInfoImpl::getOneParam( m_floatParams, REC_LUMI );
}

cond::Time_t const LHCInfo::createTime() const {
  return LHCInfoImpl::getOneParam( m_timeParams, CREATE_TIME );
}

cond::Time_t const LHCInfo::beginTime() const {
  return LHCInfoImpl::getOneParam(m_timeParams, BEGIN_TIME );
}

cond::Time_t const LHCInfo::endTime() const {
  return LHCInfoImpl::getOneParam(m_timeParams, END_TIME );
}

std::string const & LHCInfo::injectionScheme() const {
  return LHCInfoImpl::getOneParam(m_stringParams, INJECTION_SCHEME );
}

std::vector<float> const & LHCInfo::lumiPerBX() const {
  return LHCInfoImpl::getParams(m_floatParams, LUMI_PER_B );
}

//returns a boolean, true if the injection scheme has a leading 25ns
//TODO: parse the circulating bunch configuration, instead of the string.
bool LHCInfo::is25nsBunchSpacing() const {
  const std::string prefix( "25ns" );
  return std::equal( prefix.begin(), prefix.end(), injectionScheme().begin() );
}

//returns a boolean, true if the bunch slot number is in the circulating bunch configuration
bool LHCInfo::isBunchInBeam1( size_t const & bunch ) const {
  if( bunch == 0 )
    throw std::out_of_range( "0 not allowed" ); //CMS starts counting bunch crossing from 1!
  return m_bunchConfiguration1.test( bunch );
}

bool LHCInfo::isBunchInBeam2( size_t const & bunch ) const {
  if( bunch == 0 )
    throw std::out_of_range( "0 not allowed" ); //CMS starts counting bunch crossing from 1!
  return m_bunchConfiguration2.test( bunch );
}

//member functions returning *by value* a vector with all filled bunch slots
std::vector<unsigned short> LHCInfo::bunchConfigurationForBeam1() const {
  return bitsetToVector( m_bunchConfiguration1 );
}

std::vector<unsigned short> LHCInfo::bunchConfigurationForBeam2() const {
  return bitsetToVector( m_bunchConfiguration2 );
}

//setters
void LHCInfo::setBunchesInBeam1( unsigned short const & bunches ) {
  LHCInfoImpl::setOneParam( m_intParams, BUNCHES_1, static_cast<unsigned int>(bunches) );
}

void LHCInfo::setBunchesInBeam2( unsigned short const & bunches ) {
  LHCInfoImpl::setOneParam( m_intParams, BUNCHES_2, static_cast<unsigned int>(bunches) );
}

void LHCInfo::setCollidingBunches( unsigned short const & collidingBunches ) {
  LHCInfoImpl::setOneParam( m_intParams, COLLIDING_BUNCHES, static_cast<unsigned int>(collidingBunches) );
}

void LHCInfo::setTargetBunches( unsigned short const & targetBunches ) {
  LHCInfoImpl::setOneParam( m_intParams, TARGET_BUNCHES, static_cast<unsigned int>(targetBunches) );  
}

void LHCInfo::setFillType( LHCInfo::FillTypeId const & fillType ) {
  LHCInfoImpl::setOneParam( m_intParams, FILL_TYPE, static_cast<unsigned int>(fillType) );
}

void LHCInfo::setParticleTypeForBeam1( LHCInfo::ParticleTypeId const & particleType ) {
  LHCInfoImpl::setOneParam( m_intParams, PARTICLES_1, static_cast<unsigned int>(particleType) );
}

void LHCInfo::setParticleTypeForBeam2( LHCInfo::ParticleTypeId const & particleType ) {
  LHCInfoImpl::setOneParam( m_intParams, PARTICLES_2, static_cast<unsigned int>(particleType) );
}

void LHCInfo::setCrossingAngle( float const & angle ) {
  LHCInfoImpl::setOneParam( m_floatParams, CROSSING_ANGLE, angle );
}
  
void LHCInfo::setBetaStar( float const & betaStar ) {
  LHCInfoImpl::setOneParam( m_floatParams, BETA_STAR, betaStar );
}

void LHCInfo::setIntensityForBeam1( float const & intensity ) {
  LHCInfoImpl::setOneParam( m_floatParams, INTENSITY_1, intensity );
}

void LHCInfo::setIntensityForBeam2( float const & intensity ) {
  LHCInfoImpl::setOneParam( m_floatParams, INTENSITY_2, intensity );
}

void LHCInfo::setEnergy( float const & energy ) {
  LHCInfoImpl::setOneParam( m_floatParams, ENERGY, energy );
}

void LHCInfo::setDelivLumi( float const & delivLumi ) {
  LHCInfoImpl::setOneParam( m_floatParams, DELIV_LUMI, delivLumi );
}

void LHCInfo::setRecLumi( float const & recLumi ) {
  LHCInfoImpl::setOneParam( m_floatParams, REC_LUMI, recLumi );
}

void LHCInfo::setCreationTime( cond::Time_t const & createTime ) {
  LHCInfoImpl::setOneParam( m_timeParams, CREATE_TIME, createTime );
}

void LHCInfo::setBeginTime( cond::Time_t const & beginTime ) {
  LHCInfoImpl::setOneParam( m_timeParams, BEGIN_TIME, beginTime );
}

void LHCInfo::setEndTime( cond::Time_t const & endTime ) {
  LHCInfoImpl::setOneParam( m_timeParams, END_TIME, endTime );
}

void LHCInfo::setInjectionScheme( std::string const & injectionScheme ) {
  LHCInfoImpl::setOneParam( m_stringParams, INJECTION_SCHEME, injectionScheme );
}

void LHCInfo::setLumiPerBX( std::vector<float> const & lumiPerBX) {
  LHCInfoImpl::setParams( m_floatParams, LUMI_PER_B, lumiPerBX );
}

//sets all values in one go
void LHCInfo::setInfo( unsigned short const & bunches1
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
		       ,float const & delivLumi
		       ,float const & recLumi
		       ,cond::Time_t const & createTime
		       ,cond::Time_t const & beginTime
		       ,cond::Time_t const & endTime
		       ,std::string const & scheme
		       ,std::vector<float> const & lumiPerBX
		       ,std::bitset<bunchSlots+1> const & bunchConf1
		       ,std::bitset<bunchSlots+1> const & bunchConf2 ) {
  this->setBunchesInBeam1( bunches1 );
  this->setBunchesInBeam2( bunches2 );
  this->setCollidingBunches( collidingBunches );
  this->setTargetBunches( targetBunches );
  this->setFillType( fillType );
  this->setParticleTypeForBeam1( particleType1 );
  this->setParticleTypeForBeam2( particleType2 );
  this->setCrossingAngle( angle );
  this->setBetaStar( beta );
  this->setIntensityForBeam1( intensity1 );
  this->setIntensityForBeam2( intensity2 );
  this->setEnergy( energy );
  this->setDelivLumi( delivLumi );
  this->setRecLumi( recLumi );
  this->setCreationTime( createTime );
  this->setBeginTime( beginTime );
  this->setEndTime( endTime );
  this->setInjectionScheme( scheme );
  this->setLumiPerBX( lumiPerBX );
  this->setBunchBitsetForBeam1( bunchConf1 );
  this->setBunchBitsetForBeam2( bunchConf2 );
}

void LHCInfo::print( std::stringstream & ss ) const {
  ss << "LHC fill: " << fillNumber() << std::endl
     << "Bunches in Beam 1: " << bunchesInBeam1() << std::endl
     << "Bunches in Beam 2: " << bunchesInBeam2() << std::endl
     << "Colliding bunches at IP5: " << collidingBunches() << std::endl
     << "Target bunches at IP5: " << targetBunches() << std::endl
     << "Fill type: " << fillTypeToString( fillType() ) << std::endl
     << "Particle type for Beam 1: " << particleTypeToString( particleTypeForBeam1() ) << std::endl
     << "Particle type for Beam 2: " << particleTypeToString( particleTypeForBeam2() ) << std::endl
     << "Crossing angle (urad): " << crossingAngle() << std::endl
     << "Beta star (cm): " << betaStar() << std::endl
     << "Average Intensity for Beam 1 (number of charges): " << intensityForBeam1() << std::endl
     << "Average Intensity for Beam 2 (number of charges): " << intensityForBeam2() << std::endl
     << "Energy (GeV): " << energy() << std::endl
     << "Delivered Luminosity (max): " << delivLumi() << std::endl
     << "Recorded Luminosity (max): " << recLumi() << std::endl
     << "Creation time of the fill: " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( createTime() ) ) << std::endl
     << "Begin time of Stable Beam flag: " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( beginTime() ) ) << std::endl
     << "End time of the fill: " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( endTime() ) ) << std::endl
     << "Injection scheme as given by LPC: " << injectionScheme() << std::endl;
  ss << "Luminosity per bunch  (total " << lumiPerBX().size() << "): ";
  std::copy( lumiPerBX().begin(), lumiPerBX().end(), std::ostream_iterator<float>( ss, ", " ) );
  ss << std::endl;
  
  std::vector<unsigned short> bunchVector1 = this->bunchConfigurationForBeam1();
  std::vector<unsigned short> bunchVector2 = this->bunchConfigurationForBeam2();
  ss << "Bunches filled for Beam 1 (total " << bunchVector1.size() << "): ";
  std::copy( bunchVector1.begin(), bunchVector1.end(), std::ostream_iterator<unsigned short>( ss, ", " ) );
  ss << std::endl;
  ss << "Bunches filled for Beam 2 (total " << bunchVector2.size() << "): ";
  std::copy( bunchVector2.begin(), bunchVector2.end(), std::ostream_iterator<unsigned short>( ss, ", " ) );
  ss << std::endl;
}

//protected getters
std::bitset<LHCInfo::bunchSlots+1> const & LHCInfo::bunchBitsetForBeam1() const {
  return m_bunchConfiguration1;  
}

std::bitset<LHCInfo::bunchSlots+1> const & LHCInfo::bunchBitsetForBeam2() const {
  return m_bunchConfiguration2;  
}

//protected setters
void LHCInfo::setBunchBitsetForBeam1( std::bitset<LHCInfo::bunchSlots+1> const & bunchConfiguration ) {
  m_bunchConfiguration1 = bunchConfiguration;
}

void LHCInfo::setBunchBitsetForBeam2( std::bitset<LHCInfo::bunchSlots+1> const & bunchConfiguration ) {
  m_bunchConfiguration2 = bunchConfiguration;
}

std::ostream & operator<<( std::ostream & os, LHCInfo beamInfo ) {
  std::stringstream ss;
  beamInfo.print( ss );
  os << ss.str();
  return os;
}
