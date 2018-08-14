#ifndef CondFormats_RunInfo_LHCInfo_H
#define CondFormats_RunInfo_LHCInfo_H

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/Common/interface/Time.h"
#include <bitset>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

class LHCInfo {
 public:
  enum FillType { UNKNOWN = 0, PROTONS = 1, IONS = 2, COSMICS = 3, GAP = 4 };
  enum ParticleType { NONE = 0, PROTON = 1, PB82 = 2, AR18 = 3, D = 4, XE54 = 5 };

  enum IntParamIndex { LHC_FILL = 0, BUNCHES_1 = 1, BUNCHES_2 = 2, COLLIDING_BUNCHES = 3, TARGET_BUNCHES = 4, FILL_TYPE = 5, PARTICLES_1 = 6, PARTICLES_2 = 7, LUMI_SECTION = 8, ISIZE = 9 };
  enum FloatParamIndex { CROSSING_ANGLE = 0, BETA_STAR = 1, INTENSITY_1 = 2, INTENSITY_2 = 3, ENERGY = 4, DELIV_LUMI = 5, REC_LUMI = 7, LUMI_PER_B = 8, BEAM1_VC = 9, BEAM2_VC = 10, BEAM1_RF = 11, BEAM2_RF = 12, INST_LUMI = 13, INST_LUMI_ERR = 14, FSIZE = 15};
  enum TimeParamIndex { CREATE_TIME = 0, BEGIN_TIME = 1, END_TIME = 2, TSIZE =3};
  enum StringParamIndex { INJECTION_SCHEME = 0, LHC_STATE = 1, LHC_COMMENT = 2, CTPPS_STATUS = 3, SSIZE =4};

  typedef FillType FillTypeId;
  typedef ParticleType ParticleTypeId;
  LHCInfo();
  LHCInfo( const LHCInfo& rhs );
  ~LHCInfo();

  LHCInfo* cloneFill() const;
  
  //constant static unsigned integer hosting the maximum number of LHC bunch slots
  static size_t const bunchSlots = 3564;
  
  //constant static unsigned integer hosting the available number of LHC bunch slots
  static size_t const availableBunchSlots = 2808;
  
  void setFillNumber( unsigned short lhcFill );
  
  //getters
  unsigned short const fillNumber() const;
  
  unsigned short const bunchesInBeam1() const;
  
  unsigned short const bunchesInBeam2() const;
  
  unsigned short const collidingBunches() const;
  
  unsigned short const targetBunches() const;
  
  FillTypeId const fillType() const;
  
  ParticleTypeId const particleTypeForBeam1() const;
  
  ParticleTypeId const particleTypeForBeam2() const;
  
  float const crossingAngle() const;

  float const betaStar() const;
  
  float const intensityForBeam1() const;
  
  float const intensityForBeam2() const;
  
  float const energy() const;
  
  float const delivLumi() const;
  
  float const recLumi() const;

  float const instLumi() const;

  float const instLumiError() const;
  
  cond::Time_t const createTime() const;
  
  cond::Time_t const beginTime() const;
  
  cond::Time_t const endTime() const;
  
  std::string const & injectionScheme() const;
  
  std::vector<float> const & lumiPerBX() const;
  
  std::string const & lhcState() const;
  
  std::string const & lhcComment() const;
  
  std::string const & ctppsStatus() const;
  
  unsigned int const & lumiSection() const;
  
  std::vector<float> const & beam1VC() const;

  std::vector<float> const & beam2VC() const;

  std::vector<float> const & beam1RF() const;

  std::vector<float> const & beam2RF() const;

  std::vector<float>& beam1VC();

  std::vector<float>& beam2VC();

  std::vector<float>& beam1RF();

  std::vector<float>& beam2RF();

  //returns a boolean, true if the injection scheme has a leading 25ns
  //TODO: parse the circulating bunch configuration, instead of the string.
  bool is25nsBunchSpacing() const;

  //returns a boolean, true if the bunch slot number is in the circulating bunch configuration
  bool isBunchInBeam1( size_t const & bunch ) const;
  
  bool isBunchInBeam2( size_t const & bunch ) const;
  
  //member functions returning *by value* a vector with all filled bunch slots
  std::vector<unsigned short> bunchConfigurationForBeam1() const;
  
  std::vector<unsigned short> bunchConfigurationForBeam2() const;
  
  //setters
  void setBunchesInBeam1( unsigned short const & bunches );
  
  void setBunchesInBeam2( unsigned short const & bunches );
  
  void setCollidingBunches( unsigned short const & collidingBunches );
  
  void setTargetBunches( unsigned short const & targetBunches );
  
  void setFillType( FillTypeId const & fillType );
  
  void setParticleTypeForBeam1( ParticleTypeId const & particleType );
  
  void setParticleTypeForBeam2( ParticleTypeId const & particleType );
  
  void setCrossingAngle( float const & angle );
  
  void setBetaStar( float const & betaStar );
  
  void setIntensityForBeam1( float const & intensity );
  
  void setIntensityForBeam2( float const & intensity );
  
  void setEnergy( float const & energy );
  
  void setDelivLumi( float const & delivLumi );

  void setRecLumi( float const & recLumi );

  void setInstLumi( float const& instLumi );

  void setInstLumiError( float const& instLumiError );

  void setCreationTime( cond::Time_t const & createTime );
  
  void setBeginTime( cond::Time_t const & beginTime );
  
  void setEndTime( cond::Time_t const & endTime );
  
  void setInjectionScheme( std::string const & injectionScheme );
  
  void setLumiPerBX( std::vector<float> const & lumiPerBX);
  
  void setLhcState( std::string const & lhcState);
  
  void setLhcComment( std::string const & lhcComment);

  void setCtppsStatus( std::string const & ctppsStatus);
  
  void setLumiSection( unsigned int const & lumiSection);
  
  void setBeam1VC( std::vector<float> const & beam1VC);
  
  void setBeam2VC( std::vector<float> const & beam2VC);
  
  void setBeam1RF( std::vector<float> const & beam1RF);
  
  void setBeam2RF( std::vector<float> const & beam2RF);
  
  //sets all values in one go
  void setInfo( unsigned short const & bunches1
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
	,float const & instLumi
	,float const & instLumiError
	,cond::Time_t const & createTime
	,cond::Time_t const & beginTime
	,cond::Time_t const & endTime
	,std::string const & scheme
	,std::vector<float> const & lumiPerBX
	,std::string const & lhcState
	,std::string const & lhcComment
	,std::string const & ctppsStatus
	,unsigned int const & lumiSection
	,std::vector<float> const & beam1VC
	,std::vector<float> const & beam2VC
	,std::vector<float> const & beam1RF
	,std::vector<float> const & beam2RF
	,std::bitset<bunchSlots+1> const & bunchConf1 
	,std::bitset<bunchSlots+1> const & bunchConf2 );

  bool equals( const LHCInfo& rhs ) const;

  bool empty() const;

  //dumping values on output stream
  void print(std::stringstream & ss) const;
  
  std::bitset<bunchSlots+1> const & bunchBitsetForBeam1() const;
  
  std::bitset<bunchSlots+1> const & bunchBitsetForBeam2() const;
  
  void setBunchBitsetForBeam1( std::bitset<bunchSlots+1> const & bunchConfiguration );
  
  void setBunchBitsetForBeam2( std::bitset<bunchSlots+1> const & bunchConfiguration );
  
 private:
  bool m_isData = false;
  std::vector<std::vector<unsigned int> > m_intParams;
  std::vector<std::vector<float> > m_floatParams;
  std::vector<std::vector<unsigned long long> > m_timeParams;
  std::vector<std::vector<std::string> > m_stringParams;
  std::bitset<bunchSlots+1> m_bunchConfiguration1, m_bunchConfiguration2;

 COND_SERIALIZABLE;
};

std::ostream & operator<<( std::ostream &, LHCInfo lhcInfo );

#endif // CondFormats_RunInfo_LHCInfo_H
