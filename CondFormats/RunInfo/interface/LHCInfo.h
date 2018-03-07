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

  enum IntParamIndex { LHC_FILL = 0, BUNCHES_1 = 1, BUNCHES_2 = 2, COLLIDING_BUNCHES = 3, TARGET_BUNCHES = 4, FILL_TYPE = 5, PARTICLES_1 = 6, PARTICLES_2 = 7, ISIZE = 8 };
  enum FloatParamIndex { CROSSING_ANGLE = 0, BETA_STAR = 1, INTENSITY_1 = 2, INTENSITY_2 = 3, ENERGY = 4, DELIV_LUMI = 5, REC_LUMI = 7, LUMI_PER_B=8, FSIZE=9};
  enum TimeParamIndex { CREATE_TIME = 0, BEGIN_TIME = 1, END_TIME = 2, TSIZE=3};
  enum StringParamIndex { INJECTION_SCHEME = 0, SSIZE=1 };

  typedef FillType FillTypeId;
  typedef ParticleType ParticleTypeId;
  LHCInfo();
  LHCInfo( unsigned short const & lhcFill, bool const & fromData = true );
  ~LHCInfo();
  
  //constant static unsigned integer hosting the maximum number of LHC bunch slots
  static size_t const bunchSlots = 3564;
  
  //constant static unsigned integer hosting the available number of LHC bunch slots
  static size_t const availableBunchSlots = 2808;
  
  //reset instance
  void setFill( unsigned short const & lhcFill, bool const & fromData = true );
  
  //getters
  unsigned short const fillNumber() const;
  
  bool const isData() const;
  
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
  
  cond::Time_t const createTime() const;
  
  cond::Time_t const beginTime() const;
  
  cond::Time_t const endTime() const;
  
  std::string const & injectionScheme() const;
  
  std::vector<float> const & lumiPerBX() const;

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

  void setCreationTime( cond::Time_t const & createTime );
  
  void setBeginTime( cond::Time_t const & beginTime );
  
  void setEndTime( cond::Time_t const & endTime );
  
  void setInjectionScheme( std::string const & injectionScheme );
  
  void setLumiPerBX( std::vector<float> const & lumiPerBX);
  
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
		,cond::Time_t const & createTime
		,cond::Time_t const & beginTime
		,cond::Time_t const & endTime
		,std::string const & scheme
		,std::vector<float> const & lumiPerBX
		,std::bitset<bunchSlots+1> const & bunchConf1 
		,std::bitset<bunchSlots+1> const & bunchConf2 );
  
  //dumping values on output stream
  void print(std::stringstream & ss) const;
  
 protected:
  std::bitset<bunchSlots+1> const & bunchBitsetForBeam1() const;
  
  std::bitset<bunchSlots+1> const & bunchBitsetForBeam2() const;
  
  void setBunchBitsetForBeam1( std::bitset<bunchSlots+1> const & bunchConfiguration );
  
  void setBunchBitsetForBeam2( std::bitset<bunchSlots+1> const & bunchConfiguration );
  
 private:
  bool m_isData;
  std::vector<std::vector<unsigned int> > m_intParams;
  //unsigned short m_lhcFill;
  //unsigned short m_bunches1, m_bunches2, m_collidingBunches, m_targetBunches;
  //FillTypeId m_fillType;
  //ParticleTypeId m_particles1, m_particles2;
  std::vector<std::vector<float> > m_floatParams;
  //float m_crossingAngle, m_betastar, m_intensity1, m_intensity2, m_energy, m_delivLumi, m_recLumi;
  std::vector<std::vector<unsigned long long> > m_timeParams;
  //cond::Time_t m_createTime, m_beginTime, m_endTime;
  std::vector<std::vector<std::string> > m_stringParams;
  //std::string m_injectionScheme;
  //std::vector<std::vector<float> > m_farrayParams;
  //std::vector<std::vector<float> > m_iarrayParams;
  //std::vector<float> m_lumiPerBX;
  //BEWARE: since CMS counts bunches starting from one,
  //the size of the bitset must be incremented by one,
  //in order to avoid off-by-one
  std::bitset<bunchSlots+1> m_bunchConfiguration1, m_bunchConfiguration2;

 COND_SERIALIZABLE;
};

std::ostream & operator<<( std::ostream &, LHCInfo fillInfo );

#endif // CondFormats_RunInfo_LHCInfo_H
