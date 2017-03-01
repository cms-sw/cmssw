#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/Common/interface/Time.h"
#include <bitset>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

class FillInfo {
 public:
  enum FillType { UNKNOWN = 0, PROTONS = 1, IONS = 2, COSMICS = 3, GAP = 4 };
  enum ParticleType { NONE = 0, PROTON = 1, PB82 = 2, AR18 = 3, D = 4, XE54 = 5 };
  typedef FillType FillTypeId;
  typedef ParticleType ParticleTypeId;
  FillInfo();
  FillInfo( unsigned short const & lhcFill, bool const & fromData = true );
  ~FillInfo();
  
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
  
  cond::Time_t const createTime() const;
  
  cond::Time_t const beginTime() const;
  
  cond::Time_t const endTime() const;
  
  std::string const & injectionScheme() const;

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
  
  void setCreationTime( cond::Time_t const & createTime );
  
  void setBeginTime( cond::Time_t const & beginTime );
  
  void setEndTime( cond::Time_t const & endTime );
  
  void setInjectionScheme( std::string const & injectionScheme );
  
  //sets all values in one go
  void setBeamInfo( unsigned short const & bunches1
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
  unsigned short m_lhcFill;
  unsigned short m_bunches1, m_bunches2, m_collidingBunches, m_targetBunches;
  FillTypeId m_fillType;
  ParticleTypeId m_particles1, m_particles2;
  float m_crossingAngle, m_betastar, m_intensity1, m_intensity2, m_energy;
  cond::Time_t m_createTime, m_beginTime, m_endTime;
  std::string m_injectionScheme;
  //BEWARE: since CMS counts bunches starting from one,
  //the size of the bitset must be incremented by one,
  //in order to avoid off-by-one
  std::bitset<bunchSlots+1> m_bunchConfiguration1, m_bunchConfiguration2;

 COND_SERIALIZABLE;
};

std::ostream & operator<<( std::ostream &, FillInfo fillInfo );
