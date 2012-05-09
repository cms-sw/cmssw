#ifndef MEVarVector_hh
#define MEVarVector_hh

//
// Authors  : Gautier Hamel de Monchenault and Julie Malcles, Saclay 
//

#include <vector>
#include "MusEcal.hh"
#include "METimeInterval.hh"

class MEVarVector
{

public:

  MEVarVector( int size );
  virtual  ~MEVarVector();

  void addTime( ME::Time t );
  void setVal(  ME::Time t, int ii, float val, bool check=false );
  void setFlag( ME::Time t, int ii, bool flag, bool check=false );
  void setValAndFlag( ME::Time t, int ii, float val, bool flag, 
		      bool check=false );

  // get value of variable var by the time
  bool getValByTime( ME::Time time, int ii, 
		     float& val, bool& flag );
  
  // get times in interval
  void getTime(	std::vector< ME::Time >& time,
		const METimeInterval* timeInterval=0 );

  // get values of a variable for a given vector of Times
  void getValAndFlag( int ii,
		      const std::vector< ME::Time >& time,
		      std::vector< float >& val,
		      std::vector< bool >& flag );

  // get times and values of a variable in interval
  void getTimeValAndFlag( int ii,
			  std::vector< ME::Time >& time, 
			  std::vector< float >& val,
			  std::vector< bool >& flag,
			  const METimeInterval* timeInterval=0 );

private:

  int _size;
  MusEcal::VarVecTimeMap _map;

  ClassDef(MEVarVector,0) // MEVarVector 
};

#endif
