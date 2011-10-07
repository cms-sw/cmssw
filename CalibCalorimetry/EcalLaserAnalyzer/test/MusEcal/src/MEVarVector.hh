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

  void getValFlagAndNorm( int ii, 
			    const vector< ME::Time >& time, 
			    vector< float >& val,
			    vector< bool >& flag,
			  double& norm	);

  // get times and values of a variable in interval
  void getTimeValAndFlag( int ii,
			  std::vector< ME::Time >& time, 
			  std::vector< float >& val,
			  std::vector< bool >& flag,
			  const METimeInterval* timeInterval=0 );
  
  // get times and values of a variable in interval
  void getTimeValFlagAndNorm( int ii,
			  std::vector< ME::Time >& time, 
			  std::vector< float >& val,
			  std::vector< bool >& flag,
			      double& norm,
			  const METimeInterval* timeInterval=0 );
  
  void getNormsInInterval(int ii, unsigned int nbuf, unsigned int nave,
					   const METimeInterval* timeInterval,
					   std::vector< double >& norm,
					   std::vector< bool >& normflag);
  
  void getNormsInInterval(int ivar,int irms, int inevt,
			  unsigned int nbuf, unsigned int nave, 
			  const METimeInterval* timeInterval,
			  std::vector< double >&  norm,
			  std::vector< double >&  enorm,
			  std::vector< bool >&  normflag);
  
  void getClosestValid( ME::Time timeref, int ii,  vector< ME::Time >& time, float &val, bool &flag );
  void getClosestValidInPast( ME::Time timeref, int ii,   ME::Time& time, float &val, bool &flag );
  void getClosestValidInFuture( ME::Time timeref, int ii,   ME::Time& time, float &val, bool &flag );

private:

  int _size;
  MusEcal::VarVecTimeMap _map;

  ClassDef(MEVarVector,0) // MEVarVector 
};

#endif
