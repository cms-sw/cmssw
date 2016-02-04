#ifndef ExpAlgo_AlgoImpl_h
#define ExpAlgo_AlgoImpl_h

#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"
#include "DetectorDescription/Base/interface/DDAlgoPar.h"
#include <string>


class AlgoPos;

//! implementation of an algorithm, non generated checking code.
/** 
  objects of this class must register themselves with the representation
  of the algorithm AlgoPos.
  
  All methods will be called appropriately by AlgoPos.
*/  
class AlgoImpl 
{ 
  friend class AlgoPos;
public:
  //! subclass must provide a similar constructor and call this one
  AlgoImpl(AlgoPos*, std::string label);
  
protected:  
  virtual ~AlgoImpl();
  
  //! subclass must calculate a translation std::vector
  /**
    depending on the current position curr_ in the range [start_,end_,incr_]
    and the user supplied parameters ParE_, ParS_
  */
  virtual DD3Vector translation() = 0;
  
  //! subclass must calculate a rotation matrix
  /**
    depending on the current position curr_ in the range [start_,end_,incr_]
    and the user supplied parameters ParE_, ParS_
  */
  virtual DDRotationMatrix rotation() = 0;
    
  //! subclass must check the supplied parameters ParE_, ParS_
  /**
    whether they are correct and should select this paricular algorithm.
    
    If the parameters are correct by should not select this particular
    algorithm, checkParamters must return false otherwise true.
    
    The std::string err_ is to be used to be extended with error information in
    case any errors have been detected. Error information must be attached
    to err_ because of the possibility of already contained error information.
    
    In case of errors: 
    If an DDException is thrown by the algorithm implementation, further processing of
    any other implementations of the algorithm will be stopped.
    If no exception is thrown, checkParamters must return false.
    It's preferable not to throw an exception in case of errors. The algorithm
    implementation will throw if all checkParamters() of all registered 
    algorithm implementations have returned false.
  */
  virtual bool checkParameters() = 0;
  
  //! stop the current iteration of the algorithm (for incr_==0 types of algorithms)
  /**
    terminate() should be called in translation() or rotation() whenever
    the algorithm detects its termination condition. The current iteration
    of the algorithm then is not taken into account for algorithmic positioning.
    
    If the algorithm is of type incr_ != 0 it will terminate automatically 
    after its range [start_,end_,incr_] has been covered unless the algorithm
    calls terminate() from within translation() or rotation().
    
    If the algorithm is of type incr_ == 0 the algorithm implementation 
    has to provide code in checkTermination() which must call terminate()
    when it detects a termination condition (depending on the invocation-count
    of the algorithm , ...)
  */
  void terminate();
  
  //! copy-number calculation
  /**
    In case incr_==0 it makes sense to overload this method, otherwise
    the invocation-count count_ will be returned as copy-number
    
    If incr_ !=0 the copy-number will be curr_, the actual position
    in the range [start_,end_,incr_], unless this methods is overloaded.
  */   
  virtual int copyno() const;
  
  //! for algorithms with incr_==0 the algorithm must check whether to terminate 
  /**
    Overload this function in case the algorithm is a 'incr_==0' type.
    In this case provide some code which checks using perhaps the value of
    count_ and/or supplied algorithm parameters to check whether terminate()
    has to be called or not. If terminate() is called, the current iteration
    of the algorithm is not taken into account!
    
    The default implementation will immidiately terminate the algorithm in
    case incr_==0. 
    
    In case of incr_!=0: checkTermination() is not called at all; the algorithm
    will terminate automatically when the specified range [start_, end_, incr_]
    has been covered or terminate() has been called from within
    translation() or rotation().
  */
  virtual void checkTermination();
   
  //! ahh, converts a double into a std::string ... yet another one of this kind!
  static std::string d2s(double x);
 
  parS_type & ParS_; 
  parE_type & ParE_;
  const int & start_;
  const int & end_; 
  const int & incr_;
  const int & curr_;
  const int & count_;
  bool & terminate_;
  std::string & err_;
  std::string label_;
};

#endif
