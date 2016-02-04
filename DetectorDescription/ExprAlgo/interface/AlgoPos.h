#ifndef ExprAlgo_AlgoPos_h
#define ExprAlgo_AlgoPos_h

#include <iostream>
#include <vector>
#include <string>

#include "DetectorDescription/Base/interface/DDAlgoPar.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
//#include "DetectorDescription/Core/src/DDNamedObject.h"


class AlgoImpl;
class AlgoCheck;

//! class for algorithmic positioning, represents an algorithm
/**
  Implementations of the algorithm (AlgoImpl objects) have to register
  with AlgoPos. At least one implementation must register, so that AlgoPos
  is a valid representation of an algorithm.
  In case of multiple registration of AlgoImpl-implementations, it must be assured 
  by the implementations  that given a set of user-supplied parameters only
  one or none of these implementations will execute the algorithm 
  (see AlgoImpl::checkParamters()).
*/
class AlgoPos
{
  friend class AlgoImpl;
public:  
  //! creates an algorithm named name 
  AlgoPos(AlgoCheck * check=0);
  //AlgoPos();
  
  //! destructor clears registered AlgoImpl
  ~AlgoPos();

  //! sets mandatory and optional parameters 
  void setParameters(int start, int end, int incr,
                     const parS_type &, const parE_type &);
    
  //! no further algorithm processing if go() returns false 
  bool go() const;
  
  //! update for next iteration of the algorithm, don't call it yourself! 
  void next();
    
  //! translation calculations 
  /** 
    Will only work, if one registered algorithm AlgoImpl has been selected
    by checkParameters() 
  */    
  DD3Vector translation();
  
  //! rotation calculations 
  /** 
    Will only work, if one registered algorithm AlgoImpl has been selected
    by checkParameters() 
  */    
  DDRotationMatrix rotation();
  
  //! copy-number calculation 
  int copyno() const;
  
  //! streams some information about common parameters and the name of the algorithm
  void stream(std::ostream & os) const;
  
  //! registers an implementation of the algorithm
  /**
    At least 1 implementation must be registered.
    
    In case multiple implementations are registered, only one checkParamters()
    member function of these is allowed to return true given a set of
    user-supplied parameters. The particular registered algorithm is then
    selected. If all checkParamters() of the registered AlgoImpls return false,
    an exception is raised.
  */  
  void registerAlgo(AlgoImpl *);
  
  //! return number of registered implementations
  size_t numRegistered();
  
  int start() const { return start_; }
  int end() const { return end_; }
  int incr() const { return incr_; }
  const parS_type & parS() const { return ParS_; }
  const parE_type & parE() const { return ParE_; }

protected:  
  //! terminates current algorithmic processing 
  /**
    current algorithm iteration will not result in a positioning,
    if terminate() was called from within translation() or rotation()
  */  
  void terminate();

  //! for algorithms with incr_==0 the algorithm must check whether to terminate 
  /**
    Overload this function in case the algorithm is a 'incr_==0' type.
    In this case provide some code which checks using perhaps the value of
    count_ and/or supplied algorithm parameters to check whether terminate()
    has to be called or not.
    
    Will only work, if one registered algorithm AlgoImpl has been selected
    by checkParameters() 

    The default implementation will immidiately terminate the algorithm in
    case incr_==0. 
    
    In case of incr_!=0: checkTermination() is not called then, the algorithm
    will terminate automatically when the specified range [start_, end_, incr_]
    has been covered.
  */
  void checkTermination();
  
  //! selection of one of the registered AlgoImpl 
  /**
    checkParamters() of the registered AlgoImpl objects will be called
    sequentially until either an error is detected in the parameters or
    true is returned. In the latter case this AlgoImpl is choosen.
    
    select() returns true as soon as a checkParamters() of a registered
    AlgoImpl returns true. Otherwise it returns false.
  */
  bool select();
  //bool checkParameters();

  //! range of the algorithm 
  /**
    The algorithm will be invoked like
    /c for(curr_=start_; curr_<=end_; curr_ += incr_) { algo code } // incr > 0
    /c for(curr_=end_; curr_>=start_; curr_ += incr_) { algo code } // incr < 0
    
    If incr_==0 the algorithm code will be invoked until term() is called
    from within the algorithm code. The iteration in wich term() is called
    will not result in a positioning.
  */
  int start_, end_, incr_;
  
  //! current position in the range of the algorithm 
  /**
    see doc of start_, end_, incr_ as well.
    
    In case of incr_==0, the value of curr_  is undefined.
  */
  int curr_;
  
  //! invocation count 
  /** 
    count_ will be set to 1 at the first invocation of the algorithm and
    will be increased by 1 at every subsequent invocation.
  */
  int count_;
  
  //! std::string valued parameters passed to the algorithm 
  /** 
    Before the first invocation of the algorithm ParS_ will be filled
    with the std::string-valued parameters passed to the algorithm implementation object
    AlgoImpl.     
  */  
  parS_type ParS_;
  
  //! double valued parameters passed to the algorithm 
  /**
    Before the first invocation of the algorithm ParS_ will be filled
    with the std::string-valued parameters passed to the algorithm implementation object
    AlgoImpl.     
  */
  parE_type ParE_;
  
  //! flag to signal whether the algorithm has finished (true) 
  bool terminate_;
  
  //! std::string to hold potential error messages found inside checkParameters()
  std::string err_;
  
  //! registry of algorithm implementation objects
  std::vector<AlgoImpl*> regAlgos_;
  
  //! selected algorithm implementation
  AlgoImpl * selAlgo_;
  
  //! checking object which checks the user-supplied parameters whether they fulfill their spec.
  AlgoCheck * checkAlgo_;
  
private:
};
#endif
