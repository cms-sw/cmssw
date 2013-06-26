#ifndef RECOCALOTOOLS_NAVIGATION_CALONAVIGATOR_H
#define RECOCALOTOOLS_NAVIGATION_CALONAVIGATOR_H 1

#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

template <class T, class TOPO=CaloSubdetectorTopology>
class CaloNavigator GCC11_FINAL {
 public:

  CaloNavigator(const T& home, const TOPO * topology) : myTopology_(topology) 
    {
      setHome(home);
    }

  /// set the starting position
  inline void setHome(const T& startingPoint);

  /// set the starting position
  inline void setTopology(const TOPO *);

  /// set the starting position
  const TOPO * getTopology() const
    {
      return myTopology_;
    }

  /// move the navigator back to the starting point
  inline void home() const ;

  /// get the current position
  T pos() const { return currentPoint_; }

  /// get the current position
  T operator*() const { return currentPoint_; } 

  /// move the navigator north
  T north() const 
    { 
     currentPoint_=myTopology_->goNorth(currentPoint_);
     return currentPoint_;
    } ;

  /// move the navigator south
  T south()  const 
    { 
     currentPoint_=myTopology_->goSouth(currentPoint_);
      return currentPoint_;
    } ;

  /// move the navigator east
  T east() const
    { 
     currentPoint_=myTopology_->goEast(currentPoint_);
     return currentPoint_;
    } ;

  /// move the navigator west
  T west() const
    { 
     currentPoint_=myTopology_->goWest(currentPoint_);
     return currentPoint_;
    } ;

  /// move the navigator west
  T up() const
    { 
      currentPoint_=myTopology_->goUp(currentPoint_);
      return currentPoint_;
    } ;

  /// move the navigator west
  T down() const
    { 
     currentPoint_=myTopology_->goDown(currentPoint_);
     return currentPoint_;
    } ;

  /// Free movement of arbitray steps
  T offsetBy(int deltaX, int deltaY) const
    {
      for(int x=0; x < abs(deltaX) && currentPoint_ != T(0); x++)
	{
	  if(deltaX > 0) east();
	  else           west();
	}

      for(int y=0; y < abs(deltaY) && currentPoint_ != T(0); y++)
	{
	  if(deltaY > 0) north();
	  else           south();
	}

      return currentPoint_;

    }

 protected:
  
  const TOPO * myTopology_;
  mutable T startingPoint_, currentPoint_;
};

template <class T, class TOPO>
inline
void CaloNavigator<T,TOPO>::setHome(const T& startingPoint)
{
  startingPoint_=startingPoint;
  home();
}

template <class T, class TOPO>
inline
void CaloNavigator<T,TOPO>::home() const
{
  currentPoint_=startingPoint_;
}

template <class T, class TOPO>
inline
void CaloNavigator<T,TOPO>::setTopology(const TOPO * topology) 
{
  if (myTopology_ == 0)
    myTopology_=topology;
  else
    return;
}

#endif
