#ifndef RECOCALOTOOLS_NAVIGATION_CALONAVIGATOR_H
#define RECOCALOTOOLS_NAVIGATION_CALONAVIGATOR_H 1

#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

template <class T>
class CaloNavigator GCC11_FINAL {
 public:

  CaloNavigator(const T& home, const CaloSubdetectorTopology* topology) : myTopology_(topology) 
    {
      setHome(home);
    };

  /// set the starting position
  inline void setHome(const T& startingPoint);

  /// set the starting position
  inline void setTopology(const CaloSubdetectorTopology*);

  /// set the starting position
  const CaloSubdetectorTopology* getTopology() const
    {
      return myTopology_;
    }

  /// move the navigator back to the starting point
  void home() const ;

  /// get the current position
  T pos() const { return currentPoint_; }

  /// get the current position
  T operator*() const { return currentPoint_; } 

  /// move the navigator north
  T north() const 
    { 
      if ((myTopology_->north(currentPoint_)).size()==1)
	currentPoint_=(myTopology_->north(currentPoint_))[0];
      else
	currentPoint_=T(0);
      return currentPoint_;
    } ;

  /// move the navigator south
  T south()  const 
    { 
      if ((myTopology_->south(currentPoint_)).size()==1)
	currentPoint_=(myTopology_->south(currentPoint_))[0];
      else
	currentPoint_=T(0);
      return currentPoint_;
    } ;

  /// move the navigator east
  T east() const
    { 
      if ((myTopology_->east(currentPoint_)).size()==1)
	currentPoint_=(myTopology_->east(currentPoint_))[0];
      else
	currentPoint_=T(0);
      return currentPoint_;
    } ;

  /// move the navigator west
  T west() const
    { 
      if ((myTopology_->west(currentPoint_)).size()==1)
	currentPoint_=(myTopology_->west(currentPoint_))[0];
      else
	currentPoint_=T(0);
      return currentPoint_;
    } ;

  /// move the navigator west
  T up() const
    { 
      if ((myTopology_->up(currentPoint_)).size()==1)
	currentPoint_=(myTopology_->up(currentPoint_))[0];
      else
	currentPoint_=T(0);
      return currentPoint_;
    } ;

  /// move the navigator west
  T down() const
    { 
      if ((myTopology_->down(currentPoint_)).size()==1)
	currentPoint_=(myTopology_->down(currentPoint_))[0];
      else
	currentPoint_=T(0);
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
  
  const CaloSubdetectorTopology* myTopology_;
  mutable T startingPoint_, currentPoint_;
};

template <class T>
void CaloNavigator<T>::setHome(const T& startingPoint)
{
  startingPoint_=startingPoint;
  home();
}

template <class T>
inline
void CaloNavigator<T>::home() const
{
  currentPoint_=startingPoint_;
}

template <class T>
inline
void CaloNavigator<T>::setTopology(const CaloSubdetectorTopology* topology) 
{
  if (myTopology_ == 0)
    myTopology_=topology;
  else
    return;
}

#endif
