/** \file
 * 
 *
 * \author Ilaria Segoni
 *
 * modified by Borislav Pavlov - University of Sofia
 * modification to be used for upgrade and for "pseudodigi"
 *
 *
 */


#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include <iostream>

RPCDigi::RPCDigi (int strip, int bx) :
  strip_(strip),
  bx_(bx),
  time_(0),
  coordinateX_(0),
  coordinateY_(0),
  deltaTime_(0),
  deltaX_(0),
  deltaY_(0),
  hasTime_(false),
  hasX_(false),
  hasY_(false)
{}

RPCDigi::RPCDigi ():
  strip_(0),
  bx_(0),
  time_(0),
  coordinateX_(0),
  coordinateY_(0),
  deltaTime_(0),
  deltaX_(0),
  deltaY_(0),
  hasTime_(false),
  hasX_(false),
  hasY_(false)
{}


// Comparison
bool RPCDigi::operator == (const RPCDigi& digi) const {
  if ( strip_ != digi.strip() ||
       bx_    != digi.bx() ) return false;
  return true;
}

///Precedence operator
bool RPCDigi::operator<(const RPCDigi& digi) const{
  if(digi.bx() == this->bx())
    return digi.strip()<this->strip();
  else 
    return digi.bx()<this->bx();
}

std::ostream & operator<<(std::ostream & o, const RPCDigi& digi) {
  return o << " " << digi.strip()
           << " " << digi.bx();
}

int RPCDigi::strip() const { return strip_; }

int RPCDigi::bx() const { return bx_; }

void RPCDigi::print() const {
  std::cout << "Strip " << strip() 
	    << " bx " << bx() <<std::endl;
}

double RPCDigi::time() const 
{
  return time_;
}

double RPCDigi::coordinateX() const
{
  return coordinateY_;
}

double RPCDigi::coordinateY() const
{
  return coordinateY_;
}

bool RPCDigi::hasTime() const
{
  return hasTime_;
}

bool RPCDigi::hasX() const
{
  return hasX_;
}

bool RPCDigi::hasY() const
{
  return hasY_;
}

void RPCDigi::hasTime(bool has)
{
  hasTime_ = has;
}

void RPCDigi::hasX(bool has)
{
  hasX_ = has;
}


void RPCDigi::hasY(bool has)
{
  hasY_ = has;
}

double RPCDigi::deltaTime() const
{
  return deltaTime_;
}

double RPCDigi::deltaX() const
{
  return deltaX_;
}

double RPCDigi::deltaY() const
{
  return deltaY_;
}

void RPCDigi::setTime(double time)
{
  time_ = time;
}

void RPCDigi::setDeltaTime(double dt)
{
  deltaTime_ = dt;
}

void RPCDigi::setX(double x)
{
  coordinateX_ = x;
}

void RPCDigi::setY(double y)
{
  coordinateY_ = y;
}

void RPCDigi::setDeltaX(double dx)
{
  deltaX_ = dx;
}

void RPCDigi::setDeltaY(double dy)
{
  deltaY_ = dy;
}

bool RPCDigi::isPseudoDigi() const
{
  return hasX_ || hasY_ ;
}

