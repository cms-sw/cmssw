//$Id: SprBoxFilter.cc,v 1.2 2007/09/21 22:32:09 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBoxFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <iostream>
#include <string>
#include <algorithm>

using namespace std;


bool SprBoxFilter::pass(const SprPoint* p) const
{
  assert( p != 0 );
  for( SprBox::const_iterator iter=box_.begin();iter!=box_.end();iter++ ) {
    unsigned d = iter->first;
    if( d < p->dim() ) {
      const SprInterval& range = iter->second;
      double r = (p->x_)[d];
      if( r>range.first && r<range.second )
        continue;
      else
        return false;
    }
  }
  return true;
}


bool SprBoxFilter::setRange(int d, const SprInterval& range)
{
  // sanity check
  if( d < 0 ) {
    cerr << "Index out of range for SprBoxFilter::setRange " << d << endl;
    return false;
  }

  // find out if a cut on this dimension exists already
  SprBox::iterator iter = box_.find(d);
  if( iter == box_.end() )
    box_.insert(pair<const unsigned,SprInterval>(d,range));
  else
    iter->second = range;

  // exit
  return true;
}


SprInterval SprBoxFilter::range(int d) const
{
  // sanity check
  if( d < 0 )
    return SprInterval(SprUtils::min(),SprUtils::max());

  // find the cut
  SprBox::const_iterator iter = box_.find(d);

  // if not found, infty range
  if( iter == box_.end() )
    return SprInterval(SprUtils::min(),SprUtils::max());

  // exit
  return iter->second;
}


bool SprBoxFilter::setBox(const std::vector<SprInterval>& box)
{
  if( !this->reset() ) {
    cerr << "Unable to reset SprBoxFilter." << endl;
    return false;
  }
  for( unsigned int d=0;d<box.size();d++ )
    box_.insert(pair<const unsigned,SprInterval>(d,box[d]));
  return true;
}
