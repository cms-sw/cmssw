//$Id: SprFomCalculator.cc,v 1.4 2006/11/26 02:04:31 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprFomCalculator.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAverageLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <cassert>

using namespace std;

double SprFomCalculator::fom(const SprAbsFilter* data, 
			     const SprAbsTrainedClassifier* t,
			     const SprAbsTwoClassCriterion* crit, 
			     SprAverageLoss* loss,
			     const SprClass& cls0, const SprClass& cls1)
{
  // sanity check
  assert( data != 0 );
  assert( t != 0 );
  assert( cls0 != cls1 );
  assert( crit!=0 || loss!=0 );

  // init
  int size = data->size();
  double wcor0(0), wcor1(0), wmis0(0), wmis1(0);
  if( loss != 0 ) loss->reset();

  // loop thru data
  for( int i=0;i<size;i++ ) {
    const SprPoint* p = (*data)[i];
    double w = data->w(i);
    if( loss == 0 ) {
      if( t->accept(p) ) {
	if(      p->class_ == cls0 )
	  wmis0 += w;
	else if( p->class_ == cls1 )
	  wcor1 += w;
      }
      else {
	if(      p->class_ == cls0 )
	  wcor0 += w;
	else if( p->class_ == cls1 )
	  wmis1 += w;
      }
    }
    else {
      double r = t->response(p);
      if(      p->class_ == cls0 )
	loss->update(0,r,w);
      else if( p->class_ == cls1 )
	loss->update(1,r,w);
    }
  }

  // compute fom
  double fom = 0;
  if( loss == 0 ) 
    fom = crit->fom(wcor0,wmis0,wcor1,wmis1);
  else
    fom = loss->value();

  // exit
  return fom;
}
