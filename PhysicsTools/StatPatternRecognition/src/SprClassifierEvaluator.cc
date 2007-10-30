// $Id: SprClassifierEvaluator.cc,v 1.1 2007/10/29 22:10:40 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierEvaluator.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedMultiClassLearner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprCoordinateMapper.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAverageLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedFisher.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedLogitR.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerPermutator.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"

#include <iostream>
#include <memory>
#include <algorithm>
#include <cassert>
#include <cmath>


using namespace std;


bool SprClassifierEvaluator::variableImportance(
			       const SprAbsFilter* data,
			       SprAbsTrainedClassifier* trained,
			       SprTrainedMultiClassLearner* mcTrained,
			       SprCoordinateMapper* mapper,
			       unsigned nPerm,
			       std::vector<NameAndValue>& lossIncrease)
{
  // sanity check
  if( data == 0 ) {
    cerr << "No data supplied for variableImportance." << endl;
    return false;
  }
  if( trained==0 && mcTrained==0 ) {
    cerr << "No classifiers provided for variableImportance." << endl;
    return false;
  }
  if( trained!=0 && mcTrained!=0 ) {
    cerr << "variableImportance cannot process both two-class " 
	 << "and multi-class learners." << endl;
    return false;
  }
  if( nPerm == 0 ) {
    cout << "No permutations requested. Will use one by default." << endl;
    nPerm = 1;
  }

  // check classes
  vector<SprClass> classes; 
  data->classes(classes); 
  if( classes.size() < 2 ) {
    cerr << "Classes have not been set." << endl;
    return false; 
  }
  vector<int> mcClasses;
  if( mcTrained != 0 ) 
    mcTrained->classes(mcClasses);

  // make loss
  auto_ptr<SprAverageLoss> loss;
  if( mcTrained != 0 )
    loss.reset(new SprAverageLoss(&SprLoss::correct_id));
  else
    loss.reset(new SprAverageLoss(&SprLoss::quadratic));
  if( trained != 0 ) {
    if(      trained->name() == "AdaBoost" ) {
      SprTrainedAdaBoost* specific
	= static_cast<SprTrainedAdaBoost*>(trained);
      specific->useNormalized();
    }
    else if( trained->name() == "Fisher" ) {
      SprTrainedFisher* specific
	= static_cast<SprTrainedFisher*>(trained);
      specific->useNormalized();
    }
    else if( trained->name() == "LogitR" ) {
      SprTrainedLogitR* specific
	= static_cast<SprTrainedLogitR*>(trained);
      specific->useNormalized();
    }
  }

  //
  // pass through all variables
  //
  vector<string> testVars;
  if( mcTrained != 0 )
    mcTrained->vars(testVars);
  else
    trained->vars(testVars);
  int N = data->size();
  SprIntegerPermutator permu(N);

  // make first pass without permutations
  for( int n=0;n<N;n++ ) {
    const SprPoint* p = (*data)[n];
    const SprPoint* mappedP = p;
    int icls = p->class_;
    if( mcTrained != 0 ) {
      if( find(mcClasses.begin(),mcClasses.end(),icls) == mcClasses.end() )
	continue;
    }
    else {
      if(      icls == classes[0] )
	icls = 0;
      else if( icls == classes[1] )
	icls = 1;
      else
	continue;
    }
    if( mapper != 0 ) mappedP = mapper->output(p);
    if( mcTrained != 0 )
      loss->update(icls,mcTrained->response(mappedP),data->w(n));
    else
      loss->update(icls,trained->response(mappedP),data->w(n));
    if(  mapper != 0 ) mapper->clear();
  }
  double nominalLoss = loss->value();

  //
  // loop over permutations
  //
  cout << "Will perform " << nPerm << " permutations per variable." << endl;
  int nVars = testVars.size();
  lossIncrease.clear();
  lossIncrease.resize(nVars);
  for( int d=0;d<nVars;d++ ) {
    cout << "Permuting variable " << testVars[d].c_str() << endl;

    // map this var
    int mappedD = d;
    if( mapper != 0 )
      mappedD = mapper->mappedIndex(d);
    assert( mappedD>=0 && mappedD<data->dim() );

    // pass through all points permuting them
    vector<double> losses(nPerm);
    double aveLoss = 0;
    for( int i=0;i<nPerm;i++ ) {

      // permute this variable
      vector<unsigned> seq;
      if( !permu.sequence(seq) ) {
        cerr << "variableImportance is unable to permute points." << endl;
        return false;
      }

      // pass through points
      loss->reset();
      for( int n=0;n<N;n++ ) {
        SprPoint p(*(*data)[n]);
        p.x_[mappedD] = (*data)[seq[n]]->x_[mappedD];
        const SprPoint* mappedP = &p;
        int icls = p.class_;
	if( mcTrained != 0 ) {
	  if( find(mcClasses.begin(),mcClasses.end(),icls) == mcClasses.end() )
	    continue;
	}
	else {
	  if(      icls == classes[0] )
	    icls = 0;
	  else if( icls == classes[1] )
	    icls = 1;
	  else
	    continue;
	}
        if( mapper != 0 ) mappedP = mapper->output(&p);
	if( mcTrained != 0 )
	  loss->update(icls,mcTrained->response(mappedP),data->w(n));
	else
	  loss->update(icls,trained->response(mappedP),data->w(n));
        if( mapper != 0 ) mapper->clear();
      }

      // store loss
      losses[i] = loss->value();
      aveLoss += losses[i];
    }// end loop over permutations

    // compute error
    aveLoss /= nPerm;
    double err = 0;
    for( int i=0;i<nPerm;i++ )
      err += (losses[i]-aveLoss)*(losses[i]-aveLoss);
    err /= nPerm;
    err = sqrt(err);

    // store values
    lossIncrease[d] = NameAndValue(testVars[d],
				   ValueWithError(aveLoss-nominalLoss,err));
  }// end loop over variables

  // exit
  return true;
}
