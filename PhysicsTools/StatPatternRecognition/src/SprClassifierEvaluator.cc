// $Id: SprClassifierEvaluator.cc,v 1.2 2007/12/01 01:29:46 narsky Exp $

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
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

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
  unsigned int N = data->size();
  SprIntegerPermutator permu(N);

  // make first pass without permutations
  for( unsigned int n=0;n<N;n++ ) {
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
  unsigned int nVars = testVars.size();
  lossIncrease.clear();
  lossIncrease.resize(nVars);
  for( unsigned int d=0;d<nVars;d++ ) {
    cout << "Permuting variable " << testVars[d].c_str() << endl;

    // map this var
    int mappedD = d;
    if( mapper != 0 )
      mappedD = mapper->mappedIndex(d);
    assert( mappedD>=0 && mappedD<static_cast<int>(data->dim()) );

    // pass through all points permuting them
    vector<double> losses(nPerm);
    double aveLoss = 0;
    for( unsigned int i=0;i<nPerm;i++ ) {

      // permute this variable
      vector<unsigned> seq;
      if( !permu.sequence(seq) ) {
        cerr << "variableImportance is unable to permute points." << endl;
        return false;
      }

      // pass through points
      loss->reset();
      for( unsigned int n=0;n<N;n++ ) {
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
    for( unsigned int i=0;i<nPerm;i++ )
      err += (losses[i]-aveLoss)*(losses[i]-aveLoss);
    if( nPerm > 1 )
      err /= (nPerm-1);
    err = sqrt(err);

    // store values
    lossIncrease[d] = NameAndValue(testVars[d],
				   ValueWithError(aveLoss-nominalLoss,err));
  }// end loop over variables

  // exit
  return true;
}


bool SprClassifierEvaluator::variableInteraction(
				       const SprAbsFilter* data,
				       SprAbsTrainedClassifier* trained,
				       SprTrainedMultiClassLearner* mcTrained,
				       SprCoordinateMapper* mapper,
				       const char* vars,
				       unsigned nPoints,
				       std::vector<NameAndValue>& interaction,
				       int verbose)
{
  // sanity check
  if( data == 0 ) {
    cerr << "No data supplied for variableInteraction." << endl;
    return false;
  }
  if( trained==0 && mcTrained==0 ) {
    cerr << "No classifiers provided for variableInteraction." << endl;
    return false;
  }
  if( trained!=0 && mcTrained!=0 ) {
    cerr << "variableInteraction cannot process both two-class " 
	 << "and multi-class learners." << endl;
    return false;
  }
  if( nPoints > data->size() ) {
    cerr << "Number of points for integration " << nPoints 
	 << " cannot exceed the data size " << data->size() << endl;
    return false;
  }

  // set number of points and passes
  unsigned int nPass = 2;
  if( nPoints == 0 ) {
    nPoints = data->size();
    nPass = 1;
  }

  // get input vars
  vector<vector<string> > v_of_vars;
  SprStringParser::parseToStrings(vars,v_of_vars);

  // can only handle 1st and 2nd order interactions for now
  vector<string> svars;
  if( !v_of_vars.empty() && !v_of_vars[0].empty() )
    svars = v_of_vars[0];

  // get trained vars
  vector<string> testVars;
  if(      trained != 0 )
    trained->vars(testVars);
  else if( mcTrained != 0 )
    mcTrained->vars(testVars);
  assert( !testVars.empty() );
  unsigned dim        = testVars.size();
  unsigned dim_subset = svars.size();
  if( dim <= dim_subset ) {
    cerr << "Too many variables requested in variableInteraction." << endl;
    return false;
  }

  // map subset vars onto classifier vars
  vector<int> subsetIndex(dim_subset,-1);
  for( unsigned int d=0;d<dim_subset;d++ ) {
    vector<string>::const_iterator found 
      = find(testVars.begin(),testVars.end(),svars[d]);
    if( found == testVars.end() ) {
      cerr << "Variable " << svars[d].c_str() 
	   << " not found among trained variables in " 
	   << "variableInteraction." << endl;
      return false;
    }
    subsetIndex[d] = found - testVars.begin();
  }

  // find classifier vars not included in subset vars
  map<string,int> analyzeVarIndex;
  for( unsigned int d=0;d<dim;d++ ) {
    if( find(svars.begin(),svars.end(),testVars[d]) == svars.end() ) {
      if( !analyzeVarIndex.insert(pair<string,int>(testVars[d],d)).second ) {
	cerr << "Unable to insert into analyzeVarIndex." << endl;
	return false;
      }
    }
  }

  // use normalized output
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

  // init interaction
  interaction.clear();
  interaction.resize(dim,NameAndValue("UserSubset",ValueWithError(1,0)));

  // reduce data size for integration by randomly choosing
  //   the number of points defined by the user
  unsigned N = data->size();
  SprIntegerPermutator permu(N);

  //
  // two passes to get a rough error estimate
  //
  vector<vector<double> > pass(nPass,vector<double>(dim));
  for( unsigned int ipass=0;ipass<nPass;ipass++ ) {

    // permute indices
    vector<unsigned> indices;
    if( !permu.sequence(indices) ) {
    cerr << "Unable to permute input indices." << endl;
    return false;
    }
    double wtot = 0;
    for( unsigned int i=0;i<nPoints;i++ )
      wtot += data->w(indices[i]);

    //
    // compute correlation between F(S) and each variable in \S
    //
    for( map<string,int>::const_iterator iter=analyzeVarIndex.begin();
	 iter!=analyzeVarIndex.end();iter++ ) {

      // get var index in the classifier list
      unsigned int d = iter->second;
      assert( d < dim );
      if( verbose > 0 ) {
	cout << "Computing interaction for variable "
	     << testVars[d].c_str() << " at pass " << ipass+1 << endl;
      }
      
      // if no vars specified, use all vars but this one
      if( svars.empty() ) {
	dim_subset = dim-1;
	subsetIndex.clear();
	for( unsigned int k=0;k<dim;k++ ) {
	  if( k == d ) continue;
	  subsetIndex.push_back(k);
	}
      }
      
      // compute mean Fd and FS at each point
      vector<double> FS(nPoints,0), Fd(nPoints,0);
      for( unsigned int i=0;i<nPoints;i++ ) {
	if( verbose>1 && (i+1)%1000==0 )
	  cout << "Processing point " << i+1 << endl;
	int ii = indices[i];
	//double wi = data->w(ii);
	const SprPoint* pi = (*data)[ii];
	const SprPoint* mappedPi = pi;
	if( mapper != 0 ) mappedPi = mapper->output(pi);
	const vector<double>& xi = mappedPi->x_;
	vector<double> xi_subset(dim_subset);
	for( unsigned int k=0;k<dim_subset;k++ )
	  xi_subset[k] = xi[subsetIndex[k]];
	
	for( unsigned int j=0;j<nPoints;j++ ) {
	  int jj = indices[j];
	  const SprPoint* pj = (*data)[jj];
	  vector<double> x_S(pj->x_), x_d(pj->x_);
	  double wj = data->w(jj);
	  if( mapper != 0 ) mapper->map(pj->x_,x_S);
	  if( mapper != 0 ) mapper->map(pj->x_,x_d);
	  x_d[d] = xi[d];
	  for( unsigned int k=0;k<dim_subset;k++ )
	    x_S[subsetIndex[k]] = xi_subset[k];
	  if(      trained != 0 ) {
	    Fd[i] += wj * trained->response(x_d);
	    FS[i] += wj * trained->response(x_S);
	  }
	  else if( mcTrained != 0 ) {
	    Fd[i] += wj * mcTrained->response(x_d);
	    FS[i] += wj * mcTrained->response(x_S);
	  }
	}// end loop over j
	Fd[i] /= wtot;
	FS[i] /= wtot;
	
	// cleanup
	if( mapper != 0 ) mapper->clear();
      }// end loop over i
      
      // compute correlation between FS and Fd
      double FS_mean(0), Fd_mean(0);
      for( unsigned int i=0;i<nPoints;i++ ) {
	int ii = indices[i];
	double wi = data->w(ii);
	Fd_mean += wi*Fd[i];
	FS_mean += wi*FS[i];
      }
      Fd_mean /= wtot;
      FS_mean /= wtot;
      double var_S(0), var_d(0), cov(0);
      for( unsigned int i=0;i<nPoints;i++ ) {
	int ii = indices[i];
	double wi = data->w(ii);
	var_d += wi*pow(Fd[i]-Fd_mean,2);
	var_S += wi*pow(FS[i]-FS_mean,2);
	cov   += wi*(Fd[i]-Fd_mean)*(FS[i]-FS_mean);
      }
      var_d /= wtot;
      var_S /= wtot;
      cov   /= wtot;
      
      // set correlation
      if( var_d<SprUtils::eps() || var_S<SprUtils::eps() ) {
	cerr << "Variance too small: " << var_d << " " << var_S 
	     << " for variable " << testVars[d].c_str()
	     << ". Unable to compute variable interaction." << endl;
	pass[ipass][d] = 0;
      }
      else      
	pass[ipass][d] = cov/(sqrt(var_d)*sqrt(var_S));
    }// end loop over d
  }// end loop over ipass

  //
  // compute average over passes
  //
  for( map<string,int>::const_iterator iter=analyzeVarIndex.begin();
       iter!=analyzeVarIndex.end();iter++ ) {
    int d = iter->second;
    double mean = 0;
    for( unsigned int ipass=0;ipass<nPass;ipass++ )
      mean += pass[ipass][d];
    mean /= nPass;
    double var = 0;
    for( unsigned int ipass=0;ipass<nPass;ipass++ )
      var += pow(pass[ipass][d]-mean,2);
    if( nPass > 1 )
      var /= (nPass-1);
    double sigma = ( var>0 ? sqrt(var) : 0 );
    interaction[d] = NameAndValue(testVars[d],ValueWithError(mean,sigma));
  }

  // exit
  return true;
}
