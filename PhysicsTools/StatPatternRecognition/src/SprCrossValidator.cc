//$Id: SprCrossValidator.cc,v 1.2 2007/09/21 22:32:09 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprCrossValidator.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAverageLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprFomCalculator.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerPermutator.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"

#include <list>
#include <iostream>

using namespace std;


SprCrossValidator::~SprCrossValidator()
{
  for( unsigned int i=0;i<samples_.size();i++ )
    delete samples_[i];
}


bool SprCrossValidator::divide(unsigned nPieces)
{
  // sanity check
  unsigned size = data_->size();
  if( nPieces > size ) {
    cerr << "Too many pieces requested for cross-validation: " 
	 << nPieces << ">" << size << endl;
    return false;
  }

  // randomize point indices
  vector<unsigned> index;
  SprIntegerPermutator permu(size);
  if( !permu.sequence(index) ) {
    cerr << "CrossValidator is unable to randomize indices." << endl;
    return false;
  }
  else
    cout << "Indices for cross-validation permuted." << endl;

  // fill subsamples
  unsigned nupdate = size/nPieces;
  for( unsigned i=0;i<nPieces;i++ ) {
    // subsamples own SprData which does not own points
    SprData* sample = data_->emptyCopy();
    vector<SprClass> classes;
    data_->classes(classes);
    samples_[i] = new SprEmptyFilter(sample,classes,true);

    //make subsample
    vector<double> w;
    for( unsigned j=0;j<nupdate;j++ ) {
      unsigned icurr = index[j+i*nupdate];
      sample->uncheckedInsert((*data_)[icurr]);
      w.push_back(data_->w(icurr));
    }

    // set weights
    if( !samples_[i]->setWeights(w) ) {
      cerr << "Unable to set weights for subsample " << i << endl;
      return false;
    }

    // sanity check
    assert( !samples_[i]->empty() );

    // message
    cout << "Obtained subsample " << i << " for cross-validation." << endl;
  }

  // exit
  return true;
}


bool SprCrossValidator::validate(const SprAbsTwoClassCriterion* crit,
				 SprAverageLoss* loss,
				 const std::vector<SprAbsClassifier*>& 
				 classifiers,
				 const SprClass& cls0, const SprClass& cls1,
				 const SprCut& cut,
				 std::vector<double>& crossFom,
				 int verbose) 
  const
{
  // print out
  if( verbose > 0 ) {
    cout << "Will cross-validate using " 
	 << samples_.size() << " subsamples: " << endl;
    for( unsigned int i=0;i<samples_.size();i++ ) {
      cout << "Subsample " << i 
	   << "  W1=" << samples_[i]->weightInClass(cls1)
	   << "  W0=" << samples_[i]->weightInClass(cls0)
	   << "  N1=" << samples_[i]->ptsInClass(cls1)
	   << "  N0=" << samples_[i]->ptsInClass(cls0) << endl;
    }
  }

  // sanity check
  assert( !classifiers.empty() && !samples_.empty() );

  // make a local copy of data
  SprEmptyFilter data(data_);

  // init
  crossFom.clear();
  crossFom.resize(classifiers.size());

  // loop over classifiers
  for( unsigned int ic=0;ic<classifiers.size();ic++ ) {
    SprAbsClassifier* c = classifiers[ic];
    assert( c != 0 );

    // message
    cout << "Cross-validator processing classifier " << ic << endl;

    // init fom vector
    vector<double> fom;

    // loop over subsamples
    for( unsigned int i=0;i<samples_.size();i++ ) {
      // message
      cout << "Cross-validator processing sub-sample " << i 
	   << " for classifier " << ic << endl;

      // remove subsample from training data
      data.clear();
      data.remove(samples_[i]->data());

      // print out
      if( verbose > 0 ) {
	cout << "Will train classifier " << c->name().c_str()
	     << " on a sample: " << endl;
	cout << "  W1=" << data.weightInClass(cls1)
	     << "  W0=" << data.weightInClass(cls0)
	     << "  N1=" << data.ptsInClass(cls1)
	     << "  N0=" << data.ptsInClass(cls0) << endl;
      }

      // reset classifier
      if( !c->setData(&data) ) {
	cerr << "Cross-validator unable to set data for classifier " 
	     << ic << endl;
	return false;
      }

      // train
      if( !c->train(verbose-1) ) {
	cerr << "Unable to train classifier " << ic << endl;
	continue;
      }
      SprAbsTrainedClassifier* t = c->makeTrained();
      if( t == 0 ) {
	cerr << "Cross-validator unable to get trained classifier "
	     << ic << " for subsample " << i << endl;
	continue;
      }
      t->setCut(cut);

      // compute FOM
      fom.push_back(SprFomCalculator::fom(samples_[i],t,crit,loss,cls0,cls1));

      // cleanup
      delete t;
    }// end loop over subsamples

    // sanity check
    if( fom.empty() ) {
      cerr << "Cross-validator unable to compute FOM for classifier " 
	   << ic << endl;
      crossFom[ic] = SprUtils::min();
      continue;
    }

    // compute average FOM
    double ave = 0;
    for( unsigned int i=0;i<fom.size();i++ )
      ave += fom[i];
    ave /= fom.size();

    // print out
    if( verbose > 0 ) {
      cout << "Computed FOMs for subsamples:" << endl;
      for( unsigned int i=0;i<fom.size();i++ )
	cout << i << "   FOM=" << fom[i] << endl;
    }

    // fill cross-validation FOM
    crossFom[ic] = ave;

    // reset classifier to point to the original data
    if( !c->setData(const_cast<SprAbsFilter*>(data_)) ) {
      cerr << "Cross-validator unable to restore data for classifier " 
	   << ic << endl;
      return false;
    }
  }// end loop over classifiers

  // exit
  return true;
}
