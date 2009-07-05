//$Id: SprTransformerFilter.cc,v 1.2 2008/11/26 22:59:20 elmer Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformerFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsVarTransformer.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprCoordinateMapper.hh"

#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include <iterator>
#include <memory>

using namespace std;


bool SprTransformerFilter::transform(const SprAbsVarTransformer* trans,
				     bool replaceOriginalData)
{
  // sanity check
  assert( trans != 0 );
  if( !trans->ready() ) {
    cerr << "Variable transformer not ready. " 
	 << "No transformation will be applied." << endl;
    return false;
  }

  // get transformation vars and data vars
  vector<string> dataVars;
  data_->vars(dataVars);
  vector<string> transVars;
  trans->oldVars(transVars);
  vector<string> newVars;
  trans->newVars(newVars);

  // map data vars onto transformation vars
  auto_ptr<SprCoordinateMapper> 
    mapper(SprCoordinateMapper::createMapper(transVars,dataVars));
  if( mapper.get() == 0 ) {
    cerr << "SprTransformerFilter unable to map transformation vars onto " 
	 << "data vars." << endl;
    return false;
  }

  // prepare a list of data vars with transformed vars excluded
  vector<string> newDataVars;
  vector<int> useVars;
  for( unsigned int i=0;i<dataVars.size();i++ ) {
    if( ::find(transVars.begin(),transVars.end(),dataVars[i])
	== transVars.end() ) {
      newDataVars.push_back(dataVars[i]);
      useVars.push_back(i);
    }
  }
  assert( newDataVars.size() == (dataVars.size()-transVars.size()) );
  ::copy(newVars.begin(),newVars.end(),back_inserter(newDataVars));

  // clean up copy
  SprData* copy = 0;
  if( !replaceOriginalData ) 
    copy = data_->emptyCopy();

  // loop over points
  for( unsigned int i=0;i<data_->size();i++ ) {
    SprPoint* p = (*data_)[i];
    vector<double>& oldV = p->x_;
    vector<double> newV;
    for( unsigned int d=0;d<useVars.size();d++ )
      newV.push_back(oldV[useVars[d]]);
    vector<double> oldExtraV;
    mapper->map(oldV,oldExtraV);
    vector<double> newExtraV;
    trans->transform(oldExtraV,newExtraV);
    ::copy(newExtraV.begin(),newExtraV.end(),back_inserter(newV));
    assert( newV.size() == newDataVars.size() );
    if( replaceOriginalData )
      oldV = newV;
    else
      copy->uncheckedInsert(new SprPoint(p->index_,p->class_,newV));
  }

  // make new data
  if( replaceOriginalData ) {
    SprData* data = const_cast<SprData*>(data_);
    data->setDim(newDataVars.size());
    data->setVars(newDataVars);
    if( ownCopy_ ) delete copy_;
    copy_ = data_;
    ownCopy_ = false;
  }
  else {
    copy->setDim(newDataVars.size());
    copy->setVars(newDataVars);
    if( ownCopy_ ) delete copy_;
    copy_ = copy;
    ownCopy_ = true;
  }

  // weights are copied without changes
  copyWeights_ = dataWeights_;
  
  // exit
  return true;
}
