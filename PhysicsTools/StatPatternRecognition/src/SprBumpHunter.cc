//$Id: SprBumpHunter.cc,v 1.2 2007/09/21 22:32:09 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBumpHunter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <stdio.h>
#include <cassert>
#include <algorithm>
#include <functional>
#include <cmath>

using namespace std;


struct SBHCmpPairFirst 
  : public binary_function<pair<double,int>,pair<double,int>,bool> {
  bool operator()(const pair<double,int>& l, const pair<double,int>& r)
    const {
    return (l.first < r.first);
  }
};


SprBumpHunter::SprBumpHunter(SprAbsFilter* data, 
			     const SprAbsTwoClassCriterion* crit,
			     int nbump,
			     int nmin,
			     double apeel)
  :
  SprAbsClassifier(data),
  crit_(crit),
  nbump_(nbump),
  nmin_(nmin),
  apeel_(apeel),
  box_(new SprBoxFilter(data)),
  boxes_(),
  fom_(),
  n0_(),
  n1_(),
  w0_(),
  w1_(),
  nsplit_(0),
  cls0_(0),
  cls1_(1)
{
  assert( crit_ != 0 );
  assert( nbump_ > 0 );
  assert( nmin_ > 0 );
  assert( apeel_ > 0 );
  this->setClasses();
}


void SprBumpHunter::setClasses() 
{
  vector<SprClass> classes;
  box_->classes(classes);
  int size = classes.size();
  if( size > 0 ) cls0_ = classes[0];
  if( size > 1 ) cls1_ = classes[1];
  cout << "Classes for bump hunter are set to " 
       << cls0_ << " " << cls1_ << endl;
}


bool SprBumpHunter::train(int verbose)
{
  // continue until all bumps are found
  SprBox limits;
  while( fom_.size() < nbump_ ) {
    // init
    int status = -1;
    double fom0(0), w0(0), w1(0);
    unsigned n0(0), n1(0);

    // find box by shrinking
    while( (status = this->shrink(limits,n0,n1,w0,w1,fom0,verbose)) == 0 ) {
      if( (nsplit_++%100) == 0 ) 
	cout << "Performing shrinking split " << (nsplit_-1) << " ..." << endl;
    }
    if( status < 0 ) return false;

    // expand box
    while( (status = this->expand(limits,n0,n1,w0,w1,fom0,verbose)) == 0 ) {
      if( (nsplit_++%100) == 0 ) 
	cout << "Performing expanding split " << (nsplit_-1) << " ..." << endl;
    }
    if( status < 0 ) return false;

    // box found
    if( w1 < SprUtils::eps() ) {
      cout << "Unable to find a new box. Exiting." << endl;
      break;
    }
    cout << "Found box " << fom_.size() << endl;
    fom_.push_back(fom0);
    n0_.push_back(n0);
    n1_.push_back(n1);
    w0_.push_back(w0);
    w1_.push_back(w1);
    boxes_.push_back(limits);

    // exclude the found box
    if( fom_.size() < nbump_ ) {
      SprData forRemoval(false);
      for( unsigned int i=0;i<box_->size();i++ ) {
	forRemoval.insert((*box_)[i]);
      }
      if( verbose > 0 ) {
	cout << "Will remove " << forRemoval.size() 
	     << " points for the found box." << endl;
      }
      box_->clear();
      unsigned int beforeRemoval = box_->size();
      if( !box_->fastRemove(&forRemoval) ) {
	cerr << "Cannot remove data from box." << endl;
	return false;
      }
      unsigned int afterRemoval = box_->size();
      if( verbose > 0 ) {
	cout << "Sample has been reduced from " 
	     << beforeRemoval << " to " << afterRemoval << " points." << endl;
      }
      SprBoxFilter* box = new SprBoxFilter(box_);
      delete box_;
      box_ = box;
      assert( box_->size() == afterRemoval );
      
      // reset limits
      limits.clear();
    }
  }

  // exit
  return !fom_.empty();
}


bool SprBumpHunter::reset()
{
  delete box_;
  box_ = new SprBoxFilter(data_);
  boxes_.clear();
  fom_.clear();
  n0_.clear();
  n1_.clear();
  w0_.clear();
  w1_.clear();
  nsplit_ = 0;
  return true;
}


bool SprBumpHunter::setData(SprAbsFilter* data)
{
  assert( data != 0 );
  data_ = data;
  return this->reset();
}


void SprBumpHunter::print(std::ostream& os) const
{
  os << "Bumps: " << boxes_.size() << " " << SprVersion << endl;
  os << "-------------------------------------------------------" << endl;
  vector<string> vars;
  box_->vars(vars);
  for( unsigned int i=0;i<boxes_.size();i++ ) {
    int size = boxes_[i].size();
    char s [200];
    sprintf(s,"Bump %6i    Size %-4i    FOM=%-10g W0=%-10g W1=%-10g N0=%-10i N1=%-10i",i,size,fom_[i],w0_[i],w1_[i],n0_[i],n1_[i]);
    os << s << endl;
    for( SprBox::const_iterator j=boxes_[i].begin();j!=boxes_[i].end();j++ ) {
      assert( j->first < vars.size() );
      char s [200];
      sprintf(s,"Variable %30s    Limits  %15g %15g",
	      vars[j->first].c_str(),j->second.first,j->second.second);
      os << s << endl;
    }
    os << "-------------------------------------------------------" << endl;
  }
}


SprTrainedDecisionTree* SprBumpHunter::makeTrained() const
{
  // make tree
  SprTrainedDecisionTree* t =  new SprTrainedDecisionTree(boxes_);

  // vars
  vector<string> vars;
  data_->vars(vars);
  t->setVars(vars);

  // exit
  return t;
}


bool SprBumpHunter::sort(int dsort, 
			 std::vector<std::vector<int> >& sorted,
			 std::vector<std::vector<double> >& division) const
{
  // sanity check
  unsigned int size = box_->size();
  if( size == 0 ) {
    cerr << "Unable to sort an empty box." << endl;
    return false;
  }

  // init
  unsigned dim = box_->dim();
  sorted.clear();
  sorted.resize(dim,vector<int>(size));
  division.clear();
  division.resize(dim);

  // loop through dimensions
  for( unsigned int d=0;d<dim;d++ ) {
    // check dim
    if( dsort>=0 && static_cast<int>(d)!=dsort ) continue;

    // make vector
    vector<pair<double,int> > r(size);

    // loop through points
    for( unsigned int j=0;j<size;j++ )
      r[j] = pair<double,int>((*box_)[j]->x_[d],j);
    
    // sort
    stable_sort(r.begin(),r.end(),SBHCmpPairFirst());
    
    // fill out sorted indices
    division[d].push_back(SprUtils::min());
    double xprev = r[0].first;
    sorted[d][0] = r[0].second;
    for( unsigned int j=1;j<size;j++ ) {
      sorted[d][j] = r[j].second;
      double xcurr = r[j].first;
      if( (xcurr-xprev) > SprUtils::eps() ) {
	division[d].push_back(0.5*(xcurr+xprev));
	xprev = xcurr;
      }
    }
    division[d].push_back(SprUtils::max());
  }

  // exit
  return true;
}


int SprBumpHunter::shrink(SprBox& limits,
			  unsigned& n0, unsigned& n1,
			  double& w0, double& w1, double& fom0, int verbose)
{
  // sanity check
  if( box_->empty() ) {
    if( verbose > 0 ) {
      cout << "Unable to shrink - the box is empty." << endl;
    }
    return 1;
  }

  // save initial number of events
  unsigned int initSize = box_->size();

  // sort
  vector<vector<int> > sorted;
  vector<vector<double> > division;
  if( !this->sort(-1,sorted,division) ) {
    cerr << "Unable to sort box." << endl;
    return -1;
  }

  // set limits
  if( verbose > 1 ) {
    cout << "Limits:" << endl;
    for( SprBox::const_iterator d=limits.begin();d!=limits.end();d++ ) {
      cout << "Dimension " << d->first << "    " 
	   << d->second.first << " " << d->second.second << endl;
    }
  }

  // check number of events
  n0 = box_->ptsInClass(cls0_);
  n1 = box_->ptsInClass(cls1_);
  unsigned int ntot = n0 + n1;
  if( ntot < nmin_ ) {
    if( verbose > 1 ) {
      cout << "Split failed - not enough events." << endl;
    }
    return 1;
  }

  // get total weights
  w0 = box_->weightInClass(cls0_);
  w1 = box_->weightInClass(cls1_);
  if( w0<SprUtils::eps() || w1<SprUtils::eps() ) {
    if( verbose > 1 ) {
      cout << "Data missing one of categories." << endl;
    }
    return 1;
  }

  // compute fom
  fom0 = crit_->fom(0,w0,w1,0);

  // message
  if( verbose > 1 ) {
    cout << "===================" << endl;
    cout << "Splitting with " << w0 << " background and " 
	 << w1 << " signal weights and " << ntot << " events." << endl;
    cout << "Starting FOM=" << fom0 << endl;
  }

  // init
  unsigned dim = box_->dim();
  vector<double> fom(dim,SprUtils::min());
  SprBox savedBox;

  // loop through dimensions
  for( unsigned int d=0;d<dim;d++ ) {

    // check divisions
    if( division[d].empty() ) continue;

    // make vectors
    vector<double> flo, fhi;

    // loop through points to find low cuts
    double wmis0 = w0;
    double wcor1 = w1;
    double wmis1(0), wcor0(0);
    unsigned int nmis0 = n0;
    unsigned int ncor1 = n1;
    unsigned int nmis1(0), ncor0(0);
    int istart(0), isplit(0);
    for( unsigned int k=0;k<division[d].size();k++ ) {
      double z = division[d][k];
      bool lbreak = false;
      for( isplit=istart;isplit<static_cast<int>(sorted[d].size());isplit++ ) {
	if( ((*box_)[sorted[d][isplit]])->x_[d] > z ) {
	  lbreak = true;
	  break;
	}
      }
      if( !lbreak ) isplit = sorted[d].size();
      for( int i=istart;i<isplit;i++ ) {
	unsigned int index = sorted[d][i];
	const SprPoint* p = (*box_)[index];
	double w = box_->w(index);
	if(      p->class_ == cls0_ ) {
	  wmis0 -= w;
	  wcor0 += w;
	  nmis0--;
	  ncor0++;
	}
	else if( p->class_ == cls1_ ) {
	  wcor1 -= w;
	  wmis1 += w;
	  ncor1--;
	  nmis1++;
	}
      }
      istart = isplit;
      if( (wcor0+wmis1) > apeel_*(w0+w1) ) break;
      if( (ncor1+nmis0) < nmin_ ) break;
      flo.push_back(crit_->fom(wcor0,wmis0,wcor1,wmis1));
      if( verbose > 2 ) {
	cout << "Imposing shrinking low split K=" << k 
	     << " in dimension " << d << " at location=" << z
	     << " with FOM=" << flo[flo.size()-1]
	     << endl;
      }
    }

    // loop through points to find high cuts
    wmis0 = w0; wcor1 = w1;
    wmis1 = 0; wcor0 = 0;
    nmis0 = n0; ncor1 = n1;
    nmis1 = 0; ncor0 = 0;
    istart = sorted[d].size()-1;
    for( unsigned int k=division[d].size()-1;k>=0;k-- ) {
      double z = division[d][k];
      bool lbreak = false;
      for( isplit=istart;isplit>=0;isplit-- ) {
	if( ((*box_)[sorted[d][isplit]])->x_[d] < z ) {
	  lbreak = true;
	  break;
	}
      }
      if( !lbreak ) isplit = -1;
      for( int i=istart;i>isplit;i-- ) {
	int index = sorted[d][i];
	const SprPoint* p = (*box_)[index];
	double w = box_->w(index);
	if(      p->class_ == cls0_ ) {
	  wmis0 -= w;
	  wcor0 += w;
	  nmis0--;
	  ncor0++;
	}
	else if( p->class_ == cls1_ ) {
	  wcor1 -= w;
	  wmis1 += w;
	  ncor1--;
	  nmis1++;
	}
      }
      istart = isplit;
      if( (wcor0+wmis1) > apeel_*(w0+w1) ) break;
      if( (ncor1+nmis0) < nmin_ ) break;
      fhi.push_back(crit_->fom(wcor0,wmis0,wcor1,wmis1));
      if( verbose > 2 ) {
	cout << "Imposing shrinking high split K=" << k 
	     << " in dimension " << d << "  at location=" << z
	     << " with FOM=" << fhi[fhi.size()-1]
	     << endl;
      }
    }

    // find optimal point and sign of cut
    if( flo.empty() && fhi.empty() ) continue;
    int klo(-1), khi(-1);
    double lfom = SprUtils::min();
    double hfom = SprUtils::min();
    if( !flo.empty() ) {
      vector<double>::iterator ilo = max_element(flo.begin(),flo.end());
      klo = ilo - flo.begin();
      lfom = *ilo;
    }
    if( !fhi.empty() ) {
      vector<double>::iterator ihi = max_element(fhi.begin(),fhi.end());
      khi = ihi - fhi.begin();
      hfom = *ihi;
    }
    if( lfom>hfom && klo!=0 && klo!=((int)division[d].size()-1) ) {
      if( !savedBox.insert(pair<const unsigned,SprInterval>(d,(SprUtils::lowerBound(division[d][klo]))[0])).second ) {
	cerr << "Unable to insert interval for dimension " << d << endl;
	return -1;
      }
      fom[d] = lfom;
    }
    else if( khi!=0 && khi!=(static_cast<int>(division[d].size())-1) ) {
      if( !savedBox.insert(pair<const unsigned,SprInterval>(d,(SprUtils::upperBound(division[d][division[d].size()-1-khi]))[0])).second ) {
	cerr << "Unable to insert interval for dimension " << d << endl;
	return -1;
      }
      fom[d] = hfom;
    }
  }

  // find optimal fom
  vector<double>::iterator imax = max_element(fom.begin(),fom.end());

  // update box
  bool lexit = true;
  if( *imax > fom0 ) {
    lexit = false;
    int d = imax - fom.begin();
    SprBox::const_iterator imposed = savedBox.find(d);
    assert( imposed != savedBox.end() );
    SprInterval z = imposed->second;
    if( verbose > 0 ) {
      cout << "Making shrinking split " << nsplit_ 
	   << " in dimension " << d << " with " 
	   << n1 << " signal and " << n0 << " background events" 
	   << "     FOM=" << *imax 
	   << "    interval=" << z.first << " " << z.second << endl;
    }

    // update limits
    SprBox::iterator found = limits.find(d);
    if( found == limits.end() ) {
      limits.insert(pair<const unsigned,SprInterval>(d,z));
    }
    else {
      if( found->second.first < z.first )
	found->second.first = z.first;
      if( found->second.second > z.second )
	found->second.second = z.second;
      assert( found->second.first < found->second.second );
    }

    // cut off tails
    box_->setBox(limits);
    if( !box_->filter() ) {
      cerr << "Cannot filter box." << endl;
      return -1;
    }

    // print out
    if( verbose > 1 ) {
      cout << "Imposed cut in dimension " << d << "  : "
	   << found->second.first << " " <<  found->second.second 
	   << "  Box reduced to " << box_->size() << " events." << endl;
    }

    // check if the size has changed
    if( box_->size() >= initSize ) {
      if( verbose > 0 ) 
	cout << "Warning: box has not been reduced." << endl;
      lexit = true;
    }
  }

  // see if time to exit
  if( lexit ) {
    // reset weights
    w0 = box_->weightInClass(cls0_);
    w1 = box_->weightInClass(cls1_);
    n0 = box_->ptsInClass(cls0_);
    n1 = box_->ptsInClass(cls1_);
    fom0 = crit_->fom(0,w0,w1,0);

    // message
    if( verbose > 0 ) {
      cout << "Found box " << boxes_.size() << " with "
	   << w0 << " background and " << w1 << " signal weights;   "
	   << n0 << " background and " << n1 << " signal events"
	   << "     FOM=" << fom0  << endl;
    }
    return 1;
  }

  // exit
  if( verbose > 1 )
    cout << "===================" << endl;
  return 0;
}


int SprBumpHunter::expand(SprBox& limits, 
			  unsigned& n0, unsigned& n1,
			  double& w0, double& w1, double& fom0, int verbose)
{
  // sanity check
  if( box_->empty() ) {
    if( verbose > 0 ) {
      cout << "Unable to expand - the box is empty." << endl;
    }
    return 1;
  }

  // save initial number of events
  unsigned int initSize = box_->size();

  // get total weights, fom and node type
  w0 = box_->weightInClass(cls0_);
  w1 = box_->weightInClass(cls1_);

  // check number of events
  n0 = box_->ptsInClass(cls0_);
  n1 = box_->ptsInClass(cls1_);
  int ntot = n0 + n1;

  // compute fom
  fom0 = crit_->fom(0,w0,w1,0);

  // message
  if( verbose > 1 ) {
    cout << "===================" << endl;
    cout << "Expanding with " << w0 << " background and " 
	 << w1 << " signal weights and " << ntot << " events." << endl;
    cout << "Starting FOM=" << fom0 << endl;
  }

  // init
  unsigned dim = box_->dim();
  vector<double> fom(dim,SprUtils::min());
  SprBox savedBox = limits;

  // set limits
  if( verbose > 1 ) {
    cout << "Limits:" << endl;
    for( SprBox::const_iterator d=limits.begin();d!=limits.end();d++ ) {
      cout << "Dimension " << d->first << "    " 
	   << d->second.first << " " << d->second.second << endl;
    }
  }

  // loop through dimensions
  for( unsigned int d=0;d<dim;d++ ) {
    // reset limits
    box_->setBox(limits);

    // check if any limits exist    
    SprBox::iterator found = savedBox.find(d);
    if( found == savedBox.end() ) {
      if( verbose > 1 )
	cout << "No cuts on dimension " << d << " imposed. " 
	     << " will not attempt to expand."<< endl;
      continue;
    }

    // look at both sides of the box
    for( int side=-1;side<2;side+=2 ) {

      // compute starting weights
      if( verbose > 1 ) {
	cout << "Resetting cuts on dimension " << d << " at "
	     << "     Low=" << found->second.first 
	     << "    High=" << found->second.second << endl;
      }
      box_->setRange(d,found->second);
      if( !box_->filter() ) {
	cerr << "Cannot filter box." << endl;
	return -1;
      }
      if( box_->empty() ) {
	cerr << "Box is empty after cutting on dimension " << d 
	     << " at " << found->second.first << " " 
	     << found->second.second << endl;
	return 1;
      }

      // get total weights
      double w0d = box_->weightInClass(cls0_);
      double w1d = box_->weightInClass(cls1_);
      if( verbose > 1 ) {
	cout << "Starting optimization in dimension " << d
	     << " with " << w1d << " signal and "
	     << w0d << " background weights." << endl;
      }
      double startfom = crit_->fom(0,w0d,w1d,0);

      // check number of events
      int n0d = box_->ptsInClass(cls0_);
      int n1d = box_->ptsInClass(cls1_);

      // invert cut on dimension d to accept events outside the box
      SprInterval outer;
      if(      side < 0 )
	outer = SprInterval(SprUtils::min(),found->second.first);
      else if( side > 0 )
	outer = SprInterval(found->second.second,SprUtils::max());
      box_->setRange(d,outer);// invert cut on dimension d

      // filter
      if( !box_->filter() ) {
	cerr << "Cannot filter box." << endl;
	return -1;
      }
      if( box_->empty() ) continue;

      // compute weights
      double w0add = box_->weightInClass(cls0_);
      double w1add = box_->weightInClass(cls1_);
      if( verbose > 1 ) {
	cout << "Expanding in dimension " << d << " into area with " 
	     << w0add << " background and " 
	     << w1add << " signal events." << endl;
      }
      if( w1add < SprUtils::eps() ) continue;// do nothing if added area empty

      // sort
      vector<vector<int> > sorted;
      vector<vector<double> > division;
      if( !this->sort(d,sorted,division) ) {
	cerr << "Unable to sort box." << endl;
	return -1;
      }

      // prepare weights and fom
      double wmis0(w0add), wcor1(w1add);
      double wmis1(0), wcor0(0);
      int nmis0(n0d), ncor1(n1d);
      int nmis1(0), ncor0(0);
      vector<double> fomnew;

      // loop through points and find cuts
      if(      side < 0 ) {// lower cut
	int istart(0), isplit(0);
	for( unsigned int k=0;k<division[d].size();k++ ) {
	  double z = division[d][k];
	  bool lbreak = false;
	  for( isplit=istart;isplit<(int)sorted[d].size();isplit++ ) {
	    if( ((*box_)[sorted[d][isplit]])->x_[d] > z ) {
	      lbreak = true;
	      break;
	    }
	  }
	  if( !lbreak ) isplit = sorted[d].size();
	  for( int i=istart;i<isplit;i++ ) {
	    int index = sorted[d][i];
	    const SprPoint* p = (*box_)[index];
	    double w = box_->w(index);
	    if(      p->class_ == cls0_ ) {
	      wmis0 -= w;
	      wcor0 += w;
	      nmis0--;
	      ncor0++;
	    }
	    else if( p->class_ == cls1_ ) {
	      wcor1 -= w;
	      wmis1 += w;
	      ncor1--;
	      nmis1++;
	    }
	  }
	  istart = isplit;
	  fomnew.push_back(crit_->fom(wcor0,wmis0+w0d,wcor1+w1d,wmis1));
	  if( verbose > 2 ) {
	    cout << "Imposing expanding low split K=" << k 
		 << " in dimension " << d << " with FOM=" 
		 << fomnew[fomnew.size()-1] << endl;
	  }
	}
      }
      else if( side > 0 ) {// high cut
	int istart(sorted[d].size()-1), isplit(0);
	for( int k=division[d].size()-1;k>=0;k-- ) {
	  double z = division[d][k];
	  bool lbreak = false;
	  for( isplit=istart;isplit>=0;isplit-- ) {
	    if( ((*box_)[sorted[d][isplit]])->x_[d] < z ) {
	      lbreak = true;
	      break;
	    }
	  }
	  if( !lbreak ) isplit = -1;
	  for( int i=istart;i>isplit;i-- ) {
	    int index = sorted[d][i];
	    const SprPoint* p = (*box_)[index];
	    double w = box_->w(index);
	    if(      p->class_ == cls0_ ) {
	      wmis0 -= w;
	      wcor0 += w;
	      nmis0--;
	      ncor0++;
	    }
	    else if( p->class_ == cls1_ ) {
	      wcor1 -= w;
	      wmis1 += w;
	      ncor1--;
	      nmis1++;
	    }
	  }
	  istart = isplit;
	  fomnew.push_back(crit_->fom(wcor0,wmis0+w0d,wcor1+w1d,wmis1));
	  if( verbose > 2 ) {
	    cout << "Imposing expanding high split K=" << k 
		 << " in dimension " << d 
		 << " with FOM=" << fomnew[fomnew.size()-1] << endl;
	  }
	}
      }// end side > 0

      // find optimal point and sign of cut
      vector<double>::iterator imax = max_element(fomnew.begin(),fomnew.end());
      int k = imax - fomnew.begin();
      double maxfom = *imax;

      // print out
      if( verbose > 1 ) {
	cout << "FOM's for dimension " << d 
	     << ":   Side=" << side << "  Best FOM=" << maxfom 
	     << "   Start FOM=" << startfom << "  Init FOM=" << fom0 
	     << "   K=" << k << endl;
      }

      // find optimal cut
      if( maxfom > startfom ) {
	fom[d] = maxfom;
	if(      side < 0 ) {
	  double z = division[d][k];
	  if( found->second.first > z ) found->second.first = z;
	}
	else if( side > 0 ) {
	  double z = division[d][division[d].size()-1-k];
	  if( found->second.second < z ) found->second.second = z;
	}
      }
    }// end side loop
    if( verbose > 1 ) {
      cout << "Found the best cut on dimension " << d << " at " 
	   << "    Low=" << found->second.first 
	   << "   High=" << found->second.second 
	   << "  with FOM=" << fom[d] << endl;
    }
  }// end dim loop

  // find optimal fom
  vector<double>::iterator imax = max_element(fom.begin(),fom.end());

  // update box
  bool lexit = true;
  if( *imax > fom0 ) {
    lexit = false;
    int d = imax - fom.begin();
    SprBox::const_iterator imposed = savedBox.find(d);
    assert( imposed != savedBox.end() );
    SprInterval z = imposed->second;
    if( verbose > 0 ) {
      cout << "Making expansion " << nsplit_ 
	   << " in dimension " << d << " with " 
	   << n1 << " signal and " << n0 << " background events" 
	   << "     FOM=" << *imax 
	   << "    interval=" << z.first << " " << z.second << endl;
    }

    // update limits
    SprBox::iterator found = limits.find(d);
    if( found != limits.end() ) {
      if( found->second.first > z.first )
	found->second.first = z.first;
      if( found->second.second < z.second )
	found->second.second = z.second;
      assert( found->second.first < found->second.second );
    }

    // cut off tails
    box_->setBox(limits);
    if( !box_->filter() ) {
      cerr << "Cannot filter box." << endl;
      return -1;
    }

    // check if the size has changed
    if( box_->size() <= initSize ) {
      if( verbose > 0 )
	cout << "Warning: box has not been increased." << endl;
      lexit = true;
    }
  }

  // see if time to exit
  if( lexit ) {
    // cut off tails
    box_->setBox(limits);
    if( !box_->filter() ) {
      cerr << "Cannot filter box." << endl;
      return -1;
    }

    // recompute weights
    w0 = box_->weightInClass(cls0_);
    w1 = box_->weightInClass(cls1_);
    n0 = box_->ptsInClass(cls0_);
    n1 = box_->ptsInClass(cls1_);
    fom0 = crit_->fom(0,w0,w1,0);
    
    // message
    if( verbose > 0 ) {
      cout << "Found box " << boxes_.size() << " with "
	   << w0 << " background and " << w1 << " signal weights;   "
	   << n0 << " background and " << n1 << " signal events"
	   << "     FOM=" << fom0  << endl;
    }
    return 1;
  }

  // exit
  if( verbose > 1 )
    cout << "===================" << endl;
  return 0;
}





