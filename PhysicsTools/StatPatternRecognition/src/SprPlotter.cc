//$Id: SprPlotter.cc,v 1.1 2007/09/21 22:32:10 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPlotter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"

#include <algorithm>
#include <functional>
#include <iostream>
#include <cmath>

using namespace std;


struct SPCmpPairDDFirst
  : public binary_function<pair<double,double>,pair<double,double>,bool> {
  bool operator()(const pair<double,double>& l, const pair<double,double>& r)
    const {
    return (l.first < r.first);
  }
};


SprPlotter::SprPlotter(const std::vector<Response>& responses) 
  : 
  responses_(responses), 
  crit_(0), 
  useAbsolute_(false), 
  scaleS_(1.), 
  scaleB_(1.),
  sigW_(0),
  bgrW_(0),
  sigN_(0),
  bgrN_(0)
{
  bool status = this->init();
  assert( status );
}


SprPlotter::SprPlotter(const SprPlotter& other)
  :
  responses_(other.responses_), 
  crit_(other.crit_), 
  useAbsolute_(other.useAbsolute_), 
  scaleS_(other.scaleS_), 
  scaleB_(other.scaleB_),
  sigW_(other.sigW_),
  bgrW_(other.bgrW_),
  sigN_(other.sigN_),
  bgrN_(other.bgrN_)
{}


bool SprPlotter::init()
{
  sigW_ = 0;
  bgrW_ = 0;
  sigN_ = 0;
  bgrN_ = 0;
  for( unsigned int i=0;i<responses_.size();i++ ) {
    if(      responses_[i].cls == 0 ) {
      bgrN_++;
      bgrW_ += responses_[i].weight;
    }
    else if( responses_[i].cls == 1 ) {
      sigN_++;
      sigW_ += responses_[i].weight;
    }
  }
  if( sigW_<SprUtils::eps() || bgrW_<SprUtils::eps() 
      || sigN_==0 || bgrN_==0 ) {
    cerr << "One of categories missing in the vector of responses." << endl;
    return false;
  }
  return true;
}


bool SprPlotter::setScaleFactors(double scaleS, double scaleB)
{
  if( useAbsolute_ ) {
    scaleS_ = scaleS;
    scaleB_ = scaleB;
    return true;
  }
  cerr << "Use setAbsolute() first to choose absolute values "
       << "instead of relative efficiencies." << endl;
  return false;
}


bool SprPlotter::backgroundCurve(const std::vector<double>& signalEff,
				 const char* classifier,
				 std::vector<FigureOfMerit>& bgrndEff) const
{
  // sanity check
  string sclassifier = classifier;
  if( sclassifier.empty() ) {
    cerr << "No classifier has been specified for plotting." << endl;
    return false;
  }
  if( signalEff.empty() ) {
    cerr << "No signal efficiencies specified." << endl;
    return false;
  }

  // sort signal efficiencies and prepare vectors
  vector<double> sigSortedEff(signalEff);
  stable_sort(sigSortedEff.begin(),sigSortedEff.end());

  // prepare vectors
  vector<pair<double,double> > signal;
  vector<pair<double,double> > bgrnd;
  if( !this->fillSandB(sclassifier,signal,bgrnd) ) {
    cerr << "Unable to fill out signal and background vectors in " 
	 << "SprPlotter::backgroundCurve." << endl;
    return false;
  }

  // check sizes
  if( signal.empty() || bgrnd.empty() ) {
    cerr << "One of categories is empty for classifier " 
	 << sclassifier.c_str() 
	 << " in SprPlotter::backgroundCurve." << endl;
    return false;
  }

  // sort
  stable_sort(bgrnd.begin(),bgrnd.end(),not2(SPCmpPairDDFirst()));
  stable_sort(signal.begin(),signal.end(),not2(SPCmpPairDDFirst()));

  // if absolute values, check max allowed signal value
  int maxsize = sigSortedEff.size();
  vector<double>::iterator maxeff = sigSortedEff.end();
  if( useAbsolute_ ) {
    maxeff = find_if(sigSortedEff.begin(),sigSortedEff.end(),
		     bind2nd(greater<double>(),sigW_));
  }
  else {
    maxeff = find_if(sigSortedEff.begin(),sigSortedEff.end(),
		     bind2nd(greater<double>(),1.));
  }
  if( maxeff != sigSortedEff.end() )
    maxsize = maxeff-sigSortedEff.begin();
  if( maxsize == 0 ) {
    cerr << "Values of signal efficiency out of range." << endl;
    return false;
  }

  // find dividing point in classifier response
  vector<double> cuts(maxsize,signal[0].first);
  double w = 0;
  int startDivider = 0;
  unsigned int i = 0;
  while( i<signal.size() && startDivider<maxsize ) {
    w += signal[i].second;
    for( int divider=startDivider;divider<maxsize;divider++ ) {
      if( (useAbsolute_ && scaleS_*w>sigSortedEff[divider]) 
	  || (!useAbsolute_ && (w/sigW_)>sigSortedEff[divider]) ) {
	if( i > 0 )
	  cuts[divider] = 0.5 * (signal[i].first + signal[i-1].first);
	startDivider = divider + 1;
      }
      else
	break;
    }// end divider loop
    i++;
  }
	
  // find background fractions
  double defFOMvalue = 0;
  if( useAbsolute_ && crit_!=0 ) 
    defFOMvalue = crit_->fom(0,scaleB_*bgrW_,scaleS_*sigW_,0);
  FigureOfMerit defaultFOM(cuts[cuts.size()-1],
			   (useAbsolute_ ? bgrW_ : 1),
			   bgrN_,defFOMvalue);
  bgrndEff.clear();
  bgrndEff.resize(sigSortedEff.size(),defaultFOM);
  w = 0;
  startDivider = 0;
  i = 0;
  while( i<bgrnd.size() && startDivider<maxsize ) {
    for( int divider=startDivider;divider<maxsize;divider++ ) {
      if( bgrnd[i].first < cuts[divider] ) {
	if( useAbsolute_ ) {
	  bgrndEff[divider]
	    = FigureOfMerit(cuts[divider],scaleB_*w,i,
			    ( crit_==0 ? 0 : 
			      crit_->fom(scaleB_*(bgrW_-w),scaleB_*w,
					 sigSortedEff[divider],
					 scaleS_*sigW_
					 -sigSortedEff[divider])));
	}
	else {
	  bgrndEff[divider] = FigureOfMerit(cuts[divider],w/bgrW_,i);
	}
	startDivider = divider + 1;
      }
      else
	break;
    }// end divider loop
    w += bgrnd[i].second;
    i++;
  }

  // exit
  return true;
}


bool SprPlotter::fillSandB(const std::string& sclassifier,
			   std::vector<std::pair<double,double> >& signal,
			   std::vector<std::pair<double,double> >& bgrnd) const
{
  // init
  signal.clear();
  bgrnd.clear();

  // fill them
  for( unsigned int i=0;i<responses_.size();i++ ) {
    // check if classifier is present
    map<string,double>::const_iterator found = 
      responses_[i].response.find(sclassifier);
    if( found == responses_[i].response.end() ) {
      cerr << "Point " << i << " does not have response value " 
	   << "for classifier " << sclassifier.c_str() << endl;
      return false;
    }

    // fill out signal and background vectors
    int cls = responses_[i].cls;
    double w = responses_[i].weight;
    if(      cls == 0 )
      bgrnd.push_back(pair<double,double>(found->second,w));
    else if( cls == 1 )
      signal.push_back(pair<double,double>(found->second,w));
  }

  // exit
  return true;
}


int SprPlotter::histogram(const char* classifier, 
			  double xlo, double xhi, double dx,
			  std::vector<std::pair<double,double> >& sigHist,
			  std::vector<std::pair<double,double> >& bgrHist) 
  const
{
  // sanity check
  const string sclassifier = classifier;
  if( sclassifier.empty() || dx<=0. || xlo>=xhi ) {
    cerr << "Invalid histogram parameters entered." << endl;
    return 0;
  }

  // prepare vectors
  vector<pair<double,double> > signal;
  vector<pair<double,double> > bgrnd;

  // fill them
  if( !this->fillSandB(sclassifier,signal,bgrnd) ) {
    cerr << "Unable to fill out signal and background vectors " 
	 << "in SprPlotter::histogram." << endl;
    return 0;
  }

  // sort in ascending order
  stable_sort(bgrnd.begin(),bgrnd.end(),SPCmpPairDDFirst());
  stable_sort(signal.begin(),signal.end(),SPCmpPairDDFirst());

  // book histos
  int nbin = int(floor((xhi-xlo)/dx)) + 1;
  sigHist.clear(); bgrHist.clear();
  sigHist.resize(nbin); bgrHist.resize(nbin);

  // init indices
  int jsig(0), jbgr(0);

  // check starting indices
  if( !signal.empty() ) {
    if( signal[0].first < xlo ) {
      bool lbreak = false;
      for( unsigned int j=0;j<signal.size();j++ ) {
	if( signal[j].first >= xlo ) {
	  jsig = j;
	  lbreak = true;
	  break;
	}
      }
      if( !lbreak ) jsig = signal.size();
    }
  }
  if( !bgrnd.empty() ) {
    if( bgrnd[jbgr].first < xlo ) {
      bool lbreak = false;
      for( unsigned int j=jbgr;j<bgrnd.size();j++ ) {
	if( bgrnd[j].first >= xlo ) {
	  jbgr = j;
	  lbreak = true;
	  break;
	}
      }
      if( !lbreak ) jbgr = bgrnd.size();
    }
  }
      
  // fill histos
  for( int i=0;i<nbin;i++ ) {
    double xa = xlo + i*dx;
    double xb = xa + dx;
    
    // signal
    double wsig = 0;
    unsigned nsig = 0;
    bool lbreak = false;
    for( unsigned int j=jsig;j<signal.size();j++ ) {
      if( signal[j].first >= xb ) {
	jsig = j;
	lbreak = true;
	break;
      }
      wsig += signal[j].second;
      nsig++;
    }
    double errS = ( nsig>0 ? scaleS_*wsig/sqrt(double(nsig)) : 0 );
    sigHist[i] = pair<double,double>(scaleS_*wsig,errS);
    if( !lbreak ) jsig = signal.size();

    // background
    double wbgr = 0;
    unsigned nbgr = 0;
    lbreak = false;
    for( unsigned int j=jbgr;j<bgrnd.size();j++ ) {
      if( bgrnd[j].first >= xb ) {
	jbgr = j;
	lbreak = true;
	break;
      }
      wbgr += bgrnd[j].second;
      nbgr++;
    }
    double errB = ( nbgr>0 ? scaleB_*wbgr/sqrt(double(nbgr)) : 0 );
    bgrHist[i] = pair<double,double>(scaleB_*wbgr,errB);
    if( !lbreak ) jbgr = bgrnd.size();
  }// end bin loop

  // sanity check
  assert( sigHist.size() == bgrHist.size() );

  // exit
  return nbin;
}
