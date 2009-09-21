#include "CalibTracker/SiStripLorentzAngle/interface/LA_Filler_Fitter.h"
#include "CalibTracker/SiStripLorentzAngle/interface/TTREE_FOREACH_ENTRY.hh"
#include "CalibTracker/SiStripLorentzAngle/interface/Book.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <cmath>
#include <boost/foreach.hpp>
#include <boost/regex.hpp> 
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/erase.hpp>
#include <TF1.h>
#include <TTree.h>
#include <TGraphErrors.h>

void LA_Filler_Fitter::
fill(TTree* tree, Book& book) {
  TTREE_FOREACH_ENTRY(tree,
		      std::vector<unsigned>* LEAF( clusterdetid )
		      std::vector<unsigned>* LEAF( clusterwidth )
		      std::vector<float>*    LEAF( clustervariance )
		      std::vector<unsigned>* LEAF( tsostrackmulti )
		      std::vector<float>*    LEAF( tsosdriftx )
		      std::vector<float>*    LEAF( tsosdriftz )
		      std::vector<float>*    LEAF( tsoslocalpitch )
		      std::vector<float>*    LEAF( tsoslocaltheta )
		      std::vector<float>*    LEAF( tsoslocalphi )
		      std::vector<float>*    LEAF( tsosBdotY )
		      std::vector<float>*    LEAF( tsosglobalZofunitlocalY )
		      ) {
    if(maxEvents_) TTREE_FOREACH_ENTRY_total = std::min(maxEvents_,TTREE_FOREACH_ENTRY_total);
    for(unsigned i=0; i< clusterwidth->size() ; i++) {  

      SiStripDetId detid((*clusterdetid)[i]);

      if( (*tsostrackmulti)[i] != 1 ||
	  detid.subDetector()!=SiStripDetId::TIB && 
	  detid.subDetector()!=SiStripDetId::TOB        ) 
	continue;
      
      float driftx = (*tsosdriftx)[i];
      float driftz = (*tsosdriftz)[i];

      int sign = (*tsosglobalZofunitlocalY)[i] < 0  ?  -1  :  1 ;
      double projectionByDriftz = tan((*tsoslocaltheta)[i]) * cos((*tsoslocalphi)[i]);
      unsigned width = (*clusterwidth)[i];
      float sqrtVar = sqrt((*clustervariance)[i]);

      poly<std::string> granular;
      if(ensembleBins_) {
	granular+= "ensembleBin"+boost::lexical_cast<std::string>((int)(ensembleBins_*(sign*driftx/driftz-ensembleLow_)/(ensembleUp_-ensembleLow_)));
	granular+= "_pitch"+boost::lexical_cast<std::string>((int)(10000* (*tsoslocalpitch)[i] ));
	granular+= "";
	if(ensembleSize_) granular*= "_sample"+boost::lexical_cast<std::string>(TTREE_FOREACH_ENTRY_index % ensembleSize_);
      } else {
	granular+= detid.subDetector()==SiStripDetId::TOB? "TOB" : "TIB";
	granular+= "";
	if(byLayer_) { unsigned layer = detid.subDetector() == SiStripDetId::TOB ? TOBDetId(detid()).layer() : TIBDetId(detid()).layer(); 
                       granular*= "_layer"+boost::lexical_cast<std::string>(layer)+(detid.stereo()?"s":"a"); }
	if(byModule_)  granular*= "_module"+boost::lexical_cast<std::string>(detid());
      }
      poly<std::string> A1("_all"); 
      if(width==1) A1*="_width1";

                             book.fill( sign*driftx/driftz,               granular+"_reconstruction"   , 81, -0.6,0.6 );
      if(methods_ & RATIO)   book.fill( sign*projectionByDriftz,          granular+ method(RATIO,0)+A1 , 81, -0.6,0.6 );
      if(methods_ & WIDTH)   book.fill( sign*projectionByDriftz,   width, granular+ method(WIDTH)      , 81, -0.6,0.6 );
      if(methods_ & SQRTVAR) book.fill( sign*projectionByDriftz, sqrtVar, granular+ method(SQRTVAR)    , 81, -0.6,0.6 );
      if(methods_ & SYMM)    book.fill( sign*projectionByDriftz, sqrtVar, granular+ method(SYMM,0)     ,128,-1.0,1.0 );
      if(ensembleBins_==0)   book.fill( fabs((*tsosBdotY)[i]),            granular+"_field"            , 101, 1, 5 );
    }
  }
}

void LA_Filler_Fitter::
make_and_fit_ratio(Book& book, bool cleanup) {
  for(Book::const_iterator it = book.begin(".*"+method(RATIO,0)+"_width1"); it!=book.end(); ++it) {
    if((*it)->GetEntries() < 30) continue;

    std::string base = boost::erase_all_copy(it.name(),"_width1");
    std::string width1 = base+"_width1";
    std::string all    = base+"_all";
    std::string ratio  = base+"_ratio";

    TH1* p = (TH1*) book(width1)->Clone(ratio.c_str());
    p->Divide(book(all));
    p->Fit("gaus","LLQ");
    double mean = p->GetFunction("gaus")->GetParameter(1);
    double sigma = p->GetFunction("gaus")->GetParameter(2);
    p->Fit("gaus","LLMEQ","",mean-sigma,mean+sigma);
    book.book(ratio, p);
    
    if(cleanup) {
      book.erase(width1);
      book.erase(all);
    }
  }
}

void LA_Filler_Fitter::
fit_profile(Book& book, const std::string& key) {
  for(Book::const_iterator p = book.begin(".*"+key); p!=book.end(); ++p) {
    if((*p)->GetEntries() < 400) {book.erase(p.name()); continue;}
    
    float min = (*p)->GetMinimum();
    float max = (*p)->GetMaximum();
    float xofmin = (*p)->GetBinCenter((*p)->GetMinimumBin()); if( xofmin>0.0 || xofmin<-0.15) xofmin = -0.05;
    float xofmax = (*p)->GetBinCenter((*p)->GetMaximumBin());

    TF1* fit = new TF1("LA_profile_fit","[2]*(TMath::Abs(x-[0]))+[1]",-1,1);
    fit->SetParLimits(0,-0.15,0.01);
    fit->SetParLimits(1, 0.6*min, 1.25*min );
    fit->SetParLimits(2,0.1,10);
    fit->SetParameters( xofmin, min, (max-min) / fabs( xofmax - xofmin ) );

    int badfit = (*p)->Fit(fit,"IEQ","",-.5,.3);
    if( badfit ) badfit = (*p)->Fit(fit,"IEQ","", -.46,.26);
    if( badfit ) book.erase(p.name());
  }
}

void LA_Filler_Fitter::
make_and_fit_symmchi2(Book& book) {
  for(Book::const_iterator p = book.begin(".*"+method(SYMM,0)); p!=book.end(); ++p) {
    unsigned minbin = (*p)->GetMinimumBin();
    double dguess(0), weight(0);
    for(unsigned i=0; i<20; i++) { 
      double w = 1./pow((*p)->GetBinContent(minbin-10+i),4);
      dguess+= w*(minbin-10+i);
      weight+= w;
    }
    unsigned guess = (unsigned)(dguess/weight);
    TH1* chi2 = SymmetryFit::symmetryChi2(*p, std::make_pair(guess-4,guess+4));
    if(chi2) book.book(SymmetryFit::name((*p)->GetName()), chi2);
  }
}

LA_Filler_Fitter::Result LA_Filler_Fitter::
result(Method m, const std::string name, const Book& book) {
  Result p;
  std::string base = boost::erase_all_copy(name,method(m));
  if(book.contains(base+"_reconstruction")) {
    p.reco    = book(base+"_reconstruction")->GetMean();
    p.recoErr = book(base+"_reconstruction")->GetRMS();
  }
  if(book.contains(base+"_field"))
    p.field = book(base+"_field")->GetMean();

  if(book.contains(name)) {
    TH1* h = book(name);
    p.entries = (unsigned)(h->GetEntries());
    switch(m) {
    case RATIO: {
      TF1* f = h->GetFunction("gaus");
      p.measure = f->GetParameter(1);
      p.measureErr = f->GetParError(1);
      p.chi2 = f->GetChisquare();
      p.ndof = f->GetNDF();
      break; }
    case WIDTH: case SQRTVAR: {
      TF1* f = h->GetFunction("LA_profile_fit");
      p.measure = f->GetParameter(0);
      p.measureErr = f->GetParError(0);
      p.chi2 = f->GetChisquare();
      p.ndof = f->GetNDF();
      break;
    }
    case SYMM: {
      TF1* f = h->GetFunction("SymmetryFit");
      p.measure = f->GetParameter(0);
      p.measureErr = f->GetParameter(1);
      p.chi2 = f->GetParameter(2);
      p.ndof = (unsigned) (f->GetParameter(3));
    }
    default:break;
    }
  }
  return p;
}

std::map<uint32_t,LA_Filler_Fitter::Result> LA_Filler_Fitter::
module_results(const Book& book, const Method m) {
  std::map<uint32_t,Result> results;
  for(Book::const_iterator it = book.begin(".*module.*"+method(m)); it!=book.end(); ++it ) {
    uint32_t detid = boost::lexical_cast<uint32_t>( boost::regex_replace( it.name(),
									  boost::regex(".*_module(\\d*)_.*"),
									  std::string("\\1")));
    results[detid] = result(m,it.name(),book);
  }
  return results;
}

std::map<std::string,LA_Filler_Fitter::Result> LA_Filler_Fitter::
layer_results(const Book& book, const Method m) {
  std::map<std::string,Result> results;
  for(Book::const_iterator it = book.begin(".*layer.*"+method(m)); it!=book.end(); ++it ) {
    std::string name = boost::erase_all_copy(it.name(),method(m));
    results[name] = result(m,it.name(),book);
  }
  return results;
}

std::map<std::string, std::vector<LA_Filler_Fitter::Result> > LA_Filler_Fitter::
ensemble_results(const Book& book, const Method m) {
  std::map<std::string, std::vector<Result> > results;
  for(Book::const_iterator it = book.begin(".*_sample.*"+method(m)); it!=book.end(); ++it ) {
    std::string name = boost::regex_replace(it.name(),boost::regex("sample\\d*_"),"");
    results[name].push_back(result(m,it.name(),book));
  }
  return results;
}

void LA_Filler_Fitter::
summarize_ensembles(Book& book) {
  typedef std::map<std::string, std::vector<Result> > results_t;
  results_t results;
  for(int m = FIRST_METHOD; m <= LAST_METHOD; m<<=1)
    if(methods_ & m) { results_t g = ensemble_results(book,(Method)(m)); results.insert(g.begin(),g.end());}
  
  BOOST_FOREACH(const results_t::value_type group, results) {
    const std::string name = group.first;
    try {
      BOOST_FOREACH(const Result p, group.second) {
	float pad = (ensembleUp_-ensembleLow_)/10;
	book.fill( p.reco,       name+"_ensembleReco", 12*ensembleBins_, ensembleLow_-pad, ensembleUp_+pad );
	book.fill( p.measure,    name+"_measure",      12*ensembleBins_, ensembleLow_-pad, ensembleUp_+pad );
	book.fill( p.measureErr, name+"_merr",         500, 0, 0.1);
	book.fill( (p.measure-p.reco)/p.measureErr, name+"_pull", 500, -10,10);
      }
      book(name+"_measure")->Fit("gaus","LMEQ");
      book(name+"_merr")->Fit("gaus","LMEQ");
      book(name+"_pull")->Fit("gaus","LMEQ");
    } catch (edm::Exception e) {
      std::cerr << "Fit summary failed " << std::endl << e << std::endl;
      book.erase(name+"_ensembleReco");
      book.erase(name+"_measure");
      book.erase(name+"_merr");
      book.erase(name+"_pull");
    }
  }
}

std::map<std::string, std::vector<LA_Filler_Fitter::EnsembleSummary> > LA_Filler_Fitter::
ensemble_summary(const Book& book) {
  std::map<std::string, std::vector<EnsembleSummary> > summary;
  for(Book::const_iterator it = book.begin(".*_ensembleReco"); it!=book.end(); ++it) {
    std::string base = boost::erase_all_copy(it.name(),"_ensembleReco");

    TH1* reco = *it;
    TH1* measure = book(base+"_measure");
    TH1* merr = book(base+"_merr");
    TH1* pull = book(base+"_pull");

    EnsembleSummary s;
    s.truth = reco->GetMean();
    s.meanMeasured = measure->GetFunction("gaus")->GetParameter(1);
    s.SDmeanMeasured = measure->GetFunction("gaus")->GetParError(1);
    s.sigmaMeasured = measure->GetFunction("gaus")->GetParameter(2);
    s.SDsigmaMeasured = measure->GetFunction("gaus")->GetParError(2);
    s.meanUncertainty = merr->GetFunction("gaus")->GetParameter(1);
    s.SDmeanUncertainty = merr->GetFunction("gaus")->GetParError(1);
    s.pull = pull->GetFunction("gaus")->GetParameter(2);
    s.SDpull = pull->GetFunction("gaus")->GetParError(2);

    std::string name = boost::regex_replace(base,boost::regex("ensembleBin\\d*_"),"");
    summary[name].push_back(s);
  }
  return summary;
}

std::pair<std::pair<float,float>, std::pair<float,float> > LA_Filler_Fitter::
offset_slope(const std::vector<LA_Filler_Fitter::EnsembleSummary>& ensembles) { 
  try {
    std::vector<float> x,y,xerr,yerr;
    BOOST_FOREACH(EnsembleSummary ensemble, ensembles) {
      x.push_back(ensemble.truth);
      xerr.push_back(0);
      y.push_back(ensemble.meanMeasured);
      yerr.push_back(ensemble.SDmeanMeasured);
    }
    TGraphErrors graph(x.size(),&(x[0]),&(y[0]),&(xerr[0]),&(yerr[0]));
    graph.Fit("pol1");
    TF1* fit = graph.GetFunction("pol1");
    
    return std::make_pair( std::make_pair(fit->GetParameter(0), fit->GetParError(0)),
			   std::make_pair(fit->GetParameter(1), fit->GetParError(1)) );
  } catch(edm::Exception e) { 
    std::cerr << "Fitting Line Failed " << std::endl << e << std::endl;
    return std::make_pair( std::make_pair(0,0), std::make_pair(0,0));
  }
}

float LA_Filler_Fitter::
pull(const std::vector<LA_Filler_Fitter::EnsembleSummary>& ensembles) {
  float p(0),w(0);
  BOOST_FOREACH(EnsembleSummary ensemble, ensembles) {
    float unc2 = pow(ensemble.SDpull,2);
    p+=  ensemble.pull / unc2;
    w+= 1/unc2;
  }
  return p/w;
}


std::ostream& operator<<(std::ostream& strm, const LA_Filler_Fitter::Result& r) { 
  return strm << r.reco    <<"\t"<< r.recoErr <<"\t"
	      << r.measure <<"\t"<< r.measureErr <<"\t"
	      << r.calibratedMeasurement <<"\t"<< r.calibratedError <<"\t"
	      << r.field <<"\t"
	      << r.chi2 <<"\t"
	      << r.ndof <<"\t"
	      << r.entries;
}

std::ostream& operator<<(std::ostream& strm, const LA_Filler_Fitter::EnsembleSummary& e) { 
  return strm << e.truth <<"\t"
	      << e.meanMeasured    <<"\t"<< e.SDmeanMeasured <<"\t"
	      << e.sigmaMeasured   <<"\t"<< e.SDsigmaMeasured <<"\t"
	      << e.meanUncertainty <<"\t"<< e.SDmeanUncertainty << "\t"
	      << e.pull            <<"\t"<< e.SDpull;
}



